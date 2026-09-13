"""Independent paired evaluator; preserves the original v1 validation semantics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from .contracts import TaskSpec
from .gate_runner import save, sha256
from .grounding import OriginalRequest, adjudicate


def interpret(row: dict, original: dict, contract: dict):
    if row["input"] != original or row["identity"] != original["request_id"]:
        raise ValueError("Model input differs from the frozen original request")
    result = dict(
        decision="explicit_failure",
        raw_decision=None,
        schema_status="EXPLICIT_FAILURE",
        reason="model_error",
    )
    if row["status"] != "COMPLETED":
        return result
    try:
        data = json.loads(row["raw_output"])
        result["raw_decision"] = (
            data.get("decision") if isinstance(data, dict) else None
        )
    except (ValueError, TypeError):
        pass
    if row["version"] == "v2":
        verdict = adjudicate(
            OriginalRequest(original["request_id"], original["text"]),
            row["raw_output"],
            contract,
        )
        return {
            **result,
            **{key: value for key, value in verdict.items() if key != "task"},
            "raw_decision": result["raw_decision"],
        }
    if row["version"] != "v1":
        raise ValueError("Unknown comparison contract version")
    try:
        task = TaskSpec.parse(row["raw_output"], original["request_id"])
        return dict(
            result, decision=task.decision, schema_status="VALID", reason=task.reason
        )
    except (ValueError, TypeError) as exc:
        return dict(result, reason="malformed", error=str(exc))


def summarize(rows: list, expected: int):
    positive = [r for r in rows if r["expected_accept"]]
    negative = [r for r in rows if not r["expected_accept"]]
    correct = sum(r["decision"] == "accept" for r in positive)
    false = sum(r["decision"] == "accept" for r in negative)
    return dict(
        evaluated=len(rows),
        expected=expected,
        positive=len(positive),
        negative=len(negative),
        correct_accepts=correct,
        false_accepts=false,
        acceptance_recall=correct / max(1, len(positive)),
        raw_decisions=dict(Counter(str(r["raw_decision"]) for r in rows)),
        final_decisions=dict(Counter(r["decision"] for r in rows)),
        schema_valid=sum(r["schema_status"] == "VALID" for r in rows),
        explicit_failures=sum(r["decision"] == "explicit_failure" for r in rows),
        unknown=sum(r["decision"] == "unknown" for r in rows),
        wall_s_sum=sum(r["wall_s"] for r in rows),
        wall_s_p50=statistics.median(r["wall_s"] for r in rows) if rows else None,
    )


def score_comparison(receipt, inputs, regression, answers, old_answers, config):
    originals = {("new", r["request_id"]): r for r in inputs}
    originals.update({("regression", r["request_id"]): r for r in regression})
    truth = {("new", r["request_id"]): r for r in answers}
    truth.update({("regression", r["request_id"]): r for r in old_answers})
    seen, rows = set(), []
    for row in receipt.get("cases", []):
        key = row["dataset"], row["identity"]
        unique = (*key, row["version"])
        if (
            key not in truth
            or unique in seen
            or (row["dataset"] == "regression" and row["version"] != "v2")
        ):
            raise ValueError("Unexpected or duplicate comparison identity")
        seen.add(unique)
        verdict = interpret(row, originals[key], config["contract"])
        rows.append(
            dict(
                dataset=key[0],
                version=row["version"],
                identity=key[1],
                text=originals[key]["text"],
                expected_accept=truth[key]["expected_accept"],
                family=truth[key]["family"],
                wall_s=row["wall_s"],
                **verdict,
            )
        )
    groups = {}
    for dataset, version, count in (
        ("new", "v1", 48),
        ("new", "v2", 48),
        ("regression", "v2", 24),
    ):
        selected = [
            r for r in rows if (r["dataset"], r["version"]) == (dataset, version)
        ]
        groups[dataset + "_" + version] = summarize(selected, count)
        groups[dataset + "_" + version]["families"] = {
            family: summarize(
                [r for r in selected if r["family"] == family],
                sum(t["family"] == family for k, t in truth.items() if k[0] == dataset),
            )
            for family in sorted({r["family"] for r in selected})
        }
    new, old = groups["new_v2"], groups["regression_v2"]
    complete = receipt.get("status") == "COMPLETED" and len(seen) == 120
    gate = config["repair_language_gate"]
    passed = (
        complete
        and new["positive"] == new["negative"] == 24
        and old["positive"] == old["negative"] == 12
        and new["correct_accepts"] >= gate["new_minimum_correct_accepts"]
        and old["correct_accepts"] >= gate["old_minimum_correct_accepts"]
        and new["false_accepts"] == old["false_accepts"] == 0
    )
    return dict(
        status="PASS" if passed else "FAIL",
        complete_run=complete,
        groups=groups,
        rows=rows,
    )


def require_joint_gates(evidence: Path):
    for phase, receipt_path in (("language", "comparison"), ("shadow", "shadow")):
        gate = json.loads((evidence / (phase + "_score.json")).read_text())
        if (
            gate.get("status") != "PASS"
            or gate["source_sha256"]["receipt"]
            != sha256(evidence / receipt_path / "run_receipt.json")
            or gate["source_sha256"]["config"]
            != sha256(evidence / "frozen_config.json")
        ):
            raise ValueError(
                "Native v2 requires both intact real language and visual gates"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence

    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    output = root / "language_score.json"
    if output.exists():
        raise ValueError("Refusing to overwrite paired results")
    result = score_comparison(
        read("comparison/run_receipt.json"),
        read("model_inputs/comparison.json"),
        read("model_inputs/regression.json"),
        read("scorer_only/comparison_answers.json"),
        read("scorer_only/regression_answers.json"),
        read("frozen_config.json"),
    )
    result["source_sha256"] = {
        "receipt": sha256(root / "comparison/run_receipt.json"),
        "config": sha256(root / "frozen_config.json"),
    }
    save(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
