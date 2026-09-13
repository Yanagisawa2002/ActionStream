"""Independent offline evaluator; its references never enter model/runtime APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import CheckDecision, TaskSpec
from .gate_runner import save, sha256


def score(phase, receipt, answers, config):
    key = "request_id" if phase == "language" else "observation_id"
    truths = {row[key]: row for row in answers}
    if len(truths) != len(answers):
        raise ValueError("Duplicate reference identity")
    rows = receipt.get("cases", [])
    seen = set()
    scored = []
    for row in rows:
        identity = row["identity"]
        if identity not in truths or identity in seen:
            raise ValueError("Missing or duplicate model/reference identity")
        seen.add(identity)
        decision = "explicit_failure"
        error = row.get("error")
        if row["status"] == "COMPLETED":
            try:
                parsed = (TaskSpec if phase == "language" else CheckDecision).parse(
                    row["raw_output"], identity
                )
                decision = parsed.decision if phase == "language" else parsed.status
            except (ValueError, TypeError) as exc:
                error = str(exc)
        scored.append(
            dict(identity=identity, decision=decision, error=error, **truths[identity])
        )
    complete = receipt.get("status") == "COMPLETED" and seen == set(truths)
    result = dict(
        phase=phase,
        status="PASS",
        complete_run=complete,
        evaluated=len(scored),
        expected=len(answers),
        rows=scored,
        explicit_failures=sum(r["decision"] == "explicit_failure" for r in scored),
    )
    if phase == "language":
        supported = [r for r in scored if r["expected_accept"]]
        negative = [r for r in scored if not r["expected_accept"]]
        correct = sum(r["decision"] == "accept" for r in supported)
        false = sum(r["decision"] == "accept" for r in negative)
        gate = config["language_gate"]
        passed = (
            complete
            and len(supported) == gate["supported_cases"]
            and len(negative) == gate["unsupported_cases"]
            and correct >= gate["minimum_correct_accepts"]
            and false <= gate["maximum_false_accepts"]
        )
        result.update(
            supported=len(supported),
            negative=len(negative),
            correct_accepts=correct,
            false_accepts=false,
        )
    else:
        labelled = [r for r in scored if r["native_is_success"] is not None]
        positive = [r for r in labelled if r["native_is_success"] is True]
        negative = [r for r in labelled if r["native_is_success"] is False]
        matrix = {
            truth: {
                status: 0
                for status in ("complete", "incomplete", "unknown", "explicit_failure")
            }
            for truth in ("positive", "negative", "unlabeled")
        }
        for row in scored:
            truth = (
                "unlabeled"
                if row["native_is_success"] is None
                else ("positive" if row["native_is_success"] else "negative")
            )
            matrix[truth][row["decision"]] += 1
        coverage = sum(
            r["decision"] in ("complete", "incomplete") for r in labelled
        ) / max(1, len(labelled))
        all_coverage = sum(
            r["decision"] in ("complete", "incomplete") for r in scored
        ) / max(1, len(answers))
        gate = config["shadow_gate"]
        passed = (
            complete
            and len(answers) <= gate["maximum_pairs"]
            and len(negative) >= gate["minimum_labelled_negative"]
            and len(positive) >= gate["minimum_labelled_positive"]
            and matrix["negative"]["complete"] <= gate["maximum_false_complete"]
            and coverage >= gate["minimum_labelled_non_unknown_coverage"]
            and matrix["positive"]["complete"] >= gate["minimum_true_complete"]
        )
        result.update(
            labelled=len(labelled),
            positive=len(positive),
            negative=len(negative),
            confusion_matrix=matrix,
            labelled_non_unknown_coverage=coverage,
            all_non_unknown_coverage=all_coverage,
            false_complete=matrix["negative"]["complete"],
            true_complete=matrix["positive"]["complete"],
        )
    result["status"] = "PASS" if passed else "FAIL"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("language", "shadow"), required=True)
    for name in ("receipt", "answers", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a gate score")
    result = score(
        args.phase,
        *(
            json.loads(path.read_text(encoding="utf-8"))
            for path in (args.receipt, args.answers, args.config)
        ),
    )
    result["source_sha256"] = {
        key: sha256(getattr(args, key)) for key in ("receipt", "answers", "config")
    }
    save(args.output, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"}, indent=2
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
