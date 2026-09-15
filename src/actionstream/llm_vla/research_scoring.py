"""Independent DISPATCH_005 evaluator: consumed regression and diagnostic data."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from .coverage import adjudicate_covered
from .gate_runner import save, sha256
from .grounding import OriginalRequest, adjudicate
from .repair_scoring import summarize
from .visual_evidence import iou, parse_evidence
from .visual_replay import replay_visual_receipt


def score_text(receipt, inputs, answers, config):
    originals = {r["request_id"]: r for r in inputs}
    truth = {r["request_id"]: r for r in answers}
    rows, seen = [], set()
    for call in receipt["cases"]:
        identity = call["identity"]
        if (
            identity in seen
            or identity not in originals
            or call["input"] != originals[identity]
        ):
            raise ValueError("Duplicated, unknown or changed text input")
        seen.add(identity)
        for version, interpret in (("v2", adjudicate), ("v3", adjudicate_covered)):
            if call["status"] == "COMPLETED":
                result = interpret(
                    OriginalRequest(**originals[identity]),
                    call["raw_output"],
                    config["contract"],
                )
                if call[version] != result:
                    raise ValueError("Runtime and independent text adjudication differ")
            else:
                result = dict(
                    decision="explicit_failure",
                    schema_status="EXPLICIT_FAILURE",
                    raw_decision=None,
                    reason="model_error",
                )
            rows.append(
                dict(
                    identity=identity,
                    version=version,
                    text=originals[identity]["text"],
                    **truth[identity],
                    wall_s=call["wall_s"],
                    **result,
                    host_s=call.get(version + "_host_s"),
                )
            )
    groups = {}
    for dataset, count in (("new48", 48), ("old48", 48), ("old24", 24)):
        for version in ("v2", "v3"):
            selected = [
                r for r in rows if r["dataset"] == dataset and r["version"] == version
            ]
            summary = summarize(selected, count)
            summary["host_wall_s_sum"] = sum(r["host_s"] or 0 for r in selected)
            summary["reasons"] = dict(Counter(r["reason"] for r in selected))
            summary["families"] = {
                family: summarize(
                    [r for r in selected if r["family"] == family],
                    sum(r["family"] == family for r in selected),
                )
                for family in sorted({r["family"] for r in selected})
            }
            groups[dataset + "_" + version] = summary
    passed = receipt["status"] == "COMPLETED" and len(seen) == 120
    for dataset, minimum, count in (
        ("new48", 20, 48),
        ("old48", 20, 48),
        ("old24", 10, 24),
    ):
        group = groups[dataset + "_v3"]
        passed &= (
            group["evaluated"] == count
            and group["correct_accepts"] >= minimum
            and group["false_accepts"] == 0
        )
    return dict(
        status="PASS" if passed else "FAIL",
        comparison="same real v2 semantic proposal, two host adjudicators; 120 calls, not 240",
        groups=groups,
        rows=rows,
    )


def score_vision(receipt, annotations, public=None, input_root=None):
    integrity = replay_visual_receipt(receipt, public, input_root)
    truth = {r["observation_id"]: r for r in annotations}
    if len(truth) != len(annotations):
        raise ValueError("Duplicate visual annotations")
    originals, calls = {}, {}
    for row in receipt["cases"]:
        key = row["identity"], row["variant"]
        if key in calls or key[0] not in truth:
            raise ValueError("Unexpected or duplicate visual call")
        calls[key] = row
        if key[1] == "original":
            originals[key[0]] = row
    verified = integrity["verified"]
    rows = []
    for identity, annotation in truth.items():
        row = dict(
            identity=identity,
            split=annotation["split"],
            source_episode=annotation["source_episode"],
            native_is_success=annotation["native_is_success"],
            status=verified.get(identity, {}).get("status", "unknown"),
            evidence=verified.get(identity),
            region_metrics=[],
            proposal_schema="EXPLICIT_FAILURE",
        )
        try:
            proposed = parse_evidence(originals[identity]["raw_output"])
            row["proposal_schema"] = "VALID"
            for view, gold in enumerate(annotation["views"]):
                item = dict(
                    view=view,
                    target_identity_correct=proposed[f"i{view}"]
                    == gold["target_identity"],
                    target_visibility_correct=proposed[f"v{view}"]
                    == gold["target_visibility"],
                    basket_visibility_correct=proposed[f"w{view}"]
                    == gold["basket_visibility"],
                    containment_correct=proposed[f"r{view}"] == gold["containment"]
                    if gold["containment"] != "unknown"
                    else None,
                )
                for key, name in (("t", "target"), ("b", "basket")):
                    box = gold[name + "_box"]
                    if box is not None:
                        normalized = [round(n * 1000 / 360) for n in box]
                        item[name + "_iou"] = iou(proposed[f"{key}{view}"], normalized)

                        def center(b):
                            return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

                        a, b = center(proposed[f"{key}{view}"]), center(normalized)
                        item[name + "_center_error_pixels"] = (
                            (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                        ) ** 0.5 * 0.36
                item["target_identity_and_localization"] = (
                    (proposed[f"i{view}"] == "match" and item["target_iou"] >= 0.5)
                    if gold["target_identity"] == "match"
                    else None
                )
                row["region_metrics"].append(item)
        except (KeyError, ValueError, TypeError):
            pass
        rows.append(row)
    groups = {}
    for split in ("calibration", "diagnostic_evaluation"):
        selected = [r for r in rows if r["split"] == split]
        labelled = [r for r in selected if r["native_is_success"] is not None]
        positives = [r for r in labelled if r["native_is_success"]]
        negatives = [r for r in labelled if not r["native_is_success"]]
        metrics = [v for r in selected for v in r["region_metrics"]]
        aggregate = {}
        for key in (
            "target_iou",
            "basket_iou",
            "target_center_error_pixels",
            "basket_center_error_pixels",
            "target_identity_correct",
            "target_identity_and_localization",
            "target_visibility_correct",
            "basket_visibility_correct",
            "containment_correct",
        ):
            values = [v[key] for v in metrics if v.get(key) is not None]
            aggregate[key] = dict(
                n=len(values), mean=statistics.mean(values) if values else None
            )
        groups[split] = dict(
            observations=len(selected),
            labelled=len(labelled),
            positives=len(positives),
            negatives=len(negatives),
            schema_valid=sum(r["proposal_schema"] == "VALID" for r in selected),
            final_statuses=dict(Counter(r["status"] for r in selected)),
            valid_coverage=sum(
                r["status"] in ("complete", "incomplete") for r in labelled
            )
            / max(1, len(labelled)),
            false_complete=sum(r["status"] == "complete" for r in negatives),
            true_complete=sum(r["status"] == "complete" for r in positives),
            true_complete_recall=sum(r["status"] == "complete" for r in positives)
            / max(1, len(positives)),
            region_metrics=aggregate,
            verification_checks={
                key: dict(
                    passed=sum(
                        v["checks"].get(key, False)
                        for r in selected
                        for v in (r["evidence"] or {}).get("views", [])
                    ),
                    total=sum(
                        len((r["evidence"] or {}).get("views", [])) for r in selected
                    ),
                )
                for key in (
                    "t_flip",
                    "b_flip",
                    "t_appearance",
                    "b_appearance",
                    "identity",
                    "visibility",
                    "mask_response",
                    "relation_consistency",
                )
            },
        )
    causal = []
    for (identity, variant), row in calls.items():
        if not variant.startswith("causal_"):
            continue
        real = calls[identity, "causal_real"]
        try:
            status = json.loads(row["raw_output"])["status"]
            real_status = json.loads(real["raw_output"])["status"]
        except (ValueError, TypeError, KeyError):
            status = real_status = "explicit_failure"
        causal.append(
            dict(
                identity=identity,
                variant=variant,
                status=status,
                real_status=real_status,
                output_changed=status != real_status,
                prompt_same=row.get("prompt_sha256") == real.get("prompt_sha256"),
                rgb_changed=[
                    a != b for a, b in zip(row["rgb_sha256"], real["rgb_sha256"])
                ],
                tensor_changed=[
                    a != b
                    for a, b in zip(
                        row.get("image_tensor_sha256", []),
                        real.get("image_tensor_sha256", []),
                    )
                ],
                grid=row.get("image_grid_thw"),
                grid_changed=row.get("image_grid_thw") != real.get("image_grid_thw"),
            )
        )
    group = groups["diagnostic_evaluation"]
    passed = (
        integrity["status"] == "PASS"
        and receipt["status"] == "COMPLETED"
        and len(calls) == 84
        and len(verified) == 20
        and group["positives"] >= 3
        and group["negatives"] >= 8
        and group["false_complete"] == 0
        and group["valid_coverage"] >= 0.8
        and group["true_complete"] >= 3
    )
    return dict(
        status="PASS" if passed else "FAIL",
        integrity={k: v for k, v in integrity.items() if k != "verified"},
        groups=groups,
        causal=causal,
        rows=rows,
        limitations="All 20 images consumed previously. Episode-separated RGB calibration and diagnostic evaluation, not a fresh holdout. Analyst approximate boxes and identities are not independently human validated.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence

    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    for phase in ("coverage", "evidence"):
        path = root / (phase + "_score.json")
        if path.exists():
            raise ValueError("Refusing to overwrite research scores")
        receipt = read(phase + "/run_receipt.json")
        result = (
            score_text(
                receipt,
                read("model_inputs/coverage.json"),
                read("scorer_only/language_answers.json"),
                read("frozen_config.json"),
            )
            if phase == "coverage"
            else score_vision(
                receipt,
                read("scorer_only/visual_annotations.json"),
                read("model_inputs/evidence.json"),
                root / "model_inputs",
            )
        )
        result["source_sha256"] = dict(
            receipt=sha256(root / phase / "run_receipt.json"),
            config=sha256(root / "frozen_config.json"),
        )
        save(path, result)
        print(
            json.dumps(
                dict(
                    phase=phase,
                    status=result["status"],
                    groups={
                        k: {a: b for a, b in v.items() if a not in ("families",)}
                        for k, v in result["groups"].items()
                    },
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
