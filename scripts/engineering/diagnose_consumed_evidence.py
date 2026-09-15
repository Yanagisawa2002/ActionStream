"""Read-only CPU diagnosis of consumed evidence; never model evaluation."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / "src"))
    from actionstream.llm_vla.coverage import adjudicate_covered
    from actionstream.llm_vla.grounding import OriginalRequest
    from actionstream.llm_vla.research_scoring import score_vision
    from actionstream.llm_vla.visual_evidence import similarity

    inputs = {}

    def read(path):
        content = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    root = args.evidence
    public = read(root / "model_inputs/evidence.json")
    annotations = read(root / "scorer_only/visual_annotations.json")
    config = read(root / "frozen_config.json")
    old = root.parent / "embodied_llm_vla_grounding_20260913"
    receipt = read(old / "comparison/run_receipt.json")
    truth = {
        r["request_id"]: r
        for name in ("comparison_answers", "regression_answers")
        for r in read(old / f"scorer_only/{name}.json")
    }
    language = []
    for row in receipt["cases"]:
        if row["version"] != "v2":
            continue
        result = adjudicate_covered(
            OriginalRequest(**row["input"]), row["raw_output"], config["contract"]
        )
        language.append(
            dict(
                dataset=row["dataset"],
                text=row["input"]["text"],
                expected_accept=truth[row["identity"]]["expected_accept"],
                before=row["decision"],
                after=result["decision"],
            )
        )
    summary = {}
    for dataset in ("new", "regression"):
        selected = [r for r in language if r["dataset"] == dataset]
        summary[dataset] = dict(cases=len(selected))
        for version in ("before", "after"):
            summary[dataset][version] = dict(
                correct_accepts=sum(
                    r["expected_accept"] and r[version] == "accept" for r in selected
                ),
                false_accepts=sum(
                    not r["expected_accept"] and r[version] == "accept"
                    for r in selected
                ),
            )

    references = []
    for ref in public["references"]:
        path = root / "model_inputs" / ref["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"]
        with Image.open(path) as image:
            references.append(image.convert("RGB"))
    lookup = {r["observation_id"]: r for r in public["observations"]}
    appearance = []
    for annotation in annotations:
        identity = annotation["observation_id"]
        for view, regions in enumerate(annotation["views"]):
            public_view = lookup[identity]["views"][view]
            path = root / "model_inputs" / public_view["path"]
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == public_view["sha256"]
            )
            with Image.open(path) as image:
                for entity, key, reference in zip(
                    ("target", "basket"), ("t", "b"), references
                ):
                    box = regions[entity + "_box"]
                    if box is None:
                        continue
                    value = similarity(image.convert("RGB").crop(box), reference)
                    appearance.append(
                        dict(
                            identity=identity,
                            split=annotation["split"],
                            view=view,
                            entity=entity,
                            score=value,
                            threshold=public["thresholds"][key],
                            passed=value >= public["thresholds"][key],
                            native_is_success=annotation["native_is_success"],
                        )
                    )
    grouped = defaultdict(list)
    for row in appearance:
        grouped[f"{row['split']}_{row['entity']}_view{row['view']}"].append(row)
    appearance_summary = {
        k: dict(
            n=len(v),
            passed=sum(r["passed"] for r in v),
            score_min=min(r["score"] for r in v),
            score_max=max(r["score"] for r in v),
        )
        for k, v in grouped.items()
    }

    # Negative test: a corrupted worker claims perfect verified statuses although
    # ALL model calls failed. This synthetic receipt is never a model result.
    cases = []
    for obs in public["observations"]:
        for variant in ("original", "flipped", "masked"):
            cases.append(
                dict(
                    identity=obs["observation_id"],
                    variant=variant,
                    status="ERROR",
                    raw_output="{}",
                    rgb_sha256=[],
                )
            )
    for anchor in public["anchors"]:
        for variant in (
            "real",
            "blank",
            "swapped",
            "mismatched",
            "occluded",
            "cropped",
        ):
            cases.append(
                dict(
                    identity=anchor["current"],
                    variant="causal_" + variant,
                    status="ERROR",
                    raw_output="{}",
                    rgb_sha256=[],
                )
            )
    corrupted = dict(
        status="COMPLETED",
        cases=cases,
        verified=[
            dict(
                identity=r["observation_id"],
                status="complete" if r["native_is_success"] else "incomplete",
            )
            for r in annotations
        ],
    )
    scored = score_vision(corrupted, annotations)
    result = dict(
        scope="CPU replay of consumed raw outputs and analyst image boxes; zero new model calls; no changed frozen input",
        language_summary=summary,
        language_rows=language,
        appearance_summary=appearance_summary,
        appearance_rows=appearance,
        synthetic_corrupt_receipt_probe=dict(
            model_calls=len(cases),
            successful_model_calls=0,
            scorer_status=scored["status"],
            groups=scored["groups"],
            means="Integrity regression only; not evidence of tampering in any historical run",
        ),
        input_sha256=inputs,
    )
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in inputs.items()
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {k: result[k] for k in ("language_summary", "appearance_summary")}, indent=2
        )
    )
    print("Synthetic all-model-error scorer status:", scored["status"])


if __name__ == "__main__":
    main()
