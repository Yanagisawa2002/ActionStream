"""DISPATCH_005 offline text and RGB research; cannot import a native backend."""

import argparse
import json
import os
from pathlib import Path
import time
import traceback

from PIL import Image, ImageOps

from .coverage import adjudicate_covered
from .gate_runner import save, sha256
from .grounding import OriginalRequest, adjudicate
from .qwen import LocalQwen, checker_messages, parser_messages
from .structured import evidence_slots, status_slots
from .visual_evidence import (
    EvidenceHistory,
    interventions,
    mask_targets,
    parse_evidence,
    rgb_hash,
    verify,
)


def load_images(root, views):
    images = []
    for view in views:
        if set(view) != {"path", "sha256"}:
            raise ValueError("Unexpected view metadata")
        path = (root / view["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != view["sha256"]:
            raise ValueError("Input image identity mismatch")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "inputs", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("coverage", "evidence"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    public = json.loads(args.inputs.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="RUNNING",
        phase=args.phase,
        pid=os.getpid(),
        cases=[],
        expected_calls=120 if args.phase == "coverage" else 84,
        config_sha256=sha256(args.config),
        input_sha256=sha256(args.inputs),
        model_load={"status": "NOT_RUN"},
        execution_mode="OFFLINE_ONLY",
    )
    path = args.output / "run_receipt.json"
    save(path, receipt)
    started, model = time.monotonic(), None

    def call(identity, variant, images, prompt, slots, references=None):
        if len(receipt["cases"]) >= 84:
            raise ValueError("Vision call cap")
        messages = checker_messages(prompt, identity, images)
        if references:
            messages[1]["content"].extend(
                {"type": "image", "image": im} for im in references
            )
        row = dict(
            identity=identity,
            variant=variant,
            rgb_sha256=[rgb_hash(im) for im in images],
            reference_rgb_sha256=[rgb_hash(im) for im in (references or [])],
            **model.generate(
                messages,
                structured_slots=slots,
                record_tensors=True,
                max_new_tokens=320,
            ),
        )
        row["ordinal"] = len(receipt["cases"])
        receipt["cases"].append(row)
        save(path, receipt)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in ("ordinal", "identity", "variant", "status", "wall_s")
                }
            ),
            flush=True,
        )
        if row.get("error_type") == "TimeoutError":
            raise TimeoutError("Stop after one model timeout; no retries")
        return row

    try:
        model = LocalQwen(args.model, config["model"])
        receipt["model_load"] = dict(status="PASS", **model.receipt)
        save(path, receipt)
        if args.phase == "coverage":
            if len(public) != 120:
                raise ValueError(
                    "Text call schedule must contain exactly 120 originals"
                )
            for item in public:
                if set(item) != {"request_id", "text"}:
                    raise ValueError("Unexpected text metadata")
                request = OriginalRequest(**item)
                row = dict(
                    identity=request.request_id,
                    input=item,
                    original_sha256=request.text_sha256,
                    **model.generate(
                        parser_messages(
                            config["parser_system_prompt"],
                            request.request_id,
                            request.text,
                        )
                    ),
                )
                if row["status"] == "COMPLETED":
                    before = time.perf_counter()
                    row["v2"] = adjudicate(
                        request, row["raw_output"], config["contract"]
                    )
                    row["v2_host_s"] = time.perf_counter() - before
                    before = time.perf_counter()
                    row["v3"] = adjudicate_covered(
                        request, row["raw_output"], config["contract"]
                    )
                    row["v3_host_s"] = time.perf_counter() - before
                receipt["cases"].append(row)
                save(path, receipt)
                print(
                    json.dumps(
                        dict(
                            ordinal=len(receipt["cases"]),
                            status=row["status"],
                            wall_s=row["wall_s"],
                        )
                    ),
                    flush=True,
                )
                if row.get("error_type") == "TimeoutError":
                    raise TimeoutError("Text phase stops without retry")
        else:
            if set(public) != {"observations", "anchors", "references", "thresholds"}:
                raise ValueError("Unexpected visual input metadata")
            observations = public["observations"]
            if len(observations) != 20 or len(public["anchors"]) != 4:
                raise ValueError("Frozen visual schedule changed")
            lookup = {}
            for item in observations:
                if set(item) != {"observation_id", "stream", "observed_ns", "views"}:
                    raise ValueError("Unexpected observation metadata")
                lookup[item["observation_id"]] = item
            references = load_images(args.inputs.parent, public["references"])
            # Causal diagnostics run FIRST, before any evidence proposal output.
            for anchor in public["anchors"]:
                current = load_images(
                    args.inputs.parent, lookup[anchor["current"]]["views"]
                )
                earlier = load_images(
                    args.inputs.parent, lookup[anchor["earlier"]]["views"]
                )[1]
                for variant, images in interventions(current, earlier).items():
                    call(
                        anchor["current"],
                        "causal_" + variant,
                        images,
                        config["causal_prompt"],
                        status_slots(),
                    )
            history = EvidenceHistory()
            receipt["verified"] = []
            for item in observations:
                identity = item["observation_id"]
                images = load_images(args.inputs.parent, item["views"])
                original = call(
                    identity,
                    "original",
                    images,
                    config["region_prompt"],
                    evidence_slots(),
                    references,
                )
                flipped = call(
                    identity,
                    "flipped",
                    [ImageOps.mirror(im) for im in images],
                    config["region_prompt"],
                    evidence_slots(),
                    references,
                )
                try:
                    proposal = parse_evidence(original["raw_output"])
                except (ValueError, TypeError):
                    proposal = None
                masked = call(
                    identity,
                    "masked",
                    mask_targets(images, proposal),
                    config["region_prompt"],
                    evidence_slots(),
                    references,
                )
                result = verify(
                    images,
                    original["raw_output"],
                    flipped["raw_output"],
                    masked["raw_output"],
                    references,
                    public["thresholds"],
                )
                # Receipt binding checks actual model inputs, not the derived PIL
                # transforms alone. All three prompts must be byte-identical.
                changed = all(
                    len(original.get("image_tensor_sha256", [])) >= 2
                    and len(altered.get("image_tensor_sha256", [])) >= 2
                    and original["image_tensor_sha256"][v]
                    != altered["image_tensor_sha256"][v]
                    and original["rgb_sha256"][v] != altered["rgb_sha256"][v]
                    for altered in (flipped, masked)
                    for v in range(2)
                )
                prompt_same = (
                    len({r.get("prompt_sha256") for r in (original, flipped, masked)})
                    == 1
                )
                result.update(intervention_input_verified=bool(changed and prompt_same))
                if not changed or not prompt_same:
                    result.update(
                        status="unknown", reason="counterfactual_input_not_established"
                    )
                result = history.record(item["stream"], item["observed_ns"], result)
                receipt["verified"].append(dict(identity=identity, **result))
                save(path, receipt)
        receipt["status"] = "COMPLETED"
        return 0
    except BaseException as exc:
        receipt.update(
            status="ERROR",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        print(receipt["traceback"], flush=True)
        return 1
    finally:
        if model is not None:
            receipt["peak_torch_allocated_MiB"] = (
                model.torch.cuda.max_memory_allocated() / 1024**2
            )
            model.close()
        receipt["wall_s"] = time.monotonic() - started
        save(path, receipt)


if __name__ == "__main__":
    raise SystemExit(main())
