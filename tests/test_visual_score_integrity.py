"""Synthetic CPU receipt integrity tests, not visual-model accuracy tests."""

import copy
import hashlib
import json

from PIL import Image, ImageDraw, ImageOps
import pytest

from actionstream.llm_vla.research_scoring import score_vision
from actionstream.llm_vla.visual_evidence import (
    EvidenceHistory,
    interventions,
    mask_targets,
    pixel_box,
    rgb_hash,
    verify,
)


@pytest.fixture
def trial(tmp_path):
    references = [Image.new("RGB", (30, 30), color) for color in ("red", "white")]

    def saved(image, name):
        path = tmp_path / name
        image.save(path)
        return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    thresholds = dict(
        t=0.9, b=0.1, flip_iou=0.5, inside_fraction=0.8, outside_fraction=0.05
    )
    public = dict(
        observations=[],
        anchors=[],
        references=[saved(im, f"ref{i}.png") for i, im in enumerate(references)],
        thresholds=thresholds,
    )
    receipt = dict(status="COMPLETED", cases=[], verified=[])
    annotations, pairs = [], {}
    history = EvidenceHistory()

    def record(identity, variant, images, raw, refs):
        prompt = "Synthetic CPU contract prompt. No model was called."
        return dict(
            identity=identity,
            variant=variant,
            status="COMPLETED",
            raw_output=raw,
            prompt=prompt,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            rgb_sha256=[rgb_hash(im) for im in images],
            reference_rgb_sha256=[rgb_hash(im) for im in refs],
            image_tensor_sha256=[
                hashlib.sha256(("synthetic:" + rgb_hash(im)).encode()).hexdigest()
                for im in images + refs
            ],
        )

    for i in range(20):
        identity, complete = f"observation{i}", i >= 17
        target = [500, 200, 650, 450] if complete else [100, 100, 200, 250]
        basket = [400, 100, 900, 600]
        image = Image.new("RGB", (360, 360), "white")
        ImageDraw.Draw(image).rectangle(pixel_box(target, image), fill="red")
        pair = [image, image.copy()]
        pairs[identity] = pair
        public["observations"].append(
            dict(
                observation_id=identity,
                stream="stream",
                observed_ns=i + 1,
                views=[saved(im, f"{identity}-{v}.png") for v, im in enumerate(pair)],
            )
        )
        proposal = {}
        for v in range(2):
            proposal.update(
                {
                    f"t{v}": target,
                    f"b{v}": basket,
                    f"i{v}": "match",
                    f"j{v}": "match",
                    f"v{v}": "visible",
                    f"w{v}": "visible",
                    f"r{v}": "inside" if complete else "outside",
                }
            )
        flipped, masked = copy.deepcopy(proposal), copy.deepcopy(proposal)
        for v in range(2):
            for key in "tb":
                b = proposal[f"{key}{v}"]
                flipped[f"{key}{v}"] = [1000 - b[2], b[1], 1000 - b[0], b[3]]
            masked[f"v{v}"] = "absent"
            masked[f"r{v}"] = "unknown"
        raws = [json.dumps(p) for p in (proposal, flipped, masked)]
        changed = [
            pair,
            [ImageOps.mirror(im) for im in pair],
            mask_targets(pair, proposal),
        ]
        for variant, ims, raw in zip(("original", "flipped", "masked"), changed, raws):
            receipt["cases"].append(record(identity, variant, ims, raw, references))
        result = verify(pair, *raws, references, thresholds)
        assert result["status"] == ("complete" if complete else "incomplete")
        result["intervention_input_verified"] = True
        result = history.record("stream", i + 1, result)
        receipt["verified"].append(dict(identity=identity, **result))
        annotations.append(
            dict(
                observation_id=identity,
                source_episode="cal" if i < 5 else "eval",
                split="calibration" if i < 5 else "diagnostic_evaluation",
                native_is_success=complete,
                views=[
                    dict(
                        target_box=pixel_box(target, image),
                        basket_box=pixel_box(basket, image),
                        target_identity="match",
                        target_visibility="visible",
                        basket_visibility="visible",
                        containment="inside" if complete else "outside",
                    )
                    for _ in range(2)
                ],
            )
        )
    for i in (0, 5, 10, 15):
        identity = f"observation{i}"
        public["anchors"].append(dict(current=identity, earlier=identity))
        for variant, images in interventions(
            pairs[identity], pairs[identity][1]
        ).items():
            receipt["cases"].append(
                record(
                    identity, "causal_" + variant, images, '{"status":"unknown"}', []
                )
            )
    assert len(receipt["cases"]) == 84
    return receipt, annotations, public, tmp_path


def test_valid_bound_receipt_can_pass_cpu_integrity_and_scoring(trial):
    result = score_vision(*trial)
    assert result["integrity"]["status"] == "PASS"
    assert result["status"] == "PASS"
    assert result["groups"]["diagnostic_evaluation"]["true_complete"] == 3


def test_worker_status_without_independent_inputs_cannot_pass(trial):
    result = score_vision(trial[0], trial[1])
    assert result["status"] == "FAIL"


@pytest.mark.parametrize(
    "corruption",
    [
        "all_model_errors",
        "forged_worker",
        "raw_changed",
        "rgb_changed",
        "tensor_missing",
        "prompt_changed",
        "schedule_changed",
        "source_image_changed",
    ],
)
def test_corrupted_receipt_cannot_report_quality_pass(trial, corruption):
    receipt, annotations, public, root = trial
    if corruption == "all_model_errors":
        for call in receipt["cases"]:
            call.update(status="ERROR", raw_output="{}")
    elif corruption == "forged_worker":
        receipt["verified"][0]["status"] = "complete"
    elif corruption == "raw_changed":
        raw = json.loads(receipt["cases"][0]["raw_output"])
        raw["t0"] = [0, 0, 0, 0]
        receipt["cases"][0]["raw_output"] = json.dumps(raw)
    elif corruption == "rgb_changed":
        receipt["cases"][1]["rgb_sha256"][0] = "f" * 64
    elif corruption == "tensor_missing":
        receipt["cases"][0]["image_tensor_sha256"] = []
    elif corruption == "prompt_changed":
        receipt["cases"][0]["prompt"] += "changed"
    elif corruption == "schedule_changed":
        receipt["cases"][1]["variant"] = "unexpected"
    else:
        (root / public["observations"][0]["views"][0]["path"]).write_bytes(b"changed")
    result = score_vision(receipt, annotations, public, root)
    assert result["integrity"]["status"] == "FAIL"
    assert result["status"] == "FAIL"
