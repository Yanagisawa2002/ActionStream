"""Contracts and counterfactual mechanics; these do not test model ability."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageOps

from actionstream.llm_vla.coverage import adjudicate_covered, authorize, execute_covered
from actionstream.llm_vla.grounding import OriginalRequest
from actionstream.llm_vla.structured import SlotDecoder, evidence_slots
from actionstream.llm_vla.visual_evidence import (
    EvidenceHistory,
    interventions,
    mask_targets,
    parse_evidence,
    rgb_hash,
    verify,
)

CONFIG = json.loads(
    (Path(__file__).parents[1] / "configs/llm_vla/dispatch005.json").read_text(
        encoding="utf-8"
    )
)
CONTRACT = CONFIG["contract"]
RAW = '{"decision":"accept","target_quote":"tomato sauce","destination_quote":"basket"}'


@pytest.mark.parametrize(
    "text",
    [
        "Put the tomato sauce in the basket and shut the window.",
        "Put the tomato sauce and the soup in the basket.",
        "Put three tomato sauce jars into the basket.",
        "Put the basket over the tomato sauce.",
        "If I nod, put the tomato sauce in the basket.",
        "Put the tomato sauce in the basket slowly.",
        "Put the tomato sauce in the basket 外面.",
        "Put the tomato sauce in the basket and keep holding it.",
    ],
)
def test_complete_coverage_rejects_unconsumed_semantics(text):
    result = adjudicate_covered(OriginalRequest("arbitrary", text), RAW, CONTRACT)
    assert result["decision"] != "accept"
    assert not result["coverage"]["complete"]
    assert result["coverage"]["obligations"][-1]["kind"] == "unresolved_utterance"


def test_compositional_and_is_supported_and_source_is_retained():
    text = "Could you lift the tomato sauce and place it inside the basket?"
    request = OriginalRequest("not-an-answer-key", text)
    result = adjudicate_covered(request, RAW, CONTRACT)
    assert result["decision"] == "accept"
    spans = [o for o in result["coverage"]["obligations"] if "start" in o]
    assert all(text[o["start"] : o["end"]] == o["quote"] for o in spans)
    assert all(
        o["disposition"] == "supported" for o in result["coverage"]["obligations"]
    )


def test_forged_or_changed_permit_fails_before_backend_factory():
    request = OriginalRequest("test", "Put the tomato sauce in the basket.")
    permit = authorize(request, RAW, CONTRACT)
    touched = []

    def factory():
        touched.append(True)

    for bad_request, bad_permit in (
        (request, replace(permit, proof_sha256="forged")),
        (OriginalRequest("test", request.text + " Open the door."), permit),
        (request, {"complete_coverage": True}),
    ):
        with pytest.raises(ValueError):
            execute_covered(
                bad_request, bad_permit, CONTRACT, factory, None, lambda *a, **k: None
            )
    assert not touched


def test_extra_model_coverage_flag_is_not_an_authority():
    raw = RAW[:-1] + ',"complete_coverage":true}'
    result = adjudicate_covered(
        OriginalRequest("x", "Place the tomato sauce in the basket."), raw, CONTRACT
    )
    assert result["schema_status"] == "EXPLICIT_FAILURE"


class CharTokenizer:
    eos_token_id = 999999

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class Ids(list):
    def __getitem__(self, item):
        value = super().__getitem__(item)
        return Ids(value) if isinstance(item, slice) else value

    def tolist(self):
        return list(self)


def test_slot_decoder_allows_structure_not_an_answer():
    slots = evidence_slots()
    decoder = SlotDecoder(CharTokenizer(), slots, 2)
    ids = Ids([1, 2])
    # Deliberately select different valid numbers and enums, including unknown.
    text = "".join(options[-1] for options in slots)
    for char in text:
        assert ord(char) in decoder(0, ids)
        ids.append(ord(char))
    assert decoder(0, ids) == [CharTokenizer.eos_token_id]
    assert len(parse_evidence(text)) == 14


def test_identical_or_contradictory_proposals_cannot_be_evidence():
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((5, 30, 20, 60), fill="red")
    row = {}
    for view in range(2):
        row.update(
            {
                f"t{view}": [50, 300, 200, 600],
                f"b{view}": [600, 200, 950, 700],
                f"i{view}": "match",
                f"j{view}": "match",
                f"v{view}": "visible",
                f"w{view}": "visible",
                f"r{view}": "inside",
            }
        )
    raw = json.dumps(row)
    result = verify(
        [image, image],
        raw,
        raw,
        raw,
        [image.crop((5, 30, 20, 60)), image],
        dict(t=0.1, b=0.1, flip_iou=0.5, inside_fraction=0.8, outside_fraction=0.05),
    )
    assert result["status"] == "unknown"
    assert all(not v["checks"]["mask_response"] for v in result["views"])
    assert all(not v["checks"]["t_flip"] for v in result["views"])
    transformed = mask_targets([image, image], row)
    assert rgb_hash(image) != rgb_hash(transformed[0])


def test_interventions_change_actual_rgb_and_temporal_check_has_no_labels():
    image = Image.new("RGB", (360, 360), "white")
    ImageDraw.Draw(image).rectangle((10, 20, 100, 220), fill="red")
    variants = interventions(
        [image, ImageOps.mirror(image)], Image.new("RGB", image.size, "black")
    )
    assert variants["cropped"][0].size == (240, 240)
    assert variants["mismatched"][0].tobytes() == image.tobytes()
    assert variants["mismatched"][1].tobytes() != variants["real"][1].tobytes()
    history = EvidenceHistory()
    assert history.record("opaque", None, {"status": "complete"})["status"] == "unknown"
    assert history.record("opaque", 10, {"status": "complete"})["status"] == "complete"
    assert history.record("opaque", 9, {"status": "complete"})["status"] == "unknown"
    assert history.record("opaque", 11, {"status": "incomplete"})["status"] == "unknown"


def test_covered_goal_materializes_only_after_verified_permit(monkeypatch):
    from actionstream.llm_vla import integration

    request = OriginalRequest("test", "Place the tomato sauce in the basket.")
    permit = authorize(request, RAW, CONTRACT)
    calls = []
    monkeypatch.setattr(
        integration,
        "execute",
        lambda spec, port, *args, **kwargs: calls.append((spec.target, port)),
    )
    execute_covered(
        request, permit, CONTRACT, lambda: "fake-port", None, lambda *a, **k: None
    )
    assert calls == [("tomato_sauce", "fake-port")]


def test_unknown_is_not_a_visual_pass():
    from actionstream.llm_vla.research_scoring import score_vision

    annotations = [
        dict(
            observation_id=str(i),
            split="diagnostic_evaluation",
            source_episode="x",
            native_is_success=i < 3,
            views=[],
        )
        for i in range(12)
    ]
    result = score_vision(dict(status="COMPLETED", cases=[], verified=[]), annotations)
    assert result["status"] == "FAIL"
    assert result["groups"]["diagnostic_evaluation"]["valid_coverage"] == 0


def test_verified_regions_can_complete_and_cross_view_conflict_is_unknown():
    images = [Image.new("RGB", (100, 100), "white") for _ in range(2)]
    for image in images:
        ImageDraw.Draw(image).rectangle((20, 30, 40, 60), fill="red")
    proposal = {}
    for view in range(2):
        proposal.update(
            {
                f"t{view}": [200, 300, 400, 600],
                f"b{view}": [100, 200, 600, 800],
                f"i{view}": "match",
                f"j{view}": "match",
                f"v{view}": "partial",
                f"w{view}": "visible",
                f"r{view}": "inside",
            }
        )
    flip = dict(proposal)
    mask = dict(proposal)
    for view in range(2):
        for key in "tb":
            b = proposal[f"{key}{view}"]
            flip[f"{key}{view}"] = [1000 - b[2], b[1], 1000 - b[0], b[3]]
        mask[f"v{view}"] = "absent"
        mask[f"r{view}"] = "unknown"
    refs = [images[0].crop((20, 30, 40, 60)), images[0].crop((10, 20, 60, 80))]
    thresholds = dict(
        t=0.99, b=0.99, flip_iou=0.5, inside_fraction=0.8, outside_fraction=0.05
    )
    result = verify(
        images,
        json.dumps(proposal),
        json.dumps(flip),
        json.dumps(mask),
        refs,
        thresholds,
    )
    assert result["status"] == "complete"
    # A contradictory stated relation is not repaired to the geometry's answer.
    proposal["r1"] = flip["r1"] = "outside"
    result = verify(
        images,
        json.dumps(proposal),
        json.dumps(flip),
        json.dumps(mask),
        refs,
        thresholds,
    )
    assert result["status"] == "unknown"
