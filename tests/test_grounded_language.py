"""CPU authorization boundaries, not model quality or native task evidence."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from actionstream.llm_vla.contracts import CAPABILITY, TaskSpec
from actionstream.llm_vla.gate_runner import sha256
from actionstream.llm_vla.grounding import (
    OriginalRequest,
    adjudicate,
    execute_authorized,
    materialize,
    validate_authorization,
)
from actionstream.llm_vla.repair_scoring import interpret


@pytest.fixture
def contract():
    return json.loads(
        (Path(__file__).parents[1] / "configs/llm_vla_dispatch004.json").read_text(
            encoding="utf-8"
        )
    )["contract"]


def accept(target="tomato sauce", destination="basket"):
    return json.dumps(
        dict(decision="accept", target_quote=target, destination_quote=destination)
    )


def test_original_substitution_and_forged_bindings_cannot_execute(contract):
    request = OriginalRequest(
        "opaque", "Please place the tomato sauce inside the basket."
    )
    task = materialize(request, accept(), contract)
    validate_authorization(request, task, contract)
    changed = OriginalRequest("opaque", "Leave the tomato sauce next to the basket.")

    class ForbiddenPort:
        def start(self, *args):
            pytest.fail("Invalid authorization reached environment reset")

    for original, authorization in (
        (changed, task),
        (request, replace(task, original_sha256="forged")),
        (request, replace(task, target=replace(task.target, start=0))),
        (request, replace(task, model_output_sha256="forged")),
        (request, TaskSpec("1.0", "opaque", "accept", **CAPABILITY, reason="forged")),
    ):
        with pytest.raises(ValueError):
            execute_authorized(
                original,
                authorization,
                contract,
                ForbiddenPort(),
                None,
                lambda *a, **k: None,
            )


def test_quotes_require_exact_unique_alias_spans_not_inferred_context(contract):
    for text, raw in (
        ("Could you carry it to the basket?", accept()),
        ("Place the tomato sauce in the wastebasket.", accept()),
        (
            "Place the tomato sauce near the basket, then inspect the tomato sauce.",
            accept(),
        ),
        ("Put some sauce inside the basket.", accept("sauce")),
    ):
        verdict = adjudicate(OriginalRequest("opaque", text), raw, contract)
        assert verdict["schema_status"] == "VALID"
        assert verdict["decision"] == "explicit_failure" and verdict["task"] is None


def test_explicit_alias_and_case_are_bound_to_original_spelling(contract):
    request = OriginalRequest("opaque", "Place THE JAR OF TOMATO SAUCE in the basket.")
    task = materialize(
        request, accept("THE JAR OF TOMATO SAUCE", "the basket"), contract
    )
    assert (
        request.text[task.target.start : task.target.end] == "THE JAR OF TOMATO SAUCE"
    )
    assert (
        task.control_spec().canonical_instruction == CAPABILITY["canonical_instruction"]
    )
    altered = dict(contract, aliases={"target": ["milk"], "destination": ["basket"]})
    with pytest.raises(ValueError):
        validate_authorization(request, task, altered)


def test_minimal_nonaccept_and_generic_negation_guards_have_no_authorization(contract):
    request = OriginalRequest(
        "opaque", "Never transfer the tomato sauce to the basket."
    )
    for raw in (
        '{"decision":"reject","reason":"negated"}',
        '{"decision":"unknown","reason":"missing_or_ambiguous_reference"}',
        accept(),
    ):
        verdict = adjudicate(request, raw, contract)
        assert verdict["decision"] in ("reject", "unknown") and verdict["task"] is None
        with pytest.raises(ValueError):
            materialize(request, raw, contract)
    malformed = adjudicate(
        request, '{"decision":"accept","decision":"reject"}', contract
    )
    assert (
        malformed["reason"] == "malformed"
        and malformed["schema_status"] == "EXPLICIT_FAILURE"
    )


def test_textual_mentions_are_not_a_general_relation_verifier(contract):
    # An intentionally false semantic model claim can still carry real mentions.
    # This documents the residual boundary; the real negative challenge must
    # assess role reversal independently, rather than claiming host entailment.
    request = OriginalRequest(
        "opaque", "Move the basket into the tomato sauce container."
    )
    verdict = adjudicate(request, accept("tomato sauce container"), contract)
    assert verdict["decision"] == "accept"
    assert "argument roles" in contract["semantic_limit"]


def test_v1_and_v2_scorers_never_repair_each_others_outputs(contract):
    original = {
        "request_id": "opaque",
        "text": "Do not move the tomato sauce into the basket.",
    }
    raw_v1 = json.dumps(
        dict(
            version="1.0",
            request_id="opaque",
            decision="reject",
            skill=None,
            target=None,
            destination=None,
            canonical_instruction=CAPABILITY["canonical_instruction"],
            reason="negated",
        )
    )
    row = dict(
        input=original,
        identity="opaque",
        status="COMPLETED",
        version="v1",
        raw_output=raw_v1,
    )
    assert interpret(row, original, contract)["decision"] == "explicit_failure"
    row.update(version="v2", raw_output='{"decision":"reject","reason":"negated"}')
    assert interpret(row, original, contract)["decision"] == "reject"
    row["version"] = "v1"
    assert interpret(row, original, contract)["decision"] == "explicit_failure"


@pytest.mark.parametrize("failed", ["language", "shadow"])
def test_actual_v2_entrypoint_joint_gate_blocks_model_and_simulator(
    tmp_path, monkeypatch, failed
):
    from actionstream.llm_vla import grounded_native, native_runner

    config = tmp_path / "frozen_config.json"
    config.write_text("{}")
    for phase, directory in (("language", "comparison"), ("shadow", "shadow")):
        target = tmp_path / directory
        target.mkdir()
        receipt = target / "run_receipt.json"
        receipt.write_text("{}")
        (tmp_path / (phase + "_score.json")).write_text(
            json.dumps(
                dict(
                    status="FAIL" if phase == failed else "PASS",
                    source_sha256={
                        "receipt": sha256(receipt),
                        "config": sha256(config),
                    },
                )
            )
        )
    output = tmp_path / "native"
    monkeypatch.setattr(
        "sys.argv",
        [
            "grounded_native",
            "--config",
            str(config),
            "--model",
            str(tmp_path),
            "--output",
            str(output),
            "--base-store",
            str(tmp_path),
            "--evidence",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        native_runner,
        "LocalQwen",
        lambda *a: pytest.fail("Model loaded with a failed joint gate"),
    )
    with pytest.raises(ValueError, match="both intact"):
        grounded_native.run()
    assert not output.exists()
