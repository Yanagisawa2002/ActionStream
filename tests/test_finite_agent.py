"""Meaningful authorization, history, stop and finite-budget regressions."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from actionstream.llm_vla.agent_cli import execute_parsed
from actionstream.llm_vla.coverage import authorize
from actionstream.llm_vla.finite_agent import TemporalVerifier, execute_request
from actionstream.llm_vla.grounding import OriginalRequest

ROOT = Path(__file__).parents[1]
CONFIG = json.loads((ROOT / "configs/finite_agent_language.json").read_text())
REQUEST = OriginalRequest("test", "Put the tomato sauce in the basket.")
RAW = '{"decision":"accept","target_quote":"tomato sauce","destination_quote":"basket"}'


def frame(control, request_id="test", revision=0):
    values = dict(
        request_id=request_id,
        revision=revision,
        observation_id=f"o{control}",
        captured_monotonic=float(control + 1),
    )
    pixels = {
        k: np.full((1, 360, 360, 3), (control + i) % 256, np.uint8)
        for i, k in enumerate(("image", "image2"))
    }
    return SimpleNamespace(**values, pixels=pixels, binding=lambda: dict(values))


class Predictor:
    def __init__(self, decisions=(), default="complete"):
        self.decisions = iter(decisions)
        self.default = default
        self.clips = []

    def predict(self, clip):
        assert clip.shape == (1, 3, 2, 192, 192, 3) and clip.dtype == np.uint8
        self.clips.append(clip[0, :, :, 0, 0, 0].tolist())
        decision = next(self.decisions, self.default)
        return np.array(
            [
                {
                    "complete": [0.01, 0.98, 0.01],
                    "incomplete": [0.98, 0.01, 0.01],
                    "unknown": [0.3, 0.4, 0.3],
                }[decision]
            ]
        )


class Port:
    def __init__(self, bad_chunk=False, fail_post=False):
        self.steps, self.holds, self.infers = [], [], 0
        self.bad_chunk, self.fail_post = bad_chunk, fail_post
        self.closed = False

    def start(self, request_id, revision):
        return frame(0, request_id, revision)

    def infer(self, observation, instruction):
        self.infers += 1
        assert instruction == "pick up the tomato sauce and place it in the basket"
        return np.zeros((29 if self.bad_chunk else 30, 7)), {}

    def begin_hold(self):
        self.holds.append(len(self.steps))

    def step(self, action, request_id, revision, control):
        if self.fail_post and self.holds:
            raise RuntimeError("physical post-stop failure")
        self.steps.append(action.copy())
        return frame(control, request_id, revision)

    def close(self):
        self.closed = True


def run(port, model):
    events = []
    result = execute_request(
        REQUEST,
        authorize(REQUEST, RAW, CONFIG["contract"]),
        CONFIG["contract"],
        lambda: port,
        model,
        lambda event, **row: events.append(dict(event=event, **row)),
    )
    return result, events


@pytest.mark.parametrize("change", ["text", "proof", "contract", "type"])
def test_invalid_permission_never_constructs_execution_port(change):
    request, contract = REQUEST, dict(CONFIG["contract"])
    permit = authorize(request, RAW, contract)
    if change == "text":
        request = OriginalRequest("test", request.text + " Close the drawer.")
    elif change == "proof":
        permit = replace(permit, proof_sha256="forged")
    elif change == "contract":
        contract["version"] = "changed"
    else:
        permit = {"decision": "accept"}
    touched = []
    with pytest.raises(ValueError):
        execute_request(
            request,
            permit,
            contract,
            lambda: touched.append(True),
            None,
            lambda *a, **k: None,
        )
    assert not touched


def test_stored_accept_flag_cannot_authorize_an_extra_obligation(tmp_path):
    parsed = dict(
        original=dict(request_id="test", text=REQUEST.text + " Open the door."),
        call=dict(status="COMPLETED", raw_output=RAW),
        verdict=dict(decision="accept"),
    )
    touched = []
    result = execute_parsed(
        parsed, CONFIG, 1, tmp_path, lambda: touched.append(True), None
    )
    assert result["status"] == "LANGUAGE_BLOCKED" and not touched


def test_full_history_precedes_eleven_positive_observations():
    model, verifier = Predictor(), None
    verifier = TemporalVerifier(model, "test")
    results = [verifier.update(c, frame(c)) for c in range(21)]
    assert all(r["probabilities"] is None and not r["confirmed"] for r in results[:10])
    assert not any(r["confirmed"] for r in results[:20]) and results[20]["confirmed"]
    assert model.clips[0] == [[0, 1], [5, 6], [10, 11]]


@pytest.mark.parametrize("break_decision", ["unknown", "incomplete"])
def test_unknown_or_negative_restarts_confirmation(break_decision):
    model = Predictor(["complete"] * 5 + [break_decision])
    result, _ = run(Port(), model)
    assert result["status"] == "complete" and result["first_claim_control"] == 26


def test_gap_requires_fresh_history_and_fresh_confirmation():
    verifier = TemporalVerifier(Predictor(), "test")
    for c in range(20):
        assert not verifier.update(c, frame(c))["confirmed"]
    rows = [verifier.update(c, frame(c)) for c in range(21, 42)]
    assert not any(r["history_ready"] for r in rows[:10])
    assert not any(r["confirmed"] for r in rows[:-1]) and rows[-1]["confirmed"]


@pytest.mark.parametrize(
    "change", ["request", "revision", "reuse", "capture", "control"]
)
def test_foreign_or_reused_evidence_is_rejected(change):
    verifier = TemporalVerifier(Predictor(), "test")
    verifier.update(0, frame(0))
    value, c = frame(1), 1
    if change == "request":
        value.request_id = "other"
    elif change == "revision":
        value.revision = 1
    elif change == "reuse":
        value.observation_id = "o0"
    elif change == "capture":
        value.captured_monotonic = 1.0
    else:
        c = 0
    with pytest.raises(ValueError):
        verifier.update(c, value)


def test_confirmed_stop_discards_chunk_and_executes_all_post_stop_controls():
    port = Port()
    result, events = run(port, Predictor())
    assert result["first_claim_control"] == 20 and result["discarded_actions"] == 10
    assert port.infers == 1 and port.holds == [20] and port.closed
    assert result["control_steps"] == 20 and result["post_stop_controls"] == 40
    assert len(port.steps) == 60 and all(a[6] == -1 for a in port.steps[20:])
    assert sum(r["event"] == "verification" for r in events) == 21


def test_all_unknown_exhausts_frozen_budget_without_success_or_extra_vla():
    port = Port()
    result, _ = run(port, Predictor(default="unknown"))
    assert (
        result["status"] == "unconfirmed_horizon"
        and result["first_claim_control"] is None
    )
    assert len(port.steps) == 360 and port.infers == 10 and port.holds == [300]
    assert result["post_stop_controls"] == 0 and port.closed


def test_invalid_chunk_never_reaches_physical_step():
    port = Port(bad_chunk=True)
    result, _ = run(port, Predictor())
    assert result["status"] == "ERROR" and not port.steps and port.closed


def test_failed_post_stop_is_error_even_after_confirmation():
    port = Port(fail_post=True)
    result, _ = run(port, Predictor())
    assert result["status"] == "ERROR" and result["first_claim_control"] == 20
    assert result["post_stop_controls"] == 1 and len(port.steps) == 20 and port.closed


def test_new_instance_does_not_inherit_prior_confirmation():
    first = TemporalVerifier(Predictor(), "test")
    for c in range(21):
        last = first.update(c, frame(c))
    assert last["confirmed"]
    assert not TemporalVerifier(Predictor(), "next").update(0, frame(0, "next"))[
        "confirmed"
    ]


def test_nonfinite_model_output_cannot_stop():
    model = SimpleNamespace(predict=lambda clip: np.array([[0, float("nan"), 0]]))
    port = Port()
    result, _ = run(port, model)
    assert result["status"] == "ERROR" and result["first_claim_control"] is None
    assert len(port.steps) == 10


def test_acceptance_is_disjoint_and_keeps_frozen_combination():
    protocol = json.loads(
        (ROOT / "configs/finite_agent_acceptance_v1.json").read_text()
    )
    old = json.loads((ROOT / "configs/completion_acceptance_v3.json").read_text())
    seeds = [r["seed"] for r in protocol["requests"] if "seed" in r]
    assert len(set(seeds)) == len(seeds) == 20 and not set(seeds) & set(old["seeds"])
    assert protocol["checkpoint_sha256"] == old["checkpoint_sha256"]
    assert protocol["frozen_behavior"]["thresholds"] == old["thresholds"]
    assert (
        protocol["frozen_behavior"]["confirmation_controls"]
        == old["confirmation_controls"]
    )


@pytest.mark.parametrize(
    "failure",
    [
        "false_accept",
        "schema_error",
        "missing_episode",
        "seed",
        "premature",
        "missed",
        "unstable",
        "no_completion",
        "late",
    ],
)
def test_whole_agent_gate_rejects_failures_and_denominator_changes(failure):
    from actionstream.llm_vla.finite_scoring import score_run

    p = json.loads((ROOT / "configs/finite_agent_acceptance_v1.json").read_text())
    language = [
        dict(
            id=r["id"],
            decision="accept" if "seed" in r else "reject",
            model_status="COMPLETED",
            schema_status="VALID",
        )
        for r in p["requests"]
    ]
    episodes = [
        dict(
            id=r["id"],
            seed=r["seed"],
            outcome=dict(status="complete"),
            score=dict(
                first_truth_control=100,
                premature_stop=False,
                missed_completed_event=False,
                post_stop_stable=True,
                integrity_passed=True,
                confirmation_delay_s=0.5,
            ),
        )
        for r in p["requests"]
        if "seed" in r
    ]
    assert score_run(p, language, episodes)["status"] == "GO"
    if failure == "false_accept":
        language[-1]["decision"] = "accept"
    elif failure == "schema_error":
        language[-1]["schema_status"] = "EXPLICIT_FAILURE"
    elif failure == "missing_episode":
        episodes.pop()
    elif failure == "seed":
        episodes[0]["seed"] = 1
    elif failure == "no_completion":
        for e in episodes:
            e["score"].update(
                first_truth_control=None,
                confirmation_delay_s=None,
                post_stop_stable=None,
            )
            e["outcome"]["status"] = "unconfirmed_horizon"
    else:
        key, value = {
            "premature": ("premature_stop", True),
            "missed": ("missed_completed_event", True),
            "unstable": ("post_stop_stable", False),
            "late": ("confirmation_delay_s", 2.05),
        }[failure]
        episodes[0]["score"][key] = value
    assert score_run(p, language, episodes)["status"] == "NO-GO"


def test_agent_help_has_no_gpu_or_simulator_startup():
    import os
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-m", "actionstream.llm_vla.agent_cli", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert (
        result.returncode == 0
        and "--checkpoint" in result.stdout
        and "--request" in result.stdout
    )
