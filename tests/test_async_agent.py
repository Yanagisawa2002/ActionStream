"""Integration tests for the real worker plus adversarial dispatch boundaries."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from actionstream.llm_vla.async_agent import AsyncAgentConfig, execute_async_request
from actionstream.llm_vla.coverage import authorize
from actionstream.llm_vla.grounding import OriginalRequest

CONTRACT = json.loads(Path("configs/finite_agent_language.json").read_text())[
    "contract"
]
REQUEST = OriginalRequest("async-test", "Put the tomato sauce in the basket.")
RAW = '{"decision":"accept","target_quote":"tomato sauce","destination_quote":"basket"}'


class Port:
    def __init__(self, delay=0):
        self.control = 0
        self.steps = []
        self.closed = False
        self.delay = delay
        self.infer_steps = []
        self.infer_threads = []
        self.bindings = {}
        self.owner = threading.get_ident()
        self.pixels = {
            k: np.zeros((1, 360, 360, 3), np.uint8) for k in ("image", "image2")
        }

    def frame(self, request, revision):
        binding = dict(
            request_id=request,
            revision=revision,
            observation_id=f"{revision}:{self.control}",
            captured_monotonic=time.monotonic() - 0.001 + self.control * 1e-8,
        )
        self.bindings[self.control] = dict(binding)
        return SimpleNamespace(
            **binding, binding=lambda: dict(binding), pixels=self.pixels
        )

    def start(self, request, revision):
        return self.frame(request, revision)

    def refresh(self, request, revision):
        return self.frame(request, revision)

    def warmup(self, observation, instruction, predictor):
        pass

    def reset_inference(self):
        assert threading.get_ident() != self.owner

    def infer(self, observation, instruction):
        self.infer_threads.append(threading.get_ident())
        self.infer_steps.append(self.control)
        time.sleep(self.delay)
        actions = np.zeros((30, 7), np.float32)
        actions[:, 0] = 0.25
        return actions, dict(actions=actions.tolist())

    def pose_hold(self, observation, gripper):
        return np.array([0, 0, 0, 0, 0, 0, gripper], np.float32)

    def step(self, action, request, revision, control):
        assert threading.get_ident() == self.owner
        assert control == self.control + 1
        self.control = control
        self.steps.append(action.copy())
        return self.frame(request, revision)

    def close(self):
        self.closed = True


class Predictor:
    def __init__(self, port, complete_after=None):
        self.port, self.after = port, complete_after

    def predict(self, clips):
        done = self.after is not None and self.port.control >= self.after
        return np.array([[0.01, 0.98, 0.01] if done else [0.98, 0.01, 0.01]])


def run(port, complete_after=None, worker_factory=None, **overrides):
    values = dict(
        period_s=0.02,
        observation_max_age_s=1.0,
        policy_controls=35,
        settle_controls=12,
        recovery_open_controls=4,
        post_stop_controls=4,
        request_interval_controls=5,
        max_starvation_controls=40,
    )
    values.update(overrides)
    events = []
    kwargs = {} if worker_factory is None else {"worker_factory": worker_factory}
    result = execute_async_request(
        REQUEST,
        authorize(REQUEST, RAW, CONTRACT),
        CONTRACT,
        lambda: port,
        Predictor(port, complete_after),
        lambda event, **row: events.append(dict(event=event, **row)),
        AsyncAgentConfig(**values),
        **kwargs,
    )
    return result, events


def test_real_engine_advances_physics_while_inference_is_pending():
    port = Port(delay=0.055)
    result, events = run(port, complete_after=20)
    assert result["status"] == "complete", result
    assert port.closed and port.infer_threads
    assert all(thread != port.owner for thread in port.infer_threads)
    chunks = [e for e in events if e["event"] == "async_chunk"]
    assert (
        chunks
        and sum(e["event"] == "dispatch" for e in events[: events.index(chunks[0])])
        >= 2
    )
    stop = next(i for i, e in enumerate(events) if e["event"] == "confirmed_stop")
    assert all(
        e["source"] is None and e["action"][-1] == -1
        for e in events[stop:]
        if e["event"] == "dispatch"
    )
    assert result["post_stop_controls"] == 4


def test_retry_is_bounded_and_clears_rgb_history_and_old_action_revisions():
    port = Port()
    result, events = run(port, complete_after=60, max_attempts=2)
    assert result["status"] == "complete", result
    assert result["attempts"] == 2
    retry = [e for e in events if e["event"] == "recovery_started"]
    assert len(retry) == 1 and retry[0]["control"] == 47
    revision_checks = [
        e
        for e in events
        if e["event"] == "verification" and e["observation"]["revision"] == 1
    ]
    assert len(revision_checks) > 10
    assert all(not e["history_ready"] for e in revision_checks[:10])
    assert revision_checks[10]["history_ready"]
    for event in events:
        if event["event"] == "dispatch" and event["source"]:
            assert event["source"]["observation"]["revision"] == event["revision"]


def test_thread_cold_start_finishes_before_dispatch_and_warmup_actions_are_discarded():
    class ColdPort(Port):
        def infer(self, observation, instruction):
            cold = not self.infer_threads
            self.delay = 0.12 if cold else 0
            actions, _ = super().infer(observation, instruction)
            actions[:, 0] = 0.9 if cold else 0.25
            return actions, dict(actions=actions.tolist())

    port = ColdPort()
    outcome, events = run(port, complete_after=20, max_starvation_controls=3)
    assert outcome["status"] == "complete", outcome
    warm = next(
        i for i, e in enumerate(events) if e["event"] == "worker_warmup_complete"
    )
    assert not any(e["event"] == "dispatch" for e in events[:warm])
    assert outcome["engine_epoch_offset"] == 1
    assert len(set(port.infer_threads)) == 1
    assert all(action[0] != np.float32(0.9) for action in port.steps)
    assert all(
        e["source"]["epoch"] == 1
        for e in events
        if e["event"] == "dispatch" and e["source"]
    )


def test_all_incomplete_exhausts_exact_two_attempts_without_success():
    result, events = run(Port(), max_attempts=2)
    assert result["status"] == "unconfirmed_horizon"
    assert result["control_steps"] == 2 * (35 + 12) + 4
    assert result["first_claim_control"] is None
    assert sum(e["event"] == "recovery_started" for e in events) == 1


class StaleWorker:
    def __init__(self, port, instruction, config, emit):
        self.source = None
        self.age = config.action_source_max_age_s

    def warmup(self, observation):
        return 0

    def publish(self, control, observation):
        if self.source is None:
            binding = dict(
                observation.binding(),
                captured_monotonic=time.monotonic() - self.age - 1,
            )
            self.source = SimpleNamespace(**binding, binding=lambda: dict(binding))

    def take(self):
        return dict(
            action=np.ones(7),
            source=self.source,
            current=True,
            source_control=0,
            epoch=0,
            ordinal=0,
        )

    def close(self):
        return {}


def test_stale_packets_never_reach_physics_and_starvation_is_bounded():
    port = Port()
    result, events = run(port, worker_factory=StaleWorker, max_starvation_controls=3)
    assert result["status"] == "inference_unavailable"
    assert result["control_steps"] == 3
    assert sum(e["event"] == "stale_action_rejected" for e in events) == 4
    assert all(action[0] == 0 for action in port.steps)


def test_foreign_authorization_cannot_create_worker_or_port():
    touched = []
    with pytest.raises(ValueError):
        execute_async_request(
            REQUEST,
            {"decision": "accept"},
            CONTRACT,
            lambda: touched.append(True),
            None,
            lambda *a, **k: None,
        )
    assert not touched


@pytest.mark.parametrize(
    "change", [{"max_attempts": 3}, {"period_s": 0}, {"policy_controls": 0}]
)
def test_budget_cannot_be_unbounded(change):
    with pytest.raises(ValueError):
        AsyncAgentConfig(**change)


def retained_episode(tmp_path):
    port = Port()
    outcome, events = run(port, complete_after=20)
    assert outcome["status"] == "complete", outcome
    facts = [
        dict(
            inside=c >= 20,
            basket_contact=c >= 20,
            finger_contact=False,
            linear_speed=0.0,
            angular_speed=0.0,
            strict_complete=c >= 30,
        )
        for c in range(port.control + 1)
    ]
    payloads = {
        "outcome.json": outcome,
        "private_truth.json": facts,
        "observation_bindings.json": list(port.bindings.values()),
        "actions.json": [v.tolist() for v in port.steps],
    }
    for name, value in payloads.items():
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "runtime.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    np.savez_compressed(
        tmp_path / "trajectory.npz",
        states=np.zeros((port.control + 1, 5)),
        rgb=np.zeros((port.control + 1, 2, 192, 192, 3), np.uint8),
    )
    return events


def test_independent_replay_requires_full_rgb_and_rejects_unsafe_truth(tmp_path):
    from actionstream.llm_vla.async_scoring import score_episode

    retained_episode(tmp_path)
    result = score_episode(tmp_path)
    assert result["integrity_passed"] and result["safely_completed"], result
    path = tmp_path / "private_truth.json"
    facts = json.loads(path.read_text())
    for fact in facts:
        fact.update(inside=False, basket_contact=False, strict_complete=False)
    path.write_text(json.dumps(facts))
    result = score_episode(tmp_path)
    assert result["premature_stop"] and not result["safely_completed"]


def test_independent_replay_rejects_foreign_actions_and_tampered_clip(tmp_path):
    from actionstream.llm_vla.async_scoring import score_episode

    events = retained_episode(tmp_path)
    dispatched = next(e for e in events if e["event"] == "dispatch" and e["source"])
    dispatched["source"]["epoch"] = 999
    checked = next(
        e for e in events if e["event"] == "verification" and e["history_ready"]
    )
    checked["clip_sha256"] = "0" * 64
    (tmp_path / "runtime.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    result = score_episode(tmp_path)
    assert not result["integrity_passed"]
    assert any(e.startswith("action_provenance:") for e in result["integrity_errors"])
    assert any(e.startswith("clip_identity:") for e in result["integrity_errors"])
