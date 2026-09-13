"""Audit-only regressions: deterministic scheduling, no GPU or remote calls.

Assertions describe the required contract. Failures reproduce audit findings.
The merge wrapper only pauses a worker at a real unlocked scheduling boundary;
it does not change the production merge or reset behavior.
"""

import threading
import time
from types import SimpleNamespace

import pytest
from actionstream.inference_transport import InferenceDeadlineExceeded
import torch

from actionstream.lerobot_inference import (
    ActionStreamInferenceConfig,
    ActionStreamInferenceEngine,
)


class Resettable:
    config = SimpleNamespace(use_amp=False)

    def reset(self):
        pass


def make_engine(infer, **config):
    return ActionStreamInferenceEngine(
        policy=Resettable(),
        preprocessor=Resettable(),
        postprocessor=Resettable(),
        hw_features={},
        task="audit",
        device="cpu",
        robot_type="mock",
        config=ActionStreamInferenceConfig(**config),
        infer_chunk=infer,
    )


def wait_until(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("audit synchronization timed out")


@pytest.mark.parametrize("pipelined", [False, True])
def test_reset_after_acceptance_check_must_not_repopulate_queue(pipelined, tmp_path):
    before_merge = threading.Event()
    continue_merge = threading.Event()
    merge_finished = threading.Event()
    engine = make_engine(
        lambda obs, task: torch.full((1, 4, 2), 37.0),
        delivery_scheduler_enabled=pipelined,
        telemetry_jsonl_path=str(tmp_path / "events.jsonl"),
    )
    original_merge = engine._merge_chunk

    def paused_merge(*args, **kwargs):
        before_merge.set()
        assert continue_merge.wait(3)
        try:
            return original_merge(*args, **kwargs)
        finally:
            merge_finished.set()

    engine._merge_chunk = paused_merge
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"episode": "old"})
        assert before_merge.wait(3)
        engine.reset()
        assert engine.telemetry.queue_depth == 0
        continue_merge.set()
        assert merge_finished.wait(3)
        leaked_action = engine.get_action(None)
        assert leaked_action is None, (
            "An old-episode action was delivered after reset without a new observation: "
            f"{leaked_action.tolist()}"
        )
    finally:
        continue_merge.set()
        engine.stop()


def test_old_generation_failure_must_not_fail_reset_episode():
    entered = threading.Event()
    release = threading.Event()

    def failing_old_request(obs, task):
        if obs["episode"] == "new":
            return torch.full((1, 4, 2), 73.0)
        entered.set()
        assert release.wait(3)
        raise ConnectionError("audit old-generation disconnect")

    engine = make_engine(
        failing_old_request, max_consecutive_failures=1, retry_backoff_s=0
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"episode": "old"})
        assert entered.wait(3)
        engine.reset()
        engine.notify_observation({"episode": "new"})
        release.set()
        wait_until(lambda: engine.failed or engine.telemetry.chunks_accepted > 0)
        assert not engine.failed, "A pre-reset failure poisoned the new episode"
        torch.testing.assert_close(engine.get_action(None), torch.tensor([73.0, 73.0]))
    finally:
        release.set()
        engine.stop()


@pytest.mark.parametrize("pipelined", [False, True])
def test_control_step_advance_before_merge_uses_current_age(pipelined):
    entered, release, finished = (threading.Event() for _ in range(3))
    engine = make_engine(
        lambda obs, task: torch.arange(8).reshape(1, 4, 2).float(),
        delivery_scheduler_enabled=pipelined,
        latest_only_fallback=False,
    )
    original = engine._merge_chunk

    def pause(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()

    engine._merge_chunk = pause
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"step": 0})
        assert entered.wait(3)
        engine.pause()
        engine.notify_observation({"step": 1})
        engine.notify_observation({"step": 2})
        release.set()
        assert finished.wait(3)
        torch.testing.assert_close(engine.get_action(None), torch.tensor([4.0, 5.0]))
    finally:
        release.set()
        engine.stop()


@pytest.mark.parametrize("pipelined", [False, True])
def test_stop_started_before_merge_must_not_publish_pending_action(pipelined):
    entered, release = threading.Event(), threading.Event()
    engine = make_engine(
        lambda obs, task: torch.ones(1, 4, 2), delivery_scheduler_enabled=pipelined
    )
    original = engine._merge_chunk

    def pause(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    engine._merge_chunk = pause
    engine.reset()
    engine.start()
    engine.resume()
    stopper = None
    try:
        engine.notify_observation({})
        assert entered.wait(3)
        stopper = threading.Thread(target=engine.stop)
        stopper.start()
        assert engine._shutdown.wait(3)
        release.set()
        stopper.join(3)
        assert not stopper.is_alive()
        assert not engine.failed
        assert engine.get_action(None) is None
    finally:
        release.set()
        if stopper:
            stopper.join(3)
        engine.stop()


@pytest.mark.parametrize("outcome", ["success", "timeout"])
def test_reset_isolates_late_result_metrics(outcome):
    entered, release = threading.Event(), threading.Event()

    def infer(obs, task):
        entered.set()
        assert release.wait(3)
        if outcome == "timeout":
            raise InferenceDeadlineExceeded("old request")
        return torch.ones(1, 4, 2)

    engine = make_engine(infer, max_consecutive_failures=1, retry_backoff_s=0)
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({})
        assert entered.wait(3)
        engine.reset()
        release.set()
        wait_until(lambda: engine.failed or engine.telemetry.chunks_rejected_reset > 0)
        assert not engine.failed
        metrics = engine.telemetry
        assert metrics.inference_started == metrics.inference_completed == 0
        assert metrics.inference_errors == metrics.inference_timeouts == 0
        assert metrics.latest_inference_latency_ms is None
    finally:
        release.set()
        engine.stop()


def test_reset_between_failure_accounting_and_fatal_commit():
    entered, release = threading.Event(), threading.Event()

    def infer(obs, task):
        if obs["episode"] == "new":
            return torch.full((1, 4, 2), 73.0)
        raise ConnectionError("old episode")

    engine = make_engine(infer, max_consecutive_failures=1, retry_backoff_s=0)
    original = engine._mark_fatal

    def pause(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    engine._mark_fatal = pause
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"episode": "old"})
        assert entered.wait(3)
        engine.reset()
        engine.notify_observation({"episode": "new"})
        release.set()
        wait_until(lambda: engine.failed or engine.telemetry.chunks_accepted > 0)
        assert not engine.failed
        torch.testing.assert_close(engine.get_action(None), torch.tensor([73.0, 73.0]))
    finally:
        release.set()
        engine.stop()


def test_old_provider_reset_exception_does_not_fail_new_episode():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def provider_reset():
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
            raise ConnectionError("old provider reset failed")

    engine = make_engine(lambda obs, task: torch.full((1, 4, 2), 73.0))
    engine._reset_provider = provider_reset
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"episode": "old"})
        assert entered.wait(3)
        engine.reset()
        engine.notify_observation({"episode": "new"})
        release.set()
        wait_until(lambda: engine.failed or engine.telemetry.chunks_accepted > 0)
        assert not engine.failed, (
            "Old provider-reset exception crosses the epoch boundary"
        )
        torch.testing.assert_close(engine.get_action(None), torch.tensor([73.0, 73.0]))
    finally:
        release.set()
        engine.stop()
