from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import torch

from actionstream.lerobot_inference import (
    ActionStreamInferenceConfig,
    ActionStreamInferenceEngine,
)


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


class _Resettable:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


def _engine(
    infer_chunk,
    *,
    config: ActionStreamInferenceConfig | None = None,
    shutdown_event: threading.Event | None = None,
    reset_provider=None,
) -> ActionStreamInferenceEngine:
    policy = _Resettable()
    policy.config = SimpleNamespace(use_amp=False)
    return ActionStreamInferenceEngine(
        policy=policy,
        preprocessor=_Resettable(),
        postprocessor=_Resettable(),
        hw_features={},
        task="task A",
        device="cpu",
        robot_type="mock",
        config=config,
        shutdown_event=shutdown_event,
        infer_chunk=infer_chunk,
        reset_provider=reset_provider,
    )


def test_aligned_chunk_drops_elapsed_prefix_and_bounded_hold_stops() -> None:
    entered = threading.Event()
    release = threading.Event()

    def infer(_obs, _task):
        entered.set()
        assert release.wait(2)
        return torch.arange(1, 13, dtype=torch.float32).reshape(1, 4, 3)

    engine = _engine(
        infer,
        config=ActionStreamInferenceConfig(bounded_hold_steps=2),
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"step": 0})
        assert entered.wait(2)
        engine.notify_observation({"step": 1})
        engine.notify_observation({"step": 2})
        engine.pause()
        release.set()
        _wait_for(lambda: engine.telemetry.chunks_accepted == 1)

        torch.testing.assert_close(
            engine.get_action(None), torch.tensor([7.0, 8.0, 9.0])
        )
        torch.testing.assert_close(
            engine.get_action(None), torch.tensor([10.0, 11.0, 12.0])
        )
        torch.testing.assert_close(
            engine.get_action(None), torch.tensor([10.0, 11.0, 12.0])
        )
        torch.testing.assert_close(
            engine.get_action(None), torch.tensor([10.0, 11.0, 12.0])
        )
        assert engine.get_action(None) is None

        telemetry = engine.telemetry
        assert telemetry.stale_actions_discarded == 2
        assert telemetry.actions_dequeued == 2
        assert telemetry.hold_actions == 2
        assert telemetry.hold_exhausted == 1
        assert telemetry.queue_depth == 0
    finally:
        engine.stop()


def test_latest_mailbox_coalesces_observations_while_one_call_is_in_flight() -> None:
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def infer(obs, _task):
        calls.append(obs["step"])
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        return torch.ones((1, 8, 2), dtype=torch.float32) * (obs["step"] + 1)

    engine = _engine(infer)
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"step": 0})
        assert entered.wait(2)
        engine.notify_observation({"step": 1})
        engine.notify_observation({"step": 2})
        release.set()
        _wait_for(lambda: len(calls) == 2)
        _wait_for(lambda: engine.telemetry.inference_completed == 2)
        assert calls == [0, 2]
        assert engine.telemetry.observations_superseded == 1
    finally:
        engine.stop()


def test_reset_rejects_inflight_chunk_and_resets_provider_on_worker() -> None:
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []
    reset_threads: list[str] = []

    def infer(obs, _task):
        calls.append(obs["value"])
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        return torch.full((1, 4, 2), float(obs["value"]))

    engine = _engine(
        infer,
        reset_provider=lambda: reset_threads.append(threading.current_thread().name),
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"value": 1})
        assert entered.wait(2)
        engine.reset()
        engine.notify_observation({"value": 2})
        release.set()
        _wait_for(lambda: engine.telemetry.chunks_rejected_reset == 1)
        _wait_for(lambda: engine.telemetry.chunks_accepted == 1)
        torch.testing.assert_close(engine.get_action(None), torch.tensor([2.0, 2.0]))
        assert reset_threads == ["ActionStreamInference", "ActionStreamInference"]
    finally:
        engine.stop()


def test_timeout_is_discarded_then_a_fresh_request_recovers() -> None:
    calls = 0

    def infer(_obs, _task):
        nonlocal calls
        calls += 1
        if calls == 1:
            time.sleep(0.04)
        return torch.ones((1, 4, 2), dtype=torch.float32) * calls

    engine = _engine(
        infer,
        config=ActionStreamInferenceConfig(
            inference_timeout_s=0.01,
            retry_backoff_s=0.0,
            max_consecutive_failures=3,
        ),
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"step": 0})
        _wait_for(lambda: engine.telemetry.inference_timeouts == 1)
        assert engine.get_action(None) is None
        engine.notify_observation({"step": 1})
        _wait_for(lambda: engine.telemetry.chunks_accepted == 1)
        torch.testing.assert_close(engine.get_action(None), torch.tensor([2.0, 2.0]))
        telemetry = engine.telemetry
        assert telemetry.disconnects == 1
        assert telemetry.recoveries == 1
        assert telemetry.consecutive_failures == 0
        assert telemetry.inference_completed == 1
    finally:
        engine.stop()


def test_disconnect_is_retried_and_recovery_is_observable() -> None:
    calls = 0

    def infer(_obs, _task):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("mock remote unavailable")
        return torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])

    engine = _engine(
        infer,
        config=ActionStreamInferenceConfig(retry_backoff_s=0.0),
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"step": 0})
        _wait_for(lambda: engine.telemetry.inference_errors == 1)
        engine.notify_observation({"step": 1})
        _wait_for(lambda: engine.telemetry.chunks_accepted == 1)
        assert not engine.failed
        assert engine.telemetry.disconnects == 1
        assert engine.telemetry.recoveries == 1
    finally:
        engine.stop()


def test_fully_stale_chunk_uses_latest_only_only_after_depletion() -> None:
    entered = threading.Event()
    release = threading.Event()

    def infer(_obs, _task):
        entered.set()
        assert release.wait(2)
        return torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    engine = _engine(
        infer,
        config=ActionStreamInferenceConfig(
            bounded_hold_steps=1,
            latest_only_fallback=True,
        ),
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"step": 0})
        assert entered.wait(2)
        engine.notify_observation({"step": 1})
        engine.notify_observation({"step": 2})
        assert engine.get_action(None) is None
        engine.pause()
        release.set()
        _wait_for(lambda: engine.telemetry.fallback_chunks_accepted == 1)
        torch.testing.assert_close(engine.get_action(None), torch.tensor([1.0, 2.0]))
        telemetry = engine.telemetry
        assert telemetry.fallback_activations == 1
        assert telemetry.chunks_rejected_stale == 0
        assert telemetry.stale_actions_discarded == 0
    finally:
        engine.stop()


def test_fully_stale_chunk_is_rejected_while_existing_queue_is_usable() -> None:
    second_entered = threading.Event()
    second_release = threading.Event()
    calls = 0

    def infer(_obs, _task):
        nonlocal calls
        calls += 1
        if calls == 2:
            second_entered.set()
            assert second_release.wait(2)
        return torch.ones((1, 4, 2), dtype=torch.float32) * calls

    engine = _engine(infer)
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"step": 0})
        _wait_for(lambda: engine.telemetry.chunks_accepted == 1)
        engine.notify_observation({"step": 1})
        assert second_entered.wait(2)
        for step in range(2, 6):
            engine.notify_observation({"step": step})
        engine.pause()
        second_release.set()
        _wait_for(lambda: engine.telemetry.chunks_rejected_stale == 1)
        assert engine.telemetry.queue_depth == 4
        assert engine.telemetry.fallback_activations == 0
    finally:
        engine.stop()


def test_repeated_failures_mark_engine_failed_and_signal_rollout_shutdown() -> None:
    shutdown = threading.Event()

    def infer(_obs, _task):
        raise ConnectionError("offline")

    engine = _engine(
        infer,
        shutdown_event=shutdown,
        config=ActionStreamInferenceConfig(
            retry_backoff_s=0.0,
            max_consecutive_failures=2,
        ),
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"step": 0})
        _wait_for(lambda: engine.telemetry.inference_errors == 1)
        engine.notify_observation({"step": 1})
        assert shutdown.wait(2)
        assert engine.failed
        assert "ConnectionError" in (engine.failure_traceback or "")
        assert engine.get_action(None) is None
    finally:
        engine.stop()
