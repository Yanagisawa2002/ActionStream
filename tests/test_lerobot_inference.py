from __future__ import annotations

import json
import multiprocessing
import threading
import time
from types import SimpleNamespace

import pytest
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
    delivery_delay_provider=None,
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
        delivery_delay_provider=delivery_delay_provider,
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


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_request_budget_interval_requires_a_positive_int(value) -> None:
    with pytest.raises(ValueError, match="minimum_request_interval_steps"):
        ActionStreamInferenceConfig(minimum_request_interval_steps=value)


def test_request_budget_caps_submissions_and_reset_clears_budget_state() -> None:
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def infer(obs, _task):
        calls.append(obs["step"])
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        return torch.full((1, 30, 2), float(obs["step"]))

    engine = _engine(
        infer,
        config=ActionStreamInferenceConfig(minimum_request_interval_steps=5),
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"step": 0})
        assert entered.wait(2)
        for step in range(1, 5):
            engine.notify_observation({"step": step})

        telemetry = engine.telemetry
        assert telemetry.observations_received == 5
        assert telemetry.observations_skipped_by_budget == 4
        assert telemetry.observations_superseded == 0

        release.set()
        _wait_for(lambda: engine.telemetry.inference_completed == 1)
        engine.notify_observation({"step": 5})
        _wait_for(lambda: len(calls) == 2)
        assert calls == [0, 5]

        engine.reset()
        engine.notify_observation({"step": 99})
        _wait_for(lambda: len(calls) == 3)
        assert calls == [0, 5, 99]
        assert engine.telemetry.observations_skipped_by_budget == 0
    finally:
        engine.stop()


def test_delivery_scheduler_does_not_block_inference_worker() -> None:
    calls: list[int] = []

    def infer(obs, _task):
        calls.append(obs["step"])
        return torch.full((1, 30, 2), float(obs["step"]))

    engine = _engine(
        infer,
        config=ActionStreamInferenceConfig(
            latest_only_fallback=False,
            delivery_scheduler_enabled=True,
        ),
        delivery_delay_provider=lambda _ordinal: 1.0,
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"step": 0})
        _wait_for(lambda: engine.telemetry.inference_completed == 1)
        engine.notify_observation({"step": 1})
        _wait_for(lambda: engine.telemetry.inference_completed == 2)

        assert calls == [0, 1]
        assert engine.telemetry.responses_scheduled == 2
        assert engine.telemetry.responses_delivered == 0
        assert engine.telemetry.pending_responses == 2
        assert engine.telemetry.queue_depth == 0
    finally:
        engine.stop()


def test_delivery_scheduler_reset_rejects_pending_pre_reset_response() -> None:
    def infer(obs, _task):
        return torch.full((1, 4, 2), float(obs["value"]))

    engine = ActionStreamInferenceEngine(
        policy=SimpleNamespace(config=SimpleNamespace(use_amp=False)),
        preprocessor=_Resettable(),
        postprocessor=_Resettable(),
        hw_features={},
        task="task A",
        device="cpu",
        robot_type="mock",
        config=ActionStreamInferenceConfig(
            latest_only_fallback=False,
            delivery_scheduler_enabled=True,
        ),
        infer_chunk=infer,
        reset_provider=lambda: None,
        delivery_delay_provider=lambda _ordinal: 0.1,
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"value": 1})
        _wait_for(lambda: engine.telemetry.responses_scheduled == 1)
        engine.reset()
        engine.resume()
        engine.notify_observation({"value": 2})
        _wait_for(lambda: engine.telemetry.chunks_accepted == 1)

        torch.testing.assert_close(engine.get_action(None), torch.tensor([2.0, 2.0]))
        assert engine.telemetry.chunks_rejected_reset == 1
        assert engine.telemetry.responses_delivered == 1
    finally:
        engine.stop()


def test_delivery_scheduler_rejects_out_of_order_older_response() -> None:
    def infer(obs, _task):
        return torch.full((1, 8, 2), float(obs["step"]))

    engine = ActionStreamInferenceEngine(
        policy=SimpleNamespace(config=SimpleNamespace(use_amp=False)),
        preprocessor=_Resettable(),
        postprocessor=_Resettable(),
        hw_features={},
        task="task A",
        device="cpu",
        robot_type="mock",
        config=ActionStreamInferenceConfig(
            latest_only_fallback=False,
            delivery_scheduler_enabled=True,
        ),
        infer_chunk=infer,
        reset_provider=lambda: None,
        delivery_delay_provider=lambda ordinal: 0.2 if ordinal == 0 else 0.01,
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"step": 0})
        _wait_for(lambda: engine.telemetry.inference_completed == 1)
        engine.notify_observation({"step": 1})
        _wait_for(lambda: engine.telemetry.responses_delivered == 1)
        _wait_for(
            lambda: engine.telemetry.responses_rejected_out_of_order == 1
        )

        torch.testing.assert_close(engine.get_action(None), torch.tensor([1.0, 1.0]))
        assert engine.telemetry.chunks_rejected_stale == 1
        assert engine.telemetry.stale_actions_discarded == 8
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


def test_timeout_retries_the_only_observation_and_recovers() -> None:
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
        _wait_for(lambda: engine.telemetry.chunks_accepted == 1)
        torch.testing.assert_close(engine.get_action(None), torch.tensor([2.0, 2.0]))
        telemetry = engine.telemetry
        assert telemetry.disconnects == 1
        assert telemetry.recoveries == 1
        assert telemetry.consecutive_failures == 0
        assert telemetry.inference_completed == 1
    finally:
        engine.stop()


def test_disconnect_automatically_retries_and_recovery_is_observable() -> None:
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
        _wait_for(lambda: engine.telemetry.chunks_accepted == 1)
        assert not engine.failed
        assert engine.telemetry.disconnects == 1
        assert engine.telemetry.recoveries == 1
    finally:
        engine.stop()


def test_failed_request_does_not_overwrite_a_newer_mailbox_observation() -> None:
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def infer(obs, _task):
        calls.append(obs["step"])
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
            raise ConnectionError("mock remote unavailable")
        return torch.full((1, 4, 2), float(obs["step"]))

    engine = _engine(
        infer,
        config=ActionStreamInferenceConfig(retry_backoff_s=0.0),
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"step": 0})
        assert entered.wait(2)
        engine.notify_observation({"step": 1})
        engine.notify_observation({"step": 2})
        release.set()
        _wait_for(lambda: engine.telemetry.chunks_accepted == 1)
        assert calls == [0, 2]
        torch.testing.assert_close(engine.get_action(None), torch.tensor([2.0, 2.0]))
        assert engine.telemetry.observations_superseded == 1
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
        assert shutdown.wait(2)
        assert engine.failed
        assert "ConnectionError" in (engine.failure_traceback or "")
        assert engine.get_action(None) is None
    finally:
        engine.stop()


def _process_config(telemetry_path, *, timeout_s: float = 0.15):
    return ActionStreamInferenceConfig(
        inference_timeout_s=timeout_s,
        retry_backoff_s=0.0,
        max_consecutive_failures=3,
        join_timeout_s=2.0,
        transport_mode="process",
        process_transport_factory=(
            "actionstream.transport_stress_fixture:create_transport"
        ),
        process_transport_start_method="spawn",
        process_transport_startup_timeout_s=10.0,
        process_transport_terminate_timeout_s=1.0,
        telemetry_jsonl_path=str(telemetry_path),
    )


def _assert_no_transport_children() -> None:
    assert all(
        child.name != "ActionStreamTransport"
        for child in multiprocessing.active_children()
    )


def test_process_deadline_kills_hung_transport_and_recovers_with_new_observation(
    tmp_path,
) -> None:
    telemetry_path = tmp_path / "deadline.jsonl"
    engine = _engine(
        lambda _obs, _task: torch.empty(0),
        config=_process_config(telemetry_path),
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"mode": "hang", "value": 1})
        _wait_for(lambda: engine.telemetry.transport_process_restarts == 1, timeout=5)
        engine.notify_observation({"mode": "ok", "value": 7})
        _wait_for(lambda: engine.telemetry.inference_timeouts == 1, timeout=5)
        _wait_for(lambda: engine.telemetry.chunks_accepted == 1, timeout=5)
        torch.testing.assert_close(engine.get_action(None), torch.tensor([7.0, 7.0]))
        telemetry = engine.telemetry
        assert telemetry.deadline_enforced is True
        assert telemetry.transport_process_restarts == 2
        assert telemetry.recoveries == 1
    finally:
        engine.stop()

    events = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {event["event"] for event in events} >= {
        "engine_started",
        "inference_timeout",
        "inference_completed",
        "engine_stopped",
    }
    assert all(event["schema_version"] == 1 for event in events)
    _assert_no_transport_children()


def test_process_reset_preempts_repeated_hung_calls_without_orphans(tmp_path) -> None:
    engine = _engine(
        lambda _obs, _task: torch.empty(0),
        config=_process_config(tmp_path / "reset.jsonl", timeout_s=5.0),
    )
    engine.reset()
    engine.start()
    engine.resume()
    try:
        for cycle in range(3):
            engine.notify_observation({"mode": "hang", "value": cycle})
            expected_restarts = cycle + 1
            _wait_for(
                lambda: engine.telemetry.transport_process_restarts
                >= expected_restarts,
                timeout=5,
            )
            started = time.monotonic()
            engine.reset()
            assert time.monotonic() - started < 1.0
            engine.resume()

        engine.notify_observation({"mode": "ok", "value": 9})
        _wait_for(lambda: engine.telemetry.chunks_accepted == 1, timeout=5)
        torch.testing.assert_close(engine.get_action(None), torch.tensor([9.0, 9.0]))
        assert engine.telemetry.transport_cancellations >= 3
    finally:
        engine.stop()
    _assert_no_transport_children()


def test_process_stop_preempts_hung_call_with_bounded_join(tmp_path) -> None:
    engine = _engine(
        lambda _obs, _task: torch.empty(0),
        config=_process_config(tmp_path / "stop.jsonl", timeout_s=30.0),
    )
    engine.reset()
    engine.start()
    engine.resume()
    engine.notify_observation({"mode": "hang"})
    _wait_for(lambda: engine.telemetry.transport_process_restarts == 1, timeout=5)
    started = time.monotonic()
    engine.stop()
    assert time.monotonic() - started < 1.0
    assert not engine.failed
    _assert_no_transport_children()
