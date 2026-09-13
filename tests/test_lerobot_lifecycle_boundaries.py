"""Generation boundaries with deterministic handoffs; no simulator or GPU."""

from __future__ import annotations

import multiprocessing
import threading

import pytest
import torch

from actionstream.inference_transport import (
    DirectInferenceTransport,
    InferenceCancelled,
    ProcessInferenceTransport,
)
from test_lerobot_lifecycle import make_engine, wait_until


def test_provider_reset_retries_before_any_policy_call():
    operations = []

    def reset_provider():
        operations.append("reset")
        if len(operations) == 1:
            raise ConnectionError("partial reset")

    def infer(obs, task):
        operations.append("infer")
        assert operations == ["reset", "reset", "infer"]
        return torch.ones(1, 4, 2)

    engine = make_engine(infer, retry_backoff_s=0)
    engine._reset_provider_override = reset_provider
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({})
        wait_until(lambda: engine.telemetry.chunks_accepted == 1)
        assert not engine.failed
        assert operations == ["reset", "reset", "infer"]
        assert engine.telemetry.inference_errors == 1
        assert engine.telemetry.recoveries == 1
    finally:
        engine.stop()


def test_observation_during_transport_teardown_waits_for_reset_completion():
    entered, release = threading.Event(), threading.Event()
    inspected = threading.Event()
    calls = []

    class Transport(DirectInferenceTransport):
        def reset(self):
            entered.set()
            assert release.wait(3)
            return super().reset()

    def infer(obs, task):
        assert release.is_set(), "New request entered the old teardown"
        calls.append(obs["value"])
        return torch.full((1, 4, 2), float(obs["value"]))

    engine = make_engine(infer)
    engine._transport = Transport(infer)
    original_wait = engine._condition.wait_for

    def observe_wait(predicate, timeout=None):
        def checked():
            value = predicate()
            if engine._resetting and engine._latest_observation is not None:
                assert not value
                inspected.set()
            return value

        return original_wait(checked, timeout)

    engine._condition.wait_for = observe_wait
    engine.start()
    engine.resume()
    errors = []

    def reset():
        try:
            engine.reset()
        except BaseException as exc:
            errors.append(exc)

    resetter = threading.Thread(target=reset)
    resetter.start()
    try:
        assert entered.wait(3)
        engine.notify_observation({"value": 2})
        assert inspected.wait(3)
        assert not engine.ready
        assert engine.get_action(None) is None
        release.set()
        resetter.join(3)
        assert not resetter.is_alive() and not errors
        wait_until(lambda: engine.telemetry.chunks_accepted == 1)
        assert calls == [2]
        assert engine.telemetry.inference_errors == 0
        torch.testing.assert_close(engine.get_action(None), torch.tensor([2.0, 2.0]))
    finally:
        release.set()
        resetter.join(3)
        engine.stop()


def test_invalidated_process_entry_cannot_start_an_old_child():
    entered, release = threading.Event(), threading.Event()

    class PausedEntry(ProcessInferenceTransport):
        def infer_cancellable(self, observation, task, **kwargs):
            if observation["value"] == 1:
                entered.set()
                assert release.wait(3)
            return super().infer_cancellable(observation, task, **kwargs)

    transport = PausedEntry("actionstream.transport_stress_fixture:create_transport")
    previous_children = {p.pid for p in multiprocessing.active_children()}
    engine = make_engine(lambda obs, task: None, max_consecutive_failures=1)
    engine._transport = transport
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"value": 1})
        assert entered.wait(3)
        engine.reset()
        engine.notify_observation({"value": 2})
        release.set()
        wait_until(
            lambda: engine.failed or engine.telemetry.chunks_accepted == 1, seconds=15
        )
        assert not engine.failed
        assert transport.process_restarts == 1
        assert engine.telemetry.inference_errors == 0
        torch.testing.assert_close(engine.get_action(None), torch.tensor([2.0, 2.0]))
    finally:
        release.set()
        engine.stop()
    assert not ({p.pid for p in multiprocessing.active_children()} - previous_children)


def test_cancelled_token_prevents_process_start_without_an_engine():
    transport = ProcessInferenceTransport(
        "actionstream.transport_stress_fixture:create_transport"
    )
    cancelled = threading.Event()
    cancelled.set()
    try:
        with pytest.raises(InferenceCancelled):
            transport.infer_cancellable(
                {}, "task", timeout_s=1, cancellation_event=cancelled
            )
        assert transport.process_restarts == 0
        assert transport.latest_request_latency_s is None
    finally:
        transport.close()


def test_old_direct_call_cannot_restore_reset_transport_latency():
    entered, release = threading.Event(), threading.Event()

    def infer(obs, task):
        entered.set()
        assert release.wait(3)
        return torch.ones(1, 4, 2)

    transport = DirectInferenceTransport(infer)
    thread = threading.Thread(target=lambda: transport.infer({}, "task", timeout_s=5))
    thread.start()
    try:
        assert entered.wait(3)
        transport.reset()
        release.set()
        thread.join(3)
        assert not thread.is_alive()
        assert transport.latest_request_latency_s is None
    finally:
        release.set()
        thread.join(3)
        transport.close()


def test_stop_start_does_not_replay_queue_or_last_action():
    engine = make_engine(lambda obs, task: torch.full((1, 4, 2), float(obs["value"])))
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"value": 1})
        wait_until(lambda: engine.telemetry.chunks_accepted == 1)
        packet = engine.get_action_with_revision(None)
        assert engine.is_action_current(packet)
        engine.stop()
        assert not engine.is_action_current(packet)
        assert engine.get_action(None) is None
        engine.start()
        engine.resume()
        assert engine.get_action(None) is None
        assert engine.telemetry.queue_depth == 0
        assert engine.telemetry.inference_started == 0
        engine.notify_observation({"value": 2})
        wait_until(lambda: engine.telemetry.chunks_accepted == 1)
        torch.testing.assert_close(engine.get_action(None), torch.tensor([2.0, 2.0]))
    finally:
        engine.stop()


def test_fatal_requires_stop_start_and_does_not_clear_caller_shutdown():
    def infer(obs, task):
        if obs["value"] == 1:
            raise ConnectionError("current episode failed")
        return torch.full((1, 4, 2), float(obs["value"]))

    engine = make_engine(infer, max_consecutive_failures=1)
    shutdown = threading.Event()
    engine._global_shutdown_event = shutdown
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"value": 1})
        wait_until(lambda: engine.failed)
        assert shutdown.is_set()
        assert not engine.ready
        assert engine.telemetry.disconnects == 1
        engine.stop()
        engine.start()
        engine.resume()
        assert engine.ready and not engine.failed
        assert engine.failure_traceback is None
        assert shutdown.is_set(), "Only the caller may clear its shutdown signal"
        engine.notify_observation({"value": 2})
        wait_until(lambda: engine.telemetry.chunks_accepted == 1)
        assert engine.telemetry.recoveries == 0
        assert engine.telemetry.inference_errors == 0
        engine.notify_observation({"value": 1})
        wait_until(lambda: engine.failed)
        engine.reset()
        assert engine.failed and shutdown.is_set()
    finally:
        engine.stop()


def test_failed_direct_stop_prevents_a_second_owner_until_successful_stop():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def infer(obs, task):
        calls.append(obs["value"])
        if obs["value"] == 1:
            entered.set()
            assert release.wait(3)
        return torch.full((1, 4, 2), float(obs["value"]))

    engine = make_engine(infer, join_timeout_s=0.01)
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"value": 1})
        assert entered.wait(3)
        owner = engine._worker
        engine.stop()
        assert engine.failed and owner.is_alive()
        assert engine.get_action(None) is None
        engine.start()
        assert engine._worker is owner and calls == [1]
        assert not engine.ready
        release.set()
        owner.join(3)
        assert not owner.is_alive()
        engine.stop()
        engine.start()
        engine.resume()
        engine.notify_observation({"value": 2})
        wait_until(lambda: engine.telemetry.chunks_accepted == 1)
        assert not engine.failed and calls == [1, 2]
        torch.testing.assert_close(engine.get_action(None), torch.tensor([2.0, 2.0]))
    finally:
        release.set()
        engine.stop()


def test_serialized_dispatcher_rejects_reset_task_and_foreign_receipts():
    engine = make_engine(lambda obs, task: torch.ones(1, 4, 2))
    other = make_engine(lambda obs, task: torch.ones(1, 4, 2))
    # This lock belongs to the consumer, covering both lifecycle and dispatch.
    control = threading.Lock()
    executed = []

    def dispatch(packet):
        with control:
            if not engine.is_action_current(packet):
                return False
            executed.append(packet.action.tolist())
            return True

    engine.reset()
    engine.start()
    engine.resume()
    other.reset()
    other.start()
    try:
        engine.notify_observation({})
        wait_until(lambda: engine.telemetry.chunks_accepted == 1)
        packet = engine.get_action_with_revision(None)
        assert dispatch(packet)
        assert not other.is_action_current(packet)
        with control:
            engine.set_task("new goal")
        assert not dispatch(packet)
        # The legacy tensor path keeps LeRobot's requested/dispatched task lag.
        tail = engine.get_action_with_revision(None)
        assert tail.task == "audit" and engine.dispatched_task == "audit"
        with control:
            engine.reset()
        assert not dispatch(tail)
        assert len(executed) == 1
    finally:
        engine.stop()
        other.stop()


@pytest.mark.parametrize("late", [False, True])
def test_delivery_observer_error_is_owned_by_its_response(late):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def observer(update):
        # Callback is outside both commit locks.
        assert not engine._condition._is_owned()
        assert engine._queue_lock.acquire(blocking=False)
        engine._queue_lock.release()
        if update["status"] == "delivered" and not calls:
            calls.append(dict(update))
            entered.set()
            assert release.wait(3)
            raise RuntimeError("observer failure")

    engine = make_engine(
        lambda obs, task: torch.full((1, 4, 2), float(obs["value"])),
        delivery_scheduler_enabled=True,
    )
    engine._delivery_observer = observer
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"value": 1})
        assert entered.wait(3)
        old_epoch = calls[0]["epoch"]
        if late:
            engine.reset()
            engine.notify_observation({"value": 2})
        release.set()
        if late:
            wait_until(lambda: engine.failed or engine.telemetry.chunks_accepted == 1)
            assert not engine.failed
            assert old_epoch != engine._epoch
            torch.testing.assert_close(
                engine.get_action(None), torch.tensor([2.0, 2.0])
            )
        else:
            wait_until(lambda: engine.failed)
            assert "observer failure" in engine.failure_traceback
    finally:
        release.set()
        engine.stop()


def test_old_delay_provider_error_does_not_fail_next_episode():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def delay(ordinal):
        calls.append(ordinal)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
            raise ValueError("old scheduling failure")
        return 0

    engine = make_engine(
        lambda obs, task: torch.full((1, 4, 2), float(obs["value"])),
        delivery_scheduler_enabled=True,
    )
    engine._delivery_delay_provider = delay
    engine.reset()
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"value": 1})
        assert entered.wait(3)
        engine.reset()
        engine.notify_observation({"value": 2})
        release.set()
        wait_until(lambda: engine.failed or engine.telemetry.chunks_accepted == 1)
        assert not engine.failed
        torch.testing.assert_close(engine.get_action(None), torch.tensor([2.0, 2.0]))
    finally:
        release.set()
        engine.stop()


def test_late_telemetry_failure_does_not_increment_new_episode_errors():
    entered, release = threading.Event(), threading.Event()
    seen = []

    class Sink:
        def emit(self, event, **fields):
            if event == "action_dequeued":
                seen.append(fields)
                entered.set()
                assert release.wait(3)
                raise OSError("old write failed")

    engine = make_engine(lambda obs, task: torch.ones(1, 4, 2))
    engine._telemetry_sink = Sink()
    engine.reset()
    engine.start()
    engine.resume()
    packets = []
    consumer = None
    try:
        engine.notify_observation({})
        wait_until(lambda: engine.telemetry.chunks_accepted == 1)
        consumer = threading.Thread(
            target=lambda: packets.append(engine.get_action_with_revision(None))
        )
        consumer.start()
        assert entered.wait(3)
        engine.reset()
        release.set()
        consumer.join(3)
        assert not consumer.is_alive()
        assert seen[0]["epoch"] != engine._epoch
        assert engine.telemetry.telemetry_write_errors == 0
        assert not engine.is_action_current(packets[0])
    finally:
        release.set()
        if consumer:
            consumer.join(3)
        engine.stop()
