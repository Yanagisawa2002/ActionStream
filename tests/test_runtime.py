from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from actionstream.lerobot_backend import (
    immutable_observation_snapshot,
    thaw_observation_snapshot,
)
from actionstream.runtime import (
    ActionQueue,
    InferencePayload,
    InferenceRequest,
    InferenceResult,
    LatestRequestWorker,
    QueueNotReady,
)


def test_observation_request_is_independent_and_immutable() -> None:
    source = {"pixels": np.ones((1, 2), dtype=np.float32), "nested": {"values": [1, 2]}}
    snapshot = immutable_observation_snapshot(source)
    source["pixels"][0, 0] = 9
    assert snapshot["pixels"][0, 0] == 1
    with pytest.raises(ValueError):
        snapshot["pixels"][0, 0] = 3
    with pytest.raises(TypeError):
        snapshot["new"] = "unsafe"

    thawed = thaw_observation_snapshot(snapshot)
    thawed["pixels"][0, 0] = 4
    assert thawed["pixels"].flags.writeable
    assert snapshot["pixels"][0, 0] == 1


def result(episode: str, observation_step: int, rows: int = 6) -> InferenceResult:
    actions = np.repeat(np.arange(1, rows + 1, dtype=np.float32)[:, None], 7, axis=1)
    return InferenceResult(
        actions=actions,
        episode_id=episode,
        observation_control_step=observation_step,
        request_timestamp=1.0,
        start_timestamp=2.0,
        end_timestamp=3.0,
        delivery_timestamp=4.0,
        model_inference_latency_seconds=1.0,
    )


def test_aligned_drops_stale_prefix_by_age() -> None:
    queue = ActionQueue()
    queue.reset_episode("episode")
    outcome = queue.replace(
        result("episode", observation_step=5),
        current_control_step=8,
        mode="async_aligned",
    )
    action, held = queue.next_action()
    assert outcome.age_steps == 3
    assert outcome.dropped_prefix_steps == 3
    assert outcome.queue_length == 3
    assert not held
    assert np.all(action == 4)


def test_fully_stale_chunk_is_rejected_without_destroying_pending_queue() -> None:
    queue = ActionQueue()
    queue.reset_episode("episode")
    queue.replace(result("episode", 0), current_control_step=0, mode="async_naive")
    before = queue.queue_length
    outcome = queue.replace(result("episode", 0), current_control_step=6, mode="async_aligned")
    assert outcome.fully_stale
    assert not outcome.accepted
    assert queue.queue_length == before
    assert queue.stale_chunks_discarded == 1


def test_naive_replaces_queue_with_complete_new_chunk() -> None:
    queue = ActionQueue()
    queue.reset_episode("episode")
    queue.replace(result("episode", 0, rows=3), current_control_step=0, mode="async_naive")
    first, _ = queue.next_action()
    assert np.all(first == 1)
    queue.replace(result("episode", 0, rows=2), current_control_step=10, mode="async_naive")
    replacement, _ = queue.next_action()
    assert np.all(replacement == 1)
    assert queue.queue_length == 1


def test_underrun_repeats_last_nonzero_command_and_reset_clears_it() -> None:
    queue = ActionQueue()
    queue.reset_episode("episode")
    queue.replace(result("episode", 0, rows=1), current_control_step=0, mode="async_naive")
    action, held = queue.next_action()
    repeated, held_repeat = queue.next_action()
    assert not held
    assert held_repeat
    np.testing.assert_array_equal(repeated, action)
    assert not np.allclose(repeated, 0.0)
    queue.reset_episode("next")
    with pytest.raises(QueueNotReady):
        queue.next_action()


def test_old_episode_chunk_cannot_enter_new_episode() -> None:
    queue = ActionQueue()
    queue.reset_episode("new")
    outcome = queue.replace(result("old", 0), current_control_step=0, mode="async_naive")
    assert not outcome.accepted
    assert outcome.reason == "old_episode"
    assert queue.queue_length == 0


def test_worker_has_one_inference_in_flight_and_discards_old_episode_result() -> None:
    release = threading.Event()
    active = 0
    max_active = 0
    lock = threading.Lock()

    def infer(request: InferenceRequest) -> InferencePayload:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        release.wait(timeout=2)
        with lock:
            active -= 1
        actions = np.ones((2, 7), dtype=np.float32)
        return InferencePayload(actions=actions, model_inference_latency_seconds=0.01)

    worker = LatestRequestWorker(infer)
    try:
        worker.reset_episode("old")
        worker.submit(InferenceRequest({}, "task", "old", 0, time.monotonic()))
        time.sleep(0.02)
        worker.reset_episode("new")
        worker.submit(InferenceRequest({}, "task", "new", 0, time.monotonic()))
        release.set()
        assert worker.wait_idle(timeout=2)
        delivered = worker.drain_results()
        assert max_active == 1
        assert len(delivered) == 1
        assert delivered[0].episode_id == "new"
    finally:
        worker.close()
