from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import actionstream.benchmark as benchmark
from actionstream.benchmark import (
    _coalesce_missed_control_ticks,
    _measured_control_step_seconds,
    _run_async_episode,
    build_parser,
)
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


def test_measured_control_period_falls_back_for_zero_or_nonfinite_clock_deltas() -> None:
    period = 0.05
    assert _measured_control_step_seconds([1.0, 1.0, 1.0], period) == period
    assert _measured_control_step_seconds([1.0, float("nan")], period) == period
    assert _measured_control_step_seconds([1.0], period) == period
    assert _measured_control_step_seconds([1.0, 1.04, 1.08], period) == pytest.approx(0.04)


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


def test_safe_hold_discards_pending_commands_but_retains_last_action() -> None:
    queue = ActionQueue()
    queue.reset_episode("episode")
    queue.replace(result("episode", 0, rows=4), current_control_step=0, mode="async_naive")
    first, held = queue.next_action()
    assert not held
    assert queue.discard_pending_for_hold() == 3
    repeated, held_repeat = queue.next_action()
    assert held_repeat
    np.testing.assert_array_equal(repeated, first)


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


def test_delivery_delay_does_not_occupy_inference_worker() -> None:
    observed_steps: list[int] = []

    def infer(request: InferenceRequest) -> InferencePayload:
        observed_steps.append(request.observation_control_step)
        return InferencePayload(
            actions=np.ones((2, 7), dtype=np.float32),
            model_inference_latency_seconds=0.001,
        )

    worker = LatestRequestWorker(infer, delivery_delay_seconds=1.0)
    worker.reset_episode("episode")
    worker.submit(InferenceRequest({}, "task", "episode", 0, time.monotonic()))
    assert worker.wait_idle(timeout=1)
    worker.submit(InferenceRequest({}, "task", "episode", 10, time.monotonic()))
    assert worker.wait_idle(timeout=1)
    assert observed_steps == [0, 10]
    assert worker.drain_results() == []
    worker.close()
    assert len(worker.drain_all_results()) == 2


def test_control_clock_skips_only_whole_missed_ticks() -> None:
    scheduled, missed = _coalesce_missed_control_ticks(10.0, 10.12, 0.05)
    assert missed == 2
    assert scheduled == pytest.approx(10.1)

    next_scheduled, next_missed = _coalesce_missed_control_ticks(
        scheduled + 0.05,
        10.16,
        0.05,
    )
    assert next_missed == 0
    assert next_scheduled == pytest.approx(10.15)


class FakeRealtimeBackend:
    episode_length = 5
    model_id = "fake-model"
    model_revision = "fake-revision"
    suite = "fake-suite"

    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []
        self.step_timestamps: list[float] = []

    def reset_episode(
        self,
        *,
        task_id: int,
        seed: int,
        initial_state_index: int,
    ) -> tuple[dict[str, int], dict[str, object], str]:
        return {"step": 0}, {}, "fake instruction"

    def controller_frequency_hz(self, task_id: int) -> float:
        return 200.0

    def step(self, task_id: int, action: np.ndarray) -> SimpleNamespace:
        self.actions.append(np.asarray(action).copy())
        self.step_timestamps.append(time.monotonic())
        terminal = len(self.actions) >= self.episode_length
        return SimpleNamespace(
            observation={"step": len(self.actions)},
            success=False,
            terminated=terminal,
            truncated=False,
        )

    @property
    def peak_cuda_memory_mib(self) -> float:
        return 0.0


def test_sync_hold_waits_for_first_chunk_then_advances_with_exact_hold(
    tmp_path,
    monkeypatch,
) -> None:
    backend = FakeRealtimeBackend()
    first_chunk = np.asarray(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            [4.0, 5.0, 6.0, 0.1, 0.2, 0.3, -1.0],
        ],
        dtype=np.float32,
    )
    next_chunk = np.repeat(
        np.asarray([[9.0, 8.0, 7.0, 0.3, 0.2, 0.1, 1.0]], dtype=np.float32),
        2,
        axis=0,
    )
    request_steps: list[int] = []
    inference_finished: list[float] = []
    second_inference_started: list[float] = []

    def infer(request: InferenceRequest) -> InferencePayload:
        request_steps.append(request.observation_control_step)
        if len(request_steps) == 1:
            actions = first_chunk
        else:
            second_inference_started.append(time.monotonic())
            time.sleep(0.05)
            actions = next_chunk
        inference_finished.append(time.monotonic())
        return InferencePayload(actions=actions, model_inference_latency_seconds=0.001)

    worker = LatestRequestWorker(infer)
    monkeypatch.setattr(benchmark.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(benchmark, "_lerobot_version", lambda: "test")
    try:
        record = _run_async_episode(
            backend,
            run_id="test",
            git_commit="test",
            task_id=0,
            episode_index=0,
            initial_state_index=0,
            seed=142,
            injected_delay_ms=0,
            realtime=True,
            replan_interval_steps=10,
            mode="sync_hold",
            inference_worker=worker,
            output=tmp_path / "episodes.jsonl",
        )
    finally:
        worker.close()

    assert request_steps[:2] == [0, 2]
    assert backend.step_timestamps[0] >= inference_finished[0]
    assert any(
        second_inference_started[0] <= timestamp <= inference_finished[1]
        for timestamp in backend.step_timestamps
    )
    assert record["queue_underrun_hold_steps"] == 3
    for held_action in backend.actions[2:]:
        np.testing.assert_array_equal(held_action, first_chunk[-1])
    assert record["replan_interval_steps"] == len(first_chunk)
    assert record["nominal_queue_headroom_steps"] == 0
    assert record["chunks_accepted"] == 1
    assert record["chunks_rejected_after_episode"] == 1
    trace_path = tmp_path / "traces" / "sync_hold_delay0_task0_episode0.npz"
    with np.load(trace_path, allow_pickle=False) as trace:
        np.testing.assert_array_equal(
            trace["queue_hold_mask"],
            np.asarray([False, False, True, True, True]),
        )
        np.testing.assert_array_equal(
            trace["queue_depth_before_action"],
            np.asarray([2, 1, 0, 0, 0]),
        )
        events = json.loads(str(trace["inference_events_json"].item()))
    assert events[0]["merge"]["accepted"] is True
    assert events[0]["merge"]["replacement_action_index"] == 0
    assert events[0]["chunk_accepted"] is True
    assert events[0]["merge"]["stale_fraction"] == 0.0
    assert events[0]["queue_headroom_at_request_steps"] == 0
    assert "effective_delivery_age_steps" in events[0]


def test_sync_hold_episode_reset_rejects_old_chunks_and_clears_hold() -> None:
    queue = ActionQueue()
    queue.reset_episode("old")
    queue.replace(result("old", 0, rows=1), current_control_step=0, mode="sync_hold")
    previous, held = queue.next_action()
    repeated, held_repeat = queue.next_action()
    assert not held
    assert held_repeat
    np.testing.assert_array_equal(repeated, previous)

    queue.reset_episode("new")
    assert queue.hold_steps == 0
    assert queue.accepted_chunks == 0
    with pytest.raises(QueueNotReady):
        queue.next_action()
    stale_outcome = queue.replace(
        result("old", 0, rows=1),
        current_control_step=0,
        mode="sync_hold",
    )
    assert not stale_outcome.accepted
    assert stale_outcome.reason == "old_episode"
    with pytest.raises(QueueNotReady):
        queue.next_action()


def test_replacement_event_metrics_use_last_executed_absolute_command() -> None:
    queue = ActionQueue()
    queue.reset_episode("episode")
    first_actions = np.asarray(
        [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    second_actions = np.asarray(
        [[4.0, 6.0, 3.0, 0.0, 0.0, np.pi / 2.0, 1.0]],
        dtype=np.float32,
    )

    queue.replace(
        InferenceResult(
            actions=first_actions,
            episode_id="episode",
            observation_control_step=0,
            request_timestamp=1.0,
            start_timestamp=2.0,
            end_timestamp=3.0,
            delivery_timestamp=4.0,
            model_inference_latency_seconds=1.0,
        ),
        current_control_step=0,
        mode="async_naive",
    )
    queue.next_action()
    outcome = queue.replace(
        InferenceResult(
            actions=second_actions,
            episode_id="episode",
            observation_control_step=1,
            request_timestamp=5.0,
            start_timestamp=6.0,
            end_timestamp=7.0,
            delivery_timestamp=8.0,
            model_inference_latency_seconds=1.0,
        ),
        current_control_step=1,
        mode="async_naive",
    )

    assert outcome.replacement_position_l2 == pytest.approx(5.0)
    assert outcome.replacement_rotation_geodesic_radians == pytest.approx(np.pi / 2.0)
    assert outcome.replacement_gripper_switch is True


def test_parser_accepts_derived_arbitrary_nonnegative_delay(tmp_path) -> None:
    parsed = build_parser().parse_args(
        ["--mode", "async_aligned", "--injected-delay-ms", "437", "--output", "out.jsonl"]
    )
    assert parsed.injected_delay_ms == 437

    negative = build_parser().parse_args(
        [
            "--mode",
            "async_aligned",
            "--injected-delay-ms",
            "-1",
            "--output",
            str(tmp_path / "negative.jsonl"),
        ]
    )
    with pytest.raises(ValueError, match="must be non-negative"):
        benchmark.run(negative)
