from __future__ import annotations

import inspect
import json
import threading
import time
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from actionstream.m5_runtime import (
    M5AlignedActionQueue,
    M5InferenceRequest,
    M5LatestRequestWorker,
    filter_result_generation,
    summarize_action_records,
)
from actionstream.runtime import (
    ActionQueue,
    InferencePayload,
    InferenceRequest,
    InferenceResult,
    LatestRequestWorker,
)


def request(
    *,
    observation_step: int = 0,
    generation: int = 0,
    world_epoch: int = 0,
    episode_id: str = "episode",
    request_id: str | None = None,
) -> M5InferenceRequest:
    return M5InferenceRequest(
        observation={"step": observation_step},
        task_instruction="put the bowl in the basket",
        episode_id=episode_id,
        observation_control_step=observation_step,
        request_timestamp=float(observation_step + 1),
        queue_depth_at_request_steps=2,
        queue_headroom_at_request_steps=1,
        request_id=request_id or f"request-{generation}-{observation_step}",
        request_generation_id=generation,
        condition="aligned_oracle_pose_gate_shift",
        task_id=0,
        seed=142,
        world_epoch_at_observation=world_epoch,
        entity_pose_at_observation={
            "entity_name": "basket_1",
            "position": [0.1 + world_epoch * 0.03, 0.2, 0.3],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
        },
    )


def result(
    source: M5InferenceRequest,
    *,
    rows: int = 6,
    offset: float = 0.0,
) -> InferenceResult:
    values = np.arange(1, rows + 1, dtype=np.float32) + offset
    actions = np.repeat(values[:, None], 7, axis=1)
    request_time = source.request_timestamp
    return InferenceResult(
        actions=actions,
        episode_id=source.episode_id,
        observation_control_step=source.observation_control_step,
        request_timestamp=request_time,
        start_timestamp=request_time + 0.1,
        end_timestamp=request_time + 0.2,
        delivery_timestamp=request_time + 0.3,
        model_inference_latency_seconds=0.1,
    )


def merge(
    queue: M5AlignedActionQueue,
    source: M5InferenceRequest,
    *,
    current_step: int,
    rows: int = 6,
    offset: float = 0.0,
    current_world_epoch: int | None = None,
):
    return queue.replace(
        result(source, rows=rows, offset=offset),
        request=source,
        current_control_step=current_step,
        result_arrival_step=current_step,
        result_arrival_timestamp=float(current_step + 10),
        current_world_epoch=(
            source.world_epoch_at_observation
            if current_world_epoch is None
            else current_world_epoch
        ),
    )


def m4_result(source: M5InferenceRequest, *, rows: int = 6) -> InferenceResult:
    return result(source, rows=rows)


def test_m5_request_is_a_frozen_m4_request_with_pose_snapshot() -> None:
    pose = {
        "entity_name": "basket_1",
        "position": [0.1, 0.2, 0.3],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
    }
    source = request()
    assert isinstance(source, InferenceRequest)
    with pytest.raises(FrozenInstanceError):
        source.request_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        source.entity_pose_at_observation["entity_name"] = "changed"  # type: ignore[index]

    serialized = source.provenance_dict()
    assert serialized["entity_pose_at_observation"] == pose
    json.dumps(serialized)


def test_existing_latest_request_worker_accepts_m5_request_subclass() -> None:
    observed: list[M5InferenceRequest] = []

    def infer(source: InferenceRequest) -> InferencePayload:
        assert isinstance(source, M5InferenceRequest)
        observed.append(source)
        return InferencePayload(
            actions=np.ones((2, 7), dtype=np.float32),
            model_inference_latency_seconds=0.001,
        )

    worker = LatestRequestWorker(infer)
    try:
        source = request()
        worker.reset_episode(source.episode_id)
        worker.submit(source)
        assert worker.wait_idle(timeout=2.0)
        assert worker.wait_for_result(timeout=2.0)
        delivered = worker.drain_results()
    finally:
        worker.close()
    assert observed == [source]
    assert len(delivered) == 1
    assert delivered[0].delivery_timestamp <= time.monotonic()


def test_m5_worker_reports_exact_mailbox_replacement_and_cancellation() -> None:
    started = threading.Event()
    release = threading.Event()

    def infer(source: InferenceRequest) -> InferencePayload:
        started.set()
        assert release.wait(timeout=2.0)
        return InferencePayload(
            actions=np.ones((2, 7), dtype=np.float32),
            model_inference_latency_seconds=0.001,
        )

    worker = M5LatestRequestWorker(infer)
    try:
        active = request(request_id="active")
        pending = request(request_id="pending")
        replacement = request(request_id="replacement")
        worker.reset_episode(active.episode_id)
        assert worker.submit(active) is None
        assert started.wait(timeout=2.0)
        assert worker.submit(pending) is None
        assert worker.submit(replacement) is pending
        assert worker.cancel_pending_request() is replacement
        release.set()
        assert worker.wait_idle(timeout=2.0)
    finally:
        release.set()
        worker.close()
    assert worker.pending_requests_replaced == 1
    assert worker.pending_requests_cancelled == 1


def test_no_shift_queue_is_command_equivalent_to_existing_aligned_queue() -> None:
    existing = ActionQueue()
    m5 = M5AlignedActionQueue()
    existing.reset_episode("episode")
    m5.reset_episode("episode")
    source = request(observation_step=2)
    incoming = m4_result(source)

    existing_outcome = existing.replace(
        incoming,
        current_control_step=5,
        mode="async_aligned",
    )
    m5_outcome = merge(m5, source, current_step=5)
    assert (
        m5_outcome.accepted,
        m5_outcome.reason,
        m5_outcome.age_steps,
        m5_outcome.dropped_prefix_steps,
        m5_outcome.fully_stale,
        m5_outcome.queue_length,
        m5_outcome.stale_fraction,
    ) == (
        existing_outcome.accepted,
        existing_outcome.reason,
        existing_outcome.age_steps,
        existing_outcome.dropped_prefix_steps,
        existing_outcome.fully_stale,
        existing_outcome.queue_length,
        existing_outcome.stale_fraction,
    )

    for step in range(5, 8):
        existing_action, existing_held = existing.next_action()
        m5_record = m5.next_action(
            current_control_step=step,
            current_world_epoch=0,
            execution_timestamp=float(step + 20),
        )
        m5_action, m5_held = m5_record
        np.testing.assert_array_equal(m5_action, existing_action)
        assert m5_held == existing_held
        assert not m5_record.stale

    existing_hold, existing_held = existing.next_action()
    m5_hold = m5.next_action(
        current_control_step=8,
        current_world_epoch=0,
        execution_timestamp=28.0,
    )
    np.testing.assert_array_equal(m5_hold.action, existing_hold)
    assert m5_hold.held == existing_held
    assert m5_hold.source_kind == "queue_hold"


def test_fully_stale_result_keeps_queue_and_has_complete_discards() -> None:
    queue = M5AlignedActionQueue()
    queue.reset_episode("episode")
    active = request(observation_step=0, request_id="active")
    merge(queue, active, current_step=0, rows=3)
    before = queue.queued_actions

    stale = request(observation_step=1, request_id="fully-stale")
    outcome = merge(queue, stale, current_step=5, rows=4)
    assert not outcome.accepted
    assert outcome.fully_stale
    assert outcome.dropped_prefix_steps == 4
    assert queue.queued_actions == before
    assert len(outcome.discard_records) == 4
    assert {row.discard_reason for row in outcome.discard_records} == {"fully_stale"}
    assert all(not row.was_queued for row in outcome.discard_records)
    assert [row.original_chunk_action_index for row in outcome.discard_records] == [
        0,
        1,
        2,
        3,
    ]


def test_replacement_records_old_queue_before_inserting_larger_chunk() -> None:
    queue = M5AlignedActionQueue()
    queue.reset_episode("episode")
    first = request(observation_step=0, request_id="first")
    merge(queue, first, current_step=0, rows=3)
    queue.next_action(
        current_control_step=0,
        current_world_epoch=0,
        execution_timestamp=20.0,
    )
    assert queue.queue_length == 2

    second = request(observation_step=1, request_id="second")
    outcome = merge(queue, second, current_step=1, rows=5)
    assert outcome.accepted
    assert queue.queue_length == 5
    replaced = [
        record
        for record in outcome.discard_records
        if record.discard_reason == "queue_replaced"
    ]
    assert len(replaced) == 2
    assert all(record.queue_depth_before_invalidation == 2 for record in replaced)
    assert all(record.queue_depth_after_invalidation == 0 for record in replaced)


def test_scene_invalidation_clears_old_queue_and_preserves_safe_command() -> None:
    queue = M5AlignedActionQueue()
    queue.reset_episode("episode")
    source = request()
    merge(queue, source, current_step=0, rows=4)
    executed = queue.next_action(
        current_control_step=0,
        current_world_epoch=0,
        execution_timestamp=20.0,
    )
    assert queue.queue_length == 3

    invalidation = queue.invalidate_scene(
        current_control_step=1,
        current_world_epoch=1,
        invalidation_timestamp=21.0,
    )
    assert invalidation.queue_depth_before == 3
    assert invalidation.queue_depth_after == 0
    assert len(invalidation) == 3
    assert invalidation.last_safe_absolute_command_preserved
    assert queue.queue_length == 0
    assert queue.current_request_generation_id == 1
    assert all(record.discard_reason == "scene_invalidation" for record in invalidation)
    assert all(record.gate_triggered for record in invalidation)
    assert all(record.stale for record in invalidation)
    assert all(record.queue_depth_before_invalidation == 3 for record in invalidation)
    assert all(record.queue_depth_after_invalidation == 0 for record in invalidation)

    hold = queue.safe_hold(
        current_control_step=1,
        current_world_epoch=1,
        execution_timestamp=21.1,
    )
    np.testing.assert_array_equal(hold.action, executed.action)


def test_late_old_result_is_rejected_and_fresh_result_is_accepted() -> None:
    queue = M5AlignedActionQueue()
    queue.reset_episode("episode")
    old = request(observation_step=0, generation=0, world_epoch=0)
    merge(queue, old, current_step=0, rows=2)
    queue.next_action(
        current_control_step=0,
        current_world_epoch=0,
        execution_timestamp=20.0,
    )
    queue.invalidate_scene(
        current_control_step=1,
        current_world_epoch=1,
        invalidation_timestamp=21.0,
    )

    late_outcome = merge(
        queue,
        old,
        current_step=1,
        rows=3,
        current_world_epoch=1,
    )
    assert not late_outcome.accepted
    assert late_outcome.reason == "late_generation"
    assert queue.queue_length == 0
    assert len(late_outcome.discard_records) == 3
    assert queue.stale_policy_results_discarded == 1

    fresh = request(
        observation_step=1,
        generation=1,
        world_epoch=1,
        request_id="fresh",
    )
    fresh_outcome = merge(
        queue,
        fresh,
        current_step=1,
        rows=3,
        current_world_epoch=1,
    )
    assert fresh_outcome.accepted
    assert queue.queue_length == 3
    assert queue.fresh_replan_count == 1


def test_no_old_scene_policy_action_executes_after_invalidation() -> None:
    queue = M5AlignedActionQueue()
    queue.reset_episode("episode")
    old = request(world_epoch=0, generation=0)
    merge(queue, old, current_step=0, rows=3)
    last_safe = queue.next_action(
        current_control_step=0,
        current_world_epoch=0,
        execution_timestamp=20.0,
    )
    queue.invalidate_scene(
        current_control_step=1,
        current_world_epoch=1,
        invalidation_timestamp=21.0,
    )
    gate_hold = queue.next_action(
        current_control_step=1,
        current_world_epoch=1,
        execution_timestamp=21.1,
    )
    np.testing.assert_array_equal(gate_hold.action, last_safe.action)
    assert gate_hold.source_kind == "gate_hold"
    assert gate_hold.world_epoch_at_observation == 1
    assert gate_hold.command_origin.world_epoch_at_observation == 0
    assert not gate_hold.stale

    fresh = request(observation_step=1, generation=1, world_epoch=1)
    merge(
        queue,
        fresh,
        current_step=1,
        rows=2,
        current_world_epoch=1,
    )
    fresh_action = queue.next_action(
        current_control_step=2,
        current_world_epoch=1,
        execution_timestamp=22.0,
    )
    assert fresh_action.source_kind == "policy"
    assert fresh_action.request_generation_id == 1
    assert fresh_action.world_epoch_at_observation == 1
    assert not fresh_action.stale
    post_gate_policy = [
        row
        for row in queue.action_records
        if getattr(row, "queue_execution_step", None) is not None
        and row.queue_execution_step >= 1
        and getattr(row, "source_kind", None) == "policy"
    ]
    assert all(row.world_epoch_at_observation == 1 for row in post_gate_policy)


def test_summarize_action_records_has_exact_synthetic_stale_metrics() -> None:
    rows = [
        {
            "discarded": False,
            "source_kind": "policy",
            "queue_execution_step": 0,
            "observation_control_step": 0,
            "world_epoch_at_observation": 0,
            "current_world_epoch_at_execution": 0,
            "queue_depth_before_action": 3,
            "queue_depth_after_action": 2,
        },
        {
            "discarded": False,
            "source_kind": "policy",
            "queue_execution_step": 1,
            "observation_control_step": 0,
            "world_epoch_at_observation": 0,
            "current_world_epoch_at_execution": 1,
            "queue_depth_before_action": 2,
            "queue_depth_after_action": 1,
        },
        {
            "discarded": False,
            "source_kind": "policy",
            "queue_execution_step": 3,
            "observation_control_step": 0,
            "world_epoch_at_observation": 0,
            "current_world_epoch_at_execution": 2,
            "queue_depth_before_action": 1,
            "queue_depth_after_action": 0,
        },
        {
            "discarded": False,
            "source_kind": "gate_hold",
            "queue_execution_step": 4,
            "observation_control_step": 4,
            "world_epoch_at_observation": 2,
            "current_world_epoch_at_execution": 2,
            "queue_depth_before_action": 0,
            "queue_depth_after_action": 0,
        },
        {
            "discarded": True,
            "discard_reason": "late_generation",
            "episode_id": "episode",
            "request_id": "old",
            "request_generation_id": 0,
            "world_epoch_at_observation": 0,
            "current_world_epoch_at_discard": 2,
            "queue_depth_before_invalidation": 0,
            "queue_depth_after_invalidation": 0,
        },
        {
            "discarded": True,
            "discard_reason": "scene_invalidation",
            "episode_id": "episode",
            "request_id": "queued-old-scene",
            "request_generation_id": 0,
            "world_epoch_at_observation": 0,
            "current_world_epoch_at_discard": 2,
            "was_queued": True,
            "queue_insertion_step": 0,
            "queue_depth_before_invalidation": 1,
            "queue_depth_after_invalidation": 0,
        },
    ]
    summary = summarize_action_records(rows, control_frequency_hz=20.0)
    assert summary["stale_action_steps"] == 2
    assert summary["stale_action_duration_seconds"] == pytest.approx(0.1)
    assert summary["first_stale_action_step"] == 1
    assert summary["last_stale_action_step"] == 3
    assert summary["hold_steps_introduced_by_gate"] == 1
    assert summary["stale_policy_results_discarded"] == 1
    assert summary["stale_queued_actions_discarded"] == 1
    assert summary["maximum_queue_depth"] == 3
    assert summary["action_source_observation_age_steps_mean"] == pytest.approx(1.0)


def test_safe_hold_preserves_exact_command_and_labels_hold_source() -> None:
    queue = M5AlignedActionQueue()
    queue.reset_episode("episode")
    source = request()
    merge(queue, source, current_step=0, rows=1)
    policy = queue.next_action(
        current_control_step=0,
        current_world_epoch=0,
        execution_timestamp=20.0,
    )
    queue_hold = queue.next_action(
        current_control_step=1,
        current_world_epoch=0,
        execution_timestamp=21.0,
    )
    np.testing.assert_array_equal(queue_hold.action, policy.action)
    assert queue_hold.source_kind == "queue_hold"
    assert queue_hold.world_epoch_at_observation == 0

    queue.invalidate_scene(
        current_control_step=2,
        current_world_epoch=1,
        invalidation_timestamp=22.0,
    )
    gate_hold = queue.safe_hold(
        current_control_step=2,
        current_world_epoch=1,
        execution_timestamp=22.1,
        current_entity_pose={
            "entity_name": "basket_1",
            "position": [0.13, 0.2, 0.3],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
        },
    )
    np.testing.assert_array_equal(gate_hold.action, policy.action)
    assert gate_hold.source_kind == "gate_hold"
    assert gate_hold.world_epoch_at_observation == 1
    assert gate_hold.command_origin.request_id == policy.request_id
    assert gate_hold.command_origin.world_epoch_at_observation == 0
    assert not gate_hold.stale


def test_records_have_complete_json_serializable_provenance() -> None:
    queue = M5AlignedActionQueue()
    queue.reset_episode("episode")
    source = request(observation_step=2, request_id="complete")
    outcome = merge(queue, source, current_step=4, rows=5)
    assert outcome.accepted
    assert len(outcome.discard_records) == 2
    assert [
        action.original_chunk_action_index for action in outcome.inserted_actions
    ] == [
        2,
        3,
        4,
    ]
    executed = queue.next_action(
        current_control_step=4,
        current_world_epoch=0,
        execution_timestamp=24.0,
    )
    queue.invalidate_scene(
        current_control_step=5,
        current_world_epoch=1,
        invalidation_timestamp=25.0,
    )

    required = {
        "episode_id",
        "condition",
        "task_id",
        "seed",
        "request_id",
        "request_generation_id",
        "observation_control_step",
        "observation_timestamp",
        "world_epoch_at_observation",
        "entity_pose_at_observation",
        "policy_result_arrival_step",
        "policy_result_arrival_timestamp",
        "original_chunk_action_index",
        "queue_insertion_step",
        "queue_execution_step",
        "current_world_epoch_at_execution",
        "stale",
        "discarded",
        "discard_reason",
        "gate_triggered",
        "queue_depth_before_invalidation",
        "queue_depth_after_invalidation",
    }
    serialized = [record.to_dict() for record in queue.action_records]
    assert serialized
    assert all(required <= set(record) for record in serialized)
    json.dumps(serialized)
    assert executed.original_chunk_action_index == 2
    assert executed.queue_insertion_step == 4
    assert executed.policy_result_arrival_step == 4
    assert executed.policy_result_arrival_timestamp == 14.0


def test_episode_finalization_records_every_remaining_queued_action() -> None:
    queue = M5AlignedActionQueue()
    queue.reset_episode("episode")
    source = request(observation_step=0, request_id="finalize")
    merge(queue, source, current_step=0, rows=3)

    records = queue.discard_remaining_at_episode_end(
        current_control_step=1,
        current_world_epoch=0,
        discard_timestamp=21.0,
    )

    assert len(records) == 3
    assert queue.queue_length == 0
    assert queue.queue_invalidation_count == 0
    assert queue.current_request_generation_id == 0
    assert all(record.discard_reason == "episode_ended" for record in records)
    assert all(record.was_queued for record in records)
    assert all(not record.stale for record in records)


def test_generation_filter_uses_only_generation_and_accepts_fresh() -> None:
    parameters = set(inspect.signature(filter_result_generation).parameters)
    assert parameters == {
        "request_generation_id",
        "current_request_generation_id",
    }
    late = filter_result_generation(3, 4)
    fresh = filter_result_generation(4, 4)
    future = filter_result_generation(5, 4)
    assert not late.accepted and late.reason == "late_generation"
    assert fresh.accepted and fresh.reason == "current_generation"
    assert not future.accepted and future.reason == "future_generation"
