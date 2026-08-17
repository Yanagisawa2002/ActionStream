from __future__ import annotations

import sys

import pytest

from action_stream_isaac.dynamic_request_driver import (
    DynamicRequestDriverConfig,
    DynamicRequestScheduler,
    generation_from_task_state,
    is_bootstrap_request_accepted,
    is_executor_lifecycle_ready,
    success_from_dynamic_task_state,
    sync_resolution_for_runtime_event,
)
from action_stream_isaac.dynamic_task import (
    PHASE_SUCCESS,
    DynamicTaskMachine,
    DynamicTaskMeasurement,
    pack_dynamic_task_state,
    scenario_for_seed,
)


def _packed_initial_state() -> tuple[float, ...]:
    scenario = scenario_for_seed(0)
    measurement = DynamicTaskMeasurement(
        episode_step=0,
        end_effector_xyz=(0.3, 0.0, 0.5),
        object_xyz=scenario.object_xyz,
        object_wxyz=scenario.object_wxyz,
        object_linear_velocity_xyz=(0.0, 0.0, 0.0),
        object_angular_velocity_xyz=(0.0, 0.0, 0.0),
        gripper_aperture_m=0.08,
        physically_grasped=False,
    )
    evaluation = DynamicTaskMachine(scenario).update(measurement)
    return pack_dynamic_task_state(
        scenario=scenario,
        measurement=measurement,
        evaluation=evaluation,
    )


def test_scheduler_uses_cadence_and_forces_every_generation_advance() -> None:
    scheduler = DynamicRequestScheduler(10)
    assert scheduler.observe(observation_step=0, generation_id=1).request
    assert not scheduler.observe(observation_step=1, generation_id=1).request
    cadence = scheduler.observe(observation_step=10, generation_id=1)
    assert cadence.request and cadence.scheduled

    forced = scheduler.observe(observation_step=13, generation_id=2)
    assert forced.request
    assert forced.forced_generation_advance
    assert not forced.scheduled
    assert forced.previous_generation_id == 1
    duplicate = scheduler.observe(observation_step=13, generation_id=2)
    assert not duplicate.request
    assert not duplicate.forced_generation_advance


def test_generation_advance_is_marked_forced_even_on_cadence_boundary() -> None:
    scheduler = DynamicRequestScheduler(10)
    scheduler.observe(observation_step=0, generation_id=1)
    decision = scheduler.observe(observation_step=10, generation_id=2)
    assert decision.request and decision.scheduled
    assert decision.forced_generation_advance


def test_sync_hold_never_overlaps_requests_and_retries_after_cpp_expiry() -> None:
    scheduler = DynamicRequestScheduler(10, strategy="sync_hold")
    first = scheduler.observe(observation_step=0, generation_id=1)
    assert first.request
    scheduler.mark_request_published(request_id=1, generation_id=1)
    assert scheduler.sync_request_in_flight
    for step in (1, 10, 20):
        assert not scheduler.observe(observation_step=step, generation_id=1).request

    assert scheduler.resolve_request(request_id=1, retry=True)
    retry = scheduler.observe(observation_step=21, generation_id=1)
    assert retry.request and retry.retry_after_failure and not retry.scheduled
    scheduler.mark_request_published(request_id=2, generation_id=1)
    assert not scheduler.observe(observation_step=30, generation_id=1).request
    assert scheduler.resolve_request(request_id=2, retry=False)
    assert not scheduler.observe(observation_step=31, generation_id=1).request


def test_drop_does_not_fabricate_sync_cancellation_and_bootstrap_uses_acceptance() -> None:
    assert sync_resolution_for_runtime_event("response_dropped", "trace_drop") is None
    assert sync_resolution_for_runtime_event("chunk_rejected", "invalid_command") is None
    assert sync_resolution_for_runtime_event("chunk_rejected", "fully_expired") is True
    assert sync_resolution_for_runtime_event("queue_updated", "sync_valid_rebuild") is False
    assert is_bootstrap_request_accepted("request_registered", 1, 1)
    assert not is_bootstrap_request_accepted("queue_updated", 1, 1)
    assert is_executor_lifecycle_ready("episode_started", 1)
    assert is_executor_lifecycle_ready("episode_reset", 1)
    assert not is_executor_lifecycle_ready("episode_start", 1)
    assert not is_executor_lifecycle_ready("episode_started", 2)


def test_sync_generation_advance_supersedes_old_inflight_request() -> None:
    scheduler = DynamicRequestScheduler(10, strategy="sync_hold")
    scheduler.observe(observation_step=0, generation_id=1)
    scheduler.mark_request_published(request_id=1, generation_id=1)
    forced = scheduler.observe(observation_step=7, generation_id=2)
    assert forced.request and forced.forced_generation_advance
    scheduler.mark_request_published(request_id=2, generation_id=2)
    assert not scheduler.resolve_request(request_id=1, retry=True)
    assert scheduler.sync_request_in_flight
    assert not scheduler.observe(observation_step=10, generation_id=2).request
    assert scheduler.resolve_request(request_id=2, retry=False)


def test_scheduler_rejects_generation_or_observation_regression() -> None:
    scheduler = DynamicRequestScheduler(10)
    scheduler.observe(observation_step=0, generation_id=1)
    scheduler.observe(observation_step=10, generation_id=2)
    with pytest.raises(ValueError, match="generation regressed"):
        scheduler.observe(observation_step=11, generation_id=1)
    with pytest.raises(ValueError, match="step regressed"):
        scheduler.observe(observation_step=9, generation_id=2)


def test_task_state_generation_and_success_are_strict() -> None:
    state = _packed_initial_state()
    assert generation_from_task_state(state) == 1
    assert not success_from_dynamic_task_state(state)

    successful = list(state)
    successful[25] = float(PHASE_SUCCESS)
    successful[39] = 1.0
    successful[40] = 1.0
    assert success_from_dynamic_task_state(successful)

    malformed = list(state)
    malformed[39] = 0.5
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        success_from_dynamic_task_state(malformed)


def test_config_is_frozen_and_module_import_does_not_require_ros() -> None:
    DynamicRequestDriverConfig("episode", "aligned_async").validate()
    with pytest.raises(ValueError, match="episode_id"):
        DynamicRequestDriverConfig("", "aligned_async").validate()
    with pytest.raises(ValueError, match="strategy"):
        DynamicRequestDriverConfig("episode", "").validate()
    assert "rclpy" not in sys.modules
