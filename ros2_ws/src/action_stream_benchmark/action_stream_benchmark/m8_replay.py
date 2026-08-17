"""Independent replay and metric reconstruction for M8 native-Isaac logs."""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .m8_archive import (
    ARCHIVE_EPISODE_ARTIFACT_FIELDS,
    DIRECT_EPISODE_ARTIFACT_FIELDS,
    iter_archive_members,
    read_archive_member,
    read_archive_members,
    validate_archive_matrix_binding,
)
from .m8_faults import (
    M8FaultProfile,
    M8FaultTrace,
    declared_fault_profile,
    fault_profile_binding_mismatches,
    fault_trace_from_mapping,
    load_fault_trace,
)
from .m8_protocol import (
    CONTROL_FREQUENCY_HZ,
    M8_MILESTONE,
    M8_SCHEMA_VERSION,
    NATIVE_ISAAC_EVIDENCE_CLASS,
    PAIRED_RESET_FAIRNESS_CONTRACT,
    PROFILE_STRATEGIES,
    load_protocol,
    load_seed_file,
    paired_reset_canonical_payload,
    sha256_file,
    validate_freeze_manifest,
)
from .m8_scenario import (
    command_targets_obsolete_destination,
    euclidean_distance,
    load_scenario,
    motion_targets_obsolete_destination,
    scenario_from_mapping,
)
from .schema import canonical_sha256, read_json, read_jsonl, write_json_atomic


CONTROL_PERIOD_SECONDS = 1.0 / CONTROL_FREQUENCY_HZ
FAIRNESS_FIELDS = (
    "fault_trace_sha256",
    "scenario_sha256",
    "reset_state_sha256",
    "paired_reset_canonical_sha256",
    "switch_step",
    "object_position_xyz",
    "original_destination_xyz",
    "final_destination_xyz",
    "controller_sha256",
    "protocol_sha256",
    "task_contract_sha256",
    "safety_limits_sha256",
)
HOLDOUT_FAIRNESS_FIELDS = (*FAIRNESS_FIELDS, "freeze_sha256")
PAIRED_EXACT_EQUALITY_FIELDS = tuple(
    field
    for field in FAIRNESS_FIELDS
    if field not in {"reset_state_sha256", "paired_reset_canonical_sha256"}
)
REQUIRED_METRICS = (
    "task_success",
    "correct_destination_placement_success",
    "obsolete_destination_placement_count",
    "grasp_success",
    "stable_placement_success",
    "collision",
    "timeout",
    "completion_reason",
    "completion_steps",
    "wall_clock_seconds",
    "simulation_seconds",
    "hold_control_steps",
    "hold_control_seconds",
    "response_wait_wall_seconds",
    "recovery_observed",
    "recovery_latency_steps",
    "recovery_latency_seconds",
    "penalized_recovery_latency_steps",
    "time_to_final_destination_progress_steps",
    "time_to_final_destination_progress_seconds",
    "obsolete_destination_motion_steps",
    "obsolete_destination_motion_seconds",
    "pre_switch_observation_command_steps",
    "obsolete_destination_command_steps",
    "expired_actions_executed",
    "expired_actions_removed",
    "stale_generation_responses_rejected",
    "duplicate_responses",
    "duplicate_target_actions_removed",
    "out_of_order_completions",
    "inference_requests",
    "returned_responses",
    "dropped_responses",
    "queue_rebuild_count",
    "deadline_misses",
    "deadline_miss_rate",
    "executed_actions",
    "mean_inference_latency_ms",
    "p50_inference_latency_ms",
    "p95_inference_latency_ms",
    "mean_action_age_steps",
    "p50_action_age_steps",
    "p95_action_age_steps",
    "command_motion_audit_count",
    "maximum_requested_target_delta_m",
    "maximum_applied_target_delta_m",
    "maximum_measured_end_effector_delta_m",
    "requested_target_discontinuity_count",
    "applied_target_discontinuity_count",
    "hold_requested_applied_target_mismatch_count",
)


def _development_strategy_filter(
    manifest: Mapping[str, Any], *, split: str
) -> tuple[str, ...] | None:
    """Return a rigorously scoped non-headline development method filter."""

    if "development_strategy_filter" not in manifest:
        return None
    if split != "development":
        raise ValueError("development_strategy_filter is forbidden outside development")
    raw = manifest.get("development_strategy_filter")
    if not isinstance(raw, list) or not raw:
        raise ValueError("development_strategy_filter must be a non-empty JSON list")
    selected = tuple(str(value) for value in raw)
    known = {
        strategy
        for strategies in PROFILE_STRATEGIES.values()
        for strategy in strategies
    }
    if (
        any(not isinstance(value, str) or not value for value in raw)
        or len(set(selected)) != len(selected)
        or any(strategy not in known for strategy in selected)
    ):
        raise ValueError(
            "development_strategy_filter is malformed or contains an unknown method"
        )
    if manifest.get("headline_eligible") is not False:
        raise ValueError(
            "filtered development replay must remain headline_eligible=false"
        )
    return selected


def _expected_profile_strategies(
    profile_id: str,
    *,
    split: str,
    development_filter: tuple[str, ...] | None,
) -> set[str]:
    """Keep baseline/holdout complete while allowing a declared dev-only subset."""

    expected = set(PROFILE_STRATEGIES.get(profile_id, ()))
    if split == "development" and development_filter is not None:
        expected.intersection_update(development_filter)
        if not expected:
            raise ValueError(
                f"development_strategy_filter leaves profile {profile_id!r} with no method"
            )
    return expected


def _percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _finite_xyz(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    converted = tuple(float(item) for item in value)
    return converted if all(math.isfinite(item) for item in converted) else None


def decode_dynamic_task_state(values: Sequence[float]) -> dict[str, Any]:
    """Decode the frozen 44-value M8 task layout without importing Isaac."""

    data = tuple(float(value) for value in values)
    if len(data) != 44 or not all(math.isfinite(value) for value in data):
        raise ValueError("M8 dynamic task_state must contain exactly 44 finite values")

    def integer(name: str, index: int, *, minimum: int = 0) -> int:
        converted = int(data[index])
        if float(converted) != data[index] or converted < minimum:
            raise ValueError(
                f"M8 task_state {name} must be an exact integer >= {minimum}"
            )
        return converted

    def binary(name: str, index: int) -> bool:
        if data[index] not in {0.0, 1.0}:
            raise ValueError(f"M8 task_state {name} must be exactly 0 or 1")
        return data[index] == 1.0

    active_destination_id = integer("active_destination_id", 22)
    if active_destination_id not in {0, 1}:
        raise ValueError("M8 task_state active_destination_id must be 0 or 1")
    switched = binary("disturbance_switched", 23)
    generation = integer("generation_id", 24)
    phase = integer("phase_code", 25)
    if generation not in {1, 2} or phase not in set(range(11)):
        raise ValueError(
            "M8 task_state generation or phase is outside the frozen contract"
        )
    if active_destination_id != int(switched) or generation != (2 if switched else 1):
        raise ValueError(
            "M8 task_state destination, disturbance, and generation disagree"
        )
    original = data[13:16]
    final = data[16:19]
    active = data[19:22]
    expected_active = final if switched else original
    if any(
        abs(left - right) > 1e-12
        for left, right in zip(active, expected_active, strict=True)
    ):
        raise ValueError(
            "M8 task_state active destination does not match disturbance state"
        )
    result = {
        "object_position_xyz": list(data[0:3]),
        "object_orientation_wxyz": list(data[3:7]),
        "object_linear_velocity_xyz": list(data[7:10]),
        "object_angular_velocity_xyz": list(data[10:13]),
        "original_destination_xyz": list(data[13:16]),
        "final_destination_xyz": list(data[16:19]),
        "active_destination_xyz": list(data[19:22]),
        "active_destination_id": active_destination_id,
        "disturbance_switched": switched,
        "generation_id": generation,
        "phase_code": phase,
        "phase_step": integer("phase_step", 26),
        "grasped": binary("grasped", 27),
        "gripper_aperture_m": data[28],
        "initial_object_z": data[29],
        "lift_height_m": data[30],
        "final_destination_xy_error_m": data[31],
        "obsolete_destination_xy_error_m": data[32],
        "success_streak_steps": integer("success_streak_steps", 33),
        "correct_destination_placement": binary("correct_destination_placement", 34),
        "obsolete_destination_placement": binary("obsolete_destination_placement", 35),
        "collision": binary("collision", 36),
        "joint_or_workspace_limit": binary("joint_or_workspace_limit", 37),
        "grasp_ever": binary("grasp_ever", 38),
        "success": binary("success", 39),
        "terminated": binary("terminated", 40),
        "episode_step": integer("episode_step", 41),
        "switch_step": integer("switch_step", 42, minimum=1),
        "max_disallowed_contact_force_n": data[43],
    }
    if result["success"] and (not result["terminated"] or phase != 9):
        raise ValueError("successful M8 task_state must be terminated in success phase")
    if result["terminated"] and phase not in {9, 10}:
        raise ValueError(
            "terminated M8 task_state must use success or terminated phase"
        )
    if result["grasped"] and not result["grasp_ever"]:
        raise ValueError("current grasp requires grasp_ever")
    return result


def _observation(row: Mapping[str, Any]) -> dict[str, Any]:
    task_state = row.get("task_state")
    if not isinstance(task_state, (list, tuple)) or len(task_state) != 44:
        raise ValueError(
            "native M8 observation is missing the authoritative 44-value task_state"
        )
    # Explicit decoded/derived convenience fields are deliberately ignored.
    # Only the frozen raw vector may drive metric reconstruction.
    decoded = decode_dynamic_task_state(task_state)
    decoded["observation_step"] = int(
        row.get("observation_step", row.get("sim_step", 0))
    )
    return decoded


def _event_detail(row: Mapping[str, Any]) -> dict[str, Any]:
    detail = row.get("detail")
    if isinstance(detail, Mapping):
        return dict(detail)
    if isinstance(detail, str) and detail.strip().startswith("{"):
        try:
            value = json.loads(detail)
        except (TypeError, ValueError):
            return {}
        return dict(value) if isinstance(value, Mapping) else {}
    return {}


def _merged(row: Mapping[str, Any]) -> dict[str, Any]:
    return {**_event_detail(row), **dict(row)}


def _first(rows: Sequence[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("event_type") == event_type), None)


def recompute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("event log is empty")
    starts = [row for row in rows if row.get("event_type") == "episode_start"]
    ends = [row for row in rows if row.get("event_type") == "episode_end"]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(
            "M8 event log requires exactly one episode_start and episode_end"
        )
    start, end = starts[0], ends[0]
    commands = [row for row in rows if row.get("event_type") == "command_executed"]
    policy_commands = [row for row in commands if not bool(row.get("hold"))]
    hold_commands = [row for row in commands if bool(row.get("hold"))]
    observations = [
        _observation(row)
        for row in rows
        if row.get("event_type") in {"observation", "task_state_observed"}
    ]
    observations.sort(key=lambda item: int(item.get("observation_step", 0)))
    switch_events = [
        row for row in rows if row.get("event_type") == "destination_switched"
    ]
    switch_event = switch_events[0] if switch_events else None
    switch = _merged(switch_event) if switch_event is not None else {}
    switch_step = int(
        switch.get(
            "step",
            switch.get(
                "switch_step",
                start.get(
                    "switch_step",
                    observations[0].get("switch_step", 0) if observations else 0,
                ),
            ),
        )
    )
    generation_after = int(
        switch.get(
            "generation_after",
            switch.get("generation_id", start.get("post_switch_generation", 1)),
        )
    )
    original_destination = _finite_xyz(
        switch.get("old_destination_xyz", start.get("original_destination_xyz"))
    )
    final_destination = _finite_xyz(
        switch.get("new_destination_xyz", start.get("final_destination_xyz"))
    )
    if observations:
        original_destination = original_destination or _finite_xyz(
            observations[0].get("original_destination_xyz")
        )
        final_destination = final_destination or _finite_xyz(
            observations[0].get("final_destination_xyz")
        )

    max_steps = int(
        start.get("maximum_episode_steps", end.get("maximum_episode_steps", 320))
    )
    placement_tolerance = float(start.get("placement_tolerance_m", 0.04))
    placement_height_tolerance = float(start.get("placement_height_tolerance_m", 0.035))
    stable_steps_required = int(start.get("stable_placement_steps", 20))
    stable_linear_speed = float(start.get("stable_linear_speed_mps", 0.03))
    stable_angular_speed = float(start.get("stable_angular_speed_rps", 0.50))
    lift_clearance = float(start.get("lift_clearance_m", 0.10))
    collision_threshold = float(start.get("collision_threshold_n", 40.0))
    final_observation = observations[-1] if observations else {}
    grasp_success = any(bool(item.get("grasped")) for item in observations)
    lift_success = any(
        float(item["object_position_xyz"][2]) - float(item["initial_object_z"])
        >= lift_clearance
        for item in observations
    )
    gripper_open_aperture = float(start.get("gripper_open_aperture_m", 0.07))
    stable_success = False
    stable_streak = 0
    previous_observation_step: int | None = None
    obsolete_placements = 0
    previously_obsolete = False
    for item in observations:
        object_xyz = _finite_xyz(item.get("object_position_xyz"))
        destination_xyz = (
            _finite_xyz(item.get("final_destination_xyz")) or final_destination
        )
        linear_velocity = _finite_xyz(item.get("object_linear_velocity_xyz"))
        angular_velocity = _finite_xyz(item.get("object_angular_velocity_xyz"))
        if object_xyz is None or destination_xyz is None:
            continue
        xy_error = math.hypot(
            object_xyz[0] - destination_xyz[0], object_xyz[1] - destination_xyz[1]
        )
        height_error = abs(object_xyz[2] - destination_xyz[2])
        velocity_stable = (
            linear_velocity is not None
            and angular_velocity is not None
            and math.sqrt(sum(value * value for value in linear_velocity))
            <= stable_linear_speed
            and math.sqrt(sum(value * value for value in angular_velocity))
            <= stable_angular_speed
        )
        correct_now = (
            float(item.get("gripper_aperture_m", 0.0)) >= gripper_open_aperture
            and xy_error <= placement_tolerance
            and height_error <= placement_height_tolerance
            and velocity_stable
        )
        observation_step = int(item.get("observation_step", 0))
        if correct_now and (
            previous_observation_step is None
            or observation_step == previous_observation_step + 1
        ):
            stable_streak += 1
        elif correct_now:
            stable_streak = 1
        else:
            stable_streak = 0
        previous_observation_step = observation_step
        if stable_streak >= stable_steps_required:
            stable_success = True
        obsolete_destination = (
            _finite_xyz(item.get("original_destination_xyz")) or original_destination
        )
        obsolete_now = False
        if object_xyz is not None and obsolete_destination is not None:
            obsolete_now = (
                float(item.get("gripper_aperture_m", 0.0)) >= gripper_open_aperture
                and math.hypot(
                    object_xyz[0] - obsolete_destination[0],
                    object_xyz[1] - obsolete_destination[1],
                )
                <= placement_tolerance
                and abs(object_xyz[2] - obsolete_destination[2])
                <= placement_height_tolerance
            )
        if obsolete_now and not previously_obsolete:
            obsolete_placements += 1
        previously_obsolete = obsolete_now
    released = (
        float(final_observation.get("gripper_aperture_m", 0.0)) >= gripper_open_aperture
    )
    final_object = _finite_xyz(final_observation.get("object_position_xyz"))
    final_xy_error = (
        math.hypot(
            final_object[0] - final_destination[0],
            final_object[1] - final_destination[1],
        )
        if final_object is not None and final_destination is not None
        else math.inf
    )
    final_height_error = (
        abs(final_object[2] - final_destination[2])
        if final_object is not None and final_destination is not None
        else math.inf
    )
    collision = any(
        float(item.get("max_disallowed_contact_force_n", 0.0)) >= collision_threshold
        for item in observations
    ) or any(
        row.get("event_type") == "collision_detected"
        and bool(_merged(row).get("disqualifying", True))
        for row in rows
    )
    limit_failure = any(
        bool(item.get("joint_or_workspace_limit")) for item in observations
    )
    correct_success = (
        len(switch_events) == 1
        and bool(final_observation.get("disturbance_switched"))
        and int(final_observation.get("generation_id", 0)) == 2
        and generation_after == 2
        and grasp_success
        and lift_success
        and stable_success
        and released
        and final_xy_error <= placement_tolerance
        and final_height_error <= placement_height_tolerance
        and not collision
        and not limit_failure
    )
    task_terminated = next(
        (
            _merged(row)
            for row in reversed(rows)
            if row.get("event_type") == "task_terminated"
        ),
        {},
    )
    completion_reason = str(
        task_terminated.get(
            "reason", end.get("completion_reason", "transport_or_simulator_failure")
        )
    )
    timeout = completion_reason in {
        "task_timeout",
        "step_limit",
        "destination_switch_recovery_timeout",
    }
    completion_steps = int(
        end.get(
            "completion_steps",
            final_observation.get(
                "episode_step",
                max(
                    (int(row.get("actual_target_step", 0)) for row in commands),
                    default=0,
                ),
            ),
        )
    )

    action_ages = [
        float(
            int(row.get("actual_target_step", 0))
            - int(row.get("source_observation_step", 0))
        )
        for row in policy_commands
    ]
    latencies = [
        float(row.get("latency_ms", 0.0))
        for row in rows
        if row.get("event_type") == "chunk_arrived" and not bool(row.get("duplicate"))
    ]
    expired_executed = sum(
        int(row.get("source_target_step", 0)) < int(row.get("actual_target_step", 0))
        for row in policy_commands
    )
    expired_removed = sum(
        int(row.get("expired_actions_removed", 0))
        for row in rows
        if row.get("event_type") == "queue_updated"
    ) + sum(
        row.get("event_type") == "action_discarded"
        and row.get("reason") == "expired_before_execution"
        for row in rows
    )
    duplicate_targets = sum(
        int(row.get("duplicate_target_actions_removed", 0))
        for row in rows
        if row.get("event_type") == "queue_updated"
    )
    response_wait_ns = sum(
        int(row.get("duration_ns", 0))
        for row in rows
        if row.get("event_type") == "sync_wait"
        and bool(row.get("counts_as_hold", True))
    )

    pre_switch_commands = 0
    obsolete_commands = 0
    recovery_step: int | None = None
    for row in policy_commands:
        actual = int(row.get("actual_target_step", 0))
        source_observation = int(row.get("source_observation_step", 0))
        source_generation = int(row.get("source_generation_id", 0))
        if switch_step and actual > switch_step and source_observation < switch_step:
            pre_switch_commands += 1
        if (
            switch_step
            and actual > switch_step
            and original_destination
            and final_destination
        ):
            command_xyz = _finite_xyz(row.get("command", ())[:3])
            if command_xyz is not None and command_targets_obsolete_destination(
                command_xyz,
                obsolete_destination_xyz=original_destination,
                final_destination_xyz=final_destination,
            ):
                obsolete_commands += 1
        if (
            recovery_step is None
            and switch_step
            and actual > switch_step
            and source_observation >= switch_step
            and source_generation == generation_after
        ):
            recovery_step = actual

    obsolete_motion_steps = 0
    progress_step: int | None = None
    switch_observation = next(
        (
            item
            for item in observations
            if int(item.get("observation_step", 0)) >= switch_step
        ),
        None,
    )
    initial_final_distance = None
    if switch_observation is not None and final_destination is not None:
        switch_object = _finite_xyz(switch_observation.get("object_position_xyz"))
        if switch_object is not None:
            initial_final_distance = euclidean_distance(
                switch_object, final_destination
            )
    for previous, current in zip(observations, observations[1:]):
        step = int(current.get("observation_step", 0))
        if not switch_step or step <= switch_step:
            continue
        previous_xyz = _finite_xyz(previous.get("object_position_xyz"))
        current_xyz = _finite_xyz(current.get("object_position_xyz"))
        if previous_xyz and current_xyz and original_destination and final_destination:
            if motion_targets_obsolete_destination(
                previous_xyz,
                current_xyz,
                obsolete_destination_xyz=original_destination,
                final_destination_xyz=final_destination,
            ):
                obsolete_motion_steps += 1
            if (
                progress_step is None
                and initial_final_distance is not None
                and initial_final_distance
                - euclidean_distance(current_xyz, final_destination)
                > 1e-4
            ):
                progress_step = step

    recovery_latency = None if recovery_step is None else recovery_step - switch_step
    penalized_recovery = (
        recovery_latency
        if recovery_latency is not None
        else max(0, max_steps - switch_step + 1)
    )
    progress_latency = None if progress_step is None else progress_step - switch_step
    start_wall = int(start.get("wall_time_ns", 0))
    end_wall = int(
        end.get(
            "wall_time_ns",
            max((int(row.get("wall_time_ns", 0)) for row in rows), default=start_wall),
        )
    )
    start_sim = int(start.get("sim_time_ns", 0))
    end_sim = int(
        end.get(
            "sim_time_ns",
            max((int(row.get("sim_time_ns", 0)) for row in rows), default=start_sim),
        )
    )
    duplicate_arrivals = sum(
        row.get("event_type") == "chunk_arrived" and bool(row.get("duplicate"))
        for row in rows
    )
    if duplicate_arrivals == 0:
        duplicate_arrivals = sum(
            row.get("event_type") == "chunk_rejected"
            and row.get("reason") == "duplicate_response"
            for row in rows
        )
    command_motion_audits = [
        _merged(row) for row in rows if row.get("event_type") == "command_motion_audit"
    ]
    requested_target_deltas = [
        float(row["requested_target_delta_m"])
        for row in command_motion_audits
        if row.get("requested_target_delta_m") is not None
    ]
    applied_target_deltas = [
        float(row["applied_target_delta_m"])
        for row in command_motion_audits
        if row.get("applied_target_delta_m") is not None
    ]
    measured_end_effector_deltas = [
        float(row["measured_end_effector_delta_m"])
        for row in command_motion_audits
        if row.get("measured_end_effector_delta_m") is not None
    ]
    metrics = {
        "task_success": correct_success,
        "correct_destination_placement_success": correct_success,
        "obsolete_destination_placement_count": obsolete_placements,
        "grasp_success": grasp_success,
        "stable_placement_success": stable_success,
        "collision": collision,
        "timeout": timeout,
        "completion_reason": completion_reason,
        "completion_steps": completion_steps,
        "wall_clock_seconds": max(0, end_wall - start_wall) / 1e9,
        "simulation_seconds": max(0, end_sim - start_sim) / 1e9,
        "hold_control_steps": len(hold_commands),
        "hold_control_seconds": len(hold_commands) * CONTROL_PERIOD_SECONDS,
        "response_wait_wall_seconds": response_wait_ns / 1e9,
        "recovery_observed": recovery_latency is not None,
        "recovery_latency_steps": recovery_latency,
        "recovery_latency_seconds": (
            None
            if recovery_latency is None
            else recovery_latency * CONTROL_PERIOD_SECONDS
        ),
        "penalized_recovery_latency_steps": penalized_recovery,
        "time_to_final_destination_progress_steps": progress_latency,
        "time_to_final_destination_progress_seconds": (
            None
            if progress_latency is None
            else progress_latency * CONTROL_PERIOD_SECONDS
        ),
        "obsolete_destination_motion_steps": obsolete_motion_steps,
        "obsolete_destination_motion_seconds": obsolete_motion_steps
        * CONTROL_PERIOD_SECONDS,
        "pre_switch_observation_command_steps": pre_switch_commands,
        "obsolete_destination_command_steps": obsolete_commands,
        "expired_actions_executed": expired_executed,
        "expired_actions_removed": expired_removed,
        "stale_generation_responses_rejected": sum(
            row.get("event_type") == "chunk_rejected"
            and row.get("reason") == "stale_generation"
            for row in rows
        ),
        "duplicate_responses": duplicate_arrivals,
        "duplicate_target_actions_removed": duplicate_targets,
        "out_of_order_completions": sum(
            row.get("event_type") == "chunk_arrived" and bool(row.get("out_of_order"))
            for row in rows
        ),
        "inference_requests": sum(
            row.get("event_type") == "inference_request" for row in rows
        ),
        "returned_responses": sum(
            row.get("event_type") == "chunk_arrived" for row in rows
        ),
        "dropped_responses": sum(
            row.get("event_type") == "response_dropped" for row in rows
        ),
        "queue_rebuild_count": sum(
            row.get("event_type") == "queue_updated"
            and row.get("reason") != "arrival_order_append"
            for row in rows
        ),
        "deadline_misses": len(hold_commands),
        "deadline_miss_rate": len(hold_commands) / len(commands) if commands else 0.0,
        "executed_actions": len(policy_commands),
        "mean_inference_latency_ms": sum(latencies) / len(latencies)
        if latencies
        else None,
        "p50_inference_latency_ms": _percentile(latencies, 0.50),
        "p95_inference_latency_ms": _percentile(latencies, 0.95),
        "mean_action_age_steps": sum(action_ages) / len(action_ages)
        if action_ages
        else None,
        "p50_action_age_steps": _percentile(action_ages, 0.50),
        "p95_action_age_steps": _percentile(action_ages, 0.95),
        "command_motion_audit_count": (
            len(command_motion_audits) if command_motion_audits else None
        ),
        "maximum_requested_target_delta_m": (
            max(requested_target_deltas) if requested_target_deltas else None
        ),
        "maximum_applied_target_delta_m": (
            max(applied_target_deltas) if applied_target_deltas else None
        ),
        "maximum_measured_end_effector_delta_m": (
            max(measured_end_effector_deltas) if measured_end_effector_deltas else None
        ),
        "requested_target_discontinuity_count": (
            sum(
                float(row.get("requested_target_delta_m", 0.0))
                > float(row.get("planned_row_translation_limit_m", math.inf))
                for row in command_motion_audits
            )
            if command_motion_audits
            else None
        ),
        "applied_target_discontinuity_count": (
            sum(
                float(row.get("applied_target_delta_m", 0.0))
                > float(row.get("planned_row_translation_limit_m", math.inf))
                for row in command_motion_audits
            )
            if command_motion_audits
            else None
        ),
        "hold_requested_applied_target_mismatch_count": (
            sum(
                bool(row.get("hold"))
                and abs(
                    float(row.get("requested_target_delta_m", 0.0))
                    - float(row.get("applied_target_delta_m", 0.0))
                )
                > 1e-12
                for row in command_motion_audits
            )
            if command_motion_audits
            else None
        ),
    }
    return metrics


def _metric_equal(recorded: Any, recomputed: Any) -> bool:
    if recorded is None or recomputed is None:
        return recorded is recomputed
    if isinstance(recorded, bool) or isinstance(recomputed, bool):
        return recorded is recomputed
    if isinstance(recorded, (int, float)) and isinstance(recomputed, (int, float)):
        return abs(float(recorded) - float(recomputed)) <= 1e-9 * max(
            1.0, abs(float(recorded)), abs(float(recomputed))
        )
    return recorded == recomputed


def validate_episode_rows(
    rows: list[dict[str, Any]],
    summary: Mapping[str, Any],
    *,
    source: str = "<memory>",
    required_fairness_fields: Sequence[str] = FAIRNESS_FIELDS,
) -> dict[str, Any]:
    recomputed = recompute_metrics(rows)
    recorded = summary.get("metrics", {})
    mismatches = {
        metric: {"recorded": recorded.get(metric), "recomputed": value}
        for metric, value in recomputed.items()
        if metric not in recorded or not _metric_equal(recorded.get(metric), value)
    }
    violations: list[dict[str, Any]] = []
    expected_indices = list(range(len(rows)))
    indices = [int(row.get("event_index", -1)) for row in rows]
    if indices != expected_indices:
        violations.append(
            {"event_index": None, "reason": "non_contiguous_event_indices"}
        )
    strategy = str(summary.get("strategy", ""))
    starts = [row for row in rows if row.get("event_type") == "episode_start"]
    switches = [row for row in rows if row.get("event_type") == "destination_switched"]
    ends = [row for row in rows if row.get("event_type") == "episode_end"]
    if starts and starts[0].get("command_motion_audit_required") is True:
        executed = [row for row in rows if row.get("event_type") == "command_executed"]
        motion_audits = [
            _merged(row)
            for row in rows
            if row.get("event_type") == "command_motion_audit"
        ]
        if len(motion_audits) != len(executed):
            violations.append(
                {
                    "event_index": None,
                    "reason": "command_motion_audit_count_mismatch",
                    "commands": len(executed),
                    "audits": len(motion_audits),
                }
            )
        required_motion_fields = (
            "actual_target_step",
            "requested_target_delta_m",
            "applied_target_delta_m",
            "measured_end_effector_delta_m",
            "planned_row_translation_limit_m",
            "planned_row_limit_scope",
        )
        for audit in motion_audits:
            missing = [field for field in required_motion_fields if field not in audit]
            numeric = (
                audit.get("requested_target_delta_m"),
                audit.get("applied_target_delta_m"),
                audit.get("measured_end_effector_delta_m"),
                audit.get("planned_row_translation_limit_m"),
            )
            if (
                missing
                or audit.get("planned_row_limit_scope") != "adjacent_planned_rows_only"
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    for value in numeric
                )
            ):
                violations.append(
                    {
                        "event_index": audit.get("event_index"),
                        "reason": "command_motion_audit_invalid",
                        "missing": missing,
                    }
                )
    if len(starts) != 1 or len(switches) > 1 or len(ends) != 1:
        violations.append({"event_index": None, "reason": "lifecycle_event_count"})
    elif len(switches) == 1 and not (
        int(starts[0].get("event_index", -1))
        < int(switches[0].get("event_index", -1))
        < int(ends[0].get("event_index", -1))
    ):
        violations.append({"event_index": None, "reason": "lifecycle_event_order"})
    elif len(switches) == 0:
        terminations = [
            row for row in rows if row.get("event_type") == "task_terminated"
        ]
        planned_switch = int(summary.get("switch_step", -1))
        terminal = _merged(terminations[0]) if len(terminations) == 1 else {}
        terminal_step = int(
            terminal.get(
                "terminal_step",
                terminal.get(
                    "episode_step", ends[0].get("completion_steps", -1) if ends else -1
                ),
            )
        )
        max_observation_step = max(
            (
                int(row.get("observation_step", row.get("sim_step", -1)))
                for row in rows
                if row.get("event_type") in {"observation", "task_state_observed"}
            ),
            default=-1,
        )
        if not (
            len(terminations) == 1
            and starts
            and ends
            and int(starts[0].get("event_index", -1))
            < int(terminations[0].get("event_index", -1))
            < int(ends[0].get("event_index", -1))
            and terminal.get("success") is False
            and ends[0].get("success") is False
            and 0 <= terminal_step < planned_switch
            and max_observation_step < planned_switch
        ):
            violations.append(
                {"event_index": None, "reason": "invalid_zero_switch_failure"}
            )
    terminations = [row for row in rows if row.get("event_type") == "task_terminated"]
    if len(terminations) != 1:
        violations.append(
            {"event_index": None, "reason": "task_termination_event_count"}
        )
    else:
        terminal_success = _merged(terminations[0]).get("success")
        if (
            not isinstance(terminal_success, bool)
            or terminal_success != recomputed["task_success"]
        ):
            violations.append(
                {
                    "event_index": terminations[0].get("event_index"),
                    "reason": "recomputed_terminal_success_mismatch",
                }
            )
    if len(ends) == 1:
        end_success = ends[0].get("success")
        if (
            not isinstance(end_success, bool)
            or end_success != recomputed["task_success"]
        ):
            violations.append(
                {
                    "event_index": ends[0].get("event_index"),
                    "reason": "recomputed_episode_end_success_mismatch",
                }
            )
    if starts:
        start = starts[0]
        for field in required_fairness_fields:
            if field not in start:
                violations.append(
                    {
                        "event_index": start.get("event_index"),
                        "reason": "raw_fairness_field_missing",
                        "field": field,
                    }
                )
            elif start.get(field) != summary.get(field):
                violations.append(
                    {
                        "event_index": start.get("event_index"),
                        "reason": "raw_summary_fairness_mismatch",
                        "field": field,
                    }
                )
        for field in ("episode_id", "profile_id", "seed", "strategy"):
            if start.get(field) != summary.get(field):
                violations.append(
                    {
                        "event_index": start.get("event_index"),
                        "reason": "raw_summary_identity_mismatch",
                        "field": field,
                    }
                )
        scenario_payload = start.get("scenario")
        reset_payload = start.get("reset_state")
        if not isinstance(scenario_payload, Mapping):
            violations.append(
                {
                    "event_index": start.get("event_index"),
                    "reason": "raw_scenario_payload_missing",
                }
            )
        else:
            scenario_core = dict(scenario_payload)
            embedded_scenario_sha = scenario_core.pop("scenario_sha256", None)
            recomputed_scenario_sha = canonical_sha256(scenario_core)
            if (
                embedded_scenario_sha is not None
                and embedded_scenario_sha != recomputed_scenario_sha
            ):
                violations.append(
                    {
                        "event_index": start.get("event_index"),
                        "reason": "raw_scenario_self_hash_mismatch",
                    }
                )
            if recomputed_scenario_sha != start.get("scenario_sha256"):
                violations.append(
                    {
                        "event_index": start.get("event_index"),
                        "reason": "raw_scenario_sha256_mismatch",
                    }
                )
            for nested_name, flat_name in (
                ("seed", "seed"),
                ("switch_step", "switch_step"),
                ("object_position_xyz", "object_position_xyz"),
                ("original_destination_xyz", "original_destination_xyz"),
                ("final_destination_xyz", "final_destination_xyz"),
            ):
                if scenario_core.get(nested_name) != summary.get(flat_name):
                    violations.append(
                        {
                            "event_index": start.get("event_index"),
                            "reason": "raw_scenario_summary_mismatch",
                            "field": nested_name,
                        }
                    )
        if not isinstance(reset_payload, Mapping):
            violations.append(
                {
                    "event_index": start.get("event_index"),
                    "reason": "raw_reset_state_payload_missing",
                }
            )
        else:
            reset_core = dict(reset_payload)
            embedded_reset_sha = reset_core.pop("reset_state_sha256", None)
            embedded_paired_reset_sha = reset_core.pop(
                "paired_reset_canonical_sha256", None
            )
            recomputed_reset_sha = canonical_sha256(reset_core)
            if (
                embedded_reset_sha is not None
                and embedded_reset_sha != recomputed_reset_sha
            ):
                violations.append(
                    {
                        "event_index": start.get("event_index"),
                        "reason": "raw_reset_state_self_hash_mismatch",
                    }
                )
            if recomputed_reset_sha != start.get("reset_state_sha256"):
                violations.append(
                    {
                        "event_index": start.get("event_index"),
                        "reason": "raw_reset_state_sha256_mismatch",
                    }
                )
            try:
                recomputed_paired_reset_sha = canonical_sha256(
                    paired_reset_canonical_payload(reset_core)
                )
            except (KeyError, TypeError, ValueError) as exc:
                violations.append(
                    {
                        "event_index": start.get("event_index"),
                        "reason": "raw_paired_reset_canonical_payload_invalid",
                        "detail": str(exc),
                    }
                )
            else:
                if (
                    embedded_paired_reset_sha is not None
                    and embedded_paired_reset_sha != recomputed_paired_reset_sha
                ):
                    violations.append(
                        {
                            "event_index": start.get("event_index"),
                            "reason": "raw_paired_reset_canonical_self_hash_mismatch",
                        }
                    )
                if recomputed_paired_reset_sha != start.get(
                    "paired_reset_canonical_sha256"
                ):
                    violations.append(
                        {
                            "event_index": start.get("event_index"),
                            "reason": "raw_paired_reset_canonical_sha256_mismatch",
                        }
                    )
            required_reset_fields = (
                "seed",
                "robot_joint_positions",
                "robot_joint_velocities",
                "end_effector_position_xyz",
                "end_effector_orientation_wxyz",
                "object_position_xyz",
                "object_orientation_wxyz",
                "zone_a_xyz",
                "zone_b_xyz",
                "physics_dt_seconds",
                "rendering_dt_seconds",
                "stage_units_in_meters",
                "gravity_xyz",
            )
            missing_reset = [
                name for name in required_reset_fields if name not in reset_core
            ]
            if missing_reset:
                violations.append(
                    {
                        "event_index": start.get("event_index"),
                        "reason": "raw_reset_state_fields_missing",
                        "fields": missing_reset,
                    }
                )
            if reset_core.get("seed") != summary.get("seed"):
                violations.append(
                    {
                        "event_index": start.get("event_index"),
                        "reason": "raw_reset_state_seed_mismatch",
                    }
                )
            if isinstance(scenario_payload, Mapping):
                for field in ("zone_a_xyz", "zone_b_xyz"):
                    if reset_core.get(field) != scenario_core.get(field):
                        violations.append(
                            {
                                "event_index": start.get("event_index"),
                                "reason": "raw_reset_scenario_mismatch",
                                "field": field,
                            }
                        )
    if starts and switches:
        switch_detail = _merged(switches[0])
        switch_step = int(
            switch_detail.get("step", switch_detail.get("switch_step", -1))
        )
        if switch_step != int(summary.get("switch_step", -2)):
            violations.append(
                {
                    "event_index": switches[0].get("event_index"),
                    "reason": "raw_summary_switch_step_mismatch",
                }
            )
        for raw_name, summary_name in (
            ("old_destination_xyz", "original_destination_xyz"),
            ("new_destination_xyz", "final_destination_xyz"),
        ):
            if _finite_xyz(switch_detail.get(raw_name)) != _finite_xyz(
                summary.get(summary_name)
            ):
                violations.append(
                    {
                        "event_index": switches[0].get("event_index"),
                        "reason": "raw_summary_destination_mismatch",
                        "field": summary_name,
                    }
                )
    active_episode = str(starts[0].get("episode_id", "")) if starts else ""
    active_generation = int(starts[0].get("generation_id", 0)) if starts else 0
    rejected = {
        int(row.get("request_id", 0))
        for row in rows
        if row.get("event_type") == "chunk_rejected"
        and row.get("reason") != "duplicate_response"
    }
    semantic_switch_step = int(summary.get("switch_step", 0))
    generation_before_switch = int(starts[0].get("generation_id", 1)) if starts else 1
    generation_after_switch = 2
    if switches:
        switch_contract = _merged(switches[0])
        semantic_switch_step = int(
            switch_contract.get(
                "step", switch_contract.get("switch_step", semantic_switch_step)
            )
        )
        generation_before_switch = int(
            switch_contract.get("generation_before", generation_before_switch)
        )
        generation_after_switch = int(
            switch_contract.get("generation_after", generation_after_switch)
        )
    terminal_rows = [row for row in rows if row.get("event_type") == "task_terminated"]
    semantic_terminal_step: int | None = None
    if len(terminal_rows) == 1:
        terminal_detail = _merged(terminal_rows[0])
        explicit_terminal = terminal_detail.get(
            "terminal_step", terminal_detail.get("source_observation_step")
        )
        if explicit_terminal is not None and int(explicit_terminal) >= 0:
            semantic_terminal_step = int(explicit_terminal)
        elif len(ends) == 1 and int(ends[0].get("completion_steps", -1)) >= 0:
            # Older/lean event fixtures do not duplicate the terminal step on
            # task_terminated.  The recomputed completion step is still a
            # semantic boundary, unlike cross-topic event delivery order.
            semantic_terminal_step = int(ends[0]["completion_steps"])
    executed_targets: Counter[int] = Counter()
    # Cross-topic recorder delivery order is not a semantic ordering. Index
    # complete raw chunks up front, while event_index remains authoritative for
    # executor RuntimeEvents within a topic.
    arrived_chunks: dict[int, dict[str, Any]] = {}
    for candidate in rows:
        if candidate.get("event_type") == "chunk_arrived" and not bool(
            candidate.get("duplicate")
        ):
            arrived_chunks.setdefault(int(candidate.get("request_id", 0)), candidate)
    first_actions: dict[tuple[int, int], Any] = {}
    accepted_requests: set[int] = set()
    latest_plan_key_by_generation: dict[int, tuple[int, int]] = {}
    reconstructed_queue: dict[int, dict[str, Any]] = {}
    naive_queue: list[dict[str, Any]] = []
    latest_observation_step = -1
    latest_runtime_actual = -1
    runtime_commands: dict[int, dict[str, Any]] = {}
    runtime_expected_commands: dict[int, Any] = {}
    runtime_holds: set[int] = set()
    obsolete_classifications = {
        (
            int(row.get("actual_target_step", -1)),
            int(row.get("source_request_id", 0)),
        ): bool(row.get("obsolete_destination_command"))
        for row in rows
        if row.get("event_type") == "obsolete_command_classified"
    }
    last_actual = -1
    for row in rows:
        event_type = row.get("event_type")
        if event_type == "generation_advanced":
            active_generation = int(
                row.get("generation_id", row.get("generation_after", 0))
            )
            if strategy in {"sync_hold", "aligned_async"}:
                if int(row.get("queue_length_before", -1)) != len(reconstructed_queue):
                    violations.append(
                        {
                            "event_index": row.get("event_index"),
                            "reason": "generation_queue_before_mismatch",
                        }
                    )
                if int(row.get("queue_length_after", -1)) != 0:
                    violations.append(
                        {
                            "event_index": row.get("event_index"),
                            "reason": "generation_queue_not_invalidated",
                        }
                    )
                if int(row.get("action_count", 0)) != len(reconstructed_queue):
                    violations.append(
                        {
                            "event_index": row.get("event_index"),
                            "reason": "generation_removed_count_mismatch",
                        }
                    )
                reconstructed_queue.clear()
            latest_plan_key_by_generation.clear()
        elif event_type == "destination_switched":
            detail = _merged(row)
            active_generation = int(
                detail.get(
                    "generation_after", detail.get("generation_id", active_generation)
                )
            )
        elif event_type in {
            "observation",
            "task_state_observed",
            "observation_received",
        }:
            latest_observation_step = int(
                row.get("observation_step", row.get("sim_step", -1))
            )
            if event_type == "observation_received":
                latest_observation_step = int(
                    row.get("actual_target_step", latest_observation_step)
                )
        elif event_type == "chunk_arrived":
            arrived_chunks.setdefault(int(row.get("request_id", 0)), row)
        elif event_type == "queue_cleared":
            if strategy != "sync_hold" or row.get("reason") != "sync_periodic_replan":
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "unexpected_queue_clear",
                    }
                )
            if int(row.get("queue_length_before", -1)) != len(reconstructed_queue):
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "queue_clear_before_mismatch",
                    }
                )
            if int(row.get("queue_length_after", -1)) != 0 or int(
                row.get("action_count", -1)
            ) != len(reconstructed_queue):
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "queue_clear_count_mismatch",
                    }
                )
            reconstructed_queue.clear()
        elif event_type == "queue_updated" and strategy == "naive_async":
            request_id = int(row.get("request_id", 0))
            arrived = arrived_chunks.get(request_id)
            actions = None if arrived is None else arrived.get("actions")
            if not isinstance(actions, list) or not actions:
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "naive_append_missing_raw_chunk",
                    }
                )
                continue
            if int(row.get("queue_length_before", -1)) != len(naive_queue):
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "naive_fifo_before_mismatch",
                    }
                )
            for action in actions:
                if (
                    not isinstance(action, Mapping)
                    or "target_step" not in action
                    or "command" not in action
                ):
                    violations.append(
                        {
                            "event_index": row.get("event_index"),
                            "reason": "naive_fifo_malformed_action",
                        }
                    )
                    continue
                naive_queue.append(
                    {
                        "request_id": request_id,
                        "generation_id": int(row.get("generation_id", 0)),
                        "source_observation_step": int(
                            row.get("source_observation_step", 0)
                        ),
                        "source_target_step": int(action["target_step"]),
                        "command": action["command"],
                    }
                )
            if row.get("reason") != "arrival_order_append":
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "naive_append_reason_mismatch",
                    }
                )
            if int(row.get("queue_length_after", -1)) != len(naive_queue) or int(
                row.get("action_count", -1)
            ) != len(actions):
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "naive_fifo_after_mismatch",
                    }
                )
        elif event_type == "queue_updated" and strategy in {
            "sync_hold",
            "aligned_async",
        }:
            generation = int(row.get("generation_id", active_generation))
            key = (
                int(row.get("source_observation_step", 0)),
                int(row.get("request_id", 0)),
            )
            if strategy == "aligned_async":
                latest = latest_plan_key_by_generation.get(generation)
                if latest is not None and key <= latest:
                    violations.append(
                        {
                            "event_index": row.get("event_index"),
                            "reason": "same_generation_freshness_violation",
                        }
                    )
                latest_plan_key_by_generation[generation] = key
            arrived = arrived_chunks.get(key[1])
            if arrived is None:
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "queue_rebuild_missing_raw_chunk",
                    }
                )
                continue
            actions = arrived.get("actions", [])
            if not isinstance(actions, list) or not actions:
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "queue_rebuild_actions_missing",
                    }
                )
                continue
            # The executor snapshots the observation boundary used for atomic
            # replacement. Cross-topic recorder arrival order is not semantic:
            # a later observation can be written before this RuntimeEvent.
            queue_detail = _event_detail(row)
            insertion_observation_step = queue_detail.get(
                "insertion_observation_step"
            )
            if insertion_observation_step is None:
                # Backward compatibility for fixtures and evidence recorded
                # before the executor emitted the explicit boundary snapshot.
                insertion_step = latest_observation_step + 1
            else:
                try:
                    insertion_observation_step = int(insertion_observation_step)
                except (TypeError, ValueError):
                    insertion_observation_step = -1
                if insertion_observation_step < 0:
                    violations.append(
                        {
                            "event_index": row.get("event_index"),
                            "reason": "queue_insertion_observation_step_invalid",
                        }
                    )
                if (
                    "actual_target_step" in row
                    and int(row.get("actual_target_step", -1))
                    != insertion_observation_step
                ):
                    violations.append(
                        {
                            "event_index": row.get("event_index"),
                            "reason": "queue_insertion_observation_step_mismatch",
                        }
                    )
                insertion_step = insertion_observation_step + 1
            expected: dict[int, dict[str, Any]] = {}
            expired = 0
            duplicates = 0
            malformed = False
            for action in actions:
                if (
                    not isinstance(action, Mapping)
                    or "target_step" not in action
                    or "command" not in action
                ):
                    malformed = True
                    continue
                target = int(action["target_step"])
                if target < insertion_step or target <= latest_runtime_actual:
                    expired += 1
                    continue
                if target in expected:
                    duplicates += 1
                    continue
                expected[target] = {
                    "request_id": key[1],
                    "generation_id": generation,
                    "source_observation_step": key[0],
                    "command": action["command"],
                }
                first_actions[(key[1], target)] = action["command"]
            if malformed:
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "queue_rebuild_malformed_action",
                    }
                )
            expected = dict(sorted(expected.items()))
            if int(row.get("queue_length_before", -1)) != len(reconstructed_queue):
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "atomic_rebuild_queue_before_mismatch",
                    }
                )
            if int(row.get("queue_length_after", -1)) != len(expected):
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "atomic_rebuild_queue_after_mismatch",
                    }
                )
            if int(row.get("action_count", -1)) != len(expected):
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "atomic_rebuild_action_count_mismatch",
                    }
                )
            expected_reason = (
                "sync_valid_rebuild"
                if strategy == "sync_hold"
                else "aligned_atomic_rebuild"
            )
            if row.get("reason") != expected_reason:
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "atomic_rebuild_reason_mismatch",
                    }
                )
            if int(row.get("expired_actions_removed", -1)) != expired:
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "expired_prefix_count_mismatch",
                    }
                )
            if int(row.get("duplicate_target_actions_removed", -1)) != duplicates:
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "duplicate_target_count_mismatch",
                    }
                )
            reconstructed_queue = expected
            accepted_requests.add(key[1])
        elif event_type == "action_discarded" and strategy in {
            "sync_hold",
            "aligned_async",
        }:
            target = int(row.get("source_target_step", -1))
            if (
                row.get("reason") != "expired_before_execution"
                or target not in reconstructed_queue
            ):
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "unreconstructable_expired_discard",
                    }
                )
            else:
                reconstructed_queue.pop(target)
            if int(row.get("queue_length_after", -1)) != len(reconstructed_queue):
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "discard_queue_after_mismatch",
                    }
                )
        elif event_type == "action_executed":
            actual = int(row.get("actual_target_step", -1))
            latest_runtime_actual = max(latest_runtime_actual, actual)
            if actual in runtime_commands:
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "duplicate_runtime_target_execution",
                    }
                )
            runtime_commands[actual] = row
            if strategy == "naive_async":
                if not naive_queue:
                    violations.append(
                        {
                            "event_index": row.get("event_index"),
                            "reason": "naive_execution_from_empty_fifo",
                        }
                    )
                else:
                    expected = naive_queue.pop(0)
                    for raw_name, expected_name in (
                        ("request_id", "request_id"),
                        ("generation_id", "generation_id"),
                        ("source_observation_step", "source_observation_step"),
                        ("source_target_step", "source_target_step"),
                    ):
                        if int(row.get(raw_name, -1)) != int(expected[expected_name]):
                            violations.append(
                                {
                                    "event_index": row.get("event_index"),
                                    "reason": "naive_fifo_provenance_mismatch",
                                    "field": raw_name,
                                }
                            )
                    runtime_expected_commands[actual] = expected["command"]
                if int(row.get("queue_length_after", -1)) != len(naive_queue):
                    violations.append(
                        {
                            "event_index": row.get("event_index"),
                            "reason": "naive_execution_queue_after_mismatch",
                        }
                    )
            elif strategy in {"sync_hold", "aligned_async"}:
                target = int(row.get("source_target_step", -1))
                expected = reconstructed_queue.get(target)
                if target != actual or expected is None:
                    violations.append(
                        {
                            "event_index": row.get("event_index"),
                            "reason": "executed_action_not_in_reconstructed_queue",
                        }
                    )
                else:
                    for raw_name, expected_name in (
                        ("request_id", "request_id"),
                        ("generation_id", "generation_id"),
                        ("source_observation_step", "source_observation_step"),
                    ):
                        if int(row.get(raw_name, -1)) != int(expected[expected_name]):
                            violations.append(
                                {
                                    "event_index": row.get("event_index"),
                                    "reason": "executed_action_provenance_mismatch",
                                    "field": raw_name,
                                }
                            )
                    reconstructed_queue.pop(target)
                    runtime_expected_commands[actual] = expected["command"]
                if int(row.get("queue_length_after", -1)) != len(reconstructed_queue):
                    violations.append(
                        {
                            "event_index": row.get("event_index"),
                            "reason": "execution_queue_after_mismatch",
                        }
                    )
        elif event_type == "hold_executed":
            actual = int(row.get("actual_target_step", -1))
            latest_runtime_actual = max(latest_runtime_actual, actual)
            runtime_holds.add(actual)
            if strategy in {"sync_hold", "aligned_async"} and int(
                row.get("queue_length_after", -1)
            ) != len(reconstructed_queue):
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "hold_queue_after_mismatch",
                    }
                )
        elif event_type in {"task_terminated", "episode_terminated"}:
            if event_type == "episode_terminated":
                if int(row.get("queue_length_after", -1)) != 0:
                    violations.append(
                        {
                            "event_index": row.get("event_index"),
                            "reason": "termination_queue_not_cleared",
                        }
                    )
                reconstructed_queue.clear()
                naive_queue.clear()
        elif event_type == "command_executed":
            actual = int(row.get("actual_target_step", -1))
            if actual <= last_actual:
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "target_step_not_monotonic",
                    }
                )
            last_actual = actual
            executed_targets[actual] += 1
            if str(row.get("episode_id", "")) != active_episode:
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "previous_episode_execution",
                    }
                )
            if semantic_terminal_step is not None and actual > semantic_terminal_step:
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "post_termination_execution",
                    }
                )
            if bool(row.get("hold")):
                if actual not in runtime_holds and actual not in runtime_commands:
                    # RuntimeEvent delivery may follow RobotCommand in the raw
                    # stream; the complete-log check below resolves ordering.
                    pass
                continue
            request_id = int(row.get("source_request_id", 0))
            generation = int(row.get("source_generation_id", 0))
            source_target = int(row.get("source_target_step", 0))
            if request_id in rejected:
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "rejected_response_execution",
                    }
                )
            if strategy != "naive_async":
                expected_generation = (
                    generation_after_switch
                    if switches and actual > semantic_switch_step
                    else generation_before_switch
                )
                if generation != expected_generation:
                    violations.append(
                        {
                            "event_index": row.get("event_index"),
                            "reason": "stale_generation_execution",
                        }
                    )
            if strategy != "naive_async" and source_target < actual:
                violations.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "expired_action_execution",
                    }
                )
            if (
                strategy in {"sync_hold", "aligned_async"}
                and request_id in accepted_requests
            ):
                first_command = first_actions.get((request_id, source_target))
                if first_command is not None and row.get("command") != first_command:
                    violations.append(
                        {
                            "event_index": row.get("event_index"),
                            "reason": "duplicate_target_first_wins_violation",
                            "target": source_target,
                        }
                    )
            if switches:
                switch_detail = _merged(switches[0])
                switch_step = int(
                    switch_detail.get("step", switch_detail.get("switch_step", 0))
                )
                if actual > switch_step:
                    original = _finite_xyz(
                        switch_detail.get(
                            "old_destination_xyz",
                            starts[0].get("original_destination_xyz"),
                        )
                    )
                    final = _finite_xyz(
                        switch_detail.get(
                            "new_destination_xyz",
                            starts[0].get("final_destination_xyz"),
                        )
                    )
                    command_xyz = _finite_xyz(row.get("command", ())[:3])
                    if original and final and command_xyz:
                        reconstructed_obsolete = command_targets_obsolete_destination(
                            command_xyz,
                            obsolete_destination_xyz=original,
                            final_destination_xyz=final,
                        )
                        classification_key = (actual, request_id)
                        logged_obsolete = row.get(
                            "obsolete_destination_command",
                            obsolete_classifications.get(classification_key),
                        )
                        if logged_obsolete is None:
                            violations.append(
                                {
                                    "event_index": row.get("event_index"),
                                    "reason": "obsolete_command_attribution_missing",
                                }
                            )
                        elif bool(logged_obsolete) != reconstructed_obsolete:
                            violations.append(
                                {
                                    "event_index": row.get("event_index"),
                                    "reason": "obsolete_command_attribution_mismatch",
                                }
                            )
    command_rows = [row for row in rows if row.get("event_type") == "command_executed"]
    for command in command_rows:
        actual = int(command.get("actual_target_step", -1))
        if bool(command.get("hold")):
            if actual not in runtime_holds:
                violations.append(
                    {
                        "event_index": command.get("event_index"),
                        "reason": "hold_missing_runtime_event",
                    }
                )
            continue
        runtime = runtime_commands.get(actual)
        if runtime is None:
            violations.append(
                {
                    "event_index": command.get("event_index"),
                    "reason": "command_missing_runtime_execution_event",
                }
            )
            continue
        for command_name, runtime_name in (
            ("source_request_id", "request_id"),
            ("source_generation_id", "generation_id"),
            ("source_observation_step", "source_observation_step"),
            ("source_target_step", "source_target_step"),
        ):
            if int(command.get(command_name, -1)) != int(runtime.get(runtime_name, -2)):
                violations.append(
                    {
                        "event_index": command.get("event_index"),
                        "reason": "command_runtime_provenance_mismatch",
                        "field": command_name,
                    }
                )
        expected_command = runtime_expected_commands.get(actual)
        if expected_command is None or command.get("command") != expected_command:
            violations.append(
                {
                    "event_index": command.get("event_index"),
                    "reason": "command_payload_not_reconstructed_from_chunk",
                }
            )
    for target, count in executed_targets.items():
        if count > 1:
            violations.append(
                {
                    "event_index": None,
                    "reason": "duplicate_actual_target_execution",
                    "target": target,
                }
            )
    if summary.get("milestone") != M8_MILESTONE:
        violations.append({"event_index": None, "reason": "summary_milestone_mismatch"})
    if summary.get("evidence_class") != NATIVE_ISAAC_EVIDENCE_CLASS:
        violations.append({"event_index": None, "reason": "non_native_isaac_evidence"})
    if strategy == "aligned_async" and recomputed["expired_actions_executed"] != 0:
        violations.append(
            {"event_index": None, "reason": "aligned_expired_execution_count_nonzero"}
        )
    counts = Counter(item["reason"] for item in violations)
    return {
        "schema_version": M8_SCHEMA_VERSION,
        "milestone": M8_MILESTONE,
        "source": source,
        "strategy": strategy,
        "event_count": len(rows),
        "metrics_match": not mismatches,
        "metric_mismatches": mismatches,
        "invariant_violation_counts": dict(sorted(counts.items())),
        "violations": violations,
        "passed": not mismatches and not violations,
        "recomputed_metrics": recomputed,
    }


def validate_episode_log(
    event_log_path: Path | str, summary_path: Path | str
) -> dict[str, Any]:
    return validate_episode_rows(
        read_jsonl(event_log_path),
        read_json(summary_path),
        source=str(Path(event_log_path)),
    )


def validate_fault_trace_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    trace_path: Path | None = None,
    trace: M8FaultTrace | None = None,
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if (trace_path is None) == (trace is None):
        raise ValueError("provide exactly one of trace_path or parsed trace")
    active_trace = trace if trace is not None else load_fault_trace(trace_path)
    assert active_trace is not None
    errors: list[dict[str, Any]] = []
    if active_trace.sha256 != summary.get("fault_trace_sha256"):
        errors.append(
            {"event_index": None, "reason": "fault_trace_summary_sha256_mismatch"}
        )
    requests = [row for row in rows if row.get("event_type") == "inference_request"]
    request_ids = {int(row.get("request_id", 0)) for row in requests}
    fault_response_types = {
        "chunk_scheduled",
        "response_dropped",
        "chunk_delivered",
        "chunk_arrived",
    }
    for row in rows:
        if row.get("event_type") in fault_response_types:
            request_id = int(row.get("request_id", 0))
            if request_id not in request_ids:
                errors.append(
                    {
                        "event_index": row.get("event_index"),
                        "reason": "fault_event_without_inference_request",
                        "request_id": request_id,
                    }
                )

    def steady_time_ns(row: Mapping[str, Any] | None) -> int | None:
        if row is None:
            return None
        try:
            value = int(_merged(row).get("steady_time_ns"))
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    episode_end = next(
        (row for row in reversed(rows) if row.get("event_type") == "episode_end"),
        None,
    )
    episode_end_ns = steady_time_ns(episode_end)
    for request in requests:
        request_id = int(request.get("request_id", 0))
        ordinal = request_id - 1
        try:
            expected = active_trace.entry(ordinal)
        except IndexError:
            errors.append(
                {
                    "event_index": request.get("event_index"),
                    "reason": "fault_trace_request_out_of_range",
                }
            )
            continue
        related = [row for row in rows if int(row.get("request_id", 0)) == request_id]
        drops = [row for row in related if row.get("event_type") == "response_dropped"]
        scheduled = [
            row for row in related if row.get("event_type") == "chunk_scheduled"
        ]
        delivered = [
            row for row in related if row.get("event_type") == "chunk_delivered"
        ]
        arrived = [row for row in related if row.get("event_type") == "chunk_arrived"]
        if expected.dropped:
            if len(drops) != 1 or scheduled or delivered or arrived:
                errors.append(
                    {
                        "event_index": request.get("event_index"),
                        "reason": "fault_trace_drop_outcome_mismatch",
                    }
                )
            continue
        if len(scheduled) != 1 or drops:
            errors.append(
                {
                    "event_index": request.get("event_index"),
                    "reason": "fault_trace_schedule_outcome_mismatch",
                }
            )
            continue
        scheduled_detail = _merged(scheduled[0])
        if (
            int(scheduled_detail.get("request_ordinal", -1)) != ordinal
            or int(scheduled_detail.get("latency_ms", -1))
            != expected.total_delivery_delay_ms
            or scheduled_detail.get("trace_sha256") != active_trace.sha256
        ):
            errors.append(
                {
                    "event_index": scheduled[0].get("event_index"),
                    "reason": "fault_trace_schedule_detail_mismatch",
                }
            )
        originals = [
            row for row in delivered if not bool(_merged(row).get("duplicate"))
        ]
        duplicates = [row for row in delivered if bool(_merged(row).get("duplicate"))]
        original_arrivals = [
            row for row in arrived if not bool(_merged(row).get("duplicate"))
        ]
        duplicate_arrivals = [
            row for row in arrived if bool(_merged(row).get("duplicate"))
        ]
        if len(originals) > 1 or len(duplicates) > expected.duplicate_count:
            errors.append(
                {
                    "event_index": request.get("event_index"),
                    "reason": "fault_trace_delivery_count_mismatch",
                }
            )
        if (
            len(original_arrivals) > 1
            or len(duplicate_arrivals) > expected.duplicate_count
        ):
            errors.append(
                {
                    "event_index": request.get("event_index"),
                    "reason": "fault_trace_arrival_count_mismatch",
                }
            )
        for original in originals:
            detail = _merged(original)
            if (
                int(detail.get("latency_ms", -1)) != expected.total_delivery_delay_ms
                or int(detail.get("request_ordinal", -1)) != ordinal
                or detail.get("trace_sha256") != active_trace.sha256
            ):
                errors.append(
                    {
                        "event_index": original.get("event_index"),
                        "reason": "fault_trace_original_latency_mismatch",
                    }
                )
        for duplicate in duplicates:
            expected_duplicate_latency = (
                expected.total_delivery_delay_ms + expected.duplicate_delivery_offset_ms
            )
            detail = _merged(duplicate)
            if (
                int(detail.get("latency_ms", -1)) != expected_duplicate_latency
                or int(detail.get("request_ordinal", -1)) != ordinal
                or detail.get("trace_sha256") != active_trace.sha256
            ):
                errors.append(
                    {
                        "event_index": duplicate.get("event_index"),
                        "reason": "fault_trace_duplicate_offset_mismatch",
                    }
                )
        scheduled_ns = steady_time_ns(scheduled[0])
        if scheduled_ns is not None and episode_end_ns is not None:
            expected_counts = (
                (
                    "original",
                    1,
                    scheduled_ns + expected.total_delivery_delay_ms * 1_000_000,
                    originals,
                    original_arrivals,
                ),
                (
                    "duplicate",
                    expected.duplicate_count,
                    scheduled_ns
                    + (
                        expected.total_delivery_delay_ms
                        + expected.duplicate_delivery_offset_ms
                    )
                    * 1_000_000,
                    duplicates,
                    duplicate_arrivals,
                ),
            )
            for (
                delivery_kind,
                expected_count,
                due_ns,
                delivered_kind,
                arrived_kind,
            ) in expected_counts:
                if (
                    expected_count
                    and episode_end_ns >= due_ns
                    and (
                        len(delivered_kind) != expected_count
                        or len(arrived_kind) != expected_count
                    )
                ):
                    errors.append(
                        {
                            "event_index": request.get("event_index"),
                            "reason": "fault_trace_delivery_arrival_count_mismatch",
                            "delivery_kind": delivery_kind,
                            "expected": expected_count,
                            "delivered": len(delivered_kind),
                            "arrived": len(arrived_kind),
                        }
                    )

    arrivals = [row for row in rows if row.get("event_type") == "chunk_arrived"]
    arrival_request_ids = [int(row.get("request_id", 0)) for row in arrivals]
    for index, arrival in enumerate(arrivals):
        expected_out_of_order = arrival_request_ids[index] < max(
            arrival_request_ids[:index], default=0
        )
        if bool(_merged(arrival).get("out_of_order")) != expected_out_of_order:
            errors.append(
                {
                    "event_index": arrival.get("event_index"),
                    "reason": "out_of_order_attribution_mismatch",
                }
            )
    return errors


def write_summary_from_rows(
    rows: list[dict[str, Any]],
    summary_path: Path | str,
    *,
    profile_id: str,
    seed: int,
    strategy: str,
    episode_id: str,
    fairness: Mapping[str, Any],
    split: str = "frozen_holdout",
) -> dict[str, Any]:
    required_fairness = (
        HOLDOUT_FAIRNESS_FIELDS if split == "frozen_holdout" else FAIRNESS_FIELDS
    )
    missing = [name for name in required_fairness if name not in fairness]
    if missing:
        raise ValueError(f"summary fairness metadata missing: {missing}")
    summary = {
        "schema_version": M8_SCHEMA_VERSION,
        "milestone": M8_MILESTONE,
        "evidence_class": NATIVE_ISAAC_EVIDENCE_CLASS,
        "native_isaac_physics": True,
        "headline_eligible": split == "frozen_holdout",
        "profile_id": profile_id,
        "seed": int(seed),
        "strategy": strategy,
        "episode_id": episode_id,
        "split": split,
        **{name: fairness[name] for name in required_fairness},
        "metrics": recompute_metrics(rows),
    }
    write_json_atomic(summary_path, summary)
    return summary


def _jsonl_bytes(data: bytes) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"archive JSONL line {line_number} is not an object")
        rows.append(value)
    return rows


def _repository_root_for(path: Path) -> Path | None:
    resolved = path.resolve()
    return next(
        (
            candidate
            for candidate in (resolved.parent, *resolved.parents)
            if (candidate / ".git").exists()
        ),
        None,
    )


def _portable_path_identity(
    target: Path,
    *,
    manifest_path: Path,
    repository_root: Path | None = None,
) -> str:
    """Return a clone-stable repository- or manifest-relative artifact ID."""
    resolved = target.resolve()
    root = repository_root or _repository_root_for(manifest_path)
    if root is not None:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    try:
        return Path(
            os.path.relpath(resolved, manifest_path.resolve().parent)
        ).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"artifact cannot be represented by a portable path: {resolved}"
        ) from exc


@lru_cache(maxsize=8)
def _cached_archive_context(
    resolved_manifest_path: str,
    modified_time_ns: int,
    size_bytes: int,
) -> dict[str, Any] | None:
    del modified_time_ns, size_bytes
    return validate_archive_matrix_binding(Path(resolved_manifest_path))


def _load_episode_artifacts(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    archive_context: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    repository_root = _repository_root_for(manifest_path)
    if "archive_member" in entry:
        if archive_context is None:
            stat = manifest_path.stat()
            archive_context = _cached_archive_context(
                str(manifest_path.resolve()),
                stat.st_mtime_ns,
                stat.st_size,
            )
        if archive_context is None:
            raise ValueError(
                "archive-backed episode lacks validated matrix archive binding"
            )
        archive = Path(archive_context["archive_path"])
        rows = _jsonl_bytes(
            _archive_member_bytes(archive_context, str(entry["archive_member"]))
        )
        summary_member = str(entry.get("summary_archive_member", ""))
        if not summary_member:
            raise ValueError("archive-backed episode lacks summary_archive_member")
        summary = json.loads(
            _archive_member_bytes(archive_context, summary_member).decode("utf-8")
        )
        archive_id = _portable_path_identity(
            archive,
            manifest_path=manifest_path,
            repository_root=repository_root,
        )
        return rows, summary, f"{archive_id}!{entry['archive_member']}"
    event_path = Path(str(entry["event_log_path"]))
    summary_path = Path(str(entry["summary_path"]))
    if not event_path.is_absolute():
        event_path = manifest_path.parent / event_path
    if not summary_path.is_absolute():
        summary_path = manifest_path.parent / summary_path
    source = _portable_path_identity(
        event_path,
        manifest_path=manifest_path,
        repository_root=repository_root,
    )
    return read_jsonl(event_path), read_json(summary_path), source


def _archive_member_bytes(context: Mapping[str, Any], member_name: str) -> bytes:
    payloads = context.get("member_payloads", {})
    if member_name in payloads:
        return payloads[member_name]
    return read_archive_member(context["archive_path"], member_name)


def _archive_episode_stream(
    entries: Sequence[Mapping[str, Any]],
    context: dict[str, Any],
) -> Iterable[tuple[int, Mapping[str, Any]]]:
    event_to_entry: dict[str, tuple[int, Mapping[str, Any]]] = {}
    small_members: set[str] = set()
    for index, entry in enumerate(entries):
        event_member = str(entry["archive_member"])
        if event_member in event_to_entry:
            raise ValueError(f"duplicate episode event archive member: {event_member}")
        event_to_entry[event_member] = (index, entry)
        small_members.update(
            str(entry[field])
            for field in (
                "summary_archive_member",
                "fault_trace_archive_member",
                "scenario_archive_member",
            )
        )
    payloads = read_archive_members(context["archive_path"], small_members)
    context["member_payloads"] = payloads
    try:
        for member_name, data in iter_archive_members(
            context["archive_path"],
            event_to_entry,
        ):
            payloads[member_name] = data
            try:
                yield event_to_entry[member_name]
            finally:
                del payloads[member_name]
    finally:
        context.pop("member_payloads", None)


def _finite_reset_values(
    reset: Mapping[str, Any],
    field: str,
    *,
    length: int,
) -> tuple[float, ...]:
    value = reset.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"paired reset {field} is not a vector")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"paired reset {field} is not numeric") from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(
            f"paired reset {field} must contain exactly {length} finite values"
        )
    return result


def _quaternion_pair_metrics(
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[float, float]:
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    norm_error = max(abs(first_norm - 1.0), abs(second_norm - 1.0))
    if first_norm <= 0.0 or second_norm <= 0.0:
        return math.inf, norm_error
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    normalized_dot = min(1.0, max(0.0, abs(dot / (first_norm * second_norm))))
    return 2.0 * math.acos(normalized_dot), norm_error


def compare_paired_reset_states(
    resets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare native reset measurements using preregistered physical tolerances."""

    if len(resets) < 2:
        return {
            "passed": True,
            "comparison_count": 0,
            "canonical_digest_equal": True,
            "canonical_bin_boundary_crossed": False,
            "max_deltas": None,
            "errors": [],
        }
    tolerances = PAIRED_RESET_FAIRNESS_CONTRACT["pairwise_tolerances"]
    reference = resets[0]
    errors: list[dict[str, Any]] = []
    maxima = {
        "robot_joint_position_max_abs_rad": 0.0,
        "robot_joint_velocity_max_abs_rad_s": 0.0,
        "end_effector_position_axis_max_abs_m": 0.0,
        "end_effector_position_l2_m": 0.0,
        "object_position_axis_max_abs_m": 0.0,
        "object_position_l2_m": 0.0,
        "end_effector_quaternion_geodesic_rad": 0.0,
        "object_quaternion_geodesic_rad": 0.0,
        "quaternion_norm_abs_error": 0.0,
    }
    canonical_digests = [
        canonical_sha256(paired_reset_canonical_payload(reset)) for reset in resets
    ]
    exact_fields = (
        "seed",
        "zone_a_xyz",
        "zone_b_xyz",
        "physics_dt_seconds",
        "rendering_dt_seconds",
        "stage_units_in_meters",
        "gravity_xyz",
    )
    for comparison_index, candidate in enumerate(resets[1:], start=1):
        mismatched = [
            field
            for field in exact_fields
            if candidate.get(field) != reference.get(field)
        ]
        if mismatched:
            errors.append(
                {
                    "comparison_index": comparison_index,
                    "reason": "paired_reset_exact_field_mismatch",
                    "fields": mismatched,
                }
            )
        joint_position_delta = max(
            abs(a - b)
            for a, b in zip(
                _finite_reset_values(reference, "robot_joint_positions", length=9),
                _finite_reset_values(candidate, "robot_joint_positions", length=9),
                strict=True,
            )
        )
        joint_velocity_delta = max(
            abs(a - b)
            for a, b in zip(
                _finite_reset_values(reference, "robot_joint_velocities", length=9),
                _finite_reset_values(candidate, "robot_joint_velocities", length=9),
                strict=True,
            )
        )
        maxima["robot_joint_position_max_abs_rad"] = max(
            maxima["robot_joint_position_max_abs_rad"], joint_position_delta
        )
        maxima["robot_joint_velocity_max_abs_rad_s"] = max(
            maxima["robot_joint_velocity_max_abs_rad_s"], joint_velocity_delta
        )
        for field, prefix in (
            ("end_effector_position_xyz", "end_effector_position"),
            ("object_position_xyz", "object_position"),
        ):
            deltas = tuple(
                abs(a - b)
                for a, b in zip(
                    _finite_reset_values(reference, field, length=3),
                    _finite_reset_values(candidate, field, length=3),
                    strict=True,
                )
            )
            axis_delta = max(deltas)
            l2_delta = math.sqrt(sum(delta * delta for delta in deltas))
            maxima[f"{prefix}_axis_max_abs_m"] = max(
                maxima[f"{prefix}_axis_max_abs_m"], axis_delta
            )
            maxima[f"{prefix}_l2_m"] = max(maxima[f"{prefix}_l2_m"], l2_delta)
        for field, prefix in (
            ("end_effector_orientation_wxyz", "end_effector"),
            ("object_orientation_wxyz", "object"),
        ):
            angle, norm_error = _quaternion_pair_metrics(
                _finite_reset_values(reference, field, length=4),
                _finite_reset_values(candidate, field, length=4),
            )
            maxima[f"{prefix}_quaternion_geodesic_rad"] = max(
                maxima[f"{prefix}_quaternion_geodesic_rad"], angle
            )
            maxima["quaternion_norm_abs_error"] = max(
                maxima["quaternion_norm_abs_error"], norm_error
            )

    limits = {
        "robot_joint_position_max_abs_rad": float(
            tolerances["robot_joint_position_max_abs_rad"]
        ),
        "robot_joint_velocity_max_abs_rad_s": float(
            tolerances["robot_joint_velocity_max_abs_rad_s"]
        ),
        "end_effector_position_axis_max_abs_m": float(
            tolerances["position_axis_max_abs_m"]
        ),
        "end_effector_position_l2_m": float(tolerances["position_l2_m"]),
        "object_position_axis_max_abs_m": float(tolerances["position_axis_max_abs_m"]),
        "object_position_l2_m": float(tolerances["position_l2_m"]),
        "end_effector_quaternion_geodesic_rad": float(
            tolerances["quaternion_geodesic_rad"]
        ),
        "object_quaternion_geodesic_rad": float(tolerances["quaternion_geodesic_rad"]),
        "quaternion_norm_abs_error": float(tolerances["quaternion_norm_abs_error"]),
    }
    exceeded = {
        name: {"observed": maxima[name], "limit": limit}
        for name, limit in limits.items()
        if maxima[name] > limit
    }
    if exceeded:
        errors.append(
            {
                "reason": "paired_reset_numeric_tolerance_exceeded",
                "metrics": exceeded,
            }
        )
    canonical_equal = len(set(canonical_digests)) == 1
    return {
        "passed": not errors,
        "comparison_count": len(resets) - 1,
        "canonical_digest_equal": canonical_equal,
        "canonical_bin_boundary_crossed": not canonical_equal and not errors,
        "canonical_digests": canonical_digests,
        "max_deltas": maxima,
        "limits": limits,
        "errors": errors,
    }


def validate_manifest(
    manifest_path: Path | str,
    *,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = read_json(path)
    if manifest.get("milestone") != M8_MILESTONE:
        raise ValueError("manifest is not M8-G0")
    split = str(manifest.get("split", ""))
    if split not in {"baseline_gate", "development", "frozen_holdout"}:
        raise ValueError("M8 manifest has an invalid split")
    if split == "frozen_holdout" and manifest.get("headline_eligible") is not True:
        raise ValueError("frozen holdout manifest must be headline_eligible=true")
    development_filter = _development_strategy_filter(manifest, split=split)
    archive_binding = validate_archive_matrix_binding(path, manifest)
    archive_context = dict(archive_binding) if archive_binding is not None else None
    manifest_sha256 = sha256_file(path)
    required_fairness = (
        HOLDOUT_FAIRNESS_FIELDS if split == "frozen_holdout" else FAIRNESS_FIELDS
    )
    entries = list(manifest.get("episodes", []))
    indexed_audits: list[tuple[int, dict[str, Any]]] = []
    indexed_summaries: list[tuple[int, dict[str, Any]]] = []
    indexed_reset_states: list[tuple[int, dict[str, Any] | None]] = []
    indexed_trace_profiles: list[tuple[int, str, str, str, M8FaultProfile]] = []
    entry_stream: Iterable[tuple[int, Mapping[str, Any]]]
    if archive_context is None:
        entry_stream = enumerate(entries)
    else:
        entry_stream = _archive_episode_stream(entries, archive_context)
    for entry_index, entry in entry_stream:
        active_trace: M8FaultTrace | None = None
        rows, summary, source = _load_episode_artifacts(
            path,
            manifest,
            entry,
            archive_context=archive_context,
        )
        indexed_summaries.append((entry_index, summary))
        starts = [row for row in rows if row.get("event_type") == "episode_start"]
        reset_state = starts[0].get("reset_state") if len(starts) == 1 else None
        indexed_reset_states.append(
            (
                entry_index,
                dict(reset_state) if isinstance(reset_state, Mapping) else None,
            )
        )
        audit = validate_episode_rows(
            rows,
            summary,
            source=source,
            required_fairness_fields=required_fairness,
        )
        if archive_context is not None:
            trace_member = str(entry["fault_trace_archive_member"])
            trace_payload = json.loads(
                _archive_member_bytes(archive_context, trace_member).decode("utf-8")
            )
            active_trace = fault_trace_from_mapping(
                trace_payload,
                source=f"{source.rsplit('!', 1)[0]}!{trace_member}",
            )
            trace_errors = validate_fault_trace_rows(
                rows,
                trace=active_trace,
                summary=summary,
            )
            if entry.get("fault_trace_sha256") != summary.get("fault_trace_sha256"):
                trace_errors.append(
                    {
                        "event_index": None,
                        "reason": "matrix_episode_fault_trace_sha256_mismatch",
                    }
                )
        else:
            trace_value = entry.get("fault_trace_file")
            if not trace_value:
                trace_errors = [
                    {"event_index": None, "reason": "fault_trace_file_missing"}
                ]
            else:
                trace_path = Path(str(trace_value))
                if trace_path.is_absolute():
                    trace_errors = [
                        {"event_index": None, "reason": "fault_trace_path_not_portable"}
                    ]
                else:
                    trace_path = (path.parent / trace_path).resolve()
                    if not trace_path.is_file():
                        trace_errors = [
                            {
                                "event_index": None,
                                "reason": "fault_trace_file_not_found",
                            }
                        ]
                    else:
                        active_trace = load_fault_trace(trace_path)
                        trace_errors = validate_fault_trace_rows(
                            rows,
                            trace=active_trace,
                            summary=summary,
                        )
                        if entry.get("fault_trace_sha256") != summary.get(
                            "fault_trace_sha256"
                        ):
                            trace_errors.append(
                                {
                                    "event_index": None,
                                    "reason": "matrix_episode_fault_trace_sha256_mismatch",
                                }
                            )
        if archive_context is not None:
            scenario_member = str(entry["scenario_archive_member"])
            scenario_payload = json.loads(
                _archive_member_bytes(archive_context, scenario_member).decode("utf-8")
            )
            try:
                scenario = scenario_from_mapping(
                    scenario_payload,
                    source=f"{source.rsplit('!', 1)[0]}!{scenario_member}",
                )
            except (KeyError, TypeError, ValueError) as exc:
                trace_errors.append(
                    {
                        "event_index": None,
                        "reason": "scenario_file_invalid",
                        "detail": str(exc),
                    }
                )
            else:
                if (
                    scenario.sha256 != entry.get("scenario_sha256")
                    or scenario.sha256 != summary.get("scenario_sha256")
                    or scenario.seed != int(entry.get("seed", -1))
                ):
                    trace_errors.append(
                        {"event_index": None, "reason": "scenario_provenance_mismatch"}
                    )
        else:
            scenario_value = entry.get("scenario_file")
            if not scenario_value:
                trace_errors.append(
                    {"event_index": None, "reason": "scenario_file_missing"}
                )
            else:
                scenario_path = Path(str(scenario_value))
                if scenario_path.is_absolute():
                    trace_errors.append(
                        {"event_index": None, "reason": "scenario_path_not_portable"}
                    )
                else:
                    scenario_path = (path.parent / scenario_path).resolve()
                    if not scenario_path.is_file():
                        trace_errors.append(
                            {"event_index": None, "reason": "scenario_file_not_found"}
                        )
                    else:
                        try:
                            scenario = load_scenario(scenario_path)
                        except (KeyError, TypeError, ValueError) as exc:
                            trace_errors.append(
                                {
                                    "event_index": None,
                                    "reason": "scenario_file_invalid",
                                    "detail": str(exc),
                                }
                            )
                        else:
                            if (
                                scenario.sha256 != entry.get("scenario_sha256")
                                or scenario.sha256 != summary.get("scenario_sha256")
                                or scenario.seed != int(entry.get("seed", -1))
                            ):
                                trace_errors.append(
                                    {
                                        "event_index": None,
                                        "reason": "scenario_provenance_mismatch",
                                    }
                                )
        if trace_errors:
            audit["violations"].extend(trace_errors)
            counts = Counter(item["reason"] for item in audit["violations"])
            audit["invariant_violation_counts"] = dict(sorted(counts.items()))
            audit["passed"] = False
        audit["fault_trace_conformance_passed"] = not trace_errors
        indexed_audits.append((entry_index, audit))
        if active_trace is not None:
            indexed_trace_profiles.append(
                (
                    entry_index,
                    str(entry.get("episode_id", summary.get("episode_id", ""))),
                    str(entry.get("profile_id", summary.get("profile_id", ""))),
                    str(entry.get("strategy", summary.get("strategy", ""))),
                    active_trace.profile,
                )
            )
    audits = [audit for _, audit in sorted(indexed_audits)]
    summaries = [summary for _, summary in sorted(indexed_summaries)]
    reset_states = [reset for _, reset in sorted(indexed_reset_states)]
    if not audits:
        raise ValueError("manifest contains no episodes")

    repository_root = _repository_root_for(path)
    provenance_errors: list[str] = []
    if any(summary.get("split") != split for summary in summaries):
        provenance_errors.append("episode_summary_split_mismatch")
    if any(summary.get("native_isaac_physics") is not True for summary in summaries):
        provenance_errors.append("episode_summary_native_physics_missing")
    if any(
        summary.get("protocol_sha256") != manifest.get("protocol_sha256")
        for summary in summaries
    ):
        provenance_errors.append("episode_summary_protocol_sha256_mismatch")
    freeze_errors: list[str] = []
    freeze_audit: dict[str, Any] | None = None
    freeze_validation_passed = False
    candidate: Mapping[str, Any] | None = None
    if repository_root is None:
        provenance_errors.append("repository_root_not_found")
    elif split == "frozen_holdout":
        freeze_sha256 = manifest.get("freeze_sha256")
        freeze_manifest_value = manifest.get("freeze_manifest")
        if not freeze_manifest_value:
            freeze_errors.append("freeze_manifest_missing")
        else:
            freeze_path = Path(str(freeze_manifest_value))
            if not freeze_path.is_absolute():
                freeze_path = (path.parent / freeze_path).resolve()
            if not freeze_path.is_file():
                freeze_errors.append("freeze_manifest_not_found")
            else:
                freeze_audit = validate_freeze_manifest(
                    freeze_path,
                    repository_root=repository_root,
                )
                freeze_errors.extend(str(error) for error in freeze_audit["errors"])
                if freeze_audit["manifest"].get("freeze_sha256") != freeze_sha256:
                    freeze_errors.append("run_manifest_freeze_sha256_mismatch")
        if manifest.get("frozen_before_first_holdout_result") is not True:
            freeze_errors.append("frozen_before_holdout_declaration_missing")
        if manifest.get("holdout_freeze_status") != "frozen":
            freeze_errors.append("holdout_freeze_status_mismatch")
        if not isinstance(freeze_sha256, str) or len(freeze_sha256) != 64:
            freeze_errors.append("freeze_sha256_invalid")
        if any(summary.get("freeze_sha256") != freeze_sha256 for summary in summaries):
            freeze_errors.append("episode_freeze_sha256_mismatch")
        freeze_validation_passed = not freeze_errors
        provenance_errors.extend(freeze_errors)
    else:
        candidate_value = manifest.get("candidate_manifest")
        if manifest.get("frozen_before_first_holdout_result") is not False:
            provenance_errors.append("candidate_frozen_declaration_mismatch")
        if manifest.get("holdout_freeze_status") != "pending":
            provenance_errors.append("candidate_holdout_status_mismatch")
        if not candidate_value:
            provenance_errors.append("candidate_manifest_missing")
        else:
            candidate_path = Path(str(candidate_value))
            if not candidate_path.is_absolute():
                candidate_path = (path.parent / candidate_path).resolve()
            if not candidate_path.is_file():
                provenance_errors.append("candidate_manifest_not_found")
            else:
                candidate = load_protocol(candidate_path)
                from .schema import canonical_sha256

                if canonical_sha256(candidate) != manifest.get("protocol_sha256"):
                    provenance_errors.append("candidate_protocol_sha256_mismatch")
    provenance_validation_passed = not provenance_errors

    seed_errors: list[str] = []
    seed_file_value = manifest.get("seed_file")
    frozen_seeds: tuple[int, ...] = ()
    seed_path: Path | None = None
    if not seed_file_value:
        seed_errors.append("seed_file_missing")
    else:
        seed_path = Path(str(seed_file_value))
        if seed_path.is_absolute():
            seed_errors.append("seed_file_path_not_portable")
        if not seed_path.is_absolute():
            seed_path = (path.parent / seed_path).resolve()
        if not seed_path.is_file():
            seed_errors.append("seed_file_not_found")
        else:
            frozen_seeds = load_seed_file(seed_path)
            if sha256_file(seed_path) != manifest.get("seed_file_sha256"):
                seed_errors.append("seed_file_sha256_mismatch")
    if split == "baseline_gate" and len(frozen_seeds) < 20:
        seed_errors.append("baseline_seed_count_below_20")
    if split == "development" and not 1 <= len(frozen_seeds) <= 12:
        seed_errors.append("development_seed_count_outside_1_to_12")
    if split == "frozen_holdout" and len(frozen_seeds) < 40:
        seed_errors.append("holdout_seed_count_below_40")
    if split == "frozen_holdout" and freeze_audit is not None and seed_path is not None:
        try:
            relative_seed = (
                seed_path.resolve().relative_to(repository_root.resolve()).as_posix()
            )
        except ValueError:
            seed_errors.append("holdout_seed_file_outside_repository")
        else:
            frozen_input = next(
                (
                    record
                    for record in freeze_audit["manifest"].get("inputs", [])
                    if record.get("path") == relative_seed
                    and record.get("role") == "holdout_seeds"
                ),
                None,
            )
            if frozen_input is None or frozen_input.get("sha256") != manifest.get(
                "seed_file_sha256"
            ):
                seed_errors.append("holdout_seed_file_not_bound_by_freeze")

    profile_errors: list[str] = []
    declared_profiles: dict[str, M8FaultProfile] = {}
    profile_records = manifest.get("profiles")
    if not isinstance(profile_records, Mapping):
        profile_errors.append("profile_records_missing")
        profile_records = {}
    expected_profile_ids = (
        {"profile_0_sanity"}
        if split == "baseline_gate"
        else set(PROFILE_STRATEGIES)
        if split == "frozen_holdout"
        else set(str(key) for key in profile_records)
    )
    if set(str(key) for key in profile_records) != expected_profile_ids:
        profile_errors.append("profile_record_set_mismatch")
    for raw_profile_id, record in profile_records.items():
        profile_id = str(raw_profile_id)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or not record.get("path")
        ):
            profile_errors.append(f"profile_record_invalid:{profile_id}")
            continue
        profile_path = Path(str(record["path"]))
        if profile_path.is_absolute():
            profile_errors.append(f"profile_path_not_portable:{profile_id}")
        profile_path = (path.parent / profile_path).resolve()
        if not profile_path.is_file():
            profile_errors.append(f"profile_file_missing:{profile_id}")
            continue
        actual_hash = sha256_file(profile_path)
        if actual_hash != record.get("sha256"):
            profile_errors.append(f"profile_sha256_mismatch:{profile_id}")
        profile_payload = read_json(profile_path)
        expected_strategies = PROFILE_STRATEGIES.get(profile_id)
        if expected_strategies is None:
            profile_errors.append(f"profile_contract_mismatch:{profile_id}")
            continue
        try:
            declared_profiles[profile_id] = declared_fault_profile(
                profile_payload,
                expected_profile_id=profile_id,
                expected_strategies=expected_strategies,
            )
        except (KeyError, TypeError, ValueError) as exc:
            profile_errors.append(f"profile_contract_mismatch:{profile_id}")
            profile_errors.append(f"profile_contract_detail:{profile_id}:{exc}")
        if split != "frozen_holdout" and candidate is not None:
            candidate_profiles = candidate.get("profiles")
            candidate_record = (
                candidate_profiles.get(profile_id)
                if isinstance(candidate_profiles, Mapping)
                else None
            )
            if not isinstance(candidate_record, Mapping) or repository_root is None:
                profile_errors.append(f"profile_not_bound_by_candidate:{profile_id}")
            else:
                candidate_relative = Path(str(candidate_record.get("profile_file", "")))
                candidate_profile_path = (
                    repository_root.resolve() / candidate_relative
                ).resolve()
                if (
                    not candidate_record.get("profile_file")
                    or candidate_relative.is_absolute()
                    or ".." in candidate_relative.parts
                    or tuple(candidate_record.get("strategies", ()))
                    != expected_strategies
                    or candidate_profile_path != profile_path
                    or candidate_record.get("profile_sha256") != actual_hash
                ):
                    profile_errors.append(
                        f"profile_not_bound_by_candidate:{profile_id}"
                    )
        if split == "frozen_holdout" and freeze_audit is not None:
            try:
                relative_profile = profile_path.relative_to(
                    repository_root.resolve()
                ).as_posix()
            except ValueError:
                profile_errors.append(f"profile_outside_repository:{profile_id}")
            else:
                frozen_profile = next(
                    (
                        item
                        for item in freeze_audit["manifest"].get("inputs", [])
                        if item.get("role") == f"profile:{profile_id}"
                        and item.get("path") == relative_profile
                    ),
                    None,
                )
                if (
                    frozen_profile is None
                    or frozen_profile.get("sha256") != actual_hash
                ):
                    profile_errors.append(f"profile_not_bound_by_freeze:{profile_id}")

    audit_by_index = dict(indexed_audits)
    for (
        entry_index,
        episode_id,
        profile_id,
        strategy,
        trace_profile,
    ) in indexed_trace_profiles:
        binding_violations: list[dict[str, Any]] = []
        declared_profile = declared_profiles.get(profile_id)
        if declared_profile is None:
            binding_violations.append(
                {
                    "event_index": None,
                    "reason": "fault_trace_declared_profile_missing",
                }
            )
        else:
            mismatches = fault_profile_binding_mismatches(
                trace_profile,
                declared_profile,
            )
            if mismatches:
                binding_violations.append(
                    {
                        "event_index": None,
                        "reason": "fault_trace_profile_payload_mismatch",
                        "mismatches": mismatches,
                    }
                )
        expected_strategies = PROFILE_STRATEGIES.get(profile_id, ())
        if strategy not in expected_strategies:
            binding_violations.append(
                {
                    "event_index": None,
                    "reason": "fault_trace_strategy_not_declared",
                    "strategy": strategy,
                }
            )
        if not binding_violations:
            continue
        audit = audit_by_index[entry_index]
        audit["violations"].extend(binding_violations)
        counts = Counter(item["reason"] for item in audit["violations"])
        audit["invariant_violation_counts"] = dict(sorted(counts.items()))
        audit["fault_trace_conformance_passed"] = False
        audit["passed"] = False
        profile_errors.extend(
            f"{violation['reason']}:{episode_id}" for violation in binding_violations
        )

    if archive_context is not None:
        for entry in manifest.get("episodes", []):
            if any(field in entry for field in DIRECT_EPISODE_ARTIFACT_FIELDS):
                profile_errors.append("archive_episode_retains_loose_raw_reference")
            if any(field not in entry for field in ARCHIVE_EPISODE_ARTIFACT_FIELDS):
                profile_errors.append("archive_episode_member_reference_missing")
    elif split == "frozen_holdout":
        for entry in manifest.get("episodes", []):
            for field in (
                "fault_trace_file",
                "scenario_file",
                "event_log_path",
                "summary_path",
            ):
                if Path(str(entry.get(field, ""))).is_absolute():
                    profile_errors.append(f"episode_path_not_portable:{field}")
                    break

    by_pair: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_pair[(str(summary["profile_id"]), int(summary["seed"]))].append(summary)
    reset_by_episode_id = {
        str(summary.get("episode_id")): reset
        for summary, reset in zip(summaries, reset_states, strict=True)
    }
    fairness_errors: list[dict[str, Any]] = []
    fairness_notes: list[dict[str, Any]] = []
    paired_reset_audits: list[dict[str, Any]] = []
    paired_exact_fields = (
        *PAIRED_EXACT_EQUALITY_FIELDS,
        *(("freeze_sha256",) if split == "frozen_holdout" else ()),
    )
    for (profile_id, seed), pair in sorted(by_pair.items()):
        expected = _expected_profile_strategies(
            profile_id,
            split=split,
            development_filter=development_filter,
        )
        actual = {str(summary["strategy"]) for summary in pair}
        if len(pair) != len(actual):
            fairness_errors.append(
                {
                    "profile_id": profile_id,
                    "seed": seed,
                    "reason": "duplicate_strategy_episode",
                }
            )
        if actual != expected:
            fairness_errors.append(
                {
                    "profile_id": profile_id,
                    "seed": seed,
                    "reason": "method_set_mismatch",
                    "expected": sorted(expected),
                    "actual": sorted(actual),
                }
            )
        for field in paired_exact_fields:
            values = [summary.get(field) for summary in pair]
            if any(value is None for value in values) or any(
                value != values[0] for value in values[1:]
            ):
                fairness_errors.append(
                    {
                        "profile_id": profile_id,
                        "seed": seed,
                        "reason": "paired_field_mismatch",
                        "field": field,
                        "values": values,
                    }
                )
        pair_resets = [
            reset_by_episode_id.get(str(summary.get("episode_id"))) for summary in pair
        ]
        if any(reset is None for reset in pair_resets):
            reset_audit = {
                "profile_id": profile_id,
                "seed": seed,
                "passed": False,
                "reason": "paired_reset_state_missing",
            }
            fairness_errors.append(dict(reset_audit))
        else:
            try:
                reset_audit = {
                    "profile_id": profile_id,
                    "seed": seed,
                    **compare_paired_reset_states(
                        [reset for reset in pair_resets if reset is not None]
                    ),
                }
            except (KeyError, TypeError, ValueError) as exc:
                reset_audit = {
                    "profile_id": profile_id,
                    "seed": seed,
                    "passed": False,
                    "reason": "paired_reset_state_invalid",
                    "detail": str(exc),
                }
            if reset_audit.get("passed") is not True:
                fairness_errors.append(
                    {
                        "profile_id": profile_id,
                        "seed": seed,
                        "reason": "paired_reset_fairness_failed",
                        "errors": reset_audit.get("errors", [reset_audit]),
                    }
                )
            elif reset_audit.get("canonical_bin_boundary_crossed") is True:
                fairness_notes.append(
                    {
                        "profile_id": profile_id,
                        "seed": seed,
                        "reason": "canonical_bin_boundary_crossed",
                        "numeric_tolerances_passed": True,
                    }
                )
        paired_reset_audits.append(reset_audit)
        if any(
            summary.get("evidence_class") != NATIVE_ISAAC_EVIDENCE_CLASS
            for summary in pair
        ):
            fairness_errors.append(
                {
                    "profile_id": profile_id,
                    "seed": seed,
                    "reason": "non_native_isaac_pair",
                }
            )
    expected_blocks: dict[tuple[str, int], set[str]] = {}
    if split == "baseline_gate":
        expected_profiles = {"profile_0_sanity"}
    elif split == "frozen_holdout":
        expected_profiles = set(PROFILE_STRATEGIES)
    else:
        expected_profiles = {str(summary.get("profile_id")) for summary in summaries}
    for profile_id in expected_profiles:
        if profile_id not in PROFILE_STRATEGIES:
            fairness_errors.append(
                {"profile_id": profile_id, "seed": None, "reason": "unknown_profile"}
            )
            continue
        expected_strategies = _expected_profile_strategies(
            profile_id,
            split=split,
            development_filter=development_filter,
        )
        for seed in frozen_seeds:
            expected_blocks[(profile_id, seed)] = set(expected_strategies)
    actual_blocks = {
        block: {str(summary["strategy"]) for summary in pair}
        for block, pair in by_pair.items()
    }
    if set(actual_blocks) != set(expected_blocks):
        fairness_errors.append(
            {
                "profile_id": None,
                "seed": None,
                "reason": "paired_block_set_mismatch",
                "missing": [
                    list(block)
                    for block in sorted(set(expected_blocks) - set(actual_blocks))
                ],
                "unexpected": [
                    list(block)
                    for block in sorted(set(actual_blocks) - set(expected_blocks))
                ],
            }
        )
    for block in sorted(set(actual_blocks) & set(expected_blocks)):
        if actual_blocks[block] != expected_blocks[block]:
            fairness_errors.append(
                {
                    "profile_id": block[0],
                    "seed": block[1],
                    "reason": "exact_strategy_set_mismatch",
                    "expected": sorted(expected_blocks[block]),
                    "actual": sorted(actual_blocks[block]),
                }
            )
    expected_episode_count = sum(
        len(strategies) for strategies in expected_blocks.values()
    )
    if (
        len(summaries) != expected_episode_count
        or int(manifest.get("expected_episode_count", -1)) != expected_episode_count
    ):
        fairness_errors.append(
            {
                "profile_id": None,
                "seed": None,
                "reason": "exact_episode_count_mismatch",
                "expected": expected_episode_count,
                "summary_count": len(summaries),
                "manifest_expected": manifest.get("expected_episode_count"),
            }
        )
    result = {
        "schema_version": M8_SCHEMA_VERSION,
        "milestone": M8_MILESTONE,
        "manifest": _portable_path_identity(
            path,
            manifest_path=path,
            repository_root=repository_root,
        ),
        "manifest_sha256": manifest_sha256,
        "split": split,
        "episode_count": len(audits),
        "paired_block_count": len(by_pair),
        "episode_audits_passed": sum(audit["passed"] for audit in audits),
        "fairness_passed": not fairness_errors,
        "freeze_validation_passed": freeze_validation_passed,
        "freeze_errors": freeze_errors,
        "freeze_audit": freeze_audit,
        "provenance_validation_passed": provenance_validation_passed,
        "provenance_errors": provenance_errors,
        "seed_validation_passed": not seed_errors,
        "seed_errors": seed_errors,
        "seed_count": len(frozen_seeds),
        "seed_file_sha256": manifest.get("seed_file_sha256"),
        "profile_validation_passed": not profile_errors,
        "profile_errors": profile_errors,
        "fairness_errors": fairness_errors,
        "fairness_notes": fairness_notes,
        "paired_reset_fairness_contract": PAIRED_RESET_FAIRNESS_CONTRACT,
        "paired_reset_audits": paired_reset_audits,
        "audits": audits,
        "passed": (
            all(audit["passed"] for audit in audits)
            and not fairness_errors
            and provenance_validation_passed
            and not seed_errors
            and not profile_errors
        ),
    }
    if output_path is not None:
        write_json_atomic(output_path, result)
    return result
