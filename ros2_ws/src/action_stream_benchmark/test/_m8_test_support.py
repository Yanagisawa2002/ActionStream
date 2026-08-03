from __future__ import annotations

from copy import deepcopy
from typing import Any

from action_stream_benchmark.m8_protocol import (
    NATIVE_ISAAC_EVIDENCE_CLASS,
    paired_reset_canonical_payload,
)
from action_stream_benchmark.m8_replay import recompute_metrics
from action_stream_benchmark.schema import canonical_sha256


OLD = (0.48, -0.22, 0.02575)
NEW = (0.48, 0.22, 0.02575)
START = (0.45, 0.0, 0.02575)


def scenario_payload(seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "milestone": "M8-G0",
        "task_id": "dynamic_target_pick_place_v1",
        "disturbance_type": "destination_switch",
        "seed": seed,
        "object_position_xyz": list(START),
        "object_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "zone_a_xyz": list(OLD),
        "zone_b_xyz": list(NEW),
        "original_destination_xyz": list(OLD),
        "final_destination_xyz": list(NEW),
        "original_zone_label": "A",
        "final_zone_label": "B",
        "original_destination_id": 0,
        "final_destination_id": 1,
        "switch_step": 100,
        "initial_generation_id": 1,
        "disturbed_generation_id": 2,
    }


def reset_state_payload(seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "robot_joint_positions": [0.0] * 9,
        "robot_joint_velocities": [0.0] * 9,
        "end_effector_position_xyz": [0.307015, 0.0, 0.589907],
        "end_effector_orientation_wxyz": [0.0, 1.0, 0.0, 0.0],
        "object_position_xyz": list(START),
        "object_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "zone_a_xyz": list(OLD),
        "zone_b_xyz": list(NEW),
        "physics_dt_seconds": 0.05,
        "rendering_dt_seconds": 0.05,
        "stage_units_in_meters": 1.0,
        "gravity_xyz": [0.0, 0.0, -9.81],
    }


def fairness(*, holdout: bool = True, seed: int = 2026081200) -> dict[str, Any]:
    scenario = scenario_payload(seed)
    reset_state = reset_state_payload(seed)
    result = {
        "fault_trace_sha256": "1" * 64,
        "scenario_sha256": canonical_sha256(scenario),
        "reset_state_sha256": canonical_sha256(reset_state),
        "paired_reset_canonical_sha256": canonical_sha256(
            paired_reset_canonical_payload(reset_state)
        ),
        "switch_step": 100,
        "object_position_xyz": list(START),
        "original_destination_xyz": list(OLD),
        "final_destination_xyz": list(NEW),
        "controller_sha256": "4" * 64,
        "protocol_sha256": "5" * 64,
        "task_contract_sha256": "6" * 64,
        "safety_limits_sha256": "7" * 64,
    }
    if holdout:
        result["freeze_sha256"] = "8" * 64
    return result


def task_state(
    step: int,
    *,
    object_xyz: tuple[float, float, float],
    switched: bool,
    phase: int,
    grasped: bool,
    grasp_ever: bool,
    aperture: float,
    success: bool = False,
    terminated: bool = False,
    streak: int = 0,
) -> list[float]:
    active = NEW if switched else OLD
    final_error = ((object_xyz[0] - NEW[0]) ** 2 + (object_xyz[1] - NEW[1]) ** 2) ** 0.5
    old_error = ((object_xyz[0] - OLD[0]) ** 2 + (object_xyz[1] - OLD[1]) ** 2) ** 0.5
    correct = aperture >= 0.07 and final_error <= 0.04 and abs(object_xyz[2] - NEW[2]) <= 0.035
    obsolete = aperture >= 0.07 and old_error <= 0.04 and abs(object_xyz[2] - OLD[2]) <= 0.035
    return [
        *object_xyz,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        *OLD,
        *NEW,
        *active,
        float(switched),
        float(switched),
        2.0 if switched else 1.0,
        float(phase),
        0.0,
        float(grasped),
        float(aperture),
        START[2],
        object_xyz[2] - START[2],
        final_error,
        old_error,
        float(streak),
        float(correct),
        float(obsolete),
        0.0,
        0.0,
        float(grasp_ever),
        float(success),
        float(terminated),
        float(step),
        100.0,
        0.0,
    ]


def make_episode(
    *,
    strategy: str = "aligned_async",
    profile_id: str = "profile_1_fixed",
    seed: int = 2026081200,
    episode_id: str | None = None,
    split: str = "frozen_holdout",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episode = episode_id or f"episode-{profile_id}-{seed}-{strategy}"
    shared_fairness = fairness(holdout=split == "frozen_holdout", seed=seed)
    rows: list[dict[str, Any]] = []

    def add(event_type: str, **payload: Any) -> None:
        rows.append(
            {
                "schema_version": 1,
                "milestone": "M8-G0",
                "evidence_class": NATIVE_ISAAC_EVIDENCE_CLASS,
                "event_type": event_type,
                "episode_id": episode,
                "profile_id": profile_id,
                "seed": seed,
                "strategy": strategy,
                "split": split,
                **payload,
            }
        )

    add(
        "episode_start",
        generation_id=1,
        maximum_episode_steps=320,
        placement_tolerance_m=0.04,
        placement_height_tolerance_m=0.035,
        stable_placement_steps=20,
        stable_linear_speed_mps=0.03,
        stable_angular_speed_rps=0.5,
        lift_clearance_m=0.10,
        gripper_open_aperture_m=0.07,
        wall_time_ns=0,
        sim_time_ns=0,
        scenario=scenario_payload(seed),
        reset_state=reset_state_payload(seed),
        **shared_fairness,
    )
    add(
        "observation",
        observation_step=0,
        sim_step=0,
        task_state=task_state(
            0,
            object_xyz=START,
            switched=False,
            phase=0,
            grasped=False,
            grasp_ever=False,
            aperture=0.07,
        ),
    )
    old_command = [*OLD, 3.141592653589793, 0.0, 0.0, -1.0]
    add("inference_request", request_id=1, generation_id=1, source_observation_step=0)
    add(
        "chunk_arrived",
        request_id=1,
        generation_id=1,
        source_observation_step=0,
        latency_ms=0.0,
        duplicate=False,
        out_of_order=False,
        actions=[{"target_step": 1, "command": old_command}],
    )
    add(
        "queue_updated",
        reason="arrival_order_append" if strategy == "naive_async" else (
            "sync_valid_rebuild" if strategy == "sync_hold" else "aligned_atomic_rebuild"
        ),
        request_id=1,
        generation_id=1,
        source_observation_step=0,
        queue_length_before=0,
        queue_length_after=1,
        action_count=1,
        expired_actions_removed=0,
        duplicate_target_actions_removed=0,
    )
    add(
        "action_executed",
        reason="arrival_order_execution" if strategy == "naive_async" else "target_step_match",
        request_id=1,
        generation_id=1,
        source_observation_step=0,
        source_target_step=1,
        actual_target_step=1,
        queue_length_before=1,
        queue_length_after=0,
        action_count=1,
    )
    add(
        "command_executed",
        actual_target_step=1,
        source_request_id=1,
        source_generation_id=1,
        source_observation_step=0,
        source_target_step=1,
        command=old_command,
        hold=False,
    )
    add(
        "destination_switched",
        step=100,
        switch_step=100,
        old_destination_xyz=list(OLD),
        new_destination_xyz=list(NEW),
        generation_before=1,
        generation_after=2,
    )
    add(
        "generation_advanced",
        generation_id=2,
        source_observation_step=100,
        queue_length_before=0,
        queue_length_after=0,
        action_count=0,
    )
    add(
        "observation",
        observation_step=100,
        sim_step=100,
        task_state=task_state(
            100,
            object_xyz=(START[0], START[1], 0.15),
            switched=True,
            phase=4,
            grasped=True,
            grasp_ever=True,
            aperture=0.02,
        ),
    )
    new_command = [*NEW, 3.141592653589793, 0.0, 0.0, 1.0]
    add("inference_request", request_id=2, generation_id=2, source_observation_step=100)
    add(
        "chunk_arrived",
        request_id=2,
        generation_id=2,
        source_observation_step=100,
        latency_ms=0.0,
        duplicate=False,
        out_of_order=False,
        actions=[{"target_step": 101, "command": new_command}],
    )
    add(
        "queue_updated",
        reason="arrival_order_append" if strategy == "naive_async" else (
            "sync_valid_rebuild" if strategy == "sync_hold" else "aligned_atomic_rebuild"
        ),
        request_id=2,
        generation_id=2,
        source_observation_step=100,
        queue_length_before=0,
        queue_length_after=1,
        action_count=1,
        expired_actions_removed=0,
        duplicate_target_actions_removed=0,
    )
    add(
        "action_executed",
        reason="arrival_order_execution" if strategy == "naive_async" else "target_step_match",
        request_id=2,
        generation_id=2,
        source_observation_step=100,
        source_target_step=101,
        actual_target_step=101,
        queue_length_before=1,
        queue_length_after=0,
        action_count=1,
    )
    add(
        "command_executed",
        actual_target_step=101,
        source_request_id=2,
        source_generation_id=2,
        source_observation_step=100,
        source_target_step=101,
        command=new_command,
        hold=False,
        obsolete_destination_command=False,
    )
    for index, step in enumerate(range(101, 121), start=1):
        final = step == 120
        add(
            "observation",
            observation_step=step,
            sim_step=step,
            task_state=task_state(
                step,
                object_xyz=NEW,
                switched=True,
                phase=9 if final else 8,
                grasped=False,
                grasp_ever=True,
                aperture=0.07,
                success=final,
                terminated=final,
                streak=index,
            ),
        )
    add("task_terminated", reason="correct_destination_stable_placement", success=True)
    add("episode_terminated", reason="success", queue_length_before=0, queue_length_after=0)
    add(
        "episode_end",
        success=True,
        completion_reason="correct_destination_stable_placement",
        completion_steps=120,
        wall_time_ns=10_000_000_000,
        sim_time_ns=6_000_000_000,
    )
    for index, row in enumerate(rows):
        row["event_index"] = index
    summary = {
        "schema_version": 1,
        "milestone": "M8-G0",
        "evidence_class": NATIVE_ISAAC_EVIDENCE_CLASS,
        "native_isaac_physics": True,
        "headline_eligible": split == "frozen_holdout",
        "split": split,
        "profile_id": profile_id,
        "seed": seed,
        "strategy": strategy,
        "episode_id": episode,
        **shared_fairness,
        "metrics": recompute_metrics(rows),
    }
    return rows, summary


def cloned_episode(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = make_episode(**kwargs)
    return deepcopy(rows), deepcopy(summary)
