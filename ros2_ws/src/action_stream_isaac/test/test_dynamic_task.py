from __future__ import annotations

import hashlib
import json

import pytest

from action_stream_isaac.dynamic_task import (
    DISTURBED_GENERATION_ID,
    DYNAMIC_TASK_STATE_SIZE,
    FAILURE_COLLISION,
    FAILURE_RECOVERY_TIMEOUT,
    FAILURE_RELEASED_AT_OBSOLETE,
    FAILURE_SWITCH_PRECONDITION_MISSED,
    PHASE_CLOSE,
    PHASE_DESCEND,
    PHASE_LIFT,
    PHASE_LOWER,
    PHASE_RELEASE,
    PHASE_RETRACT,
    PHASE_STABILIZE,
    PHASE_SUCCESS,
    PHASE_TRANSPORT,
    DynamicScenario,
    DynamicTaskConfig,
    DynamicTaskMachine,
    DynamicTaskMeasurement,
    pack_dynamic_task_state,
    scenario_for_seed,
    scenario_payload,
    scenario_sha256,
    unpack_dynamic_task_state,
)


def _measurement(
    *,
    step: int,
    object_xyz: tuple[float, float, float],
    eef_xyz: tuple[float, float, float] | None = None,
    grasped: bool = False,
    aperture_m: float = 0.08,
    collision: bool = False,
) -> DynamicTaskMeasurement:
    return DynamicTaskMeasurement(
        episode_step=step,
        end_effector_xyz=eef_xyz or (0.30, 0.0, 0.50),
        object_xyz=object_xyz,
        object_wxyz=(1.0, 0.0, 0.0, 0.0),
        object_linear_velocity_xyz=(0.0, 0.0, 0.0),
        object_angular_velocity_xyz=(0.0, 0.0, 0.0),
        gripper_aperture_m=aperture_m,
        physically_grasped=grasped,
        collision=collision,
    )


def _fast_contract() -> tuple[DynamicTaskConfig, DynamicScenario]:
    config = DynamicTaskConfig(switch_steps=(10,), max_steps=80)
    return config, scenario_for_seed(0, config)


def _drive_through_release(
    *, object_at_release: str
) -> tuple[DynamicTaskMachine, DynamicTaskConfig, DynamicScenario, int]:
    config, scenario = _fast_contract()
    machine = DynamicTaskMachine(scenario, config)
    object_xyz = scenario.object_xyz
    approach = (object_xyz[0], object_xyz[1], object_xyz[2] + config.approach_height_m)
    grasp = (object_xyz[0], object_xyz[1], object_xyz[2] + config.grasp_hand_offset_m)

    assert machine.update(_measurement(step=0, object_xyz=object_xyz, eef_xyz=approach)).phase_code == PHASE_DESCEND
    assert machine.update(_measurement(step=1, object_xyz=object_xyz, eef_xyz=grasp)).phase_code == PHASE_CLOSE
    for step in range(2, 8):
        evaluation = machine.update(
            _measurement(
                step=step,
                object_xyz=object_xyz,
                eef_xyz=grasp,
                grasped=True,
                aperture_m=0.0,
            )
        )
    assert evaluation.phase_code == PHASE_LIFT

    lifted_object = (object_xyz[0], object_xyz[1], object_xyz[2] + 0.11)
    evaluation = machine.update(
        _measurement(
            step=8,
            object_xyz=lifted_object,
            eef_xyz=(object_xyz[0], object_xyz[1], config.carry_hand_height_m),
            grasped=True,
            aperture_m=0.0,
        )
    )
    assert evaluation.phase_code == PHASE_TRANSPORT
    machine.update(
        _measurement(
            step=9,
            object_xyz=lifted_object,
            eef_xyz=(
                scenario.original_destination_xyz[0],
                scenario.original_destination_xyz[1],
                config.carry_hand_height_m,
            ),
            grasped=True,
            aperture_m=0.0,
        )
    )
    evaluation = machine.update(
        _measurement(
            step=10,
            object_xyz=lifted_object,
            eef_xyz=(
                scenario.final_destination_xyz[0],
                scenario.final_destination_xyz[1],
                config.carry_hand_height_m,
            ),
            grasped=True,
            aperture_m=0.0,
        )
    )
    assert evaluation.phase_code == PHASE_LOWER
    assert evaluation.destination_switched_now
    assert evaluation.switch_precondition_met
    assert evaluation.generation_id == DISTURBED_GENERATION_ID

    final_grasp = (
        scenario.final_destination_xyz[0],
        scenario.final_destination_xyz[1],
        scenario.final_destination_xyz[2] + config.grasp_hand_offset_m,
    )
    evaluation = machine.update(
        _measurement(
            step=11,
            object_xyz=lifted_object,
            eef_xyz=final_grasp,
            grasped=True,
            aperture_m=0.0,
        )
    )
    assert evaluation.phase_code == PHASE_RELEASE
    for step in range(12, 17):
        machine.update(
            _measurement(
                step=step,
                object_xyz=lifted_object,
                eef_xyz=final_grasp,
                grasped=True,
                aperture_m=0.0,
            )
        )

    release_xyz = (
        scenario.final_destination_xyz
        if object_at_release == "final"
        else scenario.original_destination_xyz
    )
    evaluation = machine.update(
        _measurement(
            step=17,
            object_xyz=release_xyz,
            eef_xyz=final_grasp,
            aperture_m=0.08,
        )
    )
    assert evaluation.phase_code == PHASE_RETRACT
    retract = (
        scenario.final_destination_xyz[0],
        scenario.final_destination_xyz[1],
        scenario.final_destination_xyz[2] + config.approach_height_m,
    )
    evaluation = machine.update(
        _measurement(
            step=18,
            object_xyz=release_xyz,
            eef_xyz=retract,
            aperture_m=0.08,
        )
    )
    assert evaluation.phase_code == PHASE_STABILIZE
    return machine, config, scenario, 19


def test_scenario_generation_and_hash_are_canonical() -> None:
    even = scenario_for_seed(42)
    odd = scenario_for_seed(43)
    assert scenario_for_seed(42) == even
    assert even.original_zone_label == "A" and even.final_zone_label == "B"
    assert odd.original_zone_label == "B" and odd.final_zone_label == "A"
    assert 0.42 <= even.object_xyz[0] <= 0.48
    assert -0.04 <= even.object_xyz[1] <= 0.04
    assert even.switch_step in {90, 100, 110}

    payload = scenario_payload(even)
    assert payload["object_position_xyz"] == list(even.object_xyz)
    assert payload["zone_a_xyz"] == list(even.zone_a_xyz)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert scenario_sha256(even) == hashlib.sha256(encoded).hexdigest()


def test_switch_precondition_miss_terminates_exactly_at_boundary() -> None:
    config, scenario = _fast_contract()
    machine = DynamicTaskMachine(scenario, config)
    evaluation = None
    for step in range(scenario.switch_step + 1):
        evaluation = machine.update(
            _measurement(step=step, object_xyz=scenario.object_xyz)
        )
        if step < scenario.switch_step:
            assert not evaluation.terminated
    assert evaluation is not None
    assert evaluation.destination_switched_now
    assert not evaluation.switch_precondition_met
    assert evaluation.generation_id == DISTURBED_GENERATION_ID
    assert evaluation.terminated
    assert evaluation.termination_reason == FAILURE_SWITCH_PRECONDITION_MISSED


def test_stable_twenty_step_final_zone_b_placement_succeeds() -> None:
    machine, config, scenario, first_stable_step = _drive_through_release(
        object_at_release="final"
    )
    assert scenario.final_zone_label == "B"
    for offset in range(config.stable_placement_steps):
        evaluation = machine.update(
            _measurement(
                step=first_stable_step + offset,
                object_xyz=scenario.final_destination_xyz,
                aperture_m=0.08,
            )
        )
        assert evaluation.success == (offset == config.stable_placement_steps - 1)
    assert evaluation.phase_code == PHASE_SUCCESS
    assert evaluation.terminated
    assert evaluation.success_streak_steps == config.stable_placement_steps


def test_stable_twenty_step_obsolete_zone_a_release_fails() -> None:
    machine, config, scenario, first_stable_step = _drive_through_release(
        object_at_release="obsolete"
    )
    assert scenario.original_zone_label == "A"
    for offset in range(config.stable_placement_steps):
        evaluation = machine.update(
            _measurement(
                step=first_stable_step + offset,
                object_xyz=scenario.original_destination_xyz,
                aperture_m=0.08,
            )
        )
    assert evaluation.terminated and not evaluation.success
    assert evaluation.termination_reason == FAILURE_RELEASED_AT_OBSOLETE


def test_collision_and_recovery_timeout_are_structured_failures() -> None:
    config, scenario = _fast_contract()
    collision_machine = DynamicTaskMachine(scenario, config)
    collision = collision_machine.update(
        _measurement(step=0, object_xyz=scenario.object_xyz, collision=True)
    )
    assert collision.terminated and collision.termination_reason == FAILURE_COLLISION

    timeout_machine = DynamicTaskMachine(scenario, config)
    timeout_machine.phase_code = PHASE_TRANSPORT
    timeout_machine.grasp_ever = True
    lifted = (
        scenario.object_xyz[0],
        scenario.object_xyz[1],
        scenario.object_xyz[2] + 0.11,
    )
    for step in range(config.max_steps + 1):
        timeout = timeout_machine.update(
            _measurement(
                step=step,
                object_xyz=lifted,
                eef_xyz=(0.30, 0.0, config.carry_hand_height_m),
                grasped=True,
                aperture_m=0.0,
            )
        )
        if timeout.terminated:
            break
    assert timeout.termination_reason == FAILURE_RECOVERY_TIMEOUT
    assert step == config.max_steps


def test_task_state_round_trip_and_binary_validation() -> None:
    scenario = scenario_for_seed(2)
    machine = DynamicTaskMachine(scenario)
    measurement = _measurement(step=0, object_xyz=scenario.object_xyz)
    evaluation = machine.update(measurement)
    packed = pack_dynamic_task_state(
        scenario=scenario, measurement=measurement, evaluation=evaluation
    )
    assert len(packed) == DYNAMIC_TASK_STATE_SIZE
    unpacked = unpack_dynamic_task_state(packed)
    assert unpacked["generation_id"] == 1
    assert unpacked["disturbance_switched"] is False

    for index in (22, 23, 27, 34, 35, 36, 37, 38, 39, 40):
        malformed = list(packed)
        malformed[index] = 0.5
        with pytest.raises(ValueError, match="exactly 0 or 1"):
            unpack_dynamic_task_state(malformed)
    malformed = list(packed)
    malformed[26] = 0.5
    with pytest.raises(ValueError, match="non-negative integer"):
        unpack_dynamic_task_state(malformed)

    invalid_measurement = _measurement(step=1, object_xyz=scenario.object_xyz)
    object.__setattr__(invalid_measurement, "physically_grasped", 1)
    with pytest.raises(ValueError, match="exactly bool"):
        invalid_measurement.validated()
