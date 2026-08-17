from __future__ import annotations

from dataclasses import replace
import math

import pytest

from action_stream_isaac.dynamic_task import (
    DISTURBED_GENERATION_ID,
    INITIAL_GENERATION_ID,
    PHASE_NAMES,
    DynamicTaskMachine,
    DynamicTaskMeasurement,
    pack_dynamic_task_state,
    scenario_for_seed,
)

from action_stream_policy.dynamic_pick_place import (
    CHUNK_HORIZON,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    PHASE_APPROACH_ABOVE,
    PHASE_CLOSE,
    PHASE_TRANSPORT,
    DynamicPickPlacePolicy,
    DynamicPolicyObservation,
    DynamicPolicyRequestData,
)


def _task_state(
    *,
    phase: int = PHASE_APPROACH_ABOVE,
    switched: bool = False,
    grasped: bool = False,
    object_xyz: tuple[float, float, float] = (0.45, 0.0, 0.02575),
) -> tuple[float, ...]:
    scenario = scenario_for_seed(0)
    bootstrap_measurement = DynamicTaskMeasurement(
        episode_step=0,
        end_effector_xyz=(0.40, 0.0, 0.30),
        object_xyz=scenario.object_xyz,
        object_wxyz=scenario.object_wxyz,
        object_linear_velocity_xyz=(0.0, 0.0, 0.0),
        object_angular_velocity_xyz=(0.0, 0.0, 0.0),
        gripper_aperture_m=0.08,
        physically_grasped=False,
    )
    base = DynamicTaskMachine(scenario).update(bootstrap_measurement)
    active = (
        scenario.final_destination_xyz if switched else scenario.original_destination_xyz
    )
    measurement = replace(
        bootstrap_measurement,
        episode_step=10,
        object_xyz=object_xyz,
        gripper_aperture_m=0.0 if grasped else 0.08,
        physically_grasped=grasped,
    )
    evaluation = replace(
        base,
        phase_code=phase,
        phase=PHASE_NAMES[phase],
        disturbance_switched=switched,
        generation_id=(DISTURBED_GENERATION_ID if switched else INITIAL_GENERATION_ID),
        active_destination_id=int(switched),
        active_destination_xyz=active,
        lift_height_m=object_xyz[2] - scenario.object_xyz[2],
        final_destination_xy_error_m=math.hypot(
            object_xyz[0] - scenario.final_destination_xyz[0],
            object_xyz[1] - scenario.final_destination_xyz[1],
        ),
        obsolete_destination_xy_error_m=math.hypot(
            object_xyz[0] - scenario.original_destination_xyz[0],
            object_xyz[1] - scenario.original_destination_xyz[1],
        ),
        grasped=grasped,
        grasp_ever=grasped,
    )
    return pack_dynamic_task_state(
        scenario=scenario, measurement=measurement, evaluation=evaluation
    )


def _request(
    *,
    phase: int = PHASE_APPROACH_ABOVE,
    switched: bool = False,
    grasped: bool = False,
    object_xyz: tuple[float, float, float] = (0.45, 0.0, 0.02575),
    robot_xyz: tuple[float, float, float] = (0.40, 0.0, 0.30),
) -> DynamicPolicyRequestData:
    generation = 2 if switched else 1
    observation = DynamicPolicyObservation(
        episode_id="episode-1",
        observation_step=10,
        generation_id=generation,
        task_id="dynamic_target_pick_place_v1",
        robot_state=(*robot_xyz, math.pi, 0.0, 0.0, GRIPPER_OPEN),
        task_state=_task_state(
            phase=phase,
            switched=switched,
            grasped=grasped,
            object_xyz=object_xyz,
        ),
    )
    return DynamicPolicyRequestData(
        episode_id=observation.episode_id,
        request_id=1,
        generation_id=generation,
        source_observation_step=observation.observation_step,
        expected_horizon=CHUNK_HORIZON,
        observation=observation,
    )


def test_chunk_is_finite_workspace_bounded_smooth_and_exactly_o_plus_one_to_h() -> None:
    policy = DynamicPickPlacePolicy()
    request = _request(
        phase=PHASE_TRANSPORT,
        switched=True,
        grasped=True,
        robot_xyz=(0.40, -0.05, 0.30),
    )
    chunk = policy.predict(request)
    assert [action.target_step for action in chunk.actions] == list(range(11, 41))
    previous = request.observation.robot_state[0:3]
    for action in chunk.actions:
        assert all(math.isfinite(value) for value in action.command)
        xyz = action.command[0:3]
        assert 0.25 <= xyz[0] <= 0.70
        assert -0.35 <= xyz[1] <= 0.35
        assert 0.08 <= xyz[2] <= 0.60
        assert math.dist(previous, xyz) <= 0.010000000001
        assert action.command[3:6] == (math.pi, 0.0, 0.0)
        assert action.command[6] == GRIPPER_CLOSED
        previous = xyz


def test_policy_is_sensitive_to_destination_switch_and_object_displacement() -> None:
    policy = DynamicPickPlacePolicy()
    original = policy.predict(
        _request(phase=PHASE_TRANSPORT, switched=False, grasped=True)
    )
    final = policy.predict(
        _request(phase=PHASE_TRANSPORT, switched=True, grasped=True)
    )
    assert original.actions[-1].command[1] < 0.0
    assert final.actions[-1].command[1] > 0.0
    assert original.actions != final.actions

    left = policy.predict(
        _request(phase=PHASE_APPROACH_ABOVE, object_xyz=(0.44, -0.03, 0.02575))
    )
    right = policy.predict(
        _request(phase=PHASE_APPROACH_ABOVE, object_xyz=(0.44, 0.03, 0.02575))
    )
    assert left.actions != right.actions
    assert left.actions[-1].command[1] < right.actions[-1].command[1]


def test_phase_and_physical_grasp_change_goal_and_gripper() -> None:
    policy = DynamicPickPlacePolicy()
    approach = policy.predict(_request(phase=PHASE_APPROACH_ABOVE))
    close = policy.predict(_request(phase=PHASE_CLOSE))
    assert approach.actions[0].command[6] == GRIPPER_OPEN
    assert close.actions[0].command[6] == GRIPPER_CLOSED
    assert approach.actions != close.actions

    recovered = policy.predict(
        _request(phase=PHASE_TRANSPORT, switched=True, grasped=False)
    )
    carrying = policy.predict(
        _request(phase=PHASE_TRANSPORT, switched=True, grasped=True)
    )
    assert recovered.actions != carrying.actions
    assert recovered.actions[-1].command[1] < carrying.actions[-1].command[1]


@pytest.mark.parametrize(
    ("object_xyz", "robot_xyz"),
    (
        ((0.80, 0.0, 0.02575), (0.40, 0.0, 0.30)),
        ((0.45, 0.0, 0.02575), (0.10, 0.0, 0.30)),
    ),
)
def test_policy_refuses_out_of_workspace_goal_or_start(
    object_xyz: tuple[float, float, float],
    robot_xyz: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError, match="outside workspace"):
        DynamicPickPlacePolicy().predict(
            _request(object_xyz=object_xyz, robot_xyz=robot_xyz)
        )


def test_policy_rejects_generation_and_binary_state_drift() -> None:
    state = list(_task_state())
    state[36] = 0.5
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        DynamicPolicyObservation(
            episode_id="episode-1",
            observation_step=10,
            generation_id=1,
            task_id="dynamic_target_pick_place_v1",
            robot_state=(0.4, 0.0, 0.3, math.pi, 0.0, 0.0, 1.0),
            task_state=tuple(state),
        )

    with pytest.raises(ValueError, match="generation"):
        DynamicPolicyObservation(
            episode_id="episode-1",
            observation_step=10,
            generation_id=2,
            task_id="dynamic_target_pick_place_v1",
            robot_state=(0.4, 0.0, 0.3, math.pi, 0.0, 0.0, 1.0),
            task_state=_task_state(),
        )
