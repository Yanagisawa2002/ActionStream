from action_stream_isaac.task_logic import (
    TaskEvaluation,
    deterministic_spawn_position,
    pack_task_state,
    split_sim_time,
)


def test_isaac_public_layout_and_time_conversion_are_frozen() -> None:
    evaluation = TaskEvaluation(
        phase_code=0,
        phase="reach",
        lift_height_m=0.0,
        end_effector_object_distance_m=0.2,
        success_streak_steps=0,
        success=False,
        terminated=False,
        termination_reason="",
    )
    task_state = pack_task_state(
        object_xyz=(0.45, 0.0, 0.04),
        lift_target_xyz=(0.45, 0.0, 0.19),
        grasped=False,
        initial_object_z=0.04,
        evaluation=evaluation,
        episode_step=0,
    )
    assert len(task_state) == 12
    assert deterministic_spawn_position("paired-episode") == deterministic_spawn_position(
        "paired-episode"
    )
    assert split_sim_time(1.9999999996) == (2, 0)
