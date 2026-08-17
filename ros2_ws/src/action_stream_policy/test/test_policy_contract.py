import inspect

from action_stream_policy import policy_node
from action_stream_policy.scripted_policy import (
    CHUNK_HORIZON,
    TASK_ID,
    PolicyObservation,
    PolicyRequestData,
    ReachLiftScriptedPolicy,
)


def test_scripted_policy_node_factory_is_ros_import_safe() -> None:
    parameter = inspect.signature(policy_node.create_scripted_policy_node).parameters[
        "parameter_overrides"
    ]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert "rclpy" not in policy_node.__dict__


def test_policy_labels_every_action_o_plus_one_through_o_plus_h() -> None:
    observation = PolicyObservation(
        episode_id="package-test",
        observation_step=10,
        task_id=TASK_ID,
        robot_state=(0.27, 0.0, 0.22, 0.0, 0.0, 0.0, 1.0),
        task_state=(
            0.45,
            0.0,
            0.04,
            0.45,
            0.0,
            0.19,
            0.0,
            0.04,
            0.0,
            0.0,
            0.0,
            10.0,
        ),
    )
    chunk = ReachLiftScriptedPolicy().predict(
        PolicyRequestData(
            episode_id=observation.episode_id,
            request_id=1,
            generation_id=0,
            source_observation_step=observation.observation_step,
            expected_horizon=CHUNK_HORIZON,
            observation=observation,
        )
    )
    assert [item.target_step for item in chunk.actions] == list(range(11, 41))
