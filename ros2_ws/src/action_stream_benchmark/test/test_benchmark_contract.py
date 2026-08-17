import inspect

from action_stream_benchmark import event_recorder_node, fault_injector_node
from action_stream_benchmark.faults import generate_fault_trace, resolve_profile
from action_stream_benchmark.plant import ReachLiftPlant


def test_embedded_node_factories_are_ros_import_safe() -> None:
    for factory in (
        fault_injector_node.create_fault_injector_node,
        event_recorder_node.create_event_recorder_node,
    ):
        parameter = inspect.signature(factory).parameters["parameter_overrides"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert "rclpy" not in fault_injector_node.__dict__
    assert "rclpy" not in event_recorder_node.__dict__


def test_fault_trace_and_test_plant_are_seed_deterministic() -> None:
    profile = resolve_profile("profile_b")
    first = generate_fault_trace(profile, seed=2026080300, request_count=32)
    second = generate_fault_trace(profile, seed=2026080300, request_count=32)
    assert first.sha256 == second.sha256

    first_observation = ReachLiftPlant().reset("first", seed=2026080300)
    second_observation = ReachLiftPlant().reset("second", seed=2026080300)
    assert first_observation.robot_state == second_observation.robot_state
    assert first_observation.task_state == second_observation.task_state
