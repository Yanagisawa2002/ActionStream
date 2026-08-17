"""Launch one ROS/C++ deterministic test-plant benchmark episode."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    strategy = LaunchConfiguration("strategy")
    seed = LaunchConfiguration("seed")
    episode_id = LaunchConfiguration("episode_id")
    trace_file = LaunchConfiguration("trace_file")
    output_path = LaunchConfiguration("output_path")
    profile_id = LaunchConfiguration("profile_id")
    trace_sha256 = LaunchConfiguration("trace_sha256")
    max_steps = LaunchConfiguration("max_steps")
    request_interval_steps = LaunchConfiguration("request_interval_steps")
    common_topics = {
        "observation_topic": "/action_stream/observation",
        "inference_request_topic": "/action_stream/inference_request",
        "action_chunk_topic": "/action_stream/action_chunk",
        "robot_command_topic": "/action_stream/robot_command",
        "diagnostics_topic": "/action_stream/diagnostics",
        "runtime_event_topic": "/action_stream/events",
        "episode_control_topic": "/action_stream/episode_control",
    }
    executor = Node(
        package="action_stream_executor",
        executable="action_stream_executor_node",
        name="action_stream_executor",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "strategy": strategy,
                "action_dimension": 7,
                "safe_hold_command": [0.27, 0.0, 0.22, 0.0, 0.0, 0.0, 1.0],
                **common_topics,
            }
        ],
    )
    policy = Node(
        package="action_stream_policy",
        executable="scripted_policy_node",
        name="action_stream_scripted_policy",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "request_topic": "/action_stream/inference_request",
                "raw_chunk_topic": "/action_stream/raw_action_chunk",
                "event_topic": "/action_stream/events",
            }
        ],
    )
    injector = Node(
        package="action_stream_benchmark",
        executable="fault-injector-node",
        name="action_stream_fault_injector",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "trace_file": trace_file,
                "raw_chunk_topic": "/action_stream/raw_action_chunk",
                "action_chunk_topic": "/action_stream/action_chunk",
                "event_topic": "/action_stream/events",
            }
        ],
    )
    recorder = Node(
        package="action_stream_benchmark",
        executable="event-recorder-node",
        name="action_stream_event_recorder",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "output_path": output_path,
                "strategy": strategy,
                "profile_id": profile_id,
                "seed": seed,
                "trace_sha256": trace_sha256,
            }
        ],
    )
    plant = Node(
        package="action_stream_benchmark",
        executable="test-plant-node",
        name="action_stream_test_plant",
        output="screen",
        parameters=[
            {
                "strategy": strategy,
                "seed": seed,
                "episode_id": episode_id,
                "max_steps": max_steps,
                "request_interval_steps": request_interval_steps,
            }
        ],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("strategy", default_value="aligned_async"),
            DeclareLaunchArgument("seed", default_value="2026080300"),
            DeclareLaunchArgument("episode_id", default_value="m7-ros-test-plant"),
            DeclareLaunchArgument("profile_id", default_value="profile_a"),
            DeclareLaunchArgument("trace_file"),
            DeclareLaunchArgument("trace_sha256"),
            DeclareLaunchArgument("output_path"),
            DeclareLaunchArgument("max_steps", default_value="180"),
            DeclareLaunchArgument("request_interval_steps", default_value="10"),
            executor,
            policy,
            injector,
            recorder,
            plant,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=plant,
                    on_exit=[EmitEvent(event=Shutdown(reason="test plant episode complete"))],
                )
            ),
        ]
    )
