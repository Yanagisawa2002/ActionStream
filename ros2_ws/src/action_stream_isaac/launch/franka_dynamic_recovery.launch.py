"""Structural M8 native launch: one router, C++ executor, and Isaac adapter.

The adapter owns exactly one in-process dynamic policy, fault injector,
request driver, and M8 recorder per episode.  They must not also be launched
as standalone nodes.  On the audited Windows deployment the PowerShell runner
starts these same processes directly because pre-Kit ``ros2 launch`` imports
are not supported by the Isaac environment.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


SAFE_HOLD_COMMAND = [
    0.307015,
    0.0,
    0.589907,
    3.141592653589793,
    0.0,
    0.0,
    1.0,
]


def generate_launch_description() -> LaunchDescription:
    strategy = LaunchConfiguration("strategy")
    adapter = Node(
        package="action_stream_isaac",
        executable="dynamic_isaac_adapter",
        name="action_stream_dynamic_isaac_adapter_process",
        output="screen",
        arguments=[
            "--matrix-manifest",
            LaunchConfiguration("matrix_manifest"),
            "--expected-strategy",
            strategy,
            "--headless",
            LaunchConfiguration("headless"),
            "--max-episodes",
            LaunchConfiguration("max_episodes"),
        ],
    )
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
                "safe_hold_command": SAFE_HOLD_COMMAND,
                "sync_periodic_replan": True,
                "observation_topic": "/action_stream/observation",
                "inference_request_topic": "/action_stream/inference_request",
                "action_chunk_topic": "/action_stream/action_chunk",
                "robot_command_topic": "/action_stream/robot_command",
                "diagnostics_topic": "/action_stream/diagnostics",
                "runtime_event_topic": "/action_stream/events",
                "episode_control_topic": "/action_stream/episode_control",
            }
        ],
    )
    router = ExecuteProcess(
        cmd=[LaunchConfiguration("zenohd_executable")],
        name="action_stream_owned_zenoh_router",
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("matrix_manifest"),
            DeclareLaunchArgument("strategy", default_value="aligned_async"),
            DeclareLaunchArgument("headless", default_value="true"),
            DeclareLaunchArgument("max_episodes", default_value="0"),
            DeclareLaunchArgument("zenohd_executable", default_value="rmw_zenohd"),
            router,
            executor,
            adapter,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=adapter,
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(reason="M8 native Isaac adapter complete")
                        )
                    ],
                )
            ),
        ]
    )
