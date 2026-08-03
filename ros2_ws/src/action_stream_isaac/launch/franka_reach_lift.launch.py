"""Launch the production Isaac adapter; this file has no test-plant fallback."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="true"),
            DeclareLaunchArgument("auto_start_episode", default_value=""),
            DeclareLaunchArgument("exit_after_auto_episode", default_value="false"),
            Node(
                package="action_stream_isaac",
                executable="isaac_adapter",
                name="action_stream_isaac_adapter",
                namespace="action_stream",
                output="screen",
                arguments=[
                    "--headless",
                    LaunchConfiguration("headless"),
                    "--auto-start-episode",
                    LaunchConfiguration("auto_start_episode"),
                    "--exit-after-auto-episode",
                    LaunchConfiguration("exit_after_auto_episode"),
                ],
            ),
        ]
    )
