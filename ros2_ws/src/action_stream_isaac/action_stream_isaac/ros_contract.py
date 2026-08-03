"""Centralized validation for the frozen ActionStream ROS message contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

OBSERVATION_TOPIC: Final = "/action_stream/observation"
BOOTSTRAP_OBSERVATION_TOPIC: Final = "/action_stream/isaac/bootstrap_observation"
ROBOT_COMMAND_TOPIC: Final = "/action_stream/robot_command"
EPISODE_CONTROL_TOPIC: Final = "/action_stream/episode_control"
CLOCK_TOPIC: Final = "/clock"

OBSERVATION_FIELDS: Final = (
    "sim_stamp",
    "steady_time_ns",
    "wall_time_ns",
    "episode_id",
    "observation_step",
    "task_id",
    "robot_state",
    "task_state",
    "terminated",
)
ROBOT_COMMAND_FIELDS: Final = (
    "sim_stamp",
    "steady_time_ns",
    "wall_time_ns",
    "episode_id",
    "actual_target_step",
    "source_request_id",
    "source_generation_id",
    "source_observation_step",
    "source_target_step",
    "command",
    "hold",
    "reason",
)
EPISODE_CONTROL_FIELDS: Final = (
    "message_kind",
    "command",
    "sim_stamp",
    "steady_time_ns",
    "wall_time_ns",
    "episode_id",
    "generation_id",
    "active",
    "terminated",
    "success",
)
EPISODE_CONTROL_CONSTANTS: Final = (
    "REQUEST",
    "STATUS",
    "START",
    "RESET",
    "TERMINATE",
)


@dataclass(frozen=True)
class RosMessageTypes:
    Observation: type
    RobotCommand: type
    EpisodeControl: type
    Clock: type


def _fields(message_type: type) -> tuple[str, ...]:
    getter = getattr(message_type, "get_fields_and_field_types", None)
    if getter is None:
        raise RuntimeError(
            f"{message_type!r} is not a generated ROS 2 message type: "
            "get_fields_and_field_types() is missing"
        )
    return tuple(getter())


def require_exact_fields(message_type: type, expected: tuple[str, ...]) -> None:
    """Fail loudly if a generated message has drifted from the frozen schema."""

    actual = _fields(message_type)
    if actual != expected:
        missing = tuple(field for field in expected if field not in actual)
        unexpected = tuple(field for field in actual if field not in expected)
        raise RuntimeError(
            f"ROS schema mismatch for {message_type.__module__}.{message_type.__name__}: "
            f"expected ordered fields={expected}, actual={actual}, missing={missing}, "
            f"unexpected={unexpected}. Rebuild action_stream_msgs and this workspace."
        )


def load_and_validate_message_types() -> RosMessageTypes:
    """Import generated interfaces only when a sourced ROS workspace is present."""

    try:
        from action_stream_msgs.msg import EpisodeControl, Observation, RobotCommand
        from rosgraph_msgs.msg import Clock
    except ImportError as exc:
        raise RuntimeError(
            "ActionStream ROS interfaces are unavailable. Source ROS 2 Jazzy, build "
            "ros2_ws with Python 3.12, and source the platform-appropriate "
            "ros2_ws/install/setup script before "
            "launching the Isaac adapter. The deterministic test plant is not a "
            "fallback for this executable."
        ) from exc

    require_exact_fields(Observation, OBSERVATION_FIELDS)
    require_exact_fields(RobotCommand, ROBOT_COMMAND_FIELDS)
    require_exact_fields(EpisodeControl, EPISODE_CONTROL_FIELDS)
    missing_constants = tuple(
        name for name in EPISODE_CONTROL_CONSTANTS if not hasattr(EpisodeControl, name)
    )
    if missing_constants:
        raise RuntimeError(
            "EpisodeControl is missing frozen constants "
            f"{missing_constants}; rebuild action_stream_msgs"
        )
    return RosMessageTypes(
        Observation=Observation,
        RobotCommand=RobotCommand,
        EpisodeControl=EpisodeControl,
        Clock=Clock,
    )


def set_ros_time(stamp: Any, *, sec: int, nanosec: int) -> None:
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError(f"invalid ROS time: sec={sec}, nanosec={nanosec}")
    stamp.sec = sec
    stamp.nanosec = nanosec
