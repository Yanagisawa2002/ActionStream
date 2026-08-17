"""Pure reach-and-lift task semantics shared by the Isaac adapter and audits.

This module deliberately imports only the Python standard library.  It can be
unit tested on a machine that has neither ROS 2 nor Isaac Sim installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Final, Sequence

ISAAC_SIM_RELEASE: Final = "6.0.1"
ISAAC_PIP_VERSION: Final = "6.0.1.0"
TASK_ID: Final = "scripted_reach_lift_v1"
ACTION_DIMENSION: Final = 7
GRIPPER_OPEN: Final = 1.0
GRIPPER_CLOSED: Final = -1.0
PANDA_FINGER_OPEN_POSITION_M: Final = 0.04
PANDA_FINGER_CLOSED_POSITION_M: Final = 0.0
ACTION_LAYOUT: Final = (
    "absolute_end_effector_position_xyz[0:3]",
    "absolute_end_effector_axis_angle_xyz[3:6]",
    "normalized_gripper_command[6]",
)

PHASE_REACH: Final = 0
PHASE_GRASP: Final = 1
PHASE_LIFT: Final = 2
PHASE_SUCCESS: Final = 3
PHASE_TERMINATED: Final = 4

PHASE_NAMES: Final = {
    PHASE_REACH: "reach",
    PHASE_GRASP: "grasp",
    PHASE_LIFT: "lift",
    PHASE_SUCCESS: "success",
    PHASE_TERMINATED: "terminated",
}

# Observation.robot_state is the canonical, simulator-independent policy input.
# Panda joint and contact state remains private to the Isaac adapter.
ROBOT_STATE_LAYOUT: Final = (
    "end_effector_position_xyz[0:3]",
    "end_effector_axis_angle_xyz[3:6]",
    "normalized_gripper_command[6]",
)

# Observation.task_state exactly matches action_stream_policy. Boolean values
# are encoded as exactly 0.0 or 1.0.
TASK_STATE_LAYOUT: Final = (
    "object_position_xyz[0:3]",
    "lift_target_xyz[3:6]",
    "grasped[6]",
    "initial_object_z[7]",
    "success_streak_steps[8]",
    "success[9]",
    "terminated[10]",
    "episode_step[11]",
)


@dataclass(frozen=True)
class TaskThresholds:
    """Frozen geometric and termination thresholds for the task scaffold."""

    reach_distance_m: float = 0.075
    lift_success_m: float = 0.12
    max_grasp_distance_m: float = 0.18
    grasp_radius_m: float = 0.045
    success_hold_steps: int = 20
    max_steps: int = 180
    workspace_radius_m: float = 1.25
    floor_failure_z_m: float = -0.02

    def validate(self) -> None:
        numeric = (
            self.reach_distance_m,
            self.lift_success_m,
            self.max_grasp_distance_m,
            self.grasp_radius_m,
            self.workspace_radius_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
            raise ValueError("Task distance thresholds must be finite and positive")
        if not math.isfinite(self.floor_failure_z_m):
            raise ValueError("floor_failure_z_m must be finite")
        if self.success_hold_steps <= 0 or self.max_steps <= 0:
            raise ValueError("Task step thresholds must be positive")


@dataclass(frozen=True)
class TaskEvaluation:
    phase_code: int
    phase: str
    lift_height_m: float
    end_effector_object_distance_m: float
    success_streak_steps: int
    success: bool
    terminated: bool
    termination_reason: str


@dataclass(frozen=True)
class CommandValidation:
    accepted: bool
    reason: str
    command: tuple[float, ...] | None


def _finite_vector(name: str, values: Sequence[float], length: int) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if len(converted) != length:
        raise ValueError(f"{name} must have length {length}, got {len(converted)}")
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} contains a non-finite value")
    return converted


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the Euclidean distance between two finite xyz vectors."""

    av = _finite_vector("a", a, 3)
    bv = _finite_vector("b", b, 3)
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(av, bv, strict=True)))


def deterministic_spawn_position(
    episode_id: str,
    *,
    base_xyz: Sequence[float] = (0.45, 0.0, 0.026),
    xy_jitter_m: float = 0.025,
) -> tuple[float, float, float]:
    """Map an episode ID to a deterministic object position.

    Paired strategies receive the same pose whenever the benchmark reuses the
    same episode ID.  SHA-256 is used instead of Python's randomized ``hash``.
    """

    if not episode_id:
        raise ValueError("episode_id must be non-empty")
    base = _finite_vector("base_xyz", base_xyz, 3)
    if not math.isfinite(xy_jitter_m) or xy_jitter_m < 0.0:
        raise ValueError("xy_jitter_m must be finite and non-negative")
    digest = hashlib.sha256(episode_id.encode("utf-8")).digest()
    x_unit = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
    y_unit = int.from_bytes(digest[8:16], "big") / float((1 << 64) - 1)
    return (
        base[0] + (2.0 * x_unit - 1.0) * xy_jitter_m,
        base[1] + (2.0 * y_unit - 1.0) * xy_jitter_m,
        base[2],
    )


def evaluate_task(
    *,
    end_effector_xyz: Sequence[float],
    object_xyz: Sequence[float],
    initial_object_z: float,
    episode_step: int,
    previous_success_streak: int,
    thresholds: TaskThresholds,
) -> TaskEvaluation:
    """Evaluate deterministic reach/lift success and automated termination."""

    thresholds.validate()
    eef = _finite_vector("end_effector_xyz", end_effector_xyz, 3)
    obj = _finite_vector("object_xyz", object_xyz, 3)
    initial_z = float(initial_object_z)
    if not math.isfinite(initial_z):
        raise ValueError("initial_object_z must be finite")
    if episode_step < 0 or previous_success_streak < 0:
        raise ValueError("step counters must be non-negative")

    distance = euclidean_distance(eef, obj)
    lift_height = obj[2] - initial_z
    qualifying = (
        lift_height >= thresholds.lift_success_m and distance <= thresholds.max_grasp_distance_m
    )
    streak = previous_success_streak + 1 if qualifying else 0
    success = streak >= thresholds.success_hold_steps

    radial_distance = math.sqrt(obj[0] ** 2 + obj[1] ** 2)
    reason = ""
    terminated = False
    if success:
        phase_code = PHASE_SUCCESS
        terminated = True
        reason = "lift_success"
    elif obj[2] < thresholds.floor_failure_z_m:
        phase_code = PHASE_TERMINATED
        terminated = True
        reason = "object_below_floor"
    elif radial_distance > thresholds.workspace_radius_m:
        phase_code = PHASE_TERMINATED
        terminated = True
        reason = "object_left_workspace"
    elif episode_step >= thresholds.max_steps:
        phase_code = PHASE_TERMINATED
        terminated = True
        reason = "step_limit"
    elif lift_height > 0.02:
        phase_code = PHASE_LIFT
    elif distance <= thresholds.reach_distance_m:
        phase_code = PHASE_GRASP
    else:
        phase_code = PHASE_REACH

    return TaskEvaluation(
        phase_code=phase_code,
        phase=PHASE_NAMES[phase_code],
        lift_height_m=lift_height,
        end_effector_object_distance_m=distance,
        success_streak_steps=streak,
        success=success,
        terminated=terminated,
        termination_reason=reason,
    )


def validate_ee_command(command: Sequence[float]) -> tuple[float, ...]:
    """Validate canonical 7D absolute-EE ActionStream commands.

    Joint-space conversion is intentionally private to the Isaac adapter.  A
    seven-dimensional test-plant command must never be mistaken for nine Panda
    joint targets.
    """

    validated = _finite_vector("command", command, ACTION_DIMENSION)
    if validated[6] not in {GRIPPER_OPEN, GRIPPER_CLOSED}:
        raise ValueError("gripper command must be exactly +1 (open) or -1 (closed)")
    return validated


def validate_robot_command_for_step(
    *,
    active_episode_id: str | None,
    latest_observation_step: int,
    pending_target_step: int | None,
    message_episode_id: str,
    actual_target_step: int,
    command: Sequence[float],
    hold: bool,
) -> CommandValidation:
    """Validate the adapter boundary without importing ROS or Isaac.

    ``source_target_step`` is deliberately absent: naive asynchronous execution
    may legally apply a source-labelled action at a different actual step.  The
    actuator boundary is keyed only by ``actual_target_step``.
    """

    if active_episode_id is None:
        return CommandValidation(False, "episode_not_active", None)
    if message_episode_id != active_episode_id:
        return CommandValidation(False, "previous_episode", None)
    if latest_observation_step < 0:
        return CommandValidation(False, "invalid_observation_step", None)
    expected = latest_observation_step + 1
    if actual_target_step != expected:
        return CommandValidation(False, "unexpected_actual_target_step", None)
    if pending_target_step == actual_target_step:
        return CommandValidation(False, "duplicate_actual_target_step", None)
    try:
        if hold:
            validated = _finite_vector("hold command", command, ACTION_DIMENSION)
        else:
            validated = validate_ee_command(command)
    except (TypeError, ValueError):
        return CommandValidation(False, "invalid_command", None)
    return CommandValidation(True, "accepted_hold" if hold else "accepted", validated)


def gripper_command_to_finger_positions(command: float) -> tuple[float, float]:
    """Sign-quantize a gripper command to the two Panda finger joints.

    The public policy emits exactly ``+1`` or ``-1``.  Sign quantization keeps
    the repository's gripper-switch semantics explicit for audit/replay while
    still making this low-level mapping safe for a finite out-of-range value.
    Zero is intentionally rejected rather than assigned an invented state.
    """

    value = float(command)
    if not math.isfinite(value):
        raise ValueError("gripper command must be finite")
    value = min(1.0, max(-1.0, value))
    if value > 0.0:
        return (PANDA_FINGER_OPEN_POSITION_M, PANDA_FINGER_OPEN_POSITION_M)
    if value < 0.0:
        return (PANDA_FINGER_CLOSED_POSITION_M, PANDA_FINGER_CLOSED_POSITION_M)
    raise ValueError("gripper command zero has no open/closed semantic state")


def axis_angle_to_quaternion_wxyz(axis_angle_xyz: Sequence[float]) -> tuple[float, ...]:
    """Convert an exponential-coordinate rotation vector to quaternion wxyz."""

    rotation = _finite_vector("axis_angle_xyz", axis_angle_xyz, 3)
    angle = math.sqrt(sum(component * component for component in rotation))
    if angle < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    scale = math.sin(0.5 * angle) / angle
    return (
        math.cos(0.5 * angle),
        rotation[0] * scale,
        rotation[1] * scale,
        rotation[2] * scale,
    )


def quaternion_wxyz_to_axis_angle(quaternion_wxyz: Sequence[float]) -> tuple[float, ...]:
    """Convert a quaternion to the canonical shortest rotation vector."""

    quaternion = _finite_vector("quaternion_wxyz", quaternion_wxyz, 4)
    magnitude = math.sqrt(sum(component * component for component in quaternion))
    if magnitude < 1e-12:
        raise ValueError("quaternion_wxyz must have non-zero norm")
    normalized = tuple(component / magnitude for component in quaternion)
    if normalized[0] < 0.0:
        normalized = tuple(-component for component in normalized)
    vector_norm = math.sqrt(sum(component * component for component in normalized[1:]))
    if vector_norm < 1e-12:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vector_norm, min(1.0, max(-1.0, normalized[0])))
    scale = angle / vector_norm
    return tuple(component * scale for component in normalized[1:])


def pack_robot_state(
    *,
    end_effector_xyz: Sequence[float],
    end_effector_axis_angle_xyz: Sequence[float],
    gripper_command: float,
) -> tuple[float, ...]:
    eef_xyz = _finite_vector("end_effector_xyz", end_effector_xyz, 3)
    eef_axis_angle = _finite_vector("end_effector_axis_angle_xyz", end_effector_axis_angle_xyz, 3)
    gripper = float(gripper_command)
    if gripper not in {GRIPPER_OPEN, GRIPPER_CLOSED}:
        raise ValueError("gripper_command must be exactly +1 or -1")
    return eef_xyz + eef_axis_angle + (gripper,)


def pack_task_state(
    *,
    object_xyz: Sequence[float],
    lift_target_xyz: Sequence[float],
    grasped: bool,
    initial_object_z: float,
    evaluation: TaskEvaluation,
    episode_step: int,
) -> tuple[float, ...]:
    obj_xyz = _finite_vector("object_xyz", object_xyz, 3)
    target = _finite_vector("lift_target_xyz", lift_target_xyz, 3)
    initial_z = float(initial_object_z)
    if not math.isfinite(initial_z):
        raise ValueError("initial_object_z must be finite")
    return (
        *obj_xyz,
        *target,
        1.0 if grasped else 0.0,
        initial_z,
        float(evaluation.success_streak_steps),
        1.0 if evaluation.success else 0.0,
        1.0 if evaluation.terminated else 0.0,
        float(episode_step),
    )


def split_sim_time(seconds: float) -> tuple[int, int]:
    """Convert non-negative simulation seconds to normalized ROS sec/nanosec."""

    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("simulation time must be finite and non-negative")
    sec = int(math.floor(seconds))
    nanosec = int(round((seconds - sec) * 1_000_000_000))
    if nanosec == 1_000_000_000:
        sec += 1
        nanosec = 0
    return sec, nanosec
