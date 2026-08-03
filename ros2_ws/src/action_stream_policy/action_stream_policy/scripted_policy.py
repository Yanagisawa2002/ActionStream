"""Pure deterministic reach-and-lift policy used by the ROS adapter.

The public command is the repository's frozen seven-dimensional absolute
end-effector contract: xyz, axis-angle, and a signed gripper command.  ``+1``
means open and ``-1`` means closed.  Every chunk explicitly labels action
index ``i`` with target step ``O + i + 1``; arrival order is never used to
recover logical time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Iterable, Sequence


CONTROL_FREQUENCY_HZ: Final[float] = 20.0
CHUNK_HORIZON: Final[int] = 30
ACTION_DIMENSION: Final[int] = 7
TASK_ID: Final[str] = "scripted_reach_lift_v1"
GRIPPER_OPEN: Final[float] = 1.0
GRIPPER_CLOSED: Final[float] = -1.0

# Observation.task_state layout shared with the deterministic test plant.
TASK_STATE_LAYOUT: Final[tuple[str, ...]] = (
    "object_x",
    "object_y",
    "object_z",
    "lift_target_x",
    "lift_target_y",
    "lift_target_z",
    "grasped",
    "initial_object_z",
    "success_streak_steps",
    "success",
    "terminated",
    "episode_step",
)


def _finite_tuple(
    name: str,
    values: Iterable[float],
    *,
    expected_length: int | None = None,
) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if expected_length is not None and len(converted) != expected_length:
        raise ValueError(f"{name} must have length {expected_length}, got {len(converted)}")
    if not converted or not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain finite values")
    return converted


def _lerp(start: Sequence[float], end: Sequence[float], fraction: float) -> tuple[float, ...]:
    bounded = min(1.0, max(0.0, float(fraction)))
    return tuple(
        float(left) + (float(right) - float(left)) * bounded
        for left, right in zip(start, end, strict=True)
    )


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    episode_id: str
    observation_step: int
    task_id: str
    robot_state: tuple[float, ...]
    task_state: tuple[float, ...]
    terminated: bool = False

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if self.observation_step < 0:
            raise ValueError("observation_step must be non-negative")
        if self.task_id != TASK_ID:
            raise ValueError(f"unsupported task_id: {self.task_id!r}")
        object.__setattr__(
            self,
            "robot_state",
            _finite_tuple("robot_state", self.robot_state, expected_length=ACTION_DIMENSION),
        )
        object.__setattr__(
            self,
            "task_state",
            _finite_tuple("task_state", self.task_state, expected_length=len(TASK_STATE_LAYOUT)),
        )


@dataclass(frozen=True, slots=True)
class PolicyRequestData:
    episode_id: str
    request_id: int
    generation_id: int
    source_observation_step: int
    expected_horizon: int
    observation: PolicyObservation

    def __post_init__(self) -> None:
        if not self.episode_id or self.episode_id != self.observation.episode_id:
            raise ValueError("request and observation episode IDs must match")
        for name, value in (
            ("request_id", self.request_id),
            ("generation_id", self.generation_id),
            ("source_observation_step", self.source_observation_step),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.source_observation_step != self.observation.observation_step:
            raise ValueError("source observation step does not match the observation")
        if self.expected_horizon != CHUNK_HORIZON:
            raise ValueError(
                f"expected_horizon must preserve H={CHUNK_HORIZON}, "
                f"got {self.expected_horizon}"
            )


@dataclass(frozen=True, slots=True)
class TargetActionData:
    target_step: int
    command: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.target_step <= 0:
            raise ValueError("target_step must be positive")
        object.__setattr__(
            self,
            "command",
            _finite_tuple("command", self.command, expected_length=ACTION_DIMENSION),
        )
        if self.command[-1] not in {GRIPPER_OPEN, GRIPPER_CLOSED}:
            raise ValueError("gripper command must be exactly +1 (open) or -1 (closed)")


@dataclass(frozen=True, slots=True)
class ActionChunkData:
    episode_id: str
    request_id: int
    generation_id: int
    source_observation_step: int
    actions: tuple[TargetActionData, ...]
    action_dimension: int = ACTION_DIMENSION

    def __post_init__(self) -> None:
        if self.action_dimension != ACTION_DIMENSION:
            raise ValueError(f"action_dimension must be {ACTION_DIMENSION}")
        if len(self.actions) != CHUNK_HORIZON:
            raise ValueError(f"a policy chunk must contain {CHUNK_HORIZON} actions")
        expected = tuple(
            range(
                self.source_observation_step + 1,
                self.source_observation_step + CHUNK_HORIZON + 1,
            )
        )
        actual = tuple(action.target_step for action in self.actions)
        if actual != expected:
            raise ValueError(f"target steps must be O+1..O+H, got {actual}")


class ReachLiftScriptedPolicy:
    """Deterministic, training-free absolute-pose chunk producer.

    The chunk is closed-loop at request boundaries.  If the object is already
    grasped, every action continues lifting/holding it.  Otherwise the chunk
    approaches, descends, closes the gripper, and lifts.  This deliberately
    gives stale prefixes observable physical meaning without learning a policy.
    """

    horizon: int = CHUNK_HORIZON
    action_dimension: int = ACTION_DIMENSION

    def predict(self, request: PolicyRequestData) -> ActionChunkData:
        if request.observation.terminated:
            raise ValueError("cannot infer from a terminated observation")
        task = request.observation.task_state
        object_xyz = task[0:3]
        lift_target_xyz = task[3:6]
        grasped = task[6] >= 0.5
        current_xyz = request.observation.robot_state[0:3]
        orientation = request.observation.robot_state[3:6]

        actions: list[TargetActionData] = []
        for index in range(CHUNK_HORIZON):
            command = self._command_for_index(
                index=index,
                current_xyz=current_xyz,
                object_xyz=object_xyz,
                lift_target_xyz=lift_target_xyz,
                orientation=orientation,
                grasped=grasped,
            )
            actions.append(
                TargetActionData(
                    target_step=request.source_observation_step + index + 1,
                    command=command,
                )
            )
        return ActionChunkData(
            episode_id=request.episode_id,
            request_id=request.request_id,
            generation_id=request.generation_id,
            source_observation_step=request.source_observation_step,
            actions=tuple(actions),
        )

    @staticmethod
    def _command_for_index(
        *,
        index: int,
        current_xyz: Sequence[float],
        object_xyz: Sequence[float],
        lift_target_xyz: Sequence[float],
        orientation: Sequence[float],
        grasped: bool,
    ) -> tuple[float, ...]:
        if grasped:
            fraction = (index + 1) / 12.0
            xyz = _lerp(current_xyz, lift_target_xyz, fraction)
            gripper = GRIPPER_CLOSED
        elif index < 8:
            approach = (object_xyz[0], object_xyz[1], object_xyz[2] + 0.06)
            xyz = _lerp(current_xyz, approach, (index + 1) / 8.0)
            gripper = GRIPPER_OPEN
        elif index < 14:
            approach = (object_xyz[0], object_xyz[1], object_xyz[2] + 0.06)
            xyz = _lerp(approach, object_xyz, (index - 7) / 6.0)
            gripper = GRIPPER_OPEN
        elif index < 17:
            xyz = tuple(float(value) for value in object_xyz)
            gripper = GRIPPER_CLOSED
        else:
            xyz = _lerp(object_xyz, lift_target_xyz, (index - 16) / 13.0)
            gripper = GRIPPER_CLOSED
        return (*xyz, *tuple(float(value) for value in orientation), gripper)


def validate_chunk_contract(chunk: ActionChunkData) -> None:
    """Re-run the public O+1..O+H and finite-command contract."""

    ActionChunkData(
        episode_id=chunk.episode_id,
        request_id=chunk.request_id,
        generation_id=chunk.generation_id,
        source_observation_step=chunk.source_observation_step,
        actions=tuple(chunk.actions),
        action_dimension=chunk.action_dimension,
    )
