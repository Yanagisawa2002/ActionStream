"""Observation-conditioned M8-G0 dynamic pick-and-place chunk policy.

The deterministic controller mirrors the installed Isaac Sim 6.0.1 Franka
pick/place example's geometry: the Panda hand approaches 0.20 m above the cube
and grasps with its hand frame 0.10 m above the cube center.  It has no hidden
trajectory state; every 30-step chunk is derived only from the supplied
observation and explicitly targets ``O+1 .. O+30``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Iterable, Sequence


TASK_ID: Final = "dynamic_target_pick_place_v1"
CONTROL_FREQUENCY_HZ: Final = 20.0
CHUNK_HORIZON: Final = 30
ACTION_DIMENSION: Final = 7
GRIPPER_OPEN: Final = 1.0
GRIPPER_CLOSED: Final = -1.0
DYNAMIC_TASK_STATE_SIZE: Final = 44

PHASE_APPROACH_ABOVE: Final = 0
PHASE_DESCEND: Final = 1
PHASE_CLOSE: Final = 2
PHASE_LIFT: Final = 3
PHASE_TRANSPORT: Final = 4
PHASE_LOWER: Final = 5
PHASE_RELEASE: Final = 6
PHASE_RETRACT: Final = 7
PHASE_STABILIZE: Final = 8
PHASE_SUCCESS: Final = 9
PHASE_TERMINATED: Final = 10
PHASE_CODES: Final = frozenset(range(PHASE_TERMINATED + 1))


def _finite_tuple(
    name: str, values: Iterable[float], *, expected_length: int
) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if len(converted) != expected_length:
        raise ValueError(f"{name} must have length {expected_length}, got {len(converted)}")
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True))
    )


def _strict_binary(name: str, value: float) -> bool:
    if value not in {0.0, 1.0}:
        raise ValueError(f"task_state {name} must be exactly 0 or 1")
    return value == 1.0


def _nonnegative_integer(name: str, value: float) -> int:
    converted = int(value)
    if float(converted) != value or converted < 0:
        raise ValueError(f"task_state {name} must be a non-negative integer")
    return converted


@dataclass(frozen=True, slots=True)
class DynamicPolicyConfig:
    approach_height_m: float = 0.20
    grasp_hand_offset_m: float = 0.10
    carry_hand_height_m: float = 0.32
    recovery_hover_offset_m: float = 0.13
    maximum_translation_per_step_m: float = 0.01
    workspace_x_m: tuple[float, float] = (0.25, 0.70)
    workspace_y_m: tuple[float, float] = (-0.35, 0.35)
    workspace_z_m: tuple[float, float] = (0.08, 0.60)
    downward_axis_angle_xyz: tuple[float, float, float] = (math.pi, 0.0, 0.0)

    def validate(self) -> None:
        positive = (
            self.approach_height_m,
            self.grasp_hand_offset_m,
            self.carry_hand_height_m,
            self.recovery_hover_offset_m,
            self.maximum_translation_per_step_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("policy distances must be finite and positive")
        for name, bounds in (
            ("workspace_x_m", self.workspace_x_m),
            ("workspace_y_m", self.workspace_y_m),
            ("workspace_z_m", self.workspace_z_m),
        ):
            pair = _finite_tuple(name, bounds, expected_length=2)
            if pair[0] >= pair[1]:
                raise ValueError(f"{name} lower bound must be smaller than upper bound")
        _finite_tuple(
            "downward_axis_angle_xyz",
            self.downward_axis_angle_xyz,
            expected_length=3,
        )


@dataclass(frozen=True, slots=True)
class DynamicPolicyObservation:
    episode_id: str
    observation_step: int
    generation_id: int
    task_id: str
    robot_state: tuple[float, ...]
    task_state: tuple[float, ...]
    terminated: bool = False

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if self.observation_step < 0:
            raise ValueError("observation_step must be non-negative")
        if self.generation_id <= 0:
            raise ValueError("generation_id must be positive")
        if type(self.terminated) is not bool:
            raise ValueError("terminated must be exactly bool")
        if self.task_id != TASK_ID:
            raise ValueError(f"unsupported task_id: {self.task_id!r}")
        object.__setattr__(
            self,
            "robot_state",
            _finite_tuple("robot_state", self.robot_state, expected_length=ACTION_DIMENSION),
        )
        state = _finite_tuple(
            "task_state", self.task_state, expected_length=DYNAMIC_TASK_STATE_SIZE
        )
        object.__setattr__(self, "task_state", state)
        state_generation = int(state[24])
        if (
            float(state_generation) != state[24]
            or state_generation not in {1, 2}
            or state_generation != self.generation_id
        ):
            raise ValueError("observation generation does not match task_state generation")
        active_destination_id = _strict_binary("active_destination_id", state[22])
        disturbance_switched = _strict_binary("disturbance_switched", state[23])
        for name, index in (
            ("grasped", 27),
            ("correct_destination_placement", 34),
            ("obsolete_destination_placement", 35),
            ("collision", 36),
            ("joint_or_workspace_limit", 37),
            ("grasp_ever", 38),
            ("success", 39),
            ("terminated", 40),
        ):
            _strict_binary(name, state[index])
        phase = _nonnegative_integer("phase_code", state[25])
        if phase not in PHASE_CODES:
            raise ValueError(f"unsupported task_state phase code: {phase}")
        for name, index in (
            ("phase_step", 26),
            ("success_streak_steps", 33),
            ("episode_step", 41),
            ("switch_step", 42),
        ):
            _nonnegative_integer(name, state[index])
        if active_destination_id != disturbance_switched:
            raise ValueError("active destination ID and disturbance state disagree")
        if disturbance_switched != (state_generation == 2):
            raise ValueError("disturbance state and generation disagree")
        if self.terminated != (state[40] == 1.0):
            raise ValueError("observation terminated flag and task_state disagree")


@dataclass(frozen=True, slots=True)
class DynamicPolicyRequestData:
    episode_id: str
    request_id: int
    generation_id: int
    source_observation_step: int
    expected_horizon: int
    observation: DynamicPolicyObservation

    def __post_init__(self) -> None:
        if not self.episode_id or self.episode_id != self.observation.episode_id:
            raise ValueError("request and observation episode IDs must match")
        if self.request_id <= 0:
            raise ValueError("request_id must be positive")
        if self.generation_id != self.observation.generation_id:
            raise ValueError("request and observation generation IDs must match")
        if self.source_observation_step != self.observation.observation_step:
            raise ValueError("source observation step does not match observation")
        if self.expected_horizon != CHUNK_HORIZON:
            raise ValueError(f"expected_horizon must be {CHUNK_HORIZON}")


@dataclass(frozen=True, slots=True)
class DynamicTargetActionData:
    target_step: int
    command: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.target_step <= 0:
            raise ValueError("target_step must be positive")
        command = _finite_tuple("command", self.command, expected_length=ACTION_DIMENSION)
        if command[6] not in {GRIPPER_OPEN, GRIPPER_CLOSED}:
            raise ValueError("gripper command must be exactly +1 or -1")
        object.__setattr__(self, "command", command)


@dataclass(frozen=True, slots=True)
class DynamicActionChunkData:
    episode_id: str
    request_id: int
    generation_id: int
    source_observation_step: int
    actions: tuple[DynamicTargetActionData, ...]
    action_dimension: int = ACTION_DIMENSION

    def __post_init__(self) -> None:
        if self.action_dimension != ACTION_DIMENSION:
            raise ValueError(f"action_dimension must be {ACTION_DIMENSION}")
        if len(self.actions) != CHUNK_HORIZON:
            raise ValueError(f"a chunk must contain {CHUNK_HORIZON} actions")
        expected = tuple(
            range(
                self.source_observation_step + 1,
                self.source_observation_step + CHUNK_HORIZON + 1,
            )
        )
        actual = tuple(action.target_step for action in self.actions)
        if actual != expected:
            raise ValueError("target steps must be exactly O+1 through O+H")


class DynamicPickPlacePolicy:
    """Deterministic observation-conditioned Cartesian waypoint controller."""

    horizon: int = CHUNK_HORIZON
    action_dimension: int = ACTION_DIMENSION

    def __init__(self, config: DynamicPolicyConfig | None = None) -> None:
        self.config = config or DynamicPolicyConfig()
        self.config.validate()

    def _validate_goal(self, goal: Sequence[float]) -> tuple[float, float, float]:
        xyz = _finite_tuple("Cartesian goal", goal, expected_length=3)
        for value, bounds, axis in zip(
            xyz,
            (
                self.config.workspace_x_m,
                self.config.workspace_y_m,
                self.config.workspace_z_m,
            ),
            "xyz",
            strict=True,
        ):
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(
                    f"Cartesian {axis} goal {value} is outside workspace {bounds}"
                )
        return xyz

    def _goal_and_gripper(
        self, observation: DynamicPolicyObservation
    ) -> tuple[tuple[float, float, float], float]:
        state = observation.task_state
        obj = state[0:3]
        original_destination = state[13:16]
        final_destination = state[16:19]
        active_destination = state[19:22]
        disturbance_switched = state[23] == 1.0
        phase_code = int(state[25])
        grasped = state[27] == 1.0
        expected_active = final_destination if disturbance_switched else original_destination
        if _distance(active_destination, expected_active) > 1e-9:
            raise ValueError("active destination disagrees with disturbance state")
        if phase_code in {PHASE_SUCCESS, PHASE_TERMINATED} or observation.terminated:
            raise ValueError("cannot infer from a terminated dynamic-task observation")

        if phase_code == PHASE_APPROACH_ABOVE:
            goal = (obj[0], obj[1], obj[2] + self.config.approach_height_m)
            gripper = GRIPPER_OPEN
        elif phase_code == PHASE_DESCEND:
            goal = (obj[0], obj[1], obj[2] + self.config.grasp_hand_offset_m)
            gripper = GRIPPER_OPEN
        elif phase_code == PHASE_CLOSE:
            goal = (obj[0], obj[1], obj[2] + self.config.grasp_hand_offset_m)
            gripper = GRIPPER_CLOSED
        elif phase_code in {PHASE_LIFT, PHASE_TRANSPORT, PHASE_LOWER} and not grasped:
            # A lost/uncertain grasp must never send the hand toward a destination.
            goal = (obj[0], obj[1], obj[2] + self.config.recovery_hover_offset_m)
            gripper = GRIPPER_CLOSED
        elif phase_code == PHASE_LIFT:
            goal = (obj[0], obj[1], self.config.carry_hand_height_m)
            gripper = GRIPPER_CLOSED
        elif phase_code == PHASE_TRANSPORT:
            goal = (
                active_destination[0],
                active_destination[1],
                self.config.carry_hand_height_m,
            )
            gripper = GRIPPER_CLOSED
        elif phase_code == PHASE_LOWER:
            goal = (
                active_destination[0],
                active_destination[1],
                active_destination[2] + self.config.grasp_hand_offset_m,
            )
            gripper = GRIPPER_CLOSED
        elif phase_code == PHASE_RELEASE:
            goal = (
                active_destination[0],
                active_destination[1],
                active_destination[2] + self.config.grasp_hand_offset_m,
            )
            gripper = GRIPPER_OPEN
        elif phase_code in {PHASE_RETRACT, PHASE_STABILIZE}:
            goal = (
                active_destination[0],
                active_destination[1],
                active_destination[2] + self.config.approach_height_m,
            )
            gripper = GRIPPER_OPEN
        else:
            raise ValueError(f"unsupported dynamic task phase code: {phase_code}")
        return self._validate_goal(goal), gripper

    def _advance(
        self, current: Sequence[float], goal: Sequence[float]
    ) -> tuple[float, float, float]:
        start = _finite_tuple("current end-effector xyz", current, expected_length=3)
        target = _finite_tuple("goal xyz", goal, expected_length=3)
        delta = tuple(right - left for left, right in zip(start, target, strict=True))
        distance = _distance(start, target)
        if distance <= self.config.maximum_translation_per_step_m:
            return target
        scale = self.config.maximum_translation_per_step_m / distance
        return tuple(left + scale * change for left, change in zip(start, delta, strict=True))

    def predict(self, request: DynamicPolicyRequestData) -> DynamicActionChunkData:
        goal, gripper = self._goal_and_gripper(request.observation)
        current = self._validate_goal(request.observation.robot_state[0:3])
        orientation = self.config.downward_axis_angle_xyz
        actions: list[DynamicTargetActionData] = []
        for index in range(CHUNK_HORIZON):
            current = self._advance(current, goal)
            command = (*current, *orientation, gripper)
            actions.append(
                DynamicTargetActionData(
                    target_step=request.source_observation_step + index + 1,
                    command=command,
                )
            )
        return DynamicActionChunkData(
            episode_id=request.episode_id,
            request_id=request.request_id,
            generation_id=request.generation_id,
            source_observation_step=request.source_observation_step,
            actions=tuple(actions),
        )
