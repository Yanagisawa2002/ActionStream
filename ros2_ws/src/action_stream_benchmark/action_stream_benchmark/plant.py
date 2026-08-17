"""Pure deterministic seven-dimensional reach-and-lift test plant."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Final, Sequence

from action_stream_policy.scripted_policy import (
    ACTION_DIMENSION,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    TASK_ID,
)


CONTROL_FREQUENCY_HZ: Final[float] = 20.0
CONTROL_PERIOD_SECONDS: Final[float] = 1.0 / CONTROL_FREQUENCY_HZ


def _finite(name: str, values: Sequence[float], length: int) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if len(converted) != length or not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain {length} finite values")
    return converted


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(first, second, strict=True)))


def deterministic_object_position(seed: int) -> tuple[float, float, float]:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    digest = hashlib.sha256(f"m7-plant:{seed}".encode("ascii")).digest()
    x = 0.45 + (int.from_bytes(digest[:4], "big") / (2**32 - 1) - 0.5) * 0.04
    y = (int.from_bytes(digest[4:8], "big") / (2**32 - 1) - 0.5) * 0.08
    return (x, y, 0.04)


@dataclass(frozen=True, slots=True)
class PlantObservation:
    episode_id: str
    observation_step: int
    task_id: str
    robot_state: tuple[float, ...]
    task_state: tuple[float, ...]
    terminated: bool
    success: bool


@dataclass(frozen=True, slots=True)
class PlantStep:
    observation: PlantObservation
    success: bool
    terminated: bool
    termination_reason: str


class ReachLiftPlant:
    """Bounded kinematic plant with deterministic grasp/lift detection.

    It is intentionally simple, but stale approach/open commands can release or
    lower the object, giving arrival-order chunk append an observable cost.
    """

    def __init__(
        self,
        *,
        max_steps: int = 180,
        success_hold_steps: int = 20,
        lift_height_m: float = 0.12,
        max_translation_per_step_m: float = 0.035,
        grasp_radius_m: float = 0.045,
    ) -> None:
        if min(max_steps, success_hold_steps) <= 0:
            raise ValueError("step limits must be positive")
        numeric = (lift_height_m, max_translation_per_step_m, grasp_radius_m)
        if any(not math.isfinite(value) or value <= 0 for value in numeric):
            raise ValueError("plant distances must be finite and positive")
        self.max_steps = max_steps
        self.success_hold_steps = success_hold_steps
        self.lift_height_m = lift_height_m
        self.max_translation_per_step_m = max_translation_per_step_m
        self.grasp_radius_m = grasp_radius_m
        self.reset("uninitialized", seed=0)

    def reset(self, episode_id: str, *, seed: int) -> PlantObservation:
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        self.episode_id = episode_id
        self.seed = seed
        self.step_count = 0
        self.object_xyz = list(deterministic_object_position(seed))
        self.initial_object_z = self.object_xyz[2]
        self.lift_target_xyz = [
            self.object_xyz[0],
            self.object_xyz[1],
            self.initial_object_z + self.lift_height_m + 0.03,
        ]
        self.eef_pose = [
            self.object_xyz[0] - 0.18,
            self.object_xyz[1],
            self.object_xyz[2] + 0.18,
            0.0,
            0.0,
            0.0,
            GRIPPER_OPEN,
        ]
        self.grasped = False
        self.success_streak = 0
        self.success = False
        self.terminated = False
        self.termination_reason = ""
        return self.observation()

    def observation(self) -> PlantObservation:
        task_state = (
            *self.object_xyz,
            *self.lift_target_xyz,
            1.0 if self.grasped else 0.0,
            self.initial_object_z,
            float(self.success_streak),
            1.0 if self.success else 0.0,
            1.0 if self.terminated else 0.0,
            float(self.step_count),
        )
        return PlantObservation(
            episode_id=self.episode_id,
            observation_step=self.step_count,
            task_id=TASK_ID,
            robot_state=tuple(self.eef_pose),
            task_state=task_state,
            terminated=self.terminated,
            success=self.success,
        )

    def step(self, command: Sequence[float]) -> PlantStep:
        if self.terminated:
            raise RuntimeError("cannot step a terminated plant")
        target = _finite("command", command, ACTION_DIMENSION)
        if target[-1] == 0.0:
            raise ValueError("zero has no gripper meaning")

        current_xyz = self.eef_pose[:3]
        delta = [target[index] - current_xyz[index] for index in range(3)]
        norm = math.sqrt(sum(value * value for value in delta))
        scale = min(1.0, self.max_translation_per_step_m / norm) if norm > 0 else 1.0
        self.eef_pose[:3] = [current_xyz[index] + delta[index] * scale for index in range(3)]
        self.eef_pose[3:6] = list(target[3:6])
        self.eef_pose[6] = GRIPPER_OPEN if target[6] > 0 else GRIPPER_CLOSED

        if self.eef_pose[6] > 0:
            self.grasped = False
        elif (
            not self.grasped
            and _distance(self.eef_pose[:3], self.object_xyz) <= self.grasp_radius_m
        ):
            self.grasped = True
        if self.grasped:
            self.object_xyz = list(self.eef_pose[:3])
        else:
            self.object_xyz[2] = max(self.initial_object_z, self.object_xyz[2] - 0.02)

        lifted = self.object_xyz[2] - self.initial_object_z >= self.lift_height_m
        self.success_streak = self.success_streak + 1 if self.grasped and lifted else 0
        self.step_count += 1
        if self.success_streak >= self.success_hold_steps:
            self.success = True
            self.terminated = True
            self.termination_reason = "lift_success"
        elif self.step_count >= self.max_steps:
            self.terminated = True
            self.termination_reason = "step_limit"
        observation = self.observation()
        return PlantStep(
            observation=observation,
            success=self.success,
            terminated=self.terminated,
            termination_reason=self.termination_reason,
        )
