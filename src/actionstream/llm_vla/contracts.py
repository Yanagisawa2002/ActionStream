"""Fail-closed public contracts, with no simulator truth or model dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import cached_property
import hashlib
import json
import time
from types import MappingProxyType

import numpy as np

CAPABILITY = dict(
    skill="pick_place",
    target="tomato_sauce",
    destination="basket",
    canonical_instruction="pick up the tomato sauce and place it in the basket",
)


def strict_object(raw: str, fields: set[str]) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=unique)
    if type(value) is not dict or set(value) != fields:
        raise ValueError("JSON fields do not match the contract")
    return value


@dataclass(frozen=True)
class TaskSpec:
    version: str
    request_id: str
    decision: str
    skill: str | None
    target: str | None
    destination: str | None
    canonical_instruction: str | None
    reason: str

    @classmethod
    def parse(cls, raw: str, request_id: str) -> TaskSpec:
        data = strict_object(raw, set(cls.__dataclass_fields__))
        if data["version"] != "1.0" or data["request_id"] != request_id:
            raise ValueError("Task identity mismatch")
        if data["decision"] not in ("accept", "reject", "unknown"):
            raise ValueError("Invalid task decision")
        if not isinstance(data["reason"], str) or not 1 <= len(data["reason"]) <= 1000:
            raise ValueError("Invalid task explanation")
        expected = (
            CAPABILITY if data["decision"] == "accept" else dict.fromkeys(CAPABILITY)
        )
        if any(data[key] != value for key, value in expected.items()):
            raise ValueError("Task capability mismatch")
        return cls(**data)

    def require_accepted(self):
        # Revalidate even directly constructed dataclasses at the execution boundary.
        TaskSpec.parse(json.dumps(asdict(self)), self.request_id)
        if self.decision != "accept":
            raise ValueError("Only an accepted TaskSpec may create an execution")


@dataclass(frozen=True)
class CheckDecision:
    observation_id: str
    status: str
    evidence: str

    @classmethod
    def parse(cls, raw: str, observation_id: str) -> CheckDecision:
        data = strict_object(raw, set(cls.__dataclass_fields__))
        if data["observation_id"] != observation_id:
            raise ValueError("Checker observation identity mismatch")
        if data["status"] not in ("complete", "incomplete", "unknown"):
            raise ValueError("Invalid checker decision")
        if (
            not isinstance(data["evidence"], str)
            or not 1 <= len(data["evidence"].split()) <= 30
        ):
            raise ValueError("Checker evidence must contain 1 to 30 words")
        return cls(**data)


def frozen_array(value, shape, dtype=None):
    array = np.array(value, copy=True)
    if list(array.shape) != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid runtime observation array: {array.shape}")
    if dtype is not None and array.dtype != dtype:
        raise ValueError("Runtime image dtype changed")
    # Bytes-backed arrays cannot be made writeable by a downstream consumer.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class Observation:
    request_id: str
    revision: int
    observation_id: str
    captured_monotonic: float
    frame_hashes: tuple[str, str]
    pixels: object
    robot_state: object

    @cached_property
    def completion_rgb(self):
        """One immutable preprocessing result for this immutable observation.

        The recorder and verifier share pixels, never simulator truth. A new
        observation (including a new revision) owns a separate cache.
        """
        from .temporal_completion import camera_rgb

        rgb = camera_rgb(
            {
                "agentview_image": self.pixels["image"][0],
                "robot0_eye_in_hand_image": self.pixels["image2"][0],
            }
        )
        return np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(rgb.shape)

    @classmethod
    def project(
        cls,
        raw: dict,
        request_id: str,
        revision: int,
        observation_id: str,
        captured_monotonic: float | None = None,
    ) -> Observation:
        pixels = {
            key: frozen_array(raw["pixels"][key], [1, 360, 360, 3], np.uint8)
            for key in ("image", "image2")
        }
        dimensions = {
            "eef.pos": 3,
            "eef.quat": 4,
            "eef.mat": (3, 3),
            "gripper.qpos": 2,
            "gripper.qvel": 2,
            "joints.pos": 7,
            "joints.vel": 7,
        }
        state = {}
        for key, size in dimensions.items():
            group, name = key.split(".")
            state.setdefault(group, {})[name] = frozen_array(
                raw["robot_state"][group][name],
                [1, *(size if isinstance(size, tuple) else (size,))],
            )
        state = {key: MappingProxyType(value) for key, value in state.items()}
        hashes = tuple(
            hashlib.sha256(value.tobytes()).hexdigest() for value in pixels.values()
        )
        return cls(
            request_id,
            revision,
            observation_id,
            time.monotonic() if captured_monotonic is None else captured_monotonic,
            hashes,
            MappingProxyType(pixels),
            MappingProxyType(state),
        )

    def native_input(self):
        return {
            "pixels": {key: value.copy() for key, value in self.pixels.items()},
            "robot_state": {
                group: {key: value.copy() for key, value in fields.items()}
                for group, fields in self.robot_state.items()
            },
        }

    def checker_images(self):
        from PIL import Image

        return [
            Image.fromarray(self.pixels["image"][0]).transpose(
                Image.Transpose.ROTATE_180
            ),
            Image.fromarray(self.pixels["image2"][0]),
        ]

    def binding(self):
        return dict(
            request_id=self.request_id,
            revision=self.revision,
            observation_id=self.observation_id,
            frame_hashes=self.frame_hashes,
            captured_monotonic=self.captured_monotonic,
        )


@dataclass
class Execution:
    task: TaskSpec
    max_steps: int = 300
    max_age_s: float = 30
    revision: int = 0
    steps: int = 0
    recoveries: int = 0
    status: str = "accepted"
    pending: list = field(default_factory=list)
    chunk_revision: int | None = None

    def __post_init__(self):
        self.task.require_accepted()
        if self.max_steps <= 0 or self.max_steps > 300:
            raise ValueError("Total control budget must be within 1..300")

    def validate_current(self, observation: Observation, now: float | None = None):
        age = (
            time.monotonic() if now is None else now
        ) - observation.captured_monotonic
        if (
            observation.request_id != self.task.request_id
            or observation.revision != self.revision
            or not 0 <= age <= self.max_age_s
        ):
            raise ValueError("Stale or foreign observation")

    def queue(self, actions, observation: Observation):
        self.validate_current(observation)
        if self.status not in ("accepted", "executing") or self.pending:
            raise ValueError("Execution is not ready for a new chunk")
        array = np.asarray(actions)
        if (
            array.shape != (30, 7)
            or not np.isfinite(array).all()
            or np.abs(array[:, 6]).max() > 1
        ):
            raise ValueError("Invalid native processed action chunk")
        self.pending = [row.copy() for row in array]
        self.chunk_revision = self.revision
        self.status = "executing"

    def take_action(self):
        if (
            self.status != "executing"
            or self.chunk_revision != self.revision
            or self.steps >= self.max_steps
            or not self.pending
        ):
            raise ValueError("No authorized action in this generation/budget")
        # Reserve a physical call before issuing it: a raised step cannot be retried for free.
        self.steps += 1
        return self.pending.pop(0)

    def interrupt(self):
        if self.status != "executing" or self.recoveries:
            raise ValueError("At most one interruption recovery is permitted")
        discarded = len(self.pending)
        self.pending.clear()
        self.chunk_revision = None
        self.revision += 1
        self.status = "interrupted"
        return discarded

    def apply_check(
        self,
        decision: CheckDecision,
        submitted: Observation,
        current: Observation,
        native_boundary: bool = False,
    ):
        self.validate_current(current)
        if (
            submitted.binding() != current.binding()
            or decision.observation_id != current.observation_id
        ):
            raise ValueError("Checker result is bound to another observation")
        CheckDecision.parse(json.dumps(asdict(decision)), current.observation_id)
        if self.status not in ("accepted", "executing", "interrupted"):
            raise ValueError("Execution already stopped")
        self.pending.clear()
        self.chunk_revision = None
        if decision.status == "complete":
            self.status = "complete"
        elif decision.status == "unknown":
            self.status = "unknown_stop"
        elif native_boundary:
            self.status = "native_boundary_stop"
        elif self.steps >= self.max_steps:
            self.status = "control_cap_stop"
        elif self.status == "interrupted":
            if self.recoveries >= 1:
                raise ValueError("Recovery cap exhausted")
            self.recoveries += 1
            self.status = "executing"
        else:
            self.status = "executing"
        return self.status
