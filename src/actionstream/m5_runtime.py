"""M5 scene-validity runtime primitives with action-level provenance.

This module intentionally leaves the frozen M4 runtime untouched.  It reuses
the M4 request, result, and worker types while adding the generation and
provenance needed by the M5 oracle scene-shift experiment.
"""

from __future__ import annotations

import copy
import math
import time
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import numpy as np

from .runtime import (
    InferenceRequest,
    InferenceResult,
    LatestRequestWorker,
    QueueNotReady,
    _rotation_geodesic_radians,
)


ActionSourceKind = Literal["policy", "queue_hold", "gate_hold"]
DiscardReason = Literal[
    "stale_prefix",
    "fully_stale",
    "queue_replaced",
    "scene_invalidation",
    "old_episode",
    "late_generation",
    "future_generation",
    "episode_ended",
]


def _freeze_value(value: Any) -> Any:
    """Take an independent, read-only snapshot suitable for provenance."""

    if isinstance(value, np.ndarray):
        frozen = np.asarray(value).copy()
        frozen.setflags(write=False)
        return frozen
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_value(item) for item in value)
    return copy.deepcopy(value)


def _thaw_value(value: Any) -> Any:
    """Convert an immutable provenance snapshot to JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_value(item) for item in value]
    if isinstance(value, frozenset | set):
        return sorted((_thaw_value(item) for item in value), key=repr)
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)


def _immutable_action(action: np.ndarray) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.shape != (7,):
        raise ValueError(
            f"An M5 action must be one finite final [7] command, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("An M5 action contains non-finite values")
    immutable = array.copy()
    immutable.setflags(write=False)
    return immutable


def _require_nonnegative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, kw_only=True)
class M5InferenceRequest(InferenceRequest):
    """Frozen M4-compatible request carrying M5 scene provenance."""

    request_id: str
    request_generation_id: int
    condition: str
    task_id: int | str
    seed: int
    world_epoch_at_observation: int
    entity_pose_at_observation: Mapping[str, Any]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.condition:
            raise ValueError("condition must be non-empty")
        _require_nonnegative("request_generation_id", self.request_generation_id)
        _require_nonnegative(
            "world_epoch_at_observation", self.world_epoch_at_observation
        )
        _require_finite("request_timestamp", self.request_timestamp)
        if not isinstance(self.entity_pose_at_observation, Mapping):
            raise TypeError("entity_pose_at_observation must be a mapping")
        object.__setattr__(
            self,
            "entity_pose_at_observation",
            _freeze_value(self.entity_pose_at_observation),
        )

    @property
    def observation_step(self) -> int:
        return self.observation_control_step

    @property
    def observation_timestamp(self) -> float:
        return self.request_timestamp

    @property
    def selected_entity_pose_at_observation(self) -> Mapping[str, Any]:
        return self.entity_pose_at_observation

    def provenance_dict(self) -> dict[str, Any]:
        """Return the request fields needed to join worker results to a request."""

        return {
            "episode_id": self.episode_id,
            "condition": self.condition,
            "task_id": self.task_id,
            "task_instruction": self.task_instruction,
            "seed": self.seed,
            "request_id": self.request_id,
            "request_generation_id": self.request_generation_id,
            "observation_control_step": self.observation_control_step,
            "observation_timestamp": self.request_timestamp,
            "world_epoch_at_observation": self.world_epoch_at_observation,
            "entity_pose_at_observation": _thaw_value(self.entity_pose_at_observation),
        }


@dataclass(frozen=True)
class GenerationDecision:
    """Pure result of comparing one result generation with the live generation."""

    accepted: bool
    reason: Literal["current_generation", "late_generation", "future_generation"]
    request_generation_id: int
    current_request_generation_id: int

    def __bool__(self) -> bool:
        return self.accepted


def filter_result_generation(
    request_generation_id: int,
    current_request_generation_id: int,
) -> GenerationDecision:
    """Accept only the current request generation.

    Deliberately, this helper has no world-epoch, pose, or perturbation input.
    It handles only asynchronous request-generation races.
    """

    _require_nonnegative("request_generation_id", request_generation_id)
    _require_nonnegative("current_request_generation_id", current_request_generation_id)
    if request_generation_id < current_request_generation_id:
        return GenerationDecision(
            accepted=False,
            reason="late_generation",
            request_generation_id=request_generation_id,
            current_request_generation_id=current_request_generation_id,
        )
    if request_generation_id > current_request_generation_id:
        return GenerationDecision(
            accepted=False,
            reason="future_generation",
            request_generation_id=request_generation_id,
            current_request_generation_id=current_request_generation_id,
        )
    return GenerationDecision(
        accepted=True,
        reason="current_generation",
        request_generation_id=request_generation_id,
        current_request_generation_id=current_request_generation_id,
    )


class GenerationFilter:
    """Small stateful wrapper around :func:`filter_result_generation`."""

    def __init__(self, current_request_generation_id: int = 0) -> None:
        self.accepted_results = 0
        self.late_results_rejected = 0
        self.future_results_rejected = 0
        self.reset(current_request_generation_id)

    @property
    def current_request_generation_id(self) -> int:
        return self._current_request_generation_id

    @property
    def current_generation_id(self) -> int:
        return self._current_request_generation_id

    def reset(self, current_request_generation_id: int = 0) -> None:
        _require_nonnegative(
            "current_request_generation_id", current_request_generation_id
        )
        self._current_request_generation_id = current_request_generation_id
        self.accepted_results = 0
        self.late_results_rejected = 0
        self.future_results_rejected = 0

    def advance(self, next_request_generation_id: int | None = None) -> int:
        if next_request_generation_id is None:
            next_request_generation_id = self._current_request_generation_id + 1
        if next_request_generation_id <= self._current_request_generation_id:
            raise ValueError(
                "next_request_generation_id must be greater than the live generation"
            )
        self._current_request_generation_id = next_request_generation_id
        return self._current_request_generation_id

    def evaluate(self, request_generation_id: int) -> GenerationDecision:
        decision = filter_result_generation(
            request_generation_id,
            self._current_request_generation_id,
        )
        if decision.accepted:
            self.accepted_results += 1
        elif decision.reason == "late_generation":
            self.late_results_rejected += 1
        else:
            self.future_results_rejected += 1
        return decision

    def accepts(self, request_generation_id: int) -> bool:
        return bool(self.evaluate(request_generation_id))


RequestGenerationFilter = GenerationFilter


@dataclass(frozen=True, kw_only=True)
class ActionProvenance:
    """Chunk-action provenance before or after queue insertion."""

    action: np.ndarray
    episode_id: str
    condition: str
    task_id: int | str
    task_instruction: str
    seed: int
    request_id: str
    request_generation_id: int
    observation_control_step: int
    observation_timestamp: float
    world_epoch_at_observation: int
    entity_pose_at_observation: Mapping[str, Any]
    policy_result_arrival_step: int
    policy_result_arrival_timestamp: float
    original_chunk_action_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _immutable_action(self.action))
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if not self.condition:
            raise ValueError("condition must be non-empty")
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        for name, value in (
            ("request_generation_id", self.request_generation_id),
            ("observation_control_step", self.observation_control_step),
            ("world_epoch_at_observation", self.world_epoch_at_observation),
            ("policy_result_arrival_step", self.policy_result_arrival_step),
            ("original_chunk_action_index", self.original_chunk_action_index),
        ):
            _require_nonnegative(name, value)
        _require_finite("observation_timestamp", self.observation_timestamp)
        _require_finite(
            "policy_result_arrival_timestamp",
            self.policy_result_arrival_timestamp,
        )
        if not isinstance(self.entity_pose_at_observation, Mapping):
            raise TypeError("entity_pose_at_observation must be a mapping")
        object.__setattr__(
            self,
            "entity_pose_at_observation",
            _freeze_value(self.entity_pose_at_observation),
        )

    @property
    def command(self) -> np.ndarray:
        return self.action.copy()

    @property
    def observation_step(self) -> int:
        return self.observation_control_step

    @property
    def result_arrival_step(self) -> int:
        return self.policy_result_arrival_step

    @property
    def result_arrival_timestamp(self) -> float:
        return self.policy_result_arrival_timestamp

    @property
    def chunk_action_index(self) -> int:
        return self.original_chunk_action_index

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.tolist(),
            "episode_id": self.episode_id,
            "condition": self.condition,
            "task_id": self.task_id,
            "task_instruction": self.task_instruction,
            "seed": self.seed,
            "request_id": self.request_id,
            "request_generation_id": self.request_generation_id,
            "observation_control_step": self.observation_control_step,
            "observation_step": self.observation_control_step,
            "observation_timestamp": self.observation_timestamp,
            "world_epoch_at_observation": self.world_epoch_at_observation,
            "entity_pose_at_observation": _thaw_value(self.entity_pose_at_observation),
            "policy_result_arrival_step": self.policy_result_arrival_step,
            "policy_result_arrival_timestamp": self.policy_result_arrival_timestamp,
            "result_arrival_step": self.policy_result_arrival_step,
            "result_arrival_timestamp": self.policy_result_arrival_timestamp,
            "original_chunk_action_index": self.original_chunk_action_index,
            "chunk_action_index": self.original_chunk_action_index,
            "queue_insertion_step": getattr(self, "queue_insertion_step", None),
        }


@dataclass(frozen=True, kw_only=True)
class QueuedAction(ActionProvenance):
    """One validated final command in the aligned queue."""

    queue_insertion_step: int

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_nonnegative("queue_insertion_step", self.queue_insertion_step)
        if np.allclose(self.action, 0.0):
            raise ValueError("Refusing to queue an all-zero absolute command")

    def to_dict(self) -> dict[str, Any]:
        record = super().to_dict()
        record["queue_insertion_step"] = self.queue_insertion_step
        return record


@dataclass(frozen=True, kw_only=True)
class ActionRecord:
    """One command actually sent to the environment."""

    provenance: ActionProvenance
    queue_execution_step: int
    execution_timestamp: float
    current_world_epoch_at_execution: int
    source_kind: ActionSourceKind
    held: bool
    queue_depth_before_action: int
    queue_depth_after_action: int
    gate_triggered: bool = False
    command_origin: ActionProvenance | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("queue_execution_step", self.queue_execution_step),
            (
                "current_world_epoch_at_execution",
                self.current_world_epoch_at_execution,
            ),
            ("queue_depth_before_action", self.queue_depth_before_action),
            ("queue_depth_after_action", self.queue_depth_after_action),
        ):
            _require_nonnegative(name, value)
        _require_finite("execution_timestamp", self.execution_timestamp)
        if self.source_kind not in {"policy", "queue_hold", "gate_hold"}:
            raise ValueError(f"Unsupported action source kind: {self.source_kind}")
        if (self.source_kind == "policy") == self.held:
            raise ValueError(
                "Policy actions must not be holds, and runtime holds must be held"
            )
        if self.queue_depth_after_action > self.queue_depth_before_action:
            raise ValueError("Executing an action cannot increase queue depth")
        if self.source_kind == "policy" and not isinstance(
            self.provenance, QueuedAction
        ):
            raise TypeError("A policy execution must refer to a QueuedAction")
        if self.source_kind != "policy" and self.command_origin is None:
            raise ValueError("A runtime hold must preserve its command origin")
        expected_stale = (
            self.provenance.world_epoch_at_observation
            < self.current_world_epoch_at_execution
        )
        if self.source_kind != "policy" and expected_stale:
            raise ValueError(
                "A deliberate runtime hold must be labeled at the live epoch"
            )

    @property
    def action(self) -> np.ndarray:
        return self.provenance.action.copy()

    @property
    def command(self) -> np.ndarray:
        return self.action

    @property
    def stale(self) -> bool:
        return (
            self.provenance.world_epoch_at_observation
            < self.current_world_epoch_at_execution
        )

    @property
    def discarded(self) -> bool:
        return False

    @property
    def discard_reason(self) -> None:
        return None

    def __iter__(self) -> Iterator[np.ndarray | bool]:
        """Allow ``action, held = queue.next_action(...)`` like the M4 queue."""

        yield self.action
        yield self.held

    def __getattr__(self, name: str) -> Any:
        provenance_names = {
            "episode_id",
            "condition",
            "task_id",
            "task_instruction",
            "seed",
            "request_id",
            "request_generation_id",
            "observation_control_step",
            "observation_step",
            "observation_timestamp",
            "world_epoch_at_observation",
            "entity_pose_at_observation",
            "policy_result_arrival_step",
            "policy_result_arrival_timestamp",
            "result_arrival_step",
            "result_arrival_timestamp",
            "original_chunk_action_index",
            "chunk_action_index",
            "queue_insertion_step",
        }
        if name in provenance_names:
            if name == "queue_insertion_step":
                return getattr(self.provenance, name, None)
            return getattr(self.provenance, name)
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        record = self.provenance.to_dict()
        record.update(
            {
                "record_type": "executed",
                "source_kind": self.source_kind,
                "held": self.held,
                "queue_execution_step": self.queue_execution_step,
                "execution_timestamp": self.execution_timestamp,
                "current_world_epoch_at_execution": (
                    self.current_world_epoch_at_execution
                ),
                "stale": self.stale,
                "discarded": False,
                "discard_reason": None,
                "gate_triggered": self.gate_triggered,
                "queue_depth_before_action": self.queue_depth_before_action,
                "queue_depth_after_action": self.queue_depth_after_action,
                "queue_depth_before_invalidation": None,
                "queue_depth_after_invalidation": None,
            }
        )
        record["command_origin"] = (
            None if self.command_origin is None else self.command_origin.to_dict()
        )
        return record


ExecutedActionRecord = ActionRecord


@dataclass(frozen=True, kw_only=True)
class DiscardRecord:
    """Complete provenance for one action that was never executed."""

    provenance: ActionProvenance
    discard_step: int
    discard_timestamp: float
    current_world_epoch_at_discard: int
    discard_reason: DiscardReason
    gate_triggered: bool
    queue_depth_before_invalidation: int
    queue_depth_after_invalidation: int

    def __post_init__(self) -> None:
        for name, value in (
            ("discard_step", self.discard_step),
            ("current_world_epoch_at_discard", self.current_world_epoch_at_discard),
            (
                "queue_depth_before_invalidation",
                self.queue_depth_before_invalidation,
            ),
            (
                "queue_depth_after_invalidation",
                self.queue_depth_after_invalidation,
            ),
        ):
            _require_nonnegative(name, value)
        _require_finite("discard_timestamp", self.discard_timestamp)
        if self.queue_depth_after_invalidation > self.queue_depth_before_invalidation:
            raise ValueError("Discarding actions cannot increase queue depth")

    @property
    def action(self) -> np.ndarray:
        return self.provenance.action.copy()

    @property
    def command(self) -> np.ndarray:
        return self.action

    @property
    def discarded(self) -> bool:
        return True

    @property
    def stale(self) -> bool:
        return (
            self.provenance.world_epoch_at_observation
            < self.current_world_epoch_at_discard
        )

    @property
    def was_queued(self) -> bool:
        return isinstance(self.provenance, QueuedAction)

    @property
    def queue_execution_step(self) -> None:
        return None

    @property
    def current_world_epoch_at_execution(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        provenance_names = {
            "episode_id",
            "condition",
            "task_id",
            "task_instruction",
            "seed",
            "request_id",
            "request_generation_id",
            "observation_control_step",
            "observation_step",
            "observation_timestamp",
            "world_epoch_at_observation",
            "entity_pose_at_observation",
            "policy_result_arrival_step",
            "policy_result_arrival_timestamp",
            "result_arrival_step",
            "result_arrival_timestamp",
            "original_chunk_action_index",
            "chunk_action_index",
            "queue_insertion_step",
        }
        if name in provenance_names:
            if name == "queue_insertion_step":
                return getattr(self.provenance, name, None)
            return getattr(self.provenance, name)
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        record = self.provenance.to_dict()
        record.update(
            {
                "record_type": "discarded",
                "source_kind": "discarded",
                "held": False,
                "queue_execution_step": None,
                "execution_timestamp": None,
                "current_world_epoch_at_execution": None,
                "stale": self.stale,
                "discarded": True,
                "discard_reason": self.discard_reason,
                "discard_step": self.discard_step,
                "discard_timestamp": self.discard_timestamp,
                "current_world_epoch_at_discard": (self.current_world_epoch_at_discard),
                "gate_triggered": self.gate_triggered,
                "was_queued": self.was_queued,
                "queue_depth_before_action": None,
                "queue_depth_after_action": None,
                "queue_depth_before_invalidation": (
                    self.queue_depth_before_invalidation
                ),
                "queue_depth_after_invalidation": (self.queue_depth_after_invalidation),
                "command_origin": None,
            }
        )
        return record


ActionEventRecord: TypeAlias = ActionRecord | DiscardRecord


@dataclass(frozen=True)
class M5MergeOutcome:
    """M4-compatible merge metrics plus complete per-action discard records."""

    accepted: bool
    reason: str
    age_steps: int
    dropped_prefix_steps: int
    fully_stale: bool
    queue_length: int
    incoming_chunk_steps: int
    stale_fraction: float
    replacement_position_l2: float | None
    replacement_rotation_geodesic_radians: float | None
    replacement_gripper_switch: bool | None
    request_id: str
    request_generation_id: int
    current_request_generation_id: int
    discard_records: tuple[DiscardRecord, ...] = field(default_factory=tuple)
    inserted_actions: tuple[QueuedAction, ...] = field(default_factory=tuple)

    @property
    def discarded_actions(self) -> tuple[DiscardRecord, ...]:
        return self.discard_records

    @property
    def accepted_actions(self) -> tuple[QueuedAction, ...]:
        return self.inserted_actions


@dataclass(frozen=True)
class InvalidationOutcome(Sequence[DiscardRecord]):
    """One explicit scene invalidation and every queued action it removed."""

    discarded_actions: tuple[DiscardRecord, ...]
    queue_depth_before: int
    queue_depth_after: int
    previous_request_generation_id: int
    current_request_generation_id: int
    invalidation_step: int
    invalidation_timestamp: float
    current_world_epoch: int
    discard_reason: str
    gate_triggered: bool
    last_safe_absolute_command_preserved: bool

    def __len__(self) -> int:
        return len(self.discarded_actions)

    def __getitem__(
        self, index: int | slice
    ) -> DiscardRecord | tuple[DiscardRecord, ...]:
        return self.discarded_actions[index]

    def __iter__(self) -> Iterator[DiscardRecord]:
        return iter(self.discarded_actions)


class M5AlignedActionQueue:
    """Aligned M5 queue with scene invalidation and complete provenance."""

    def __init__(self) -> None:
        self._queue: deque[QueuedAction] = deque()
        self._episode_id: str | None = None
        self._last_action: np.ndarray | None = None
        self._last_command_origin: ActionProvenance | None = None
        self._last_execution_step: int | None = None
        self._last_execution_world_epoch: int | None = None
        self._gate_hold_active = False
        self._generation_filter = GenerationFilter()
        self._action_records: list[ActionEventRecord] = []
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self.hold_steps = 0
        self.queue_hold_steps = 0
        self.gate_hold_steps = 0
        self.stale_chunks_discarded = 0
        self.old_episode_chunks_rejected = 0
        self.replacements = 0
        self.accepted_chunks = 0
        self.rejected_chunks = 0
        self.stale_actions_discarded = 0
        self.incoming_actions = 0
        self.returned_actions = 0
        self.stale_prefix_lengths: list[int] = []
        self.stale_policy_results_discarded = 0
        self.future_policy_results_discarded = 0
        self.stale_queued_actions_discarded = 0
        self.queue_invalidation_count = 0
        self.fresh_replan_count = 0
        self.maximum_queue_depth = 0

    def reset_episode(
        self,
        episode_id: str,
        *,
        request_generation_id: int = 0,
    ) -> None:
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        self._episode_id = episode_id
        self._queue.clear()
        self._last_action = None
        self._last_command_origin = None
        self._last_execution_step = None
        self._last_execution_world_epoch = None
        self._gate_hold_active = False
        self._generation_filter.reset(request_generation_id)
        self._action_records.clear()
        self._reset_metrics()

    @property
    def queue_length(self) -> int:
        return len(self._queue)

    @property
    def has_safe_action(self) -> bool:
        return bool(self._queue) or self._last_action is not None

    @property
    def current_request_generation_id(self) -> int:
        return self._generation_filter.current_request_generation_id

    @property
    def generation_filter(self) -> GenerationFilter:
        return self._generation_filter

    @property
    def action_records(self) -> tuple[ActionEventRecord, ...]:
        return tuple(self._action_records)

    @property
    def queued_actions(self) -> tuple[QueuedAction, ...]:
        return tuple(self._queue)

    def advance_generation(self, next_request_generation_id: int | None = None) -> int:
        return self._generation_filter.advance(next_request_generation_id)

    @staticmethod
    def _validate_result_request(
        result: InferenceResult,
        request: M5InferenceRequest,
    ) -> None:
        if result.episode_id != request.episode_id:
            raise ValueError(
                "InferenceResult and M5InferenceRequest episode IDs differ"
            )
        if result.observation_control_step != request.observation_control_step:
            raise ValueError(
                "InferenceResult and M5InferenceRequest observation steps differ"
            )
        if not math.isclose(
            result.request_timestamp,
            request.request_timestamp,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "InferenceResult and M5InferenceRequest request timestamps differ"
            )

    @staticmethod
    def _incoming_provenance(
        result: InferenceResult,
        request: M5InferenceRequest,
        *,
        action_index: int,
        result_arrival_step: int,
        result_arrival_timestamp: float,
    ) -> ActionProvenance:
        return ActionProvenance(
            action=result.actions[action_index],
            episode_id=request.episode_id,
            condition=request.condition,
            task_id=request.task_id,
            task_instruction=request.task_instruction,
            seed=request.seed,
            request_id=request.request_id,
            request_generation_id=request.request_generation_id,
            observation_control_step=request.observation_control_step,
            observation_timestamp=request.request_timestamp,
            world_epoch_at_observation=request.world_epoch_at_observation,
            entity_pose_at_observation=request.entity_pose_at_observation,
            policy_result_arrival_step=result_arrival_step,
            policy_result_arrival_timestamp=result_arrival_timestamp,
            original_chunk_action_index=action_index,
        )

    @staticmethod
    def _as_queued(
        provenance: ActionProvenance,
        *,
        queue_insertion_step: int,
    ) -> QueuedAction:
        return QueuedAction(
            action=provenance.action,
            episode_id=provenance.episode_id,
            condition=provenance.condition,
            task_id=provenance.task_id,
            task_instruction=provenance.task_instruction,
            seed=provenance.seed,
            request_id=provenance.request_id,
            request_generation_id=provenance.request_generation_id,
            observation_control_step=provenance.observation_control_step,
            observation_timestamp=provenance.observation_timestamp,
            world_epoch_at_observation=provenance.world_epoch_at_observation,
            entity_pose_at_observation=provenance.entity_pose_at_observation,
            policy_result_arrival_step=provenance.policy_result_arrival_step,
            policy_result_arrival_timestamp=(
                provenance.policy_result_arrival_timestamp
            ),
            original_chunk_action_index=provenance.original_chunk_action_index,
            queue_insertion_step=queue_insertion_step,
        )

    def _discard(
        self,
        provenance: ActionProvenance,
        *,
        discard_step: int,
        discard_timestamp: float,
        current_world_epoch: int,
        reason: DiscardReason,
        gate_triggered: bool,
        queue_depth_before: int,
        queue_depth_after: int,
    ) -> DiscardRecord:
        record = DiscardRecord(
            provenance=provenance,
            discard_step=discard_step,
            discard_timestamp=discard_timestamp,
            current_world_epoch_at_discard=current_world_epoch,
            discard_reason=reason,
            gate_triggered=gate_triggered,
            queue_depth_before_invalidation=queue_depth_before,
            queue_depth_after_invalidation=queue_depth_after,
        )
        self._action_records.append(record)
        return record

    def _outcome(
        self,
        *,
        accepted: bool,
        reason: str,
        age_steps: int,
        dropped_prefix_steps: int,
        fully_stale: bool,
        incoming_chunk_steps: int,
        stale_fraction: float,
        request: M5InferenceRequest,
        discard_records: Iterable[DiscardRecord] = (),
        inserted_actions: Iterable[QueuedAction] = (),
        replacement_position_l2: float | None = None,
        replacement_rotation_geodesic_radians: float | None = None,
        replacement_gripper_switch: bool | None = None,
    ) -> M5MergeOutcome:
        return M5MergeOutcome(
            accepted=accepted,
            reason=reason,
            age_steps=age_steps,
            dropped_prefix_steps=dropped_prefix_steps,
            fully_stale=fully_stale,
            queue_length=len(self._queue),
            incoming_chunk_steps=incoming_chunk_steps,
            stale_fraction=stale_fraction,
            replacement_position_l2=replacement_position_l2,
            replacement_rotation_geodesic_radians=(
                replacement_rotation_geodesic_radians
            ),
            replacement_gripper_switch=replacement_gripper_switch,
            request_id=request.request_id,
            request_generation_id=request.request_generation_id,
            current_request_generation_id=self.current_request_generation_id,
            discard_records=tuple(discard_records),
            inserted_actions=tuple(inserted_actions),
        )

    def replace(
        self,
        result: InferenceResult,
        *,
        request: M5InferenceRequest,
        current_control_step: int,
        result_arrival_step: int | None = None,
        result_arrival_timestamp: float | None = None,
        current_world_epoch: int | None = None,
    ) -> M5MergeOutcome:
        """Merge a current-generation result using the exact M4 aligned rule."""

        if self._episode_id is None:
            raise RuntimeError("Reset the M5 action queue before merging a chunk")
        _require_nonnegative("current_control_step", current_control_step)
        self._validate_result_request(result, request)
        if result_arrival_step is None:
            result_arrival_step = current_control_step
        if result_arrival_timestamp is None:
            result_arrival_timestamp = result.delivery_timestamp
        if current_world_epoch is None:
            current_world_epoch = request.world_epoch_at_observation
        _require_nonnegative("result_arrival_step", result_arrival_step)
        _require_nonnegative("current_world_epoch", current_world_epoch)
        _require_finite("result_arrival_timestamp", result_arrival_timestamp)
        if result_arrival_step < request.observation_control_step:
            raise ValueError("A result cannot arrive before its observation step")

        incoming = tuple(
            self._incoming_provenance(
                result,
                request,
                action_index=index,
                result_arrival_step=result_arrival_step,
                result_arrival_timestamp=result_arrival_timestamp,
            )
            for index in range(len(result.actions))
        )
        self.returned_actions += len(incoming)
        age_steps = current_control_step - result.observation_control_step

        if result.episode_id != self._episode_id:
            self.old_episode_chunks_rejected += 1
            self.rejected_chunks += 1
            depth = len(self._queue)
            records = tuple(
                self._discard(
                    action,
                    discard_step=current_control_step,
                    discard_timestamp=result_arrival_timestamp,
                    current_world_epoch=current_world_epoch,
                    reason="old_episode",
                    gate_triggered=False,
                    queue_depth_before=depth,
                    queue_depth_after=depth,
                )
                for action in incoming
            )
            return self._outcome(
                accepted=False,
                reason="old_episode",
                age_steps=max(0, age_steps),
                dropped_prefix_steps=0,
                fully_stale=False,
                incoming_chunk_steps=len(incoming),
                stale_fraction=0.0,
                request=request,
                discard_records=records,
            )

        if age_steps < 0:
            raise ValueError(
                "A result cannot be newer than the current control step: "
                f"current={current_control_step}, "
                f"observation={result.observation_control_step}"
            )

        generation = self._generation_filter.evaluate(request.request_generation_id)
        if not generation.accepted:
            self.rejected_chunks += 1
            if generation.reason == "late_generation":
                self.stale_policy_results_discarded += 1
            else:
                self.future_policy_results_discarded += 1
            depth = len(self._queue)
            records = tuple(
                self._discard(
                    action,
                    discard_step=current_control_step,
                    discard_timestamp=result_arrival_timestamp,
                    current_world_epoch=current_world_epoch,
                    reason=generation.reason,
                    gate_triggered=False,
                    queue_depth_before=depth,
                    queue_depth_after=depth,
                )
                for action in incoming
            )
            return self._outcome(
                accepted=False,
                reason=generation.reason,
                age_steps=age_steps,
                dropped_prefix_steps=0,
                fully_stale=False,
                incoming_chunk_steps=len(incoming),
                stale_fraction=0.0,
                request=request,
                discard_records=records,
            )

        self.incoming_actions += len(incoming)
        drop = age_steps
        bounded_drop = min(drop, len(incoming))
        stale_fraction = bounded_drop / len(incoming)
        self.stale_prefix_lengths.append(bounded_drop)
        self.stale_actions_discarded += bounded_drop
        depth_before_merge = len(self._queue)

        if drop >= len(incoming):
            self.stale_chunks_discarded += 1
            self.rejected_chunks += 1
            records = tuple(
                self._discard(
                    action,
                    discard_step=current_control_step,
                    discard_timestamp=result_arrival_timestamp,
                    current_world_epoch=current_world_epoch,
                    reason="fully_stale",
                    gate_triggered=False,
                    queue_depth_before=depth_before_merge,
                    queue_depth_after=depth_before_merge,
                )
                for action in incoming
            )
            return self._outcome(
                accepted=False,
                reason="fully_stale",
                age_steps=age_steps,
                dropped_prefix_steps=bounded_drop,
                fully_stale=True,
                incoming_chunk_steps=len(incoming),
                stale_fraction=stale_fraction,
                request=request,
                discard_records=records,
            )

        prefix_discards = tuple(
            self._discard(
                incoming[index],
                discard_step=current_control_step,
                discard_timestamp=result_arrival_timestamp,
                current_world_epoch=current_world_epoch,
                reason="stale_prefix",
                gate_triggered=False,
                queue_depth_before=depth_before_merge,
                queue_depth_after=depth_before_merge,
            )
            for index in range(drop)
        )
        replacement = tuple(
            self._as_queued(
                incoming[index],
                queue_insertion_step=current_control_step,
            )
            for index in range(drop, len(incoming))
        )

        replacement_position_l2: float | None = None
        replacement_rotation_geodesic_radians: float | None = None
        replacement_gripper_switch: bool | None = None
        if self._last_action is not None:
            first = replacement[0].action
            replacement_position_l2 = float(
                np.linalg.norm(first[:3] - self._last_action[:3])
            )
            replacement_rotation_geodesic_radians = _rotation_geodesic_radians(
                self._last_action[3:6],
                first[3:6],
            )
            replacement_gripper_switch = bool(
                (self._last_action[6] >= 0.0) != (first[6] >= 0.0)
            )

        replaced_discards = tuple(
            self._discard(
                queued,
                discard_step=current_control_step,
                discard_timestamp=result_arrival_timestamp,
                current_world_epoch=current_world_epoch,
                reason="queue_replaced",
                gate_triggered=False,
                queue_depth_before=depth_before_merge,
                queue_depth_after=0,
            )
            for queued in self._queue
        )
        self._queue.clear()
        self._queue.extend(replacement)
        self.maximum_queue_depth = max(self.maximum_queue_depth, len(self._queue))
        self.replacements += 1
        self.accepted_chunks += 1
        if self._gate_hold_active:
            self.fresh_replan_count += 1
            self._gate_hold_active = False
        return self._outcome(
            accepted=True,
            reason="replaced",
            age_steps=age_steps,
            dropped_prefix_steps=bounded_drop,
            fully_stale=False,
            incoming_chunk_steps=len(incoming),
            stale_fraction=stale_fraction,
            request=request,
            discard_records=prefix_discards + replaced_discards,
            inserted_actions=replacement,
            replacement_position_l2=replacement_position_l2,
            replacement_rotation_geodesic_radians=(
                replacement_rotation_geodesic_radians
            ),
            replacement_gripper_switch=replacement_gripper_switch,
        )

    def invalidate_scene(
        self,
        *,
        current_control_step: int,
        current_world_epoch: int,
        invalidation_timestamp: float | None = None,
        next_request_generation_id: int | None = None,
        discard_reason: Literal["scene_invalidation"] = "scene_invalidation",
        gate_triggered: bool = True,
    ) -> InvalidationOutcome:
        """Clear old-scene queued actions and preserve the last safe command."""

        if self._episode_id is None:
            raise RuntimeError("Reset the M5 action queue before invalidating it")
        _require_nonnegative("current_control_step", current_control_step)
        _require_nonnegative("current_world_epoch", current_world_epoch)
        if invalidation_timestamp is None:
            invalidation_timestamp = time.monotonic()
        _require_finite("invalidation_timestamp", invalidation_timestamp)

        depth_before = len(self._queue)
        previous_generation = self.current_request_generation_id
        current_generation = self._generation_filter.advance(next_request_generation_id)
        records = tuple(
            self._discard(
                queued,
                discard_step=current_control_step,
                discard_timestamp=invalidation_timestamp,
                current_world_epoch=current_world_epoch,
                reason=discard_reason,
                gate_triggered=gate_triggered,
                queue_depth_before=depth_before,
                queue_depth_after=0,
            )
            for queued in self._queue
        )
        self._queue.clear()
        self.stale_queued_actions_discarded += sum(record.stale for record in records)
        self.queue_invalidation_count += 1
        self._gate_hold_active = True
        return InvalidationOutcome(
            discarded_actions=records,
            queue_depth_before=depth_before,
            queue_depth_after=0,
            previous_request_generation_id=previous_generation,
            current_request_generation_id=current_generation,
            invalidation_step=current_control_step,
            invalidation_timestamp=invalidation_timestamp,
            current_world_epoch=current_world_epoch,
            discard_reason=discard_reason,
            gate_triggered=gate_triggered,
            last_safe_absolute_command_preserved=self._last_action is not None,
        )

    def discard_remaining_at_episode_end(
        self,
        *,
        current_control_step: int,
        current_world_epoch: int,
        discard_timestamp: float | None = None,
    ) -> tuple[DiscardRecord, ...]:
        """Record every still-queued command before closing an episode.

        This is provenance finalization, not a scene invalidation: it does not
        advance the request generation or increment gate metrics.
        """

        if self._episode_id is None:
            raise RuntimeError("Reset the M5 action queue before finalizing it")
        _require_nonnegative("current_control_step", current_control_step)
        _require_nonnegative("current_world_epoch", current_world_epoch)
        if discard_timestamp is None:
            discard_timestamp = time.monotonic()
        _require_finite("discard_timestamp", discard_timestamp)
        depth_before = len(self._queue)
        records = tuple(
            self._discard(
                queued,
                discard_step=current_control_step,
                discard_timestamp=discard_timestamp,
                current_world_epoch=current_world_epoch,
                reason="episode_ended",
                gate_triggered=False,
                queue_depth_before=depth_before,
                queue_depth_after=0,
            )
            for queued in self._queue
        )
        self._queue.clear()
        return records

    def next_action(
        self,
        *,
        current_control_step: int | None = None,
        current_world_epoch: int | None = None,
        execution_timestamp: float | None = None,
        current_entity_pose: Mapping[str, Any] | None = None,
        hold_source_kind: Literal["queue_hold", "gate_hold"] | None = None,
    ) -> ActionRecord:
        """Execute one queued policy action or repeat the last safe command.

        Runtime holds are explicitly labeled at ``current_world_epoch``.  Their
        ``command_origin`` preserves the policy action whose absolute command is
        repeated, while they are not counted as old-scene policy executions.
        """

        if self._episode_id is None:
            raise RuntimeError("Reset the M5 action queue before executing it")
        if execution_timestamp is None:
            execution_timestamp = time.monotonic()
        _require_finite("execution_timestamp", execution_timestamp)
        if current_control_step is None:
            current_control_step = (
                0
                if self._last_execution_step is None
                else self._last_execution_step + 1
            )
        _require_nonnegative("current_control_step", current_control_step)
        if (
            self._last_execution_step is not None
            and current_control_step <= self._last_execution_step
        ):
            raise ValueError("queue_execution_step must increase monotonically")

        depth_before = len(self._queue)
        if self._queue:
            queued = self._queue.popleft()
            if current_world_epoch is None:
                current_world_epoch = queued.world_epoch_at_observation
            _require_nonnegative("current_world_epoch", current_world_epoch)
            record = ActionRecord(
                provenance=queued,
                queue_execution_step=current_control_step,
                execution_timestamp=execution_timestamp,
                current_world_epoch_at_execution=current_world_epoch,
                source_kind="policy",
                held=False,
                queue_depth_before_action=depth_before,
                queue_depth_after_action=len(self._queue),
                gate_triggered=False,
            )
            self._last_action = queued.action.copy()
            self._last_command_origin = queued
        else:
            if self._last_action is None or self._last_command_origin is None:
                raise QueueNotReady(
                    "Wait for the first chunk instead of stepping with an unsafe command"
                )
            if np.allclose(self._last_action, 0.0):
                raise RuntimeError("Refusing to use an all-zero absolute hold command")
            if current_world_epoch is None:
                current_world_epoch = (
                    self._last_execution_world_epoch
                    if self._last_execution_world_epoch is not None
                    else self._last_command_origin.world_epoch_at_observation
                )
            _require_nonnegative("current_world_epoch", current_world_epoch)
            source_kind = hold_source_kind
            if source_kind is None:
                source_kind = "gate_hold" if self._gate_hold_active else "queue_hold"
            pose = (
                self._last_command_origin.entity_pose_at_observation
                if current_entity_pose is None
                else current_entity_pose
            )
            hold_provenance = ActionProvenance(
                action=self._last_action,
                episode_id=self._last_command_origin.episode_id,
                condition=self._last_command_origin.condition,
                task_id=self._last_command_origin.task_id,
                task_instruction=self._last_command_origin.task_instruction,
                seed=self._last_command_origin.seed,
                request_id=self._last_command_origin.request_id,
                request_generation_id=self.current_request_generation_id,
                observation_control_step=current_control_step,
                observation_timestamp=execution_timestamp,
                world_epoch_at_observation=current_world_epoch,
                entity_pose_at_observation=pose,
                policy_result_arrival_step=(
                    self._last_command_origin.policy_result_arrival_step
                ),
                policy_result_arrival_timestamp=(
                    self._last_command_origin.policy_result_arrival_timestamp
                ),
                original_chunk_action_index=(
                    self._last_command_origin.original_chunk_action_index
                ),
            )
            record = ActionRecord(
                provenance=hold_provenance,
                queue_execution_step=current_control_step,
                execution_timestamp=execution_timestamp,
                current_world_epoch_at_execution=current_world_epoch,
                source_kind=source_kind,
                held=True,
                queue_depth_before_action=0,
                queue_depth_after_action=0,
                gate_triggered=source_kind == "gate_hold",
                command_origin=self._last_command_origin,
            )
            self.hold_steps += 1
            if source_kind == "gate_hold":
                self.gate_hold_steps += 1
            else:
                self.queue_hold_steps += 1

        self._last_execution_step = current_control_step
        self._last_execution_world_epoch = current_world_epoch
        self._action_records.append(record)
        return record

    def safe_hold(
        self,
        *,
        current_control_step: int,
        current_world_epoch: int,
        execution_timestamp: float | None = None,
        current_entity_pose: Mapping[str, Any] | None = None,
    ) -> ActionRecord:
        """Explicit gate-hold convenience wrapper."""

        if self._queue:
            raise RuntimeError(
                "safe_hold requires the old queue to be invalidated first"
            )
        return self.next_action(
            current_control_step=current_control_step,
            current_world_epoch=current_world_epoch,
            execution_timestamp=execution_timestamp,
            current_entity_pose=current_entity_pose,
            hold_source_kind="gate_hold",
        )


class M5LatestRequestWorker(LatestRequestWorker):
    """M5 worker with auditable coalescing/cancellation identities.

    The base M4 worker behavior is preserved.  These additive return values let
    the M5 runner reconcile request provenance when its latest-only mailbox
    replaces or cancels a request before inference starts.
    """

    def submit(self, request: InferenceRequest) -> InferenceRequest | None:
        with self._condition:
            self._raise_if_failed_locked()
            if self._closed:
                raise RuntimeError("Inference worker is closed")
            if request.episode_id != self._episode_id:
                raise ValueError(
                    f"Request episode {request.episode_id!r} does not match active "
                    f"episode {self._episode_id!r}"
                )
            replaced = None if self._pending is None else self._pending[1]
            if replaced is not None:
                self.pending_requests_replaced += 1
            self._pending = (self._generation, request)
            self._condition.notify_all()
            return replaced

    def cancel_pending_request(self) -> InferenceRequest | None:
        """Cancel and return the exact mailbox request that had not started."""

        with self._condition:
            self._raise_if_failed_locked()
            cancelled = None if self._pending is None else self._pending[1]
            if cancelled is not None:
                self._pending = None
                self.pending_requests_cancelled += 1
                self._condition.notify_all()
            return cancelled


M5ActionQueue = M5AlignedActionQueue


def _record_mapping(record: ActionEventRecord | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(record, ActionRecord | DiscardRecord):
        return record.to_dict()
    if isinstance(record, Mapping):
        return record
    raise TypeError("Action records must be ActionRecord, DiscardRecord, or mappings")


def summarize_action_records(
    records: Iterable[ActionEventRecord | Mapping[str, Any]],
    control_frequency_hz: float,
) -> dict[str, Any]:
    """Purely summarize exact action staleness and hold/discard metrics."""

    if not math.isfinite(control_frequency_hz) or control_frequency_hz <= 0:
        raise ValueError("control_frequency_hz must be finite and positive")
    rows = [dict(_record_mapping(record)) for record in records]
    executed = [row for row in rows if not bool(row.get("discarded", False))]
    discarded = [row for row in rows if bool(row.get("discarded", False))]

    stale_rows: list[Mapping[str, Any]] = []
    source_ages: list[int] = []
    for row in executed:
        observation_epoch = int(row["world_epoch_at_observation"])
        execution_epoch = int(row["current_world_epoch_at_execution"])
        derived_stale = observation_epoch < execution_epoch
        if derived_stale:
            stale_rows.append(row)
        execution_step = int(row["queue_execution_step"])
        observation_step_value = row.get("observation_control_step")
        if observation_step_value is None:
            observation_step_value = row["observation_step"]
        observation_step = int(observation_step_value)
        source_ages.append(execution_step - observation_step)

    stale_steps = sorted(int(row["queue_execution_step"]) for row in stale_rows)
    reason_counts = Counter(
        str(row.get("discard_reason"))
        for row in discarded
        if row.get("discard_reason") is not None
    )
    stale_discarded = [
        row
        for row in discarded
        if int(row["world_epoch_at_observation"])
        < int(row["current_world_epoch_at_discard"])
        and (
            bool(row.get("was_queued", False))
            or row.get("queue_insertion_step") is not None
        )
    ]
    late_request_keys = {
        (
            str(row.get("episode_id")),
            str(row.get("request_id")),
            int(row.get("request_generation_id", -1)),
        )
        for row in discarded
        if row.get("discard_reason") == "late_generation"
    }
    queue_depths = [
        int(depth)
        for row in rows
        for depth in (
            row.get("queue_depth_before_action"),
            row.get("queue_depth_after_action"),
            row.get("queue_depth_before_invalidation"),
            row.get("queue_depth_after_invalidation"),
        )
        if depth is not None
    ]
    gate_holds = sum(row.get("source_kind") == "gate_hold" for row in executed)
    queue_holds = sum(row.get("source_kind") == "queue_hold" for row in executed)

    return {
        "action_record_count": len(rows),
        "executed_action_steps": len(executed),
        "discarded_action_steps": len(discarded),
        "stale_action_steps": len(stale_rows),
        "stale_action_duration_seconds": len(stale_rows) / control_frequency_hz,
        "first_stale_action_step": stale_steps[0] if stale_steps else None,
        "last_stale_action_step": stale_steps[-1] if stale_steps else None,
        "stale_queued_actions_discarded": len(stale_discarded),
        "stale_policy_results_discarded": len(late_request_keys),
        "hold_steps_introduced_by_gate": gate_holds,
        "gate_hold_steps": gate_holds,
        "queue_hold_steps": queue_holds,
        "maximum_queue_depth": max(queue_depths, default=0),
        "action_source_observation_age_steps_mean": (
            float(np.mean(source_ages)) if source_ages else None
        ),
        "action_source_observation_age_steps_median": (
            float(np.median(source_ages)) if source_ages else None
        ),
        "action_source_observation_age_steps_max": (
            max(source_ages) if source_ages else None
        ),
        "discard_reason_counts": dict(sorted(reason_counts.items())),
    }


__all__ = [
    "ActionEventRecord",
    "ActionProvenance",
    "ActionRecord",
    "ActionSourceKind",
    "DiscardReason",
    "DiscardRecord",
    "ExecutedActionRecord",
    "GenerationDecision",
    "GenerationFilter",
    "InferenceResult",
    "InvalidationOutcome",
    "LatestRequestWorker",
    "M5ActionQueue",
    "M5AlignedActionQueue",
    "M5InferenceRequest",
    "M5LatestRequestWorker",
    "M5MergeOutcome",
    "QueuedAction",
    "QueueNotReady",
    "RequestGenerationFilter",
    "filter_result_generation",
    "summarize_action_records",
]
