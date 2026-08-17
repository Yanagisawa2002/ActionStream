"""Minimal local async worker and final-action queue semantics."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


RuntimeMode = Literal["sync_hold", "async_naive", "async_aligned"]


class QueueNotReady(RuntimeError):
    """Raised before the first safe final command exists."""


@dataclass(frozen=True)
class InferenceRequest:
    observation: Mapping[str, Any]
    task_instruction: str
    episode_id: str
    observation_control_step: int
    request_timestamp: float
    queue_depth_at_request_steps: int | None = None
    queue_headroom_at_request_steps: int | None = None

    def __post_init__(self) -> None:
        if self.observation_control_step < 0:
            raise ValueError("observation_control_step must be non-negative")
        for name, value in (
            ("queue_depth_at_request_steps", self.queue_depth_at_request_steps),
            ("queue_headroom_at_request_steps", self.queue_headroom_at_request_steps),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class InferencePayload:
    actions: np.ndarray
    model_inference_latency_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceResult:
    actions: np.ndarray
    episode_id: str
    observation_control_step: int
    request_timestamp: float
    start_timestamp: float
    end_timestamp: float
    delivery_timestamp: float
    model_inference_latency_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    queue_depth_at_request_steps: int | None = None
    queue_headroom_at_request_steps: int | None = None

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Inference result must contain [T,7] final actions, got {actions.shape}")
        if actions.shape[0] == 0:
            raise ValueError("Inference result must contain at least one final action")
        if not np.isfinite(actions).all():
            raise ValueError("Inference result contains non-finite final actions")
        immutable = actions.copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "actions", immutable)

        if not (
            self.request_timestamp
            <= self.start_timestamp
            <= self.end_timestamp
            <= self.delivery_timestamp
        ):
            raise ValueError("Inference timestamps are not monotonic")
        for name, value in (
            ("queue_depth_at_request_steps", self.queue_depth_at_request_steps),
            ("queue_headroom_at_request_steps", self.queue_headroom_at_request_steps),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class MergeOutcome:
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


def _rotation_geodesic_radians(first: np.ndarray, second: np.ndarray) -> float:
    """Return the SO(3) geodesic angle between two axis-angle rotations."""

    def quaternion(axis_angle: np.ndarray) -> np.ndarray:
        vector = np.asarray(axis_angle, dtype=np.float64)
        angle = float(np.linalg.norm(vector))
        if angle <= 1e-12:
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        half_angle = angle / 2.0
        return np.concatenate(
            (
                np.asarray([np.cos(half_angle)], dtype=np.float64),
                vector / angle * np.sin(half_angle),
            )
        )

    first_quaternion = quaternion(first)
    second_quaternion = quaternion(second)
    cosine_half_angle = float(abs(np.dot(first_quaternion, second_quaternion)))
    return float(2.0 * np.arccos(np.clip(cosine_half_angle, 0.0, 1.0)))


class ActionQueue:
    """Queue only final finite 7D commands and implement the two merge rules."""

    def __init__(self) -> None:
        self._queue: deque[np.ndarray] = deque()
        self._episode_id: str | None = None
        self._last_action: np.ndarray | None = None
        self.hold_steps = 0
        self.stale_chunks_discarded = 0
        self.old_episode_chunks_rejected = 0
        self.replacements = 0
        self.accepted_chunks = 0
        self.rejected_chunks = 0
        self.stale_actions_discarded = 0
        self.incoming_actions = 0
        self.stale_prefix_lengths: list[int] = []

    def reset_episode(self, episode_id: str) -> None:
        self._episode_id = episode_id
        self._queue.clear()
        self._last_action = None
        self.hold_steps = 0
        self.stale_chunks_discarded = 0
        self.old_episode_chunks_rejected = 0
        self.replacements = 0
        self.accepted_chunks = 0
        self.rejected_chunks = 0
        self.stale_actions_discarded = 0
        self.incoming_actions = 0
        self.stale_prefix_lengths.clear()

    @property
    def queue_length(self) -> int:
        return len(self._queue)

    @property
    def has_safe_action(self) -> bool:
        return bool(self._queue) or self._last_action is not None

    def discard_pending_for_hold(self) -> int:
        """Drop queued commands while retaining only the last dispatched command."""

        discarded = len(self._queue)
        self._queue.clear()
        return discarded

    def replace(
        self,
        result: InferenceResult,
        *,
        current_control_step: int,
        mode: RuntimeMode,
    ) -> MergeOutcome:
        if self._episode_id is None:
            raise RuntimeError("Reset the action queue before merging a chunk")
        if mode not in {"sync_hold", "async_naive", "async_aligned"}:
            raise ValueError(f"Unsupported action-queue mode: {mode}")
        if result.episode_id != self._episode_id:
            self.old_episode_chunks_rejected += 1
            self.rejected_chunks += 1
            return MergeOutcome(
                accepted=False,
                reason="old_episode",
                age_steps=max(0, current_control_step - result.observation_control_step),
                dropped_prefix_steps=0,
                fully_stale=False,
                queue_length=len(self._queue),
                incoming_chunk_steps=len(result.actions),
                stale_fraction=0.0,
                replacement_position_l2=None,
                replacement_rotation_geodesic_radians=None,
                replacement_gripper_switch=None,
            )

        age_steps = current_control_step - result.observation_control_step
        if age_steps < 0:
            raise ValueError(
                "A result cannot be newer than the current control step: "
                f"current={current_control_step}, observation={result.observation_control_step}"
            )

        self.incoming_actions += len(result.actions)
        drop = age_steps if mode == "async_aligned" else 0
        bounded_drop = min(drop, len(result.actions))
        stale_fraction = bounded_drop / len(result.actions)
        self.stale_prefix_lengths.append(bounded_drop)
        self.stale_actions_discarded += bounded_drop
        if drop >= len(result.actions):
            self.stale_chunks_discarded += 1
            self.rejected_chunks += 1
            return MergeOutcome(
                accepted=False,
                reason="fully_stale",
                age_steps=age_steps,
                dropped_prefix_steps=bounded_drop,
                fully_stale=True,
                queue_length=len(self._queue),
                incoming_chunk_steps=len(result.actions),
                stale_fraction=stale_fraction,
                replacement_position_l2=None,
                replacement_rotation_geodesic_radians=None,
                replacement_gripper_switch=None,
            )

        replacement = result.actions[drop:]
        for action in replacement:
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Action queue accepts only finite final 7D commands")
            if np.allclose(action, 0.0):
                raise ValueError("Refusing to queue an all-zero absolute command")

        replacement_position_l2: float | None = None
        replacement_rotation_geodesic_radians: float | None = None
        replacement_gripper_switch: bool | None = None
        if self._last_action is not None:
            first_replacement = np.asarray(replacement[0], dtype=np.float32)
            replacement_position_l2 = float(
                np.linalg.norm(first_replacement[:3] - self._last_action[:3])
            )
            replacement_rotation_geodesic_radians = _rotation_geodesic_radians(
                self._last_action[3:6],
                first_replacement[3:6],
            )
            replacement_gripper_switch = bool(
                (self._last_action[6] >= 0.0) != (first_replacement[6] >= 0.0)
            )

        self._queue.clear()
        self._queue.extend(np.asarray(action, dtype=np.float32).copy() for action in replacement)
        self.replacements += 1
        self.accepted_chunks += 1
        return MergeOutcome(
            accepted=True,
            reason="replaced",
            age_steps=age_steps,
            dropped_prefix_steps=bounded_drop,
            fully_stale=False,
            queue_length=len(self._queue),
            incoming_chunk_steps=len(result.actions),
            stale_fraction=stale_fraction,
            replacement_position_l2=replacement_position_l2,
            replacement_rotation_geodesic_radians=replacement_rotation_geodesic_radians,
            replacement_gripper_switch=replacement_gripper_switch,
        )

    def next_action(self) -> tuple[np.ndarray, bool]:
        if self._queue:
            action = self._queue.popleft()
            self._last_action = action.copy()
            return action.copy(), False
        if self._last_action is None:
            raise QueueNotReady("Wait for the first chunk instead of stepping with an unsafe command")
        if np.allclose(self._last_action, 0.0):
            raise RuntimeError("Refusing to use an all-zero absolute hold command")
        self.hold_steps += 1
        return self._last_action.copy(), True


class LatestRequestWorker:
    """One inference thread with a coalescing latest-request mailbox."""

    def __init__(
        self,
        infer_fn: Callable[[InferenceRequest], InferencePayload],
        *,
        delivery_delay_seconds: float | Callable[[InferenceRequest, int], float] = 0.0,
    ) -> None:
        if not callable(delivery_delay_seconds) and delivery_delay_seconds < 0:
            raise ValueError("delivery_delay_seconds must be non-negative")
        self._infer_fn = infer_fn
        self._delivery_delay_seconds = delivery_delay_seconds
        self._condition = threading.Condition()
        self._pending: tuple[int, InferenceRequest] | None = None
        self._results: deque[InferenceResult] = deque()
        self._error: BaseException | None = None
        self._active = False
        self._closed = False
        self._episode_id: str | None = None
        self._generation = 0
        self.calls_started = 0
        self.calls_completed = 0
        self.pending_requests_replaced = 0
        self.pending_requests_cancelled = 0
        self.old_episode_results_discarded = 0
        self._thread = threading.Thread(target=self._run, name="actionstream-inference", daemon=True)
        self._thread.start()

    def reset_episode(self, episode_id: str) -> None:
        with self._condition:
            self._generation += 1
            self._episode_id = episode_id
            self._pending = None
            self._results.clear()
            self._error = None
            self.calls_started = 0
            self.calls_completed = 0
            self.pending_requests_replaced = 0
            self.pending_requests_cancelled = 0
            self.old_episode_results_discarded = 0
            self._condition.notify_all()

    def submit(self, request: InferenceRequest) -> None:
        with self._condition:
            self._raise_if_failed_locked()
            if self._closed:
                raise RuntimeError("Inference worker is closed")
            if request.episode_id != self._episode_id:
                raise ValueError(
                    f"Request episode {request.episode_id!r} does not match active "
                    f"episode {self._episode_id!r}"
                )
            if self._pending is not None:
                self.pending_requests_replaced += 1
            self._pending = (self._generation, request)
            self._condition.notify_all()

    def drain_results(self) -> list[InferenceResult]:
        with self._condition:
            self._raise_if_failed_locked()
            now = time.monotonic()
            results: list[InferenceResult] = []
            while self._results and self._results[0].delivery_timestamp <= now:
                results.append(self._results.popleft())
            return results

    def drain_all_results(self) -> list[InferenceResult]:
        """Drain completed results regardless of scheduled delivery while idle."""
        with self._condition:
            self._raise_if_failed_locked()
            if self._active or self._pending is not None:
                raise RuntimeError("drain_all_results requires an idle worker")
            results = list(self._results)
            self._results.clear()
            return results

    def cancel_pending(self) -> bool:
        """Cancel a mailbox request that has not started inference."""
        with self._condition:
            self._raise_if_failed_locked()
            cancelled = self._pending is not None
            if cancelled:
                self._pending = None
                self.pending_requests_cancelled += 1
                self._condition.notify_all()
            return cancelled

    def wait_for_result(self, timeout: float | None = None) -> bool:
        """Wait until at least one completed result reaches its delivery time."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                self._raise_if_failed_locked()
                now = time.monotonic()
                if self._results and self._results[0].delivery_timestamp <= now:
                    return True
                if self._closed and not self._active and self._pending is None:
                    return False

                wait_seconds: float | None = None
                if self._results:
                    wait_seconds = max(0.0, self._results[0].delivery_timestamp - now)
                if deadline is not None:
                    timeout_remaining = deadline - now
                    if timeout_remaining <= 0:
                        return False
                    wait_seconds = (
                        timeout_remaining
                        if wait_seconds is None
                        else min(wait_seconds, timeout_remaining)
                    )
                self._condition.wait(wait_seconds)

    def wait_idle(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._active or self._pending is not None:
                self._raise_if_failed_locked()
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            self._raise_if_failed_locked()
            return True

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("Inference worker did not stop")

    def _raise_if_failed_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError("Inference worker failed") from self._error

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                generation, request = self._pending
                self._pending = None
                self._active = True
                request_ordinal = self.calls_started
                self.calls_started += 1

            start_timestamp = time.monotonic()
            try:
                payload = self._infer_fn(request)
                end_timestamp = time.monotonic()
                if callable(self._delivery_delay_seconds):
                    delivery_delay_seconds = float(
                        self._delivery_delay_seconds(request, request_ordinal)
                    )
                else:
                    delivery_delay_seconds = float(self._delivery_delay_seconds)
                if delivery_delay_seconds < 0:
                    raise ValueError("delivery delay callback returned a negative delay")
                metadata = dict(payload.metadata)
                metadata.setdefault("delay_trace_index", request_ordinal)
                metadata.setdefault(
                    "injected_delivery_delay_seconds", delivery_delay_seconds
                )
                delivery_timestamp = end_timestamp + delivery_delay_seconds
                result = InferenceResult(
                    actions=payload.actions,
                    episode_id=request.episode_id,
                    observation_control_step=request.observation_control_step,
                    request_timestamp=request.request_timestamp,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    delivery_timestamp=delivery_timestamp,
                    model_inference_latency_seconds=payload.model_inference_latency_seconds,
                    metadata=metadata,
                    queue_depth_at_request_steps=request.queue_depth_at_request_steps,
                    queue_headroom_at_request_steps=request.queue_headroom_at_request_steps,
                )
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._active = False
                    self._condition.notify_all()
                continue

            with self._condition:
                self._active = False
                if generation == self._generation and request.episode_id == self._episode_id:
                    self._results.append(result)
                    self.calls_completed += 1
                else:
                    self.old_episode_results_discarded += 1
                self._condition.notify_all()
