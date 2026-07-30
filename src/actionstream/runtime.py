"""Minimal local async worker and final-action queue semantics."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


RuntimeMode = Literal["async_naive", "async_aligned"]


class QueueNotReady(RuntimeError):
    """Raised before the first safe final command exists."""


@dataclass(frozen=True)
class InferenceRequest:
    observation: Mapping[str, Any]
    task_instruction: str
    episode_id: str
    observation_control_step: int
    request_timestamp: float

    def __post_init__(self) -> None:
        if self.observation_control_step < 0:
            raise ValueError("observation_control_step must be non-negative")


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

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Inference result must contain [T,7] final actions, got {actions.shape}")
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


@dataclass(frozen=True)
class MergeOutcome:
    accepted: bool
    reason: str
    age_steps: int
    dropped_prefix_steps: int
    fully_stale: bool
    queue_length: int


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
        self.stale_prefix_lengths: list[int] = []

    def reset_episode(self, episode_id: str) -> None:
        self._episode_id = episode_id
        self._queue.clear()
        self._last_action = None
        self.hold_steps = 0
        self.stale_chunks_discarded = 0
        self.old_episode_chunks_rejected = 0
        self.replacements = 0
        self.stale_prefix_lengths.clear()

    @property
    def queue_length(self) -> int:
        return len(self._queue)

    @property
    def has_safe_action(self) -> bool:
        return bool(self._queue) or self._last_action is not None

    def replace(
        self,
        result: InferenceResult,
        *,
        current_control_step: int,
        mode: RuntimeMode,
    ) -> MergeOutcome:
        if self._episode_id is None:
            raise RuntimeError("Reset the action queue before merging a chunk")
        if result.episode_id != self._episode_id:
            self.old_episode_chunks_rejected += 1
            return MergeOutcome(
                accepted=False,
                reason="old_episode",
                age_steps=max(0, current_control_step - result.observation_control_step),
                dropped_prefix_steps=0,
                fully_stale=False,
                queue_length=len(self._queue),
            )

        age_steps = current_control_step - result.observation_control_step
        if age_steps < 0:
            raise ValueError(
                "A result cannot be newer than the current control step: "
                f"current={current_control_step}, observation={result.observation_control_step}"
            )

        drop = 0 if mode == "async_naive" else age_steps
        bounded_drop = min(drop, len(result.actions))
        self.stale_prefix_lengths.append(bounded_drop)
        if drop >= len(result.actions):
            self.stale_chunks_discarded += 1
            return MergeOutcome(
                accepted=False,
                reason="fully_stale",
                age_steps=age_steps,
                dropped_prefix_steps=bounded_drop,
                fully_stale=True,
                queue_length=len(self._queue),
            )

        replacement = result.actions[drop:]
        for action in replacement:
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Action queue accepts only finite final 7D commands")
            if np.allclose(action, 0.0):
                raise ValueError("Refusing to queue an all-zero absolute command")

        self._queue.clear()
        self._queue.extend(np.asarray(action, dtype=np.float32).copy() for action in replacement)
        self.replacements += 1
        return MergeOutcome(
            accepted=True,
            reason="replaced",
            age_steps=age_steps,
            dropped_prefix_steps=bounded_drop,
            fully_stale=False,
            queue_length=len(self._queue),
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
        delivery_delay_seconds: float = 0.0,
    ) -> None:
        if delivery_delay_seconds < 0:
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
                self.calls_started += 1

            start_timestamp = time.monotonic()
            try:
                payload = self._infer_fn(request)
                end_timestamp = time.monotonic()
                delivery_timestamp = end_timestamp + self._delivery_delay_seconds
                result = InferenceResult(
                    actions=payload.actions,
                    episode_id=request.episode_id,
                    observation_control_step=request.observation_control_step,
                    request_timestamp=request.request_timestamp,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    delivery_timestamp=delivery_timestamp,
                    model_inference_latency_seconds=payload.model_inference_latency_seconds,
                    metadata=payload.metadata,
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
