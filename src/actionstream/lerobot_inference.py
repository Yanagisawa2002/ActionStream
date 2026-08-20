"""LeRobot rollout backend for delay-aligned action-chunk inference.

The engine in this module intentionally implements LeRobot's public
``InferenceEngine`` contract instead of introducing a second rollout loop.  It is
therefore usable by ``lerobot-rollout`` strategies through the same ``start`` /
``stop`` / ``reset`` / ``get_action`` / ``notify_observation`` lifecycle as the
upstream sync and RTC engines.

This is an engineering runtime, not a real-robot safety controller.  Its bounded
hold and fallback behaviour only prevent unbounded command replay and queue
starvation; hardware limits and an independent emergency stop remain external
requirements.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.rollout.inference.base import InferenceEngine
from lerobot.utils.feature_utils import build_dataset_frame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionStreamInferenceConfig:
    """Runtime knobs whose defaults favour bounded, observable behaviour."""

    inference_timeout_s: float = 5.0
    bounded_hold_steps: int = 2
    retry_backoff_s: float = 0.05
    max_consecutive_failures: int = 10
    join_timeout_s: float = 3.0
    latest_only_fallback: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.inference_timeout_s) or self.inference_timeout_s <= 0:
            raise ValueError("inference_timeout_s must be finite and positive")
        if self.bounded_hold_steps < 0:
            raise ValueError("bounded_hold_steps must be non-negative")
        if not math.isfinite(self.retry_backoff_s) or self.retry_backoff_s < 0:
            raise ValueError("retry_backoff_s must be finite and non-negative")
        if self.max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive")
        if not math.isfinite(self.join_timeout_s) or self.join_timeout_s <= 0:
            raise ValueError("join_timeout_s must be finite and positive")


@dataclass(frozen=True)
class ActionStreamTelemetry:
    """Thread-safe point-in-time telemetry returned by :attr:`telemetry`."""

    observations_received: int
    observations_superseded: int
    inference_started: int
    inference_completed: int
    inference_timeouts: int
    inference_errors: int
    disconnects: int
    recoveries: int
    consecutive_failures: int
    chunks_accepted: int
    chunks_rejected_stale: int
    chunks_rejected_reset: int
    stale_actions_discarded: int
    actions_dequeued: int
    hold_actions: int
    hold_exhausted: int
    fallback_activations: int
    fallback_chunks_accepted: int
    queue_depth: int
    queue_age_steps: int | None
    latest_inference_latency_ms: float | None
    max_inference_latency_ms: float | None
    failed: bool

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


@dataclass(frozen=True)
class _ObservationEnvelope:
    epoch: int
    step: int
    observation: Mapping[str, Any]


@dataclass(frozen=True)
class _QueuedAction:
    tensor: torch.Tensor
    source_step: int
    task: str


InferChunk = Callable[[Mapping[str, Any], str], torch.Tensor]
ResetProvider = Callable[[], None]


class ActionStreamInferenceEngine(InferenceEngine):
    """Latest-observation async inference with delay-aligned queue replacement.

    One worker owns policy inference.  ``notify_observation`` publishes into a
    single-slot mailbox, so inference cannot fan out when the policy is slower than
    the control loop.  Returned chunks are aligned to the current observation step;
    expired prefixes are discarded and fully stale responses are rejected while a
    usable queue remains.

    Once the queue is depleted, the last dispatched action may be repeated for at
    most ``bounded_hold_steps`` pulls.  A subsequent *healthy* but fully stale chunk
    may be accepted using latest-only semantics when fallback is enabled.  Timed-out
    responses and pre-reset responses are never accepted, including as fallback.

    ``infer_chunk`` and ``reset_provider`` are injectable so the same lifecycle and
    failure paths can be tested with MockRobot or a deterministic transport stub.  In
    normal LeRobot use they are omitted and the engine calls the supplied policy and
    processor pipelines directly.
    """

    def __init__(
        self,
        *,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        hw_features: dict,
        task: str,
        device: str | None,
        robot_type: str,
        config: ActionStreamInferenceConfig | None = None,
        shutdown_event: threading.Event | None = None,
        infer_chunk: InferChunk | None = None,
        reset_provider: ResetProvider | None = None,
    ) -> None:
        super().__init__(task=task)
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._hw_features = hw_features
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._config = config or ActionStreamInferenceConfig()
        self._global_shutdown_event = shutdown_event
        self._infer_chunk_override = infer_chunk
        self._reset_provider_override = reset_provider

        if infer_chunk is None and not callable(
            getattr(policy, "predict_action_chunk", None)
        ):
            raise TypeError(
                "ActionStreamInferenceEngine requires policy.predict_action_chunk()"
            )

        self._condition = threading.Condition()
        self._queue_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._active = threading.Event()
        self._started = False
        self._worker: threading.Thread | None = None

        self._epoch = 0
        self._current_step = -1
        self._latest_observation: _ObservationEnvelope | None = None
        self._provider_reset_pending = True
        self._queue: deque[_QueuedAction] = deque()
        self._last_action: _QueuedAction | None = None
        self._hold_steps_used = 0
        self._starved_pulls = 0

        self._failure_traceback: str | None = None
        self._fatal_error = threading.Event()
        self._connected = True
        self._metrics: dict[str, int | float | None] = {}
        self._reset_metrics_locked()

    # ------------------------------------------------------------------
    # LeRobot lifecycle
    # ------------------------------------------------------------------

    @property
    def control_thread_owns_policy(self) -> bool:
        return False

    @property
    def ready(self) -> bool:
        with self._condition:
            return self._started and not self._fatal_error.is_set()

    @property
    def failed(self) -> bool:
        return self._fatal_error.is_set()

    @property
    def failure_traceback(self) -> str | None:
        return self._failure_traceback

    def start(self) -> None:
        """Start one inference worker; repeated starts are idempotent."""
        with self._condition:
            if self._started:
                return
            self._shutdown.clear()
            self._fatal_error.clear()
            self._failure_traceback = None
            self._started = True
            self._worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="ActionStreamInference",
            )
            self._worker.start()
            self._condition.notify_all()

    def stop(self) -> None:
        """Stop and join the worker without replaying pending observations."""
        with self._condition:
            if not self._started:
                return
            self._shutdown.set()
            self._active.clear()
            self._latest_observation = None
            worker = self._worker
            self._condition.notify_all()
        if worker is not None and worker.is_alive():
            worker.join(timeout=self._config.join_timeout_s)
        with self._condition:
            still_alive = bool(worker is not None and worker.is_alive())
            if not still_alive:
                self._started = False
                self._worker = None
        if still_alive:
            self._mark_fatal(
                RuntimeError(
                    "ActionStream inference worker did not stop within "
                    f"{self._config.join_timeout_s:.3f}s"
                )
            )

    def pause(self) -> None:
        self._active.clear()

    def resume(self) -> None:
        self._active.set()
        with self._condition:
            self._condition.notify_all()

    def reset(self) -> None:
        """Invalidate in-flight work and clear all episode-scoped state.

        Provider reset is executed by the worker immediately before its next policy
        call.  This keeps policy mutation on the same thread as inference and avoids a
        control-thread reset racing an in-flight CUDA or remote call.
        """
        with self._condition:
            self._epoch += 1
            self._current_step = -1
            self._latest_observation = None
            self._provider_reset_pending = True
            self._connected = True
            self._reset_metrics_locked()
            with self._queue_lock:
                self._queue.clear()
                self._last_action = None
                self._hold_steps_used = 0
                self._starved_pulls = 0
            self._condition.notify_all()
        self._discard_task_change()

    # ------------------------------------------------------------------
    # Observation and action path
    # ------------------------------------------------------------------

    def notify_observation(self, obs: dict) -> None:
        """Publish a shallow snapshot into the coalescing latest mailbox."""
        with self._condition:
            self._current_step += 1
            if self._latest_observation is not None:
                self._metrics["observations_superseded"] += 1
            self._latest_observation = _ObservationEnvelope(
                epoch=self._epoch,
                step=self._current_step,
                observation=dict(obs),
            )
            self._metrics["observations_received"] += 1
            self._condition.notify_all()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Pop a queued action, apply bounded hold, or report starvation."""
        del obs_frame
        if self.failed:
            return None
        with self._queue_lock:
            if self._queue:
                queued = self._queue.popleft()
                self._last_action = queued
                self._hold_steps_used = 0
                self._starved_pulls = 0
                self._metrics["actions_dequeued"] += 1
                self._set_dispatched_task(queued.task)
                return queued.tensor.clone()

            if (
                self._last_action is not None
                and self._hold_steps_used < self._config.bounded_hold_steps
            ):
                self._hold_steps_used += 1
                self._metrics["hold_actions"] += 1
                self._set_dispatched_task(self._last_action.task)
                return self._last_action.tensor.clone()

            self._starved_pulls += 1
            self._metrics["hold_exhausted"] += 1
            return None

    @property
    def telemetry(self) -> ActionStreamTelemetry:
        """Return one lock-consistent metrics snapshot."""
        with self._condition, self._queue_lock:
            queue_age: int | None = None
            if self._queue:
                queue_age = max(0, self._current_step - self._queue[0].source_step)
            latest_latency = self._metrics["latest_inference_latency_ms"]
            max_latency = self._metrics["max_inference_latency_ms"]
            return ActionStreamTelemetry(
                observations_received=int(self._metrics["observations_received"]),
                observations_superseded=int(self._metrics["observations_superseded"]),
                inference_started=int(self._metrics["inference_started"]),
                inference_completed=int(self._metrics["inference_completed"]),
                inference_timeouts=int(self._metrics["inference_timeouts"]),
                inference_errors=int(self._metrics["inference_errors"]),
                disconnects=int(self._metrics["disconnects"]),
                recoveries=int(self._metrics["recoveries"]),
                consecutive_failures=int(self._metrics["consecutive_failures"]),
                chunks_accepted=int(self._metrics["chunks_accepted"]),
                chunks_rejected_stale=int(self._metrics["chunks_rejected_stale"]),
                chunks_rejected_reset=int(self._metrics["chunks_rejected_reset"]),
                stale_actions_discarded=int(self._metrics["stale_actions_discarded"]),
                actions_dequeued=int(self._metrics["actions_dequeued"]),
                hold_actions=int(self._metrics["hold_actions"]),
                hold_exhausted=int(self._metrics["hold_exhausted"]),
                fallback_activations=int(self._metrics["fallback_activations"]),
                fallback_chunks_accepted=int(self._metrics["fallback_chunks_accepted"]),
                queue_depth=len(self._queue),
                queue_age_steps=queue_age,
                latest_inference_latency_ms=(
                    None if latest_latency is None else float(latest_latency)
                ),
                max_inference_latency_ms=None
                if max_latency is None
                else float(max_latency),
                failed=self._fatal_error.is_set(),
            )

    # ------------------------------------------------------------------
    # Worker-owned provider and merge path
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._shutdown.is_set()
                        or (
                            self._active.is_set()
                            and self._latest_observation is not None
                            and not self._fatal_error.is_set()
                        )
                    )
                    if self._shutdown.is_set() or self._fatal_error.is_set():
                        return
                    envelope = self._latest_observation
                    self._latest_observation = None
                    reset_provider = self._provider_reset_pending
                    self._provider_reset_pending = False
                    self._metrics["inference_started"] += 1

                if reset_provider:
                    self._reset_provider()

                task, _ = self._take_task()
                started = time.perf_counter()
                try:
                    chunk = self._infer_chunk(envelope.observation, task)
                    latency_s = time.perf_counter() - started
                    self._record_latency(latency_s)
                    if latency_s > self._config.inference_timeout_s:
                        self._record_transient_failure(timeout=True)
                        self._retry_failed_observation(envelope)
                        continue
                    actions = self._validate_chunk(chunk)
                except Exception as exc:
                    self._record_transient_failure(error=exc)
                    self._retry_failed_observation(envelope)
                    continue

                with self._condition:
                    self._metrics["inference_completed"] += 1
                    if not self._connected:
                        self._connected = True
                        self._metrics["recoveries"] += 1
                    self._metrics["consecutive_failures"] = 0
                    if envelope.epoch != self._epoch or self._shutdown.is_set():
                        self._metrics["chunks_rejected_reset"] += 1
                        continue
                    current_step = self._current_step

                self._merge_chunk(
                    actions,
                    source_step=envelope.step,
                    current_step=current_step,
                    task=task,
                )
        except BaseException as exc:
            self._mark_fatal(exc)

    def _retry_failed_observation(self, envelope: _ObservationEnvelope) -> None:
        """Retry a failed request unless a newer observation already won.

        Before the first action is available, the control loop cannot publish a
        fresher observation.  Dropping a timed-out or disconnected request there
        would leave the rollout waiting forever.  During normal control, the
        single-slot mailbox still gives precedence to any observation published
        while inference was in flight.
        """

        with self._condition:
            if (
                self._fatal_error.is_set()
                or self._shutdown.is_set()
                or envelope.epoch != self._epoch
                or self._latest_observation is not None
            ):
                return
            self._latest_observation = envelope
            self._condition.notify_all()

    def _reset_provider(self) -> None:
        if self._reset_provider_override is not None:
            self._reset_provider_override()
            return
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def _infer_chunk(self, observation: Mapping[str, Any], task: str) -> torch.Tensor:
        if self._infer_chunk_override is not None:
            return self._infer_chunk_override(observation, task)

        obs_batch = build_dataset_frame(
            self._hw_features, dict(observation), prefix="observation"
        )
        obs_batch = prepare_observation_for_inference(
            obs_batch,
            self._device,
            task,
            self._robot_type,
        )
        obs_batch["task"] = [task]
        with torch.inference_mode():
            processed = self._preprocessor(obs_batch)
            actions = self._policy.predict_action_chunk(processed)
            actions = self._postprocessor(actions)
        return actions

    @staticmethod
    def _validate_chunk(chunk: torch.Tensor) -> torch.Tensor:
        if not isinstance(chunk, torch.Tensor):
            raise TypeError(
                f"predict_action_chunk must return torch.Tensor, got {type(chunk)!r}"
            )
        actions = chunk.detach()
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError(
                    f"Expected batch size 1, got chunk shape {tuple(actions.shape)}"
                )
            actions = actions.squeeze(0)
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
            raise ValueError(
                f"Expected non-empty [T,A] action chunk, got {tuple(actions.shape)}"
            )
        if not torch.isfinite(actions).all():
            raise ValueError("Action chunk contains non-finite values")
        return actions.to(device="cpu").clone()

    def _merge_chunk(
        self,
        actions: torch.Tensor,
        *,
        source_step: int,
        current_step: int,
        task: str,
    ) -> None:
        age_steps = max(0, current_step - source_step)
        dropped = min(age_steps, len(actions))
        with self._queue_lock:
            fully_stale = dropped >= len(actions)
            allow_fallback = (
                fully_stale
                and self._config.latest_only_fallback
                and not self._queue
                and (
                    self._starved_pulls > 0
                    or self._hold_steps_used >= self._config.bounded_hold_steps
                )
            )
            if fully_stale and not allow_fallback:
                self._metrics["chunks_rejected_stale"] += 1
                self._metrics["stale_actions_discarded"] += dropped
                return

            if allow_fallback:
                replacement = actions
                self._metrics["fallback_activations"] += 1
                self._metrics["fallback_chunks_accepted"] += 1
            else:
                replacement = actions[dropped:]
                self._metrics["stale_actions_discarded"] += dropped

            self._queue.clear()
            self._queue.extend(
                _QueuedAction(tensor=action.clone(), source_step=source_step, task=task)
                for action in replacement
            )
            self._hold_steps_used = 0
            self._starved_pulls = 0
            self._metrics["chunks_accepted"] += 1

    def _record_latency(self, latency_s: float) -> None:
        latency_ms = latency_s * 1000.0
        with self._condition:
            self._metrics["latest_inference_latency_ms"] = latency_ms
            current_max = self._metrics["max_inference_latency_ms"]
            if current_max is None or latency_ms > float(current_max):
                self._metrics["max_inference_latency_ms"] = latency_ms

    def _record_transient_failure(
        self,
        *,
        timeout: bool = False,
        error: Exception | None = None,
    ) -> None:
        with self._condition:
            if timeout:
                self._metrics["inference_timeouts"] += 1
            else:
                self._metrics["inference_errors"] += 1
            if self._connected:
                self._connected = False
                self._metrics["disconnects"] += 1
            self._metrics["consecutive_failures"] += 1
            failures = int(self._metrics["consecutive_failures"])

        if error is not None:
            logger.warning(
                "ActionStream inference failure (%d/%d): %s",
                failures,
                self._config.max_consecutive_failures,
                error,
            )
        if failures >= self._config.max_consecutive_failures:
            cause = error or TimeoutError(
                f"Inference exceeded {self._config.inference_timeout_s:.3f}s "
                f"for {failures} consecutive requests"
            )
            self._mark_fatal(cause)
            return
        if self._config.retry_backoff_s:
            self._shutdown.wait(self._config.retry_backoff_s)

    def _mark_fatal(self, exc: BaseException) -> None:
        if self._fatal_error.is_set():
            return
        formatted = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        self._failure_traceback = formatted
        self._fatal_error.set()
        if self._global_shutdown_event is not None:
            self._global_shutdown_event.set()
        with self._condition:
            self._condition.notify_all()
        logger.error("Fatal ActionStream inference error: %s", exc)

    def _reset_metrics_locked(self) -> None:
        self._metrics = {
            "observations_received": 0,
            "observations_superseded": 0,
            "inference_started": 0,
            "inference_completed": 0,
            "inference_timeouts": 0,
            "inference_errors": 0,
            "disconnects": 0,
            "recoveries": 0,
            "consecutive_failures": 0,
            "chunks_accepted": 0,
            "chunks_rejected_stale": 0,
            "chunks_rejected_reset": 0,
            "stale_actions_discarded": 0,
            "actions_dequeued": 0,
            "hold_actions": 0,
            "hold_exhausted": 0,
            "fallback_activations": 0,
            "fallback_chunks_accepted": 0,
            "latest_inference_latency_ms": None,
            "max_inference_latency_ms": None,
        }
