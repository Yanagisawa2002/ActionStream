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
import heapq
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

from actionstream.inference_transport import (
    DirectInferenceTransport,
    InferenceCancelled,
    InferenceDeadlineExceeded,
    InferenceTransport,
    ProcessInferenceTransport,
)
from actionstream.telemetry import JsonlTelemetrySink
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
    transport_mode: str = "direct"
    process_transport_factory: str | None = None
    process_transport_start_method: str = "spawn"
    process_transport_startup_timeout_s: float = 30.0
    process_transport_terminate_timeout_s: float = 1.0
    delivery_scheduler_enabled: bool = False
    minimum_request_interval_steps: int = 1
    telemetry_jsonl_path: str | None = None

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
        if self.transport_mode not in {"direct", "process"}:
            raise ValueError("transport_mode must be 'direct' or 'process'")
        if self.transport_mode == "process" and not self.process_transport_factory:
            raise ValueError("process transport requires process_transport_factory")
        if (
            self.transport_mode == "direct"
            and self.process_transport_factory is not None
        ):
            raise ValueError(
                "process_transport_factory requires transport_mode='process'"
            )
        if not math.isfinite(self.process_transport_startup_timeout_s) or (
            self.process_transport_startup_timeout_s <= 0
        ):
            raise ValueError("process_transport_startup_timeout_s must be positive")
        if not math.isfinite(self.process_transport_terminate_timeout_s) or (
            self.process_transport_terminate_timeout_s <= 0
        ):
            raise ValueError("process_transport_terminate_timeout_s must be positive")
        if not isinstance(self.delivery_scheduler_enabled, bool):
            raise TypeError("delivery_scheduler_enabled must be bool")
        if (
            type(self.minimum_request_interval_steps) is not int
            or self.minimum_request_interval_steps <= 0
        ):
            raise ValueError("minimum_request_interval_steps must be a positive int")


@dataclass(frozen=True)
class ActionStreamTelemetry:
    """Thread-safe point-in-time telemetry returned by :attr:`telemetry`."""

    observations_received: int
    observations_superseded: int
    observations_skipped_by_budget: int
    inference_started: int
    inference_completed: int
    inference_timeouts: int
    inference_errors: int
    responses_scheduled: int
    responses_delivered: int
    responses_rejected_out_of_order: int
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
    pending_responses: int
    queue_age_steps: int | None
    latest_inference_latency_ms: float | None
    max_inference_latency_ms: float | None
    transport_mode: str
    deadline_enforced: bool
    transport_process_restarts: int
    transport_cancellations: int
    transport_startup_latency_ms: float | None
    transport_request_latency_ms: float | None
    delivery_scheduler_enabled: bool
    telemetry_write_errors: int
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


@dataclass(frozen=True)
class _PendingDelivery:
    epoch: int
    request_ordinal: int
    source_step: int
    task: str
    actions: torch.Tensor
    model_completed_timestamp: float
    scheduled_delivery_timestamp: float


InferChunk = Callable[[Mapping[str, Any], str], torch.Tensor]
ResetProvider = Callable[[], None]
DeliveryDelayProvider = Callable[[int], float]
DeliveryObserver = Callable[[Mapping[str, Any]], None]


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
        transport: InferenceTransport | None = None,
        delivery_delay_provider: DeliveryDelayProvider | None = None,
        delivery_observer: DeliveryObserver | None = None,
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
        self._delivery_delay_provider = delivery_delay_provider
        self._delivery_observer = delivery_observer

        if not self._config.delivery_scheduler_enabled and (
            delivery_delay_provider is not None or delivery_observer is not None
        ):
            raise ValueError(
                "delivery callbacks require delivery_scheduler_enabled=True"
            )

        if (
            transport is None
            and self._config.transport_mode == "direct"
            and infer_chunk is None
            and not callable(getattr(policy, "predict_action_chunk", None))
        ):
            raise TypeError(
                "ActionStreamInferenceEngine requires policy.predict_action_chunk()"
            )

        if transport is not None:
            self._transport = transport
        elif self._config.transport_mode == "process":
            self._transport = ProcessInferenceTransport(
                str(self._config.process_transport_factory),
                start_method=self._config.process_transport_start_method,
                startup_timeout_s=self._config.process_transport_startup_timeout_s,
                terminate_timeout_s=self._config.process_transport_terminate_timeout_s,
            )
        else:
            self._transport = DirectInferenceTransport(self._infer_chunk_direct)
        self._telemetry_sink = (
            None
            if self._config.telemetry_jsonl_path is None
            else JsonlTelemetrySink(self._config.telemetry_jsonl_path)
        )

        self._condition = threading.Condition()
        self._queue_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._active = threading.Event()
        self._started = False
        self._worker: threading.Thread | None = None
        self._delivery_worker: threading.Thread | None = None

        self._epoch = 0
        self._current_step = -1
        self._next_request_ordinal = 0
        self._latest_observation: _ObservationEnvelope | None = None
        self._last_submitted_source_step = -1
        self._provider_reset_pending = True
        self._pending_deliveries: list[tuple[float, int, _PendingDelivery]] = []
        self._latest_delivered_source_step = -1
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
            if self._config.delivery_scheduler_enabled:
                self._delivery_worker = threading.Thread(
                    target=self._delivery_loop,
                    daemon=True,
                    name="ActionStreamDelivery",
                )
                self._delivery_worker.start()
            self._worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="ActionStreamInference",
            )
            self._worker.start()
            self._condition.notify_all()
        self._emit("engine_started")

    def stop(self) -> None:
        """Stop and join the worker without replaying pending observations."""
        self._emit("engine_stop_requested")
        with self._condition:
            if not self._started:
                self._transport.close()
                return
            self._shutdown.set()
            self._active.clear()
            self._latest_observation = None
            worker = self._worker
            delivery_worker = self._delivery_worker
            cancelled_deliveries = [item[2] for item in self._pending_deliveries]
            self._pending_deliveries.clear()
            self._condition.notify_all()
        for pending in cancelled_deliveries:
            self._notify_delivery_observer(pending, status="cancelled_stop")
        cancelled = self._transport.cancel()
        if cancelled:
            self._emit("transport_cancelled", reason="stop")
        if worker is not None and worker.is_alive():
            worker.join(timeout=self._config.join_timeout_s)
        if delivery_worker is not None and delivery_worker.is_alive():
            delivery_worker.join(timeout=self._config.join_timeout_s)
        with self._condition:
            worker_alive = bool(worker is not None and worker.is_alive())
            delivery_worker_alive = bool(
                delivery_worker is not None and delivery_worker.is_alive()
            )
            still_alive = worker_alive or delivery_worker_alive
            if not still_alive:
                self._started = False
                self._worker = None
                self._delivery_worker = None
        if still_alive:
            self._mark_fatal(
                RuntimeError(
                    "ActionStream inference worker did not stop within "
                    f"{self._config.join_timeout_s:.3f}s"
                )
            )
            self._emit(
                "engine_stop_failed",
                worker_alive=worker_alive,
                delivery_worker_alive=delivery_worker_alive,
            )
        else:
            self._transport.close()
            self._emit(
                "engine_stopped",
                worker_alive=False,
                delivery_worker_alive=False,
            )

    def pause(self) -> None:
        self._active.clear()
        self._emit("engine_paused")

    def resume(self) -> None:
        self._active.set()
        with self._condition:
            self._condition.notify_all()
        self._emit("engine_resumed")

    def reset(self) -> None:
        """Invalidate in-flight work and clear all episode-scoped state.

        Provider reset is executed by the worker immediately before its next policy
        call.  This keeps policy mutation on the same thread as inference and avoids a
        control-thread reset racing an in-flight CUDA or remote call.
        """
        with self._condition:
            self._epoch += 1
            epoch = self._epoch
            self._current_step = -1
            self._next_request_ordinal = 0
            self._latest_observation = None
            self._last_submitted_source_step = -1
            self._provider_reset_pending = True
            self._connected = True
            self._reset_metrics_locked()
            cancelled_deliveries = [item[2] for item in self._pending_deliveries]
            self._pending_deliveries.clear()
            self._metrics["chunks_rejected_reset"] += len(cancelled_deliveries)
            self._latest_delivered_source_step = -1
            with self._queue_lock:
                self._queue.clear()
                self._last_action = None
                self._hold_steps_used = 0
                self._starved_pulls = 0
            self._condition.notify_all()
        for pending in cancelled_deliveries:
            self._notify_delivery_observer(pending, status="rejected_reset")
        cancelled = self._transport.reset()
        self._discard_task_change()
        self._emit("engine_reset", epoch=epoch, transport_cancelled=cancelled)

    # ------------------------------------------------------------------
    # Observation and action path
    # ------------------------------------------------------------------

    def notify_observation(self, obs: dict) -> None:
        """Publish a shallow snapshot into the coalescing latest mailbox."""
        with self._condition:
            self._current_step += 1
            self._metrics["observations_received"] += 1
            step = self._current_step
            epoch = self._epoch
            if (
                self._last_submitted_source_step >= 0
                and step
                < self._last_submitted_source_step
                + self._config.minimum_request_interval_steps
            ):
                self._metrics["observations_skipped_by_budget"] += 1
                skipped_by_budget = True
                superseded = False
            else:
                skipped_by_budget = False
                superseded = self._latest_observation is not None
                if self._latest_observation is not None:
                    self._metrics["observations_superseded"] += 1
                self._latest_observation = _ObservationEnvelope(
                    epoch=epoch,
                    step=step,
                    observation=dict(obs),
                )
                self._condition.notify_all()
        if skipped_by_budget:
            self._emit(
                "observation_budget_skipped",
                epoch=epoch,
                step=step,
                minimum_request_interval_steps=(
                    self._config.minimum_request_interval_steps
                ),
            )
            return
        self._emit(
            "observation_received", epoch=epoch, step=step, superseded=superseded
        )

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Pop a queued action, apply bounded hold, or report starvation."""
        del obs_frame
        if self.failed:
            return None
        event = "queue_depleted"
        fields: dict[str, Any] = {}
        result: torch.Tensor | None = None
        with self._queue_lock:
            if self._queue:
                queued = self._queue.popleft()
                self._last_action = queued
                self._hold_steps_used = 0
                self._starved_pulls = 0
                self._metrics["actions_dequeued"] += 1
                self._set_dispatched_task(queued.task)
                result = queued.tensor.clone()
                event = "action_dequeued"
                fields = {
                    "source_step": queued.source_step,
                    "queue_depth": len(self._queue),
                }

            elif (
                self._last_action is not None
                and self._hold_steps_used < self._config.bounded_hold_steps
            ):
                self._hold_steps_used += 1
                self._metrics["hold_actions"] += 1
                self._set_dispatched_task(self._last_action.task)
                result = self._last_action.tensor.clone()
                event = "bounded_hold"
                fields = {"hold_step": self._hold_steps_used}
            else:
                self._starved_pulls += 1
                self._metrics["hold_exhausted"] += 1
                fields = {"starved_pulls": self._starved_pulls}
        self._emit(event, **fields)
        return result

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
                observations_skipped_by_budget=int(
                    self._metrics["observations_skipped_by_budget"]
                ),
                inference_started=int(self._metrics["inference_started"]),
                inference_completed=int(self._metrics["inference_completed"]),
                inference_timeouts=int(self._metrics["inference_timeouts"]),
                inference_errors=int(self._metrics["inference_errors"]),
                responses_scheduled=int(self._metrics["responses_scheduled"]),
                responses_delivered=int(self._metrics["responses_delivered"]),
                responses_rejected_out_of_order=int(
                    self._metrics["responses_rejected_out_of_order"]
                ),
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
                pending_responses=len(self._pending_deliveries),
                queue_age_steps=queue_age,
                latest_inference_latency_ms=(
                    None if latest_latency is None else float(latest_latency)
                ),
                max_inference_latency_ms=None
                if max_latency is None
                else float(max_latency),
                transport_mode=self._config.transport_mode,
                deadline_enforced=self._transport.deadline_enforced,
                transport_process_restarts=self._transport.process_restarts,
                transport_cancellations=self._transport.cancellations,
                transport_startup_latency_ms=(
                    None
                    if self._transport.latest_startup_latency_s is None
                    else self._transport.latest_startup_latency_s * 1000.0
                ),
                transport_request_latency_ms=(
                    None
                    if self._transport.latest_request_latency_s is None
                    else self._transport.latest_request_latency_s * 1000.0
                ),
                delivery_scheduler_enabled=self._config.delivery_scheduler_enabled,
                telemetry_write_errors=int(self._metrics["telemetry_write_errors"]),
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
                    self._last_submitted_source_step = envelope.step
                    reset_provider = self._provider_reset_pending
                    self._provider_reset_pending = False
                    self._metrics["inference_started"] += 1
                    request_ordinal = self._next_request_ordinal
                    self._next_request_ordinal += 1

                self._emit(
                    "inference_started",
                    epoch=envelope.epoch,
                    source_step=envelope.step,
                    request_ordinal=request_ordinal,
                    deadline_s=self._config.inference_timeout_s,
                )

                if reset_provider:
                    self._reset_provider()

                task, _ = self._take_task()
                started = time.perf_counter()
                try:
                    chunk = self._transport.infer(
                        envelope.observation,
                        task,
                        timeout_s=self._config.inference_timeout_s,
                    )
                    latency_s = time.perf_counter() - started
                    self._record_latency(latency_s, epoch=envelope.epoch)
                    actions = self._validate_chunk(chunk)
                except InferenceCancelled:
                    with self._condition:
                        reset_or_stop = (
                            envelope.epoch != self._epoch or self._shutdown.is_set()
                        )
                        if envelope.epoch != self._epoch:
                            self._metrics["chunks_rejected_reset"] += 1
                    self._emit(
                        "inference_cancelled",
                        epoch=envelope.epoch,
                        source_step=envelope.step,
                        reset_or_stop=reset_or_stop,
                    )
                    if reset_or_stop:
                        continue
                    self._record_transient_failure(
                        epoch=envelope.epoch, error=InferenceCancelled()
                    )
                    self._retry_failed_observation(envelope)
                    continue
                except InferenceDeadlineExceeded as exc:
                    latency_s = time.perf_counter() - started
                    self._record_latency(latency_s, epoch=envelope.epoch)
                    self._emit(
                        "inference_timeout",
                        epoch=envelope.epoch,
                        source_step=envelope.step,
                        latency_ms=latency_s * 1000.0,
                        deadline_enforced=self._transport.deadline_enforced,
                    )
                    self._record_transient_failure(
                        epoch=envelope.epoch, timeout=True, error=exc
                    )
                    self._retry_failed_observation(envelope)
                    continue
                except Exception as exc:
                    latency_s = time.perf_counter() - started
                    self._record_latency(latency_s, epoch=envelope.epoch)
                    self._emit(
                        "inference_error",
                        epoch=envelope.epoch,
                        source_step=envelope.step,
                        error_type=type(exc).__name__,
                    )
                    self._record_transient_failure(epoch=envelope.epoch, error=exc)
                    self._retry_failed_observation(envelope)
                    continue

                with self._condition:
                    if envelope.epoch != self._epoch or self._shutdown.is_set():
                        self._metrics["chunks_rejected_reset"] += 1
                        continue
                    self._metrics["inference_completed"] += 1
                    if not self._connected:
                        self._connected = True
                        self._metrics["recoveries"] += 1
                        recovered = True
                    else:
                        recovered = False
                    self._metrics["consecutive_failures"] = 0
                    current_step = self._current_step

                self._emit(
                    "inference_completed",
                    epoch=envelope.epoch,
                    source_step=envelope.step,
                    request_ordinal=request_ordinal,
                    current_step=current_step,
                    latency_ms=latency_s * 1000.0,
                    recovered=recovered,
                )

                if self._config.delivery_scheduler_enabled:
                    self._schedule_delivery(
                        actions,
                        epoch=envelope.epoch,
                        request_ordinal=request_ordinal,
                        source_step=envelope.step,
                        task=task,
                    )
                else:
                    self._merge_chunk(
                        actions,
                        epoch=envelope.epoch,
                        source_step=envelope.step,
                        current_step=current_step,
                        task=task,
                    )
        except BaseException as exc:
            self._mark_fatal(exc)

    def _schedule_delivery(
        self,
        actions: torch.Tensor,
        *,
        epoch: int,
        request_ordinal: int,
        source_step: int,
        task: str,
    ) -> None:
        delay_s = (
            0.0
            if self._delivery_delay_provider is None
            else float(self._delivery_delay_provider(request_ordinal))
        )
        if not math.isfinite(delay_s) or delay_s < 0:
            raise ValueError(
                "delivery delay provider must return a finite non-negative value"
            )
        completed = time.monotonic()
        pending = _PendingDelivery(
            epoch=epoch,
            request_ordinal=request_ordinal,
            source_step=source_step,
            task=task,
            actions=actions,
            model_completed_timestamp=completed,
            scheduled_delivery_timestamp=completed + delay_s,
        )
        with self._condition:
            if epoch != self._epoch or self._shutdown.is_set():
                self._metrics["chunks_rejected_reset"] += 1
                rejected = True
            else:
                heapq.heappush(
                    self._pending_deliveries,
                    (
                        pending.scheduled_delivery_timestamp,
                        pending.request_ordinal,
                        pending,
                    ),
                )
                self._metrics["responses_scheduled"] += 1
                self._condition.notify_all()
                rejected = False
        if rejected:
            self._notify_delivery_observer(pending, status="rejected_reset")
            return
        self._emit(
            "response_scheduled",
            epoch=epoch,
            request_ordinal=request_ordinal,
            source_step=source_step,
            delivery_delay_ms=delay_s * 1000.0,
        )
        self._notify_delivery_observer(pending, status="scheduled")

    def _delivery_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                with self._condition:
                    while not self._shutdown.is_set():
                        if not self._pending_deliveries:
                            self._condition.wait()
                            continue
                        ready_at, _, pending = self._pending_deliveries[0]
                        remaining = ready_at - time.monotonic()
                        if remaining > 0:
                            self._condition.wait(timeout=remaining)
                            continue
                        heapq.heappop(self._pending_deliveries)
                        if pending.epoch != self._epoch:
                            self._metrics["chunks_rejected_reset"] += 1
                            status = "rejected_reset"
                            current_step = self._current_step
                        elif pending.source_step <= self._latest_delivered_source_step:
                            self._metrics["responses_rejected_out_of_order"] += 1
                            self._metrics["chunks_rejected_stale"] += 1
                            self._metrics["stale_actions_discarded"] += len(
                                pending.actions
                            )
                            status = "rejected_out_of_order"
                            current_step = self._current_step
                        else:
                            status = "delivered"
                            current_step = self._current_step
                        break
                    else:  # pragma: no cover - loop exits through shutdown guard
                        return
                    if self._shutdown.is_set():
                        return

                delivered_at = time.monotonic()
                if status == "delivered":
                    status = self._merge_chunk(
                        pending.actions,
                        epoch=pending.epoch,
                        source_step=pending.source_step,
                        current_step=current_step,
                        task=pending.task,
                        scheduled=True,
                    )
                else:
                    self._emit(
                        f"response_{status}",
                        epoch=pending.epoch,
                        request_ordinal=pending.request_ordinal,
                        source_step=pending.source_step,
                    )
                self._notify_delivery_observer(
                    pending,
                    status=status,
                    delivery_timestamp=delivered_at,
                    current_step=current_step,
                )
        except BaseException as exc:
            self._mark_fatal(exc)

    def _notify_delivery_observer(
        self,
        pending: _PendingDelivery,
        *,
        status: str,
        delivery_timestamp: float | None = None,
        current_step: int | None = None,
    ) -> None:
        if self._delivery_observer is None:
            return
        self._delivery_observer(
            {
                "status": status,
                "epoch": pending.epoch,
                "request_ordinal": pending.request_ordinal,
                "source_step": pending.source_step,
                "model_completed_timestamp": pending.model_completed_timestamp,
                "scheduled_delivery_timestamp": (pending.scheduled_delivery_timestamp),
                "delivery_timestamp": delivery_timestamp,
                "current_step": current_step,
            }
        )

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

    def _infer_chunk_direct(
        self, observation: Mapping[str, Any], task: str
    ) -> torch.Tensor:
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
        epoch: int,
        source_step: int,
        current_step: int,
        task: str,
        scheduled: bool = False,
    ) -> str:
        # Reset takes condition -> queue. Commit the generation check, control
        # step, queue and episode metrics under the same lock order.
        with self._condition:
            if epoch != self._epoch or self._shutdown.is_set():
                self._metrics["chunks_rejected_reset"] += 1
                return "rejected_reset"
            if scheduled:
                if source_step <= self._latest_delivered_source_step:
                    self._metrics["responses_rejected_out_of_order"] += 1
                    return "rejected_out_of_order"
                self._latest_delivered_source_step = source_step
                self._metrics["responses_delivered"] += 1
            current_step = self._current_step
            age_steps = max(0, current_step - source_step)
            dropped = min(age_steps, len(actions))
            rejected = False
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
                    event = "chunk_rejected_stale"
                    queue_depth = len(self._queue)
                    rejected = True

                elif allow_fallback:
                    replacement = actions
                    self._metrics["fallback_activations"] += 1
                    self._metrics["fallback_chunks_accepted"] += 1
                    event = "fallback_activated"
                else:
                    replacement = actions[dropped:]
                    self._metrics["stale_actions_discarded"] += dropped
                    event = "chunk_accepted"

                if not rejected:
                    self._queue.clear()
                    self._queue.extend(
                        _QueuedAction(
                            tensor=action.clone(), source_step=source_step, task=task
                        )
                        for action in replacement
                    )
                    self._hold_steps_used = 0
                    self._starved_pulls = 0
                    self._metrics["chunks_accepted"] += 1
                    queue_depth = len(self._queue)
        self._emit(
            event,
            age_steps=age_steps,
            dropped_steps=dropped,
            queue_depth=queue_depth,
        )

        return "delivered"

    def _record_latency(self, latency_s: float, *, epoch: int) -> None:
        latency_ms = latency_s * 1000.0
        with self._condition:
            if epoch != self._epoch or self._shutdown.is_set():
                return
            self._metrics["latest_inference_latency_ms"] = latency_ms
            current_max = self._metrics["max_inference_latency_ms"]
            if current_max is None or latency_ms > float(current_max):
                self._metrics["max_inference_latency_ms"] = latency_ms

    def _record_transient_failure(
        self,
        *,
        epoch: int,
        timeout: bool = False,
        error: Exception | None = None,
    ) -> None:
        with self._condition:
            if epoch != self._epoch or self._shutdown.is_set():
                self._metrics["chunks_rejected_reset"] += 1
                return
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
            self._mark_fatal(cause, epoch=epoch)
            return
        if self._config.retry_backoff_s:
            self._shutdown.wait(self._config.retry_backoff_s)

    def _mark_fatal(self, exc: BaseException, *, epoch: int | None = None) -> None:
        formatted = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        with self._condition:
            if epoch is not None and (epoch != self._epoch or self._shutdown.is_set()):
                return
            if self._fatal_error.is_set():
                return
            self._failure_traceback = formatted
            self._fatal_error.set()
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
            self._condition.notify_all()
        self._emit("engine_fatal", error_type=type(exc).__name__)
        logger.error("Fatal ActionStream inference error: %s", exc)

    def _emit(self, event: str, **fields: Any) -> None:
        if self._telemetry_sink is None:
            return
        try:
            self._telemetry_sink.emit(event, **fields)
        except Exception:
            with self._condition:
                self._metrics["telemetry_write_errors"] += 1
            logger.exception("Failed to write ActionStream telemetry event %s", event)

    def _reset_metrics_locked(self) -> None:
        self._metrics = {
            "observations_received": 0,
            "observations_superseded": 0,
            "observations_skipped_by_budget": 0,
            "inference_started": 0,
            "inference_completed": 0,
            "inference_timeouts": 0,
            "inference_errors": 0,
            "responses_scheduled": 0,
            "responses_delivered": 0,
            "responses_rejected_out_of_order": 0,
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
            "telemetry_write_errors": 0,
        }
