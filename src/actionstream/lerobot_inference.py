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
import uuid
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
    """Engine snapshot with independently sampled transport diagnostics."""

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
class ActionStreamAction:
    """Caller-owned action plus provenance for a serialized dispatch boundary.

    ``is_action_current`` checks this receipt; it does not execute or revoke a
    robot command. The caller must serialize validation/execution with resets.
    """

    action: torch.Tensor
    engine_id: str
    epoch: int
    task_revision: int
    request_ordinal: int
    source_step: int
    task: str


@dataclass(frozen=True)
class _QueuedAction:
    tensor: torch.Tensor
    epoch: int
    task_revision: int
    request_ordinal: int
    source_step: int
    task: str


@dataclass(frozen=True)
class _PendingDelivery:
    epoch: int
    task_revision: int
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

        # Lifecycle calls serialize here; model/IPC/join never hold _condition
        # or _queue_lock. Workers never acquire _lifecycle_lock.
        self._lifecycle_lock = threading.Lock()
        self._condition = threading.Condition()
        self._queue_lock = threading.Lock()
        self._resetting = False
        self._epoch_cancelled = threading.Event()
        self._task_revision = 0
        self._engine_id = (
            self._telemetry_sink.engine_id if self._telemetry_sink else uuid.uuid4().hex
        )
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
            return self._started and self._epoch_is_current_locked(self._epoch)

    @property
    def failed(self) -> bool:
        return self._fatal_error.is_set()

    @property
    def failure_traceback(self) -> str | None:
        return self._failure_traceback

    def start(self) -> None:
        """Start one owner thread; restart only after a successful stop."""
        if threading.current_thread() in (self._worker, self._delivery_worker):
            raise RuntimeError("start must run outside inference/delivery callbacks")
        with self._lifecycle_lock:
            with self._condition:
                if self._started:
                    return
                if self._shutdown.is_set():
                    self._reset_metrics_locked()
                    self._connected = True
                self._shutdown.clear()
                self._fatal_error.clear()
                self._failure_traceback = None
                self._started = True
                self._provider_reset_pending = True
                epoch = self._epoch
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
        self._emit("engine_started", epoch=epoch)

    def _invalidate_episode_locked(self) -> list[_PendingDelivery]:
        """Called with condition held; lock order is condition -> queue -> task."""
        self._epoch_cancelled.set()
        self._epoch += 1
        self._epoch_cancelled = threading.Event()
        self._current_step = -1
        self._next_request_ordinal = 0
        self._latest_observation = None
        self._last_submitted_source_step = -1
        self._provider_reset_pending = True
        cancelled = [item[2] for item in self._pending_deliveries]
        self._pending_deliveries.clear()
        self._latest_delivered_source_step = -1
        with self._queue_lock:
            self._queue.clear()
            self._last_action = None
            self._hold_steps_used = 0
            self._starved_pulls = 0
            self._discard_task_change()
        return cancelled

    def stop(self) -> None:
        """Invalidate actions before cancellation/join; preserve final metrics."""
        # Lifecycle methods belong to the control thread, never either worker.
        if threading.current_thread() in (self._worker, self._delivery_worker):
            raise RuntimeError("stop must run outside inference/delivery callbacks")
        with self._lifecycle_lock:
            with self._condition:
                self._shutdown.set()
                self._active.clear()
                self._resetting = True
                pending = self._invalidate_episode_locked()
                epoch = self._epoch
                worker, delivery_worker = self._worker, self._delivery_worker
                self._condition.notify_all()
            self._emit("engine_stop_requested", epoch=epoch)
            try:
                cancelled = self._transport.cancel()
                if cancelled:
                    self._emit("transport_cancelled", epoch=epoch, reason="stop")
                for thread in (worker, delivery_worker):
                    if thread is not None and thread.is_alive():
                        thread.join(timeout=self._config.join_timeout_s)
                worker_alive = bool(worker is not None and worker.is_alive())
                delivery_alive = bool(
                    delivery_worker is not None and delivery_worker.is_alive()
                )
                if worker_alive or delivery_alive:
                    self._mark_fatal(
                        RuntimeError(
                            "ActionStream worker did not stop within "
                            f"{self._config.join_timeout_s:.3f}s"
                        )
                    )
                else:
                    self._transport.close()
                    with self._condition:
                        self._started = False
                        self._worker = self._delivery_worker = None
            except BaseException as exc:
                self._mark_fatal(exc)
                raise
            finally:
                with self._condition:
                    self._resetting = False
                    self._condition.notify_all()
        for response in pending:
            self._notify_delivery_observer(response, status="cancelled_stop")
        self._emit(
            "engine_stop_failed"
            if worker_alive or delivery_alive
            else "engine_stopped",
            epoch=epoch,
            worker_alive=worker_alive,
            delivery_worker_alive=delivery_alive,
        )

    def pause(self) -> None:
        # Preserve queued actions and in-flight commits, as in LeRobot's API.
        with self._condition:
            self._active.clear()
            epoch = self._epoch
        self._emit("engine_paused", epoch=epoch)

    def resume(self) -> None:
        with self._condition:
            self._active.set()
            epoch = self._epoch
            self._condition.notify_all()
        self._emit("engine_resumed", epoch=epoch)

    def reset(self) -> None:
        """Invalidate first, then finish transport teardown before new admission.

        Observations received during teardown remain in the latest mailbox.
        Provider reset runs on its owner thread, and is retried until successful.
        Fatal state is sticky until stop/start; caller-owned shutdown is not cleared.
        """
        if threading.current_thread() in (self._worker, self._delivery_worker):
            raise RuntimeError("reset must run outside inference/delivery callbacks")
        with self._lifecycle_lock:
            with self._condition:
                self._resetting = True
                pending = self._invalidate_episode_locked()
                epoch = self._epoch
                self._connected = True
                self._reset_metrics_locked()
                # This counter measures cleanup, not new-episode model failures.
                self._metrics["chunks_rejected_reset"] = len(pending)
                self._condition.notify_all()
            try:
                cancelled = self._transport.reset()
            except BaseException as exc:
                self._mark_fatal(exc, epoch=epoch, during_reset=True)
                raise
            finally:
                with self._condition:
                    self._resetting = False
                    self._condition.notify_all()
        for response in pending:
            self._notify_delivery_observer(response, status="rejected_reset")
        self._emit("engine_reset", epoch=epoch, transport_cancelled=cancelled)

    def set_task(self, task: str) -> bool:
        """Track task revisions while retaining LeRobot's queued-tail semantics."""
        with self._condition, self._task_lock:
            if task == self._task:
                return False
            self._task = task
            self._task_changed = True
            self._task_revision += 1
            revision, epoch = self._task_revision, self._epoch
        self._emit("task_changed", epoch=epoch, task_revision=revision)
        return True

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
        packet = self.get_action_with_revision(obs_frame)
        return None if packet is None else packet.action

    def get_action_with_revision(
        self, obs_frame: dict | None
    ) -> ActionStreamAction | None:
        """Pop/hold an action with a receipt, without changing the tensor API."""
        del obs_frame
        event = "queue_depleted"
        fields: dict[str, Any] = {}
        packet = None
        with self._condition, self._queue_lock:
            if not self._started or not self._epoch_is_current_locked(self._epoch):
                return None
            epoch = self._epoch
            queued = None
            if self._queue:
                queued = self._queue.popleft()
                self._last_action = queued
                self._hold_steps_used = self._starved_pulls = 0
                self._metrics["actions_dequeued"] += 1
                event = "action_dequeued"
                fields = {"queue_depth": len(self._queue)}
            elif self._last_action is not None and (
                self._hold_steps_used < self._config.bounded_hold_steps
            ):
                queued = self._last_action
                self._hold_steps_used += 1
                self._metrics["hold_actions"] += 1
                event = "bounded_hold"
                fields = {"hold_step": self._hold_steps_used}
            else:
                self._starved_pulls += 1
                self._metrics["hold_exhausted"] += 1
                fields = {"starved_pulls": self._starved_pulls}
            if queued is not None:
                self._set_dispatched_task(queued.task)
                packet = ActionStreamAction(
                    action=queued.tensor.clone(),
                    engine_id=self._engine_id,
                    epoch=queued.epoch,
                    task_revision=queued.task_revision,
                    request_ordinal=queued.request_ordinal,
                    source_step=queued.source_step,
                    task=queued.task,
                )
                fields.update(
                    task_revision=queued.task_revision,
                    request_ordinal=queued.request_ordinal,
                    source_step=queued.source_step,
                )
        self._emit(event, epoch=epoch, **fields)
        return packet

    def is_action_current(self, action: ActionStreamAction) -> bool:
        """Check immediately before dispatch on the lifecycle/control thread.

        This is not an atomic robot write. A multi-threaded dispatcher must use
        its own serialization boundary for reset/task-change and validation/write.
        """
        with self._condition:
            return (
                self._started
                and action.engine_id == self._engine_id
                and self._epoch_is_current_locked(action.epoch)
                and action.task_revision == self._task_revision
            )

    @property
    def telemetry(self) -> ActionStreamTelemetry:
        """Read transport diagnostics outside the engine commit locks."""
        transport_restarts = self._transport.process_restarts
        transport_cancellations = self._transport.cancellations
        startup_latency = self._transport.latest_startup_latency_s
        request_latency = self._transport.latest_request_latency_s
        deadline_enforced = self._transport.deadline_enforced
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
                deadline_enforced=deadline_enforced,
                transport_process_restarts=transport_restarts,
                transport_cancellations=transport_cancellations,
                transport_startup_latency_ms=(
                    None if startup_latency is None else startup_latency * 1000.0
                ),
                transport_request_latency_ms=(
                    None if request_latency is None else request_latency * 1000.0
                ),
                delivery_scheduler_enabled=self._config.delivery_scheduler_enabled,
                telemetry_write_errors=int(self._metrics["telemetry_write_errors"]),
                failed=self._fatal_error.is_set(),
            )

    # ------------------------------------------------------------------
    # Worker-owned provider and merge path
    # ------------------------------------------------------------------

    def _epoch_is_current_locked(self, epoch: int) -> bool:
        return (
            epoch == self._epoch
            and not self._shutdown.is_set()
            and not self._resetting
            and not self._fatal_error.is_set()
        )

    def _note_rejected_epoch_locked(self, epoch: int) -> None:
        if epoch != self._epoch and not self._shutdown.is_set():
            self._metrics["chunks_rejected_reset"] += 1

    def _worker_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._shutdown.is_set()
                        or self._fatal_error.is_set()
                        or (
                            self._active.is_set()
                            and not self._resetting
                            and self._latest_observation is not None
                        )
                    )
                    if self._shutdown.is_set() or self._fatal_error.is_set():
                        return
                    envelope = self._latest_observation
                    self._latest_observation = None
                    self._last_submitted_source_step = envelope.step
                    reset_provider = self._provider_reset_pending
                    cancelled = self._epoch_cancelled
                    task, _ = self._take_task()
                    task_revision = self._task_revision
                    self._metrics["inference_started"] += 1
                    request_ordinal = self._next_request_ordinal
                    self._next_request_ordinal += 1
                identity = dict(
                    epoch=envelope.epoch,
                    source_step=envelope.step,
                    request_ordinal=request_ordinal,
                    task_revision=task_revision,
                )
                self._emit(
                    "inference_started",
                    **identity,
                    deadline_s=self._config.inference_timeout_s,
                )
                started = None
                try:
                    if reset_provider:
                        self._reset_provider()
                        with self._condition:
                            if not self._epoch_is_current_locked(envelope.epoch):
                                self._note_rejected_epoch_locked(envelope.epoch)
                                continue
                            # A failed/obsolete reset must never clear this flag.
                            self._provider_reset_pending = False
                    if cancelled.is_set():
                        raise InferenceCancelled("Episode invalidated before inference")
                    started = time.perf_counter()
                    cancellable = getattr(self._transport, "infer_cancellable", None)
                    if callable(cancellable):
                        chunk = cancellable(
                            envelope.observation,
                            task,
                            timeout_s=self._config.inference_timeout_s,
                            cancellation_event=cancelled,
                        )
                    else:
                        # Legacy injected transports retain their existing API.
                        chunk = self._transport.infer(
                            envelope.observation,
                            task,
                            timeout_s=self._config.inference_timeout_s,
                        )
                    latency_s = time.perf_counter() - started
                    actions = self._validate_chunk(chunk)
                except BaseException as exc:
                    latency_s = (
                        None if started is None else time.perf_counter() - started
                    )
                    timeout = isinstance(exc, InferenceDeadlineExceeded)
                    was_cancelled = isinstance(exc, InferenceCancelled)
                    event = (
                        "inference_cancelled"
                        if was_cancelled
                        else "inference_timeout"
                        if timeout
                        else "inference_error"
                    )
                    self._emit(
                        event,
                        **identity,
                        error_type=type(exc).__name__,
                        phase="provider_reset" if started is None else "inference",
                        reset_or_stop=cancelled.is_set(),
                        latency_ms=None if latency_s is None else latency_s * 1000.0,
                        deadline_enforced=self._transport.deadline_enforced,
                    )
                    if not isinstance(exc, Exception):
                        self._mark_fatal(exc, epoch=envelope.epoch)
                    else:
                        self._record_transient_failure(
                            epoch=envelope.epoch,
                            timeout=timeout,
                            error=exc,
                            latency_s=latency_s,
                        )
                        self._retry_failed_observation(envelope)
                    continue

                with self._condition:
                    if not self._epoch_is_current_locked(envelope.epoch):
                        self._note_rejected_epoch_locked(envelope.epoch)
                        continue
                    self._record_latency(latency_s)
                    self._metrics["inference_completed"] += 1
                    recovered = not self._connected
                    if recovered:
                        self._metrics["recoveries"] += 1
                    self._connected = True
                    self._metrics["consecutive_failures"] = 0
                    current_step = self._current_step
                self._emit(
                    "inference_completed",
                    **identity,
                    current_step=current_step,
                    latency_ms=latency_s * 1000.0,
                    recovered=recovered,
                )
                if self._config.delivery_scheduler_enabled:
                    self._schedule_delivery(actions, **identity, task=task)
                else:
                    self._merge_chunk(
                        actions, **identity, current_step=current_step, task=task
                    )
        except BaseException as exc:
            # Unexpected engine implementation failure, not a provider outcome.
            self._mark_fatal(exc)

    def _schedule_delivery(
        self,
        actions: torch.Tensor,
        *,
        epoch: int,
        request_ordinal: int,
        task_revision: int,
        source_step: int,
        task: str,
    ) -> None:
        try:
            delay_s = (
                0.0
                if self._delivery_delay_provider is None
                else float(self._delivery_delay_provider(request_ordinal))
            )
            if not math.isfinite(delay_s) or delay_s < 0:
                raise ValueError("delivery delay must be finite and non-negative")
        except BaseException as exc:
            self._emit(
                "delivery_provider_error",
                epoch=epoch,
                task_revision=task_revision,
                request_ordinal=request_ordinal,
                source_step=source_step,
                error_type=type(exc).__name__,
            )
            self._mark_fatal(exc, epoch=epoch)
            return
        completed = time.monotonic()
        pending = _PendingDelivery(
            epoch=epoch,
            task_revision=task_revision,
            request_ordinal=request_ordinal,
            source_step=source_step,
            task=task,
            actions=actions,
            model_completed_timestamp=completed,
            scheduled_delivery_timestamp=completed + delay_s,
        )
        with self._condition:
            if not self._epoch_is_current_locked(epoch):
                self._note_rejected_epoch_locked(epoch)
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
            task_revision=task_revision,
            request_ordinal=request_ordinal,
            source_step=source_step,
            delivery_delay_ms=delay_s * 1000.0,
        )
        self._notify_delivery_observer(pending, status="scheduled")

    def _delivery_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    while (
                        not self._shutdown.is_set() and not self._fatal_error.is_set()
                    ):
                        if self._resetting or not self._pending_deliveries:
                            self._condition.wait()
                            continue
                        ready_at, _, pending = self._pending_deliveries[0]
                        remaining = ready_at - time.monotonic()
                        if remaining > 0:
                            self._condition.wait(timeout=remaining)
                            continue
                        heapq.heappop(self._pending_deliveries)
                        current_step = self._current_step
                        break
                    else:
                        return
                status, current_step, delivered_at = self._merge_chunk(
                    pending.actions,
                    epoch=pending.epoch,
                    task_revision=pending.task_revision,
                    request_ordinal=pending.request_ordinal,
                    source_step=pending.source_step,
                    current_step=current_step,
                    task=pending.task,
                    scheduled=True,
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
        identity = dict(
            epoch=pending.epoch,
            task_revision=pending.task_revision,
            request_ordinal=pending.request_ordinal,
            source_step=pending.source_step,
        )
        self._emit(
            "response_delivery_status",
            **identity,
            status=status,
            delivery_timestamp=delivery_timestamp,
            current_step=current_step,
        )
        if status in {"rejected_reset", "rejected_out_of_order"}:
            self._emit(f"response_{status}", **identity)
        if self._delivery_observer is None:
            return
        try:
            self._delivery_observer(
                {
                    **identity,
                    "engine_id": self._engine_id,
                    "status": status,
                    "model_completed_timestamp": pending.model_completed_timestamp,
                    "scheduled_delivery_timestamp": pending.scheduled_delivery_timestamp,
                    "delivery_timestamp": delivery_timestamp,
                    "current_step": current_step,
                }
            )
        except BaseException as exc:
            self._emit(
                "delivery_observer_error", **identity, error_type=type(exc).__name__
            )
            self._mark_fatal(exc, epoch=pending.epoch)

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
        task_revision: int,
        request_ordinal: int,
        source_step: int,
        current_step: int,
        task: str,
        scheduled: bool = False,
    ) -> tuple[str, int, float]:
        # Response identity, latest age, delivery counters and queue commit share
        # one linearization point. The caller's step may already be stale.
        rejected = False
        with self._condition, self._queue_lock:
            current_step = self._current_step
            delivered_at = time.monotonic()
            if not self._epoch_is_current_locked(epoch):
                self._note_rejected_epoch_locked(epoch)
                return "rejected_reset", current_step, delivered_at
            if scheduled:
                if source_step <= self._latest_delivered_source_step:
                    self._metrics["responses_rejected_out_of_order"] += 1
                    self._metrics["chunks_rejected_stale"] += 1
                    self._metrics["stale_actions_discarded"] += len(actions)
                    return "rejected_out_of_order", current_step, delivered_at
                self._latest_delivered_source_step = source_step
                self._metrics["responses_delivered"] += 1
            age_steps = max(0, current_step - source_step)
            dropped = min(age_steps, len(actions))
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
                        tensor=action.clone(),
                        epoch=epoch,
                        task_revision=task_revision,
                        request_ordinal=request_ordinal,
                        source_step=source_step,
                        task=task,
                    )
                    for action in replacement
                )
                self._hold_steps_used = 0
                self._starved_pulls = 0
                self._metrics["chunks_accepted"] += 1
                queue_depth = len(self._queue)
        self._emit(
            event,
            epoch=epoch,
            task_revision=task_revision,
            request_ordinal=request_ordinal,
            source_step=source_step,
            age_steps=age_steps,
            dropped_steps=dropped,
            queue_depth=queue_depth,
        )
        # Delivered means the response reached this generation, even if stale
        # alignment rejected its actions; chunk acceptance is a separate event.
        return "delivered", current_step, delivered_at

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
        epoch: int,
        timeout: bool = False,
        error: Exception | None = None,
        latency_s: float | None = None,
    ) -> None:
        with self._condition:
            if not self._epoch_is_current_locked(epoch):
                self._note_rejected_epoch_locked(epoch)
                return
            if latency_s is not None:
                self._record_latency(latency_s)
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
            with self._condition:
                self._condition.wait_for(
                    lambda: not self._epoch_is_current_locked(epoch),
                    timeout=self._config.retry_backoff_s,
                )

    def _mark_fatal(
        self,
        exc: BaseException,
        *,
        epoch: int | None = None,
        during_reset: bool = False,
    ) -> None:
        formatted = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        with self._condition:
            if self._fatal_error.is_set():
                return
            if epoch is not None and (
                epoch != self._epoch
                or self._shutdown.is_set()
                or (self._resetting and not during_reset)
            ):
                return
            committed_epoch = self._epoch
            self._failure_traceback = formatted
            self._fatal_error.set()
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
            self._condition.notify_all()
        self._emit("engine_fatal", epoch=committed_epoch, error_type=type(exc).__name__)
        logger.error("Fatal ActionStream inference error: %s", exc)

    def _emit(self, event: str, *, epoch: int, **fields: Any) -> None:
        if self._telemetry_sink is None:
            return
        try:
            self._telemetry_sink.emit(event, epoch=epoch, **fields)
        except Exception:
            with self._condition:
                if epoch == self._epoch and not self._shutdown.is_set():
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
