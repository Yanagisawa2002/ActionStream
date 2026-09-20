"""Configuration, telemetry, provenance, and internal queue types."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch


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

    is_action_current checks this receipt; it does not execute or revoke a robot
    command. The caller must serialize validation/execution with resets.
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
