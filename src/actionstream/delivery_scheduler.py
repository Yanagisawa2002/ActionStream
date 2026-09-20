"""Delivery scheduling helpers for delayed ActionStream responses.

Scheduling and delivery still mutate the owning engine under its original
condition lock. This module only separates responsibility; it does not introduce
another queue, lock, or generation authority.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Any

import torch

from actionstream.inference_types import _PendingDelivery


def schedule_delivery(
    engine: Any,
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
            if engine._delivery_delay_provider is None
            else float(engine._delivery_delay_provider(request_ordinal))
        )
        if not math.isfinite(delay_s) or delay_s < 0:
            raise ValueError("delivery delay must be finite and non-negative")
    except BaseException as exc:
        engine._emit(
            "delivery_provider_error",
            epoch=epoch,
            task_revision=task_revision,
            request_ordinal=request_ordinal,
            source_step=source_step,
            error_type=type(exc).__name__,
        )
        engine._mark_fatal(exc, epoch=epoch)
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
    with engine._condition:
        if not engine._epoch_is_current_locked(epoch):
            engine._note_rejected_epoch_locked(epoch)
            rejected = True
        else:
            heapq.heappush(
                engine._pending_deliveries,
                (
                    pending.scheduled_delivery_timestamp,
                    pending.request_ordinal,
                    pending,
                ),
            )
            engine._metrics["responses_scheduled"] += 1
            engine._condition.notify_all()
            rejected = False
    if rejected:
        notify_delivery_observer(engine, pending, status="rejected_reset")
        return
    engine._emit(
        "response_scheduled",
        epoch=epoch,
        task_revision=task_revision,
        request_ordinal=request_ordinal,
        source_step=source_step,
        delivery_delay_ms=delay_s * 1000.0,
    )
    notify_delivery_observer(engine, pending, status="scheduled")


def delivery_loop(engine: Any) -> None:
    try:
        while True:
            with engine._condition:
                while (
                    not engine._shutdown.is_set() and not engine._fatal_error.is_set()
                ):
                    if engine._resetting or not engine._pending_deliveries:
                        engine._condition.wait()
                        continue
                    ready_at, _, pending = engine._pending_deliveries[0]
                    remaining = ready_at - time.monotonic()
                    if remaining > 0:
                        engine._condition.wait(timeout=remaining)
                        continue
                    heapq.heappop(engine._pending_deliveries)
                    current_step = engine._current_step
                    break
                else:
                    return

            status, current_step, delivered_at = engine._merge_chunk(
                pending.actions,
                epoch=pending.epoch,
                task_revision=pending.task_revision,
                request_ordinal=pending.request_ordinal,
                source_step=pending.source_step,
                current_step=current_step,
                task=pending.task,
                scheduled=True,
            )
            notify_delivery_observer(
                engine,
                pending,
                status=status,
                delivery_timestamp=delivered_at,
                current_step=current_step,
            )
    except BaseException as exc:
        engine._mark_fatal(exc)


def notify_delivery_observer(
    engine: Any,
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
    engine._emit(
        "response_delivery_status",
        **identity,
        status=status,
        delivery_timestamp=delivery_timestamp,
        current_step=current_step,
    )
    if status in {"rejected_reset", "rejected_out_of_order"}:
        engine._emit(f"response_{status}", **identity)
    if engine._delivery_observer is None:
        return
    try:
        engine._delivery_observer(
            {
                **identity,
                "engine_id": engine._engine_id,
                "status": status,
                "model_completed_timestamp": pending.model_completed_timestamp,
                "scheduled_delivery_timestamp": pending.scheduled_delivery_timestamp,
                "delivery_timestamp": delivery_timestamp,
                "current_step": current_step,
            }
        )
    except BaseException as exc:
        engine._emit(
            "delivery_observer_error",
            **identity,
            error_type=type(exc).__name__,
        )
        engine._mark_fatal(exc, epoch=pending.epoch)
