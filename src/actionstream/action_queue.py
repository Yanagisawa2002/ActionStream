"""Action queue alignment, stale rejection, hold, and fallback helpers.

These helpers deliberately operate on the engine's existing locks/state so the
reset-generation -> worker -> delivery -> queue-commit linearization point remains
unchanged while queue policy becomes independently reviewable.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from actionstream.inference_types import ActionStreamAction, _QueuedAction


def validate_chunk(chunk: torch.Tensor) -> torch.Tensor:
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


def get_action_with_revision(
    engine: Any, obs_frame: dict | None
) -> ActionStreamAction | None:
    """Pop/hold an action with a receipt, preserving the engine lock boundary."""
    del obs_frame
    event = "queue_depleted"
    fields: dict[str, Any] = {}
    packet = None
    with engine._condition, engine._queue_lock:
        if not engine._started or not engine._epoch_is_current_locked(engine._epoch):
            return None
        epoch = engine._epoch
        queued = None
        if engine._queue:
            queued = engine._queue.popleft()
            engine._last_action = queued
            engine._hold_steps_used = engine._starved_pulls = 0
            engine._metrics["actions_dequeued"] += 1
            event = "action_dequeued"
            fields = {"queue_depth": len(engine._queue)}
        elif engine._last_action is not None and (
            engine._hold_steps_used < engine._config.bounded_hold_steps
        ):
            queued = engine._last_action
            engine._hold_steps_used += 1
            engine._metrics["hold_actions"] += 1
            event = "bounded_hold"
            fields = {"hold_step": engine._hold_steps_used}
        else:
            engine._starved_pulls += 1
            engine._metrics["hold_exhausted"] += 1
            fields = {"starved_pulls": engine._starved_pulls}
        if queued is not None:
            engine._set_dispatched_task(queued.task)
            packet = ActionStreamAction(
                action=queued.tensor.clone(),
                engine_id=engine._engine_id,
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
    engine._emit(event, epoch=epoch, **fields)
    return packet


def is_action_current(engine: Any, action: ActionStreamAction) -> bool:
    with engine._condition:
        return (
            engine._started
            and action.engine_id == engine._engine_id
            and engine._epoch_is_current_locked(action.epoch)
            and action.task_revision == engine._task_revision
        )


def merge_chunk(
    engine: Any,
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
    """Commit a chunk at the existing condition+queue linearization point."""
    rejected = False
    with engine._condition, engine._queue_lock:
        current_step = engine._current_step
        delivered_at = time.monotonic()
        if not engine._epoch_is_current_locked(epoch):
            engine._note_rejected_epoch_locked(epoch)
            return "rejected_reset", current_step, delivered_at
        if scheduled:
            if source_step <= engine._latest_delivered_source_step:
                engine._metrics["responses_rejected_out_of_order"] += 1
                engine._metrics["chunks_rejected_stale"] += 1
                engine._metrics["stale_actions_discarded"] += len(actions)
                return "rejected_out_of_order", current_step, delivered_at
            engine._latest_delivered_source_step = source_step
            engine._metrics["responses_delivered"] += 1
        age_steps = max(0, current_step - source_step)
        dropped = min(age_steps, len(actions))
        fully_stale = dropped >= len(actions)
        allow_fallback = (
            fully_stale
            and engine._config.latest_only_fallback
            and not engine._queue
            and (
                engine._starved_pulls > 0
                or engine._hold_steps_used >= engine._config.bounded_hold_steps
            )
        )
        if fully_stale and not allow_fallback:
            engine._metrics["chunks_rejected_stale"] += 1
            engine._metrics["stale_actions_discarded"] += dropped
            event = "chunk_rejected_stale"
            queue_depth = len(engine._queue)
            rejected = True
        elif allow_fallback:
            replacement = actions
            engine._metrics["fallback_activations"] += 1
            engine._metrics["fallback_chunks_accepted"] += 1
            event = "fallback_activated"
        else:
            replacement = actions[dropped:]
            engine._metrics["stale_actions_discarded"] += dropped
            event = "chunk_accepted"

        if not rejected:
            engine._queue.clear()
            engine._queue.extend(
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
            engine._hold_steps_used = 0
            engine._starved_pulls = 0
            engine._metrics["chunks_accepted"] += 1
            queue_depth = len(engine._queue)

    engine._emit(
        event,
        epoch=epoch,
        task_revision=task_revision,
        request_ordinal=request_ordinal,
        source_step=source_step,
        age_steps=age_steps,
        dropped_steps=dropped,
        queue_depth=queue_depth,
    )
    return "delivered", current_step, delivered_at
