"""Observation-only bounded controller. This module has no oracle/backend access."""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Protocol

from .contracts import Execution, Observation, TaskSpec


class PublicPort(Protocol):
    def start(self, request_id: str, revision: int) -> Observation: ...
    def infer(self, observation: Observation, instruction: str): ...
    def step(self, action, request_id: str, revision: int, control_step: int): ...
    def refresh(self, request_id: str, revision: int) -> Observation: ...


def execute(task: TaskSpec, port: PublicPort, checker, emit, *, interruption_step=None):
    """Checker alone declares completion; native boundary merely forces a stop."""
    run = Execution(task)
    started = time.monotonic()
    record = dict(
        status="RUNNING",
        checker_complete=False,
        control_steps=0,
        recoveries=0,
        checks=0,
        inference_calls=0,
        native_boundary=False,
    )
    try:
        observation = port.start(task.request_id, run.revision)
        emit("accepted", task=asdict(task), observation=observation.binding())
        while run.status in ("accepted", "executing") and run.steps < run.max_steps:
            control_started = time.monotonic()
            if not run.pending:
                actions, inference_receipt = port.infer(
                    observation, task.canonical_instruction
                )
                run.queue(actions, observation)
                record["inference_calls"] += 1
                emit(
                    "chunk",
                    revision=run.revision,
                    source_control_step=run.steps,
                    observation=observation.binding(),
                    actions=actions.tolist(),
                    **inference_receipt,
                )
            action = run.take_action()
            emit(
                "step_requested",
                control_step=run.steps,
                revision=run.revision,
                action=action.tolist(),
            )
            observation, boundary = port.step(
                action, task.request_id, run.revision, run.steps
            )
            record["native_boundary"] = bool(boundary)
            emit(
                "step",
                control_step=run.steps,
                revision=run.revision,
                observation=observation.binding(),
                native_boundary=bool(boundary),
            )
            interrupted = interruption_step == run.steps and not boundary
            if interrupted:
                discarded = run.interrupt()
                observation = port.refresh(task.request_id, run.revision)
                emit(
                    "software_interruption",
                    control_step=run.steps,
                    revision=run.revision,
                    discarded_actions=discarded,
                    observation=observation.binding(),
                    environment_reset=False,
                )
            if (
                interrupted
                or boundary
                or run.steps % 30 == 0
                or run.steps == run.max_steps
            ):
                run.validate_current(observation)
                decision, call_receipt = checker(observation)
                record["checks"] += 1
                emit(
                    "checker",
                    control_step=run.steps,
                    revision=run.revision,
                    observation=observation.binding(),
                    decision=asdict(decision),
                    **call_receipt,
                )
                run.apply_check(decision, observation, observation, bool(boundary))
                if run.status not in ("accepted", "executing"):
                    break
            remaining = 1 / 20 - (time.monotonic() - control_started)
            if remaining > 0:
                time.sleep(remaining)
        record.update(status=run.status, checker_complete=run.status == "complete")
    except BaseException as exc:
        run.pending.clear()
        run.status = "ERROR"
        record.update(status="ERROR", error_type=type(exc).__name__, error=str(exc))
    finally:
        record.update(
            control_steps=run.steps,
            recoveries=run.recoveries,
            revision=run.revision,
            wall_s=time.monotonic() - started,
        )
        emit("execution_end", **record)
    return record
