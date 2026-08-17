"""Pure, atomic M8 event recording with frozen episode provenance."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .m8_protocol import (
    M8_MILESTONE,
    M8_SCHEMA_VERSION,
    NATIVE_ISAAC_EVIDENCE_CLASS,
    PROFILE_STRATEGIES,
)
from .m8_replay import FAIRNESS_FIELDS, HOLDOUT_FAIRNESS_FIELDS
from .m8_scenario import command_targets_obsolete_destination
from .schema import AtomicJsonlLog, json_safe


def _finite_command(command: Sequence[float]) -> list[float]:
    result = [float(value) for value in command]
    if len(result) != 7 or not all(math.isfinite(value) for value in result):
        raise ValueError("M8 commands must contain exactly seven finite values")
    return result


class M8EventRecorder:
    """Write one complete episode without ever publishing a partial JSONL log.

    The class is ROS-independent so adapters, unit tests, and offline importers
    share the same lifecycle and provenance behavior.  Runtime-specific event
    payloads remain extensible, while identity and event indices are injected
    here and cannot be overridden by callers.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        episode_id: str,
        seed: int,
        profile_id: str,
        strategy: str,
        fairness: Mapping[str, Any],
        split: str = "frozen_holdout",
        overwrite: bool = False,
    ) -> None:
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if profile_id not in PROFILE_STRATEGIES:
            raise ValueError(f"unknown M8 profile: {profile_id}")
        if strategy not in PROFILE_STRATEGIES[profile_id]:
            raise ValueError(f"strategy {strategy} is not enabled for {profile_id}")
        if split not in {"baseline_gate", "development", "frozen_holdout"}:
            raise ValueError(f"unknown M8 split: {split}")
        required_fairness = HOLDOUT_FAIRNESS_FIELDS if split == "frozen_holdout" else FAIRNESS_FIELDS
        missing = [field for field in required_fairness if field not in fairness]
        if missing:
            raise ValueError(f"episode fairness metadata missing: {missing}")
        self.path = Path(path)
        self.episode_id = episode_id
        self.seed = int(seed)
        self.profile_id = profile_id
        self.strategy = strategy
        self.split = split
        self.fairness = {field: json_safe(fairness[field]) for field in required_fairness}
        self._log = AtomicJsonlLog(self.path, overwrite=overwrite)
        self._event_index = 0
        self._opened = False
        self._started = False
        self._switched = False
        self._terminated = False
        self._terminal_step: int | None = None
        self._finished = False

    def __enter__(self) -> "M8EventRecorder":
        self._log.__enter__()
        self._opened = True
        return self

    def _append(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if not self._opened:
            raise RuntimeError("recorder must be used as a context manager")
        if self._finished:
            raise RuntimeError("cannot append after episode_end")
        protected = {
            "schema_version",
            "milestone",
            "evidence_class",
            "event_index",
            "event_type",
            "episode_id",
            "seed",
            "profile_id",
            "strategy",
            "split",
        }
        overlap = protected & set(payload)
        if overlap:
            raise ValueError(f"event payload overrides protected fields: {sorted(overlap)}")
        row = {
            "schema_version": M8_SCHEMA_VERSION,
            "milestone": M8_MILESTONE,
            "evidence_class": NATIVE_ISAAC_EVIDENCE_CLASS,
            "event_index": self._event_index,
            "event_type": event_type,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "profile_id": self.profile_id,
            "strategy": self.strategy,
            "split": self.split,
            **json_safe(payload),
        }
        self._log.append(row)
        self._event_index += 1

    def start(self, **payload: Any) -> None:
        if self._started:
            raise RuntimeError("episode_start already recorded")
        self._append("episode_start", {**self.fairness, **payload})
        self._started = True

    def append(self, event_type: str, **payload: Any) -> None:
        if not self._started:
            raise RuntimeError("episode_start must be recorded first")
        if event_type in {"episode_start", "destination_switched", "task_terminated", "episode_end"}:
            raise ValueError(f"use the lifecycle method for {event_type}")
        self._append(event_type, payload)

    def destination_switched(
        self,
        *,
        step: int,
        old_destination_xyz: Sequence[float],
        new_destination_xyz: Sequence[float],
        generation_before: int = 1,
        generation_after: int = 2,
        **payload: Any,
    ) -> None:
        if not self._started or self._switched:
            raise RuntimeError("destination switch requires one started, unswitched episode")
        if step <= 0 or generation_after <= generation_before:
            raise ValueError("destination switch step/generations are invalid")
        old = [float(value) for value in old_destination_xyz]
        new = [float(value) for value in new_destination_xyz]
        if len(old) != 3 or len(new) != 3 or not all(math.isfinite(value) for value in (*old, *new)):
            raise ValueError("destination poses must contain three finite values")
        if old == new:
            raise ValueError("destination switch must change the physical goal")
        self._append(
            "destination_switched",
            {
                "step": int(step),
                "switch_step": int(step),
                "old_destination_xyz": old,
                "new_destination_xyz": new,
                "generation_before": int(generation_before),
                "generation_after": int(generation_after),
                **payload,
            },
        )
        self._switched = True

    def classify_obsolete_command(
        self,
        *,
        actual_target_step: int,
        source_request_id: int,
        source_generation_id: int,
        command: Sequence[float],
        obsolete_destination_xyz: Sequence[float],
        final_destination_xyz: Sequence[float],
    ) -> bool:
        values = _finite_command(command)
        obsolete = command_targets_obsolete_destination(
            values[:3],
            obsolete_destination_xyz=obsolete_destination_xyz,
            final_destination_xyz=final_destination_xyz,
        )
        self.append(
            "obsolete_command_classified",
            actual_target_step=int(actual_target_step),
            source_request_id=int(source_request_id),
            source_generation_id=int(source_generation_id),
            command=values,
            obsolete_destination_command=obsolete,
        )
        return obsolete

    def terminate(self, *, reason: str, success: bool, **payload: Any) -> None:
        if not self._started or self._terminated:
            raise RuntimeError("task termination requires one active episode")
        if not reason:
            raise ValueError("termination reason must be non-empty")
        terminal_step = payload.get("terminal_step", payload.get("episode_step"))
        if terminal_step is not None:
            self._terminal_step = int(terminal_step)
        self._append("task_terminated", {"reason": reason, "success": bool(success), **payload})
        self._terminated = True

    def finish(self, *, success: bool, completion_reason: str, **payload: Any) -> None:
        if not self._started or not self._terminated:
            raise RuntimeError("episode_end requires start and task termination")
        if not completion_reason:
            raise ValueError("completion_reason must be non-empty")
        if not self._switched:
            planned_switch = int(self.fairness["switch_step"])
            if success or self._terminal_step is None or self._terminal_step >= planned_switch:
                raise RuntimeError(
                    "zero-switch episode is valid only for a recorded failure before the planned switch"
                )
        self._append(
            "episode_end",
            {"success": bool(success), "completion_reason": completion_reason, **payload},
        )
        self._finished = True
        self._log.commit()

    def abort(self) -> None:
        self._log.abort()
        self._opened = False

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None or not self._finished:
            self.abort()
            if exc_type is None:
                raise RuntimeError("M8 episode recorder closed before episode_end")
        self._opened = False
        return False
