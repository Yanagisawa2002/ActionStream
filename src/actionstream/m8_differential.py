"""Deterministic M4-to-M8/C++ differential replay.

The M4 implementation is imported unchanged.  A tiny line-protocol probe links
the actual C++ state-machine library and returns an immutable snapshot after
each synthetic event.  This module normalizes only the representation boundary
that M4 actually supported; every newer semantic remains visible as an
intentional difference.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from actionstream.runtime import ActionQueue, InferenceResult, QueueNotReady


SCHEMA_VERSION = "actionstream.m8.differential_report.v1"
CANONICAL_M4_COMMIT = "d84cb64e9e48b681e083828b24df5a38709c78c8"
STARTING_M7_COMMIT = "1c51d3e0489a57224c4b4d2f4f6ddafae5302713"
GENERATION_MIGRATION_DISCREPANCY = (
    "generation_advance_retained_executable_old_generation_queue"
)
PROBE_COLUMNS = (
    "event_id",
    "operation",
    "accepted",
    "reason",
    "actions_expired",
    "duplicates_removed",
    "decision_queue_length",
    "command_present",
    "command_hold",
    "command_reason",
    "command_request",
    "command_generation",
    "command_source_observation",
    "command_source_target",
    "command_actual_target",
    "command_token",
    "episode_active",
    "episode_terminated",
    "latest_observation",
    "active_generation",
    "latest_source_present",
    "latest_source_observation",
    "latest_source_request",
    "latest_executed_present",
    "latest_executed_target",
    "sync_inflight_present",
    "sync_inflight_request",
    "queue",
    "events",
)

INTENTIONAL_DIFFERENCES = (
    {
        "code": "implicit_vs_explicit_targets",
        "m4": "Dense row position implicitly denotes the future action.",
        "cpp": "Every action carries an explicit validated target_step.",
    },
    {
        "code": "request_and_generation_provenance",
        "m4": "The queue stores no request ID or within-episode invalidation epoch.",
        "cpp": "Request IDs, generation IDs, and source provenance are mandatory.",
    },
    {
        "code": "bounded_cross_topic_registration_barrier",
        "m4": (
            "The single-process benchmark calls queue merge only after its request "
            "lifecycle has been created."
        ),
        "cpp": (
            "Because ROS does not order different topics, one response may be buffered "
            "until its provenance-carrying request callback registers, then is validated "
            "normally; the bounded barrier does not change request-before-response traces."
        ),
    },
    {
        "code": "same_generation_freshness",
        "m4": "Every same-episode arrival replaces the queue.",
        "cpp": "Freshness is ordered by (source_observation_step, request_id).",
    },
    {
        "code": "duplicate_response_detection",
        "m4": "A duplicate response is another queue replacement.",
        "cpp": "A completed request ID cannot update the queue twice.",
    },
    {
        "code": "explicit_duplicate_target_first_wins",
        "m4": "Duplicate labels are unrepresentable and row order is retained.",
        "cpp": "The first valid action for one target wins and targets are sorted.",
    },
    {
        "code": "generation_invalidation_absent_m4",
        "m4": "Worker generation changes only discard reset-era results.",
        "cpp": "A newer observation/request epoch invalidates the executable queue.",
    },
    {
        "code": "observation_generation_validation",
        "m4": "ActionQueue has no observation lifecycle method.",
        "cpp": "A stale-generation observation is rejected before command selection.",
    },
    {
        "code": "terminal_lifecycle_owned_by_cpp",
        "m4": "Benchmark orchestration stops merging results after task termination.",
        "cpp": "The state machine owns terminal state and clears runtime queues.",
    },
    {
        "code": "startup_hold_behavior",
        "m4": "next_action raises QueueNotReady before the first policy chunk.",
        "cpp": "The configured finite absolute safe hold is immediately available.",
    },
    {
        "code": "numeric_representation_and_zero_guard",
        "m4": "Commands are float32 and allclose-to-zero policy rows are rejected.",
        "cpp": "Policy commands are doubles validated for dimension and finiteness.",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_text(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _git_object_sha256(repository_root: Path, revision_path: str) -> str:
    completed = subprocess.run(
        ["git", "show", revision_path],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git show {revision_path} failed with exit {completed.returncode}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    return hashlib.sha256(completed.stdout).hexdigest()


def _normalize_reason(reason: str) -> str:
    aliases = {
        "old_episode": "previous_episode",
        "fully_stale": "fully_expired",
        "replaced": "aligned_atomic_rebuild",
        "episode_ended": "episode_terminated",
    }
    return aliases.get(reason, reason)


def load_fixture(path: Path | str) -> dict[str, Any]:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "actionstream.m8.differential_trace.v1":
        raise ValueError("unsupported differential trace schema")
    if payload.get("frozen") is not True:
        raise ValueError("differential trace must be frozen")
    if payload.get("action_dimension") != 7 or payload.get("action_horizon") != 30:
        raise ValueError("differential trace must preserve the canonical 30x7 contract")
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("differential trace must contain events")
    event_ids: set[str] = set()
    for event in events:
        event_id = str(event.get("event_id", ""))
        if not event_id or event_id in event_ids:
            raise ValueError(f"duplicate or empty event_id: {event_id!r}")
        event_ids.add(event_id)
        tick = int(event["tick"])
        expected_stamp = {
            "sim_time_ns": tick * 50_000_000,
            "steady_time_ns": 1_000 + tick,
            "wall_time_ns": 10_000 + tick,
        }
        if event.get("stamp") != expected_stamp:
            raise ValueError(f"event {event_id} has a noncanonical deterministic timestamp")
        if event["operation"] == "chunk":
            actions = event.get("actions", [])
            if len(actions) != 30:
                raise ValueError(f"event {event_id} must contain exactly 30 actions")
            for action in actions:
                if int(action["target_step"]) < 0 or float(action["token"]) <= 0:
                    raise ValueError(f"event {event_id} contains an invalid action token")
    return payload


def _probe_line(event: Mapping[str, Any]) -> str:
    operation = str(event["operation"]).upper()
    event_id = str(event["event_id"])
    if operation in {"START", "RESET"}:
        fields = (
            event_id,
            operation,
            event["episode_id"],
            event["generation_id"],
            event["observation_step"],
            event["tick"],
        )
    elif operation == "OBSERVE":
        fields = (
            event_id,
            operation,
            event["episode_id"],
            event["observation_step"],
            event["generation_id"],
            int(bool(event["terminated"])),
            event["tick"],
        )
    elif operation == "REQUEST":
        fields = (
            event_id,
            operation,
            event["episode_id"],
            event["request_id"],
            event["generation_id"],
            event["source_observation_step"],
            event["expected_horizon"],
            event["tick"],
        )
    elif operation == "CHUNK":
        action_specification = ",".join(
            f"{int(action['target_step'])}:{float(action['token']):g}"
            for action in event["actions"]
        )
        fields = (
            event_id,
            operation,
            event["episode_id"],
            event["request_id"],
            event["generation_id"],
            event["source_observation_step"],
            event["tick"],
            action_specification,
        )
    elif operation == "COMMAND":
        fields = (
            event_id,
            operation,
            event["episode_id"],
            event["actual_target_step"],
            event["tick"],
        )
    elif operation == "TERMINATE":
        fields = (
            event_id,
            operation,
            event["episode_id"],
            int(bool(event["success"])),
            event["tick"],
        )
    elif operation == "DROP":
        fields = (event_id, operation, event["request_id"])
    else:
        raise ValueError(f"unsupported operation: {operation}")
    return "|".join(str(field) for field in fields)


def _parse_queue(value: str) -> list[dict[str, Any]]:
    if not value:
        return []
    result = []
    for entry in value.split(";"):
        request, generation, source, target, token = entry.split(",")
        result.append(
            {
                "request_id": int(request),
                "generation_id": int(generation),
                "source_observation_step": int(source),
                "source_target_step": int(target),
                "token": int(round(float(token))),
            }
        )
    return result


def _parse_events(value: str) -> list[dict[str, Any]]:
    if not value:
        return []
    result = []
    for entry in value.split(";"):
        fields = entry.split("~")
        if len(fields) != 11:
            raise ValueError(f"malformed probe runtime event: {entry}")
        result.append(
            {
                "event_type": fields[0],
                "reason": fields[1],
                "request_id": int(fields[2]),
                "generation_id": int(fields[3]),
                "source_observation_step": int(fields[4]),
                "source_target_step": int(fields[5]),
                "actual_target_step": int(fields[6]),
                "active_generation_id": int(fields[7]),
                "queue_length_before": int(fields[8]),
                "queue_length_after": int(fields[9]),
                "action_count": int(fields[10]),
            }
        )
    return result


def _parse_probe_output(output: str) -> dict[str, dict[str, Any]]:
    lines = [line.rstrip("\r") for line in output.splitlines() if line.strip()]
    if not lines or tuple(lines[0].lstrip("\ufeff").split("|")) != PROBE_COLUMNS:
        raise ValueError("unexpected C++ replay probe header")
    parsed: dict[str, dict[str, Any]] = {}
    boolean_fields = {
        "accepted",
        "command_present",
        "command_hold",
        "episode_active",
        "episode_terminated",
        "latest_source_present",
        "latest_executed_present",
        "sync_inflight_present",
    }
    integer_fields = {
        "actions_expired",
        "duplicates_removed",
        "decision_queue_length",
        "command_request",
        "command_generation",
        "command_source_observation",
        "command_source_target",
        "command_actual_target",
        "latest_observation",
        "active_generation",
        "latest_source_observation",
        "latest_source_request",
        "latest_executed_target",
        "sync_inflight_request",
    }
    for line in lines[1:]:
        values = line.split("|")
        if len(values) != len(PROBE_COLUMNS):
            raise ValueError(f"malformed C++ replay probe row: {line}")
        row: dict[str, Any] = dict(zip(PROBE_COLUMNS, values, strict=True))
        for field in boolean_fields:
            row[field] = bool(int(row[field]))
        for field in integer_fields:
            row[field] = int(row[field])
        row["command_token"] = int(round(float(row["command_token"])))
        row["queue"] = _parse_queue(row["queue"])
        row["events"] = _parse_events(row["events"])
        event_id = str(row["event_id"])
        if event_id in parsed:
            raise ValueError(f"duplicate probe output event: {event_id}")
        parsed[event_id] = row
    return parsed


def run_cpp_probe(
    probe_path: Path | str, events: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    trace = "\n".join(_probe_line(event) for event in events) + "\n"
    completed = subprocess.run(
        [str(probe_path), "aligned_async", "0"],
        input=trace,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "C++ differential replay probe failed "
            f"with exit {completed.returncode}: {completed.stderr.strip()}"
        )
    parsed = _parse_probe_output(completed.stdout)
    expected_ids = [str(event["event_id"]) for event in events]
    if list(parsed) != expected_ids:
        raise ValueError("C++ replay probe did not return every event in order")
    return parsed


class _M4Adapter:
    """Replay trace events through the canonical M4 ActionQueue unchanged."""

    def __init__(self) -> None:
        self.queue = ActionQueue()
        self.episode_id: str | None = None
        self.current_step = 0
        self.terminated = False
        self.requests: dict[int, Mapping[str, Any]] = {}
        self.token_metadata: dict[int, dict[str, int]] = {}
        self.latest_source: tuple[int, int] | None = None
        self.latest_executed: int | None = None

    def _queue_snapshot(self) -> list[dict[str, Any]]:
        raw_queue = getattr(self.queue, "_queue")
        result = []
        for command in raw_queue:
            token = int(round(float(np.asarray(command)[0])))
            metadata = self.token_metadata[token]
            result.append({**metadata, "token": token})
        return result

    def _base_result(self) -> dict[str, Any]:
        return {
            "accepted": True,
            "reason": "accepted",
            "actions_expired": 0,
            "duplicates_removed": 0,
            "command_present": False,
            "command_hold": False,
            "command_reason": "",
            "command_request": 0,
            "command_generation": 0,
            "command_source_observation": 0,
            "command_source_target": 0,
            "command_actual_target": 0,
            "command_token": 0,
        }

    def _finish(self, result: dict[str, Any]) -> dict[str, Any]:
        result.update(
            {
                "episode_active": self.episode_id is not None and not self.terminated,
                "episode_terminated": self.terminated,
                "latest_observation": self.current_step,
                "active_generation": None,
                "latest_source_present": self.latest_source is not None,
                "latest_source_observation": (
                    0 if self.latest_source is None else self.latest_source[0]
                ),
                "latest_source_request": 0 if self.latest_source is None else self.latest_source[1],
                "latest_executed_present": self.latest_executed is not None,
                "latest_executed_target": self.latest_executed or 0,
                "queue": self._queue_snapshot(),
                "events": [],
            }
        )
        result["decision_queue_length"] = len(result["queue"])
        return result

    def apply(self, event: Mapping[str, Any]) -> dict[str, Any]:
        operation = str(event["operation"])
        result = self._base_result()
        if operation in {"start", "reset"}:
            self.episode_id = str(event["episode_id"])
            self.current_step = int(event["observation_step"])
            self.terminated = False
            self.latest_source = None
            self.latest_executed = None
            self.queue.reset_episode(self.episode_id)
        elif operation == "observe":
            if str(event["episode_id"]) != self.episode_id or self.terminated:
                result["accepted"] = False
                result["reason"] = "episode_ended" if self.terminated else "old_episode"
            else:
                self.current_step = int(event["observation_step"])
        elif operation == "request":
            self.requests[int(event["request_id"])] = event
            if str(event["episode_id"]) != self.episode_id or self.terminated:
                result["accepted"] = False
                result["reason"] = "episode_ended" if self.terminated else "old_episode"
        elif operation == "chunk":
            for action in event["actions"]:
                token = int(round(float(action["token"])))
                metadata = {
                    "request_id": int(event["request_id"]),
                    "generation_id": int(event["generation_id"]),
                    "source_observation_step": int(event["source_observation_step"]),
                    "source_target_step": int(action["target_step"]),
                }
                previous = self.token_metadata.get(token)
                if previous is not None and previous != metadata:
                    raise ValueError(f"token {token} has conflicting provenance")
                self.token_metadata[token] = metadata
            if str(event["episode_id"]) == self.episode_id and self.terminated:
                result["accepted"] = False
                result["reason"] = "episode_ended"
            else:
                request = self.requests.get(int(event["request_id"]), event)
                request_time = float(request.get("ordinal", 0))
                delivery_time = max(request_time + 0.03, float(event["ordinal"]) + 0.03)
                actions = np.asarray(
                    [[float(action["token"])] * 7 for action in event["actions"]],
                    dtype=np.float32,
                )
                outcome = self.queue.replace(
                    InferenceResult(
                        actions=actions,
                        episode_id=str(event["episode_id"]),
                        observation_control_step=int(event["source_observation_step"]),
                        request_timestamp=request_time,
                        start_timestamp=request_time + 0.01,
                        end_timestamp=delivery_time - 0.01,
                        delivery_timestamp=delivery_time,
                        model_inference_latency_seconds=max(
                            0.0, delivery_time - request_time - 0.02
                        ),
                    ),
                    current_control_step=self.current_step,
                    mode="async_aligned",
                )
                result.update(
                    {
                        "accepted": outcome.accepted,
                        "reason": outcome.reason,
                        "actions_expired": outcome.dropped_prefix_steps,
                        "duplicates_removed": 0,
                    }
                )
                if outcome.accepted:
                    self.latest_source = (
                        int(event["source_observation_step"]),
                        int(event["request_id"]),
                    )
        elif operation == "command":
            try:
                command, held = self.queue.next_action()
            except QueueNotReady:
                result.update(
                    {"accepted": False, "reason": "queue_not_ready", "command_reason": "queue_not_ready"}
                )
            else:
                token = int(round(float(command[0])))
                metadata = self.token_metadata[token]
                result.update(
                    {
                        "reason": "no_state_decision",
                        "command_present": True,
                        "command_hold": held,
                        "command_reason": "queue_empty" if held else "target_step_match",
                        "command_request": metadata["request_id"],
                        "command_generation": metadata["generation_id"],
                        "command_source_observation": metadata["source_observation_step"],
                        "command_source_target": metadata["source_target_step"],
                        "command_actual_target": int(event["actual_target_step"]),
                        "command_token": token,
                    }
                )
                self.latest_executed = int(event["actual_target_step"])
        elif operation == "terminate":
            if str(event["episode_id"]) != self.episode_id or self.terminated:
                result["accepted"] = False
                result["reason"] = "old_episode" if not self.terminated else "episode_ended"
            else:
                self.terminated = True
                result["reason"] = "terminated"
        elif operation == "drop":
            result["reason"] = "no_state_decision"
        else:
            raise ValueError(f"unsupported M4 trace operation: {operation}")
        return self._finish(result)


def run_m4(events: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    adapter = _M4Adapter()
    return {str(event["event_id"]): adapter.apply(event) for event in events}


def _event_comparison(
    event: Mapping[str, Any], m4: Mapping[str, Any], cpp: Mapping[str, Any]
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    operation = str(event["operation"])
    if operation != "drop":
        checks["accepted"] = bool(m4["accepted"]) == bool(cpp["accepted"])
        checks["normalized_reason"] = _normalize_reason(str(m4["reason"])) == _normalize_reason(
            str(cpp["reason"])
        )
    checks["queue_tokens"] = [row["token"] for row in m4["queue"]] == [
        row["token"] for row in cpp["queue"]
    ]
    checks["queue_provenance"] = m4["queue"] == cpp["queue"]
    checks["episode_active"] = bool(m4["episode_active"]) == bool(cpp["episode_active"])
    checks["episode_terminated"] = bool(m4["episode_terminated"]) == bool(
        cpp["episode_terminated"]
    )
    checks["latest_observation"] = int(m4["latest_observation"]) == int(
        cpp["latest_observation"]
    )
    if operation == "chunk":
        checks["expired_prefix"] = int(m4["actions_expired"]) == int(cpp["actions_expired"])
        checks["duplicates_removed"] = int(m4["duplicates_removed"]) == int(
            cpp["duplicates_removed"]
        )
    if operation == "command":
        for field in (
            "command_present",
            "command_hold",
            "command_request",
            "command_generation",
            "command_source_observation",
            "command_source_target",
            "command_actual_target",
            "command_token",
        ):
            checks[field] = m4[field] == cpp[field]
    return {
        "all_equal": all(checks.values()),
        "checks": checks,
        "different_fields": sorted(field for field, equal in checks.items() if not equal),
    }


def build_report(
    *,
    fixture_path: Path | str,
    probe_path: Path | str,
    repository_root: Path | str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    fixture_file = Path(fixture_path).resolve()
    probe_file = Path(probe_path).resolve()
    fixture = load_fixture(fixture_file)
    events = fixture["events"]
    m4_rows = run_m4(events)
    cpp_rows = run_cpp_probe(probe_file, events)

    transcript = []
    common_mismatches = []
    observed_difference_codes: set[str] = set()
    for event in events:
        event_id = str(event["event_id"])
        comparison = _event_comparison(event, m4_rows[event_id], cpp_rows[event_id])
        if event["domain"] == "common" and not comparison["all_equal"]:
            common_mismatches.append(
                {"event_id": event_id, "different_fields": comparison["different_fields"]}
            )
        expected_difference = event.get("expected_difference")
        if expected_difference and not comparison["all_equal"]:
            observed_difference_codes.add(str(expected_difference))
        transcript.append(
            {
                "event": event,
                "m4": m4_rows[event_id],
                "cpp": cpp_rows[event_id],
                "comparison": comparison,
            }
        )

    expected_difference_codes = set(fixture["intentional_difference_codes"])
    missing_expected_differences = sorted(expected_difference_codes - observed_difference_codes)
    generation_row = cpp_rows["x059"]
    generation_events = generation_row["events"]
    generation_advanced_indices = [
        index
        for index, row in enumerate(generation_events)
        if row["event_type"] == "generation_advanced"
    ]
    generation_invalidated_indices = [
        index
        for index, row in enumerate(generation_events)
        if row["event_type"] == "generation_invalidated"
    ]
    generation_advanced = bool(generation_advanced_indices)
    invalidated_count = sum(
        row["event_type"] == "generation_invalidated" for row in generation_events
    )
    generation_queue_clean = not generation_row["queue"]
    observation_invalidation_ordered = (
        generation_row["accepted"]
        and generation_advanced
        and bool(generation_invalidated_indices)
        and generation_advanced_indices[0] < generation_invalidated_indices[0]
        and generation_queue_clean
    )
    post_switch_command_current = (
        cpp_rows["x063"]["command_present"]
        and not cpp_rows["x063"]["command_hold"]
        and cpp_rows["x063"]["command_generation"] == 2
    )
    stale_observation_rejected = (
        not cpp_rows["x067"]["accepted"]
        and cpp_rows["x067"]["reason"] == "stale_observation_generation"
    )

    unintended_discrepancies = []
    if common_mismatches:
        unintended_discrepancies.append("common_domain_behavior_mismatch")
    if not observation_invalidation_ordered:
        unintended_discrepancies.append("generation_advance_did_not_atomically_purge_queue")
    if not post_switch_command_current:
        unintended_discrepancies.append("old_generation_command_executed_after_switch")
    if not stale_observation_rejected:
        unintended_discrepancies.append("stale_observation_generation_not_rejected")
    if missing_expected_differences:
        unintended_discrepancies.append("expected_extended_difference_not_observed")

    canonical_runtime_revision = f"{CANONICAL_M4_COMMIT}:src/actionstream/runtime.py"
    canonical_benchmark_revision = f"{CANONICAL_M4_COMMIT}:src/actionstream/benchmark.py"
    pre_fix_cpp_path = "ros2_ws/src/action_stream_executor/src/executor_state_machine.cpp"
    pre_fix_cpp_revision = f"{STARTING_M7_COMMIT}:{pre_fix_cpp_path}"
    working_runtime_path = root / "src/actionstream/runtime.py"
    canonical_runtime_sha256 = _git_object_sha256(root, canonical_runtime_revision)
    working_runtime_sha256 = _sha256(working_runtime_path)
    runtime_matches_accepted_m4 = canonical_runtime_sha256 == working_runtime_sha256
    if not runtime_matches_accepted_m4:
        unintended_discrepancies.append("working_m4_runtime_differs_from_accepted_m4")

    source_paths = {
        "working_m4_runtime": working_runtime_path,
        "cpp_header": root
        / "ros2_ws/src/action_stream_executor/include/action_stream_executor/executor_state_machine.hpp",
        "cpp_implementation": root
        / "ros2_ws/src/action_stream_executor/src/executor_state_machine.cpp",
        "cpp_probe_source": root
        / "ros2_ws/src/action_stream_executor/src/differential_replay_probe.cpp",
        "executor_node": root
        / "ros2_ws/src/action_stream_executor/src/executor_node.cpp",
        "observation_message": root
        / "ros2_ws/src/action_stream_msgs/msg/Observation.msg",
    }
    source_hashes = {
        name: {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
        for name, path in source_paths.items()
    }
    headline_evaluation_unblocked = not unintended_discrepancies
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": "M8-G0",
        "phase": "Phase 0 M4-to-M7 differential replay",
        "full_behavioral_equivalence": False,
        "common_domain_behavioral_equivalence": not common_mismatches,
        "headline_evaluation_unblocked": headline_evaluation_unblocked,
        "normalization": fixture["normalization"],
        "exact_common_domain": fixture["exact_common_domain"],
        "fixture": {
            "path": fixture_file.relative_to(root).as_posix(),
            "sha256": _sha256(fixture_file),
            "event_count": len(events),
            "common_event_count": sum(event["domain"] == "common" for event in events),
            "extended_event_count": sum(event["domain"] == "extended" for event in events),
        },
        "cpp_probe": {"filename": probe_file.name, "sha256": _sha256(probe_file)},
        "source_hashes": source_hashes,
        "canonical_m4": {
            "accepted_commit": CANONICAL_M4_COMMIT,
            "runtime": {
                "path": "src/actionstream/runtime.py",
                "git_blob": _git_text(root, "rev-parse", canonical_runtime_revision),
                "sha256": canonical_runtime_sha256,
                "working_tree_sha256": working_runtime_sha256,
                "working_tree_byte_identical": runtime_matches_accepted_m4,
            },
            "historical_benchmark": {
                "path": "src/actionstream/benchmark.py",
                "git_blob": _git_text(root, "rev-parse", canonical_benchmark_revision),
                "sha256": _git_object_sha256(root, canonical_benchmark_revision),
            },
            "adapter_statement": (
                (
                    "The Phase 0 adapter instantiates ActionQueue and InferenceResult "
                    "from a byte-identical checkout of the accepted M4 runtime."
                )
                if runtime_matches_accepted_m4
                else (
                    "The adapter executed the current runtime, whose tracked bytes no "
                    "longer match the accepted M4 commit. Common-domain checks remain "
                    "useful regression evidence, but this run is not an exact canonical "
                    "M4 replay and remains explicitly blocked."
                )
            ),
            "working_benchmark_not_used_as_executor": True,
        },
        "common_mismatches": common_mismatches,
        "intentional_differences": list(INTENTIONAL_DIFFERENCES),
        "expected_trace_difference_codes": sorted(expected_difference_codes),
        "observed_trace_difference_codes": sorted(observed_difference_codes),
        "missing_expected_trace_difference_codes": missing_expected_differences,
        "migration_corrections": {
            "observation_generation_orders_invalidation_before_command": (
                observation_invalidation_ordered and post_switch_command_current
            ),
            "generation_advanced_event_emitted": generation_advanced,
            "generation_invalidated_event_count": invalidated_count,
            "queue_empty_immediately_after_generation_advance": generation_queue_clean,
            "first_post_switch_policy_command_uses_active_generation": post_switch_command_current,
            "stale_observation_generation_rejected": stale_observation_rejected,
        },
        "unintended_discrepancies_found": [GENERATION_MIGRATION_DISCREPANCY],
        "migration_discrepancy_history": [
            {
                "discrepancy_id": GENERATION_MIGRATION_DISCREPANCY,
                "classification": "unintended_m7_migration_defect",
                "discovered_during_phase_0_before_benchmark_design": True,
                "affected_pre_fix_revision": {
                    "commit": STARTING_M7_COMMIT,
                    "path": pre_fix_cpp_path,
                    "git_blob": _git_text(root, "rev-parse", pre_fix_cpp_revision),
                    "sha256": _git_object_sha256(root, pre_fix_cpp_revision),
                },
                "pre_fix_probe_retained": False,
                "pre_fix_evidence_kind": "source_code_inspection",
                "pre_fix_behavior": (
                    "A newer request generation advanced active_generation_id and reset "
                    "freshness, but retained executable actions from the older generation; "
                    "command selection had no defensive generation guard."
                ),
                "benchmark_risk": (
                    "Aligned execution could issue an obsolete pre-switch command before "
                    "the first new-generation response, directly confounding the proposed "
                    "destination-switch comparison."
                ),
                "trigger_trace_event_id": "x059",
                "fix_summary": (
                    "Propagate generation on observations, advance it before command "
                    "selection, atomically invalidate non-naive old-generation queue "
                    "entries, and retain a defensive execution-time purge."
                ),
                "fixed_files": [
                    "ros2_ws/src/action_stream_msgs/msg/Observation.msg",
                    "ros2_ws/src/action_stream_executor/include/action_stream_executor/executor_state_machine.hpp",
                    "ros2_ws/src/action_stream_executor/src/executor_state_machine.cpp",
                    "ros2_ws/src/action_stream_executor/src/executor_node.cpp",
                ],
                "regression_tests": [
                    "GenerationTest.ObservationAdvancePurgesAlignedQueueBeforeNextCommand",
                    "GenerationTest.NaiveObservationAdvanceRetainsArrivalOrderedQueue",
                    "GenerationTest.StaleObservationCannotDispatchAfterGenerationAdvance",
                    "test_actual_cpp_probe_has_exact_common_domain_and_fixed_generation_guard",
                ],
                "post_fix_probe_evidence": {
                    "generation_advance_event_id": "x059",
                    "old_generation_actions_invalidated": invalidated_count,
                    "queue_empty_after_advance": generation_queue_clean,
                    "first_post_switch_command_event_id": "x063",
                    "first_post_switch_command_generation": cpp_rows["x063"][
                        "command_generation"
                    ],
                    "active_generation": 2,
                    "stale_observation_rejection_event_id": "x067",
                },
                "status": "resolved",
            }
        ],
        "unresolved_unintended_discrepancies": unintended_discrepancies,
        "unintended_discrepancies": unintended_discrepancies,
        "events": transcript,
    }


def write_report_atomic(report: Mapping[str, Any], output_path: Path | str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        fixture_path=args.fixture,
        probe_path=args.probe,
        repository_root=args.repository_root,
    )
    write_report_atomic(report, args.output)
    if not report["headline_evaluation_unblocked"]:
        raise RuntimeError(
            "Phase 0 differential replay has unresolved discrepancies: "
            f"{report['unintended_discrepancies']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
