"""Independent event-log replay and metric validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .plant import CONTROL_FREQUENCY_HZ
from .schema import MILESTONE, SCHEMA_VERSION, read_json, read_jsonl, write_json_atomic


_COUNTED_EVENTS = {
    "inference_requests": ("inference_request", None),
    "returned_responses": ("chunk_arrived", None),
    "dropped_responses": ("response_dropped", None),
    "out_of_order_responses": ("chunk_arrived", ("out_of_order", True)),
    "duplicate_responses": ("chunk_rejected", ("reason", "duplicate_response")),
    "accepted_chunks": ("queue_updated", None),
    "rejected_chunks": ("chunk_rejected", None),
    "stale_generations_rejected": ("chunk_rejected", ("reason", "stale_generation")),
    "superseded_sources_rejected": (
        "chunk_rejected",
        ("reason", "superseded_source_observation"),
    ),
    "queue_rebuild_count": ("queue_updated", ("reason__not", "arrival_order_append")),
}


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _matches(row: dict[str, Any], event_type: str, condition: tuple[str, Any] | None) -> bool:
    if row.get("event_type") != event_type:
        return False
    if condition is None:
        return True
    key, expected = condition
    if key.endswith("__not"):
        return row.get(key.removesuffix("__not")) != expected
    return row.get(key) == expected


def recompute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("event log is empty")
    starts = [row for row in rows if row.get("event_type") == "episode_start"]
    ends = [row for row in rows if row.get("event_type") == "episode_end"]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("event log must contain exactly one episode_start and episode_end")
    commands = [row for row in rows if row.get("event_type") == "command_executed"]
    policy_commands = [row for row in commands if not bool(row.get("hold"))]
    hold_commands = [row for row in commands if bool(row.get("hold"))]
    action_ages = [
        float(row["actual_target_step"] - row["source_observation_step"])
        for row in policy_commands
    ]
    latencies = [
        float(row["latency_ms"])
        for row in rows
        if row.get("event_type") == "chunk_arrived" and not bool(row.get("duplicate"))
    ]
    expired_ingest = sum(
        int(row.get("expired_actions_removed", 0))
        for row in rows
        if row.get("event_type") == "queue_updated"
    )
    expired_execution = sum(
        1
        for row in rows
        if row.get("event_type") == "action_discarded"
        and row.get("reason") == "expired_before_execution"
    )
    duplicate_actions = sum(
        int(row.get("duplicate_target_actions_removed", 0))
        for row in rows
        if row.get("event_type") == "queue_updated"
    )
    sync_wait_ns = sum(
        int(row.get("duration_ns", 0))
        for row in rows
        if row.get("event_type") == "sync_wait" and bool(row.get("counts_as_hold"))
    )
    counters: dict[str, Any] = {}
    for metric, (event_type, condition) in _COUNTED_EVENTS.items():
        counters[metric] = sum(_matches(row, event_type, condition) for row in rows)
    simulation_steps = len(commands)
    start_wall_time_ns = int(starts[0].get("wall_time_ns", 0))
    wall_time_ns = max(int(row.get("wall_time_ns", 0)) for row in rows) - start_wall_time_ns
    counters.update(
        {
            "task_success": bool(ends[0].get("success")),
            "completion_reason": str(ends[0].get("completion_reason", "")),
            "simulation_steps": simulation_steps,
            "wall_clock_seconds": wall_time_ns / 1_000_000_000,
            "simulation_seconds": simulation_steps / CONTROL_FREQUENCY_HZ,
            "total_hold_seconds": (
                sync_wait_ns + len(hold_commands) * int(1_000_000_000 / CONTROL_FREQUENCY_HZ)
            )
            / 1_000_000_000,
            "expired_actions_removed": expired_ingest + expired_execution,
            "duplicate_target_actions_removed": duplicate_actions,
            "deadline_misses": len(hold_commands),
            "hold_steps": len(hold_commands),
            "executed_actions": len(policy_commands),
            "mean_action_age_steps": (sum(action_ages) / len(action_ages) if action_ages else 0.0),
            "p50_action_age_steps": _percentile(action_ages, 0.50),
            "p95_action_age_steps": _percentile(action_ages, 0.95),
            "median_action_age_steps": _percentile(action_ages, 0.50),
            "mean_inference_latency_ms": (sum(latencies) / len(latencies) if latencies else 0.0),
            "p50_inference_latency_ms": _percentile(latencies, 0.50),
            "p95_inference_latency_ms": _percentile(latencies, 0.95),
            "deadline_miss_rate": (
                len(hold_commands) / simulation_steps if simulation_steps else 0.0
            ),
        }
    )
    return counters


def _metric_equal(recorded: Any, recomputed: Any) -> bool:
    if isinstance(recorded, bool) or isinstance(recomputed, bool):
        return recorded is recomputed
    if isinstance(recorded, (int, float)) and isinstance(recomputed, (int, float)):
        return abs(float(recorded) - float(recomputed)) <= 1e-9 * max(
            1.0, abs(float(recorded)), abs(float(recomputed))
        )
    return recorded == recomputed


def validate_episode_log(
    event_log_path: Path | str,
    summary_path: Path | str,
) -> dict[str, Any]:
    rows = read_jsonl(event_log_path)
    summary = read_json(summary_path)
    recomputed = recompute_metrics(rows)
    recorded = summary.get("metrics", {})
    mismatches = {
        key: {"recorded": recorded.get(key), "recomputed": value}
        for key, value in recomputed.items()
        if key not in recorded or not _metric_equal(recorded[key], value)
    }

    active_episode = ""
    active_generation = 0
    rejected_requests: set[int] = set()
    executed_targets: Counter[int] = Counter()
    violations: list[dict[str, Any]] = []
    for row in rows:
        event_type = row.get("event_type")
        if event_type == "episode_start":
            active_episode = str(row.get("episode_id", ""))
            active_generation = int(row.get("generation_id", 0))
        elif event_type == "generation_advanced":
            active_generation = int(row["generation_id"])
        elif event_type == "chunk_rejected" and row.get("reason") != "duplicate_response":
            rejected_requests.add(int(row.get("request_id", 0)))
        elif event_type == "command_executed":
            actual = int(row["actual_target_step"])
            executed_targets[actual] += 1
            if str(row.get("episode_id")) != active_episode:
                violations.append(
                    {
                        "event_index": row["event_index"],
                        "reason": "previous_episode_execution",
                    }
                )
            if bool(row.get("hold")):
                continue
            request_id = int(row.get("source_request_id", 0))
            source_generation = int(row.get("source_generation_id", 0))
            source_target = int(row.get("source_target_step", 0))
            if request_id in rejected_requests:
                violations.append(
                    {
                        "event_index": row["event_index"],
                        "reason": "rejected_response_execution",
                    }
                )
            if source_generation != active_generation:
                violations.append(
                    {
                        "event_index": row["event_index"],
                        "reason": "stale_generation_execution",
                    }
                )
            if source_target < actual:
                violations.append(
                    {
                        "event_index": row["event_index"],
                        "reason": "expired_action_execution",
                        "source_target_step": source_target,
                        "actual_target_step": actual,
                    }
                )
    for target, count in executed_targets.items():
        if count > 1:
            violations.append(
                {
                    "event_index": None,
                    "reason": "duplicate_actual_target_execution",
                    "target": target,
                }
            )

    strategy = str(summary.get("strategy"))
    violation_counts = Counter(str(item["reason"]) for item in violations)
    always_fatal = {
        "previous_episode_execution",
        "rejected_response_execution",
        "stale_generation_execution",
        "duplicate_actual_target_execution",
    }
    semantic_fatal = set(always_fatal)
    if strategy != "naive_async":
        semantic_fatal.add("expired_action_execution")
    fatal_count = sum(violation_counts[reason] for reason in semantic_fatal)
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "event_log": str(Path(event_log_path)),
        "summary": str(Path(summary_path)),
        "strategy": strategy,
        "event_count": len(rows),
        "metrics_match": not mismatches,
        "metric_mismatches": mismatches,
        "invariant_violation_counts": dict(sorted(violation_counts.items())),
        "violations": violations,
        "naive_expired_execution_is_expected_baseline_behavior": strategy == "naive_async",
        "semantic_invariants_pass": fatal_count == 0,
        "passed": not mismatches and fatal_count == 0,
    }


def write_summary_from_event_log(
    event_log_path: Path | str,
    summary_path: Path | str,
    *,
    profile_id: str,
    profile: dict[str, Any],
    seed: int,
    strategy: str,
    trace_sha256: str,
    evidence_class: str = "ros_cpp_test_plant",
    request_interval_steps: int = 10,
) -> dict[str, Any]:
    rows = read_jsonl(event_log_path)
    metrics = recompute_metrics(rows)
    starts = [row for row in rows if row.get("event_type") == "episode_start"]
    if len(starts) != 1:
        raise ValueError("ROS event log must contain exactly one episode_start")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "evidence_class": evidence_class,
        "not_isaac_sim_result": evidence_class != "ros_cpp_isaac_sim",
        "profile_id": profile_id,
        "profile": profile,
        "seed": int(seed),
        "strategy": strategy,
        "episode_id": str(starts[0]["episode_id"]),
        "trace_sha256": trace_sha256,
        "request_interval_steps": request_interval_steps,
        "metrics": metrics,
    }
    write_json_atomic(summary_path, summary)
    return summary


def validate_manifest(
    manifest_path: Path | str,
    *,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    audits = []
    for episode in manifest.get("episodes", []):
        event_path = Path(episode["event_log"])
        summary_path = Path(episode["summary"])
        if not event_path.is_absolute():
            event_path = manifest_file.parent / event_path
        if not summary_path.is_absolute():
            summary_path = manifest_file.parent / summary_path
        audits.append(validate_episode_log(event_path, summary_path))
    result = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "manifest": str(manifest_file),
        "episode_count": len(audits),
        "passed": bool(audits) and all(audit["passed"] for audit in audits),
        "audits": audits,
    }
    if output_path is not None:
        write_json_atomic(output_path, result)
    return result
