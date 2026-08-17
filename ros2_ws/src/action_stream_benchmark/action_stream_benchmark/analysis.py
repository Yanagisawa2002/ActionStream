"""Paired, seed-level analysis for M7 benchmark artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from statistics import median
from typing import Any, Iterable

from .schema import (
    MILESTONE,
    SCHEMA_VERSION,
    read_json,
    read_jsonl,
    write_json_atomic,
)


REQUIRED_STRATEGIES = ("sync_hold", "naive_async", "aligned_async")
REQUIRED_EPISODE_METRICS = (
    "task_success",
    "completion_reason",
    "simulation_steps",
    "wall_clock_seconds",
    "simulation_seconds",
    "total_hold_seconds",
    "inference_requests",
    "returned_responses",
    "dropped_responses",
    "out_of_order_responses",
    "duplicate_responses",
    "stale_generations_rejected",
    "expired_actions_removed",
    "duplicate_target_actions_removed",
    "queue_rebuild_count",
    "mean_action_age_steps",
    "p50_action_age_steps",
    "p95_action_age_steps",
    "mean_inference_latency_ms",
    "p50_inference_latency_ms",
    "p95_inference_latency_ms",
    "deadline_miss_rate",
    "executed_actions",
)


def paired_bootstrap_ci(
    differences: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = 20_000,
    seed: int = 20260803,
) -> tuple[float, float]:
    values = tuple(float(value) for value in differences)
    if not values:
        raise ValueError("paired bootstrap requires at least one pair")
    if not 0.0 < confidence < 1.0 or resamples < 1_000:
        raise ValueError("invalid confidence or bootstrap resample count")
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        estimates.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    lower = estimates[int(alpha * (resamples - 1))]
    upper = estimates[int((1.0 - alpha) * (resamples - 1))]
    return lower, upper


def _resolve_summary(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _load_rows(manifest_path: Path | str) -> list[dict[str, Any]]:
    path = Path(manifest_path)
    manifest = read_json(path)
    rows = []
    for entry in manifest.get("episodes", []):
        summary = read_json(_resolve_summary(path, entry["summary"]))
        if "event_log" in entry:
            summary["_event_log_path"] = str(_resolve_summary(path, str(entry["event_log"])))
        rows.append(summary)
    if not rows:
        raise ValueError("manifest contains no episode summaries")
    return rows


def _summarize_metric_values(values: list[Any]) -> dict[str, Any]:
    """Return a compact, lossless-with-respect-to-count descriptive summary."""

    if all(isinstance(value, bool) for value in values):
        true_count = sum(values)
        return {
            "kind": "boolean",
            "count": len(values),
            "true_count": true_count,
            "false_count": len(values) - true_count,
            "rate": true_count / len(values),
        }
    if all(isinstance(value, str) for value in values):
        counts: dict[str, int] = defaultdict(int)
        for value in values:
            counts[value] += 1
        return {
            "kind": "categorical",
            "count": len(values),
            "counts": dict(sorted(counts.items())),
        }
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        numeric = [float(value) for value in values]
        return {
            "kind": "numeric",
            "count": len(numeric),
            "sum": sum(numeric),
            "mean": sum(numeric) / len(numeric),
            "median": median(numeric),
            "min": min(numeric),
            "max": max(numeric),
        }
    raise ValueError(
        "cannot summarize metric with mixed or unsupported value types: "
        f"{sorted({type(value).__name__ for value in values})}"
    )


def _descriptive_metrics(method_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_key_sets = [set(row["metrics"]) for row in method_rows]
    # Keep the error tied to seeds rather than only reporting a set
    # intersection, which makes malformed artifacts straightforward to audit.
    missing_required = {
        metric: [int(row["seed"]) for row in method_rows if metric not in row["metrics"]]
        for metric in REQUIRED_EPISODE_METRICS
        if any(metric not in row["metrics"] for row in method_rows)
    }
    if missing_required:
        raise ValueError(f"episode summaries missing required metrics: {missing_required}")
    if any(keys != metric_key_sets[0] for keys in metric_key_sets[1:]):
        raise ValueError("episode summaries have inconsistent metric fields")
    return {
        metric: _summarize_metric_values([row["metrics"][metric] for row in method_rows])
        for metric in sorted(metric_key_sets[0])
    }


def _source_generation_summary(
    method_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    generation_counts: dict[int, int] = defaultdict(int)
    policy_execution_count = 0
    missing_generation_count = 0
    missing_event_logs: list[int] = []
    for row in method_rows:
        event_log_path = row.get("_event_log_path")
        if event_log_path is None:
            missing_event_logs.append(int(row["seed"]))
            continue
        for event in read_jsonl(event_log_path):
            if event.get("event_type") != "command_executed" or bool(event.get("hold")):
                continue
            policy_execution_count += 1
            if "source_generation_id" not in event:
                missing_generation_count += 1
                continue
            generation_counts[int(event["source_generation_id"])] += 1
    recorded_executed_actions = sum(int(row["metrics"]["executed_actions"]) for row in method_rows)
    return {
        "source_field": "command_executed.source_generation_id",
        "recorded_executed_actions": recorded_executed_actions,
        "event_log_policy_executions": policy_execution_count,
        "events_with_source_generation": sum(generation_counts.values()),
        "events_missing_source_generation": missing_generation_count,
        "missing_event_log_seeds": sorted(missing_event_logs),
        "generation_counts": {
            str(generation): count for generation, count in sorted(generation_counts.items())
        },
        "all_executed_actions_accounted_for": (
            not missing_event_logs
            and missing_generation_count == 0
            and policy_execution_count == recorded_executed_actions
        ),
    }


def analyze_manifest(
    manifest_path: Path | str,
    *,
    output_path: Path | str | None = None,
    bootstrap_resamples: int = 20_000,
) -> dict[str, Any]:
    rows = _load_rows(manifest_path)
    recorded_metric_fields = set(rows[0]["metrics"])
    if any(set(row["metrics"]) != recorded_metric_fields for row in rows[1:]):
        raise ValueError("episode summaries have inconsistent metric fields")
    evidence_classes = sorted({str(row.get("evidence_class", "unknown")) for row in rows})
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_profile[str(row["profile_id"])].append(row)

    profiles: dict[str, Any] = {}
    for profile_id, profile_rows in sorted(by_profile.items()):
        paired: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in profile_rows:
            paired[int(row["seed"])][str(row["strategy"])] = row
        incomplete = {
            seed: sorted(set(REQUIRED_STRATEGIES) - set(methods))
            for seed, methods in paired.items()
            if set(methods) != set(REQUIRED_STRATEGIES)
        }
        if incomplete:
            raise ValueError(f"unpaired profile {profile_id}: {incomplete}")
        seeds = sorted(paired)
        strategy_summary: dict[str, Any] = {}
        for strategy in REQUIRED_STRATEGIES:
            method_rows = [paired[seed][strategy] for seed in seeds]
            metrics = [row["metrics"] for row in method_rows]
            successes = sum(bool(metric["task_success"]) for metric in metrics)
            strategy_summary[strategy] = {
                "successes": successes,
                "trials": len(metrics),
                "success_rate": successes / len(metrics),
                "median_wall_clock_seconds": median(
                    float(metric["wall_clock_seconds"]) for metric in metrics
                ),
                "median_simulation_steps": median(
                    int(metric["simulation_steps"]) for metric in metrics
                ),
                "mean_total_hold_seconds": sum(
                    float(metric["total_hold_seconds"]) for metric in metrics
                )
                / len(metrics),
                "mean_deadline_miss_rate": sum(
                    float(metric["deadline_miss_rate"]) for metric in metrics
                )
                / len(metrics),
                "descriptive_metrics": _descriptive_metrics(method_rows),
                "executed_action_source_generations": (_source_generation_summary(method_rows)),
            }

        success_differences = [
            float(paired[seed]["aligned_async"]["metrics"]["task_success"])
            - float(paired[seed]["naive_async"]["metrics"]["task_success"])
            for seed in seeds
        ]
        ci_low, ci_high = paired_bootstrap_ci(
            success_differences,
            resamples=bootstrap_resamples,
            seed=20260803 + sum(ord(character) for character in profile_id),
        )
        sync_hold = strategy_summary["sync_hold"]["mean_total_hold_seconds"]
        aligned_hold = strategy_summary["aligned_async"]["mean_total_hold_seconds"]
        hold_reduction = (
            1.0
            if sync_hold == 0.0 and aligned_hold == 0.0
            else (0.0 if sync_hold == 0.0 else 1.0 - aligned_hold / sync_hold)
        )
        aligned_wall = strategy_summary["aligned_async"]["median_wall_clock_seconds"]
        sync_wall = strategy_summary["sync_hold"]["median_wall_clock_seconds"]
        wall_reduction = 0.0 if sync_wall == 0.0 else 1.0 - aligned_wall / sync_wall
        difference = sum(success_differences) / len(success_differences)
        profiles[profile_id] = {
            "paired_trial_count": len(seeds),
            "seeds": seeds,
            "strategies": strategy_summary,
            "aligned_minus_naive_success_rate": difference,
            "aligned_minus_naive_percentage_points": difference * 100.0,
            "paired_bootstrap_95_ci": [ci_low, ci_high],
            "aligned_hold_reduction_vs_sync": hold_reduction,
            "aligned_median_wall_reduction_vs_sync": wall_reduction,
            "raw_paired_outcomes": [
                {
                    "seed": seed,
                    **{
                        strategy: bool(paired[seed][strategy]["metrics"]["task_success"])
                        for strategy in REQUIRED_STRATEGIES
                    },
                }
                for seed in seeds
            ],
        }

    profile_b = profiles.get("profile_b")
    gate: dict[str, Any]
    if profile_b is None:
        gate = {
            "evaluated": False,
            "passed": False,
            "reason": "profile_b_missing",
        }
    else:
        aligned_rate = profile_b["strategies"]["aligned_async"]["success_rate"]
        sync_rate = profile_b["strategies"]["sync_hold"]["success_rate"]
        conditions = {
            "success_difference_at_least_20pp": (
                profile_b["aligned_minus_naive_success_rate"] >= 0.20
            ),
            "paired_ci_excludes_zero": (profile_b["paired_bootstrap_95_ci"][0] > 0.0),
            "hold_reduction_at_least_50pct": (profile_b["aligned_hold_reduction_vs_sync"] >= 0.50),
        }
        primary_pass = all(conditions.values())
        strong_conditions = {
            "aligned_within_5pp_of_sync": aligned_rate >= sync_rate - 0.05,
            "median_wall_reduction_at_least_20pct": (
                profile_b["aligned_median_wall_reduction_vs_sync"] >= 0.20
            ),
        }
        gate = {
            "evaluated": True,
            "numerical_conditions_only": True,
            "semantic_and_replay_acceptance_must_be_combined_separately": True,
            "conditions": conditions,
            "passed": primary_pass,
            "strong_conditions": strong_conditions,
            "strong_passed": primary_pass and all(strong_conditions.values()),
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "manifest": str(Path(manifest_path)),
        "evidence_classes": evidence_classes,
        "formal_ros_or_isaac_claim_allowed": all(
            value in {"ros_cpp_test_plant", "ros_cpp_isaac_sim"} for value in evidence_classes
        ),
        "required_metric_coverage": {
            "episode_summary_metrics": list(REQUIRED_EPISODE_METRICS),
            "all_recorded_episode_metrics": sorted(recorded_metric_fields),
            "executed_action_source_generation_field": ("command_executed.source_generation_id"),
            "numeric_descriptive_statistics": [
                "count",
                "sum",
                "mean",
                "median",
                "min",
                "max",
            ],
            "boolean_descriptive_statistics": [
                "count",
                "true_count",
                "false_count",
                "rate",
            ],
            "categorical_descriptive_statistics": ["count", "counts"],
            "all_required_metrics_present": True,
            "all_recorded_metrics_descriptively_summarized": True,
            "all_executed_action_sources_accounted_for": all(
                strategy["executed_action_source_generations"][
                    "all_executed_actions_accounted_for"
                ]
                for profile in profiles.values()
                for strategy in profile["strategies"].values()
            ),
        },
        "profiles": profiles,
        "primary_numerical_gate": gate,
    }
    if output_path is not None:
        write_json_atomic(output_path, result)
    return result
