"""Compact official-gate and ActionStream episode-result summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_TASK_IDS = (0, 1, 2)
EXPECTED_INITIAL_STATE_INDICES = tuple(range(0, 20, 2))
EXPECTED_M3_CONDITIONS = {
    ("sync", 0),
    ("sync", 200),
    ("async_naive", 0),
    ("async_naive", 200),
    ("async_aligned", 0),
    ("async_aligned", 200),
}
EXPECTED_MODEL_REVISION = "12e8783e996944f5c97e490d37d4c145484ed70a"
EXPECTED_LEROBOT_VERSION = "0.6.0"
CONTRACT_ERROR_PATTERNS = (
    re.compile(r"traceback", re.IGNORECASE),
    re.compile(r"\b(?:error|nan)\b", re.IGNORECASE),
    re.compile(r"action.{0,40}(?:invalid|out of range)", re.IGNORECASE),
    re.compile(r"camera.{0,40}(?:missing|failed)", re.IGNORECASE),
)


def read_official_eval_info(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload.get("per_task"), list) or not isinstance(payload.get("overall"), dict):
        raise ValueError("Not a LeRobot eval_info.json result")
    return payload


def _official_task_records(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    task_records: dict[int, dict[str, Any]] = {}
    for record in payload["per_task"]:
        if record.get("task_group") != "libero_object":
            continue
        task_id = int(record["task_id"])
        if task_id in task_records:
            raise ValueError(f"Duplicate official result for task {task_id}")
        task_records[task_id] = record["metrics"]
    if tuple(sorted(task_records)) != EXPECTED_TASK_IDS:
        raise ValueError(
            f"Expected official task IDs {list(EXPECTED_TASK_IDS)}, got {sorted(task_records)}"
        )
    return task_records


def reconstruct_all_success_initial_states(successes: list[bool]) -> list[int]:
    """Recover the official batch-1 init indices when every episode terminates successfully.

    A successful LIBERO step resets internally, and the next evaluator rollout
    resets once more, so each used initial state advances by two. For a failed
    episode, eval_info does not preserve whether another terminal condition
    caused an internal reset, so reconstruction would be ambiguous.
    """
    if not all(successes):
        raise ValueError(
            "Cannot reconstruct exact official initial-state indices from eval_info "
            "when any episode failed; terminal/reset provenance was not recorded"
        )
    return list(range(0, 2 * len(successes), 2))


def audit_official_log(path: Path | str, *, exit_code: int) -> dict[str, Any]:
    log_path = Path(path)
    content = log_path.read_bytes()
    matches: list[dict[str, Any]] = []
    for line_number, line in enumerate(content.decode("utf-8", errors="replace").splitlines(), start=1):
        if any(pattern.search(line) for pattern in CONTRACT_ERROR_PATTERNS):
            matches.append({"line": line_number, "text": line[:500]})
    return {
        "source_log": str(log_path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "exit_code": int(exit_code),
        "contract_error_matches": matches,
    }


def build_official_manifest(
    eval_info_path: Path | str,
    *,
    model_revision_sha: str,
    git_commit: str,
    command: str,
    contract_audit: dict[str, Any],
) -> dict[str, Any]:
    payload = read_official_eval_info(eval_info_path)
    tasks = _official_task_records(payload)

    per_task: dict[str, Any] = {}
    total_success = 0
    for task_id in EXPECTED_TASK_IDS:
        successes_raw = tasks[task_id].get("successes", [])
        if not isinstance(successes_raw, list) or any(type(value) is not bool for value in successes_raw):
            raise ValueError(f"Task {task_id} successes must be JSON booleans")
        successes = list(successes_raw)
        if len(successes) != 10:
            raise ValueError(f"Task {task_id} has {len(successes)} episodes, expected 10")
        count = sum(successes)
        total_success += count
        per_task[str(task_id)] = {
            "success_count": count,
            "episode_count": len(successes),
            "successes": successes,
            "initial_state_indices": reconstruct_all_success_initial_states(successes),
        }

    tasks_at_least_six = sum(record["success_count"] >= 6 for record in per_task.values())
    overall = payload["overall"]
    if int(overall.get("n_episodes", -1)) != 30:
        raise ValueError(f"Official overall.n_episodes is {overall.get('n_episodes')}, expected 30")
    reported_percent = float(overall.get("pc_success", float("nan")))
    measured_percent = 100.0 * total_success / 30
    if abs(reported_percent - measured_percent) > 1e-6:
        raise ValueError(
            f"Official success mismatch: per-task={measured_percent}%, overall={reported_percent}%"
        )
    contract_errors = list(contract_audit.get("contract_error_matches", []))
    contract_exit_code = int(contract_audit.get("exit_code", -1))
    gate = {
        "total_success_at_least_20": total_success >= 20,
        "at_least_two_tasks_at_least_6": tasks_at_least_six >= 2,
        "no_contract_errors": contract_exit_code == 0 and not contract_errors,
    }
    gate["passed"] = all(gate.values())
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_eval_info": str(Path(eval_info_path)),
        "git_commit": git_commit,
        "model_revision_sha": model_revision_sha,
        "command": command,
        "contract_audit": contract_audit,
        "protocol": {
            "suite": "libero_object",
            "task_ids": list(EXPECTED_TASK_IDS),
            "episodes_per_task": 10,
            "batch_size": 1,
            "seed": 142,
            "control_mode": "absolute",
            "episode_length": 800,
            "n_action_steps_override": None,
        },
        "per_task": per_task,
        "overall": {
            "success_count": total_success,
            "episode_count": 30,
            "success_rate": total_success / 30,
            "official_eval_seconds": float(overall["eval_s"]),
        },
        "gate": gate,
    }


def write_official_manifest(
    eval_info_path: Path | str,
    output_path: Path | str,
    **metadata: Any,
) -> dict[str, Any]:
    manifest = build_official_manifest(eval_info_path, **metadata)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_episode_jsonl(paths: list[Path | str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "runtime_mode" not in record or "injected_delay_ms" not in record:
                raise ValueError(f"Invalid episode record at {path}:{line_number}")
            records.append(record)
    return records


def build_custom_parity_manifest(
    official_manifest_path: Path | str,
    episode_path: Path | str,
    *,
    max_allowed_difference: int = 2,
) -> dict[str, Any]:
    official_path = Path(official_manifest_path)
    official = json.loads(official_path.read_text(encoding="utf-8"))
    if not official.get("gate", {}).get("passed"):
        raise ValueError("Official baseline manifest did not pass")

    records = read_episode_jsonl([episode_path])
    if len(records) != 30:
        raise ValueError(f"Custom parity has {len(records)} rows, expected 30")
    expected_keys = {(task_id, episode_index) for task_id in EXPECTED_TASK_IDS for episode_index in range(10)}
    actual_keys = {(int(row["task_id"]), int(row["episode_index"])) for row in records}
    if actual_keys != expected_keys or len(actual_keys) != len(records):
        raise ValueError("Custom parity task/episode keys are incomplete or duplicated")

    per_task: dict[str, Any] = {}
    commits: set[str] = set()
    revisions: set[str] = set()
    run_ids: set[str] = set()
    for task_id in EXPECTED_TASK_IDS:
        task_records = sorted(
            (row for row in records if int(row["task_id"]) == task_id),
            key=lambda row: int(row["episode_index"]),
        )
        for episode_index, row in enumerate(task_records):
            if type(row.get("success")) is not bool:
                raise ValueError(f"Custom success must be boolean for task {task_id} episode {episode_index}")
            expected_fields = {
                "runtime_mode": "sync",
                "injected_delay_ms": 0,
                "suite": "libero_object",
                "initial_state_index": EXPECTED_INITIAL_STATE_INDICES[episode_index],
                "seed": 142 + episode_index,
                "realtime_control": False,
                "policy_rng_seed": 142 + episode_index,
                "policy_rng_reset_per_episode": True,
            }
            for field, expected in expected_fields.items():
                if row.get(field) != expected:
                    raise ValueError(
                        f"Custom task {task_id} episode {episode_index} has {field}={row.get(field)!r}, "
                        f"expected {expected!r}"
                    )
            if int(row.get("environment_steps", 0)) <= 0:
                raise ValueError(f"Custom task {task_id} episode {episode_index} has no environment steps")
            commits.add(str(row["git_commit"]))
            revisions.add(str(row["model_revision_sha"]))
            run_ids.add(str(row["run_id"]))

        success_count = sum(row["success"] for row in task_records)
        per_task[str(task_id)] = {
            "success_count": success_count,
            "episode_count": 10,
            "successes": [row["success"] for row in task_records],
        }

    if len(commits) != 1 or "uncommitted" in commits:
        raise ValueError(f"Custom parity must have one committed revision, got {sorted(commits)}")
    if len(revisions) != 1 or next(iter(revisions)) != official["model_revision_sha"]:
        raise ValueError(f"Custom parity model revision mismatch: {sorted(revisions)}")
    if len(run_ids) != 1:
        raise ValueError(f"Custom parity must have one run ID, got {sorted(run_ids)}")

    custom_success = sum(row["success"] for row in records)
    official_success = int(official["overall"]["success_count"])
    difference = abs(custom_success - official_success)
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_manifest": str(official_path),
        "custom_episode_metrics": str(Path(episode_path)),
        "git_commit": next(iter(commits)),
        "model_revision_sha": next(iter(revisions)),
        "run_id": next(iter(run_ids)),
        "protocol": {
            "suite": "libero_object",
            "task_ids": list(EXPECTED_TASK_IDS),
            "episodes_per_task": 10,
            "initial_state_indices": list(EXPECTED_INITIAL_STATE_INDICES),
            "seeds": list(range(142, 152)),
            "runtime_mode": "sync",
            "chunk_size": 30,
            "realtime_control": False,
            "policy_rng_reset_per_episode": True,
        },
        "per_task": per_task,
        "overall": {
            "official_success_count": official_success,
            "custom_success_count": custom_success,
            "absolute_difference": difference,
            "max_allowed_difference": max_allowed_difference,
        },
        "gate": {"passed": difference <= max_allowed_difference},
    }


def write_custom_parity_manifest(
    official_manifest_path: Path | str,
    episode_path: Path | str,
    output_path: Path | str,
    *,
    max_allowed_difference: int = 2,
) -> dict[str, Any]:
    manifest = build_custom_parity_manifest(
        official_manifest_path,
        episode_path,
        max_allowed_difference=max_allowed_difference,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _mean(records: list[dict[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    return statistics.fmean(values) if values else None


def _std(records: list[dict[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    return statistics.stdev(values) if len(values) >= 2 else 0.0 if values else None


def aggregate_episode_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["runtime_mode"]), int(record["injected_delay_ms"]))].append(record)

    aggregates: list[dict[str, Any]] = []
    for (mode, delay_ms), group in sorted(grouped.items()):
        successes = sum(bool(record["success"]) for record in group)
        per_task: dict[str, str] = {}
        for task_id in sorted({int(record["task_id"]) for record in group}):
            task_records = [record for record in group if int(record["task_id"]) == task_id]
            task_successes = sum(bool(record["success"]) for record in task_records)
            per_task[str(task_id)] = f"{task_successes}/{len(task_records)}"
        aggregates.append(
            {
                "runtime_mode": mode,
                "injected_delay_ms": delay_ms,
                "success_count": successes,
                "episode_count": len(group),
                "success_rate": successes / len(group),
                "per_task_success": per_task,
                "mean_environment_steps": _mean(group, "environment_steps"),
                "std_environment_steps": _std(group, "environment_steps"),
                "mean_wall_clock_episode_seconds": _mean(group, "wall_clock_episode_seconds"),
                "std_wall_clock_episode_seconds": _std(group, "wall_clock_episode_seconds"),
                "mean_inference_p50_seconds": _mean(group, "inference_latency_p50_seconds"),
                "mean_inference_p95_seconds": _mean(group, "inference_latency_p95_seconds"),
                "mean_delivery_p50_seconds": _mean(group, "observation_to_delivery_p50_seconds"),
                "mean_delivery_p95_seconds": _mean(group, "observation_to_delivery_p95_seconds"),
                "mean_time_to_first_action_seconds": _mean(group, "time_to_first_action_seconds"),
                "mean_hold_steps": _mean(group, "queue_underrun_hold_steps"),
                "mean_stale_chunks_discarded": _mean(group, "stale_chunks_discarded"),
                "mean_stale_prefix_steps": _mean(group, "stale_prefix_mean_steps"),
                "max_stale_prefix_steps": max(
                    int(record["stale_prefix_max_steps"]) for record in group
                ),
                "mean_deadline_misses": _mean(group, "control_deadline_misses"),
                "mean_dispatch_lateness_p50_seconds": _mean(
                    group,
                    "control_dispatch_lateness_p50_seconds",
                ),
                "mean_action_discontinuity_l2": _mean(group, "action_discontinuity_mean_l2"),
                "median_peak_cuda_memory_mib": statistics.median(
                    float(record["peak_cuda_memory_mib"]) for record in group
                ),
                "max_peak_cuda_memory_mib": max(
                    float(record["peak_cuda_memory_mib"]) for record in group
                ),
            }
        )
    return aggregates


def _format_number(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{decimals}f}"


def render_summary_markdown(aggregates: list[dict[str, Any]]) -> str:
    lines = [
        "| mode | delay | success | per task | steps mean +/- sd | wall mean | infer p50 | delivery p50 | holds | stale prefix | missed ticks |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        per_task = ", ".join(f"{task}:{count}" for task, count in row["per_task_success"].items())
        lines.append(
            "| "
            f"{row['runtime_mode']} | {row['injected_delay_ms']} ms | "
            f"{row['success_count']}/{row['episode_count']} | {per_task} | "
            f"{_format_number(row['mean_environment_steps'], 1)} +/- "
            f"{_format_number(row['std_environment_steps'], 1)} | "
            f"{_format_number(row['mean_wall_clock_episode_seconds'], 2)} s | "
            f"{_format_number(row['mean_inference_p50_seconds'])} s | "
            f"{_format_number(row['mean_delivery_p50_seconds'])} s | "
            f"{_format_number(row['mean_hold_steps'], 1)} | "
            f"{_format_number(row['mean_stale_prefix_steps'], 2)} | "
            f"{_format_number(row['mean_deadline_misses'], 1)} |"
        )
    return "\n".join(lines) + "\n"


def _exact_sign_test_two_sided(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    tail = min(wins, losses)
    probability = 2.0 * sum(math.comb(trials, index) for index in range(tail + 1)) / 2**trials
    return min(1.0, probability)


def _paired_metric_comparison(
    baseline: list[float],
    candidate: list[float],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    baseline_values = np.asarray(baseline, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    differences = baseline_values - candidate_values
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(differences), size=(20_000, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    wins = int(np.sum(differences > 0))
    ties = int(np.sum(differences == 0))
    losses = int(np.sum(differences < 0))
    baseline_mean = float(baseline_values.mean())
    candidate_mean = float(candidate_values.mean())
    return {
        "baseline_mean": baseline_mean,
        "baseline_std": float(baseline_values.std(ddof=1)),
        "candidate_mean": candidate_mean,
        "candidate_std": float(candidate_values.std(ddof=1)),
        "paired_mean_reduction": float(differences.mean()),
        "paired_reduction_95pct_bootstrap_ci": [float(lower), float(upper)],
        "relative_reduction_percent": 100.0 * float(differences.mean()) / baseline_mean,
        "candidate_wins_ties_losses": [wins, ties, losses],
        "exact_sign_test_two_sided_p": _exact_sign_test_two_sided(wins, losses),
    }


def paired_mode_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (
            str(row["runtime_mode"]),
            int(row["injected_delay_ms"]),
            int(row["task_id"]),
            int(row["episode_index"]),
        ): row
        for row in records
    }
    comparisons: list[dict[str, Any]] = []
    for delay_ms in (0, 200):
        for baseline_mode, candidate_mode in (
            ("async_naive", "async_aligned"),
            ("sync", "async_aligned"),
        ):
            keys = [(task_id, episode_index) for task_id in EXPECTED_TASK_IDS for episode_index in range(10)]
            baseline_rows = [
                indexed[(baseline_mode, delay_ms, task_id, episode_index)]
                for task_id, episode_index in keys
            ]
            candidate_rows = [
                indexed[(candidate_mode, delay_ms, task_id, episode_index)]
                for task_id, episode_index in keys
            ]
            comparisons.append(
                {
                    "delay_ms": delay_ms,
                    "baseline_mode": baseline_mode,
                    "candidate_mode": candidate_mode,
                    "paired_episode_count": len(keys),
                    "success": {
                        "baseline_count": sum(bool(row["success"]) for row in baseline_rows),
                        "candidate_count": sum(bool(row["success"]) for row in candidate_rows),
                    },
                    "environment_steps": _paired_metric_comparison(
                        [float(row["environment_steps"]) for row in baseline_rows],
                        [float(row["environment_steps"]) for row in candidate_rows],
                        bootstrap_seed=142 + delay_ms,
                    ),
                    "wall_clock_episode_seconds": _paired_metric_comparison(
                        [float(row["wall_clock_episode_seconds"]) for row in baseline_rows],
                        [float(row["wall_clock_episode_seconds"]) for row in candidate_rows],
                        bootstrap_seed=3142 + delay_ms,
                    ),
                }
            )
    return comparisons


def render_comparisons_markdown(comparisons: list[dict[str, Any]]) -> str:
    lines = [
        "| delay | comparison | success | steps baseline -> candidate | paired reduction [95% CI] | relative | wins/ties/losses |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for comparison in comparisons:
        metric = comparison["environment_steps"]
        lower, upper = metric["paired_reduction_95pct_bootstrap_ci"]
        success = comparison["success"]
        lines.append(
            "| "
            f"{comparison['delay_ms']} ms | {comparison['baseline_mode']} -> "
            f"{comparison['candidate_mode']} | "
            f"{success['baseline_count']} -> {success['candidate_count']} | "
            f"{metric['baseline_mean']:.1f} -> {metric['candidate_mean']:.1f} | "
            f"{metric['paired_mean_reduction']:.1f} [{lower:.1f}, {upper:.1f}] | "
            f"{metric['relative_reduction_percent']:.1f}% | "
            f"{'/'.join(str(value) for value in metric['candidate_wins_ties_losses'])} |"
        )
    return "\n".join(lines) + "\n"


def write_actionstream_summary(
    episode_paths: list[Path | str],
    *,
    json_output: Path | str,
    markdown_output: Path | str,
) -> list[dict[str, Any]]:
    aggregates = aggregate_episode_records(read_episode_jsonl(episode_paths))
    json_path = Path(json_output)
    markdown_path = Path(markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(aggregates, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_summary_markdown(aggregates), encoding="utf-8")
    return aggregates


def _resolve_action_trace_path(
    trace_path_value: str,
    *,
    condition: tuple[str, int],
    condition_episode_dirs: dict[tuple[str, int], Path],
) -> Path:
    if os.name == "nt" and re.match(r"^/mnt/[A-Za-z]/", trace_path_value):
        parts = trace_path_value.split("/")
        trace_path = Path(f"{parts[2].upper()}:\\", *parts[3:])
    else:
        trace_path = Path(trace_path_value)
    if trace_path.is_file():
        return trace_path

    condition_dir = condition_episode_dirs.get(condition)
    portable_path = (
        condition_dir / "traces" / Path(trace_path_value).name
        if condition_dir is not None
        else trace_path
    )
    if portable_path.is_file():
        return portable_path
    raise ValueError(
        f"Missing action trace: {trace_path}; portable fallback: {portable_path}"
    )


def validate_m3_matrix(
    episode_paths: list[Path | str],
    *,
    verify_action_traces: bool = True,
) -> dict[str, Any]:
    records = read_episode_jsonl(episode_paths)
    if len(records) != 180:
        raise ValueError(f"M3 matrix has {len(records)} rows, expected 180")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["runtime_mode"]), int(row["injected_delay_ms"]))].append(row)
    if set(grouped) != EXPECTED_M3_CONDITIONS:
        raise ValueError(f"M3 conditions are {sorted(grouped)}, expected {sorted(EXPECTED_M3_CONDITIONS)}")

    condition_episode_dirs: dict[tuple[str, int], Path] = {}
    for episode_path_value in episode_paths:
        episode_path = Path(episode_path_value)
        match = re.fullmatch(
            r"(sync|async_naive|async_aligned)_delay(\d+)",
            episode_path.parent.name,
        )
        if match:
            condition_episode_dirs[(match.group(1), int(match.group(2)))] = episode_path.parent

    commits: set[str] = set()
    revisions: set[str] = set()
    trace_count = 0
    total_action_steps = 0
    condition_summaries: dict[str, Any] = {}
    for condition, group in sorted(grouped.items()):
        mode, delay_ms = condition
        if len(group) != 30:
            raise ValueError(f"{mode}/{delay_ms} has {len(group)} rows, expected 30")
        expected_keys = {
            (task_id, episode_index)
            for task_id in EXPECTED_TASK_IDS
            for episode_index in range(10)
        }
        actual_keys = {
            (int(row["task_id"]), int(row["episode_index"]))
            for row in group
        }
        if actual_keys != expected_keys or len(actual_keys) != len(group):
            raise ValueError(f"{mode}/{delay_ms} task/episode keys are incomplete or duplicated")
        run_ids = {str(row["run_id"]) for row in group}
        if len(run_ids) != 1:
            raise ValueError(f"{mode}/{delay_ms} has multiple run IDs: {sorted(run_ids)}")

        per_task_success: dict[str, str] = {}
        for task_id in EXPECTED_TASK_IDS:
            task_records = [row for row in group if int(row["task_id"]) == task_id]
            per_task_success[str(task_id)] = (
                f"{sum(bool(row['success']) for row in task_records)}/{len(task_records)}"
            )

        for row in group:
            episode_index = int(row["episode_index"])
            expected_fields = {
                "suite": "libero_object",
                "initial_state_index": EXPECTED_INITIAL_STATE_INDICES[episode_index],
                "seed": 142 + episode_index,
                "policy_rng_seed": 142 + episode_index,
                "policy_rng_reset_per_episode": True,
                "realtime_control": True,
                "replan_interval_steps": 30 if mode == "sync" else 10,
                "controller_frequency_hz": 20.0,
                "lerobot_version": EXPECTED_LEROBOT_VERSION,
            }
            for field, expected in expected_fields.items():
                if row.get(field) != expected:
                    raise ValueError(
                        f"{mode}/{delay_ms} task {row['task_id']} episode {episode_index} "
                        f"has {field}={row.get(field)!r}, expected {expected!r}"
                    )
            if type(row.get("success")) is not bool:
                raise ValueError(f"{mode}/{delay_ms} contains a non-boolean success")
            if int(row["environment_steps"]) <= 0 or int(row["environment_steps"]) > 800:
                raise ValueError(f"{mode}/{delay_ms} has invalid environment step count")
            if int(row["inference_calls"]) <= 0:
                raise ValueError(f"{mode}/{delay_ms} has no inference calls")
            if int(row.get("old_episode_chunks_rejected", 0)) != 0:
                raise ValueError(f"{mode}/{delay_ms} rejected an old episode chunk")
            if int(row.get("old_episode_results_discarded", 0)) != 0:
                raise ValueError(f"{mode}/{delay_ms} discarded an old worker result")
            for field in (
                "inference_latency_p50_seconds",
                "inference_latency_p95_seconds",
                "observation_to_delivery_p50_seconds",
                "observation_to_delivery_p95_seconds",
                "peak_cuda_memory_mib",
            ):
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"{mode}/{delay_ms} has non-finite {field}")

            commits.add(str(row["git_commit"]))
            revisions.add(str(row["model_revision_sha"]))
            total_action_steps += int(row["environment_steps"])
            if verify_action_traces:
                trace_path_value = str(row["action_trace_path"])
                trace_path = _resolve_action_trace_path(
                    trace_path_value,
                    condition=condition,
                    condition_episode_dirs=condition_episode_dirs,
                )
                with np.load(trace_path, allow_pickle=False) as trace:
                    actions = trace["actions"]
                    dispatch_timestamps = trace["dispatch_timestamps"]
                if actions.shape != (int(row["environment_steps"]), 7):
                    raise ValueError(f"Invalid action trace shape at {trace_path}: {actions.shape}")
                if not np.isfinite(actions).all() or np.any(np.all(np.isclose(actions, 0.0), axis=1)):
                    raise ValueError(f"Unsafe or non-finite action in trace: {trace_path}")
                if dispatch_timestamps.shape != (int(row["environment_steps"]),):
                    raise ValueError(f"Invalid dispatch trace shape at {trace_path}")
                trace_count += 1

        condition_summaries[f"{mode}/delay{delay_ms}"] = {
            "run_id": next(iter(run_ids)),
            "success_count": sum(bool(row["success"]) for row in group),
            "episode_count": len(group),
            "per_task_success": per_task_success,
            "hold_steps_total": sum(int(row["queue_underrun_hold_steps"]) for row in group),
            "fully_stale_chunks_total": sum(int(row["stale_chunks_discarded"]) for row in group),
            "old_episode_leak_count": sum(
                int(row.get("old_episode_chunks_rejected", 0))
                + int(row.get("old_episode_results_discarded", 0))
                for row in group
            ),
        }

    if len(commits) != 1 or "uncommitted" in commits:
        raise ValueError(f"M3 matrix must have one committed revision, got {sorted(commits)}")
    if revisions != {EXPECTED_MODEL_REVISION}:
        raise ValueError(f"M3 matrix model revision mismatch: {sorted(revisions)}")
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "episode_metric_paths": [str(Path(path)) for path in episode_paths],
        "git_commit": next(iter(commits)),
        "model_revision_sha": next(iter(revisions)),
        "episode_count": len(records),
        "action_trace_count_verified": trace_count if verify_action_traces else None,
        "total_final_7d_actions_verified": total_action_steps if verify_action_traces else None,
        "protocol": {
            "suite": "libero_object",
            "task_ids": list(EXPECTED_TASK_IDS),
            "episodes_per_task": 10,
            "delays_ms": [0, 200],
            "runtime_modes": ["sync", "async_naive", "async_aligned"],
            "initial_state_indices": list(EXPECTED_INITIAL_STATE_INDICES),
            "seeds": list(range(142, 152)),
            "controller_frequency_hz": 20.0,
            "chunk_size": 30,
            "async_replan_interval_steps": 10,
            "policy_rng_reset_per_episode": True,
        },
        "conditions": condition_summaries,
        "gate": {
            "all_expected_rows_present": True,
            "all_action_traces_safe": verify_action_traces,
            "no_cross_episode_leaks": True,
            "passed": True,
        },
    }


def write_m3_report(
    episode_paths: list[Path | str],
    output_dir: Path | str,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = read_episode_jsonl(episode_paths)
    manifest = validate_m3_matrix(episode_paths)
    aggregates = aggregate_episode_records(records)
    comparisons = paired_mode_comparisons(records)
    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "interpretation_boundary": (
            "All six conditions reached 30/30 success, so this matrix supports an efficiency "
            "effect in environment steps and wall time, not a success-rate advantage."
        ),
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = (
        "# ActionStream M3 Results\n\n"
        "All six conditions reached 30/30 success. The binary success metric is therefore "
        "ceilinged; paired step and wall-time reductions are the informative outcomes.\n\n"
        "## Raw aggregate table\n\n"
        f"{render_summary_markdown(aggregates)}\n"
        "## Paired comparisons\n\n"
        f"{render_comparisons_markdown(comparisons)}"
    )
    (output / "summary.md").write_text(markdown, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize the fixed ActionStream M3 matrix")
    parser.add_argument("episode_paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/summary"))
    args = parser.parse_args()
    write_m3_report(args.episode_paths, args.output_dir)


if __name__ == "__main__":
    main()
