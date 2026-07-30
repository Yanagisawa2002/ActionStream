"""Compact official-gate and ActionStream episode-result summaries."""

from __future__ import annotations

import json
import hashlib
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_TASK_IDS = (0, 1, 2)
EXPECTED_INITIAL_STATE_INDICES = tuple(range(0, 20, 2))
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
                "mean_wall_clock_episode_seconds": _mean(group, "wall_clock_episode_seconds"),
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
                "mean_action_discontinuity_l2": _mean(group, "action_discontinuity_mean_l2"),
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
        "| mode | delay | success | per task | infer p50 | delivery p50 | holds | stale drops | deadline misses |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        per_task = ", ".join(f"{task}:{count}" for task, count in row["per_task_success"].items())
        lines.append(
            "| "
            f"{row['runtime_mode']} | {row['injected_delay_ms']} ms | "
            f"{row['success_count']}/{row['episode_count']} | {per_task} | "
            f"{_format_number(row['mean_inference_p50_seconds'])} s | "
            f"{_format_number(row['mean_delivery_p50_seconds'])} s | "
            f"{_format_number(row['mean_hold_steps'], 1)} | "
            f"{_format_number(row['mean_stale_chunks_discarded'], 1)} | "
            f"{_format_number(row['mean_deadline_misses'], 1)} |"
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
