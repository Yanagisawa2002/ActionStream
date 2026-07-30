"""Compact official-gate and ActionStream episode-result summaries."""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_TASK_IDS = (0, 1, 2)


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


def build_official_manifest(
    eval_info_path: Path | str,
    *,
    model_revision_sha: str,
    git_commit: str,
    command: str,
) -> dict[str, Any]:
    payload = read_official_eval_info(eval_info_path)
    tasks = _official_task_records(payload)

    per_task: dict[str, Any] = {}
    total_success = 0
    for task_id in EXPECTED_TASK_IDS:
        successes = [bool(value) for value in tasks[task_id].get("successes", [])]
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
    gate = {
        "total_success_at_least_20": total_success >= 20,
        "at_least_two_tasks_at_least_6": tasks_at_least_six >= 2,
        "no_contract_errors": True,
    }
    gate["passed"] = all(gate.values())
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_eval_info": str(Path(eval_info_path)),
        "git_commit": git_commit,
        "model_revision_sha": model_revision_sha,
        "command": command,
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
            "official_eval_seconds": float(payload["overall"]["eval_s"]),
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
                "mean_delivery_p50_seconds": _mean(group, "observation_to_delivery_p50_seconds"),
                "mean_hold_steps": _mean(group, "queue_underrun_hold_steps"),
                "mean_stale_chunks_discarded": _mean(group, "stale_chunks_discarded"),
                "mean_stale_prefix_steps": _mean(group, "stale_prefix_mean_steps"),
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
