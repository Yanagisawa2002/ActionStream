"""Derive and select an ActionStream M4 queue-pressure calibration point."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from actionstream.results import read_episode_jsonl


TARGET_HEADROOM_RATIOS = (0.5, 0.75, 1.0, 1.2)
EXTENSION_HEADROOM_RATIO = 1.5
DELAY_ROUNDING_MS = 50


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "uncommitted"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty distribution")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Distribution contains non-finite values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def _source_dirs(episode_paths: list[Path]) -> dict[tuple[str, int], Path]:
    sources: dict[tuple[str, int], Path] = {}
    for path in episode_paths:
        match = re.fullmatch(
            r"(sync|async_naive|async_aligned|sync_hold)_delay(\d+)",
            path.parent.name,
        )
        if match:
            sources[(match.group(1), int(match.group(2)))] = path.parent
    return sources


def _trace_path(
    row: dict[str, Any],
    sources: dict[tuple[str, int], Path],
) -> Path:
    recorded = Path(str(row["action_trace_path"]))
    if recorded.is_file():
        return recorded
    condition = (str(row["runtime_mode"]), int(row["injected_delay_ms"]))
    source_dir = sources.get(condition)
    portable = (
        source_dir / "traces" / recorded.name
        if source_dir is not None
        else recorded
    )
    if portable.is_file():
        return portable
    raise ValueError(f"Missing trace {recorded}; portable fallback {portable}")


def _trace_payload(path: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as trace:
        dispatch = np.asarray(trace["dispatch_timestamps"], dtype=np.float64)
        events = json.loads(str(trace["inference_events_json"]))
    if dispatch.ndim != 1 or not np.isfinite(dispatch).all():
        raise ValueError(f"Invalid dispatch timestamps in {path}")
    if not isinstance(events, list):
        raise ValueError(f"Invalid inference event list in {path}")
    return dispatch, events


def _rounded_delay_ms(value_seconds: float) -> int:
    raw_ms = max(0.0, value_seconds * 1000.0)
    return int(round(raw_ms / DELAY_ROUNDING_MS) * DELAY_ROUNDING_MS)


def derive_candidate_delays(
    *,
    control_step_seconds: float,
    base_delivery_seconds: float,
    queue_headroom_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if control_step_seconds <= 0 or base_delivery_seconds < 0:
        raise ValueError("Calibration timings must be non-negative and have a positive control step")
    if queue_headroom_steps <= 0:
        raise ValueError("Queue headroom must be positive")

    candidates: list[dict[str, Any]] = []
    used_delays: set[int] = set()
    for ratio in TARGET_HEADROOM_RATIOS:
        target_age_steps = queue_headroom_steps * ratio
        target_total_seconds = target_age_steps * control_step_seconds
        delay_ms = _rounded_delay_ms(target_total_seconds - base_delivery_seconds)
        if delay_ms in used_delays:
            continue
        used_delays.add(delay_ms)
        candidates.append(
            {
                "target_headroom_ratio": ratio,
                "target_delivery_age_steps": target_age_steps,
                "target_total_delivery_seconds": target_total_seconds,
                "injected_delay_ms": delay_ms,
                "predicted_delivery_age_steps": (
                    base_delivery_seconds + delay_ms / 1000.0
                )
                / control_step_seconds,
            }
        )
    if len(candidates) > 4:
        raise AssertionError("M4 permits at most four initial calibration candidates")

    extension_target_steps = queue_headroom_steps * EXTENSION_HEADROOM_RATIO
    extension_total_seconds = extension_target_steps * control_step_seconds
    extension_delay_ms = _rounded_delay_ms(
        extension_total_seconds - base_delivery_seconds
    )
    extension = {
        "target_headroom_ratio": EXTENSION_HEADROOM_RATIO,
        "target_delivery_age_steps": extension_target_steps,
        "target_total_delivery_seconds": extension_total_seconds,
        "injected_delay_ms": extension_delay_ms,
        "predicted_delivery_age_steps": (
            base_delivery_seconds + extension_delay_ms / 1000.0
        )
        / control_step_seconds,
    }
    return candidates, extension


def build_calibration_plan(
    episode_paths: list[Path],
    *,
    m3_manifest_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(m3_manifest_path.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    chunk_size = int(protocol["chunk_size"])
    replan_interval = int(protocol["async_replan_interval_steps"])
    queue_headroom = chunk_size - replan_interval
    if queue_headroom <= 0:
        raise ValueError(
            f"Invalid derived queue headroom: {chunk_size} - {replan_interval}"
        )

    records = [
        row
        for row in read_episode_jsonl(episode_paths)
        if str(row["runtime_mode"]) in {"async_naive", "async_aligned"}
    ]
    if len(records) != 120:
        raise ValueError(f"Expected 120 final async M3 episodes, got {len(records)}")
    sources = _source_dirs(episode_paths)

    control_steps: list[float] = []
    inference_all: list[float] = []
    delivery_all: list[float] = []
    delivery_by_delay: dict[int, list[float]] = defaultdict(list)
    trace_hashes: dict[str, str] = {}
    for row in records:
        trace_path = _trace_path(row, sources)
        dispatch, events = _trace_payload(trace_path)
        intervals = np.diff(dispatch)
        control_steps.extend(float(value) for value in intervals if value > 0)
        trace_hashes[str(trace_path)] = _sha256(trace_path)
        for event in events:
            if not bool(event.get("published_before_episode_end", True)):
                continue
            inference = float(event["model_inference_latency_seconds"])
            delivery = float(event["delivery_timestamp"]) - float(
                event["request_timestamp"]
            )
            if not (math.isfinite(inference) and math.isfinite(delivery)):
                raise ValueError(f"Non-finite timing in {trace_path}")
            inference_all.append(inference)
            delivery_all.append(delivery)
            delivery_by_delay[int(row["injected_delay_ms"])].append(delivery)

    control_summary = _distribution(control_steps)
    inference_summary = _distribution(inference_all)
    delivery_summary = _distribution(delivery_all)
    delivery_summaries = {
        str(delay_ms): _distribution(values)
        for delay_ms, values in sorted(delivery_by_delay.items())
    }
    if "0" not in delivery_summaries:
        raise ValueError("M3 async traces do not contain a 0 ms delivery baseline")

    candidates, extension = derive_candidate_delays(
        control_step_seconds=float(control_summary["median"]),
        base_delivery_seconds=float(delivery_summaries["0"]["median"]),
        queue_headroom_steps=queue_headroom,
    )
    delay_300_predicted_age = (
        float(delivery_summaries["0"]["median"]) + 0.3
    ) / float(control_summary["median"])
    nearest_target = min(
        abs(delay_300_predicted_age - queue_headroom * ratio)
        for ratio in TARGET_HEADROOM_RATIOS
    )
    include_300 = nearest_target <= 0.5

    return {
        "status": "planned",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "source": {
            "m3_manifest": str(m3_manifest_path),
            "m3_manifest_sha256": _sha256(m3_manifest_path),
            "episode_paths": [str(path) for path in episode_paths],
            "trace_count": len(trace_hashes),
        },
        "runtime_configuration": {
            "chunk_size": chunk_size,
            "replan_interval_steps": replan_interval,
            "queue_headroom_steps": queue_headroom,
            "queue_headroom_seconds_at_median_step": (
                queue_headroom * float(control_summary["median"])
            ),
            "queue_headroom_seconds_at_p95_step": (
                queue_headroom * float(control_summary["p95"])
            ),
        },
        "measured_m3_timings": {
            "control_step_seconds": control_summary,
            "model_inference_seconds": inference_summary,
            "observation_to_delivery_seconds": delivery_summary,
            "observation_to_delivery_by_delay_ms": delivery_summaries,
        },
        "candidate_derivation": {
            "formula": (
                "injected_delay = target_age_steps * median_control_step "
                "- median_0ms_observation_to_delivery"
            ),
            "delay_rounding_ms": DELAY_ROUNDING_MS,
            "target_headroom_ratios": list(TARGET_HEADROOM_RATIOS),
            "initial_candidates": candidates,
            "include_300_ms": include_300,
            "delay_300_predicted_delivery_age_steps": delay_300_predicted_age,
            "delay_300_nearest_target_distance_steps": nearest_target,
            "delay_300_rationale": (
                "included because it is within 0.5 control step of a target"
                if include_300
                else "excluded because a derived candidate is closer to every target"
            ),
            "extend_once_if_needed": extension,
        },
        "calibration_protocol": {
            "suite": "libero_object",
            "task_ids": [3],
            "episodes_per_condition": 3,
            "initial_state_indices": [0, 2, 4],
            "seeds": [142, 143, 144],
            "modes": ["async_naive", "async_aligned"],
            "selection_rule": (
                "Choose the smallest tested delay where the pooled median effective "
                "delivery age reaches queue headroom, or any queue hold or fully stale "
                "chunk occurs. If none qualifies, test the single 1.5x extension."
            ),
        },
    }


def _calibration_condition_metrics(
    episode_paths: list[Path],
) -> list[dict[str, Any]]:
    records = read_episode_jsonl(episode_paths)
    sources = _source_dirs(episode_paths)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["runtime_mode"]), int(row["injected_delay_ms"]))].append(
            row
        )

    summaries: list[dict[str, Any]] = []
    for (mode, delay_ms), group in sorted(grouped.items()):
        if mode not in {"async_naive", "async_aligned"}:
            raise ValueError(f"Unexpected calibration mode {mode}")
        if len(group) != 3 or {int(row["task_id"]) for row in group} != {3}:
            raise ValueError(
                f"{mode}/{delay_ms} must contain three task-3 episodes"
            )
        delivery_ages: list[float] = []
        for row in group:
            trace_path = _trace_path(row, sources)
            dispatch, events = _trace_payload(trace_path)
            intervals = np.diff(dispatch)
            positive = intervals[intervals > 0]
            if not positive.size:
                raise ValueError(f"No positive control intervals in {trace_path}")
            step_seconds = float(np.median(positive))
            for event in events:
                if not bool(event.get("published_before_episode_end", True)):
                    continue
                latency = float(event["delivery_timestamp"]) - float(
                    event["request_timestamp"]
                )
                delivery_ages.append(
                    float(
                        event.get(
                            "effective_delivery_age_steps",
                            latency / step_seconds,
                        )
                    )
                )
        age_summary = _distribution(delivery_ages)
        summaries.append(
            {
                "runtime_mode": mode,
                "injected_delay_ms": delay_ms,
                "episode_count": len(group),
                "success_count": sum(bool(row["success"]) for row in group),
                "effective_delivery_age_steps": age_summary,
                "effective_delivery_age_step_samples": delivery_ages,
                "queue_hold_steps_total": sum(
                    int(row["queue_underrun_hold_steps"]) for row in group
                ),
                "fully_stale_chunks_total": sum(
                    int(row["stale_chunks_discarded"]) for row in group
                ),
            }
        )
    return summaries


def choose_pressure_delay(
    condition_metrics: list[dict[str, Any]],
    *,
    queue_headroom_steps: int,
) -> tuple[int | None, list[dict[str, Any]]]:
    by_delay: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in condition_metrics:
        by_delay[int(row["injected_delay_ms"])].append(row)

    decisions: list[dict[str, Any]] = []
    selected: int | None = None
    for delay_ms, rows in sorted(by_delay.items()):
        if {str(row["runtime_mode"]) for row in rows} != {
            "async_naive",
            "async_aligned",
        }:
            raise ValueError(f"Delay {delay_ms} is missing a calibration mode")
        pooled_age_samples = [
            float(value)
            for row in rows
            for value in row.get(
                "effective_delivery_age_step_samples",
                [row["effective_delivery_age_steps"]["median"]],
            )
        ]
        pooled_age_summary = _distribution(pooled_age_samples)
        pooled_median_age = float(pooled_age_summary["median"])
        hold_steps = sum(int(row["queue_hold_steps_total"]) for row in rows)
        stale_chunks = sum(int(row["fully_stale_chunks_total"]) for row in rows)
        triggers = {
            "median_age_reaches_headroom": pooled_median_age
            >= queue_headroom_steps,
            "nonzero_queue_holds": hold_steps > 0,
            "nonzero_fully_stale_chunks": stale_chunks > 0,
        }
        qualifies = any(triggers.values())
        decisions.append(
            {
                "injected_delay_ms": delay_ms,
                "pooled_effective_delivery_age_steps": pooled_age_summary,
                "queue_hold_steps_total": hold_steps,
                "fully_stale_chunks_total": stale_chunks,
                "triggers": triggers,
                "qualifies_as_pressure": qualifies,
            }
        )
        if qualifies and selected is None:
            selected = delay_ms
    return selected, decisions


def build_selection(
    episode_paths: list[Path],
    *,
    plan_path: Path,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    queue_headroom = int(
        plan["runtime_configuration"]["queue_headroom_steps"]
    )
    condition_metrics = _calibration_condition_metrics(episode_paths)
    selected, decisions = choose_pressure_delay(
        condition_metrics,
        queue_headroom_steps=queue_headroom,
    )
    tested_delays = sorted(
        {int(row["injected_delay_ms"]) for row in condition_metrics}
    )
    extension_delay = int(
        plan["candidate_derivation"]["extend_once_if_needed"][
            "injected_delay_ms"
        ]
    )
    extension_tested = extension_delay in tested_delays
    if selected is not None:
        status = "selected"
    elif extension_tested:
        status = "no_pressure_after_extension"
    else:
        status = "needs_extension"

    return {
        "status": status,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "calibration_episode_paths": [str(path) for path in episode_paths],
        "tested_delays_ms": tested_delays,
        "queue_headroom_steps": queue_headroom,
        "condition_metrics": condition_metrics,
        "delay_decisions": decisions,
        "selected_pressure_delay_ms": selected,
        "extension_delay_ms": extension_delay,
        "extension_tested": extension_tested,
        "selection_rule_frozen_before_full_pressure_run": True,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or select the ActionStream M4 pressure calibration"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("episode_paths", nargs="+", type=Path)
    plan_parser.add_argument("--m3-manifest", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("episode_paths", nargs="+", type=Path)
    select_parser.add_argument("--plan", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "plan":
        payload = build_calibration_plan(
            args.episode_paths,
            m3_manifest_path=args.m3_manifest,
        )
    else:
        payload = build_selection(args.episode_paths, plan_path=args.plan)
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
