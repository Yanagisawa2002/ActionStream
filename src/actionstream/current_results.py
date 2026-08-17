"""Validate and summarize paired current-LeRobot Async/RTC experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from actionstream.m4_analysis import paired_metric_effect


PAIR_FIELDS = ("task_id", "episode_index", "initial_state_index", "seed")
PAIR_INVARIANT_FIELDS = (
    "source_commit",
    "model_id",
    "model_revision",
    "control_mode",
    "delay_trace_sha256",
    "controller_frequency_hz",
    "chunk_size",
    "request_interval_steps",
)
RAW_FIELDS = (
    "model_key",
    "model_revision",
    "runtime",
    "delay_profile",
    *PAIR_FIELDS,
    "success",
    "environment_steps",
    "hold_fraction",
    "dropped_prefix_steps",
    "delivery_latency_p50_seconds",
    "delivery_latency_p95_seconds",
    "inference_latency_p50_seconds",
    "inference_latency_p95_seconds",
    "action_discontinuity_mean_l2",
    "action_discontinuity_max_l2",
    "action_acceleration_max_l2",
    "peak_cuda_memory_mib",
    "wall_clock_episode_seconds",
    "source_commit",
    "trace_sha256",
    "video_sha256",
)
METRICS = (
    "success",
    "environment_steps",
    "hold_fraction",
    "dropped_prefix_steps",
    "delivery_latency_p50_seconds",
    "delivery_latency_p95_seconds",
    "action_discontinuity_mean_l2",
    "action_discontinuity_max_l2",
    "action_acceleration_max_l2",
)
RUNTIME_ORDER = (
    "sync_hold",
    "lerobot_weighted_average",
    "lerobot_latest_only",
    "actionstream_aligned",
    "actionstream_adaptive",
    "lerobot_rtc",
)
PROFILE_ORDER = (
    "fixed_0000",
    "fixed_0250",
    "fixed_0500",
    "fixed_0950",
    "jitter_0500_pm0250",
    "jitter_0500_pm0250_seed01",
    "jitter_0500_pm0250_seed02",
    "jitter_0500_pm0250_seed03",
    "burst_outage_seed01",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_episode_rows(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    row = json.loads(line)
                    row["_episode_path"] = str(path.resolve())
                    row["_line_number"] = line_number
                    rows.append(row)
    if not rows:
        raise ValueError("No episode rows were found")
    return rows


def _pair_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return tuple(int(row[field]) for field in PAIR_FIELDS)  # type: ignore[return-value]


def _condition_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["model_key"]), str(row["runtime"]), str(row["delay_profile"])


def _task_condition_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row["model_key"]),
        int(row["task_id"]),
        str(row["runtime"]),
        str(row["delay_profile"]),
    )


def _metric_value(row: dict[str, Any], metric: str) -> float:
    value = float(bool(row[metric])) if metric == "success" else float(row[metric])
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite {metric} at {row['_episode_path']}:{row['_line_number']}"
        )
    return value


def _resolve_artifact(row: dict[str, Any], field: str) -> Path | None:
    raw = row.get(field)
    if not raw:
        return None
    episode_root = Path(row["_episode_path"]).parent
    category = "traces" if field == "trace_path" else "videos"
    local = episode_root / category / Path(str(raw)).name
    return local if local.is_file() else Path(str(raw))


def validate_episode_rows(rows: list[dict[str, Any]], *, verify_artifacts: bool = True) -> dict[str, Any]:
    seen: set[tuple[str, str, str, tuple[int, int, int, int]]] = set()
    sources: set[str] = set()
    models: dict[str, str] = {}
    trace_count = 0
    video_count = 0
    action_count = 0
    for row in rows:
        location = f"{row['_episode_path']}:{row['_line_number']}"
        if row.get("status") != "completed":
            raise ValueError(f"Non-completed episode at {location}")
        if type(row.get("success")) is not bool:
            raise ValueError(f"Non-boolean success at {location}")
        for metric in METRICS:
            if row.get(metric) is None:
                raise ValueError(f"Missing {metric} at {location}")
            _metric_value(row, metric)
        condition = _condition_key(row)
        key = (*condition, _pair_key(row))
        if key in seen:
            raise ValueError(f"Duplicate paired condition row {key}")
        seen.add(key)
        sources.add(str(row["source_commit"]))
        previous_revision = models.setdefault(condition[0], str(row["model_revision"]))
        if previous_revision != str(row["model_revision"]):
            raise ValueError(f"Model revision drift for {condition[0]}")
        if verify_artifacts:
            for path_field, hash_field in (("trace_path", "trace_sha256"), ("video_path", "video_sha256")):
                artifact = _resolve_artifact(row, path_field)
                expected = row.get(hash_field)
                if artifact is None:
                    if path_field == "trace_path":
                        raise ValueError(f"Missing trace reference at {location}")
                    continue
                if not artifact.is_file():
                    raise ValueError(f"Missing artifact {artifact} referenced at {location}")
                if expected != _sha256(artifact):
                    raise ValueError(f"Artifact hash mismatch for {artifact}")
                if path_field == "trace_path":
                    payload = json.loads(artifact.read_text(encoding="utf-8"))
                    actions = payload.get("actions")
                    events = payload.get("inference_events")
                    if not isinstance(actions, list) or not isinstance(events, list):
                        raise ValueError(f"Malformed trace payload in {artifact}")
                    if len(actions) != int(row["environment_steps"]):
                        raise ValueError(f"Trace/action count mismatch in {artifact}")
                    for action_row in actions:
                        vector = action_row.get("action") if isinstance(action_row, dict) else None
                        if (
                            not isinstance(vector, list)
                            or len(vector) != 7
                            or not all(math.isfinite(float(value)) for value in vector)
                        ):
                            raise ValueError(f"Non-finite or non-7D action in {artifact}")
                    trace_count += 1
                    action_count += len(actions)
                else:
                    video_count += 1

    grouped_pairs: dict[tuple[str, str], dict[str, set[tuple[int, int, int, int]]]] = defaultdict(dict)
    for row in rows:
        model, runtime, profile = _condition_key(row)
        grouped_pairs[(model, profile)].setdefault(runtime, set()).add(_pair_key(row))
    for group, runtimes in grouped_pairs.items():
        expected: set[tuple[int, int, int, int]] | None = None
        for runtime, keys in runtimes.items():
            if expected is None:
                expected = keys
            elif keys != expected:
                raise ValueError(f"Unpaired runtime cells for {group}: {runtime}")

    paired_rows: dict[
        tuple[str, str, tuple[int, int, int, int]], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        model, _, profile = _condition_key(row)
        paired_rows[(model, profile, _pair_key(row))].append(row)
    for key, pair in paired_rows.items():
        for field in PAIR_INVARIANT_FIELDS:
            values = {json.dumps(row.get(field), sort_keys=True) for row in pair}
            if len(values) != 1:
                raise ValueError(f"Pair invariant mismatch for {key}: {field}")

    return {
        "episode_count": len(rows),
        "source_commits": sorted(sources),
        "model_revisions": models,
        "trace_count_verified": trace_count,
        "final_7d_action_count_verified": action_count,
        "video_count_verified": video_count,
        "pairing_fields": list(PAIR_FIELDS),
        "pair_invariant_fields": list(PAIR_INVARIANT_FIELDS),
    }


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Summary requires finite values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _sorted_conditions(keys: Iterable[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    runtime_rank = {name: index for index, name in enumerate(RUNTIME_ORDER)}
    profile_rank = {name: index for index, name in enumerate(PROFILE_ORDER)}
    return sorted(
        keys,
        key=lambda item: (
            item[0],
            profile_rank.get(item[2], len(profile_rank)),
            runtime_rank.get(item[1], len(runtime_rank)),
        ),
    )


def _paired_effect(
    reference_rows: dict[tuple[int, int, int, int], dict[str, Any]],
    estimate_rows: dict[tuple[int, int, int, int], dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    keys = sorted(set(reference_rows) & set(estimate_rows))
    if not keys or set(reference_rows) != set(estimate_rows):
        raise ValueError("Paired comparison requires identical nonempty episode keys")
    return paired_metric_effect(
        [_metric_value(reference_rows[key], metric) for key in keys],
        [_metric_value(estimate_rows[key], metric) for key in keys],
        bootstrap_seed=20260815,
        bootstrap_resamples=10_000,
    )


def build_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexed: dict[tuple[str, str, str], dict[tuple[int, int, int, int], dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        indexed[_condition_key(row)][_pair_key(row)] = row

    task_indexed: dict[
        tuple[str, int, str, str],
        dict[tuple[int, int, int, int], dict[str, Any]],
    ] = defaultdict(dict)
    for row in rows:
        task_indexed[_task_condition_key(row)][_pair_key(row)] = row

    summaries = []
    for model, runtime, profile in _sorted_conditions(indexed):
        cell = indexed[(model, runtime, profile)]
        summaries.append(
            {
                "model_key": model,
                "runtime": runtime,
                "delay_profile": profile,
                "metrics": {
                    metric: _summary(_metric_value(row, metric) for row in cell.values())
                    for metric in METRICS
                },
            }
        )

    comparisons = []
    comparison_pairs = (
        ("actionstream_aligned", "lerobot_weighted_average"),
        ("actionstream_aligned", "lerobot_latest_only"),
        ("actionstream_adaptive", "lerobot_latest_only"),
        ("actionstream_adaptive", "actionstream_aligned"),
        ("lerobot_rtc", "actionstream_aligned"),
        ("lerobot_rtc", "lerobot_weighted_average"),
        ("lerobot_rtc", "lerobot_latest_only"),
    )
    models = sorted({key[0] for key in indexed})
    profiles = sorted({key[2] for key in indexed})
    for model in models:
        for profile in profiles:
            for estimate, reference in comparison_pairs:
                estimate_key = model, estimate, profile
                reference_key = model, reference, profile
                if estimate_key not in indexed or reference_key not in indexed:
                    continue
                comparisons.append(
                    {
                        "comparison_type": "runtime",
                        "model_key": model,
                        "delay_profile": profile,
                        "difference_definition": "estimate minus reference",
                        "estimate_runtime": estimate,
                        "reference_runtime": reference,
                        "metrics": {
                            metric: _paired_effect(
                                indexed[reference_key], indexed[estimate_key], metric
                            )
                            for metric in METRICS
                        },
                    }
                )

    runtime_rank = {name: index for index, name in enumerate(RUNTIME_ORDER)}
    profile_rank = {name: index for index, name in enumerate(PROFILE_ORDER)}
    task_summaries = []
    for model, task_id, runtime, profile in sorted(
        task_indexed,
        key=lambda item: (
            item[0],
            item[1],
            profile_rank.get(item[3], len(profile_rank)),
            runtime_rank.get(item[2], len(runtime_rank)),
        ),
    ):
        cell = task_indexed[(model, task_id, runtime, profile)]
        task_summaries.append(
            {
                "model_key": model,
                "task_id": task_id,
                "runtime": runtime,
                "delay_profile": profile,
                "metrics": {
                    metric: _summary(_metric_value(row, metric) for row in cell.values())
                    for metric in METRICS
                },
            }
        )

    task_comparisons = []
    task_groups = sorted({(key[0], key[1], key[3]) for key in task_indexed})
    for model, task_id, profile in task_groups:
        for estimate, reference in comparison_pairs:
            estimate_key = model, task_id, estimate, profile
            reference_key = model, task_id, reference, profile
            if estimate_key not in task_indexed or reference_key not in task_indexed:
                continue
            task_comparisons.append(
                {
                    "comparison_type": "runtime_by_task",
                    "model_key": model,
                    "task_id": task_id,
                    "delay_profile": profile,
                    "difference_definition": "estimate minus reference",
                    "estimate_runtime": estimate,
                    "reference_runtime": reference,
                    "metrics": {
                        metric: _paired_effect(
                            task_indexed[reference_key], task_indexed[estimate_key], metric
                        )
                        for metric in METRICS
                    },
                }
            )

    for model in models:
        runtimes = sorted({key[1] for key in indexed if key[0] == model})
        for runtime in runtimes:
            reference_key = model, runtime, "fixed_0000"
            if reference_key not in indexed:
                continue
            for profile in PROFILE_ORDER[1:]:
                estimate_key = model, runtime, profile
                if estimate_key not in indexed:
                    continue
                comparisons.append(
                    {
                        "comparison_type": "delay_degradation",
                        "model_key": model,
                        "runtime": runtime,
                        "difference_definition": "delayed profile minus fixed_0000",
                        "estimate_profile": profile,
                        "reference_profile": "fixed_0000",
                        "metrics": {
                            metric: _paired_effect(
                                indexed[reference_key], indexed[estimate_key], metric
                            )
                            for metric in METRICS
                        },
                    }
                )

    return {
        "analysis_unit": "paired episode",
        "bootstrap": {
            "method": "paired nonparametric percentile bootstrap",
            "seed": 20260815,
            "resamples": 10_000,
            "confidence_level": 0.95,
        },
        "metric_definitions": {
            "success": "binary task success before the 280-step task horizon",
            "environment_steps": "simulator control steps until success or horizon",
            "hold_fraction": "fraction of dispatched actions that repeated the last command",
            "dropped_prefix_steps": "incoming action steps discarded as obsolete at merge",
            "delivery_latency_p50_seconds": "episode p50 request-to-delivery latency",
            "delivery_latency_p95_seconds": "episode p95 request-to-delivery latency",
            "action_discontinuity_mean_l2": "episode mean adjacent-command 7D L2",
            "action_discontinuity_max_l2": "episode maximum adjacent-command 7D L2",
            "action_acceleration_max_l2": "episode maximum second-difference 7D L2",
        },
        "condition_summaries": summaries,
        "paired_comparisons": comparisons,
        "task_condition_summaries": task_summaries,
        "task_paired_comparisons": task_comparisons,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    validation = report["validation"]
    lines = [
        "# Current LeRobot Async/RTC paired results",
        "",
        f"Validated {validation['episode_count']} completed episodes and "
        f"{validation['trace_count_verified']} raw traces containing "
        f"{validation['final_7d_action_count_verified']} finite 7D actions. "
        "The statistical unit is the "
        "episode; task, initial state, episode index, and seed are paired within each model.",
        "",
        "These results are paired within a policy only. They do not rank policy quality "
        "across X-VLA and SmolVLA.",
        "",
        "## Raw condition table",
        "",
        "| model | delay | runtime | n | success | steps mean±sd | hold mean±sd | "
        "delivery p50 mean±sd | discontinuity mean±sd |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["analysis"]["condition_summaries"]:
        metrics = cell["metrics"]
        lines.append(
            f"| {cell['model_key']} | {cell['delay_profile']} | {cell['runtime']} | "
            f"{metrics['success']['count']} | {_fmt(metrics['success']['mean'])} | "
            f"{_fmt(metrics['environment_steps']['mean'])}±"
            f"{_fmt(metrics['environment_steps']['std'])} | "
            f"{_fmt(metrics['hold_fraction']['mean'])}±"
            f"{_fmt(metrics['hold_fraction']['std'])} | "
            f"{_fmt(metrics['delivery_latency_p50_seconds']['mean'])}±"
            f"{_fmt(metrics['delivery_latency_p50_seconds']['std'])} | "
            f"{_fmt(metrics['action_discontinuity_mean_l2']['mean'])}±"
            f"{_fmt(metrics['action_discontinuity_mean_l2']['std'])} |"
        )

    lines += [
        "",
        "## Task-stratified raw condition table",
        "",
        "| model | task | delay | runtime | n | success | steps mean±sd | "
        "hold mean±sd |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for cell in report["analysis"]["task_condition_summaries"]:
        metrics = cell["metrics"]
        lines.append(
            f"| {cell['model_key']} | {cell['task_id']} | {cell['delay_profile']} | "
            f"{cell['runtime']} | {metrics['success']['count']} | "
            f"{_fmt(metrics['success']['mean'])} | "
            f"{_fmt(metrics['environment_steps']['mean'])}±"
            f"{_fmt(metrics['environment_steps']['std'])} | "
            f"{_fmt(metrics['hold_fraction']['mean'])}±"
            f"{_fmt(metrics['hold_fraction']['std'])} |"
        )

    lines += [
        "",
        "## Paired runtime effects",
        "",
        "Differences are estimate minus reference. Intervals are descriptive; raw "
        "outcomes and task-stratified effects remain the primary evidence.",
        "",
        "| model | delay | estimate - reference | n | success Δ [95% CI] | "
        "steps Δ [95% CI] | discontinuity Δ [95% CI] |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for comparison in report["analysis"]["paired_comparisons"]:
        if comparison["comparison_type"] != "runtime":
            continue
        metrics = comparison["metrics"]

        def effect(name: str) -> str:
            metric = metrics[name]
            low, high = metric["paired_mean_difference_95pct_bootstrap_ci"]
            return f"{_fmt(metric['paired_mean_difference'])} [{_fmt(low)}, {_fmt(high)}]"

        lines.append(
            f"| {comparison['model_key']} | {comparison['delay_profile']} | "
            f"{comparison['estimate_runtime']} - {comparison['reference_runtime']} | "
            f"{metrics['success']['paired_episode_count']} | {effect('success')} | "
            f"{effect('environment_steps')} | {effect('action_discontinuity_mean_l2')} |"
        )

    lines += [
        "",
        "## Task-stratified paired runtime effects",
        "",
        "| model | task | delay | estimate - reference | n | success Δ [95% CI] | "
        "steps Δ [95% CI] | hold Δ [95% CI] |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for comparison in report["analysis"]["task_paired_comparisons"]:
        metrics = comparison["metrics"]

        def task_effect(name: str) -> str:
            metric = metrics[name]
            low, high = metric["paired_mean_difference_95pct_bootstrap_ci"]
            return f"{_fmt(metric['paired_mean_difference'])} [{_fmt(low)}, {_fmt(high)}]"

        lines.append(
            f"| {comparison['model_key']} | {comparison['task_id']} | "
            f"{comparison['delay_profile']} | {comparison['estimate_runtime']} - "
            f"{comparison['reference_runtime']} | "
            f"{metrics['success']['paired_episode_count']} | {task_effect('success')} | "
            f"{task_effect('environment_steps')} | {task_effect('hold_fraction')} |"
        )

    lines += [
        "",
        "## Evidence boundary",
        "",
        "- Smoke episodes are excluded from this report.",
        "- A 280-step horizon failure is a task outcome; a shorter intentional smoke "
        "truncation is not.",
        "- Cross-model comparisons are not paired and are not used for method claims.",
        "- Per-episode delivery p95 includes the first cold inference and depends on the "
        "number of inference calls, so the compact table uses p50; p95 remains in JSON/CSV.",
        "- Native Isaac evidence is reported separately and requires a completed native "
        "runner receipt plus a live observation-conditioned policy.",
        "",
    ]
    return "\n".join(lines)


def write_analysis(
    episode_paths: Iterable[Path | str],
    *,
    output_dir: Path | str,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    paths = [Path(path) for path in episode_paths]
    rows = read_episode_rows(paths)
    validation = validate_episode_rows(rows, verify_artifacts=verify_artifacts)
    report = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_episode_jsonl": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in paths
        ],
        "validation": validation,
        "analysis": build_analysis(rows),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "paired_analysis.md").write_text(
        render_markdown(report),
        encoding="utf-8",
        newline="\n",
    )
    with (output / "raw_episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=RAW_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-artifact-verification", action="store_true")
    args = parser.parse_args()
    write_analysis(
        args.episode_paths,
        output_dir=args.output_dir,
        verify_artifacts=not args.skip_artifact_verification,
    )


if __name__ == "__main__":
    main()
