"""Paired episode-level analysis of the frozen ActionStream M3 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from actionstream.results import (
    EXPECTED_M3_CONDITIONS,
    EXPECTED_TASK_IDS,
    _resolve_action_trace_path,
    read_episode_jsonl,
    validate_m3_matrix,
)


BOOTSTRAP_SEED = 20260730
BOOTSTRAP_RESAMPLES = 10_000
METRICS = {
    "environment_steps": "environment steps",
    "wall_clock_episode_seconds": "wall-clock episode seconds",
    "observation_to_delivery_p50_seconds": "episode p50 observation-to-delivery seconds",
    "action_discontinuity_mean_l2": "episode mean adjacent-action 7D L2",
    "success": "binary episode success",
}
PAIR_FIELDS = (
    "task_id",
    "episode_index",
    "initial_state_index",
    "seed",
    "policy_rng_seed",
    "suite",
    "model_id",
    "model_revision_sha",
)
REPLACEMENT_INDEX_FIELDS = {
    "accepted_action_index",
    "accepted_control_step",
    "merge_action_index",
    "merge_control_step",
    "replacement_action_index",
    "replacement_control_step",
    "replacement_index",
}
COMPARISONS = [
    *[
        (
            f"async_aligned_minus_{reference}_delay{delay}",
            ("async_aligned", delay),
            (reference, delay),
        )
        for delay in (0, 200)
        for reference in ("async_naive", "sync")
    ],
    *[
        (f"{mode}_delay200_minus_delay0", (mode, 200), (mode, 0))
        for mode in ("sync", "async_naive", "async_aligned")
    ],
]


def _condition(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["runtime_mode"]), int(row["injected_delay_ms"])


def _episode(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["task_id"]), int(row["episode_index"])


def _value(row: dict[str, Any], metric: str) -> float:
    value = float(bool(row[metric])) if metric == "success" else float(row[metric])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {metric} in {_condition(row)}/{_episode(row)}")
    return value


def _summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Summary requires finite episode values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }


def paired_metric_effect(
    reference: Iterable[float],
    estimate: Iterable[float],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Return the paired ``estimate - reference`` effect."""
    reference_values = np.asarray(list(reference), dtype=np.float64)
    estimate_values = np.asarray(list(estimate), dtype=np.float64)
    if (
        reference_values.ndim != 1
        or reference_values.shape != estimate_values.shape
        or not reference_values.size
    ):
        raise ValueError("Paired values must be nonempty same-length vectors")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if not np.isfinite(reference_values).all() or not np.isfinite(estimate_values).all():
        raise ValueError("Paired values must be finite")

    differences = estimate_values - reference_values
    indices = np.random.default_rng(bootstrap_seed).integers(
        0,
        len(differences),
        size=(bootstrap_resamples, len(differences)),
    )
    difference_ci = np.percentile(differences[indices].mean(axis=1), [2.5, 97.5])
    relative_ci: list[float] | None = None
    relative_mean: float | None = None
    if np.all(reference_values != 0):
        relative = differences / reference_values
        relative_mean = 100.0 * float(relative.mean())
        relative_ci = [
            float(value)
            for value in 100.0
            * np.percentile(relative[indices].mean(axis=1), [2.5, 97.5])
        ]
    return {
        "paired_episode_count": int(len(differences)),
        "reference": _summary(reference_values),
        "estimate": _summary(estimate_values),
        "paired_mean_difference": float(differences.mean()),
        "paired_median_difference": float(np.median(differences)),
        "paired_mean_difference_95pct_bootstrap_ci": [
            float(difference_ci[0]),
            float(difference_ci[1]),
        ],
        "paired_relative_difference_percent": relative_mean,
        "paired_relative_difference_95pct_bootstrap_ci": relative_ci,
        "relative_difference_available": relative_ci is not None,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": bootstrap_resamples,
    }


def _index(records: list[dict[str, Any]]) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    expected_rows = len(EXPECTED_M3_CONDITIONS) * len(EXPECTED_TASK_IDS) * 10
    if len(records) != expected_rows:
        raise ValueError(f"Expected {expected_rows} frozen M3 rows, got {len(records)}")
    indexed: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in records:
        mode, delay = _condition(row)
        task, episode = _episode(row)
        if (mode, delay) not in EXPECTED_M3_CONDITIONS:
            raise ValueError(f"Unexpected condition {(mode, delay)}")
        if task not in EXPECTED_TASK_IDS or episode not in range(10):
            raise ValueError(f"Unexpected episode key {(task, episode)}")
        if type(row.get("success")) is not bool:
            raise ValueError(f"Non-boolean success in {(mode, delay, task, episode)}")
        for metric in METRICS:
            if row.get(metric) is None:
                raise ValueError(f"Missing {metric} in {(mode, delay, task, episode)}")
            _value(row, metric)
        key = mode, delay, task, episode
        if key in indexed:
            raise ValueError(f"Duplicate episode row {key}")
        indexed[key] = row

    for task in EXPECTED_TASK_IDS:
        for episode in range(10):
            rows = [
                indexed[(mode, delay, task, episode)]
                for mode, delay in EXPECTED_M3_CONDITIONS
            ]
            for field in PAIR_FIELDS:
                if len({json.dumps(row.get(field), sort_keys=True) for row in rows}) != 1:
                    raise ValueError(
                        f"Pair metadata mismatch for task {task} episode {episode}: {field}"
                    )
    return indexed


def _rows(
    indexed: dict[tuple[str, int, int, int], dict[str, Any]],
    condition: tuple[str, int],
    task: int | None = None,
) -> list[dict[str, Any]]:
    tasks = EXPECTED_TASK_IDS if task is None else (task,)
    return [
        indexed[(condition[0], condition[1], task_id, episode)]
        for task_id in tasks
        for episode in range(10)
    ]


def _condition_stats(
    indexed: dict[tuple[str, int, int, int], dict[str, Any]],
    condition: tuple[str, int],
    task: int | None = None,
) -> dict[str, Any]:
    rows = _rows(indexed, condition, task)
    return {metric: _summary(_value(row, metric) for row in rows) for metric in METRICS}


def _comparison(
    indexed: dict[tuple[str, int, int, int], dict[str, Any]],
    comparison_id: str,
    estimate_condition: tuple[str, int],
    reference_condition: tuple[str, int],
) -> dict[str, Any]:
    def effects(task: int | None) -> dict[str, Any]:
        reference = _rows(indexed, reference_condition, task)
        estimate = _rows(indexed, estimate_condition, task)
        for reference_row, estimate_row in zip(reference, estimate, strict=True):
            if any(reference_row.get(field) != estimate_row.get(field) for field in PAIR_FIELDS):
                raise ValueError(f"{comparison_id} contains an unpaired episode")
        return {
            metric: paired_metric_effect(
                [_value(row, metric) for row in reference],
                [_value(row, metric) for row in estimate],
            )
            for metric in METRICS
        }

    return {
        "comparison_id": comparison_id.replace("_sync_", "_blocking_sync_"),
        "difference_definition": "estimate minus reference",
        "estimate_condition": {
            "runtime_mode": estimate_condition[0],
            "injected_delay_ms": estimate_condition[1],
            "semantic_label": (
                "blocking evaluator baseline"
                if estimate_condition[0] == "sync"
                else estimate_condition[0]
            ),
        },
        "reference_condition": {
            "runtime_mode": reference_condition[0],
            "injected_delay_ms": reference_condition[1],
            "semantic_label": (
                "blocking evaluator baseline"
                if reference_condition[0] == "sync"
                else reference_condition[0]
            ),
        },
        "overall": effects(None),
        "per_task": {str(task): effects(task) for task in EXPECTED_TASK_IDS},
    }


def inspect_historical_replacement_indices(
    records: list[dict[str, Any]],
    *,
    episode_paths: list[Path | str],
) -> dict[str, Any]:
    """Check direct trace fields; do not infer a boundary index from age metadata."""
    directories: dict[tuple[str, int], Path] = {}
    for value in episode_paths:
        path = Path(value)
        for mode, delay in EXPECTED_M3_CONDITIONS:
            if path.parent.name == f"{mode}_delay{delay}":
                directories[(mode, delay)] = path.parent

    trace_count = event_count = replacement_count = 0
    top_fields: set[str] = set()
    merge_fields: set[str] = set()
    index_fields: set[str] = set()
    for row in records:
        path = _resolve_action_trace_path(
            str(row["action_trace_path"]),
            condition=_condition(row),
            condition_episode_dirs=directories,
        )
        with np.load(path, allow_pickle=False) as trace:
            events = json.loads(str(trace["inference_events_json"].item()))
        trace_count += 1
        for event in events:
            event_count += 1
            top_fields.update(event)
            index_fields.update(REPLACEMENT_INDEX_FIELDS.intersection(event))
            merge = event.get("merge")
            if isinstance(merge, dict):
                merge_fields.update(merge)
                index_fields.update(REPLACEMENT_INDEX_FIELDS.intersection(merge))
                replacement_count += int(
                    bool(merge.get("accepted")) and merge.get("reason") == "replaced"
                )

    available = bool(index_fields)
    return {
        "trace_count_inspected": trace_count,
        "inference_event_count_inspected": event_count,
        "accepted_replacement_event_count": replacement_count,
        "top_level_event_fields": sorted(top_fields),
        "merge_event_fields": sorted(merge_fields),
        "explicit_replacement_index_fields_found": sorted(index_fields),
        "replacement_boundary_indices_available": available,
        "replacement_boundary_metrics": {
            "position_l2_jump": None,
            "rotation_geodesic_angle_jump": None,
            "gripper_state_changes": None,
        },
        "no_reverse_engineering_performed": not available,
        "reason": (
            "Historical M3 events have observation_control_step and merge age but no "
            "explicit accepted-chunk replacement index, so boundary component jumps "
            "were not reconstructed."
            if not available
            else "An explicit replacement index is present."
        ),
    }


def build_paired_m3_analysis(
    records: list[dict[str, Any]],
    *,
    replacement_audit: dict[str, Any],
) -> dict[str, Any]:
    indexed = _index(records)
    conditions = []
    for condition in sorted(EXPECTED_M3_CONDITIONS):
        conditions.append(
            {
                "runtime_mode": condition[0],
                "injected_delay_ms": condition[1],
                "semantic_label": (
                    "blocking evaluator baseline"
                    if condition[0] == "sync"
                    else condition[0]
                ),
                "overall": _condition_stats(indexed, condition),
                "per_task": {
                    str(task): _condition_stats(indexed, condition, task)
                    for task in EXPECTED_TASK_IDS
                },
            }
        )
    return {
        "analysis_unit": "episode",
        "episode_count": len(records),
        "pairing_fields": ["task_id", "episode_index", "initial_state_index", "seed"],
        "bootstrap": {
            "method": "paired nonparametric percentile bootstrap",
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
        },
        "metric_definitions": METRICS,
        "condition_summaries": conditions,
        "paired_comparisons": [
            _comparison(indexed, comparison_id, estimate, reference)
            for comparison_id, estimate, reference in COMPARISONS
        ],
        "historical_replacement_boundary_audit": replacement_audit,
        "interpretation_boundary": (
            "All 180 episodes succeeded. Success-rate differences are ceilinged and "
            "do not establish equal robustness."
        ),
    }


def _format(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def render_paired_analysis_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ActionStream M4 Section A: paired M3 evidence",
        "",
        "The episode is the statistical unit; the 29,073 actions are not independent "
        "samples. Pairs share task, episode/initial-state index, and seed. Differences "
        "below are **estimate minus reference**.",
        "",
        f"Bootstrap: {report['bootstrap']['resamples']:,} paired resamples, fixed seed "
        f"`{report['bootstrap']['seed']}`, percentile 95% intervals.",
        "",
        "Observation-to-delivery is the existing episode p50 summarized over episodes. "
        "Discontinuity is the existing episode mean adjacent-command 7D L2.",
        "",
        "## Raw episode summaries",
        "",
        "| mode | delay | scope | n | success mean/median | steps mean/median | "
        "wall s mean/median | delivery s mean/median | discontinuity mean/median |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in report["condition_summaries"]:
        scopes = [("all", condition["overall"])] + [
            (f"task {task}", condition["per_task"][str(task)])
            for task in EXPECTED_TASK_IDS
        ]
        for scope, metrics in scopes:
            formatted = {
                metric: f"{_format(values['mean'])}/{_format(values['median'])}"
                for metric, values in metrics.items()
            }
            lines.append(
                f"| {condition['semantic_label']} | {condition['injected_delay_ms']} | "
                f"{scope} | {metrics['success']['count']} | {formatted['success']} | "
                f"{formatted['environment_steps']} | "
                f"{formatted['wall_clock_episode_seconds']} | "
                f"{formatted['observation_to_delivery_p50_seconds']} | "
                f"{formatted['action_discontinuity_mean_l2']} |"
            )

    lines += [
        "",
        "## Paired effects",
        "",
        "Relative effects are means of paired episode-wise relative differences.",
        "",
        "| comparison | scope | metric | n | reference mean/median | estimate mean/median "
        "| mean difference [95% CI] | relative difference [95% CI] |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for comparison in report["paired_comparisons"]:
        estimate = comparison["estimate_condition"]
        reference = comparison["reference_condition"]
        label = (
            f"{estimate['semantic_label']}/{estimate['injected_delay_ms']} minus "
            f"{reference['semantic_label']}/{reference['injected_delay_ms']}"
        )
        scopes = [("all", comparison["overall"])] + [
            (f"task {task}", comparison["per_task"][str(task)])
            for task in EXPECTED_TASK_IDS
        ]
        for scope, metrics in scopes:
            for metric, name in METRICS.items():
                effect = metrics[metric]
                ci = effect["paired_mean_difference_95pct_bootstrap_ci"]
                relative = effect["paired_relative_difference_percent"]
                relative_ci = effect["paired_relative_difference_95pct_bootstrap_ci"]
                relative_text = (
                    f"{_format(relative, 2)}% "
                    f"[{_format(relative_ci[0], 2)}, {_format(relative_ci[1], 2)}]%"
                    if relative_ci is not None
                    else "n/a"
                )
                lines.append(
                    f"| {label} | {scope} | {name} | {effect['paired_episode_count']} | "
                    f"{_format(effect['reference']['mean'])}/"
                    f"{_format(effect['reference']['median'])} | "
                    f"{_format(effect['estimate']['mean'])}/"
                    f"{_format(effect['estimate']['median'])} | "
                    f"{_format(effect['paired_mean_difference'])} "
                    f"[{_format(ci[0])}, {_format(ci[1])}] | {relative_text} |"
                )

    audit = report["historical_replacement_boundary_audit"]
    lines += [
        "",
        "## Historical replacement-boundary audit",
        "",
        f"{audit['trace_count_inspected']} traces and "
        f"{audit['inference_event_count_inspected']} inference events contain "
        f"{audit['accepted_replacement_event_count']} accepted replacements. "
        f"Explicit replacement-index fields: "
        f"`{audit['explicit_replacement_index_fields_found']}`.",
        "",
        audit["reason"],
        "",
        "## Interpretation boundary",
        "",
        report["interpretation_boundary"],
        "",
    ]
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_paired_m3_analysis(
    episode_paths: list[Path | str],
    *,
    output_dir: Path | str,
) -> dict[str, Any]:
    paths = [Path(path) for path in episode_paths]
    validation = validate_m3_matrix(paths, verify_action_traces=True)
    records = read_episode_jsonl(paths)
    audit = inspect_historical_replacement_indices(records, episode_paths=paths)
    if audit["replacement_boundary_indices_available"]:
        raise ValueError(
            "Replacement indices exist; component-boundary analysis is required"
        )
    analysis = build_paired_m3_analysis(records, replacement_audit=audit)
    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_episode_jsonl": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "m3_validation": {
            "passed": validation["gate"]["passed"],
            "episode_count": validation["episode_count"],
            "action_trace_count_verified": validation["action_trace_count_verified"],
            "total_final_7d_actions_verified": validation[
                "total_final_7d_actions_verified"
            ],
            "git_commit": validation["git_commit"],
            "model_revision_sha": validation["model_revision_sha"],
        },
        **analysis,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "paired_analysis.md").write_text(
        render_paired_analysis_markdown(report),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m4/paired_m3"))
    args = parser.parse_args()
    write_paired_m3_analysis(args.episode_paths, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
