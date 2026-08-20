"""Audit and report the frozen ActionStream LeRobot GPU benchmark v1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from actionstream.m4_analysis import paired_metric_effect


PAIR_FIELDS = ("suite", "task_id", "episode_index", "initial_state_index", "seed")
PAIR_INVARIANTS = (
    "source_commit",
    "model_id",
    "model_revision",
    "control_mode",
    "delay_trace_sha256",
    "controller_frequency_hz",
    "chunk_size",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split(row: dict[str, Any]) -> str:
    experiment = str(row["experiment_id"])
    if "holdout" in experiment:
        return "holdout"
    if "canary" in experiment:
        return "canary"
    return "development"


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in PAIR_FIELDS)


def _optional_number(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field}: {value!r}")
    return number


def _summary(values: Iterable[float | None]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    if not np.isfinite(array).all():
        raise ValueError("Summary received a non-finite value")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _resolve_artifact(row: dict[str, Any], field: str) -> Path | None:
    raw = row.get(field)
    if not raw:
        return None
    episodes_path = Path(row["_episodes_path"])
    category = "traces" if field == "trace_path" else "videos"
    return episodes_path.parent / category / Path(str(raw)).name


def read_results(input_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for episodes_path in sorted(input_root.glob("*/episodes.jsonl")):
        if not episodes_path.stat().st_size:
            continue
        for line_number, line in enumerate(episodes_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_episodes_path"] = str(episodes_path.resolve())
            row["_line_number"] = line_number
            rows.append(row)
        receipt_path = episodes_path.with_name("run_receipt.json")
        if not receipt_path.is_file():
            raise ValueError(f"Missing run receipt beside {episodes_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["_receipt_path"] = str(receipt_path.resolve())
        receipt["_episodes_path"] = str(episodes_path.resolve())
        receipts.append(receipt)
    if not rows:
        raise ValueError(f"No completed result files under {input_root}")
    return rows, receipts


def validate_results(
    rows: list[dict[str, Any]], receipts: list[dict[str, Any]]
) -> dict[str, Any]:
    trace_count = 0
    video_count = 0
    action_count = 0
    seen: set[tuple[Any, ...]] = set()
    sources: set[str] = set()

    receipt_by_episodes = {item["_episodes_path"]: item for item in receipts}
    for episodes_path, receipt in receipt_by_episodes.items():
        path = Path(episodes_path)
        if receipt.get("metrics_sha256") != _sha256(path):
            raise ValueError(f"Metrics hash mismatch for {path}")
        count = sum(row["_episodes_path"] == episodes_path for row in rows)
        if int(receipt.get("record_count", -1)) != count:
            raise ValueError(f"Receipt count mismatch for {path}: {count}")

    for row in rows:
        location = f"{row['_episodes_path']}:{row['_line_number']}"
        if row.get("status") != "completed":
            raise ValueError(f"Non-completed formal record at {location}: {row.get('status')}")
        if type(row.get("success")) is not bool:
            raise ValueError(f"Non-boolean success at {location}")
        for field in ("environment_steps", "wall_clock_episode_seconds", "peak_cuda_memory_mib"):
            if _optional_number(row, field) is None:
                raise ValueError(f"Missing {field} at {location}")
        unique = (
            row["experiment_id"],
            row["model_key"],
            row["runtime"],
            row["delay_profile"],
            *_pair_key(row),
        )
        if unique in seen:
            raise ValueError(f"Duplicate result cell {unique}")
        seen.add(unique)
        sources.add(str(row["source_commit"]))

        trace = _resolve_artifact(row, "trace_path")
        if trace is None or not trace.is_file() or _sha256(trace) != row.get("trace_sha256"):
            raise ValueError(f"Missing or corrupt trace at {location}: {trace}")
        payload = json.loads(trace.read_text(encoding="utf-8"))
        actions = payload.get("actions")
        events = payload.get("inference_events")
        if not isinstance(actions, list) or not isinstance(events, list):
            raise ValueError(f"Malformed trace {trace}")
        if len(actions) != int(row["environment_steps"]):
            raise ValueError(f"Action count mismatch in {trace}")
        trace_count += 1
        action_count += len(actions)

        video = _resolve_artifact(row, "video_path")
        if row.get("video_path"):
            if video is None or not video.is_file() or _sha256(video) != row.get("video_sha256"):
                raise ValueError(f"Missing or corrupt video at {location}: {video}")
            video_count += 1

    paired: dict[tuple[str, str, str, tuple[Any, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        paired[(_split(row), str(row["model_key"]), str(row["delay_profile"]), _pair_key(row))].append(row)
    for key, pair in paired.items():
        if len(pair) < 2:
            continue
        for field in PAIR_INVARIANTS:
            values = {json.dumps(row.get(field), sort_keys=True) for row in pair}
            if len(values) != 1:
                raise ValueError(f"Pair invariant mismatch for {key}: {field}")

    return {
        "episode_count": len(rows),
        "receipt_count": len(receipts),
        "trace_count_verified": trace_count,
        "video_count_verified": video_count,
        "action_count_verified": action_count,
        "source_commits": sorted(sources),
        "pair_fields": list(PAIR_FIELDS),
        "pair_invariants": list(PAIR_INVARIANTS),
    }


def _condition_row(group: list[dict[str, Any]]) -> dict[str, Any]:
    first = group[0]
    wall = [float(row["wall_clock_episode_seconds"]) for row in group]
    steps = [float(row["environment_steps"]) for row in group]
    throughput = [step / seconds for step, seconds in zip(steps, wall, strict=True)]
    inference_throughput = [
        float(
            row.get("inference_completed")
            if row.get("inference_completed") is not None
            else row.get("inference_calls", 0)
        )
        / seconds
        for row, seconds in zip(group, wall, strict=True)
    ]
    generated_action_throughput = [
        float(row["chunk_size"]) / float(row["inference_latency_p50_seconds"])
        for row in group
        if _optional_number(row, "inference_latency_p50_seconds") not in (None, 0.0)
    ]
    result: dict[str, Any] = {
        "split": _split(first),
        "model": first["model_key"],
        "runtime": first["runtime"],
        "delay_profile": first["delay_profile"],
        "episodes": len(group),
        "successes": sum(bool(row["success"]) for row in group),
        "success_rate": float(np.mean([bool(row["success"]) for row in group])),
        "mean_environment_steps": float(np.mean(steps)),
        "environment_steps_per_second_mean": float(np.mean(throughput)),
        "completed_inferences_per_second_mean": float(np.mean(inference_throughput)),
        # GPU-synchronized inference latency is measured around
        # predict_action_chunk.  Dividing the emitted chunk length by that
        # latency reports model action-generation throughput without folding in
        # simulator stepping or injected network delay.
        "gpu_generated_actions_per_second_p50_median": (
            float(np.median(generated_action_throughput))
            if generated_action_throughput
            else None
        ),
        "peak_cuda_memory_mib_max": max(float(row["peak_cuda_memory_mib"]) for row in group),
    }
    metrics = {
        "inference_latency_p50_seconds": "inference_latency_p50_seconds_median",
        "inference_latency_p95_seconds": "inference_latency_p95_seconds_median",
        "delivery_latency_p50_seconds": "delivery_latency_p50_seconds_median",
        "delivery_latency_p95_seconds": "delivery_latency_p95_seconds_median",
        "queue_depth_p50_steps": "queue_depth_p50_steps_median",
        "queue_depth_p95_steps": "queue_depth_p95_steps_median",
        "queue_age_p50_steps": "queue_age_p50_steps_median",
        "queue_age_p95_steps": "queue_age_p95_steps_median",
    }
    for source, target in metrics.items():
        result[target] = _summary(_optional_number(row, source) for row in group)["median"]
    for field in (
        "dropped_prefix_steps",
        "hold_steps",
        "bounded_hold_steps",
        "depletion_safe_hold_steps",
        "disconnects",
        "recoveries",
        "fallback_activations",
        "fallback_chunks_accepted",
    ):
        values = [_optional_number(row, field) for row in group]
        result[f"{field}_sum"] = None if all(value is None for value in values) else sum(
            value or 0.0 for value in values
        )
    return result


def build_main_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(_split(row), str(row["model_key"]), str(row["runtime"]), str(row["delay_profile"]))].append(row)
    return [_condition_row(grouped[key]) for key in sorted(grouped)]


def build_task_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                _split(row),
                str(row["model_key"]),
                str(row["suite"]),
                str(row["runtime"]),
                str(row["delay_profile"]),
            )
        ].append(row)
    task_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        summary = _condition_row(grouped[key])
        summary["suite"] = key[2]
        task_rows.append(summary)
    return task_rows


def build_paired_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(_split(row), str(row["model_key"]), str(row["delay_profile"]))].append(row)
    effects: list[dict[str, Any]] = []
    for (split, model, profile), group in sorted(groups.items()):
        by_runtime: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
        for row in group:
            by_runtime[str(row["runtime"])][_pair_key(row)] = row
        reference_name = "lerobot_latest_only"
        if reference_name not in by_runtime:
            continue
        reference = by_runtime[reference_name]
        for runtime, estimate in sorted(by_runtime.items()):
            if runtime == reference_name:
                continue
            if set(reference) != set(estimate):
                raise ValueError(
                    f"Unpaired effect {split}/{model}/{profile}/{runtime}: "
                    f"reference={len(reference)} estimate={len(estimate)}"
                )
            keys = sorted(reference)
            for metric in ("success", "environment_steps"):
                ref_values = [
                    float(bool(reference[key][metric])) if metric == "success" else float(reference[key][metric])
                    for key in keys
                ]
                est_values = [
                    float(bool(estimate[key][metric])) if metric == "success" else float(estimate[key][metric])
                    for key in keys
                ]
                effect = paired_metric_effect(
                    ref_values,
                    est_values,
                    bootstrap_seed=20260820,
                    bootstrap_resamples=10_000,
                )
                effects.append(
                    {
                        "split": split,
                        "model": model,
                        "delay_profile": profile,
                        "reference_runtime": reference_name,
                        "estimate_runtime": runtime,
                        "metric": metric,
                        "paired_episode_count": effect["paired_episode_count"],
                        "reference_mean": effect["reference"]["mean"],
                        "estimate_mean": effect["estimate"]["mean"],
                        "paired_mean_difference": effect["paired_mean_difference"],
                        "ci95_low": effect["paired_mean_difference_95pct_bootstrap_ci"][0],
                        "ci95_high": effect["paired_mean_difference_95pct_bootstrap_ci"][1],
                        "paired_relative_difference_percent": effect["paired_relative_difference_percent"],
                    }
                )
    return effects


def build_failure_taxonomy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, str, str, str]] = Counter()
    for row in rows:
        if row["success"]:
            continue
        if int(row.get("depletion_safe_hold_steps", 0) or 0) > 0:
            category = "queue_depletion_safe_hold"
        elif int(row.get("fallback_activations", 0) or 0) > 0:
            category = "fallback_boundary_failure"
        elif int(row.get("hold_steps", 0) or 0) > 0:
            category = "queue_empty_bounded_hold"
        elif int(row.get("dropped_prefix_steps", 0) or 0) > int(row.get("chunk_size", 0) or 0):
            category = "heavy_stale_discard"
        else:
            category = "task_failure_other"
        counts[
            (
                _split(row),
                str(row["model_key"]),
                str(row["suite"]),
                str(row["runtime"]),
                str(row["delay_profile"]),
                category,
            )
        ] += 1
    return [
        {
            "split": key[0],
            "model": key[1],
            "suite": key[2],
            "runtime": key[3],
            "delay_profile": key[4],
            "category": key[5],
            "count": count,
        }
        for key, count in sorted(counts.items())
    ]


def evaluate_gates(rows: list[dict[str, Any]], receipts: list[dict[str, Any]]) -> dict[str, Any]:
    xvla_sync = [
        row
        for row in rows
        if _split(row) == "canary"
        and row["model_key"] == "xvla"
        and row["runtime"] == "sync_hold"
        and row["delay_profile"] == "fixed_0000"
    ]
    xvla_sync_suites = {row["suite"] for row in xvla_sync if row["success"]}
    expected_suites = {"libero_object", "libero_spatial", "libero_goal"}
    disconnect_rows = [
        row
        for row in rows
        if _split(row) == "canary"
        and row["model_key"] == "xvla"
        and row["delay_profile"] == "disconnect_recovery_canary"
    ]
    disconnect_ok = bool(disconnect_rows) and all(
        int(row.get("disconnects", 0)) == 2
        and int(row.get("recoveries", 0)) == 2
        and row["status"] == "completed"
        for row in disconnect_rows
    )
    smol_sync = [
        row
        for row in rows
        if _split(row) == "canary"
        and row["model_key"] == "smolvla"
        and row["runtime"] == "sync_hold"
        and row["delay_profile"] == "fixed_0000"
    ]
    smol_rtc = [
        row
        for row in rows
        if _split(row) == "canary"
        and row["model_key"] == "smolvla"
        and row["runtime"] == "lerobot_rtc"
    ]
    rtc_capability = any(
        any(
            item.get("model_key") == "smolvla"
            and item.get("rtc_supported") is True
            and item.get("rtc_expectation_match") is True
            for item in receipt.get("compatibility", [])
        )
        for receipt in receipts
    )
    return {
        "xvla_sync_zero_delay_three_suite_gate": xvla_sync_suites == expected_suites,
        "xvla_sync_success_suites": sorted(xvla_sync_suites),
        "xvla_disconnect_recovery_gate": disconnect_ok,
        "xvla_disconnect_rows": len(disconnect_rows),
        "smolvla_sync_zero_delay_gate": len(smol_sync) == 1 and smol_sync[0]["success"],
        "smolvla_rtc_completion_gate": bool(smol_rtc),
        "smolvla_upstream_rtc_capability_gate": rtc_capability,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_latency_plot(path: Path, table: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    holdout = [row for row in table if row["split"] == "holdout"]
    models = sorted({row["model"] for row in holdout}) or sorted({row["model"] for row in table})
    figure, axes = plt.subplots(1, max(1, len(models)), figsize=(7 * max(1, len(models)), 5), squeeze=False)
    for axis, model in zip(axes[0], models, strict=True):
        model_rows = [row for row in (holdout or table) if row["model"] == model]
        for runtime in sorted({row["runtime"] for row in model_rows}):
            points = [row for row in model_rows if row["runtime"] == runtime]
            points = [row for row in points if row["delivery_latency_p50_seconds_median"] is not None]
            points.sort(key=lambda row: float(row["delivery_latency_p50_seconds_median"]))
            if points:
                axis.plot(
                    [1000.0 * float(row["delivery_latency_p50_seconds_median"]) for row in points],
                    [100.0 * float(row["success_rate"]) for row in points],
                    marker="o",
                    label=runtime,
                )
        axis.set_title(f"{model}: observed operating points")
        axis.set_xlabel("Measured episode median delivery latency (ms)")
        axis.set_ylabel("Success rate (%)")
        axis.set_ylim(-3, 103)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(input_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, receipts = read_results(input_root)
    validation = validate_results(rows, receipts)
    main_table = build_main_table(rows)
    task_table = build_task_table(rows)
    paired_effects = build_paired_effects(rows)
    failures = build_failure_taxonomy(rows)
    gates = evaluate_gates(rows, receipts)
    warmup_peaks = [
        float(warmup["peak_cuda_memory_mib"])
        for receipt in receipts
        for warmup in receipt.get("gpu_warmup", [])
        if warmup.get("peak_cuda_memory_mib") is not None
    ]

    _write_csv(output_dir / "main_table.csv", main_table)
    _write_csv(output_dir / "task_table.csv", task_table)
    _write_csv(output_dir / "paired_effects.csv", paired_effects)
    _write_csv(output_dir / "failure_taxonomy.csv", failures)
    _write_latency_plot(output_dir / "latency_success_operating_points.png", main_table)

    summary = {
        "schema_version": 1,
        "validation": validation,
        "gates": gates,
        "condition_count": len(main_table),
        "paired_effect_count": len(paired_effects),
        "failure_count": sum(int(row["count"]) for row in failures),
        "gpu_memory_summary": {
            "non_scored_warmup_peak_cuda_memory_mib_max": (
                max(warmup_peaks) if warmup_peaks else None
            ),
            "scored_episode_peak_cuda_memory_mib_max": max(
                float(row["peak_cuda_memory_mib_max"]) for row in main_table
            ),
            "nvidia_smi_process_residency_mib": None,
        },
        "telemetry_boundary": {
            "queue_age": "Available for ActionStreamInferenceEngine records; N/A for upstream queues that do not expose action-source age.",
            "disconnect_recovery": "Backend-only canary, excluded from official-baseline paired effects.",
            "gpu_memory": "torch.cuda.max_memory_allocated, not total nvidia-smi process residency.",
            "gpu_throughput": "Chunk actions divided by GPU-synchronized p50 predict_action_chunk latency; excludes simulator and injected network delay.",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    holdout_rows = [row for row in main_table if row["split"] == "holdout"]
    holdout_effects = [row for row in paired_effects if row["split"] == "holdout"]

    def compact(value: Any, *, digits: int = 2) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    def milliseconds(value: Any) -> str:
        return "N/A" if value is None else f"{1000.0 * float(value):.1f}"

    report_lines = [
        "# ActionStream LeRobot GPU benchmark v1",
        "",
        f"Validated {validation['episode_count']} episodes, {validation['trace_count_verified']} traces, "
        f"{validation['video_count_verified']} videos, and {validation['action_count_verified']} final 7D actions.",
        "",
        "## Gate status",
        "",
    ]
    report_lines.extend(f"- {key}: `{value}`" for key, value in gates.items())
    report_lines.extend(
        [
            "",
            "## Holdout coverage",
            "",
            f"The report contains {sum(int(row['episodes']) for row in holdout_rows)} holdout condition-episodes across "
            f"{len({row['model'] for row in holdout_rows})} model(s) and "
            f"{len({row['runtime'] for row in holdout_rows})} runtime label(s).",
            f"Peak allocated CUDA memory was {summary['gpu_memory_summary']['scored_episode_peak_cuda_memory_mib_max']:.1f} MiB during scored episodes and "
            + (
                f"{summary['gpu_memory_summary']['non_scored_warmup_peak_cuda_memory_mib_max']:.1f} MiB during non-scored cold warmup."
                if summary["gpu_memory_summary"]["non_scored_warmup_peak_cuda_memory_mib_max"]
                is not None
                else "N/A during non-scored cold warmup."
            ),
            "",
            "See `main_table.csv`, `paired_effects.csv`, `failure_taxonomy.csv`, and "
            "`latency_success_operating_points.png` for the auditable results.",
            "",
            "## Holdout main table",
            "",
            "| Model | Profile | Runtime | Success | Mean steps | GPU actions/s | Peak CUDA MiB | Model infer p50/p95 ms | Delivery p50/p95 ms | Queue age p50/p95 steps | Discard | Depletion | Fallback |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in holdout_rows:
        report_lines.append(
            "| "
            + " | ".join(
                (
                    str(row["model"]),
                    str(row["delay_profile"]),
                    str(row["runtime"]),
                    f"{row['successes']}/{row['episodes']} ({100.0 * float(row['success_rate']):.1f}%)",
                    compact(row["mean_environment_steps"]),
                    compact(row["gpu_generated_actions_per_second_p50_median"]),
                    compact(row["peak_cuda_memory_mib_max"], digits=1),
                    f"{milliseconds(row['inference_latency_p50_seconds_median'])}/{milliseconds(row['inference_latency_p95_seconds_median'])}",
                    f"{milliseconds(row['delivery_latency_p50_seconds_median'])}/{milliseconds(row['delivery_latency_p95_seconds_median'])}",
                    f"{compact(row['queue_age_p50_steps_median'])}/{compact(row['queue_age_p95_steps_median'])}",
                    compact(row["dropped_prefix_steps_sum"]),
                    compact(row["depletion_safe_hold_steps_sum"]),
                    compact(row["fallback_activations_sum"]),
                )
            )
            + " |"
        )
    report_lines.extend(
        [
            "",
            "## Paired effects versus official latest-only",
            "",
            "Positive success differences favor the estimate runtime; negative step differences are faster.",
            "",
            "| Model | Profile | Runtime | Metric | Paired n | Mean difference | 95% bootstrap CI |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in holdout_effects:
        report_lines.append(
            "| "
            + " | ".join(
                (
                    str(row["model"]),
                    str(row["delay_profile"]),
                    str(row["estimate_runtime"]),
                    str(row["metric"]),
                    str(row["paired_episode_count"]),
                    compact(row["paired_mean_difference"], digits=4 if row["metric"] == "success" else 2),
                    f"[{compact(row['ci95_low'], digits=4 if row['metric'] == 'success' else 2)}, {compact(row['ci95_high'], digits=4 if row['metric'] == 'success' else 2)}]",
                )
            )
            + " |"
        )
    report_lines.extend(
        [
            "",
            "Queue-age cells unsupported by upstream queues are blank rather than zero. Disconnect/recovery is a "
            "backend-only canary and is not treated as a fair official-baseline effect. This benchmark is LIBERO "
            "simulation evidence, not real-robot safety evidence.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")

    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary
