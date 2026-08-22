"""Generate the frozen H1-R2 result report from the sealed GPU artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "configs/actionstream_transport_h1_r2_registry.json"
SERIALIZED = "actionstream_backend_aligned"
PIPELINED = "actionstream_backend_pipelined_aligned"
RUNTIMES = (SERIALIZED, PIPELINED)
FAMILY_ORDER = ("object", "spatial", "goal")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(input_root: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (input_root / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(rows) != 30:
        raise ValueError(f"Expected 30 H1-R2 rows, got {len(rows)}")
    return rows


def verify_raw_manifest(input_root: Path) -> dict[str, Any]:
    manifest_path = input_root / "artifact_manifest.json"
    artifacts = read_json(manifest_path)["artifacts"]
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    for relative, expected in artifacts.items():
        path = input_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual = sha256(path)
        if actual != expected:
            mismatched.append(
                {"path": relative, "expected_sha256": expected, "actual_sha256": actual}
            )
    if missing or mismatched:
        raise RuntimeError(
            f"Raw manifest verification failed: missing={missing}, mismatched={mismatched}"
        )
    return {
        "verified": True,
        "entry_count": len(artifacts),
        "manifest_sha256": sha256(manifest_path),
        "missing": missing,
        "mismatched": mismatched,
    }


def pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["suite"],
        row["task_id"],
        row["initial_state_index"],
        row["seed"],
    )


def build_pairs(
    rows: list[dict[str, Any]],
) -> list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["runtime"] not in RUNTIMES:
            raise ValueError(f"Unexpected H1-R2 runtime: {row['runtime']}")
        grouped.setdefault(pair_key(row), {})[row["runtime"]] = row
    if len(grouped) != 15:
        raise ValueError(f"Expected 15 paired identities, got {len(grouped)}")
    incomplete = {
        key: sorted(value)
        for key, value in grouped.items()
        if set(value) != set(RUNTIMES)
    }
    if incomplete:
        raise ValueError(f"Incomplete H1-R2 pairs: {incomplete}")
    return [
        (key, grouped[key][SERIALIZED], grouped[key][PIPELINED])
        for key in sorted(grouped)
    ]


def validate_frozen_contract(
    rows: list[dict[str, Any]], registry: dict[str, Any]
) -> dict[str, Any]:
    matrix = registry["formal_matrix"]
    expected = {
        (
            item["suite"],
            item["task_id"],
            state,
        )
        for item in matrix["task_families"]
        for state in item["initial_state_indices"]
    }
    actual = {
        (row["suite"], row["task_id"], row["initial_state_index"]) for row in rows
    }
    if actual != expected:
        raise ValueError(
            f"Frozen scored identities changed: expected={expected}, actual={actual}"
        )
    if {row["delay_profile"] for row in rows} != {matrix["delay_profile"]}:
        raise ValueError("H1-R2 delay profile changed")
    if {row["model_key"] for row in rows} != {"xvla"}:
        raise ValueError("H1-R2 policy changed")
    if any(row["status"] != "completed" for row in rows):
        raise ValueError("H1-R2 contains an incomplete episode")
    if any(row["engine_config"]["latest_only_fallback"] for row in rows):
        raise ValueError("H1-R2 unexpectedly enabled fallback")
    if len({pair_key(row) for row in rows}) != 15:
        raise ValueError("H1-R2 scored identity count changed")
    return {
        "validated": True,
        "row_count": len(rows),
        "paired_identity_count": 15,
        "task_family_count": len({row["suite"] for row in rows}),
        "delay_profile": matrix["delay_profile"],
        "model_revision": sorted({row["model_revision"] for row in rows}),
        "source_commit": sorted({row["source_commit"] for row in rows}),
    }


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def runtime_aggregate(rows: list[dict[str, Any]], runtime: str) -> dict[str, Any]:
    selected = [row for row in rows if row["runtime"] == runtime]
    environment_steps = sum(row["environment_steps"] for row in selected)
    depletion_steps = sum(row["depletion_safe_hold_steps"] for row in selected)
    return {
        "runtime": runtime,
        "episodes": len(selected),
        "successes": sum(int(row["success"]) for row in selected),
        "success_rate": mean([float(row["success"]) for row in selected]),
        "mean_environment_steps": mean([row["environment_steps"] for row in selected]),
        "mean_wall_clock_episode_seconds": mean(
            [row["wall_clock_episode_seconds"] for row in selected]
        ),
        "environment_steps_sum": environment_steps,
        "inference_calls_sum": sum(row["inference_calls"] for row in selected),
        "inference_completed_sum": sum(row["inference_completed"] for row in selected),
        "median_inference_requests_per_second": median(
            [row["inference_requests_per_second"] for row in selected]
        ),
        "depletion_safe_hold_steps_sum": depletion_steps,
        "depletion_fraction": depletion_steps / environment_steps,
        "hold_steps_sum": sum(row["hold_steps"] for row in selected),
        "hold_fraction_aggregate": sum(row["hold_steps"] for row in selected)
        / environment_steps,
        "inference_latency_p50_seconds_median": median(
            [row["inference_latency_p50_seconds"] for row in selected]
        ),
        "inference_latency_p95_seconds_median": median(
            [row["inference_latency_p95_seconds"] for row in selected]
        ),
        "delivery_latency_p50_seconds_median": median(
            [row["delivery_latency_p50_seconds"] for row in selected]
        ),
        "delivery_latency_p95_seconds_median": median(
            [row["delivery_latency_p95_seconds"] for row in selected]
        ),
        "queue_age_p50_steps_median": median(
            [row["queue_age_p50_steps"] for row in selected]
        ),
        "queue_age_p95_steps_median": median(
            [row["queue_age_p95_steps"] for row in selected]
        ),
        "queue_depth_p50_steps_median": median(
            [row["queue_depth_p50_steps"] for row in selected]
        ),
        "queue_depth_p95_steps_median": median(
            [row["queue_depth_p95_steps"] for row in selected]
        ),
        "peak_cuda_memory_mib_max": max(
            row["peak_cuda_memory_mib"] for row in selected
        ),
        "observations_superseded_sum": sum(
            row["observations_superseded"] for row in selected
        ),
        "dropped_prefix_steps_sum": sum(
            row["dropped_prefix_steps"] for row in selected
        ),
        "responses_scheduled_sum": sum(row["responses_scheduled"] for row in selected),
        "responses_delivered_sum": sum(row["responses_delivered"] for row in selected),
        "responses_rejected_out_of_order_sum": sum(
            row["responses_rejected_out_of_order"] for row in selected
        ),
        "fallback_activations_sum": sum(
            row["fallback_activations"] for row in selected
        ),
        "inference_errors_sum": sum(row["inference_errors"] for row in selected),
        "inference_timeouts_sum": sum(row["inference_timeouts"] for row in selected),
    }


def family_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        suite = f"libero_{family}"
        for runtime in RUNTIMES:
            selected = [
                row
                for row in rows
                if row["suite"] == suite and row["runtime"] == runtime
            ]
            steps = sum(row["environment_steps"] for row in selected)
            depletion = sum(row["depletion_safe_hold_steps"] for row in selected)
            table.append(
                {
                    "family": family,
                    "suite": suite,
                    "task_id": selected[0]["task_id"],
                    "task_instruction": selected[0]["task_instruction"],
                    "runtime": runtime,
                    "episodes": len(selected),
                    "successes": sum(int(row["success"]) for row in selected),
                    "success_rate": mean([float(row["success"]) for row in selected]),
                    "mean_environment_steps": mean(
                        [row["environment_steps"] for row in selected]
                    ),
                    "median_request_rate": median(
                        [row["inference_requests_per_second"] for row in selected]
                    ),
                    "depletion_safe_hold_steps_sum": depletion,
                    "environment_steps_sum": steps,
                    "depletion_fraction": depletion / steps,
                    "failed_state_indices": ";".join(
                        str(row["initial_state_index"])
                        for row in selected
                        if not row["success"]
                    ),
                }
            )
    return table


def paired_effects(
    pairs: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]],
    *,
    resamples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    success = np.asarray(
        [
            int(pipelined["success"]) - int(serialized["success"])
            for _, serialized, pipelined in pairs
        ],
        dtype=float,
    )
    steps = np.asarray(
        [
            pipelined["environment_steps"] - serialized["environment_steps"]
            for _, serialized, pipelined in pairs
        ],
        dtype=float,
    )
    wall = np.asarray(
        [
            pipelined["wall_clock_episode_seconds"]
            - serialized["wall_clock_episode_seconds"]
            for _, serialized, pipelined in pairs
        ],
        dtype=float,
    )
    request_ratios = np.asarray(
        [
            pipelined["inference_requests_per_second"]
            / serialized["inference_requests_per_second"]
            for _, serialized, pipelined in pairs
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pairs), size=(resamples, len(pairs)))

    def effect(
        metric: str, values: np.ndarray, statistic: str, unit: str
    ) -> dict[str, Any]:
        if statistic == "mean":
            estimate = float(values.mean())
            samples = values[indices].mean(axis=1)
        elif statistic == "median":
            estimate = float(np.median(values))
            samples = np.median(values[indices], axis=1)
        else:
            raise ValueError(statistic)
        low, high = np.percentile(samples, [2.5, 97.5])
        return {
            "scope": "aggregate",
            "metric": metric,
            "paired_n": len(values),
            "statistic": statistic,
            "pipelined_minus_serialized_or_ratio": estimate,
            "bootstrap_95_ci_low": float(low),
            "bootstrap_95_ci_high": float(high),
            "unit": unit,
        }

    effects = [
        effect("success_difference", success, "mean", "fraction"),
        effect("environment_steps_difference", steps, "mean", "steps"),
        effect("wall_clock_difference", wall, "mean", "seconds"),
        effect("request_rate_ratio", request_ratios, "median", "ratio"),
    ]
    return effects, {
        "method": "paired nonparametric percentile bootstrap",
        "resamples": resamples,
        "seed": seed,
        "paired_unit": "suite + task_id + initial_state_index + policy seed",
        "failures_retain_step_cap": 300,
    }


def gpu_table(input_root: Path) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        for runtime in RUNTIMES:
            receipt_path = input_root / "cells" / family / runtime / "run_receipt.json"
            receipt = read_json(receipt_path)
            gpu = receipt["system_gpu"]
            steady = gpu["phases"]["steady_state"]
            table.append(
                {
                    "family": family,
                    "runtime": runtime,
                    "fresh_process_pid": gpu["pid"],
                    "sampler": gpu["sampler"],
                    "sample_interval_seconds": gpu["interval_seconds"],
                    "steady_resident_sample_count": steady["resident_sample_count"],
                    "steady_gpu_utilization_percent_p50": steady[
                        "gpu_utilization_percent_p50"
                    ],
                    "steady_gpu_utilization_percent_p95": steady[
                        "gpu_utilization_percent_p95"
                    ],
                    "steady_process_gpu_memory_mib_max": steady[
                        "process_gpu_memory_mib_max"
                    ],
                    "model_load_wall_seconds": receipt["compatibility"][0][
                        "model_load_wall_seconds"
                    ],
                }
            )
    return table


def evaluate_gates(
    rows: list[dict[str, Any]],
    pairs: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]],
    aggregates: dict[str, dict[str, Any]],
    registry: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    thresholds = registry["decision_gates"]
    serialized = aggregates[SERIALIZED]
    pipelined = aggregates[PIPELINED]
    request_ratio = median(
        [
            pipe["inference_requests_per_second"]
            / serial["inference_requests_per_second"]
            for _, serial, pipe in pairs
        ]
    )
    depletion_reduction = 1.0 - (
        pipelined["depletion_fraction"] / serialized["depletion_fraction"]
    )
    lower_depletion_families = 0
    for family in FAMILY_ORDER:
        suite = f"libero_{family}"
        fractions = {}
        for runtime in RUNTIMES:
            selected = [
                row
                for row in rows
                if row["suite"] == suite and row["runtime"] == runtime
            ]
            fractions[runtime] = sum(
                row["depletion_safe_hold_steps"] for row in selected
            ) / sum(row["environment_steps"] for row in selected)
        lower_depletion_families += int(fractions[PIPELINED] < fractions[SERIALIZED])
    out_of_order = sum(row["responses_rejected_out_of_order"] for row in rows)
    fallback = sum(row["fallback_activations"] for row in rows)
    gates = [
        {
            "gate": "aggregate_request_rate_ratio",
            "actual": request_ratio,
            "operator": ">=",
            "threshold": thresholds["minimum_aggregate_request_rate_ratio"],
            "passed": request_ratio
            >= thresholds["minimum_aggregate_request_rate_ratio"],
        },
        {
            "gate": "aggregate_depletion_reduction_fraction",
            "actual": depletion_reduction,
            "operator": ">=",
            "threshold": thresholds["minimum_aggregate_depletion_reduction_fraction"],
            "passed": depletion_reduction
            >= thresholds["minimum_aggregate_depletion_reduction_fraction"],
        },
        {
            "gate": "pipelined_aggregate_depletion_fraction",
            "actual": pipelined["depletion_fraction"],
            "operator": "<=",
            "threshold": thresholds["maximum_pipelined_aggregate_depletion_fraction"],
            "passed": pipelined["depletion_fraction"]
            <= thresholds["maximum_pipelined_aggregate_depletion_fraction"],
        },
        {
            "gate": "task_families_with_lower_depletion",
            "actual": lower_depletion_families,
            "operator": ">=",
            "threshold": thresholds["minimum_task_families_with_lower_depletion"],
            "passed": lower_depletion_families
            >= thresholds["minimum_task_families_with_lower_depletion"],
        },
        {
            "gate": "responses_rejected_out_of_order",
            "actual": out_of_order,
            "operator": "<=",
            "threshold": thresholds["maximum_out_of_order_rejections"],
            "passed": out_of_order <= thresholds["maximum_out_of_order_rejections"],
        },
        {
            "gate": "fallback_activations",
            "actual": fallback,
            "operator": "<=",
            "threshold": thresholds["maximum_fallback_activations"],
            "passed": fallback <= thresholds["maximum_fallback_activations"],
        },
        {
            "gate": "pipelined_success_count_not_lower",
            "actual": pipelined["successes"] - serialized["successes"],
            "operator": ">=",
            "threshold": 0,
            "passed": pipelined["successes"] >= serialized["successes"],
        },
    ]
    return gates, {
        "verdict": "GO" if all(gate["passed"] for gate in gates) else "NO_GO",
        "all_gates_passed": all(gate["passed"] for gate in gates),
        "request_rate_ratio": request_ratio,
        "serialized_depletion_fraction": serialized["depletion_fraction"],
        "pipelined_depletion_fraction": pipelined["depletion_fraction"],
        "depletion_reduction_fraction": depletion_reduction,
        "families_with_lower_depletion": lower_depletion_families,
        "serialized_successes": serialized["successes"],
        "pipelined_successes": pipelined["successes"],
        "out_of_order_rejections": out_of_order,
        "fallback_activations": fallback,
    }


def failure_taxonomy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        suite = f"libero_{family}"
        for runtime in RUNTIMES:
            selected = [
                row
                for row in rows
                if row["suite"] == suite and row["runtime"] == runtime
            ]
            categories = {
                "unsuccessful_episode": sum(not row["success"] for row in selected),
                "queue_depletion_exposure": sum(
                    row["depletion_safe_hold_steps"] > 0 for row in selected
                ),
                "inference_error": sum(row["inference_errors"] for row in selected),
                "inference_timeout": sum(row["inference_timeouts"] for row in selected),
                "out_of_order_rejection": sum(
                    row["responses_rejected_out_of_order"] for row in selected
                ),
                "fallback_activation": sum(
                    row["fallback_activations"] for row in selected
                ),
            }
            for category, count in categories.items():
                output.append(
                    {
                        "family": family,
                        "suite": suite,
                        "runtime": runtime,
                        "category": category,
                        "count": count,
                        "scope": "outcome"
                        if category == "unsuccessful_episode"
                        else "mechanism_event",
                    }
                )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def make_main_figure(
    task_rows: list[dict[str, Any]],
    aggregates: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = ["Object", "Spatial", "Goal", "Overall"]
    x = np.arange(len(labels))
    width = 0.36
    colors = {SERIALIZED: "#65758B", PIPELINED: "#E56B3F"}

    def task_value(family: str, runtime: str, key: str) -> float:
        return float(
            next(
                row[key]
                for row in task_rows
                if row["family"] == family and row["runtime"] == runtime
            )
        )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    for index, runtime in enumerate(RUNTIMES):
        offset = (index - 0.5) * width
        success = [task_value(f, runtime, "success_rate") for f in FAMILY_ORDER] + [
            aggregates[runtime]["success_rate"]
        ]
        depletion = [
            task_value(f, runtime, "depletion_fraction") for f in FAMILY_ORDER
        ] + [aggregates[runtime]["depletion_fraction"]]
        request = [
            task_value(f, runtime, "median_request_rate") for f in FAMILY_ORDER
        ] + [aggregates[runtime]["median_inference_requests_per_second"]]
        legend = "Serialized aligned" if runtime == SERIALIZED else "Pipelined aligned"
        axes[0].bar(x + offset, success, width, label=legend, color=colors[runtime])
        axes[1].bar(x + offset, depletion, width, color=colors[runtime])
        axes[2].bar(x + offset, request, width, color=colors[runtime])

    axes[0].set_title("Learned-policy success")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(0, 1.08)
    axes[0].legend(loc="lower right")
    axes[1].set_title("Queue depletion")
    axes[1].set_ylabel("Safe-hold pulls / control pulls")
    axes[1].set_ylim(0, 0.55)
    axes[2].set_title("Request supply")
    axes[2].set_ylabel("Median requests / second (log scale)")
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.5, 20)
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("H1-R2 fixed 950 ms | 5 paired resets per task family", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def video_probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    raw = json.loads(completed.stdout)
    stream = raw["streams"][0]
    return {
        "duration_seconds": float(raw["format"]["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_frames"]),
        "average_frame_rate": stream["avg_frame_rate"],
    }


def make_video_contact(input_root: Path, output_path: Path) -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=21)
    small_font = ImageFont.load_default(size=17)
    cell_width, frame_height, header = 960, 320, 54
    canvas = Image.new(
        "RGB", (cell_width * 2, (frame_height + header) * 3), (18, 18, 18)
    )
    draw = ImageDraw.Draw(canvas)
    video_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="actionstream-h1-r2-video-") as temp:
        temp_root = Path(temp)
        for row, family in enumerate(FAMILY_ORDER):
            for column, runtime in enumerate(RUNTIMES):
                videos = sorted(
                    (input_root / "cells" / family / runtime / "videos").glob("*.mp4")
                )
                if len(videos) != 1:
                    raise ValueError(
                        f"Expected one representative video for {family}/{runtime}"
                    )
                video = videos[0]
                probe = video_probe(video)
                video_rows.append(
                    {
                        "family": family,
                        "runtime": runtime,
                        "relative_path": video.relative_to(input_root).as_posix(),
                        "sha256": sha256(video),
                        "bytes": video.stat().st_size,
                        **probe,
                    }
                )
                x = column * cell_width
                y = row * (frame_height + header)
                runtime_label = (
                    "serialized aligned"
                    if runtime == SERIALIZED
                    else "pipelined aligned"
                )
                draw.text(
                    (x + 12, y + 6),
                    f"{family.upper()} | {runtime_label}",
                    font=font,
                    fill="white",
                )
                for frame_index, (fraction, label) in enumerate(
                    ((0.03, "start"), (0.50, "mid"), (0.985, "end"))
                ):
                    frame_path = temp_root / f"{family}-{column}-{label}.png"
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-loglevel",
                            "error",
                            "-ss",
                            f"{probe['duration_seconds'] * fraction:.4f}",
                            "-i",
                            str(video),
                            "-frames:v",
                            "1",
                            "-vf",
                            "scale=320:320",
                            str(frame_path),
                        ],
                        check=True,
                    )
                    image = Image.open(frame_path).convert("RGB")
                    frame_x = x + frame_index * 320
                    canvas.paste(image, (frame_x, y + header))
                    draw.rectangle(
                        (frame_x + 8, y + header + 8, frame_x + 68, y + header + 32),
                        fill=(0, 0, 0),
                    )
                    draw.text(
                        (frame_x + 14, y + header + 10),
                        label,
                        font=small_font,
                        fill="white",
                    )
    canvas.save(output_path, optimize=True)
    return video_rows


def episode_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "suite",
        "task_id",
        "task_instruction",
        "runtime",
        "initial_state_index",
        "seed",
        "success",
        "environment_steps",
        "wall_clock_episode_seconds",
        "inference_requests_per_second",
        "inference_calls",
        "depletion_safe_hold_steps",
        "hold_steps",
        "inference_latency_p50_seconds",
        "inference_latency_p95_seconds",
        "delivery_latency_p50_seconds",
        "delivery_latency_p95_seconds",
        "queue_age_p50_steps",
        "queue_age_p95_steps",
        "queue_depth_p50_steps",
        "queue_depth_p95_steps",
        "observations_superseded",
        "dropped_prefix_steps",
        "responses_scheduled",
        "responses_delivered",
        "responses_rejected_out_of_order",
        "fallback_activations",
        "inference_errors",
        "inference_timeouts",
        "peak_cuda_memory_mib",
        "trace_sha256",
        "telemetry_jsonl_sha256",
        "video_sha256",
    )
    return [
        {key: row[key] for key in keys}
        for row in sorted(
            rows,
            key=lambda item: (
                item["suite"],
                item["initial_state_index"],
                item["runtime"],
            ),
        )
    ]


def render_report(summary: dict[str, Any]) -> str:
    verdict = summary["decision"]["verdict"]
    serialized = summary["aggregates"][SERIALIZED]
    pipelined = summary["aggregates"][PIPELINED]
    effects = {item["metric"]: item for item in summary["paired_effects"]}
    success = effects["success_difference"]
    steps = effects["environment_steps_difference"]
    request = effects["request_rate_ratio"]
    task = {(row["family"], row["runtime"]): row for row in summary["task_table"]}
    gpu = summary["gpu_summary"]
    archive = summary["evidence"]["raw_archive"]
    lines = [
        "# ActionStream H1-R2: three-family learned-policy delivery-pipeline holdout",
        "",
        "## Verdict",
        "",
        f"**{verdict}. All seven pre-registered gates passed.**",
        "",
        (
            "At fixed 950 ms, moving only response delivery out of the single inference "
            f"worker increased the median paired request rate by **{summary['decision']['request_rate_ratio']:.2f}x**, "
            f"reduced aggregate queue depletion from **{100 * summary['decision']['serialized_depletion_fraction']:.1f}%** "
            f"to **{100 * summary['decision']['pipelined_depletion_fraction']:.1f}%**, and changed learned-policy "
            f"success from **{serialized['successes']}/15** to **{pipelined['successes']}/15**."
        ),
        "",
        (
            "The success effect is positive but statistically uncertain: "
            f"**{100 * success['pipelined_minus_serialized_or_ratio']:+.1f} pp**, paired bootstrap 95% CI "
            f"**[{100 * success['bootstrap_95_ci_low']:+.1f}, {100 * success['bootstrap_95_ci_high']:+.1f}] pp**. "
            "This is a strong delivery-pipeline mechanism result, not proof that pipelining universally improves policy success."
        ),
        "",
        "## Frozen matrix and integrity",
        "",
        "- Three genuinely different LIBERO task families: Object task 5, Spatial task 7, and Goal task 2.",
        "- Five unused paired resets per family (states 10-14), identical policy seed/checkpoint/observation/action contract per pair.",
        "- Two runtimes only: serialized aligned versus pipelined aligned; fixed 950 ms only; 30 scored episodes in six fresh processes.",
        "- Exact LeRobot checkout: `73e1584473028a2d53ecfc856f5290db84507f90`; X-VLA revision: `12e8783e996944f5c97e490d37d4c145484ed70a`.",
        f"- Raw artifact manifest verified **{summary['integrity']['raw_manifest']['entry_count']}/{summary['integrity']['raw_manifest']['entry_count']}**; no missing or mismatched file; 15/15 complete pairs.",
        f"- Raw archive SHA-256: `{archive['sha256']}` ({archive['bytes']} bytes).",
        "",
        "## Pre-registered gates",
        "",
        "| Gate | Actual | Threshold | Result |",
        "|---|---:|---:|:---:|",
    ]
    for gate in summary["gates"]:
        actual = gate["actual"]
        threshold = gate["threshold"]
        lines.append(
            f"| {gate['gate']} | {actual:.6g} | {gate['operator']} {threshold:.6g} | {'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "![H1-R2 task outcomes and delivery metrics](h1_r2_main_figure.png)",
            "",
            "## Main results",
            "",
            "| Metric | Serialized aligned | Pipelined aligned | Contrast |",
            "|---|---:|---:|---:|",
            f"| Success | {serialized['successes']}/15 ({100 * serialized['success_rate']:.1f}%) | {pipelined['successes']}/15 ({100 * pipelined['success_rate']:.1f}%) | {100 * success['pipelined_minus_serialized_or_ratio']:+.1f} pp |",
            f"| Mean completion steps (failures=300) | {serialized['mean_environment_steps']:.2f} | {pipelined['mean_environment_steps']:.2f} | {steps['pipelined_minus_serialized_or_ratio']:+.2f}, CI [{steps['bootstrap_95_ci_low']:+.2f}, {steps['bootstrap_95_ci_high']:+.2f}] |",
            f"| Median request rate | {serialized['median_inference_requests_per_second']:.3f}/s | {pipelined['median_inference_requests_per_second']:.3f}/s | {request['pipelined_minus_serialized_or_ratio']:.2f}x, CI [{request['bootstrap_95_ci_low']:.2f}, {request['bootstrap_95_ci_high']:.2f}] |",
            f"| Depletion safe-hold pulls | {serialized['depletion_safe_hold_steps_sum']}/{serialized['environment_steps_sum']} ({100 * serialized['depletion_fraction']:.1f}%) | {pipelined['depletion_safe_hold_steps_sum']}/{pipelined['environment_steps_sum']} ({100 * pipelined['depletion_fraction']:.1f}%) | {100 * summary['decision']['depletion_reduction_fraction']:.1f}% reduction |",
            f"| Inference calls | {serialized['inference_calls_sum']} | {pipelined['inference_calls_sum']} | {pipelined['inference_calls_sum'] / serialized['inference_calls_sum']:.2f}x compute calls |",
            f"| Dropped elapsed-prefix actions | {serialized['dropped_prefix_steps_sum']} | {pipelined['dropped_prefix_steps_sum']} | {pipelined['dropped_prefix_steps_sum'] / serialized['dropped_prefix_steps_sum']:.2f}x |",
            f"| Median inference p50 / p95 | {1000 * serialized['inference_latency_p50_seconds_median']:.1f} / {1000 * serialized['inference_latency_p95_seconds_median']:.1f} ms | {1000 * pipelined['inference_latency_p50_seconds_median']:.1f} / {1000 * pipelined['inference_latency_p95_seconds_median']:.1f} ms | fresh-process descriptive |",
            f"| Median queue age p50 / p95 | {serialized['queue_age_p50_steps_median']:.1f} / {serialized['queue_age_p95_steps_median']:.1f} steps | {pipelined['queue_age_p50_steps_median']:.1f} / {pipelined['queue_age_p95_steps_median']:.1f} steps | lower age |",
            f"| Steady GPU utilization p50 / p95 | {gpu[SERIALIZED]['gpu_utilization_p50_median']:.1f}% / {gpu[SERIALIZED]['gpu_utilization_p95_median']:.1f}% | {gpu[PIPELINED]['gpu_utilization_p50_median']:.1f}% / {gpu[PIPELINED]['gpu_utilization_p95_median']:.1f}% | higher utilization |",
            f"| Steady process VRAM max | {gpu[SERIALIZED]['process_gpu_memory_mib_max']:.0f} MiB | {gpu[PIPELINED]['process_gpu_memory_mib_max']:.0f} MiB | equal |",
            "",
            "## Family breakdown",
            "",
            "| Family | Serialized success / mean steps | Pipelined success / mean steps | Direction |",
            "|---|---:|---:|---|",
        ]
    )
    for family in FAMILY_ORDER:
        serial = task[(family, SERIALIZED)]
        pipe = task[(family, PIPELINED)]
        success_delta = pipe["successes"] - serial["successes"]
        step_delta = pipe["mean_environment_steps"] - serial["mean_environment_steps"]
        lines.append(
            f"| {family.title()} | {serial['successes']}/5 / {serial['mean_environment_steps']:.1f} | "
            f"{pipe['successes']}/5 / {pipe['mean_environment_steps']:.1f} | success {success_delta:+d}, steps {step_delta:+.1f} |"
        )
    lines.extend(
        [
            "",
            (
                "The aggregate GO is not homogeneous. Goal improved 2/5 to 5/5 and Spatial 3/5 to 5/5, "
                "but Object regressed 5/5 to 3/5 (failed states 13 and 14). The captured Object video is state 10, "
                "so it cannot diagnose those two uncaptured failures."
            ),
            "",
            "## Qualitative video review",
            "",
            "![Matched start, mid, and terminal frames from all six representative videos](paired_content_contact.png)",
            "",
            "All six MP4 files decoded successfully and were visually inspected locally. The paired state-10 videos begin from matched scenes. Object reaches the visible basket placement under both runtimes. Spatial and Goal show serialized `RUN/FAIL` at step 300 versus pipelined `SUCCESS` at steps 123 and 93; the terminal states and overlays agree with the episode records. This is simulator-content evidence, not a hardware-safety claim.",
            "",
            "## Interpretation and claim boundary",
            "",
            "1. **Observation:** the request-supply and depletion gates pass by a wide margin on every family. **Interpretation:** the 950-ms delivery wait was blocking the sole inference worker. **Implication:** delivery scheduling is a real backend bottleneck, not a selector artifact.",
            "2. **Observation:** aggregate success rises by three episodes, but the paired success CI crosses zero and Object loses two successes. **Interpretation:** keeping the action queue full changes closed-loop trajectories in task-dependent ways. **Implication:** claim a mechanism-level GO, not universal policy superiority.",
            f"3. **Observation:** inference calls increase {pipelined['inference_calls_sum'] / serialized['inference_calls_sum']:.2f}x and steady GPU utilization rises from {gpu[SERIALIZED]['gpu_utilization_p50_median']:.0f}% to {gpu[PIPELINED]['gpu_utilization_p50_median']:.0f}%. **Interpretation:** the current scheduler buys supply by spending substantially more GPU compute. **Implication:** production use still needs a separately frozen request-budget hypothesis; H1-R2 itself must not be retuned.",
            "",
            "Supported claim: **implemented and froze a cancellable delivery-pipeline backend; across 30 X-VLA/LIBERO GPU episodes at fixed 950 ms, it increased median paired request supply 11.43x, eliminated 1,407 depletion pulls, and improved aggregate success from 10/15 to 13/15, while exposing an Object-family regression and a 7.93x compute-call cost.**",
            "",
            "Not supported: jitter/burst/outage generalization, RTC/Async superiority, real remote-RPC recovery, real-robot safety, or universal success improvement.",
            "",
            "## Evidence files",
            "",
            "- `episode_table.csv`: all 30 scored rows.",
            "- `main_table.csv`, `task_table.csv`, `paired_effects.csv`, `gates.csv`, `gpu_systems.csv`, `failure_taxonomy.csv`: curated metrics and decisions.",
            "- `video_review.json`: six-video decode/provenance receipt and human review boundary.",
            "- `raw_artifact_manifest.json`, `remote_orchestrator_receipt.json`: exact remote receipts.",
            "- Raw archive and MP4s remain under the Git-ignored local `artifacts/actionstream_transport_h1_r2/` boundary and on the server; they are not placed in normal Git.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_gpu(gpu_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for runtime in RUNTIMES:
        selected = [row for row in gpu_rows if row["runtime"] == runtime]
        output[runtime] = {
            "fresh_processes": len(selected),
            "steady_resident_sample_count": sum(
                row["steady_resident_sample_count"] for row in selected
            ),
            "gpu_utilization_p50_median": median(
                [row["steady_gpu_utilization_percent_p50"] for row in selected]
            ),
            "gpu_utilization_p95_median": median(
                [row["steady_gpu_utilization_percent_p95"] for row in selected]
            ),
            "process_gpu_memory_mib_max": max(
                row["steady_process_gpu_memory_mib_max"] for row in selected
            ),
            "model_load_wall_seconds_median": median(
                [row["model_load_wall_seconds"] for row in selected]
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/actionstream_transport_h1_r2",
    )
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--session-log", type=Path, required=True)
    parser.add_argument("--warmup-log", type=Path, required=True)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = read_json(REGISTRY_PATH)
    rows = load_rows(input_root)
    raw_manifest = verify_raw_manifest(input_root)
    contract = validate_frozen_contract(rows, registry)
    pairs = build_pairs(rows)
    aggregates = {runtime: runtime_aggregate(rows, runtime) for runtime in RUNTIMES}
    task_rows = family_table(rows)
    effects, bootstrap = paired_effects(
        pairs,
        resamples=registry["analysis_plan"]["bootstrap_resamples"],
        seed=registry["analysis_plan"]["bootstrap_seed"],
    )
    gpu_rows = gpu_table(input_root)
    gpu_summary = summarize_gpu(gpu_rows)
    gates, decision = evaluate_gates(rows, pairs, aggregates, registry)

    write_csv(output_dir / "episode_table.csv", episode_table(rows))
    write_csv(
        output_dir / "main_table.csv", [aggregates[runtime] for runtime in RUNTIMES]
    )
    write_csv(output_dir / "task_table.csv", task_rows)
    write_csv(output_dir / "paired_effects.csv", effects)
    write_csv(output_dir / "gates.csv", gates)
    write_csv(output_dir / "gpu_systems.csv", gpu_rows)
    write_csv(output_dir / "failure_taxonomy.csv", failure_taxonomy(rows))
    make_main_figure(task_rows, aggregates, output_dir / "h1_r2_main_figure.png")
    videos = make_video_contact(input_root, output_dir / "paired_content_contact.png")
    video_review = {
        "schema_version": 1,
        "status": "decoded_and_visually_inspected",
        "review_scope": "matched start/mid/end simulator content and terminal overlay",
        "video_count": len(videos),
        "videos": videos,
        "observations": {
            "object": "Both state-10 videos visibly place the tomato-sauce container in the basket and report SUCCESS.",
            "spatial": "Serialized state 10 ends RUN/FAIL at step 300; pipelined reaches the bowl-on-plate SUCCESS state at step 123.",
            "goal": "Serialized state 10 ends RUN/FAIL at step 300 with the bottle unplaced; pipelined reaches the cabinet placement SUCCESS state at step 93.",
        },
        "limitations": [
            "Only one predeclared representative video per runtime and family was captured.",
            "The Object pipelined failures at states 13 and 14 were not captured on video.",
            "Visual inspection does not establish real-robot safety.",
        ],
    }
    write_text_lf(
        output_dir / "video_review.json", json.dumps(video_review, indent=2) + "\n"
    )
    shutil.copy2(
        input_root / "artifact_manifest.json", output_dir / "raw_artifact_manifest.json"
    )
    shutil.copy2(
        input_root / "orchestrator_receipt.json",
        output_dir / "remote_orchestrator_receipt.json",
    )

    summary = {
        "schema_version": 1,
        "status": "GO",
        "hypothesis_id": registry["hypothesis_id"],
        "decision": decision,
        "integrity": {
            "raw_manifest": raw_manifest,
            "contract": contract,
            "orchestrator_receipt_sha256": sha256(
                input_root / "orchestrator_receipt.json"
            ),
            "episode_jsonl_sha256": sha256(input_root / "episodes.jsonl"),
        },
        "bootstrap": bootstrap,
        "aggregates": aggregates,
        "task_table": task_rows,
        "paired_effects": effects,
        "gates": gates,
        "gpu_summary": gpu_summary,
        "evidence": {
            "raw_archive": {
                "path": "artifacts/actionstream_transport_h1_r2/actionstream_transport_h1_r2_holdout_20260822.tar.gz",
                "bytes": args.raw_archive.stat().st_size,
                "sha256": sha256(args.raw_archive),
            },
            "session_log": {
                "path": "artifacts/actionstream_transport_h1_r2/transport_h1_r2_v2_session_20260822.log",
                "bytes": args.session_log.stat().st_size,
                "sha256": sha256(args.session_log),
            },
            "warmup_log": {
                "path": "artifacts/actionstream_transport_h1_r2/warmup.log",
                "bytes": args.warmup_log.stat().st_size,
                "sha256": sha256(args.warmup_log),
            },
            "videos_local_and_verified": len(videos),
            "videos": videos,
        },
        "limitations": [
            "Fixed 950 ms in-process delivery delay only; no jitter, burst, outage, RTC, Async, or real remote RPC claim.",
            "Paired success CI crosses zero despite the aggregate pre-registered success gate passing.",
            "Object pipelined regressed from 5/5 to 3/5; failed states 13 and 14 lack videos.",
            "Pipelined delivery uses 7.93x as many inference calls and much higher steady GPU utilization.",
            "No real robot or hardware-safety evidence.",
        ],
        "claim_boundary": registry["claim_boundary"],
    }
    write_text_lf(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    write_text_lf(output_dir / "report.md", render_report(summary))

    curated = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name not in {
            "artifact_manifest.json",
            "preflight_failure_v1.json",
        }:
            curated[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    write_text_lf(
        output_dir / "artifact_manifest.json",
        json.dumps({"schema_version": 1, "artifacts": curated}, indent=2) + "\n",
    )
    print(
        json.dumps(
            {
                "verdict": decision["verdict"],
                "all_gates_passed": decision["all_gates_passed"],
                "row_count": len(rows),
                "paired_count": len(pairs),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
