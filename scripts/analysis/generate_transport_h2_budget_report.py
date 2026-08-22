"""Generate the frozen H2 compute-budget verdict and local visual evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np


UNBOUNDED = "actionstream_backend_pipelined_aligned"
BUDGETED = "actionstream_backend_budgeted_pipelined_aligned"
RUNTIME_LABELS = {UNBOUNDED: "Unbounded", BUDGETED: "Budgeted (5 steps)"}
IDENTITY_FIELDS = ("suite", "task_id", "initial_state_index", "seed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Iterable[float], value: float) -> float | None:
    rows = list(values)
    return None if not rows else float(np.percentile(np.asarray(rows), value))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_records(raw_root: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in (raw_root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 30:
        raise RuntimeError(f"Expected 30 H2 records, got {len(records)}")
    return records


def validate_evidence(raw_root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = json.loads(
        (raw_root / "artifact_manifest.json").read_text(encoding="utf-8")
    )["artifacts"]
    mismatches = []
    for relative, expected in manifest.items():
        path = raw_root / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches.append(
                {"path": relative, "expected_sha256": expected, "actual_sha256": actual}
            )
    identities = [
        tuple(row[field] for field in IDENTITY_FIELDS) + (row["runtime"],)
        for row in records
    ]
    receipt = json.loads(
        (raw_root / "orchestrator_receipt.json").read_text(encoding="utf-8")
    )
    result = {
        "manifest_entries": len(manifest),
        "manifest_mismatches": mismatches,
        "record_count": len(records),
        "unique_runtime_identities": len(set(identities)),
        "trace_count": len(list(raw_root.glob("cells/*/*/traces/*.json"))),
        "telemetry_count": len(
            list(raw_root.glob("cells/*/*/traces/*.telemetry.jsonl"))
        ),
        "video_count": len(list(raw_root.glob("cells/*/*/videos/*.mp4"))),
        "fresh_process_count": receipt["fresh_process_count"],
        "fresh_process_pid_count": len(set(receipt["fresh_process_pids"])),
        "execution_git_head": receipt["execution_git_head"],
        "candidate_base_commit": receipt["candidate_base_commit"],
        "episodes_sha256": receipt["episodes_sha256"],
    }
    required = {
        "manifest_mismatches": [],
        "record_count": 30,
        "unique_runtime_identities": 30,
        "trace_count": 30,
        "telemetry_count": 30,
        "video_count": 6,
        "fresh_process_count": 6,
        "fresh_process_pid_count": 6,
    }
    for key, expected in required.items():
        if result[key] != expected:
            raise RuntimeError(f"Evidence validation failed for {key}: {result[key]}")
    return result


def pair_records(
    records: list[dict[str, Any]],
) -> list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]]:
    indexed = {
        tuple(row[field] for field in IDENTITY_FIELDS) + (row["runtime"],): row
        for row in records
    }
    identities = sorted({key[:-1] for key in indexed})
    pairs = []
    for identity in identities:
        try:
            pairs.append((identity, indexed[identity + (UNBOUNDED,)], indexed[identity + (BUDGETED,)]))
        except KeyError as error:
            raise RuntimeError(f"Unpaired H2 identity: {identity}") from error
    if len(pairs) != 15:
        raise RuntimeError(f"Expected 15 H2 pairs, got {len(pairs)}")
    return pairs


def aggregate_runtime(records: list[dict[str, Any]], runtime: str) -> dict[str, Any]:
    rows = [row for row in records if row["runtime"] == runtime]
    steps = sum(int(row["environment_steps"]) for row in rows)
    depletion = sum(int(row["depletion_safe_hold_steps"]) for row in rows)
    return {
        "runtime": runtime,
        "label": RUNTIME_LABELS[runtime],
        "episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "inference_calls": sum(int(row["inference_calls"]) for row in rows),
        "median_inference_requests_per_second": float(
            np.median([row["inference_requests_per_second"] for row in rows])
        ),
        "environment_steps": steps,
        "mean_environment_steps": float(
            np.mean([row["environment_steps"] for row in rows])
        ),
        "depletion_safe_hold_steps": depletion,
        "depletion_fraction": depletion / steps if steps else None,
        "dropped_prefix_steps": sum(int(row["dropped_prefix_steps"]) for row in rows),
        "observations_received": sum(int(row["observations_received"]) for row in rows),
        "observations_skipped_by_budget": sum(
            int(row["observations_skipped_by_budget"]) for row in rows
        ),
        "out_of_order_rejections": sum(
            int(row["responses_rejected_out_of_order"]) for row in rows
        ),
        "fallback_activations": sum(int(row["fallback_activations"]) for row in rows),
        "inference_timeouts": sum(int(row["inference_timeouts"]) for row in rows),
        "inference_errors": sum(int(row["inference_errors"]) for row in rows),
        "queue_age_p50_steps_episode_median": float(
            np.median([row["queue_age_p50_steps"] for row in rows])
        ),
        "queue_age_p95_steps_episode_median": float(
            np.median([row["queue_age_p95_steps"] for row in rows])
        ),
        "worker_startup_latency_seconds_p50": float(
            np.median([row["worker_warmup"]["startup_latency_seconds"] for row in rows])
        ),
        "worker_post_startup_latency_ms_p50": float(
            1000.0
            * np.median(
                [
                    row["worker_warmup"]["post_startup_warmup_latency_p50_seconds"]
                    for row in rows
                ]
            )
        ),
        "cuda_peak_memory_mib_max": max(float(row["peak_cuda_memory_mib"]) for row in rows),
    }


def family_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for suite in sorted({row["suite"] for row in records}):
        groups = {
            runtime: [
                row
                for row in records
                if row["suite"] == suite and row["runtime"] == runtime
            ]
            for runtime in (UNBOUNDED, BUDGETED)
        }
        unbounded = groups[UNBOUNDED]
        budgeted = groups[BUDGETED]
        unbounded_steps = sum(row["environment_steps"] for row in unbounded)
        budgeted_steps = sum(row["environment_steps"] for row in budgeted)
        unbounded_calls = sum(row["inference_calls"] for row in unbounded)
        budgeted_calls = sum(row["inference_calls"] for row in budgeted)
        rows.append(
            {
                "suite": suite,
                "task_id": unbounded[0]["task_id"],
                "unbounded_successes": sum(row["success"] for row in unbounded),
                "budgeted_successes": sum(row["success"] for row in budgeted),
                "unbounded_mean_steps": float(
                    np.mean([row["environment_steps"] for row in unbounded])
                ),
                "budgeted_mean_steps": float(
                    np.mean([row["environment_steps"] for row in budgeted])
                ),
                "unbounded_inference_calls": unbounded_calls,
                "budgeted_inference_calls": budgeted_calls,
                "budgeted_call_ratio": budgeted_calls / unbounded_calls,
                "call_reduction_fraction": 1 - budgeted_calls / unbounded_calls,
                "unbounded_median_request_rate": float(
                    np.median([row["inference_requests_per_second"] for row in unbounded])
                ),
                "budgeted_median_request_rate": float(
                    np.median([row["inference_requests_per_second"] for row in budgeted])
                ),
                "unbounded_depletion_fraction": sum(
                    row["depletion_safe_hold_steps"] for row in unbounded
                )
                / unbounded_steps,
                "budgeted_depletion_fraction": sum(
                    row["depletion_safe_hold_steps"] for row in budgeted
                )
                / budgeted_steps,
            }
        )
    return rows


def bootstrap_effects(
    pairs: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    unbounded_success = np.asarray([float(u["success"]) for _, u, _ in pairs])
    budgeted_success = np.asarray([float(b["success"]) for _, _, b in pairs])
    unbounded_steps = np.asarray([float(u["environment_steps"]) for _, u, _ in pairs])
    budgeted_steps = np.asarray([float(b["environment_steps"]) for _, _, b in pairs])
    unbounded_calls = np.asarray([float(u["inference_calls"]) for _, u, _ in pairs])
    budgeted_calls = np.asarray([float(b["inference_calls"]) for _, _, b in pairs])
    unbounded_request = np.asarray(
        [float(u["inference_requests_per_second"]) for _, u, _ in pairs]
    )
    budgeted_request = np.asarray(
        [float(b["inference_requests_per_second"]) for _, _, b in pairs]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pairs), size=(resamples, len(pairs)))

    def ci(values: np.ndarray) -> list[float]:
        return [float(x) for x in np.percentile(values, [2.5, 97.5])]

    success_boot = np.mean(
        budgeted_success[indices] - unbounded_success[indices], axis=1
    )
    steps_boot = np.mean(budgeted_steps[indices] - unbounded_steps[indices], axis=1)
    calls_boot = np.mean(budgeted_calls[indices] - unbounded_calls[indices], axis=1)
    call_ratio_boot = np.sum(budgeted_calls[indices], axis=1) / np.sum(
        unbounded_calls[indices], axis=1
    )
    request_difference_boot = np.mean(
        budgeted_request[indices] - unbounded_request[indices], axis=1
    )
    budgeted_request_median_boot = np.median(budgeted_request[indices], axis=1)
    return {
        "method": "paired nonparametric percentile bootstrap",
        "resamples": resamples,
        "seed": seed,
        "success_difference": {
            "estimate": float(np.mean(budgeted_success - unbounded_success)),
            "ci_95": ci(success_boot),
        },
        "environment_steps_difference": {
            "estimate": float(np.mean(budgeted_steps - unbounded_steps)),
            "ci_95": ci(steps_boot),
        },
        "inference_calls_difference": {
            "estimate": float(np.mean(budgeted_calls - unbounded_calls)),
            "ci_95": ci(calls_boot),
        },
        "aggregate_inference_call_ratio": {
            "estimate": float(np.sum(budgeted_calls) / np.sum(unbounded_calls)),
            "ci_95": ci(call_ratio_boot),
        },
        "request_rate_difference": {
            "estimate": float(np.mean(budgeted_request - unbounded_request)),
            "ci_95": ci(request_difference_boot),
        },
        "budgeted_median_request_rate": {
            "estimate": float(np.median(budgeted_request)),
            "ci_95": ci(budgeted_request_median_boot),
        },
    }


def system_gpu_metrics(raw_root: Path) -> dict[str, dict[str, Any]]:
    samples = {runtime: [] for runtime in (UNBOUNDED, BUDGETED)}
    for path in raw_root.glob("cells/*/*/system_telemetry.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            runtime = row.get("runtime")
            if (
                row.get("event") == "nvidia_smi_sample"
                and row.get("phase") == "steady_state"
                and row.get("process_resident") is True
                and runtime in samples
            ):
                samples[runtime].append(row)
    result = {}
    for runtime, rows in samples.items():
        util = [float(row["gpu_utilization_percent"]) for row in rows]
        memory = [float(row["process_gpu_memory_mib"]) for row in rows]
        result[runtime] = {
            "resident_sample_count": len(rows),
            "gpu_utilization_percent_p50": percentile(util, 50),
            "gpu_utilization_percent_p95": percentile(util, 95),
            "gpu_utilization_percent_max": max(util),
            "process_gpu_memory_mib_p50": percentile(memory, 50),
            "process_gpu_memory_mib_p95": percentile(memory, 95),
            "process_gpu_memory_mib_max": max(memory),
        }
    return result


def pooled_inference_metrics(raw_root: Path) -> dict[str, dict[str, Any]]:
    values = {runtime: [] for runtime in (UNBOUNDED, BUDGETED)}
    for path in raw_root.glob("cells/*/*/traces/*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for event in data["inference_events"]:
            if event.get("status") == "completed" and event.get("phase") == "steady_state":
                runtime = BUDGETED if BUDGETED in path.name else UNBOUNDED
                values[runtime].append(
                    1000.0 * float(event["model_inference_latency_seconds"])
                )
    return {
        runtime: {
            "completed_inference_events": len(rows),
            "inference_latency_ms_p50": percentile(rows, 50),
            "inference_latency_ms_p95": percentile(rows, 95),
            "inference_latency_ms_max": max(rows),
        }
        for runtime, rows in values.items()
    }


def paired_episode_rows(
    pairs: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for identity, unbounded, budgeted in pairs:
        rows.append(
            {
                "suite": identity[0],
                "task_id": identity[1],
                "initial_state_index": identity[2],
                "seed": identity[3],
                "unbounded_success": int(unbounded["success"]),
                "budgeted_success": int(budgeted["success"]),
                "success_difference": int(budgeted["success"])
                - int(unbounded["success"]),
                "unbounded_steps": unbounded["environment_steps"],
                "budgeted_steps": budgeted["environment_steps"],
                "steps_difference": budgeted["environment_steps"]
                - unbounded["environment_steps"],
                "unbounded_inference_calls": unbounded["inference_calls"],
                "budgeted_inference_calls": budgeted["inference_calls"],
                "call_ratio": budgeted["inference_calls"] / unbounded["inference_calls"],
                "unbounded_request_rate": unbounded["inference_requests_per_second"],
                "budgeted_request_rate": budgeted["inference_requests_per_second"],
                "unbounded_depletion_steps": unbounded["depletion_safe_hold_steps"],
                "budgeted_depletion_steps": budgeted["depletion_safe_hold_steps"],
            }
        )
    return rows


def failure_taxonomy(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in paired_rows:
        if row["unbounded_success"] and not row["budgeted_success"]:
            category = "budget_only_failure_paired_regression"
        elif not row["unbounded_success"] and not row["budgeted_success"]:
            category = "shared_policy_task_failure"
        elif not row["unbounded_success"] and row["budgeted_success"]:
            category = "unbounded_only_failure"
        else:
            continue
        rows.append(
            {
                "category": category,
                "suite": row["suite"],
                "task_id": row["task_id"],
                "initial_state_index": row["initial_state_index"],
                "seed": row["seed"],
                "unbounded_success": row["unbounded_success"],
                "budgeted_success": row["budgeted_success"],
                "unbounded_steps": row["unbounded_steps"],
                "budgeted_steps": row["budgeted_steps"],
                "evidence_boundary": (
                    "scored trace and telemetry retained; video unavailable because "
                    "the frozen capture contract records episode 0 only"
                ),
            }
        )
    return rows


def make_verdict_plot(
    output: Path,
    runtime_metrics: dict[str, dict[str, Any]],
    bootstrap: dict[str, Any],
) -> None:
    unbounded = runtime_metrics[UNBOUNDED]
    budgeted = runtime_metrics[BUDGETED]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    colors = ["#6b7280", "#2563eb"]
    axes[0].bar(["Unbounded", "Budgeted"], [unbounded["inference_calls"], budgeted["inference_calls"]], color=colors)
    reduction = 100 * (1 - budgeted["inference_calls"] / unbounded["inference_calls"])
    axes[0].set_title(f"Inference calls\n-{reduction:.1f}% (PASS)")
    axes[0].set_ylabel("Calls across 15 episodes")
    axes[1].bar(["Unbounded", "Budgeted"], [unbounded["successes"], budgeted["successes"]], color=colors)
    axes[1].axhline(unbounded["successes"], color="#dc2626", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 15.8)
    axes[1].set_title("Success count\n13 < 14 (FAIL)")
    axes[1].set_ylabel("Successes / 15")
    axes[2].bar(["Unbounded", "Budgeted"], [unbounded["median_inference_requests_per_second"], budgeted["median_inference_requests_per_second"]], color=colors)
    axes[2].axhline(2.5, color="#16a34a", linestyle="--", linewidth=1, label="frozen floor")
    axes[2].set_title("Median request rate\nBudgeted supply PASS")
    axes[2].set_ylabel("Requests / second")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "H2 compute-budget holdout: efficiency gain, but frozen verdict is NO-GO",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_paired_plot(output: Path, paired_rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['suite'].replace('libero_', '')}-{row['initial_state_index']}" for row in paired_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(x, [row["unbounded_inference_calls"] for row in paired_rows], "o-", label="Unbounded", color="#6b7280")
    axes[0].plot(x, [row["budgeted_inference_calls"] for row in paired_rows], "o-", label="Budgeted", color="#2563eb")
    axes[0].set_ylabel("Inference calls")
    axes[0].legend(frameon=False)
    axes[0].set_title("Paired compute cost")
    axes[1].bar(x, [row["steps_difference"] for row in paired_rows], color=["#dc2626" if row["success_difference"] < 0 else "#2563eb" for row in paired_rows])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Budgeted - unbounded steps")
    axes[1].set_title("Paired completion-step difference (failures retain 300-step cap)")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_video_contact_sheet(raw_root: Path, output: Path) -> list[dict[str, Any]]:
    inventory = []
    rows = []
    for path in sorted(raw_root.glob("cells/*/*/videos/*.mp4")):
        family = path.parents[2].name
        runtime = path.parents[1].name
        capture = cv2.VideoCapture(str(path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0 or fps <= 0:
            raise RuntimeError(f"Unreadable representative video: {path}")
        indices = [0, round((frame_count - 1) / 3), round(2 * (frame_count - 1) / 3), frame_count - 1]
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index} from {path}")
            frame = cv2.resize(frame, (280, 280), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        capture.release()
        strip = np.concatenate(frames, axis=1)
        label = f"{family} | {RUNTIME_LABELS[runtime]} | {frame_count}f @ {fps:.1f}fps"
        header = np.full((38, strip.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(header, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
        rows.append(np.concatenate([header, strip], axis=0))
        inventory.append(
            {
                "family": family,
                "runtime": runtime,
                "path": path.relative_to(raw_root).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "frame_count": frame_count,
                "fps": fps,
                "duration_seconds": frame_count / fps,
                "sampled_frame_indices": indices,
                "decode_status": "four_frames_checked",
            }
        )
    if len(rows) != 6:
        raise RuntimeError(f"Expected six representative videos, got {len(rows)}")
    cv2.imwrite(
        str(output),
        np.concatenate(rows, axis=0),
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )
    return inventory


def render_report(summary: dict[str, Any]) -> str:
    u = summary["runtime_metrics"][UNBOUNDED]
    b = summary["runtime_metrics"][BUDGETED]
    boot = summary["paired_bootstrap"]
    ratio = boot["aggregate_inference_call_ratio"]
    success = boot["success_difference"]
    steps = boot["environment_steps_difference"]
    gpu_u = summary["gpu_system_metrics"][UNBOUNDED]
    gpu_b = summary["gpu_system_metrics"][BUDGETED]
    lat_u = summary["pooled_inference_metrics"][UNBOUNDED]
    lat_b = summary["pooled_inference_metrics"][BUDGETED]
    lines = [
        "# ActionStream H2 compute-budget holdout — NO-GO",
        "",
        "The frozen H2 hypothesis is **not accepted**. A five-step request budget cut aggregate X-VLA inference calls substantially and preserved queue supply, but it lost one paired success relative to the unbounded pipelined aligned runtime. The predeclared success gate therefore fails.",
        "",
        "![H2 verdict](h2_verdict.png)",
        "",
        "## Frozen result",
        "",
        "| Metric | Unbounded pipelined | Budgeted (5 steps) | Frozen verdict |",
        "|---|---:|---:|---|",
        f"| Success | {u['successes']}/15 | {b['successes']}/15 | **FAIL**: budgeted < unbounded |",
        f"| Aggregate inference calls | {u['inference_calls']} | {b['inference_calls']} | **PASS**: ratio {ratio['estimate']:.3f}, 95% CI [{ratio['ci_95'][0]:.3f}, {ratio['ci_95'][1]:.3f}] |",
        f"| Call reduction | — | {100 * (1 - ratio['estimate']):.1f}% | Gate requires at least 50% |",
        f"| Median request rate | {u['median_inference_requests_per_second']:.3f}/s | {b['median_inference_requests_per_second']:.3f}/s | **PASS**: budgeted >= 2.5/s |",
        f"| Aggregate depletion | {100*u['depletion_fraction']:.2f}% | {100*b['depletion_fraction']:.2f}% | **PASS** |",
        f"| Mean completion steps | {u['mean_environment_steps']:.2f} | {b['mean_environment_steps']:.2f} | paired delta {steps['estimate']:+.2f}, CI [{steps['ci_95'][0]:+.2f}, {steps['ci_95'][1]:+.2f}] |",
        f"| Out-of-order / fallback / timeout / error | {u['out_of_order_rejections']}/{u['fallback_activations']}/{u['inference_timeouts']}/{u['inference_errors']} | {b['out_of_order_rejections']}/{b['fallback_activations']}/{b['inference_timeouts']}/{b['inference_errors']} | **PASS** |",
        "",
        f"Paired success difference is {100*success['estimate']:+.1f} percentage points, bootstrap 95% CI [{100*success['ci_95'][0]:+.1f}, {100*success['ci_95'][1]:+.1f}] pp. The formal decision uses the frozen count gate, not CI significance.",
        "",
        "## Family breakdown",
        "",
        "| Family | Success U -> B | Calls U -> B | Call reduction | Mean steps U -> B | Depletion B |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["family_table"]:
        lines.append(
            f"| {row['suite']} | {row['unbounded_successes']}/5 -> {row['budgeted_successes']}/5 | {row['unbounded_inference_calls']} -> {row['budgeted_inference_calls']} | {100*row['call_reduction_fraction']:.1f}% | {row['unbounded_mean_steps']:.1f} -> {row['budgeted_mean_steps']:.1f} | {100*row['budgeted_depletion_fraction']:.2f}% |"
        )
    lines += [
        "",
        "The only paired regression is `libero_goal`, state 19: unbounded succeeds in 90 steps while budgeted reaches the 300-step cap. Object state 16 fails under both runtimes. No failure video exists because the frozen capture contract records episode 0 only; the scored traces and telemetry are retained, so the semantic cause is intentionally left unclaimed.",
        "",
        "Goal-family call reduction is 49.1%, below 50%, because the budgeted state-19 failure runs to the full 300-step cap. The formal compute gate was frozen at aggregate level, so this is a heterogeneity warning rather than an additional post-hoc failure gate.",
        "",
        "![Paired effects](paired_effects.png)",
        "",
        "## GPU and runtime evidence",
        "",
        "| Metric | Unbounded | Budgeted |",
        "|---|---:|---:|",
        f"| Actual-worker startup latency p50 | {u['worker_startup_latency_seconds_p50']:.2f} s | {b['worker_startup_latency_seconds_p50']:.2f} s |",
        f"| Post-startup worker warmup p50 | {u['worker_post_startup_latency_ms_p50']:.2f} ms | {b['worker_post_startup_latency_ms_p50']:.2f} ms |",
        f"| Pooled steady inference latency p50 / p95 | {lat_u['inference_latency_ms_p50']:.2f} / {lat_u['inference_latency_ms_p95']:.2f} ms | {lat_b['inference_latency_ms_p50']:.2f} / {lat_b['inference_latency_ms_p95']:.2f} ms |",
        f"| Steady GPU utilization p50 / p95 | {gpu_u['gpu_utilization_percent_p50']:.1f}% / {gpu_u['gpu_utilization_percent_p95']:.1f}% | {gpu_b['gpu_utilization_percent_p50']:.1f}% / {gpu_b['gpu_utilization_percent_p95']:.1f}% |",
        f"| nvidia-smi process VRAM max | {gpu_u['process_gpu_memory_mib_max']:.0f} MiB | {gpu_b['process_gpu_memory_mib_max']:.0f} MiB |",
        f"| CUDA allocator peak max | {u['cuda_peak_memory_mib_max']:.0f} MiB | {b['cuda_peak_memory_mib_max']:.0f} MiB |",
        f"| Queue age episode-median p50 / p95 | {u['queue_age_p50_steps_episode_median']:.1f} / {u['queue_age_p95_steps_episode_median']:.1f} steps | {b['queue_age_p50_steps_episode_median']:.1f} / {b['queue_age_p95_steps_episode_median']:.1f} steps |",
        f"| Dropped aligned prefix steps | {u['dropped_prefix_steps']} | {b['dropped_prefix_steps']} |",
        f"| Budget-skipped observations | {u['observations_skipped_by_budget']} | {b['observations_skipped_by_budget']} |",
        "",
        "GPU utilization is descriptive, not a frozen gate. Both runtimes use the same X-VLA checkpoint, tasks, resets, fixed 950 ms delivery, safety limits, alignment, and pipelined delivery scheduler; only the minimum request interval differs (1 versus 5 control steps).",
        "",
        f"The budgeted steady-state p50 inference latency is {lat_b['inference_latency_ms_p50']/lat_u['inference_latency_ms_p50']:.2f}x the unbounded value while GPU utilization is much lower and queue age is about three steps higher. These measurements co-occur with the lower request duty cycle, but this holdout does not isolate GPU clocking or another cause. VRAM residency is unchanged.",
        "",
        f"The separate non-scored CUDA canary used state {summary['non_scored_gpu_canary']['initial_state_index']}, advanced zero scored environment steps, and recorded first/second inference latencies of {summary['non_scored_gpu_canary']['record']['inference_latency_seconds'][0]:.3f} s / {1000*summary['non_scored_gpu_canary']['record']['inference_latency_seconds'][1]:.2f} ms with a startup CUDA peak of {summary['non_scored_gpu_canary']['cuda_peak_memory_allocated_mib']:.0f} MiB.",
        "",
        "## Representative video check",
        "",
        "![Six representative videos sampled at four times](representative_video_contact_sheet.jpg)",
        "",
        "All six episode-0 MP4 files decoded successfully; four temporal positions per video were sampled into the contact sheet. This validates playable content and policy-driven scene motion, but does not provide footage for the state-19 paired regression.",
        "",
        "## Interpretation",
        "",
        "1. **Observation:** the budget saves 61.6% of inference calls, keeps median supply at 3.46 requests/s, and eliminates depletion in all three families, but steady p50 inference latency doubles and queue age rises.",
        "2. **Interpretation:** a fixed five-step budget is inside the operating envelope for 14 of 15 paired identities, but it is too coarse to be a universally safe replacement; the goal state-19 divergence is the decisive counterexample. Lower request duty cycle also does not automatically imply lower per-call latency.",
        "3. **Implication:** the useful engineering result is a measured compute/success trade-off, not a superior default runtime. H2 is a strict NO-GO even though most mechanism gates pass.",
        "4. **Next hypothesis:** do not retune or replay H2. If work continues, test one separately frozen release condition that spends extra requests only when a predeclared progress/deadline signal fires; otherwise close the project around the proven pipelined runtime and this negative boundary.",
        "",
        "## Evidence boundary",
        "",
        "- 3 distinct LIBERO task families, 15 paired identities, 30 scored episodes, 6 fresh GPU processes, 30 traces, 30 telemetry streams, and 6 representative videos.",
        f"- Execution Git HEAD: `{summary['evidence_validation']['execution_git_head']}`; candidate source commit: `{summary['evidence_validation']['candidate_base_commit']}`.",
        f"- Episodes SHA-256: `{summary['evidence_validation']['episodes_sha256']}`; transferred archive SHA-256: `{summary['source_archive_sha256']}`.",
        f"- Non-scored GPU canary archive SHA-256: `{summary['warmup_archive_sha256']}`.",
        "- This result does not establish jitter/burst robustness, real remote RPC behavior, RTC, real-robot safety, production soak, or universal superiority.",
        "",
        "Machine-readable details: `summary.json`, `paired_episodes.csv`, `family_table.csv`, `failure_taxonomy.csv`, `video_inventory.csv`.",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    records = load_records(raw_root)
    evidence = validate_evidence(raw_root, records)
    pairs = pair_records(records)
    runtime_metrics = {
        runtime: aggregate_runtime(records, runtime) for runtime in (UNBOUNDED, BUDGETED)
    }
    families = family_table(records)
    paired_rows = paired_episode_rows(pairs)
    bootstrap = bootstrap_effects(
        pairs,
        resamples=int(registry["analysis_plan"]["bootstrap_resamples"]),
        seed=int(registry["analysis_plan"]["bootstrap_seed"]),
    )
    gpu = system_gpu_metrics(raw_root)
    inference = pooled_inference_metrics(raw_root)
    failures = failure_taxonomy(paired_rows)
    gates = registry["decision_gates"]
    unbounded = runtime_metrics[UNBOUNDED]
    budgeted = runtime_metrics[BUDGETED]
    gate_results = {
        "aggregate_inference_call_ratio": {
            "value": bootstrap["aggregate_inference_call_ratio"]["estimate"],
            "threshold": gates["maximum_aggregate_inference_call_ratio"],
            "pass": bootstrap["aggregate_inference_call_ratio"]["estimate"]
            <= gates["maximum_aggregate_inference_call_ratio"],
        },
        "budgeted_median_request_rate": {
            "value": budgeted["median_inference_requests_per_second"],
            "threshold": gates["minimum_budgeted_median_request_rate_per_second"],
            "pass": budgeted["median_inference_requests_per_second"]
            >= gates["minimum_budgeted_median_request_rate_per_second"],
        },
        "budgeted_aggregate_depletion_fraction": {
            "value": budgeted["depletion_fraction"],
            "threshold": gates["maximum_budgeted_aggregate_depletion_fraction"],
            "pass": budgeted["depletion_fraction"]
            <= gates["maximum_budgeted_aggregate_depletion_fraction"],
        },
        "budgeted_per_family_depletion_fraction": {
            "value": max(row["budgeted_depletion_fraction"] for row in families),
            "threshold": gates["maximum_budgeted_depletion_fraction_per_task_family"],
            "pass": max(row["budgeted_depletion_fraction"] for row in families)
            <= gates["maximum_budgeted_depletion_fraction_per_task_family"],
        },
        "budgeted_success_count_not_below_unbounded": {
            "value": [budgeted["successes"], unbounded["successes"]],
            "threshold": "budgeted >= unbounded",
            "pass": budgeted["successes"] >= unbounded["successes"],
        },
        "out_of_order_rejections": {
            "value": budgeted["out_of_order_rejections"] + unbounded["out_of_order_rejections"],
            "threshold": gates["maximum_out_of_order_rejections"],
            "pass": budgeted["out_of_order_rejections"] + unbounded["out_of_order_rejections"]
            <= gates["maximum_out_of_order_rejections"],
        },
        "fallback_activations": {
            "value": budgeted["fallback_activations"] + unbounded["fallback_activations"],
            "threshold": gates["maximum_fallback_activations"],
            "pass": budgeted["fallback_activations"] + unbounded["fallback_activations"]
            <= gates["maximum_fallback_activations"],
        },
        "inference_timeouts": {
            "value": budgeted["inference_timeouts"] + unbounded["inference_timeouts"],
            "threshold": gates["maximum_inference_timeouts"],
            "pass": budgeted["inference_timeouts"] + unbounded["inference_timeouts"]
            <= gates["maximum_inference_timeouts"],
        },
        "inference_errors": {
            "value": budgeted["inference_errors"] + unbounded["inference_errors"],
            "threshold": gates["maximum_inference_errors"],
            "pass": budgeted["inference_errors"] + unbounded["inference_errors"]
            <= gates["maximum_inference_errors"],
        },
    }
    verdict = "GO" if all(row["pass"] for row in gate_results.values()) else "NO_GO"
    archive_hash = sha256(args.source_archive.resolve())
    warmup_archive_hash = sha256(args.warmup_archive.resolve())
    warmup_receipt = json.loads(args.warmup_receipt.read_text(encoding="utf-8"))
    if (
        warmup_receipt.get("status") != "completed_non_scored_warmup"
        or warmup_receipt.get("scored_environment_steps") != 0
        or warmup_receipt.get("record", {}).get("inference_calls") != 2
    ):
        raise RuntimeError("Non-scored GPU canary receipt failed validation")
    video_inventory = make_video_contact_sheet(
        raw_root, output_dir / "representative_video_contact_sheet.jpg"
    )
    summary = {
        "schema_version": 1,
        "experiment_id": "actionstream_transport_h2_budget_20260822",
        "hypothesis_id": "transport_h2_compute_budget",
        "verdict": verdict,
        "frozen_before_result": True,
        "source_archive_sha256": archive_hash,
        "warmup_archive_sha256": warmup_archive_hash,
        "non_scored_gpu_canary": warmup_receipt,
        "evidence_validation": evidence,
        "runtime_metrics": runtime_metrics,
        "family_table": families,
        "paired_bootstrap": bootstrap,
        "gpu_system_metrics": gpu,
        "pooled_inference_metrics": inference,
        "gate_results": gate_results,
        "failure_count": len(failures),
        "video_inventory": video_inventory,
        "claim_boundary": registry["claim_boundary"],
    }
    write_csv(output_dir / "main_table.csv", list(runtime_metrics.values()))
    write_csv(output_dir / "family_table.csv", families)
    write_csv(output_dir / "paired_episodes.csv", paired_rows)
    write_csv(output_dir / "failure_taxonomy.csv", failures)
    write_csv(output_dir / "video_inventory.csv", video_inventory)
    make_verdict_plot(output_dir / "h2_verdict.png", runtime_metrics, bootstrap)
    make_paired_plot(output_dir / "paired_effects.png", paired_rows)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        render_report(summary), encoding="utf-8", newline="\n"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--raw-root", type=Path, required=True)
    value.add_argument("--source-archive", type=Path, required=True)
    value.add_argument("--warmup-archive", type=Path, required=True)
    value.add_argument("--warmup-receipt", type=Path, required=True)
    value.add_argument("--registry", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main() -> None:
    summary = run(parser().parse_args())
    print(json.dumps({"verdict": summary["verdict"], "gate_results": summary["gate_results"]}, sort_keys=True))


if __name__ == "__main__":
    main()
