"""Analyze immutable GPU benchmark v2 canary and holdout evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any, Iterable


RUNTIME_LABELS = {
    "sync_hold": "Sync",
    "lerobot_weighted_average": "Official Async",
    "lerobot_latest_only": "Latest-only",
    "actionstream_backend_aligned": "Aligned",
    "actionstream_backend_guarded": "Guarded",
}
PROFILE_ORDER = [
    "fixed_0000",
    "fixed_0250",
    "fixed_0950",
    "jitter_0600_pm0400_v2",
    "burst_0250_to2000_v2",
]
PROFILE_LABELS = {
    "fixed_0000": "0 ms",
    "fixed_0250": "250 ms",
    "fixed_0950": "950 ms",
    "jitter_0600_pm0400_v2": "600 +/- 400 ms jitter",
    "burst_0250_to2000_v2": "250 -> 2000 ms burst",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values: Iterable[float | int | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return statistics.fmean(selected) if selected else None


def _paired_rows(
    rows: list[dict[str, Any]], profile: str, candidate: str, reference: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    def key(row: dict[str, Any]) -> tuple[str, int, int, int]:
        return (
            str(row["suite"]),
            int(row["task_id"]),
            int(row["initial_state_index"]),
            int(row["seed"]),
        )

    candidates = {
        key(row): row
        for row in rows
        if row["delay_profile"] == profile and row["runtime"] == candidate
    }
    references = {
        key(row): row
        for row in rows
        if row["delay_profile"] == profile and row["runtime"] == reference
    }
    if set(candidates) != set(references):
        raise RuntimeError(
            f"Unpaired cell {profile} {candidate} vs {reference}: "
            f"candidate={len(candidates)} reference={len(references)}"
        )
    return [(candidates[item], references[item]) for item in sorted(candidates)]


def _bootstrap_ci(
    values: list[float], seed: int, samples: int = 10000
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires paired values")
    rng = random.Random(seed)
    estimates = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return estimates[int(0.025 * samples)], estimates[int(0.975 * samples) - 1]


def _effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    contrasts = [
        ("primary", "actionstream_backend_aligned", "lerobot_latest_only"),
        ("secondary", "actionstream_backend_aligned", "lerobot_weighted_average"),
    ]
    for profile_index, profile in enumerate(PROFILE_ORDER):
        for contrast_index, (kind, candidate, reference) in enumerate(contrasts):
            pairs = _paired_rows(rows, profile, candidate, reference)
            success = [
                float(bool(left["success"])) - float(bool(right["success"]))
                for left, right in pairs
            ]
            steps = [
                float(left["environment_steps"]) - float(right["environment_steps"])
                for left, right in pairs
            ]
            success_ci = _bootstrap_ci(
                success, 202608219000 + profile_index * 10 + contrast_index
            )
            steps_ci = _bootstrap_ci(
                steps, 202608219100 + profile_index * 10 + contrast_index
            )
            effects.append(
                {
                    "contrast": kind,
                    "profile": profile,
                    "candidate": candidate,
                    "reference": reference,
                    "paired_n": len(pairs),
                    "success_difference": statistics.fmean(success),
                    "success_ci95_low": success_ci[0],
                    "success_ci95_high": success_ci[1],
                    "environment_steps_difference": statistics.fmean(steps),
                    "steps_ci95_low": steps_ci[0],
                    "steps_ci95_high": steps_ci[1],
                }
            )
    return effects


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["delay_profile"], row["runtime"])].append(row)
    table: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        for runtime in RUNTIME_LABELS:
            selected = grouped[(profile, runtime)]
            table.append(
                {
                    "profile": profile,
                    "runtime": runtime,
                    "episodes": len(selected),
                    "successes": sum(bool(row["success"]) for row in selected),
                    "success_rate": _mean(bool(row["success"]) for row in selected),
                    "environment_steps_mean": _mean(
                        row["environment_steps"] for row in selected
                    ),
                    "steady_inference_p50_seconds_mean": _mean(
                        row.get("inference_latency_p50_seconds") for row in selected
                    ),
                    "steady_inference_p95_seconds_mean": _mean(
                        row.get("inference_latency_p95_seconds") for row in selected
                    ),
                    "request_per_second_mean": _mean(
                        row.get("inference_requests_per_second") for row in selected
                    ),
                    "queue_age_p95_steps_mean": _mean(
                        row.get("queue_age_p95_steps") for row in selected
                    ),
                    "discard_total": sum(
                        int(
                            row.get("stale_actions_discarded")
                            or row.get("dropped_prefix_steps")
                            or 0
                        )
                        for row in selected
                    ),
                    "depletion_total": sum(
                        int(row.get("depletion_safe_hold_steps") or 0)
                        for row in selected
                    ),
                    "fallback_total": sum(
                        int(row.get("fallback_activations") or 0) for row in selected
                    ),
                }
            )
    return table


def _gpu_systems(holdout_root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for receipt_path in sorted(holdout_root.glob("cells/*/*/run_receipt.json")):
        receipt = _read_json(receipt_path)
        selection = receipt["selection"]
        runtime = selection["runtimes"][0]
        family = receipt_path.parents[1].name
        gpu = receipt["system_gpu"]
        steady = gpu["phases"]["steady_state"]
        compatibility = receipt["compatibility"][0]
        output.append(
            {
                "task_family": family,
                "runtime": runtime,
                "pid": gpu["pid"],
                "model_load_wall_seconds": compatibility["model_load_wall_seconds"],
                "nvidia_smi_samples": gpu["sample_count"],
                "nvidia_smi_errors": gpu["error_count"],
                "resident_samples": steady["resident_sample_count"],
                "process_vram_mib_max": steady["process_gpu_memory_mib_max"],
                "gpu_utilization_percent_p50": steady["gpu_utilization_percent_p50"],
                "gpu_utilization_percent_p95": steady["gpu_utilization_percent_p95"],
                "gpu_utilization_percent_max": steady["gpu_utilization_percent_max"],
            }
        )
    return output


def _telemetry_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    files = 0
    events = 0
    invalid = 0
    required = {
        "schema_version",
        "engine_id",
        "event",
        "utc_unix_ns",
        "monotonic_ns",
        "pid",
        "thread",
    }
    for row in rows:
        value = row.get("telemetry_jsonl_path")
        if not value:
            continue
        path = Path(value)
        if not path.is_file() or _sha256(path) != row.get("telemetry_jsonl_sha256"):
            invalid += 1
            continue
        files += 1
        for event in _read_jsonl(path):
            events += 1
            if event.get("schema_version") != 1 or not required <= set(event):
                invalid += 1
    return {
        "telemetry_files": files,
        "telemetry_events": events,
        "invalid_files_or_events": invalid,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(table: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5.6))
    for runtime, label in RUNTIME_LABELS.items():
        selected = [row for row in table if row["runtime"] == runtime]
        axis.plot(
            range(len(PROFILE_ORDER)),
            [100.0 * float(row["success_rate"]) for row in selected],
            marker="o",
            label=label,
        )
    axis.set_xticks(
        range(len(PROFILE_ORDER)), [PROFILE_LABELS[item] for item in PROFILE_ORDER]
    )
    axis.set_ylabel("Success (%)")
    axis.set_xlabel("Frozen network operating point (categorical)")
    axis.set_ylim(-2, 102)
    axis.grid(alpha=0.25)
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def generate(canary_root: Path, holdout_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    holdout_receipt = _read_json(holdout_root / "orchestrator_receipt.json")
    canary = _read_jsonl(canary_root / "episodes.jsonl")
    rows = _read_jsonl(holdout_root / "episodes.jsonl")
    table = _aggregate(rows)
    effects = _effects(rows)
    systems = _gpu_systems(holdout_root)
    telemetry = _telemetry_audit(rows)

    expected_holdout = 3 * 5 * 5 * 3
    expected_canary = 3 * 5 * 6
    zero_sync = [
        row
        for row in canary
        if row["runtime"] == "sync_hold" and row["delay_profile"] == "fixed_0000"
    ]
    primary_950 = next(
        row
        for row in effects
        if row["contrast"] == "primary" and row["profile"] == "fixed_0950"
    )
    fixed_950_regressed = (
        primary_950["success_difference"] < 0
        or primary_950["environment_steps_difference"] > 0
    )
    verdict = (
        "LIMITED_OPERATING_ENVELOPE"
        if fixed_950_regressed
        else "SUPPORTED_ON_FROZEN_OPERATING_POINTS"
    )
    gates = {
        "canary_record_count": len(canary) == expected_canary,
        "holdout_record_count": len(rows) == expected_holdout,
        "three_task_families": len({row["suite"] for row in rows}) == 3,
        "canary_sync_zero_delay": len(zero_sync) == 3
        and all(bool(row["success"]) for row in zero_sync),
        "fresh_runtime_processes": holdout_receipt["fresh_process_count"] == 15,
        "actual_worker_warmup": all(row.get("worker_warmup") for row in rows),
        "true_request_rate": all(
            row.get("inference_requests_per_second") is not None for row in rows
        ),
        "nvidia_smi_residency": len(systems) == 15
        and all(row["resident_samples"] > 0 for row in systems),
        "standard_telemetry": telemetry["telemetry_files"] == 90
        and telemetry["invalid_files_or_events"] == 0,
    }
    status = "PASS" if all(gates.values()) else "PARTIAL"

    episode_table = [
        {
            "suite": row["suite"],
            "task_id": row["task_id"],
            "initial_state_index": row["initial_state_index"],
            "seed": row["seed"],
            "profile": row["delay_profile"],
            "runtime": row["runtime"],
            "success": row["success"],
            "environment_steps": row["environment_steps"],
            "steady_p50_seconds": row.get("inference_latency_p50_seconds"),
            "steady_p95_seconds": row.get("inference_latency_p95_seconds"),
            "startup_seconds": row["worker_warmup"]["startup_latency_seconds"],
            "request_per_second": row.get("inference_requests_per_second"),
            "queue_age_p95_steps": row.get("queue_age_p95_steps"),
            "discard": row.get(
                "stale_actions_discarded", row.get("dropped_prefix_steps")
            ),
            "depletion": row.get("depletion_safe_hold_steps"),
            "disconnects": row.get("disconnects"),
            "recoveries": row.get("recoveries"),
            "fallback": row.get("fallback_activations"),
            "trace_sha256": row["trace_sha256"],
        }
        for row in rows
    ]
    failures: list[dict[str, Any]] = []
    failure_groups: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        if not row["success"]:
            failure_groups[(row["runtime"], row["delay_profile"], row["suite"])] += 1
    for (runtime, profile, suite), count in sorted(failure_groups.items()):
        failures.append(
            {"runtime": runtime, "profile": profile, "suite": suite, "failures": count}
        )

    _write_csv(output_dir / "episode_table.csv", episode_table)
    _write_csv(output_dir / "main_table.csv", table)
    _write_csv(output_dir / "paired_effects.csv", effects)
    _write_csv(output_dir / "gpu_systems.csv", systems)
    if failures:
        _write_csv(output_dir / "failure_taxonomy.csv", failures)
    _plot(table, output_dir / "latency_success_operating_points.png")
    summary = {
        "schema_version": 2,
        "status": status,
        "verdict": verdict,
        "fixed_950_primary_effect": primary_950,
        "gates": gates,
        "telemetry": telemetry,
        "canary_receipt_sha256": _sha256(canary_root / "orchestrator_receipt.json"),
        "holdout_receipt_sha256": _sha256(holdout_root / "orchestrator_receipt.json"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    def cell(profile: str, runtime: str) -> dict[str, Any]:
        return next(
            row
            for row in table
            if row["profile"] == profile and row["runtime"] == runtime
        )

    table_lines = [
        "| Profile | Sync | Official Async | Latest-only | Aligned | Guarded |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILE_ORDER:
        values = []
        for runtime in RUNTIME_LABELS:
            row = cell(profile, runtime)
            values.append(f"{row['successes']}/{row['episodes']}")
        table_lines.append(
            f"| {PROFILE_LABELS[profile]} | " + " | ".join(values) + " |"
        )
    report = f"""# ActionStream GPU benchmark v2

Status: **{status}**. Method verdict: **{verdict}**.

This is a new frozen namespace; GPU benchmark v1 was not rewritten. Every
runtime/task-family cell ran in a fresh process. Warmup happened inside the
actual inference execution context, startup latency is separate from scored
steady-state latency, and driver residency/utilization came from sampled
`nvidia-smi` records.

## Main result

{chr(10).join(table_lines)}

The preregistered primary reference is official latest-only. At fixed 950 ms,
aligned minus latest-only success was
`{primary_950["success_difference"] * 100:.1f}` percentage points (paired 95%
CI `[{primary_950["success_ci95_low"] * 100:.1f},
{primary_950["success_ci95_high"] * 100:.1f}]`) and the paired completion-step
difference was `{primary_950["environment_steps_difference"]:.2f}` (95% CI
`[{primary_950["steps_ci95_low"]:.2f}, {primary_950["steps_ci95_high"]:.2f}]`).

If this cell regressed, aligned is explicitly a runtime with a limited
operating envelope; no universal-superiority claim is made. Aligned versus
official Async remains a secondary contrast in `paired_effects.csv`.

## Numbered findings

1. Engineering evidence gate: `{gates}`.
2. Formal evidence contains `{len(rows)}` paired task/runtime/network episodes;
   raw episode rows are in `episode_table.csv`.
3. Process-isolated GPU evidence covers `{len(systems)}` fresh runtime processes;
   process VRAM, p50/p95 GPU utilization, model-load time, and sampler coverage
   are in `gpu_systems.csv`.
4. Standard backend telemetry validated `{telemetry["telemetry_events"]}` events
   across `{telemetry["telemetry_files"]}` JSONL files with
   `{telemetry["invalid_files_or_events"]}` invalid files/events.
5. The network axis is a set of five frozen categorical operating points, not
   an interpolated continuous latency curve.

## Next experiment

Only repair the frozen SmolVLA sync gate and, if that gate passes, run the
official RTC comparison. v6 selector and Isaac Lab-Arena remain paused.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8", newline="\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(
        json.dumps(
            generate(
                args.canary_root.resolve(),
                args.holdout_root.resolve(),
                args.output_dir.resolve(),
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
