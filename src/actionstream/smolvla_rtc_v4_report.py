"""Analyze the immutable SmolVLA v4 sync gate and official RTC holdout."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from actionstream.backend_gpu_v2_report import _bootstrap_ci, _gpu_systems


RUNTIME_LABELS = {
    "sync_hold": "Sync",
    "lerobot_latest_only": "Latest-only",
    "lerobot_rtc": "Official RTC",
}
PROFILE_ORDER = [
    "fixed_0000",
    "fixed_0250",
    "fixed_0950",
    "jitter_0600_pm0400_smol_v4",
    "burst_0250_to2000_smol_v4",
]
PROFILE_LABELS = {
    "fixed_0000": "0 ms",
    "fixed_0250": "250 ms",
    "fixed_0950": "950 ms",
    "jitter_0600_pm0400_smol_v4": "600 +/- 400 ms jitter",
    "burst_0250_to2000_smol_v4": "250 -> 2000 ms burst",
}


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pair_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["suite"]),
        int(row["task_id"]),
        int(row["initial_state_index"]),
        int(row["seed"]),
    )


def _paired_rows(
    rows: list[dict[str, Any]], profile: str, candidate: str, reference: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidates = {
        _pair_key(row): row
        for row in rows
        if row["delay_profile"] == profile and row["runtime"] == candidate
    }
    references = {
        _pair_key(row): row
        for row in rows
        if row["delay_profile"] == profile and row["runtime"] == reference
    }
    if set(candidates) != set(references):
        raise RuntimeError(
            f"Unpaired cell {profile} {candidate} vs {reference}: "
            f"candidate={len(candidates)} reference={len(references)}"
        )
    return [(candidates[key], references[key]) for key in sorted(candidates)]


def _effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    contrasts = [
        ("primary", "lerobot_rtc", "lerobot_latest_only"),
        ("secondary", "lerobot_rtc", "sync_hold"),
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
                success, 202608221000 + profile_index * 10 + contrast_index
            )
            steps_ci = _bootstrap_ci(
                steps, 202608221100 + profile_index * 10 + contrast_index
            )
            output.append(
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
    return output


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
                    "hold_steps_total": sum(
                        int(row.get("hold_steps") or 0) for row in selected
                    ),
                    "discard_total": sum(
                        int(
                            row.get("stale_actions_discarded")
                            or row.get("dropped_prefix_steps")
                            or 0
                        )
                        for row in selected
                    ),
                }
            )
    return table


def _plot(table: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9.2, 5.2))
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
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _gate_rows(canary_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(canary_root / "episodes.jsonl")
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_suite[str(row["suite"])].append(row)
    gate = {
        "record_count_6": len(rows) == 6,
        "three_task_families": len(by_suite) == 3,
        "two_unseen_resets_per_family": all(
            len(items) == 2 for items in by_suite.values()
        ),
        "all_sync_success": all(bool(row["success"]) for row in rows),
    }
    return rows, gate


def generate_gate(
    dev_root: Path, canary_root: Path, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    dev_rows = _read_jsonl(dev_root / "episodes.jsonl")
    canary_rows, gate = _gate_rows(canary_root)
    passed = all(gate.values())
    summary = {
        "schema_version": 1,
        "status": "PASS" if passed else "NO_GO",
        "formal_rtc_status": "ELIGIBLE" if passed else "NOT_RUN_BY_FROZEN_SYNC_GATE",
        "development_records": len(dev_rows),
        "canary_records": len(canary_rows),
        "gates": gate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# SmolVLA v4 sync gate",
        "",
        f"Verdict: **{summary['status']}**.",
        "",
        f"Development records: {len(dev_rows)}; canary records: {len(canary_rows)}.",
        f"Frozen gates: `{gate}`.",
        "",
        "Formal RTC remains sealed unless every suite passes both unseen sync resets.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(output_dir / "canary_table.csv", canary_rows)
    return summary


def generate(
    dev_root: Path, canary_root: Path, holdout_root: Path, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    dev_rows = _read_jsonl(dev_root / "episodes.jsonl")
    canary_rows, canary_gate = _gate_rows(canary_root)
    if not all(canary_gate.values()):
        raise RuntimeError("Formal RTC cannot be reported after a failed sync gate")
    rows = _read_jsonl(holdout_root / "episodes.jsonl")
    receipt = _read_json(holdout_root / "orchestrator_receipt.json")
    table = _aggregate(rows)
    effects = _effects(rows)
    systems = _gpu_systems(holdout_root)
    gates = {
        **{f"canary_{key}": value for key, value in canary_gate.items()},
        "holdout_record_count_225": len(rows) == 225,
        "three_task_families": len({row["suite"] for row in rows}) == 3,
        "three_official_runtimes": set(row["runtime"] for row in rows)
        == set(RUNTIME_LABELS),
        "fresh_runtime_processes_9": receipt.get("fresh_process_count") == 9,
        "actual_worker_warmup": all(row.get("worker_warmup") for row in rows),
        "true_request_rate": all(
            row.get("inference_requests_per_second") is not None for row in rows
        ),
        "nvidia_smi_residency": len(systems) == 9
        and all(row["resident_samples"] > 0 for row in systems),
        "upstream_rtc_records": sum(row["runtime"] == "lerobot_rtc" for row in rows)
        == 75,
    }
    status = "PASS" if all(gates.values()) else "PARTIAL"
    _write_csv(output_dir / "development_records.csv", dev_rows)
    _write_csv(output_dir / "canary_table.csv", canary_rows)
    _write_csv(output_dir / "episode_table.csv", rows)
    _write_csv(output_dir / "main_table.csv", table)
    _write_csv(output_dir / "paired_effects.csv", effects)
    _write_csv(output_dir / "gpu_systems.csv", systems)
    failures: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        if not row["success"]:
            failures[(row["runtime"], row["delay_profile"], row["suite"])] += 1
    if failures:
        _write_csv(
            output_dir / "failure_taxonomy.csv",
            [
                {
                    "runtime": key[0],
                    "profile": key[1],
                    "suite": key[2],
                    "failures": value,
                }
                for key, value in sorted(failures.items())
            ],
        )
    _plot(table, output_dir / "latency_success_operating_points.png")
    summary = {
        "schema_version": 1,
        "status": status,
        "verdict": "FORMAL_RTC_MEASURED" if status == "PASS" else "PARTIAL_EVIDENCE",
        "gates": gates,
        "development_records": len(dev_rows),
        "canary_records": len(canary_rows),
        "holdout_records": len(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def cell(profile: str, runtime: str) -> dict[str, Any]:
        return next(
            row
            for row in table
            if row["profile"] == profile and row["runtime"] == runtime
        )

    table_lines = [
        "| Profile | Sync | Latest-only | Official RTC |",
        "|---|---:|---:|---:|",
    ]
    for profile in PROFILE_ORDER:
        values = [
            f"{cell(profile, runtime)['successes']}/{cell(profile, runtime)['episodes']}"
            for runtime in RUNTIME_LABELS
        ]
        table_lines.append(
            f"| {PROFILE_LABELS[profile]} | " + " | ".join(values) + " |"
        )
    report = f"""# SmolVLA official RTC v4

Status: **{status}**. Verdict: **{summary["verdict"]}**.

The old v2/v3 sync-gate NO-GO remains unchanged. This v4 namespace used a
predeclared development selection rule, two new gate resets per task family,
and five disjoint formal resets only after the gate passed.

## Raw success table

{chr(10).join(table_lines)}

`paired_effects.csv` reports official RTC minus latest-only as the primary
contrast and RTC minus sync as the secondary contrast, with paired bootstrap
95% intervals. No universal RTC superiority claim is implied by completion.

## Numbered findings

1. Evidence gates: `{gates}`.
2. Formal evidence contains `{len(rows)}` episodes from three task families,
   three official runtimes, five network profiles, and five paired resets.
3. GPU evidence covers `{len(systems)}` fresh processes with process residency.
4. v6 selector and Isaac Lab-Arena were not run.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8", newline="\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-root", type=Path, required=True)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--holdout-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.holdout_root is None:
        result = generate_gate(
            args.dev_root.resolve(),
            args.canary_root.resolve(),
            args.output_dir.resolve(),
        )
    else:
        result = generate(
            args.dev_root.resolve(),
            args.canary_root.resolve(),
            args.holdout_root.resolve(),
            args.output_dir.resolve(),
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
