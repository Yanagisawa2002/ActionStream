"""Audit and report the frozen SmolVLA synchronous gate attempts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from actionstream.backend_gpu_report import read_results, validate_results


EXPECTED_SUITES = ("libero_object", "libero_spatial", "libero_goal")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage(row: dict[str, Any]) -> str:
    experiment = str(row["experiment_id"])
    if "_v2" in experiment:
        version = "v2"
    elif "_v3" in experiment:
        version = "v3"
    else:
        raise ValueError(f"Unexpected SmolVLA experiment namespace: {experiment}")
    if "dev_screen" in experiment:
        return f"{version}_development"
    if "canary" in experiment:
        return f"{version}_canary"
    if "holdout" in experiment:
        return f"{version}_holdout"
    raise ValueError(f"Unknown SmolVLA experiment stage: {experiment}")


def _median(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.median(values)) if values else None


def _maximum(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return max(values, default=None)


def build_development_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stage = _stage(row)
        if stage.endswith("_development"):
            groups[(stage, str(row["suite"]), int(row["task_id"]))].append(row)
    table = []
    for (stage, suite, task_id), group in sorted(groups.items()):
        successes = sum(bool(row["success"]) for row in group)
        table.append(
            {
                "stage": stage,
                "suite": suite,
                "task_id": task_id,
                "states": ",".join(str(row["initial_state_index"]) for row in group),
                "episodes": len(group),
                "successes": successes,
                "success_rate": successes / len(group),
                "mean_environment_steps": float(
                    np.mean([float(row["environment_steps"]) for row in group])
                ),
                "inference_latency_p50_seconds_median": _median(
                    group, "inference_latency_p50_seconds"
                ),
                "inference_latency_p95_seconds_median": _median(
                    group, "inference_latency_p95_seconds"
                ),
                "peak_cuda_memory_mib_max": _maximum(group, "peak_cuda_memory_mib"),
            }
        )
    return table


def build_canary_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stage = _stage(row)
        if stage.endswith("_canary"):
            groups[(stage, str(row["suite"]))].append(row)
    table = []
    for (stage, suite), group in sorted(groups.items()):
        group.sort(key=lambda row: int(row["episode_index"]))
        successes = sum(bool(row["success"]) for row in group)
        table.append(
            {
                "stage": stage,
                "suite": suite,
                "task_id": int(group[0]["task_id"]),
                "states": ",".join(str(row["initial_state_index"]) for row in group),
                "episodes": len(group),
                "successes": successes,
                "gate_pass": len(group) == 2 and successes == 2,
                "environment_steps": ",".join(
                    str(int(row["environment_steps"])) for row in group
                ),
                "inference_latency_p50_seconds_median": _median(
                    group, "inference_latency_p50_seconds"
                ),
                "inference_latency_p95_seconds_median": _median(
                    group, "inference_latency_p95_seconds"
                ),
                "peak_cuda_memory_mib_max": _maximum(group, "peak_cuda_memory_mib"),
            }
        )
    return table


def evaluate_gates(
    rows: list[dict[str, Any]], canary_table: list[dict[str, Any]]
) -> dict[str, Any]:
    holdout_rows = [row for row in rows if _stage(row).endswith("_holdout")]
    result: dict[str, Any] = {}
    for version in ("v2", "v3"):
        version_rows = [
            row for row in canary_table if row["stage"] == f"{version}_canary"
        ]
        suites = {str(row["suite"]) for row in version_rows}
        result[f"smolvla_{version}_three_suite_sync_gate"] = (
            suites == set(EXPECTED_SUITES)
            and len(version_rows) == len(EXPECTED_SUITES)
            and all(row["gate_pass"] for row in version_rows)
        )
        result[f"smolvla_{version}_suite_successes"] = {
            str(row["suite"]): f"{row['successes']}/{row['episodes']}"
            for row in version_rows
        }
    result.update(
        {
            "formal_holdout_record_count": len(holdout_rows),
            "formal_rtc_record_count": sum(
                row.get("runtime") == "lerobot_rtc" for row in holdout_rows
            ),
            "formal_rtc_status": "NOT_RUN_BY_FROZEN_SYNC_GATE",
            "formal_rtc_paired_effect_available": False,
        }
    )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_artifact_manifest(output_dir: Path) -> dict[str, dict[str, int | str]]:
    """Hash report artifacts without creating an impossible self-hash."""
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }


def _write_gate_plot(path: Path, canary_table: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, version in zip(axes, ("v2", "v3"), strict=True):
        by_suite = {
            str(row["suite"]): int(row["successes"])
            for row in canary_table
            if row["stage"] == f"{version}_canary"
        }
        values = [by_suite.get(suite, 0) for suite in EXPECTED_SUITES]
        colors = ["#2e8b57" if value == 2 else "#c44e52" for value in values]
        axis.bar(
            [suite.removeprefix("libero_") for suite in EXPECTED_SUITES],
            values,
            color=colors,
        )
        axis.axhline(
            2, color="#222222", linestyle="--", linewidth=1, label="required 2/2"
        )
        axis.set_title(f"SmolVLA {version} sync canary")
        axis.set_ylim(0, 2.25)
        axis.set_ylabel("Successful resets (of 2)")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Formal RTC remained sealed because the three-suite sync gate failed"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(
    v2_root: Path,
    v3_root: Path,
    output_dir: Path,
    *,
    raw_archive_sha256: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    validations: dict[str, Any] = {}
    for version, root in (("v2", v2_root), ("v3", v3_root)):
        rows, receipts = read_results(root)
        validations[version] = validate_results(rows, receipts)
        all_rows.extend(rows)

    for row in all_rows:
        if row["model_key"] != "smolvla":
            raise ValueError(
                f"Unexpected model in SmolVLA gate report: {row['model_key']}"
            )
        if row["runtime"] != "sync_hold" or row["delay_profile"] != "fixed_0000":
            raise ValueError(
                "Gate evidence must contain only zero-delay sync_hold records"
            )

    development_table = build_development_table(all_rows)
    canary_table = build_canary_table(all_rows)
    gates = evaluate_gates(all_rows, canary_table)
    if (
        gates["smolvla_v2_three_suite_sync_gate"]
        or gates["smolvla_v3_three_suite_sync_gate"]
    ):
        raise ValueError("This NO-GO report cannot be emitted for a passing sync gate")
    if gates["formal_holdout_record_count"]:
        raise ValueError("Formal holdout records exist despite a failed frozen gate")

    failures = [
        {
            "stage": row["stage"],
            "suite": row["suite"],
            "task_id": row["task_id"],
            "states": row["states"],
            "category": "sync_gate_task_failure",
            "failed_resets": int(row["episodes"]) - int(row["successes"]),
        }
        for row in canary_table
        if not row["gate_pass"]
    ]
    _write_csv(output_dir / "development_task_screen.csv", development_table)
    _write_csv(output_dir / "canary_table.csv", canary_table)
    _write_csv(output_dir / "failure_taxonomy.csv", failures)
    _write_gate_plot(output_dir / "sync_gate_by_suite.png", canary_table)

    summary = {
        "schema_version": 1,
        "verdict": "NO_GO",
        "model_id": "lerobot/smolvla_libero",
        "model_revision": "31d453f7edd78c839a8bbc39744a292686daf0de",
        "lerobot_commit": "d451fe4f1f1b00a812f95aa9534389b5e42ab155",
        "validation": validations,
        "gates": gates,
        "development_condition_count": len(development_table),
        "canary_condition_count": len(canary_table),
        "raw_archive_sha256": raw_archive_sha256,
        "formal_rtc_effect": None,
        "claim_boundary": (
            "Learned SmolVLA sync motion and several successful unseen resets are real; "
            "three-suite reset robustness did not pass, so official RTC holdout/effect was not run."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    canary_by_stage = defaultdict(list)
    for row in canary_table:
        canary_by_stage[row["stage"]].append(row)
    report = [
        "# SmolVLA official-RTC sync gate report",
        "",
        "**Verdict: NO-GO.** Formal RTC was not run because neither frozen three-suite "
        "zero-delay sync gate passed. This is a gate result, not a measured RTC effect.",
        "",
        "## What was real",
        "",
        f"- v2 validated {validations['v2']['episode_count']} learned-policy episodes and "
        f"{validations['v2']['trace_count_verified']} traces; v3 validated "
        f"{validations['v3']['episode_count']} episodes and "
        f"{validations['v3']['trace_count_verified']} traces.",
        "- The checkpoint drove the LIBERO robot through the complete 50-action sync FIFO. "
        "Recorded successful Object, Spatial, and Goal episodes were visually inspected.",
        "- Development task selection and both canary attempts used named, disjoint task/reset "
        "pairs. v2 was preserved after failure; v3 used a new namespace and stronger Goal screen.",
        "",
        "## Frozen canary results",
        "",
        "| Attempt | Suite | Task | States | Success | Steps | Infer p50/p95 ms | Peak CUDA MiB |",
        "|---|---|---:|---|---:|---|---:|---:|",
    ]
    for stage in ("v2_canary", "v3_canary"):
        for row in sorted(canary_by_stage[stage], key=lambda item: str(item["suite"])):
            report.append(
                f"| {stage.removesuffix('_canary')} | {row['suite']} | {row['task_id']} | "
                f"{row['states']} | {row['successes']}/{row['episodes']} | {row['environment_steps']} | "
                f"{1000 * float(row['inference_latency_p50_seconds_median']):.1f}/"
                f"{1000 * float(row['inference_latency_p95_seconds_median']):.1f} | "
                f"{float(row['peak_cuda_memory_mib_max']):.1f} |"
            )
    report.extend(
        [
            "",
            "v2 failed on Goal (1/2). v3 used a separately frozen stronger Goal screen and then "
            "failed on Object (1/2). Spatial passed both attempts; Goal v3 passed 2/2. The gate "
            "requires every suite to pass 2/2, so partial success cannot unlock the holdout.",
            "",
            "## Formal RTC boundary",
            "",
            "- Formal holdout records: **0**.",
            "- Official RTC records in the formal holdout: **0**.",
            "- Paired RTC effect / 95% CI: **unavailable by design**.",
            "- The pre-frozen holdout configs remain sealed for outcomes: states 40–44 were never "
            "loaded and their network traces were never consumed. No additional task/reset tuning "
            "was performed after the v3 failure.",
            "",
            "The legacy diagnostic also ruled out chunk-vs-step postprocessing as the gate fix "
            "(maximum action difference 0.0); the remaining limitation is reset-dependent "
            "open-loop sync robustness, not missing learned motion or a fabricated RTC result.",
            "",
            "## Evidence",
            "",
            f"Raw v2/v3 archive SHA-256: `{raw_archive_sha256}`.",
            "See `development_task_screen.csv`, `canary_table.csv`, `failure_taxonomy.csv`, "
            "`sync_gate_by_suite.png`, and `summary.json`. Videos are retained in the raw archive; "
            "the failed v3 Object reset has trace/predicate evidence but was not recorded because "
            "capture was predeclared for episode 0 only.",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8", newline="\n"
    )

    manifest = _build_artifact_manifest(output_dir)
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary
