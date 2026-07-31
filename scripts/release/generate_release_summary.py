"""Write a portable, path-sanitized summary of the frozen M4 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "outputs" / "m4" / "report" / "m4_report.json"
OUTPUT_PATH = ROOT / "release" / "v1.0.0" / "evidence_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["status"] == "validated"
    assert report["validation"]["passed"] is True

    pressure_results = []
    for condition in report["pressure_condition_summaries"]:
        overall = condition["overall"]
        episodes = overall["success"]["episode_count"]
        successes = round(overall["success"]["mean"] * episodes)
        pressure_results.append(
            {
                "runtime_mode": condition["runtime_mode"],
                "injected_delay_ms": condition["injected_delay_ms"],
                "episodes": episodes,
                "successes": successes,
                "success_rate": overall["success"]["mean"],
                "environment_steps_mean": overall["environment_steps"]["mean"],
                "wall_clock_seconds_mean": overall[
                    "wall_clock_episode_seconds"
                ]["mean"],
                "queue_hold_steps_mean": overall[
                    "queue_underrun_hold_steps"
                ]["mean"],
                "fully_stale_chunks_mean": overall[
                    "stale_chunks_discarded"
                ]["mean"],
            }
        )

    paired = next(
        item
        for item in report["m4_paired_comparisons"]
        if item["comparison_id"] == "pressure_async_aligned_minus_async_naive"
    )
    paired_metrics = {}
    for metric in (
        "success",
        "environment_steps",
        "wall_clock_episode_seconds",
        "queue_underrun_hold_steps",
        "stale_chunks_discarded",
    ):
        value = paired["overall"][metric]
        paired_metrics[metric] = {
            "aligned_minus_naive_mean": value["paired_mean_difference"],
            "paired_bootstrap_95pct_ci": value[
                "paired_mean_difference_95pct_bootstrap_ci"
            ],
            "paired_episode_count": value["paired_episode_count"],
        }

    summary = {
        "schema_version": 1,
        "release": "v1.0.0",
        "source": {
            "path": REPORT_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256(REPORT_PATH),
            "status": report["status"],
        },
        "protocol": {
            "suite": "libero_object",
            "task_ids": [0, 1, 2],
            "episodes_per_task": 10,
            "paired_episode_count_per_mode": 30,
            "selected_pressure_delay_ms": report["calibration"][
                "selected_pressure_delay_ms"
            ],
            "queue_headroom_steps": report["calibration"]["queue_headroom_steps"],
            "bootstrap": report["bootstrap"],
        },
        "pressure_results": pressure_results,
        "paired_async_aligned_minus_async_naive": paired_metrics,
        "claim_statuses": report["claim_statuses"],
        "limitations": report["limitations"],
        "suggested_next_experiment": report["suggested_next_experiment"],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Saved {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
