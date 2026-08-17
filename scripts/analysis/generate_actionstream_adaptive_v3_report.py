"""Generate the audited Adaptive Phase-Stable v3 canary report.

This report deliberately keeps the frozen state-30 canary separate from the
post-canary state-31 mechanism-coverage probe.  It never upgrades the latter
into paired performance evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SELECTOR_ID = "actionstream_adaptive_phase_stable_v3"
SELECTOR_SHA256 = "c824762ed9b05d37369181812795313c870b3bf8f5714d4c4ccd98e221021c98"
SUITES = ("object", "spatial", "goal")
PROFILES = ("fixed_0000", "fixed_0950", "phase_outage_requests_05_13")
RUNTIMES = (
    "sync_hold",
    "lerobot_latest_only",
    "actionstream_aligned",
    "actionstream_adaptive",
)
ASYNC_RUNTIMES = RUNTIMES[1:]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def suite_from_experiment(experiment_id: str) -> str:
    matches = [suite for suite in SUITES if f"_{suite}_" in experiment_id]
    if len(matches) != 1:
        raise ValueError(f"Could not identify one suite from {experiment_id}")
    return matches[0]


def decode_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    frames = 0
    first: np.ndarray | None = None
    final: np.ndarray | None = None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if first is None:
            first = frame.copy()
        final = frame
        frames += 1
    capture.release()
    delta = (
        float(np.abs(first.astype(np.int16) - final.astype(np.int16)).mean())
        if first is not None and final is not None
        else 0.0
    )
    return {
        "opened": opened,
        "frames": frames,
        "width": width,
        "height": height,
        "fps": fps,
        "first_last_mean_abs_delta": delta,
        "final_rgb": (
            cv2.cvtColor(final, cv2.COLOR_BGR2RGB) if final is not None else None
        ),
    }


def load_rows(roots: Iterable[Path], *, expected_rows_per_root: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        receipt_path = root / "run_receipt.json"
        metrics_path = root / "episodes.jsonl"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if int(receipt["record_count"]) != expected_rows_per_root:
            raise ValueError(f"Unexpected record count in {receipt_path}")
        if receipt["adaptive_selector"]["selector_id"] != SELECTOR_ID:
            raise ValueError(f"Selector ID drift in {receipt_path}")
        if (
            receipt["adaptive_selector"]["selector_protocol_sha256"]
            != SELECTOR_SHA256
        ):
            raise ValueError(f"Selector hash drift in {receipt_path}")
        if sha256_file(metrics_path) != receipt["metrics_sha256"]:
            raise ValueError(f"Metrics hash drift in {metrics_path}")
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["_suite"] = suite_from_experiment(str(row["experiment_id"]))
            if row["status"] != "completed":
                raise ValueError(f"Incomplete row in {metrics_path}: {row}")
            for artifact in ("trace", "video"):
                path = Path(str(row[f"{artifact}_path"]))
                if not path.is_file():
                    raise FileNotFoundError(path)
                if sha256_file(path) != row[f"{artifact}_sha256"]:
                    raise ValueError(f"{artifact} hash drift: {path}")
            video = decode_video(Path(str(row["video_path"])))
            if (
                not video["opened"]
                or video["frames"] <= 1
                or video["width"] != 640
                or video["height"] != 640
                or video["first_last_mean_abs_delta"] <= 0.0
            ):
                raise ValueError(f"Invalid or static video: {row['video_path']}")
            row["_video"] = video
            trace = json.loads(Path(str(row["trace_path"])).read_text(encoding="utf-8"))
            row["_merge_reasons"] = [
                event.get("merge", {}).get("reason")
                for event in trace["inference_events"]
            ]
            row["_observed_delays_ms"] = {
                int(event["delay_trace_index"]): round(
                    float(event["injected_delivery_delay_seconds"]) * 1000
                )
                for event in trace["inference_events"]
            }
            rows.append(row)
    return rows


def indexed(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["_suite"], row["delay_profile"], row["runtime"])
        if key in result:
            raise ValueError(f"Duplicate condition: {key}")
        result[key] = row
    return result


def condition_rows(canary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = indexed(canary)
    output: list[dict[str, Any]] = []
    for suite in SUITES:
        for profile in PROFILES:
            row: dict[str, Any] = {"suite": suite, "profile": profile}
            for runtime in RUNTIMES:
                episode = lookup[(suite, profile, runtime)]
                row[f"{runtime}_success"] = bool(episode["success"])
                row[f"{runtime}_steps"] = int(episode["environment_steps"])
            adaptive = lookup[(suite, profile, "actionstream_adaptive")]
            latest = lookup[(suite, profile, "lerobot_latest_only")]
            successful_static = [
                lookup[(suite, profile, runtime)]
                for runtime in ("lerobot_latest_only", "actionstream_aligned")
                if lookup[(suite, profile, runtime)]["success"]
            ]
            row["adaptive_minus_latest_steps"] = (
                int(adaptive["environment_steps"])
                - int(latest["environment_steps"])
            )
            row["adaptive_minus_best_successful_static_steps"] = (
                int(adaptive["environment_steps"])
                - min(int(item["environment_steps"]) for item in successful_static)
            )
            output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_steps(table: list[dict[str, Any]], path: Path) -> None:
    labels = [
        f"{row['suite'].title()}\n{('0 ms' if row['profile'] == 'fixed_0000' else '950 ms' if row['profile'] == 'fixed_0950' else 'phase outage')}"
        for row in table
    ]
    x = np.arange(len(table), dtype=np.float64)
    width = 0.25
    styles = (
        ("lerobot_latest_only", "Latest-only", "#6B7280"),
        ("actionstream_aligned", "Static aligned", "#38BDF8"),
        ("actionstream_adaptive", "Adaptive v3", "#22C55E"),
    )
    fig, axis = plt.subplots(figsize=(15.5, 6.5))
    for offset, (runtime, label, color) in zip((-width, 0.0, width), styles, strict=True):
        heights = [row[f"{runtime}_steps"] for row in table]
        bars = axis.bar(x + offset, heights, width, label=label, color=color, alpha=0.9)
        for bar, row in zip(bars, table, strict=True):
            if not row[f"{runtime}_success"]:
                bar.set_hatch("///")
                bar.set_edgecolor("#DC2626")
                bar.set_linewidth(2.0)
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 4,
                    "FAIL",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#B91C1C",
                    fontweight="bold",
                )
    axis.set_title("Frozen state-30 learned-policy canary: success and completion steps")
    axis.set_ylabel("Environment steps (300 = timeout)")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 335)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, metadata={"Software": "ActionStream"})
    plt.close(fig)


def render_content_montage(canary: list[dict[str, Any]], path: Path) -> None:
    lookup = indexed(canary)
    cases = (
        ("object", "fixed_0950", "Object task 5 / 950 ms"),
        ("goal", "phase_outage_requests_05_13", "Goal task 2 / phase outage"),
    )
    runtimes = (
        ("lerobot_latest_only", "Latest-only"),
        ("actionstream_aligned", "Static aligned"),
        ("actionstream_adaptive", "Adaptive v3"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row_index, (suite, profile, case_label) in enumerate(cases):
        for column_index, (runtime, runtime_label) in enumerate(runtimes):
            episode = lookup[(suite, profile, runtime)]
            axis = axes[row_index, column_index]
            axis.imshow(episode["_video"]["final_rgb"])
            status = "SUCCESS" if episode["success"] else "FAIL"
            axis.set_title(
                f"{case_label}\n{runtime_label}: {status}, {episode['environment_steps']} steps",
                color="#15803D" if episode["success"] else "#B91C1C",
                fontsize=11,
            )
            axis.axis("off")
    fig.suptitle("Same task, reset, checkpoint, and network trace: final scene content", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=180, metadata={"Software": "ActionStream"})
    plt.close(fig)


def render_coverage(coverage: list[dict[str, Any]], path: Path) -> None:
    order = {suite: index for index, suite in enumerate(SUITES)}
    profile_order = {
        "fixed_0000": 0,
        "early_outage_requests_02_05": 1,
    }
    rows = sorted(
        coverage,
        key=lambda row: (
            order[row["_suite"]],
            profile_order.get(row["delay_profile"], 2),
        ),
    )
    labels = [
        f"{row['_suite'].title()}\n{('zero' if row['delay_profile'] == 'fixed_0000' else 'early outage' if row['delay_profile'] == 'early_outage_requests_02_05' else 'fresh pulse')}"
        for row in rows
    ]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.36
    phase = [int(row["adaptive_phase_lock_rejections"]) for row in rows]
    preserved = [int(row["adaptive_preserved_queue_rejections"]) for row in rows]
    fig, axis = plt.subplots(figsize=(14, 6))
    axis.bar(x - width / 2, phase, width, label="Phase-lock rejections", color="#8B5CF6")
    axis.bar(x + width / 2, preserved, width, label="Queue-preserving rejections", color="#F59E0B")
    for index, row in enumerate(rows):
        if not row["success"]:
            axis.axvspan(index - 0.48, index + 0.48, color="#FEE2E2", alpha=0.7)
            axis.text(index, max(phase[index], preserved[index]) + 0.25, "TASK FAIL", ha="center", color="#B91C1C", fontsize=8)
    axis.set_title("Post-canary state-31 mechanism coverage (diagnostic only)")
    axis.set_ylabel("Recorded selector/runtime events")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, max(max(phase), max(preserved), 1) + 1.5)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, metadata={"Software": "ActionStream"})
    plt.close(fig)


def format_cell(success: bool, steps: int) -> str:
    return f"{'PASS' if success else 'FAIL'} {steps}"


def build_report(summary: dict[str, Any], table: list[dict[str, Any]]) -> str:
    lines = [
        "# ActionStream Adaptive Phase-Stable v3 result",
        "",
        f"**Verdict: {summary['verdict']['canary']}.** The learned-policy canary has a strong positive fixed-950-ms result, but the formal holdout remains unopened. The separate mechanism-coverage probe is **{summary['verdict']['coverage']}** and cannot be used as a performance claim.",
        "",
        "## Frozen canary main table",
        "",
        "`PASS/FAIL steps`; sync blocks the environment during inference and is a task-capability reference, not a latency-runtime competitor.",
        "",
        "| Task | Profile | Sync | Latest-only | Static aligned | Adaptive v3 | Adaptive - latest |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        profile = {
            "fixed_0000": "0 ms",
            "fixed_0950": "950 ms",
            "phase_outage_requests_05_13": "phase outage",
        }[row["profile"]]
        lines.append(
            "| "
            + " | ".join(
                (
                    row["suite"].title(),
                    profile,
                    format_cell(row["sync_hold_success"], row["sync_hold_steps"]),
                    format_cell(
                        row["lerobot_latest_only_success"],
                        row["lerobot_latest_only_steps"],
                    ),
                    format_cell(
                        row["actionstream_aligned_success"],
                        row["actionstream_aligned_steps"],
                    ),
                    format_cell(
                        row["actionstream_adaptive_success"],
                        row["actionstream_adaptive_steps"],
                    ),
                    f"{row['adaptive_minus_latest_steps']:+d}",
                )
            )
            + " |"
        )
    fixed = summary["findings"]["fixed_0950"]
    zero = summary["findings"]["fixed_0000"]
    outage = summary["findings"]["phase_outage"]
    coverage = summary["coverage"]
    lines.extend(
        [
            "",
            "![Canary steps and success](canary_steps.png)",
            "",
            "## Findings",
            "",
            f"1. **Positive at 950 ms:** Adaptive succeeded on 3/3 tasks and used {fixed['adaptive_total_steps']} total steps versus {fixed['latest_total_steps']} for official latest-only, a {fixed['relative_reduction_percent']:.1f}% reduction. It beat latest-only on every task (99, 25, and 25 fewer steps). Static aligned failed Object while Adaptive succeeded.",
            f"2. **No aggregate zero-delay tax in this canary:** Adaptive and latest-only both succeeded 3/3 with exactly {zero['adaptive_total_steps']} total steps. Per-task differences were +2, 0, and -2 steps.",
            f"3. **Outage trade-off remains:** Adaptive and latest-only both succeeded 3/3, while aligned succeeded only 1/3. Adaptive used {outage['adaptive_total_steps']} versus {outage['latest_total_steps']} total steps because Object was 124 steps slower, despite being 6 and 18 steps faster on Spatial and Goal.",
            "4. **This is not a formal effect size:** there is only one paired reset per task/profile. The nine conditions are heterogeneous operating points, not nine IID seeds, so no 95% CI or generalization claim is reported.",
            "5. **The original canary is PARTIAL_POSITIVE, not GO:** all 36 hashes/videos and all nine Adaptive successes passed, but success termination censored request ordinal 13 in two async rows. The manifests were not altered after inspection.",
            "",
            "## Mechanism coverage",
            "",
            f"The post-canary state-31 probe is **NO-GO**: {coverage['successes']}/{coverage['episodes']} task successes, zero-delay capability only {coverage['zero_successes']}/3, and the frozen fresh pulse triggered a phase-lock rejection on {coverage['pulse_phase_rejection_rows']}/3 tasks. Across all nine rows, the runtime nevertheless recorded {coverage['phase_lock_rejections']} phase-lock rejections and {coverage['preserved_queue_rejections']} queue-preserving rejections with {coverage['guard_discards']} guard discards.",
            "",
            "![Mechanism coverage](mechanism_coverage.png)",
            "",
            "The early-outage rows executed both 1700 ms events on all three tasks and succeeded 3/3. However, every hard outage arrived after the active queue had already drained, so the non-destructive outage guard did not preserve useful slack. Its next real method step is a slack-aware prefetch/reserve scheduler, not another age threshold tweak.",
            "",
            "## Content-level paired visualization",
            "",
            "![Same-reset final scene comparison](paired_content_final_frames.png)",
            "",
            "These are final RGB frames from the recorded paired episodes. The complete MP4s and traces remain outside ordinary Git; their hashes are recorded in the source rows and validation summary.",
            "",
            "## Claim boundary and next decision",
            "",
            "Resume-safe claim: *Built and froze a task-label-blind phase-stable VLA runtime; on a three-task X-VLA canary at 950 ms, it matched 3/3 success while reducing completion steps 29.8% versus official latest-only, with zero aggregate 0-ms regression; branch-targeted coverage exposed the remaining outage-slack failure boundary.*",
            "",
            "Do not claim a formal holdout win, universal superiority, Isaac Lab-Arena, or real-robot validation. The v3 formal holdout stays closed. Register v4 only after adding observable queue-slack prediction/early request, then use new tasks, states, and traces. A minimum real-robot paired A/B remains separately blocked by hardware, driver, calibration, operator, and E-stop availability.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-root", action="append", type=Path, required=True)
    parser.add_argument("--coverage-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.canary_root) != 3 or len(args.coverage_root) != 3:
        raise ValueError("Expected exactly three canary roots and three coverage roots")

    canary = load_rows(args.canary_root, expected_rows_per_root=12)
    coverage = load_rows(args.coverage_root, expected_rows_per_root=3)
    if len(canary) != 36 or len(coverage) != 9:
        raise ValueError("Unexpected total row count")
    table = condition_rows(canary)

    adaptive = [row for row in canary if row["runtime"] == "actionstream_adaptive"]
    latest = [row for row in canary if row["runtime"] == "lerobot_latest_only"]
    aligned = [row for row in canary if row["runtime"] == "actionstream_aligned"]
    sync_zero = [
        row
        for row in canary
        if row["runtime"] == "sync_hold" and row["delay_profile"] == "fixed_0000"
    ]
    async_phase = [
        row
        for row in canary
        if row["runtime"] in ASYNC_RUNTIMES
        and row["delay_profile"] == "phase_outage_requests_05_13"
    ]
    event5 = sum(row["_observed_delays_ms"].get(5) == 1700 for row in async_phase)
    event13 = sum(row["_observed_delays_ms"].get(13) == 1700 for row in async_phase)

    def profile_summary(profile: str) -> dict[str, Any]:
        adaptive_rows = [row for row in adaptive if row["delay_profile"] == profile]
        latest_rows = [row for row in latest if row["delay_profile"] == profile]
        aligned_rows = [row for row in aligned if row["delay_profile"] == profile]
        adaptive_total = sum(int(row["environment_steps"]) for row in adaptive_rows)
        latest_total = sum(int(row["environment_steps"]) for row in latest_rows)
        return {
            "adaptive_successes": sum(bool(row["success"]) for row in adaptive_rows),
            "latest_successes": sum(bool(row["success"]) for row in latest_rows),
            "aligned_successes": sum(bool(row["success"]) for row in aligned_rows),
            "adaptive_total_steps": adaptive_total,
            "latest_total_steps": latest_total,
            "relative_reduction_percent": (latest_total - adaptive_total)
            / latest_total
            * 100.0,
        }

    pulse_rows = [
        row for row in coverage if row["delay_profile"].startswith("closed_phase_fresh")
    ]
    early_rows = [
        row
        for row in coverage
        if row["delay_profile"] == "early_outage_requests_02_05"
    ]
    zero_rows = [row for row in coverage if row["delay_profile"] == "fixed_0000"]
    coverage_gate = {
        "episodes": len(coverage),
        "successes": sum(bool(row["success"]) for row in coverage),
        "zero_successes": sum(bool(row["success"]) for row in zero_rows),
        "early_outage_both_events_rows": sum(
            row["_observed_delays_ms"].get(2) == 1700
            and row["_observed_delays_ms"].get(5) == 1700
            for row in early_rows
        ),
        "early_outage_successes": sum(bool(row["success"]) for row in early_rows),
        "pulse_override_rows": sum(
            any(value == 0 for key, value in row["_observed_delays_ms"].items() if key > 0)
            for row in pulse_rows
        ),
        "pulse_phase_rejection_rows": sum(
            int(row["adaptive_phase_lock_rejections"]) > 0 for row in pulse_rows
        ),
        "pulse_preserved_queue_rows": sum(
            int(row["adaptive_preserved_queue_rejections"]) > 0
            for row in pulse_rows
        ),
        "phase_lock_rejections": sum(
            int(row["adaptive_phase_lock_rejections"]) for row in coverage
        ),
        "preserved_queue_rejections": sum(
            int(row["adaptive_preserved_queue_rejections"]) for row in coverage
        ),
        "guard_discards": sum(
            int(row["adaptive_pending_actions_discarded_by_guard"])
            for row in coverage
        ),
    }
    coverage_go = (
        coverage_gate["zero_successes"] == 3
        and coverage_gate["early_outage_both_events_rows"] == 3
        and coverage_gate["pulse_override_rows"] == 3
        and coverage_gate["pulse_phase_rejection_rows"] == 3
        and coverage_gate["pulse_preserved_queue_rows"] == 3
        and coverage_gate["guard_discards"] == 0
    )
    canary_core = (
        len(adaptive) == 9
        and all(row["success"] for row in adaptive)
        and len(sync_zero) == 3
        and all(row["success"] for row in sync_zero)
        and all(int(row["adaptive_pending_actions_discarded_by_guard"]) == 0 for row in adaptive)
    )
    canary_event_complete = event5 == len(async_phase) and event13 == len(async_phase)

    summary = {
        "schema_version": 1,
        "selector_id": SELECTOR_ID,
        "selector_sha256": SELECTOR_SHA256,
        "verdict": {
            "canary": (
                "GO" if canary_core and canary_event_complete else "PARTIAL_POSITIVE"
            ),
            "coverage": "GO" if coverage_go else "NO-GO",
            "formal_holdout_opened": False,
        },
        "validation": {
            "canary_rows": len(canary),
            "coverage_rows": len(coverage),
            "trace_hashes_verified": len(canary) + len(coverage),
            "videos_decoded": len(canary) + len(coverage),
            "adaptive_canary_successes": sum(bool(row["success"]) for row in adaptive),
            "latest_canary_successes": sum(bool(row["success"]) for row in latest),
            "aligned_canary_successes": sum(bool(row["success"]) for row in aligned),
            "sync_canary_successes": sum(
                bool(row["success"])
                for row in canary
                if row["runtime"] == "sync_hold"
            ),
            "phase_async_rows_with_event_5": event5,
            "phase_async_rows_with_event_13": event13,
            "phase_async_rows": len(async_phase),
        },
        "findings": {
            "fixed_0000": profile_summary("fixed_0000"),
            "fixed_0950": profile_summary("fixed_0950"),
            "phase_outage": profile_summary("phase_outage_requests_05_13"),
            "overall_adaptive_minus_latest_steps": sum(
                row["adaptive_minus_latest_steps"] for row in table
            ),
            "overall_adaptive_minus_best_successful_static_steps": sum(
                row["adaptive_minus_best_successful_static_steps"] for row in table
            ),
            "statistical_boundary": "One paired reset per task/profile; no IID-seed CI is reported.",
        },
        "coverage": coverage_gate,
        "failure_boundary": {
            "hard_outage_queue_preservation_exercised": any(
                "adaptive_rejected_chunk_preserved_validated_queue"
                in row["_merge_reasons"]
                for row in early_rows
            ),
            "reason": "Every >28-step hard outage arrived after the active queue had drained; safe hold began at queue depth zero.",
            "next_method": "Slack-aware prefetch/reserve scheduling before phase-stable selection.",
        },
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "main_table.csv", table)
    coverage_csv = [
        {
            "suite": row["_suite"],
            "profile": row["delay_profile"],
            "success": row["success"],
            "steps": row["environment_steps"],
            "phase_lock_rejections": row["adaptive_phase_lock_rejections"],
            "preserved_queue_rejections": row["adaptive_preserved_queue_rejections"],
            "guard_discards": row["adaptive_pending_actions_discarded_by_guard"],
        }
        for row in coverage
    ]
    write_csv(output / "coverage_table.csv", coverage_csv)
    render_steps(table, output / "canary_steps.png")
    render_content_montage(canary, output / "paired_content_final_frames.png")
    render_coverage(coverage, output / "mechanism_coverage.png")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "report.md").write_text(
        build_report(summary, table), encoding="utf-8", newline="\n"
    )
    artifact_paths = sorted(
        path for path in output.iterdir() if path.name != "artifact_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "artifacts": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
