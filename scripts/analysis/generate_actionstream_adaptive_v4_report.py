"""Generate the audited Queue-Slack Adaptive v4 development-canary report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SELECTOR_ID = "actionstream_adaptive_queue_slack_v4"
SELECTOR_SHA256 = "7e437cbaeb3d274e098d8a068fafa482697148933eb5431c28c48cd51632a021"
SOURCE_COMMIT = "d78c81ec89ba460945454220ba276a3a3d63622e"
SUITES = ("object", "spatial", "goal")
PROFILES = ("low_jitter_v4", "high_jitter_v4", "outage_v4_requests_02_06")
RUNTIMES = (
    "sync_hold",
    "lerobot_latest_only",
    "actionstream_aligned",
    "actionstream_adaptive",
)
PROTOCOL_SHA256 = {
    "object": "88bd26354614872cfc29b1becbd71ba23a0cd11c85f50e7bdce6254034bf2451",
    "spatial": "12a81f489d603f42c831cda2113bddd5350cf075eaf8960e9df7c3b844c0b057",
    "goal": "cabe2d0bd2790598e07bc2aa1f3081e1e92007e01857b2d196782f4c8b709cdb",
}


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
        "first_last_mean_abs_delta": delta,
        "final_rgb": (
            cv2.cvtColor(final, cv2.COLOR_BGR2RGB) if final is not None else None
        ),
    }


def sampled_video_frames(path: Path, count: int = 5) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) < count:
        raise ValueError(f"Too few frames for timeline: {path}")
    indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[index] for index in indices]


def load_rows(roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        receipt_path = root / "run_receipt.json"
        metrics_path = root / "episodes.jsonl"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        suite = suite_from_experiment(str(receipt["experiment_id"]))
        if int(receipt["record_count"]) != 12:
            raise ValueError(f"Unexpected record count in {receipt_path}")
        if receipt["source_commit"] != SOURCE_COMMIT:
            raise ValueError(f"Source commit drift in {receipt_path}")
        if receipt["protocol_sha256"] != PROTOCOL_SHA256[suite]:
            raise ValueError(f"Protocol hash drift in {receipt_path}")
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
            row["_suite"] = suite
            if row["status"] != "completed":
                raise ValueError(f"Incomplete row in {metrics_path}")
            trace_path = Path(str(row["trace_path"]))
            video_path = Path(str(row["video_path"]))
            if sha256_file(trace_path) != row["trace_sha256"]:
                raise ValueError(f"Trace hash drift: {trace_path}")
            if sha256_file(video_path) != row["video_sha256"]:
                raise ValueError(f"Video hash drift: {video_path}")
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            actions = np.asarray(
                [item["action"] for item in trace["actions"]], dtype=np.float64
            )
            if (
                len(actions) != int(row["environment_steps"])
                or actions.ndim != 2
                or actions.shape[1] != 7
                or not np.isfinite(actions).all()
            ):
                raise ValueError(f"Invalid trace actions: {trace_path}")
            video = decode_video(video_path)
            if (
                not video["opened"]
                or video["frames"] != int(row["environment_steps"]) + 1
                or video["width"] != 640
                or video["height"] != 640
                or video["first_last_mean_abs_delta"] <= 0.0
            ):
                raise ValueError(f"Invalid or static video: {video_path}")
            row["_video"] = video
            row["_trace"] = trace
            row["_observed_delays_ms"] = {
                int(event["delay_trace_index"]): round(
                    float(event["injected_delivery_delay_seconds"]) * 1000
                )
                for event in trace["inference_events"]
            }
            rows.append(row)
    return rows


def indexed(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["_suite"], row["delay_profile"], row["runtime"])
        if key in output:
            raise ValueError(f"Duplicate condition: {key}")
        output[key] = row
    return output


def condition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = indexed(rows)
    output: list[dict[str, Any]] = []
    for suite in SUITES:
        for profile in PROFILES:
            item: dict[str, Any] = {"suite": suite, "profile": profile}
            for runtime in RUNTIMES:
                episode = lookup[(suite, profile, runtime)]
                item[f"{runtime}_success"] = bool(episode["success"])
                item[f"{runtime}_steps"] = int(episode["environment_steps"])
            item["adaptive_minus_latest_steps"] = (
                item["actionstream_adaptive_steps"]
                - item["lerobot_latest_only_steps"]
            )
            item["adaptive_minus_aligned_steps"] = (
                item["actionstream_adaptive_steps"]
                - item["actionstream_aligned_steps"]
            )
            output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_steps(table: list[dict[str, Any]], path: Path) -> None:
    profile_labels = {
        "low_jitter_v4": "25–125 ms",
        "high_jitter_v4": "600–1100 ms",
        "outage_v4_requests_02_06": "425 / 1850 ms",
    }
    labels = [
        f"{row['suite'].title()}\n{profile_labels[row['profile']]}" for row in table
    ]
    x = np.arange(len(table), dtype=np.float64)
    width = 0.25
    styles = (
        ("lerobot_latest_only", "Latest-only", "#6B7280"),
        ("actionstream_aligned", "Static aligned", "#38BDF8"),
        ("actionstream_adaptive", "Adaptive v4", "#22C55E"),
    )
    fig, axis = plt.subplots(figsize=(15.5, 6.5))
    for offset, (runtime, label, color) in zip((-width, 0.0, width), styles, strict=True):
        bars = axis.bar(
            x + offset,
            [row[f"{runtime}_steps"] for row in table],
            width,
            label=label,
            color=color,
            alpha=0.9,
        )
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
                    color="#B91C1C",
                    fontsize=8,
                    fontweight="bold",
                )
    axis.set_title("Frozen state-32 Queue-Slack v4 canary")
    axis.set_ylabel("Environment steps (300 = timeout)")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 335)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, metadata={"Software": "ActionStream"})
    plt.close(fig)


def render_mechanism(rows: list[dict[str, Any]], path: Path) -> None:
    adaptive = [row for row in rows if row["runtime"] == "actionstream_adaptive"]
    labels = [
        f"{row['_suite'].title()}\n{row['delay_profile'].replace('_v4', '').replace('outage_requests_02_06', 'outage')}"
        for row in adaptive
    ]
    x = np.arange(len(adaptive), dtype=np.float64)
    fig, axis = plt.subplots(figsize=(15.5, 6.5))
    axis.bar(
        x - 0.2,
        [int(row["adaptive_reserve_hold_steps"]) for row in adaptive],
        0.4,
        label="Reserve hold steps",
        color="#F59E0B",
    )
    axis.bar(
        x + 0.2,
        [int(row["adaptive_prefetch_requests"]) for row in adaptive],
        0.4,
        label="Prefetch requests",
        color="#8B5CF6",
    )
    for index, row in enumerate(adaptive):
        if not row["success"]:
            axis.axvspan(index - 0.48, index + 0.48, color="#FEE2E2", alpha=0.8)
            axis.text(index, 86, "TASK FAIL", ha="center", color="#B91C1C")
    axis.set_title("V4 scheduler activity and the high-jitter failure boundary")
    axis.set_ylabel("Recorded events / control steps")
    axis.set_xticks(x, labels, fontsize=8)
    axis.set_ylim(0, 92)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, metadata={"Software": "ActionStream"})
    plt.close(fig)


def render_content_montage(rows: list[dict[str, Any]], path: Path) -> None:
    lookup = indexed(rows)
    cases = (
        ("object", "high_jitter_v4", "Object / high jitter"),
        ("object", "outage_v4_requests_02_06", "Object / outages"),
    )
    runtimes = (
        ("lerobot_latest_only", "Latest-only"),
        ("actionstream_aligned", "Static aligned"),
        ("actionstream_adaptive", "Adaptive v4"),
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
    fig.suptitle("Same task, state, checkpoint, and network trace: final scene", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=180, metadata={"Software": "ActionStream"})
    plt.close(fig)


def render_failure_timeline(rows: list[dict[str, Any]], path: Path) -> None:
    lookup = indexed(rows)
    runtimes = (
        ("lerobot_latest_only", "Latest-only"),
        ("actionstream_aligned", "Static aligned"),
        ("actionstream_adaptive", "Adaptive v4"),
    )
    fig, axes = plt.subplots(3, 5, figsize=(18, 10.5))
    for row_index, (runtime, label) in enumerate(runtimes):
        episode = lookup[("object", "high_jitter_v4", runtime)]
        frames = sampled_video_frames(Path(str(episode["video_path"])))
        for column_index, frame in enumerate(frames):
            axis = axes[row_index, column_index]
            axis.imshow(frame)
            if column_index == 0:
                status = "SUCCESS" if episode["success"] else "FAIL"
                axis.set_ylabel(
                    f"{label}\n{status}, {episode['environment_steps']} steps",
                    color="#15803D" if episode["success"] else "#B91C1C",
                    fontsize=10,
                )
            axis.set_title(f"{column_index * 25}%")
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Object high-jitter paired timeline (state 32)", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=180, metadata={"Software": "ActionStream"})
    plt.close(fig)


def profile_summary(rows: list[dict[str, Any]], profile: str) -> dict[str, Any]:
    selected = [row for row in rows if row["delay_profile"] == profile]
    output: dict[str, Any] = {}
    for runtime in RUNTIMES[1:]:
        runtime_rows = [row for row in selected if row["runtime"] == runtime]
        output[runtime] = {
            "successes": sum(bool(row["success"]) for row in runtime_rows),
            "episodes": len(runtime_rows),
            "total_steps": sum(int(row["environment_steps"]) for row in runtime_rows),
        }
    return output


def format_cell(success: bool, steps: int) -> str:
    return f"{'PASS' if success else 'FAIL'} {steps}"


def build_report(summary: dict[str, Any], table: list[dict[str, Any]]) -> str:
    lines = [
        "# ActionStream Adaptive Queue-Slack v4 result",
        "",
        "**Verdict: NO-GO.** The predeclared 9/9 Adaptive success gate failed on Object/high-jitter. The outage mechanism worked and recovered a latest-only failure, but static aligned remained the stronger overall runtime.",
        "",
        "## Frozen state-32 canary",
        "",
        "`PASS/FAIL steps`; sync is a policy-capability reference, not a latency-runtime competitor.",
        "",
        "| Task | Profile | Sync | Latest-only | Static aligned | Adaptive v4 | Adaptive - latest | Adaptive - aligned |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "low_jitter_v4": "25–125 ms",
        "high_jitter_v4": "600–1100 ms",
        "outage_v4_requests_02_06": "425 / 1850 ms",
    }
    for row in table:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["suite"].title(),
                    labels[row["profile"]],
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
                    f"{row['adaptive_minus_aligned_steps']:+d}",
                )
            )
            + " |"
        )
    low = summary["profiles"]["low_jitter_v4"]
    high = summary["profiles"]["high_jitter_v4"]
    outage = summary["profiles"]["outage_v4_requests_02_06"]
    lines.extend(
        [
            "",
            "![Canary steps](canary_steps.png)",
            "",
            "## Findings",
            "",
            f"1. **Low jitter is not a win:** all three runtimes succeeded 3/3. Adaptive used {low['actionstream_adaptive']['total_steps']} total steps versus {low['lerobot_latest_only']['total_steps']} for latest-only (+5.3%) and exactly matched aligned's {low['actionstream_aligned']['total_steps']} steps.",
            f"2. **High jitter is the decisive regression:** Adaptive succeeded {high['actionstream_adaptive']['successes']}/3 and timed out on Object, while latest-only and aligned were both 3/3. Adaptive accumulated {high['actionstream_adaptive']['total_steps']} steps versus {high['lerobot_latest_only']['total_steps']} and {high['actionstream_aligned']['total_steps']}.",
            f"3. **Outage robustness improved relative to latest-only:** Adaptive was {outage['actionstream_adaptive']['successes']}/3 versus latest-only {outage['lerobot_latest_only']['successes']}/3, recovering the Object failure. Aligned was still 3/3 and faster in aggregate ({outage['actionstream_aligned']['total_steps']} versus Adaptive {outage['actionstream_adaptive']['total_steps']}).",
            "4. **The registered mechanism fired:** every Adaptive outage row observed both 1850 ms events, preserved a nonempty five-command reserve through rejected stale results, and recorded zero guard discards. Across the three rows there were 100 reserve-hold steps and 21 queue-preserving rejections.",
            "5. **Failure boundary:** Object/high-jitter had 25 reserve activations and 83 reserve-hold steps, versus 9/20 on Spatial and 6/8 on Goal. This supports an over-conservative reserve-fragmentation hypothesis, but one reset cannot prove causality.",
            "6. **Statistical boundary:** this is one paired state per task/profile. No IID-seed 95% CI or generalization claim is reported.",
            "",
            "![Scheduler mechanism](scheduler_mechanism.png)",
            "",
            "## Content-level paired visualization",
            "",
            "![Final frames](paired_content_final_frames.png)",
            "",
            "![High-jitter timeline](object_high_jitter_timeline.png)",
            "",
            "The videos use the same task, state 32, checkpoint, observation/action spaces, success predicate, and paired network trace within each row group.",
            "",
            "## Decision",
            "",
            "Do not open the v4 formal holdout and do not retune this selector. Preserve this exact candidate as `NO-GO`. The next candidate should replace the binary five-command freeze with a bounded reserve budget or deadline-aware release that can spend reserve when repeated holds stop making task progress; it requires a new selector ID and new state/trace split.",
            "",
            "Resume-safe claim: *Built and froze an online service-time/queue-slack scheduler for X-VLA; on three disjoint state-32 task families it preserved nonempty action reserves through all registered outages and recovered a latest-only Object failure, while a high-jitter Object timeout exposed an over-conservative reserve boundary.*",
            "",
            "Do not claim formal holdout superiority, universal latency robustness, Isaac Lab-Arena, or real-robot validation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--spatial-dir", type=Path, required=True)
    parser.add_argument("--goal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows((args.object_dir, args.spatial_dir, args.goal_dir))
    if len(rows) != 36:
        raise ValueError(f"Expected 36 rows, found {len(rows)}")
    lookup = indexed(rows)
    expected = {
        (suite, profile, runtime)
        for suite in SUITES
        for profile in PROFILES
        for runtime in RUNTIMES
    }
    if set(lookup) != expected:
        raise ValueError("Canary matrix is incomplete or contains extra rows")

    table = condition_rows(rows)
    adaptive = [row for row in rows if row["runtime"] == "actionstream_adaptive"]
    outage = [
        row
        for row in adaptive
        if row["delay_profile"] == "outage_v4_requests_02_06"
    ]
    sync_low = [
        row
        for row in rows
        if row["runtime"] == "sync_hold" and row["delay_profile"] == "low_jitter_v4"
    ]
    gate = {
        "sync_capability_3_of_3": len(sync_low) == 3 and all(row["success"] for row in sync_low),
        "adaptive_success_9_of_9": len(adaptive) == 9 and all(row["success"] for row in adaptive),
        "prefetch_all_9": all(int(row["adaptive_prefetch_requests"]) > 0 for row in adaptive),
        "outage_mechanism_all_3": all(
            int(row["adaptive_reserve_activations"]) > 0
            and int(row["adaptive_preserved_queue_rejections"]) > 0
            and int(row["adaptive_pending_actions_discarded_by_guard"]) == 0
            for row in outage
        ),
        "outage_ordinals_all_3": all(
            row["_observed_delays_ms"].get(2) == 1850
            and row["_observed_delays_ms"].get(6) == 1850
            for row in outage
        ),
        "artifacts_valid": True,
    }
    summary = {
        "schema_version": 1,
        "selector_id": SELECTOR_ID,
        "selector_sha256": SELECTOR_SHA256,
        "source_commit": SOURCE_COMMIT,
        "verdict": "GO" if all(gate.values()) else "NO-GO",
        "formal_holdout_opened": False,
        "gate": gate,
        "validation": {
            "rows": len(rows),
            "trace_hashes_verified": len(rows),
            "videos_decoded": len(rows),
            "decoded_frames": sum(int(row["_video"]["frames"]) for row in rows),
            "sync_successes": sum(row["success"] for row in rows if row["runtime"] == "sync_hold"),
            "latest_successes": sum(row["success"] for row in rows if row["runtime"] == "lerobot_latest_only"),
            "aligned_successes": sum(row["success"] for row in rows if row["runtime"] == "actionstream_aligned"),
            "adaptive_successes": sum(row["success"] for row in adaptive),
        },
        "profiles": {profile: profile_summary(rows, profile) for profile in PROFILES},
        "mechanism": {
            "prefetch_requests": sum(int(row["adaptive_prefetch_requests"]) for row in adaptive),
            "reserve_activations": sum(int(row["adaptive_reserve_activations"]) for row in adaptive),
            "reserve_hold_steps": sum(int(row["adaptive_reserve_hold_steps"]) for row in adaptive),
            "preserved_queue_rejections": sum(int(row["adaptive_preserved_queue_rejections"]) for row in adaptive),
            "guard_discards": sum(int(row["adaptive_pending_actions_discarded_by_guard"]) for row in adaptive),
            "outage_reserve_hold_steps": sum(int(row["adaptive_reserve_hold_steps"]) for row in outage),
            "outage_preserved_queue_rejections": sum(int(row["adaptive_preserved_queue_rejections"]) for row in outage),
        },
        "failure_boundary": {
            "condition": "Object/high_jitter_v4/actionstream_adaptive",
            "classification": "timeout_after_fragmented_reserve_holds",
            "reserve_activations": int(lookup[("object", "high_jitter_v4", "actionstream_adaptive")]["adaptive_reserve_activations"]),
            "reserve_hold_steps": int(lookup[("object", "high_jitter_v4", "actionstream_adaptive")]["adaptive_reserve_hold_steps"]),
            "causal_status": "trace-supported hypothesis, not identified from one reset",
        },
        "statistical_boundary": "One paired state per task/profile; no IID-seed confidence interval.",
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "main_table.csv", table)
    write_csv(
        output / "failure_taxonomy.csv",
        [
            {
                "suite": "object",
                "profile": "high_jitter_v4",
                "runtime": "actionstream_adaptive",
                "success": False,
                "steps": 300,
                "classification": "timeout_after_fragmented_reserve_holds",
                "reserve_activations": 25,
                "reserve_hold_steps": 83,
                "causal_status": "trace-supported hypothesis; paired video reviewed separately",
            }
        ],
    )
    render_steps(table, output / "canary_steps.png")
    render_mechanism(rows, output / "scheduler_mechanism.png")
    render_content_montage(rows, output / "paired_content_final_frames.png")
    render_failure_timeline(rows, output / "object_high_jitter_timeline.png")
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
