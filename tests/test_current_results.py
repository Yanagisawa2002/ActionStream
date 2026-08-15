from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from actionstream.current_results import (
    build_analysis,
    read_episode_rows,
    validate_episode_rows,
    write_analysis,
)


def _row(runtime: str, profile: str, episode: int) -> dict[str, object]:
    offset = {
        "lerobot_weighted_average": 4,
        "lerobot_latest_only": 2,
        "actionstream_aligned": 0,
    }[runtime]
    return {
        "status": "completed",
        "model_key": "xvla",
        "model_id": "lerobot/xvla-libero",
        "model_revision": "revision",
        "control_mode": "absolute",
        "runtime": runtime,
        "delay_profile": profile,
        "delay_trace_sha256": "b" * 64,
        "task_id": 0,
        "episode_index": episode,
        "initial_state_index": 2 * episode,
        "seed": 100 + episode,
        "success": offset < 4,
        "environment_steps": 100 + offset + episode,
        "hold_fraction": offset / 100,
        "dropped_prefix_steps": offset,
        "delivery_latency_p50_seconds": 0.1 + offset / 100,
        "delivery_latency_p95_seconds": 0.2 + offset / 100,
        "inference_latency_p50_seconds": 0.05,
        "inference_latency_p95_seconds": 0.06,
        "action_discontinuity_mean_l2": 0.01 + offset / 100,
        "action_discontinuity_max_l2": 0.02 + offset / 100,
        "action_acceleration_max_l2": 0.03 + offset / 100,
        "peak_cuda_memory_mib": 1000,
        "wall_clock_episode_seconds": 5.0,
        "controller_frequency_hz": 20.0,
        "chunk_size": 30,
        "request_interval_steps": 10,
        "source_commit": "a" * 40,
        "trace_path": f"/remote/trace_{runtime}_{profile}_{episode}.json",
        "video_path": None,
        "video_sha256": None,
    }


def _write_fixture(root: Path) -> Path:
    traces = root / "traces"
    traces.mkdir(parents=True)
    rows = []
    for profile in ("fixed_0000", "fixed_0500"):
        for runtime in (
            "lerobot_weighted_average",
            "lerobot_latest_only",
            "actionstream_aligned",
        ):
            for episode in range(3):
                row = _row(runtime, profile, episode)
                trace = traces / Path(str(row["trace_path"])).name
                trace.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "actions": [
                                {"action": [0.0] * 7}
                                for _ in range(int(row["environment_steps"]))
                            ],
                            "inference_events": [],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                row["trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
                rows.append(row)
    path = root / "episodes.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_current_results_validate_pair_and_build_effects(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    rows = read_episode_rows([path])
    validation = validate_episode_rows(rows)
    report = build_analysis(rows)

    assert validation["episode_count"] == 18
    assert validation["trace_count_verified"] == 18
    assert validation["final_7d_action_count_verified"] > 1_800
    comparison = next(
        item
        for item in report["paired_comparisons"]
        if item["comparison_type"] == "runtime"
        and item["delay_profile"] == "fixed_0000"
        and item["estimate_runtime"] == "actionstream_aligned"
        and item["reference_runtime"] == "lerobot_weighted_average"
    )
    assert comparison["metrics"]["environment_steps"]["paired_mean_difference"] == -4
    assert comparison["metrics"]["success"]["paired_mean_difference"] == 1


def test_current_results_reject_unpaired_runtime_cell(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    rows = read_episode_rows([path])
    rows.pop()
    with pytest.raises(ValueError, match="Unpaired runtime cells"):
        validate_episode_rows(rows, verify_artifacts=False)


def test_current_results_reject_pair_invariant_drift(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    rows = read_episode_rows([path])
    rows[0]["delay_trace_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="Pair invariant mismatch.*delay_trace_sha256"):
        validate_episode_rows(rows, verify_artifacts=False)


def test_current_results_write_raw_table_and_reports(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path / "input")
    output = tmp_path / "output"
    report = write_analysis([path], output_dir=output)

    assert report["validation"]["episode_count"] == 18
    assert len((output / "raw_episodes.csv").read_text().splitlines()) == 19
    assert (output / "paired_analysis.json").is_file()
    assert "Raw condition table" in (output / "paired_analysis.md").read_text()
