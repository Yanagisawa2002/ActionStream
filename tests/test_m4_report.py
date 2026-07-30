from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from actionstream.m4_analysis import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
from actionstream.m4_report import (
    _derive_claim_statuses,
    build_m4_report,
    write_m4_report,
)
from actionstream.results import EXPECTED_MODEL_REVISION


M4_COMMIT = "m4-commit"
M3_COMMIT = "m3-commit"
SELECTED_DELAY = 400


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _m3_summary(value: float) -> dict:
    return {"count": 30, "mean": value, "median": value}


def _m3_effect(value: float = 0.0) -> dict:
    summary = {"count": 30, "mean": value, "median": value}
    return {
        "paired_episode_count": 30,
        "reference": summary,
        "estimate": summary,
        "paired_mean_difference": 0.0,
        "paired_median_difference": 0.0,
        "paired_mean_difference_95pct_bootstrap_ci": [0.0, 0.0],
        "paired_relative_difference_percent": 0.0 if value != 0 else None,
        "paired_relative_difference_95pct_bootstrap_ci": (
            [0.0, 0.0] if value != 0 else None
        ),
        "relative_difference_available": value != 0,
    }


def _write_trace(path: Path, row: dict, *, bad_telemetry: bool = False) -> None:
    steps = int(row["environment_steps"])
    actions = np.ones((steps, 7), dtype=np.float32)
    actions[:, 0] += np.arange(steps, dtype=np.float32) / 100
    holds = int(row["queue_underrun_hold_steps"])
    hold_mask = np.zeros(steps, dtype=np.bool_)
    if holds:
        hold_mask[-holds:] = True
        for index in np.flatnonzero(hold_mask):
            actions[index] = actions[index - 1]
    if row["runtime_mode"] == "sync_hold":
        depth_before = np.maximum(
            30 - np.arange(steps, dtype=np.int32),
            0,
        )
    else:
        depth_before = np.full(steps, 3, dtype=np.int32)
        depth_before[hold_mask] = 0
    depth_after = np.maximum(depth_before - 1, 0).astype(np.int32)
    total_latency = float(row["effective_delivery_age_p50_steps"]) * 0.05
    injected_delay = row["injected_delay_ms"] / 1000
    inference_end = total_latency - injected_delay
    dispatch = (
        np.arange(steps, dtype=np.float64) * 0.05
        + total_latency
        + 0.5
    )
    event = {
        "observation_control_step": 0,
        "request_timestamp": 0.0,
        "start_timestamp": 0.01,
        "end_timestamp": inference_end,
        "delivery_timestamp": total_latency,
        "model_inference_latency_seconds": inference_end - 0.01,
        "published_before_episode_end": True,
        "observation_to_delivery_latency_seconds": total_latency,
        "effective_delivery_age_steps": row["effective_delivery_age_p50_steps"],
        "injected_delivery_delay_seconds": injected_delay,
        "queue_depth_at_request_steps": 0,
        "queue_headroom_at_request_steps": 0,
        "incoming_chunk_steps": 30,
        "chunk_accepted": True,
        "chunk_rejection_reason": None,
        "merge": {
            "control_step": 0,
            "replacement_action_index": 0,
            "accepted": True,
            "reason": "replaced",
            "age_steps": 0,
            "dropped_prefix_steps": 0,
            "fully_stale": False,
            "queue_length_before": 0,
            "queue_length_after": 30,
            "incoming_chunk_steps": 30,
            "stale_fraction": row["incoming_stale_fraction_mean"],
            "replacement_position_l2": row[
                "replacement_position_jump_mean_l2"
            ],
            "replacement_rotation_geodesic_radians": row[
                "replacement_rotation_jump_mean_radians"
            ],
            "replacement_gripper_switch": False,
        },
    }
    events = [event]
    if row["runtime_mode"] == "sync_hold" and holds:
        request_timestamp = float(dispatch[29]) + 0.001
        end_timestamp = request_timestamp + 0.11
        delivery_timestamp = end_timestamp + injected_delay
        events.append(
            {
                "observation_control_step": 30,
                "request_timestamp": request_timestamp,
                "start_timestamp": request_timestamp + 0.01,
                "end_timestamp": end_timestamp,
                "delivery_timestamp": delivery_timestamp,
                "model_inference_latency_seconds": 0.1,
                "published_before_episode_end": False,
                "observation_to_delivery_latency_seconds": (
                    delivery_timestamp - request_timestamp
                ),
                "effective_delivery_age_steps": (
                    delivery_timestamp - request_timestamp
                )
                / 0.05,
                "injected_delivery_delay_seconds": injected_delay,
                "queue_depth_at_request_steps": 0,
                "queue_headroom_at_request_steps": 0,
                "incoming_chunk_steps": 30,
                "chunk_accepted": False,
                "chunk_rejection_reason": "episode_ended",
            }
        )
    published = [
        item for item in events if item["published_before_episode_end"]
    ]
    stale_fractions = [
        item["merge"]["stale_fraction"]
        for item in events
        if "merge" in item
    ]
    row.update(
        {
            "inference_calls": len(events),
            "inference_latency_p50_seconds": float(
                np.percentile(
                    [item["model_inference_latency_seconds"] for item in events],
                    50,
                )
            ),
            "inference_latency_p95_seconds": float(
                np.percentile(
                    [item["model_inference_latency_seconds"] for item in events],
                    95,
                )
            ),
            "observation_to_delivery_p50_seconds": float(
                np.percentile(
                    [
                        item["observation_to_delivery_latency_seconds"]
                        for item in published
                    ],
                    50,
                )
            ),
            "observation_to_delivery_p95_seconds": float(
                np.percentile(
                    [
                        item["observation_to_delivery_latency_seconds"]
                        for item in published
                    ],
                    95,
                )
            ),
            "effective_delivery_age_p50_steps": float(
                np.percentile(
                    [item["effective_delivery_age_steps"] for item in published],
                    50,
                )
            ),
            "effective_delivery_age_p95_steps": float(
                np.percentile(
                    [item["effective_delivery_age_steps"] for item in published],
                    95,
                )
            ),
            "control_step_duration_p50_seconds": float(np.percentile(np.diff(dispatch), 50)),
            "control_step_duration_p95_seconds": float(np.percentile(np.diff(dispatch), 95)),
            "queue_depth_p50_steps": float(np.percentile(depth_before, 50)),
            "queue_depth_p95_steps": float(np.percentile(depth_before, 95)),
            "queue_depth_min_steps": int(depth_before.min()),
            "queue_depth_max_steps": int(depth_before.max()),
            "queue_headroom_at_request_p50_steps": 0.0,
            "queue_headroom_at_request_p95_steps": 0.0,
            "incoming_stale_fraction_mean": float(np.mean(stale_fractions)),
            "incoming_stale_fraction_p95": float(np.percentile(stale_fractions, 95)),
            "chunks_accepted": 1,
            "chunks_rejected": len(events) - 1,
            "chunks_rejected_after_episode": len(events) - 1,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        actions=actions,
        dispatch_timestamps=dispatch,
        inference_events_json=np.asarray(json.dumps(events)),
        queue_depth_before_action=(
            depth_before[:-1] if bad_telemetry else depth_before
        ),
        queue_depth_after_action=depth_after,
        queue_hold_mask=hold_mask,
    )


def _row(
    *,
    mode: str,
    delay: int,
    task: int,
    episode: int,
    trace_path: Path,
) -> dict:
    mode_offset = {"sync_hold": 2, "async_naive": 3, "async_aligned": 1}[mode]
    pressure = delay >= SELECTED_DELAY
    holds = (
        2
        if pressure and mode == "sync_hold"
        else 1
        if pressure and mode == "async_naive"
        else 0
    )
    steps = (
        32
        if pressure and mode == "sync_hold"
        else 5 + mode_offset + episode % 2
    )
    age = 21.0 if pressure else 8.0 if delay else 3.0
    return {
        "run_id": f"{mode}-delay{delay}-run",
        "git_commit": M4_COMMIT,
        "model_id": "lerobot/xvla-libero",
        "model_revision_sha": EXPECTED_MODEL_REVISION,
        "suite": "libero_object",
        "runtime_mode": mode,
        "injected_delay_ms": delay,
        "task_id": task,
        "episode_index": episode,
        "initial_state_index": 2 * episode,
        "seed": 142 + episode,
        "policy_rng_seed": 142 + episode,
        "policy_rng_reset_per_episode": True,
        "realtime_control": True,
        "controller_frequency_hz": 20.0,
        "success": True,
        "environment_steps": steps,
        "wall_clock_episode_seconds": steps / 20 + delay / 1000,
        "queue_underrun_hold_steps": holds,
        "stale_chunks_discarded": 0,
        "effective_delivery_age_p50_steps": age,
        "action_discontinuity_mean_l2": 0.01 * mode_offset,
        "incoming_stale_fraction_mean": 0.2 if mode == "async_aligned" else 0.0,
        "chunks_accepted": 1,
        "chunks_rejected": 0,
        "chunks_rejected_after_episode": 0,
        "replacement_position_jump_mean_l2": 0.01 * mode_offset,
        "replacement_rotation_jump_mean_radians": 0.02 * mode_offset,
        "replacement_gripper_switches": 0,
        "chunk_size_steps": 30,
        "replan_interval_steps": 30 if mode == "sync_hold" else 10,
        "nominal_queue_headroom_steps": 0 if mode == "sync_hold" else 20,
        "queue_depth_min_steps": 0 if holds else 3,
        "queue_depth_max_steps": 3,
        "old_episode_chunks_rejected": 0,
        "old_episode_results_discarded": 0,
        "action_trace_path": str(trace_path),
    }


def _write_condition(
    root: Path,
    *,
    mode: str,
    delay: int,
    tasks: tuple[int, ...],
    episodes: int,
    corrupt_first_trace: bool = False,
) -> Path:
    directory = root / f"{mode}_delay{delay}"
    episode_path = directory / "episodes.jsonl"
    rows = []
    for task in tasks:
        for episode in range(episodes):
            trace_path = (
                directory
                / "traces"
                / f"{mode}_delay{delay}_task{task}_episode{episode}.npz"
            )
            row = _row(
                mode=mode,
                delay=delay,
                task=task,
                episode=episode,
                trace_path=trace_path,
            )
            _write_trace(
                trace_path,
                row,
                bad_telemetry=corrupt_first_trace and not rows,
            )
            rows.append(row)
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    episode_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return episode_path


def _fixture(tmp_path: Path, *, corrupt_pressure_trace: bool = False) -> dict:
    m3_path = tmp_path / "paired_analysis.json"
    _write_json(
        m3_path,
        {
            "analysis_unit": "episode",
            "bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "resamples": BOOTSTRAP_RESAMPLES,
            },
            "m3_validation": {
                "passed": True,
                "episode_count": 180,
                "action_trace_count_verified": 180,
                "total_final_7d_actions_verified": 29073,
                "git_commit": M3_COMMIT,
                "model_revision_sha": EXPECTED_MODEL_REVISION,
            },
            "condition_summaries": [
                {
                    "runtime_mode": mode,
                    "injected_delay_ms": delay,
                    "semantic_label": (
                        "blocking evaluator baseline" if mode == "sync" else mode
                    ),
                    "overall": {
                        "success": _m3_summary(1.0),
                        "environment_steps": _m3_summary(100.0),
                        "wall_clock_episode_seconds": _m3_summary(5.0),
                        "observation_to_delivery_p50_seconds": _m3_summary(0.2),
                        "action_discontinuity_mean_l2": _m3_summary(0.01),
                    },
                }
                for mode in ("sync", "async_naive", "async_aligned")
                for delay in (0, 200)
            ],
            "paired_comparisons": [
                {
                    "comparison_id": "async_aligned_minus_async_naive_delay0",
                    "overall": {
                        "success": _m3_effect(1.0),
                        "environment_steps": _m3_effect(100.0),
                        "wall_clock_episode_seconds": _m3_effect(5.0),
                        "observation_to_delivery_p50_seconds": _m3_effect(0.2),
                        "action_discontinuity_mean_l2": _m3_effect(0.01),
                    },
                }
            ],
            "historical_replacement_boundary_audit": {
                "replacement_boundary_indices_available": False
            },
            "interpretation_boundary": "Synthetic M3 success ceiling.",
        },
    )

    plan_path = tmp_path / "calibration_plan.json"
    _write_json(
        plan_path,
        {
            "status": "planned",
            "git_commit": M4_COMMIT,
            "runtime_configuration": {
                "chunk_size": 30,
                "replan_interval_steps": 10,
                "queue_headroom_steps": 20,
                "queue_headroom_seconds_at_median_step": 1.0,
                "queue_headroom_seconds_at_p95_step": 1.2,
            },
            "measured_m3_timings": {
                "control_step_seconds": {
                    "mean": 0.051,
                    "median": 0.05,
                    "p95": 0.06,
                },
                "model_inference_seconds": {
                    "mean": 0.11,
                    "median": 0.10,
                    "p95": 0.14,
                },
                "observation_to_delivery_seconds": {
                    "mean": 0.13,
                    "median": 0.12,
                    "p95": 0.17,
                },
                "observation_to_delivery_by_delay_ms": {
                    "0": {
                        "mean": 0.11,
                        "median": 0.10,
                        "p95": 0.13,
                    },
                    "200": {
                        "mean": 0.31,
                        "median": 0.30,
                        "p95": 0.34,
                    },
                },
            },
            "candidate_derivation": {
                "initial_candidates": [
                    {
                        "injected_delay_ms": SELECTED_DELAY,
                        "target_delivery_age_steps": 20.0,
                    }
                ],
                "extend_once_if_needed": {"injected_delay_ms": 1450},
            },
            "calibration_protocol": {
                "task_ids": [3],
                "episodes_per_condition": 3,
                "initial_state_indices": [0, 2, 4],
                "seeds": [142, 143, 144],
                "modes": ["async_naive", "async_aligned"],
                "selection_rule": (
                    "Choose the smallest tested delay where queue hold or fully stale "
                    "events occur; otherwise test one extension."
                ),
            },
        },
    )

    sync_paths = [
        _write_condition(
            tmp_path / "sync_hold",
            mode="sync_hold",
            delay=delay,
            tasks=(0, 1, 2),
            episodes=10,
        )
        for delay in (0, 200)
    ]
    calibration_paths = [
        _write_condition(
            tmp_path / "calibration",
            mode=mode,
            delay=SELECTED_DELAY,
            tasks=(3,),
            episodes=3,
        )
        for mode in ("async_naive", "async_aligned")
    ]
    pressure_paths = []
    for index, mode in enumerate(("sync_hold", "async_naive", "async_aligned")):
        pressure_paths.append(
            _write_condition(
                tmp_path / "pressure",
                mode=mode,
                delay=SELECTED_DELAY,
                tasks=(0, 1, 2),
                episodes=10,
                corrupt_first_trace=corrupt_pressure_trace and index == 0,
            )
        )

    selection_path = tmp_path / "selection.json"
    _write_json(
        selection_path,
        {
            "status": "selected",
            "git_commit": M4_COMMIT,
            "plan_sha256": _sha256(plan_path),
            "tested_delays_ms": [SELECTED_DELAY],
            "queue_headroom_steps": 20,
            "delay_decisions": [
                {
                    "injected_delay_ms": SELECTED_DELAY,
                    "pooled_effective_delivery_age_steps": {
                        "count": 6,
                        "mean": 21.0,
                        "median": 21.0,
                        "p95": 21.0,
                    },
                    "queue_hold_steps_total": 3,
                    "fully_stale_chunks_total": 0,
                    "triggers": {
                        "median_age_reaches_headroom": True,
                        "nonzero_queue_holds": True,
                        "nonzero_fully_stale_chunks": False,
                    },
                    "qualifies_as_pressure": True,
                }
            ],
            "selected_pressure_delay_ms": SELECTED_DELAY,
            "selection_rule_frozen_before_full_pressure_run": True,
        },
    )
    return {
        "m3_paired_path": m3_path,
        "sync_hold_paths": sync_paths,
        "calibration_plan_path": plan_path,
        "calibration_selection_path": selection_path,
        "calibration_paths": calibration_paths,
        "pressure_paths": pressure_paths,
    }


def _mutate_first_trace(
    episode_path: Path,
    mutator,
) -> None:
    row = json.loads(episode_path.read_text(encoding="utf-8").splitlines()[0])
    trace_path = Path(row["action_trace_path"])
    with np.load(trace_path, allow_pickle=False) as trace:
        payload = {field: np.asarray(trace[field]) for field in trace.files}
    events = json.loads(str(payload["inference_events_json"].item()))
    mutator(payload, events)
    payload["inference_events_json"] = np.asarray(json.dumps(events))
    np.savez_compressed(trace_path, **payload)


def test_complete_synthetic_m4_package_validates_and_reports(tmp_path) -> None:
    arguments = _fixture(tmp_path)
    output = tmp_path / "report"

    report = write_m4_report(**arguments, output_dir=output)

    assert report["validation"]["passed"]
    assert report["validation"]["sync_hold_matrix"]["episode_count"] == 60
    assert report["validation"]["calibration_matrix"]["episode_count"] == 6
    assert report["validation"]["pressure_matrix"]["episode_count"] == 90
    assert report["calibration"]["selected_pressure_delay_ms"] == SELECTED_DELAY
    assert len(report["m4_paired_comparisons"]) == 4
    aligned_vs_naive = next(
        comparison
        for comparison in report["m4_paired_comparisons"]
        if comparison["comparison_id"]
        == "pressure_async_aligned_minus_async_naive"
    )
    assert aligned_vs_naive["per_task"]["2"]["environment_steps"][
        "paired_episode_count"
    ] == 10
    assert aligned_vs_naive["overall"]["queue_underrun_hold_steps"][
        "paired_mean_difference"
    ] == pytest.approx(-1)
    assert report["claim_statuses"]["success"]["status"] == (
        "unsupported_success_ceiling"
    )
    assert report["claim_statuses"]["underrun"]["status"] == "supported"
    assert (
        report["claim_statuses"]["underrun"]["cross_comparator_status"]
        == "supported"
    )
    assert report["claim_statuses"]["efficiency"]["status"] == "supported"
    assert report["calibration"]["measured_m3_timings"]["control_step_seconds"][
        "p95"
    ] == pytest.approx(0.06)
    assert (output / "m4_report.json").is_file()
    markdown = (output / "m4_report.md").read_text(encoding="utf-8")
    assert "Queue-pressure calibration" in markdown
    assert "0 ms observation-to-delivery median 0.100000 s" in markdown
    assert "success ceiling" in markdown.lower()
    section_titles = [
        "## 1. M3 validated results",
        "## 2. Paired statistical analysis",
        "## 3. Blocking sync versus sync_hold semantics",
        "## 4. Queue-pressure calibration",
        "## 5. Full selected-pressure results",
        "## 6. Supported, unsupported, and still-untested claims",
    ]
    assert [markdown.index(title) for title in section_titles] == sorted(
        markdown.index(title) for title in section_titles
    )
    assert "async_aligned_minus_async_naive_delay0" in markdown


def test_pressure_telemetry_length_mismatch_is_rejected(tmp_path) -> None:
    arguments = _fixture(tmp_path, corrupt_pressure_trace=True)

    with pytest.raises(ValueError, match="queue_depth_before_action shape"):
        build_m4_report(**arguments)


def test_sync_hold_changed_hold_command_is_rejected(tmp_path) -> None:
    arguments = _fixture(tmp_path)

    def corrupt(payload, events) -> None:
        actions = payload["actions"].copy()
        actions[-1, 0] += 1.0
        payload["actions"] = actions

    _mutate_first_trace(arguments["pressure_paths"][0], corrupt)

    with pytest.raises(ValueError, match="invalid sync_hold command"):
        build_m4_report(**arguments)


def test_sync_hold_nonzero_request_headroom_is_rejected(tmp_path) -> None:
    arguments = _fixture(tmp_path)
    episode_path = arguments["pressure_paths"][0]

    def corrupt(payload, events) -> None:
        events[-1]["queue_headroom_at_request_steps"] = 1

    _mutate_first_trace(episode_path, corrupt)
    rows = [
        json.loads(line)
        for line in episode_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["queue_headroom_at_request_p50_steps"] = 0.5
    rows[0]["queue_headroom_at_request_p95_steps"] = 0.95
    episode_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requested before exhaustion"):
        build_m4_report(**arguments)


def test_chunk_and_merge_acceptance_mismatch_is_rejected(tmp_path) -> None:
    arguments = _fixture(tmp_path)

    def corrupt(payload, events) -> None:
        events[0]["chunk_accepted"] = False
        events[0]["chunk_rejection_reason"] = "fully_stale"

    _mutate_first_trace(arguments["pressure_paths"][1], corrupt)

    with pytest.raises(ValueError, match="chunk/merge acceptance mismatch"):
        build_m4_report(**arguments)


def test_trace_percentile_and_episode_row_must_reconcile(tmp_path) -> None:
    arguments = _fixture(tmp_path)
    episode_path = arguments["pressure_paths"][2]
    rows = [
        json.loads(line)
        for line in episode_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["queue_depth_p50_steps"] += 1.0
    episode_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="queue_depth_p50_steps"):
        build_m4_report(**arguments)


def test_selection_must_match_recomputed_smallest_pressure_delay(tmp_path) -> None:
    arguments = _fixture(tmp_path)
    selection_path = arguments["calibration_selection_path"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["selected_pressure_delay_ms"] = 450
    _write_json(selection_path, selection)

    with pytest.raises(ValueError, match="Selected pressure delay"):
        build_m4_report(**arguments)


def test_m4_evidence_commit_mismatch_is_rejected(tmp_path) -> None:
    arguments = _fixture(tmp_path)
    for pressure_path in arguments["pressure_paths"]:
        rows = [
            json.loads(line)
            for line in pressure_path.read_text(encoding="utf-8").splitlines()
        ]
        for row in rows:
            row["git_commit"] = "different-commit"
        pressure_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="M4 evidence commit mismatch"):
        build_m4_report(**arguments)


def test_claim_status_reports_mixed_underrun_evidence() -> None:
    def metric(difference: float, low: float, high: float) -> dict:
        return {
            "paired_mean_difference": difference,
            "paired_mean_difference_95pct_bootstrap_ci": [low, high],
        }

    claims = _derive_claim_statuses(
        {
            "comparison_id": "pressure_async_aligned_minus_async_naive",
            "overall": {
                "success": metric(0.0, 0.0, 0.0),
                "queue_underrun_hold_steps": metric(-2.0, -3.0, -1.0),
                "stale_chunks_discarded": metric(1.0, 0.5, 1.5),
                "environment_steps": metric(-5.0, -8.0, -2.0),
                "wall_clock_episode_seconds": metric(-0.2, -0.4, -0.1),
            },
        },
        success_ceiling=True,
    )

    assert claims["underrun"]["status"] == "mixed_evidence"
