from __future__ import annotations

import json
from pathlib import Path

import pytest

from actionstream.m5_analysis import (
    ALIGNED_NO_SHIFT,
    ALIGNED_SHIFT,
    GATED_NO_SHIFT,
    GATED_SHIFT,
    build_m5_analysis,
)
from actionstream.m5_protocol import canonical_sha256, file_sha256, write_seed_manifests
from actionstream.m5_report import (
    _condition_summaries,
    _validate_protocol_calibration_decisions,
    build_m5_report,
    write_m5_report,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _base_inputs(tmp_path: Path) -> dict:
    config = {
        "schema_version": 1,
        "milestone": "M5-G0",
        "runtime": {
            "control_frequency_hz": 20.0,
            "injected_delay_ms": 950,
        },
        "detector": {"translation_threshold_m": 0.010},
        "perturbation": {"axis_xyz": [-1.0, 0.0, 0.0]},
    }
    audit = {
        "schema_version": 1,
        "milestone": "M5-G0",
        "m4_baseline_intact": True,
        "selected_task": {
            "task_id": 0,
            "instruction": "pick up the alphabet soup and place it in the basket",
            "moved_entity": "basket_1",
            "simulator_body": "basket_1_main",
        },
        "known_limitations": ["Synthetic fixture; no simulator was launched."],
    }
    manifest_dir = tmp_path / "protocol"
    manifests = write_seed_manifests(manifest_dir)
    return {
        "config_path": _write_json(tmp_path / "config.json", config),
        "task_audit_path": _write_json(tmp_path / "task_audit.json", audit),
        "seed_manifest_paths": manifests,
    }


def _decision(path: Path, *, status: str) -> Path:
    if status == "selected":
        core = {
            "schema_version": 1,
            "milestone": "M5-G0",
            "status": "selected",
            "task_id": 0,
            "selected_displacement_magnitude_mm": 50,
            "tested_magnitudes_mm": [50],
            "evaluations": [
                {
                    "task_id": 0,
                    "displacement_magnitude_mm": 50,
                    "pair_count": 5,
                    "aligned_success_count": 3,
                    "gate_success_count": 5,
                    "gate_only_success_count": 2,
                    "all_perturbations_physically_valid": True,
                    "paired_shift_parameters_identical": True,
                    "qualifies": True,
                }
            ],
        }
    else:
        core = {
            "schema_version": 1,
            "milestone": "M5-G0",
            "status": "candidate_no_go",
            "task_id": 2,
            "selected_displacement_magnitude_mm": None,
            "tested_magnitudes_mm": [50, 30],
            "evaluations": [
                {
                    "task_id": 2,
                    "displacement_magnitude_mm": 30,
                    "pair_count": 5,
                    "aligned_success_count": 1,
                    "gate_success_count": 2,
                    "gate_only_success_count": 1,
                    "all_perturbations_physically_valid": True,
                    "paired_shift_parameters_identical": True,
                    "qualifies": False,
                }
            ],
            "reason": "Fresh post-shift policy solved fewer than 4/5 pairs",
        }
    return _write_json(
        path,
        {**core, "selection_sha256": canonical_sha256(core)},
    )


def test_calibration_decisions_must_match_frozen_protocol(tmp_path: Path) -> None:
    decision_path = _decision(tmp_path / "decision.json", status="no_go")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    protocol = {
        "calibration_decisions": [
            {
                "path": str(decision_path.resolve()),
                "sha256": file_sha256(decision_path),
                "selection_sha256": decision["selection_sha256"],
                "task_id": decision["task_id"],
                "status": decision["status"],
            }
        ]
    }

    _validate_protocol_calibration_decisions(
        protocol,
        [decision_path],
        [decision],
    )

    replacement_path = _decision(
        tmp_path / "replacement.json",
        status="no_go",
    )
    with pytest.raises(ValueError, match="does not match the frozen protocol"):
        _validate_protocol_calibration_decisions(
            protocol,
            [replacement_path],
            [json.loads(replacement_path.read_text(encoding="utf-8"))],
        )


def _episode(
    condition: str,
    pair_index: int,
    *,
    success: bool,
    stale_duration: float,
) -> dict:
    shifted = condition in {ALIGNED_SHIFT, GATED_SHIFT}
    gated = condition in {GATED_SHIFT, GATED_NO_SHIFT}
    seed = (53001 if shifted else 52001) + pair_index
    state = (20 if shifted else 10) + pair_index
    return {
        "episode_id": f"{condition}-seed{seed}",
        "condition": condition,
        "phase": "sealed" if shifted else "no_shift",
        "task_id": 0,
        "seed": seed,
        "initial_state_index": state,
        "task_instruction": "pick up the alphabet soup and place it in the basket",
        "moved_entity": "basket_1",
        "initial_source_entity_pose": {"position": [0.0, 0.0, 0.0]},
        "initial_moved_entity_pose": {"position": [0.0, 0.25, 0.0]},
        "policy_rng_seed": seed,
        "model_id": "lerobot/xvla-libero",
        "model_revision_sha": "frozen-revision",
        "suite": "libero_object",
        "implementation_source_sha256": "source-sha",
        "experiment_config_sha256": "config-sha",
        "seed_manifest_sha256": "manifest-sha",
        "protocol_decision_sha256": "protocol-sha",
        "no_shift_decision_sha256": "no-shift-sha" if shifted else None,
        "injected_delay_ms": 950,
        "controller_frequency_hz": 20.0,
        "chunk_size_steps": 30,
        "replan_interval_steps": 10,
        "nominal_queue_headroom_steps": 20,
        "scheduled_shift_step": 19 if shifted else None,
        "displacement_magnitude_mm": 50 if shifted else 0,
        "requested_displacement_xy_m": [-0.05, 0.0] if shifted else None,
        "achieved_displacement_xyz_m": ([-0.05, 0.0, 0.0] if shifted else None),
        "success": success,
        "stale_action_duration_seconds": stale_duration,
        "stale_action_steps": int(round(stale_duration * 20.0)),
        "environment_steps": 100 + pair_index + int(gated),
        "simulated_completion_time_seconds": 5.0 + pair_index / 20.0,
        "wall_clock_episode_seconds": 4.0 + pair_index / 25.0,
        "time_from_shift_to_gate_trigger_seconds": (
            0.02 if shifted and gated else None
        ),
        "time_from_shift_to_first_fresh_action_seconds": 1.1 if shifted else None,
        "time_from_shift_to_queue_clear_seconds": 0.05 if shifted else None,
        "perturbation_valid": True,
        "invalid_reason": None,
        "false_gate_count": 0,
        "scene_gate_trigger_count": int(shifted and gated),
        "queue_invalidation_count": int(shifted and gated),
        "shift_step": 19 if shifted else None,
    }


def _full_go_episodes() -> list[dict]:
    rows: list[dict] = []
    for pair_index in range(30):
        aligned_duration = 1.0 + pair_index / 100.0
        rows.extend(
            [
                _episode(
                    ALIGNED_SHIFT,
                    pair_index,
                    success=pair_index < 18,
                    stale_duration=aligned_duration,
                ),
                _episode(
                    GATED_SHIFT,
                    pair_index,
                    success=pair_index < 24,
                    stale_duration=0.2 * aligned_duration,
                ),
            ]
        )
    for pair_index in range(10):
        rows.extend(
            [
                _episode(
                    ALIGNED_NO_SHIFT,
                    pair_index,
                    success=True,
                    stale_duration=0.0,
                ),
                _episode(
                    GATED_NO_SHIFT,
                    pair_index,
                    success=True,
                    stale_duration=0.0,
                ),
            ]
        )
    return rows


def _action(
    episode: dict,
    *,
    execution_step: int | None,
    observation_step: int,
    result_step: int,
    observation_epoch: int,
    current_epoch: int,
    discarded: bool = False,
    gate_triggered: bool = False,
) -> dict:
    row = {
        "record_type": "discarded" if discarded else "executed",
        "episode_id": episode["episode_id"],
        "condition": episode["condition"],
        "task_id": 0,
        "seed": episode["seed"],
        "initial_state_index": episode["initial_state_index"],
        "request_id": f"request-{observation_epoch}",
        "request_generation_id": observation_epoch,
        "observation_control_step": observation_step,
        "observation_step": observation_step,
        "observation_timestamp": observation_step / 20.0,
        "world_epoch_at_observation": observation_epoch,
        "entity_pose_at_observation": {
            "position": [0.0 - 0.05 * observation_epoch, 0.2, 0.1],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "policy_result_arrival_step": result_step,
        "policy_result_arrival_timestamp": result_step / 20.0,
        "chunk_action_index": 0,
        "queue_insertion_step": result_step,
        "discarded": discarded,
        "gate_triggered": gate_triggered,
        "stale": observation_epoch < current_epoch,
    }
    if discarded:
        row.update(
            {
                "queue_execution_step": None,
                "current_world_epoch_at_execution": None,
                "discard_reason": "scene_invalidation",
                "discard_step": 20,
                "current_world_epoch_at_discard": current_epoch,
                "queue_depth_before_invalidation": 10,
                "queue_depth_after_invalidation": 0,
            }
        )
    else:
        row.update(
            {
                "queue_execution_step": execution_step,
                "current_world_epoch_at_execution": current_epoch,
                "discard_reason": None,
                "queue_depth_before_action": 10,
                "queue_depth_after_action": 9,
            }
        )
    return row


def _representative_actions(episodes: list[dict]) -> list[dict]:
    actions: list[dict] = []
    for condition in (ALIGNED_SHIFT, GATED_SHIFT):
        episode = next(
            row
            for row in episodes
            if row["condition"] == condition and row["seed"] == 53019
        )
        if condition == ALIGNED_SHIFT:
            actions.extend(
                [
                    _action(
                        episode,
                        execution_step=12,
                        observation_step=10,
                        result_step=31,
                        observation_epoch=0,
                        current_epoch=0,
                    ),
                    _action(
                        episode,
                        execution_step=20,
                        observation_step=10,
                        result_step=31,
                        observation_epoch=0,
                        current_epoch=1,
                    ),
                ]
            )
        else:
            actions.extend(
                [
                    _action(
                        episode,
                        execution_step=None,
                        observation_step=10,
                        result_step=31,
                        observation_epoch=0,
                        current_epoch=1,
                        discarded=True,
                        gate_triggered=True,
                    ),
                    _action(
                        episode,
                        execution_step=41,
                        observation_step=20,
                        result_step=40,
                        observation_epoch=1,
                        current_epoch=1,
                    ),
                ]
            )
    return actions


def test_full_go_report_copies_reviewed_statistics_and_writes_all_figures(
    tmp_path: Path,
) -> None:
    inputs = _base_inputs(tmp_path)
    decision = _decision(tmp_path / "calibration.json", status="selected")
    episodes = _full_go_episodes()
    episode_path = _write_jsonl(tmp_path / "episodes.jsonl", episodes)
    action_path = _write_jsonl(
        tmp_path / "actions.jsonl", _representative_actions(episodes)
    )
    representative = next(
        row
        for row in episodes
        if row["condition"] == GATED_SHIFT and row["seed"] == 53019
    )
    event_path = _write_jsonl(
        tmp_path / "events.jsonl",
        [
            {
                "episode_id": representative["episode_id"],
                "condition": GATED_SHIFT,
                "seed": 53019,
                "event_type": "perturbation_applied",
                "control_step": 19,
            },
            {
                "episode_id": representative["episode_id"],
                "condition": GATED_SHIFT,
                "seed": 53019,
                "event_type": "scene_gate_triggered",
                "control_step": 20,
            },
        ],
    )
    statistics = build_m5_analysis(
        episodes,
        m4_baseline_intact=True,
        source_validation_passed=True,
    )
    statistics_path = _write_json(tmp_path / "statistics.json", statistics)
    output = tmp_path / "report"

    report = write_m5_report(
        **inputs,
        calibration_decision_paths=[decision],
        episode_paths=[episode_path],
        action_paths=[action_path],
        event_paths=[event_path],
        statistics_path=statistics_path,
        output_root=output,
    )

    assert report["acceptance"]["classification"]["label"] == "FULL GO"
    assert report["sealed_evaluation"]["status"] == "complete"
    assert report["representative_pair"]["seed"] == 53019
    assert report["action_provenance_validation"]["stale_executed_action_count"] == 1
    assert json.loads((output / "statistical_tests.json").read_text()) == statistics
    for name in (
        "aggregate_summary.json",
        "statistical_tests.json",
        "acceptance.json",
        "m5_g0_report.md",
        "plots/representative_paired_timeline.png",
        "plots/representative_paired_timeline.pdf",
        "plots/mechanism_outcome.png",
        "plots/mechanism_outcome.pdf",
    ):
        path = output / name
        assert path.is_file()
        assert path.stat().st_size > 100
    assert (
        (output / "plots/representative_paired_timeline.png")
        .read_bytes()
        .startswith(b"\x89PNG")
    )
    assert (output / "plots/mechanism_outcome.pdf").read_bytes().startswith(b"%PDF")
    markdown = (output / "m5_g0_report.md").read_text(encoding="utf-8")
    assert "Temporal staleness is handled by ActionStream" in markdown
    assert "semantic scene invalidation" in markdown
    assert "`world_epoch` is evaluator-only" in markdown
    assert "exact two-sided McNemar" in markdown
    assert "Paired success-difference bootstrap 95% CI" in markdown
    assert "Stale-action duration mean/median" in markdown
    assert "Paired stale-duration difference" in markdown
    assert "Shift-to-first-fresh-action mean/median" in markdown
    assert "Shift-to-queue-clear latency" in markdown
    assert "Environment steps mean/median" in markdown
    assert "Simulated completion time mean/median" in markdown
    assert "Measured wall-clock episode time mean/median" in markdown
    assert "Invalid/excluded evidence" in markdown
    assert "No FoundationPose" in markdown
    assert all(source["sha256"] for source in report["sources"])


def test_calibration_no_go_with_no_sealed_rows_is_an_honest_hard_stop(
    tmp_path: Path,
) -> None:
    inputs = _base_inputs(tmp_path)
    decision = _decision(tmp_path / "calibration_no_go.json", status="no_go")
    output = tmp_path / "report"
    statistics = build_m5_analysis(
        [],
        m4_baseline_intact=True,
        source_validation_passed=True,
        calibration_no_go_reason="Fresh post-shift policy solved fewer than 4/5 pairs",
    )
    statistics_path = _write_json(tmp_path / "no_go_statistics.json", statistics)

    report = write_m5_report(
        **inputs,
        calibration_decision_paths=[decision],
        statistics_path=statistics_path,
        output_root=output,
    )

    assert report["acceptance"]["classification"]["label"] == "NO-GO"
    assert report["sealed_evaluation"] == {
        "status": "not_run_due_to_calibration_hard_stop",
        "valid_pair_count": 0,
        "required_pair_count": 30,
        "reason": "Fresh post-shift policy solved fewer than 4/5 pairs",
    }
    assert report["statistics"]["shift_comparison"] is None
    markdown = (output / "m5_g0_report.md").read_text(encoding="utf-8")
    assert "not_run_due_to_calibration_hard_stop" in markdown
    assert "did not authorize sealed rows" in markdown
    assert (
        "| Task | Shift | Pairs | Aligned success | Oracle-gated success |" in markdown
    )
    assert "| 2 | 30 mm | 5 | 1/5 | 2/5 | 1/5 | True | False |" in markdown
    assert "Not run because calibration did not authorize" in markdown
    for name in (
        "plots/representative_paired_timeline.png",
        "plots/representative_paired_timeline.pdf",
        "plots/mechanism_outcome.png",
        "plots/mechanism_outcome.pdf",
    ):
        assert (output / name).stat().st_size > 100


def test_manifest_hash_tampering_is_rejected(tmp_path: Path) -> None:
    inputs = _base_inputs(tmp_path)
    decision = _decision(tmp_path / "calibration_no_go.json", status="no_go")
    manifest = Path(inputs["seed_manifest_paths"][0])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["pairs"][0]["seed"] += 100
    _write_json(manifest, payload)

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        build_m5_report(
            **inputs,
            calibration_decision_paths=[decision],
        )


def test_action_stale_label_must_match_world_epochs(tmp_path: Path) -> None:
    inputs = _base_inputs(tmp_path)
    decision = _decision(tmp_path / "calibration_no_go.json", status="no_go")
    action = {
        "condition": ALIGNED_SHIFT,
        "world_epoch_at_observation": 0,
        "current_world_epoch_at_execution": 1,
        "discarded": False,
        "stale": False,
    }
    action_path = _write_jsonl(tmp_path / "actions.jsonl", [action])

    with pytest.raises(ValueError, match="disagrees with world epochs"):
        build_m5_report(
            **inputs,
            calibration_decision_paths=[decision],
            action_paths=[action_path],
        )


def test_reviewed_statistics_must_match_recomputed_episode_statistics(
    tmp_path: Path,
) -> None:
    inputs = _base_inputs(tmp_path)
    decision = _decision(tmp_path / "calibration.json", status="selected")
    episodes = _full_go_episodes()
    episode_path = _write_jsonl(tmp_path / "episodes.jsonl", episodes)
    statistics = build_m5_analysis(
        episodes,
        m4_baseline_intact=True,
        source_validation_passed=True,
    )
    statistics["shift_comparison"]["success"]["aligned"]["success_count"] += 1
    statistics_path = _write_json(tmp_path / "stale_statistics.json", statistics)

    with pytest.raises(ValueError, match="do not canonically match"):
        build_m5_report(
            **inputs,
            calibration_decision_paths=[decision],
            episode_paths=[episode_path],
            statistics_path=statistics_path,
        )


def test_condition_success_rate_excludes_invalid_shifted_episode() -> None:
    valid = _episode(
        GATED_SHIFT,
        0,
        success=True,
        stale_duration=0.0,
    )
    invalid = _episode(
        GATED_SHIFT,
        1,
        success=False,
        stale_duration=0.0,
    )
    invalid["perturbation_valid"] = False
    invalid["invalid_reason"] = "workspace collision"

    summary = next(
        item
        for item in _condition_summaries([valid, invalid])
        if item["condition"] == GATED_SHIFT
    )

    assert summary["episode_count"] == 2
    assert summary["valid_episode_count"] == 1
    assert summary["invalid_episode_count"] == 1
    assert summary["success_count"] == 1
    assert summary["success_rate"] == 1.0
