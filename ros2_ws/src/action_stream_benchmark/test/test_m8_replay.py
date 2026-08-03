from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from action_stream_benchmark.m8_faults import M8FaultProfile, generate_fault_trace, write_fault_trace
from action_stream_benchmark.m8_replay import (
    HOLDOUT_FAIRNESS_FIELDS,
    _development_strategy_filter,
    _expected_profile_strategies,
    _load_episode_artifacts,
    compare_paired_reset_states,
    decode_dynamic_task_state,
    recompute_metrics,
    validate_episode_rows,
    validate_fault_trace_rows,
)
from action_stream_benchmark.schema import write_json_atomic, write_jsonl_atomic

from _m8_test_support import NEW, START, cloned_episode, reset_state_payload, task_state


def _renumber(rows):
    for index, row in enumerate(rows):
        row["event_index"] = index


def _audit(rows, summary):
    summary["metrics"] = recompute_metrics(rows)
    return validate_episode_rows(
        rows,
        summary,
        required_fairness_fields=HOLDOUT_FAIRNESS_FIELDS,
    )


def test_empty_latency_and_action_age_distributions_are_unavailable() -> None:
    rows, _summary = cloned_episode()
    rows = [
        row
        for row in rows
        if row.get("event_type") not in {"chunk_arrived", "command_executed"}
    ]
    metrics = recompute_metrics(rows)
    for name in (
        "mean_inference_latency_ms",
        "p50_inference_latency_ms",
        "p95_inference_latency_ms",
        "mean_action_age_steps",
        "p50_action_age_steps",
        "p95_action_age_steps",
    ):
        assert metrics[name] is None


def test_replay_allows_only_declared_non_headline_development_method_subset() -> None:
    manifest = {
        "headline_eligible": False,
        "development_strategy_filter": ["aligned_async"],
    }
    selected = _development_strategy_filter(manifest, split="development")
    assert selected == ("aligned_async",)
    assert _expected_profile_strategies(
        "profile_1_fixed",
        split="development",
        development_filter=selected,
    ) == {"aligned_async"}

    assert _expected_profile_strategies(
        "profile_1_fixed",
        split="frozen_holdout",
        development_filter=None,
    ) == {"sync_hold", "naive_async", "aligned_async"}
    with pytest.raises(ValueError, match="forbidden outside development"):
        _development_strategy_filter(manifest, split="frozen_holdout")


@pytest.mark.parametrize(
    "value",
    [[], ["aligned_async", "aligned_async"], ["not_a_strategy"], "aligned_async"],
)
def test_replay_rejects_malformed_development_method_filter(value: object) -> None:
    with pytest.raises(ValueError, match="development_strategy_filter"):
        _development_strategy_filter(
            {"headline_eligible": False, "development_strategy_filter": value},
            split="development",
        )


def test_episode_source_identity_survives_repository_relocation(tmp_path: Path) -> None:
    original = tmp_path / "original"
    (original / ".git").mkdir(parents=True)
    evidence = original / "outputs" / "m8"
    events = evidence / "events"
    events.mkdir(parents=True)
    rows, summary = cloned_episode()
    event_path = events / "episode.jsonl"
    summary_path = evidence / "summary.json"
    write_jsonl_atomic(event_path, rows)
    write_json_atomic(summary_path, summary)
    manifest_path = evidence / "matrix.json"
    manifest = {"episodes": []}
    entry = {
        "event_log_path": "events/episode.jsonl",
        "summary_path": "summary.json",
    }
    write_json_atomic(manifest_path, manifest)

    _, _, original_source = _load_episode_artifacts(manifest_path, manifest, entry)
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    relocated_manifest = relocated / "outputs" / "m8" / "matrix.json"
    _, _, relocated_source = _load_episode_artifacts(relocated_manifest, manifest, entry)

    assert original_source == "outputs/m8/events/episode.jsonl"
    assert relocated_source == original_source
    assert not Path(original_source).is_absolute()


@pytest.mark.parametrize("strategy", ["sync_hold", "naive_async", "aligned_async"])
def test_raw_replay_reconstructs_all_three_queue_methods(strategy: str) -> None:
    rows, summary = cloned_episode(strategy=strategy)
    audit = _audit(rows, summary)
    assert audit["passed"], audit
    assert audit["recomputed_metrics"]["task_success"] is True


def test_raw_task_state_is_authoritative_and_strict() -> None:
    rows, summary = cloned_episode()
    for row in rows:
        if row["event_type"] == "observation":
            row["task_state_decoded"] = {"collision": True, "success": False}
    assert _audit(rows, summary)["recomputed_metrics"]["task_success"] is True

    invalid = deepcopy(next(row["task_state"] for row in rows if row["event_type"] == "observation"))
    invalid[23] = 0.5
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        decode_dynamic_task_state(invalid)


def test_raw_nested_scenario_and_reset_hashes_are_recomputed() -> None:
    rows, summary = cloned_episode()
    start = rows[0]
    start["scenario"]["object_position_xyz"][0] += 0.001
    audit = _audit(rows, summary)
    assert "raw_scenario_sha256_mismatch" in audit["invariant_violation_counts"]
    assert "raw_scenario_summary_mismatch" in audit["invariant_violation_counts"]

    rows, summary = cloned_episode()
    rows[0]["reset_state"]["robot_joint_positions"][0] = 0.01
    audit = _audit(rows, summary)
    assert "raw_reset_state_sha256_mismatch" in audit["invariant_violation_counts"]

    rows, summary = cloned_episode()
    rows[0]["scenario"]["scenario_sha256"] = "0" * 64
    audit = _audit(rows, summary)
    assert "raw_scenario_self_hash_mismatch" in audit["invariant_violation_counts"]


def test_paired_reset_fairness_uses_numeric_tolerances_not_float_identity() -> None:
    reference = reset_state_payload(2026081200)
    near_boundary = deepcopy(reference)
    reference["robot_joint_positions"][0] = 4.9e-6
    near_boundary["robot_joint_positions"][0] = 5.1e-6
    near_boundary["end_effector_orientation_wxyz"] = [0.0, -1.0, 0.0, 0.0]

    audit = compare_paired_reset_states([reference, near_boundary])

    assert audit["passed"] is True
    assert audit["canonical_digest_equal"] is False
    assert audit["canonical_bin_boundary_crossed"] is True
    assert audit["max_deltas"]["robot_joint_position_max_abs_rad"] == pytest.approx(
        0.2e-6
    )


def test_paired_reset_fairness_rejects_material_physical_reset_drift() -> None:
    reference = reset_state_payload(2026081200)
    drifted = deepcopy(reference)
    drifted["object_position_xyz"][0] += 4e-5

    audit = compare_paired_reset_states([reference, drifted])

    assert audit["passed"] is False
    assert audit["errors"][0]["reason"] == "paired_reset_numeric_tolerance_exceeded"
    assert audit["max_deltas"]["object_position_axis_max_abs_m"] == pytest.approx(4e-5)


def test_command_motion_audits_reconstruct_execution_discontinuities() -> None:
    rows, summary = cloned_episode()
    rows[0]["command_motion_audit_required"] = True
    command_indices = [
        index for index, row in enumerate(rows) if row["event_type"] == "command_executed"
    ]
    for ordinal, index in reversed(list(enumerate(command_indices))):
        command = rows[index]
        requested_delta = 0.18 if ordinal == 0 else 0.005
        rows.insert(
            index + 1,
            {
                **{
                    name: command[name]
                    for name in ("episode_id", "profile_id", "seed", "strategy", "split")
                },
                "schema_version": 1,
                "milestone": "M8-G0",
                "evidence_class": command["evidence_class"],
                "event_type": "command_motion_audit",
                "actual_target_step": command["actual_target_step"],
                "requested_target_delta_m": requested_delta,
                "applied_target_delta_m": requested_delta,
                "measured_end_effector_delta_m": 0.004,
                "planned_row_translation_limit_m": 0.01,
                "planned_row_limit_scope": "adjacent_planned_rows_only",
                "hold": bool(command.get("hold")),
            },
        )
    _renumber(rows)

    metrics = recompute_metrics(rows)
    audit = _audit(rows, summary)

    assert audit["passed"], audit
    assert metrics["command_motion_audit_count"] == len(command_indices)
    assert metrics["maximum_requested_target_delta_m"] == pytest.approx(0.18)
    assert metrics["requested_target_discontinuity_count"] == 1


def test_release_and_height_thresholds_are_independently_reconstructed() -> None:
    rows, summary = cloned_episode()
    for row in rows:
        if row["event_type"] == "observation" and row["observation_step"] >= 101:
            row["task_state"][28] = 0.069
            row["task_state"][34] = 1.0  # encoded flag is intentionally dishonest
    metrics = recompute_metrics(rows)
    assert metrics["task_success"] is False
    assert metrics["stable_placement_success"] is False

    rows, summary = cloned_episode()
    for row in rows:
        if row["event_type"] == "observation" and row["observation_step"] >= 101:
            row["task_state"][2] = NEW[2] + 0.04
            row["task_state"][34] = 1.0
    assert recompute_metrics(rows)["task_success"] is False


def test_queue_rebuild_uses_complete_log_not_cross_topic_arrival_order() -> None:
    rows, summary = cloned_episode()
    chunk_index = next(i for i, row in enumerate(rows) if row["event_type"] == "chunk_arrived")
    chunk = rows.pop(chunk_index)
    queue_index = next(i for i, row in enumerate(rows) if row["event_type"] == "queue_updated")
    rows.insert(queue_index + 1, chunk)
    _renumber(rows)
    assert _audit(rows, summary)["passed"]


def test_robot_command_may_precede_cross_topic_generation_events() -> None:
    rows, summary = cloned_episode()
    command_index = next(
        index
        for index, row in enumerate(rows)
        if row["event_type"] == "command_executed" and row.get("source_generation_id") == 2
    )
    command = rows.pop(command_index)
    switch_index = next(
        index for index, row in enumerate(rows) if row["event_type"] == "destination_switched"
    )
    rows.insert(switch_index, command)
    _renumber(rows)
    audit = _audit(rows, summary)
    assert audit["passed"], audit


def test_duplicate_first_wins_and_same_generation_freshness() -> None:
    rows, summary = cloned_episode()
    second_chunk = [row for row in rows if row["event_type"] == "chunk_arrived"][1]
    first_command = deepcopy(second_chunk["actions"][0]["command"])
    second_chunk["actions"].append(
        {"target_step": 101, "command": [0.48, -0.22, 0.02575, 3.14, 0.0, 0.0, -1.0]}
    )
    second_queue = [row for row in rows if row["event_type"] == "queue_updated"][1]
    second_queue["duplicate_target_actions_removed"] = 1
    assert _audit(rows, summary)["passed"]
    command = [row for row in rows if row["event_type"] == "command_executed"][-1]
    command["command"] = second_chunk["actions"][1]["command"]
    audit = _audit(rows, summary)
    assert "command_payload_not_reconstructed_from_chunk" in audit["invariant_violation_counts"]
    command["command"] = first_command

    duplicate_update = deepcopy(second_queue)
    insert_at = rows.index(second_queue) + 1
    duplicate_update["queue_length_before"] = 1
    rows.insert(insert_at, duplicate_update)
    _renumber(rows)
    audit = _audit(rows, summary)
    assert "same_generation_freshness_violation" in audit["invariant_violation_counts"]


def test_sync_periodic_queue_clear_is_reconstructed() -> None:
    rows, summary = cloned_episode(strategy="sync_hold")
    second_queue = [row for row in rows if row["event_type"] == "queue_updated"][1]
    action = [row for row in rows if row["event_type"] == "action_executed"][-1]
    command = [row for row in rows if row["event_type"] == "command_executed"][-1]
    rows.remove(action)
    rows.remove(command)
    rows.insert(
        rows.index(second_queue) + 1,
        {
            **{key: second_queue[key] for key in ("schema_version", "milestone", "evidence_class", "episode_id", "profile_id", "seed", "strategy", "split")},
            "event_type": "queue_cleared",
            "reason": "sync_periodic_replan",
            "request_id": 3,
            "generation_id": 2,
            "source_observation_step": 100,
            "queue_length_before": 1,
            "queue_length_after": 0,
            "action_count": 1,
        },
    )
    _renumber(rows)
    assert _audit(rows, summary)["passed"]


def test_first_termination_event_blocks_late_pending_command() -> None:
    rows, summary = cloned_episode()
    terminal_index = next(i for i, row in enumerate(rows) if row["event_type"] == "task_terminated")
    late = deepcopy(next(row for row in rows if row["event_type"] == "command_executed" and not row["hold"]))
    late["actual_target_step"] = 121
    late["source_target_step"] = 121
    rows.insert(terminal_index + 1, late)
    _renumber(rows)
    audit = _audit(rows, summary)
    assert "post_termination_execution" in audit["invariant_violation_counts"]


def test_timeout_taxonomy_includes_recovery_timeout() -> None:
    rows, _summary = cloned_episode()
    terminal = next(row for row in rows if row["event_type"] == "task_terminated")
    end = next(row for row in rows if row["event_type"] == "episode_end")
    terminal["reason"] = "destination_switch_recovery_timeout"
    terminal["success"] = False
    end["completion_reason"] = "destination_switch_recovery_timeout"
    end["success"] = False
    assert recompute_metrics(rows)["timeout"] is True


def test_pre_switch_terminal_failure_is_valid_without_fabricated_switch() -> None:
    rows, summary = cloned_episode()
    rows = rows[: next(i for i, row in enumerate(rows) if row["event_type"] == "destination_switched")]
    shared = {
        key: rows[0][key]
        for key in (
            "schema_version",
            "milestone",
            "evidence_class",
            "episode_id",
            "profile_id",
            "seed",
            "strategy",
            "split",
        )
    }
    rows.extend(
        [
            {
                **shared,
                "event_type": "observation",
                "observation_step": 20,
                "sim_step": 20,
                "task_state": task_state(
                    20,
                    object_xyz=START,
                    switched=False,
                    phase=10,
                    grasped=False,
                    grasp_ever=False,
                    aperture=0.07,
                    terminated=True,
                ),
            },
            {
                **shared,
                "event_type": "task_terminated",
                "reason": "grasp_miss",
                "success": False,
                "terminal_step": 20,
            },
            {
                **shared,
                "event_type": "episode_terminated",
                "reason": "grasp_miss",
                "queue_length_before": 0,
                "queue_length_after": 0,
            },
            {
                **shared,
                "event_type": "episode_end",
                "success": False,
                "completion_reason": "grasp_miss",
                "completion_steps": 20,
                "wall_time_ns": 1_000_000_000,
                "sim_time_ns": 1_000_000_000,
            },
        ]
    )
    _renumber(rows)
    summary["metrics"] = recompute_metrics(rows)
    audit = validate_episode_rows(
        rows,
        summary,
        required_fairness_fields=HOLDOUT_FAIRNESS_FIELDS,
    )
    assert audit["passed"], audit
    assert audit["recomputed_metrics"]["task_success"] is False


def test_success_requires_switch_event_and_terminal_success_consistency() -> None:
    rows, summary = cloned_episode()
    rows = [row for row in rows if row["event_type"] != "destination_switched"]
    _renumber(rows)
    assert recompute_metrics(rows)["task_success"] is False

    rows, summary = cloned_episode()
    terminal = next(row for row in rows if row["event_type"] == "task_terminated")
    terminal["success"] = False
    audit = _audit(rows, summary)
    assert "recomputed_terminal_success_mismatch" in audit["invariant_violation_counts"]

    rows, summary = cloned_episode()
    end = next(row for row in rows if row["event_type"] == "episode_end")
    end["success"] = False
    audit = _audit(rows, summary)
    assert "recomputed_episode_end_success_mismatch" in audit["invariant_violation_counts"]


def test_fault_trace_conformance_binds_duplicate_delivery_offset(tmp_path: Path) -> None:
    profile = M8FaultProfile(
        profile_id="duplicate-test",
        base_latency_ms=850,
        jitter_ms=0,
        drop_probability=0.0,
        extra_delay_probability=0.0,
        extra_delay_ms=0,
        duplicate_probability=1.0,
        duplicate_delivery_offset_ms=50,
        communication_pause_probability=0.0,
        communication_pause_ms=0,
    )
    trace = generate_fault_trace(profile, seed=17, request_count=1)
    trace_path = tmp_path / "trace.json"
    write_fault_trace(trace_path, trace)
    rows = [
        {"event_index": 0, "event_type": "inference_request", "request_id": 1},
        {
            "event_index": 1,
            "event_type": "chunk_scheduled",
            "request_id": 1,
            "request_ordinal": 0,
            "latency_ms": 850,
            "trace_sha256": trace.sha256,
        },
        {
            "event_index": 2,
            "event_type": "chunk_delivered",
            "request_id": 1,
            "request_ordinal": 0,
            "trace_sha256": trace.sha256,
            "latency_ms": 850,
            "duplicate": False,
        },
        {
            "event_index": 3,
            "event_type": "chunk_arrived",
            "request_id": 1,
            "duplicate": False,
            "out_of_order": False,
        },
        {
            "event_index": 4,
            "event_type": "chunk_delivered",
            "request_id": 1,
            "request_ordinal": 0,
            "trace_sha256": trace.sha256,
            "latency_ms": 900,
            "duplicate": True,
        },
        {
            "event_index": 5,
            "event_type": "chunk_arrived",
            "request_id": 1,
            "duplicate": True,
            "out_of_order": False,
        },
    ]
    summary = {"fault_trace_sha256": trace.sha256}
    assert validate_fault_trace_rows(rows, trace_path=trace_path, summary=summary) == []
    rows[4]["latency_ms"] = 899
    reasons = {
        item["reason"]
        for item in validate_fault_trace_rows(rows, trace_path=trace_path, summary=summary)
    }
    assert "fault_trace_duplicate_offset_mismatch" in reasons

    rows[4]["latency_ms"] = 900
    rows.append(
        {
            "event_index": 6,
            "event_type": "chunk_arrived",
            "request_id": 2,
            "duplicate": False,
            "out_of_order": False,
        }
    )
    reasons = {
        item["reason"]
        for item in validate_fault_trace_rows(rows, trace_path=trace_path, summary=summary)
    }
    assert "fault_event_without_inference_request" in reasons


def test_fault_delivery_may_be_absent_only_when_episode_ends_before_due(tmp_path: Path) -> None:
    profile = M8FaultProfile(
        profile_id="terminal-timing-test",
        base_latency_ms=850,
        jitter_ms=0,
        drop_probability=0.0,
        extra_delay_probability=0.0,
        extra_delay_ms=0,
        duplicate_probability=0.0,
        duplicate_delivery_offset_ms=50,
        communication_pause_probability=0.0,
        communication_pause_ms=0,
    )
    trace = generate_fault_trace(profile, seed=19, request_count=1)
    trace_path = tmp_path / "trace.json"
    write_fault_trace(trace_path, trace)
    scheduled_ns = 1_000_000_000
    rows = [
        {"event_index": 0, "event_type": "inference_request", "request_id": 1},
        {
            "event_index": 1,
            "event_type": "chunk_scheduled",
            "request_id": 1,
            "request_ordinal": 0,
            "latency_ms": 850,
            "trace_sha256": trace.sha256,
            "steady_time_ns": scheduled_ns,
        },
        {
            "event_index": 2,
            "event_type": "episode_end",
            "steady_time_ns": scheduled_ns + 849_000_000,
        },
    ]
    summary = {"fault_trace_sha256": trace.sha256}
    assert validate_fault_trace_rows(rows, trace_path=trace_path, summary=summary) == []
    rows[-1]["steady_time_ns"] = scheduled_ns + 850_000_000
    reasons = {
        item["reason"]
        for item in validate_fault_trace_rows(rows, trace_path=trace_path, summary=summary)
    }
    assert "fault_trace_delivery_arrival_count_mismatch" in reasons
