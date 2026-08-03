from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from action_stream_benchmark.m8_protocol import build_episode_fairness
from action_stream_benchmark.schema import read_jsonl
from action_stream_isaac.dynamic_event_recorder import (
    DynamicM8RecorderCore,
    DynamicRecorderConfig,
)
from action_stream_isaac.dynamic_isaac_adapter import _finalize_episode_summary
from action_stream_isaac.dynamic_task import scenario_for_seed, scenario_payload


def _protocol() -> dict[str, object]:
    root = Path(__file__).resolve().parents[4]
    return json.loads((root / "configs" / "m8_g0.json").read_text(encoding="utf-8"))


def _fairness(seed: int, trace_sha256: str) -> dict[str, object]:
    scenario = scenario_for_seed(seed)
    return build_episode_fairness(
        protocol_payload=_protocol(),
        scenario_payload=scenario_payload(scenario),
        reset_state_payload={
            "seed": seed,
            "robot_joint_positions": [0.0] * 9,
            "robot_joint_velocities": [0.0] * 9,
            "end_effector_position_xyz": [0.307015, 0.0, 0.589907],
            "end_effector_orientation_wxyz": [0.0, 1.0, 0.0, 0.0],
            "object_position_xyz": list(scenario.object_xyz),
            "object_orientation_wxyz": list(scenario.object_wxyz),
            "zone_a_xyz": list(scenario.zone_a_xyz),
            "zone_b_xyz": list(scenario.zone_b_xyz),
            "physics_dt_seconds": 1.0 / 60.0,
            "rendering_dt_seconds": 1.0 / 60.0,
            "stage_units_in_meters": 1.0,
            "gravity_xyz": [0.0, 0.0, -9.81],
        },
        fault_trace_sha256=trace_sha256,
    )


def _record_complete_episode(path: Path) -> tuple[dict[str, object], int, str]:
    seed = 2026081000
    episode_id = "m8-development-profile_1_fixed-test-aligned_async"
    trace_sha256 = "a" * 64
    fairness = _fairness(seed, trace_sha256)
    scenario = scenario_for_seed(seed)
    core = DynamicM8RecorderCore(
        DynamicRecorderConfig(
            output_path=path,
            episode_id=episode_id,
            seed=seed,
            profile_id="profile_1_fixed",
            strategy="aligned_async",
            split="development",
            expected_scenario_sha256=str(fairness["scenario_sha256"]),
            expected_fault_trace_sha256=trace_sha256,
        )
    )
    core.record("simulator_reset", {"sim_time_ns": 0})
    core.record(
        "episode_start",
        {
            **fairness,
            "generation_id": 1,
            "sim_time_ns": 0,
            "steady_time_ns": 100,
            "wall_time_ns": 1_000,
            "maximum_episode_steps": 320,
        },
    )
    core.record(
        "destination_switched",
        {
            "reason": "seeded_disturbance_boundary",
            "switch_step": scenario.switch_step,
            "old_destination_xyz": list(scenario.original_destination_xyz),
            "new_destination_xyz": list(scenario.final_destination_xyz),
            "generation_before": 1,
            "generation_after": 2,
            "sim_time_ns": scenario.switch_step * 50_000_000,
            "steady_time_ns": 200,
            "wall_time_ns": 2_000,
        },
    )
    core.record(
        "task_terminated",
        {
            "reason": "destination_switch_recovery_timeout",
            "success": False,
            "terminal_step": scenario.switch_step + 20,
            "sim_time_ns": (scenario.switch_step + 20) * 50_000_000,
            "steady_time_ns": 300,
            "wall_time_ns": 3_000,
        },
    )
    core.record(
        "episode_end",
        {
            "reason": "destination_switch_recovery_timeout",
            "completion_reason": "destination_switch_recovery_timeout",
            "success": False,
            "completion_steps": scenario.switch_step + 20,
            "sim_time_ns": (scenario.switch_step + 20) * 50_000_000,
            "steady_time_ns": 300,
            "wall_time_ns": 3_000,
        },
    )
    core.record("diagnostics", {"queue_length": 0})
    core.close()
    return fairness, seed, episode_id


def test_single_authority_recorder_buffers_start_and_commits_end_last(tmp_path: Path) -> None:
    raw = tmp_path / "episode.jsonl"
    fairness, _seed, _episode_id = _record_complete_episode(raw)
    rows = read_jsonl(raw)
    assert [row["event_type"] for row in rows].count("episode_start") == 1
    assert [row["event_type"] for row in rows].count("destination_switched") == 1
    assert [row["event_type"] for row in rows].count("task_terminated") == 1
    assert rows[-1]["event_type"] == "episode_end"
    assert rows[0]["fault_trace_sha256"] == fairness["fault_trace_sha256"]
    assert rows[0]["protocol_sha256"] == fairness["protocol_sha256"]
    assert [row["event_index"] for row in rows] == list(range(len(rows)))


def test_adapter_finalizer_recomputes_summary_from_committed_raw(tmp_path: Path) -> None:
    raw = tmp_path / "episode.jsonl"
    summary_path = tmp_path / "episode.json"
    fairness, seed, episode_id = _record_complete_episode(raw)
    spec = SimpleNamespace(
        event_log_path=raw,
        summary_path=summary_path,
        profile_id="profile_1_fixed",
        seed=seed,
        strategy="aligned_async",
        episode_id=episode_id,
        scenario_sha256=fairness["scenario_sha256"],
        trace_sha256=fairness["fault_trace_sha256"],
    )
    matrix = SimpleNamespace(
        split="development",
        protocol_sha256=fairness["protocol_sha256"],
        freeze_sha256=None,
    )
    summary = _finalize_episode_summary(matrix=matrix, spec=spec)
    assert summary_path.is_file()
    assert summary["native_isaac_physics"] is True
    assert summary["headline_eligible"] is False
    assert summary["metrics"]["task_success"] is False
    assert summary["metrics"]["completion_reason"] == "destination_switch_recovery_timeout"
    assert summary["fault_trace_sha256"] == fairness["fault_trace_sha256"]
