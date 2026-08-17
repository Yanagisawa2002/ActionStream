from __future__ import annotations

import json

from actionstream.m5_analysis import ALIGNED_NO_SHIFT, GATED_NO_SHIFT
from actionstream.m5_no_shift import build_no_shift_decision, compare_action_traces
from actionstream.m5_protocol import canonical_sha256


def episode(seed: int, state: int, condition: str, protocol_sha256: str) -> dict:
    return {
        "condition": condition,
        "seed": seed,
        "initial_state_index": state,
        "phase": "no_shift",
        "task_id": 0,
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
        "protocol_decision_sha256": protocol_sha256,
        "no_shift_decision_sha256": None,
        "injected_delay_ms": 950,
        "controller_frequency_hz": 20.0,
        "chunk_size_steps": 30,
        "replan_interval_steps": 10,
        "nominal_queue_headroom_steps": 20,
        "scheduled_shift_step": None,
        "shift_step": None,
        "displacement_magnitude_mm": 0,
        "requested_displacement_xy_m": None,
        "achieved_displacement_xyz_m": None,
        "success": True,
        "stale_action_duration_seconds": 0.0,
        "stale_action_steps": 0,
        "environment_steps": 1,
        "simulated_completion_time_seconds": 0.05,
        "wall_clock_episode_seconds": 0.01,
        "time_from_shift_to_gate_trigger_seconds": None,
        "time_from_shift_to_first_fresh_action_seconds": None,
        "time_from_shift_to_queue_clear_seconds": None,
        "perturbation_valid": True,
        "invalid_reason": None,
        "false_gate_count": 0,
        "scene_gate_trigger_count": 0,
        "queue_invalidation_count": 0,
    }


def action(seed: int, condition: str, value: float = 1.0) -> dict:
    return {
        "condition": condition,
        "seed": seed,
        "discarded": False,
        "queue_execution_step": 0,
        "action": [value] * 7,
        "source_kind": "policy",
    }


def test_no_shift_action_trace_comparison_is_exact_and_detects_change() -> None:
    episodes = []
    actions = []
    for index in range(10):
        seed = 52001 + index
        for condition in (ALIGNED_NO_SHIFT, GATED_NO_SHIFT):
            episodes.append(episode(seed, index, condition, "protocol"))
            actions.append(action(seed, condition))
    exact = compare_action_traces(episodes, actions)
    assert exact["pair_count"] == 10
    assert exact["all_action_traces_bit_exact"]
    assert exact["all_action_source_traces_exact"]

    actions[-1]["action"][0] = 1.25
    changed = compare_action_traces(episodes, actions)
    assert not changed["all_action_traces_bit_exact"]
    assert changed["maximum_absolute_action_difference"] == 0.25


def test_no_shift_decision_binds_protocol_and_safety(tmp_path) -> None:
    protocol_core = {
        "schema_version": 1,
        "milestone": "M5-G0",
        "status": "selected",
        "no_shift_task_id": 0,
    }
    protocol = {
        **protocol_core,
        "protocol_decision_sha256": canonical_sha256(protocol_core),
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    episode_path = tmp_path / "episodes.jsonl"
    action_path = tmp_path / "actions.jsonl"
    episodes = []
    actions = []
    for index in range(10):
        seed = 52001 + index
        for condition in (ALIGNED_NO_SHIFT, GATED_NO_SHIFT):
            episodes.append(
                episode(
                    seed,
                    index,
                    condition,
                    protocol["protocol_decision_sha256"],
                )
            )
            actions.append(action(seed, condition))
    episode_path.write_text(
        "".join(json.dumps(row) + "\n" for row in episodes),
        encoding="utf-8",
    )
    action_path.write_text(
        "".join(json.dumps(row) + "\n" for row in actions),
        encoding="utf-8",
    )

    decision = build_no_shift_decision(
        protocol_decision_path=protocol_path,
        episode_paths=[episode_path],
        action_paths=[action_path],
    )
    assert decision["status"] == "pass"
    assert all(decision["checks"].values())
    assert decision["analysis"]["valid_pair_count"] == 10
