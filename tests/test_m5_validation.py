from __future__ import annotations

import copy
import json

import pytest

from actionstream.m5_protocol import file_sha256
from actionstream.m5_runtime import summarize_action_records
from actionstream.m5_validation import _validate_m4_baseline, _validate_trace_files


def test_m4_baseline_validation_uses_runtime_mode(tmp_path) -> None:
    smoke_path = tmp_path / "episodes.jsonl"
    smoke_path.write_text(
        json.dumps(
            {
                "runtime_mode": "async_aligned",
                "task_id": 0,
                "seed": 142,
                "initial_state_index": 0,
                "injected_delay_ms": 950,
                "success": True,
                "environment_steps": 143,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    task_audit = {
        "m4_baseline_intact": True,
        "m4_baseline_smoke": {
            "passed": True,
            "condition": "async_aligned",
            "task_id": 0,
            "seed": 142,
            "initial_state_index": 0,
            "injected_delay_ms": 950,
            "success": True,
            "environment_steps": 143,
            "path": "episodes.jsonl",
            "sha256": file_sha256(smoke_path),
        },
    }

    result = _validate_m4_baseline(tmp_path, task_audit)

    assert result["passed"]
    assert result["episode"]["runtime_mode"] == "async_aligned"


def _trace_fixture():
    episode_id = "episode-0"
    request_id = "request-0"
    condition = "aligned_no_shift"
    action = {
        "episode_id": episode_id,
        "condition": condition,
        "request_id": request_id,
        "request_generation_id": 0,
        "discarded": False,
        "source_kind": "policy",
        "stale": False,
        "world_epoch_at_observation": 0,
        "current_world_epoch_at_execution": 0,
        "queue_execution_step": 0,
        "observation_control_step": 0,
        "queue_depth_before_action": 1,
        "queue_depth_after_action": 0,
    }
    summary = summarize_action_records([action], 20.0)
    episode = {
        "episode_id": episode_id,
        "condition": condition,
        "controller_frequency_hz": 20.0,
        "environment_steps": 1,
        "request_count": 1,
        "detector_checks": 0,
        "scene_gate_trigger_count": 0,
        "queue_invalidation_count": 0,
        "fresh_replan_count": 0,
        "false_gate_count": 0,
        **summary,
    }
    request = {
        "episode_id": episode_id,
        "condition": condition,
        "request_id": request_id,
        "policy_result_arrival_timestamp": 2.0,
    }
    events = [
        {
            "episode_id": episode_id,
            "condition": condition,
            "event_type": "episode_start",
        },
        {
            "episode_id": episode_id,
            "condition": condition,
            "event_type": "policy_request",
            "request_id": request_id,
        },
        {
            "episode_id": episode_id,
            "condition": condition,
            "event_type": "policy_result_arrival",
            "request_id": request_id,
            "merge": True,
        },
        {
            "episode_id": episode_id,
            "condition": condition,
            "event_type": "queue_merge",
            "request_id": request_id,
        },
        {
            "episode_id": episode_id,
            "condition": condition,
            "event_type": "action_execution",
            "request_id": request_id,
            "step": 0,
            "source_kind": "policy",
            "stale": False,
        },
        {
            "episode_id": episode_id,
            "condition": condition,
            "event_type": "episode_end",
        },
    ]
    return [episode], [action], [request], events


def test_trace_validation_binds_requests_actions_and_events() -> None:
    episodes, actions, requests, events = _trace_fixture()
    result = _validate_trace_files(
        episodes=episodes,
        actions=actions,
        requests=requests,
        events=events,
    )
    assert result["request_action_event_linkage_validated"]

    drifted_actions = copy.deepcopy(actions)
    drifted_actions[0]["request_id"] = "missing-request"
    with pytest.raises(ValueError, match="unrecorded request"):
        _validate_trace_files(
            episodes=episodes,
            actions=drifted_actions,
            requests=requests,
            events=events,
        )
