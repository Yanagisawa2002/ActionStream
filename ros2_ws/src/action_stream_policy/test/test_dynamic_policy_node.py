from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from action_stream_policy.dynamic_policy_node import _set_dynamic_event_fields


def test_node_module_is_import_safe_without_ros() -> None:
    before = set(sys.modules)
    importlib.reload(sys.modules["action_stream_policy.dynamic_policy_node"])
    newly_loaded = set(sys.modules) - before
    assert "rclpy" not in newly_loaded
    assert "action_stream_msgs" not in newly_loaded


def test_policy_event_is_structured_and_generation_bound() -> None:
    event = SimpleNamespace()
    request = SimpleNamespace(
        episode_id="episode-1",
        request_id=7,
        generation_id=2,
        source_observation_step=100,
    )
    _set_dynamic_event_fields(
        event,
        request,
        detail={"phase_code": 4, "disturbance_switched": True},
    )
    assert event.event_type == "policy_inference_completed"
    assert event.generation_id == event.active_generation_id == 2
    assert event.action_count == 30
    assert json.loads(event.detail)["disturbance_switched"] is True


def test_dynamic_policy_config_matches_frozen_runtime() -> None:
    path = Path(__file__).parents[1] / "config" / "dynamic_pick_place.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["milestone"] == "M8-G0"
    assert payload["task_id"] == "dynamic_target_pick_place_v1"
    assert payload["control_frequency_hz"] == 20.0
    assert payload["chunk_horizon"] == 30
    assert payload["waypoints"]["approach_height_m"] == 0.2
    assert payload["waypoints"]["grasp_hand_offset_m"] == 0.1
