from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
POLICY_ROOT = ROOT / "ros2_ws" / "src" / "action_stream_policy"
BENCHMARK_ROOT = ROOT / "ros2_ws" / "src" / "action_stream_benchmark"
sys.path.insert(0, str(POLICY_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT))

from action_stream_benchmark.ros_utils import (  # noqa: E402
    combine_nanoseconds,
    split_nanoseconds,
)


def test_timestamp_conversion_round_trip_and_boundaries() -> None:
    for value in (0, 1, 999_999_999, 1_000_000_000, 123_456_789_012_345):
        seconds, nanoseconds = split_nanoseconds(value)
        assert 0 <= nanoseconds < 1_000_000_000
        assert combine_nanoseconds(seconds, nanoseconds) == value
    with pytest.raises(ValueError):
        split_nanoseconds(-1)
    with pytest.raises(ValueError):
        combine_nanoseconds(1, 1_000_000_000)


def test_ros_nodes_are_importable_without_ros_for_offline_ci() -> None:
    import action_stream_benchmark.event_recorder_node  # noqa: F401
    import action_stream_benchmark.fault_injector_node  # noqa: F401
    import action_stream_benchmark.test_plant_node  # noqa: F401
    import action_stream_policy.policy_node  # noqa: F401


def test_launch_routes_topics_through_compiled_cpp_executor() -> None:
    launch = (BENCHMARK_ROOT / "launch" / "test_plant_episode.launch.py").read_text(
        encoding="utf-8"
    )
    assert 'package="action_stream_executor"' in launch
    assert 'executable="action_stream_executor_node"' in launch
    assert '"/action_stream/raw_action_chunk"' in launch
    assert '"/action_stream/action_chunk"' in launch
    assert '"/action_stream/robot_command"' in launch
    assert "OnProcessExit" in launch


def test_console_contract_separates_reference_and_ros_evidence() -> None:
    setup = (BENCHMARK_ROOT / "setup.py").read_text(encoding="utf-8")
    cli = (BENCHMARK_ROOT / "action_stream_benchmark" / "cli.py").read_text(encoding="utf-8")
    assert "fault-injector-node" in setup
    assert "test-plant-node" in setup
    assert "event-recorder-node" in setup
    assert '"reference-run"' in cli
    assert '"ros-episode"' in cli
    assert '"ros-run"' in cli
    assert '"deterministic_ros_test_plant_reference"' in cli
    assert '"ros_cpp_test_plant"' in cli
    assert '"ros_log":' not in cli


def test_episode_requests_do_not_masquerade_as_status_messages() -> None:
    source = (BENCHMARK_ROOT / "action_stream_benchmark" / "test_plant_node.py").read_text(
        encoding="utf-8"
    )
    assert "message.active = False" in source
    assert "message.terminated = False" in source
    assert "command == EpisodeControl.TERMINATE" in source


def test_frozen_profile_b_is_within_predeclared_ranges() -> None:
    payload = json.loads(
        (BENCHMARK_ROOT / "config" / "profile_b.json").read_text(encoding="utf-8")
    )["profile"]
    assert payload["frozen"] is True
    assert 800 <= payload["base_latency_ms"] <= 950
    assert 150 <= payload["jitter_ms"] <= 250
    assert 0.02 <= payload["drop_probability"] <= 0.05
    assert payload["extra_delay_probability"] > 0
