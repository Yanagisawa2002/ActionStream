from __future__ import annotations

from pathlib import Path

from action_stream_benchmark.m8_faults import generate_fault_trace, load_profile
from action_stream_benchmark.m8_scenario import (
    command_targets_obsolete_destination,
    generate_scenario,
    motion_targets_obsolete_destination,
)
from action_stream_isaac.dynamic_task import scenario_for_seed, scenario_payload, scenario_sha256


ROOT = Path(__file__).resolve().parents[4]


def test_scenario_is_byte_canonical_with_native_task() -> None:
    for seed in (0, 1, 2026081200, 2026081259):
        benchmark = generate_scenario(seed)
        native = scenario_for_seed(seed)
        assert benchmark.core_payload() == scenario_payload(native)
        assert benchmark.sha256 == scenario_sha256(native)


def test_fault_trace_is_deterministic_and_paired() -> None:
    profile = load_profile(
        ROOT / "ros2_ws/src/action_stream_benchmark/config/m8_profile_2_faults.json"
    )
    first = generate_fault_trace(profile, seed=2026081200, request_count=64)
    second = generate_fault_trace(profile, seed=2026081200, request_count=64)
    other = generate_fault_trace(profile, seed=2026081201, request_count=64)
    assert first.to_dict() == second.to_dict()
    assert first.sha256 != other.sha256
    assert all(entry.duplicate_delivery_offset_ms == 50 for entry in first.entries)


def test_obsolete_command_and_motion_are_geometric() -> None:
    assert command_targets_obsolete_destination(
        (0.48, -0.20, 0.20),
        obsolete_destination_xyz=(0.48, -0.22, 0.02575),
        final_destination_xyz=(0.48, 0.22, 0.02575),
    )
    assert motion_targets_obsolete_destination(
        (0.45, 0.0, 0.20),
        (0.46, -0.05, 0.20),
        obsolete_destination_xyz=(0.48, -0.22, 0.02575),
        final_destination_xyz=(0.48, 0.22, 0.02575),
    )
