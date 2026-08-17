from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from actionstream.m8_differential import (
    _normalize_reason,
    build_report,
    load_fixture,
    run_m4,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "configs" / "m8_differential_trace.json"


def test_frozen_fixture_preserves_canonical_contract_and_timestamps() -> None:
    payload = load_fixture(FIXTURE)
    assert payload["frozen"] is True
    assert payload["action_horizon"] == 30
    assert payload["action_dimension"] == 7
    assert len(payload["events"]) >= 60
    assert len({event["event_id"] for event in payload["events"]}) == len(payload["events"])


def test_m4_dense_target_normalization_covers_partial_and_full_expiry() -> None:
    fixture = load_fixture(FIXTURE)
    rows = run_m4(fixture["events"])

    partial = rows["c026"]
    assert partial["accepted"] is True
    assert partial["actions_expired"] == 5
    assert partial["queue"][0]["source_target_step"] == 11
    assert partial["queue"][0]["token"] == 2011

    fully_expired = rows["c_full_chunk"]
    assert fully_expired["accepted"] is False
    assert fully_expired["reason"] == "fully_stale"
    assert fully_expired["actions_expired"] == 30
    assert _normalize_reason(fully_expired["reason"]) == "fully_expired"


def test_trace_places_stale_observation_last_so_it_cannot_contaminate_requests() -> None:
    events = load_fixture(FIXTURE)["events"]
    stale_index = next(
        index for index, event in enumerate(events) if event["event_id"] == "x067"
    )
    assert stale_index == len(events) - 1
    assert events[stale_index]["generation_id"] == 1
    assert events[stale_index - 1]["generation_id"] == 2


def test_actual_cpp_probe_has_exact_common_domain_and_fixed_generation_guard(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required for the direct state-machine probe")

    executable = tmp_path / "action_stream_executor_replay_probe"
    if shutil.which("g++") is None and Path(compiler).name.lower().startswith("clang"):
        executable = executable.with_suffix(".exe" if shutil.which("clang++.exe") else "")
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Wshadow",
            f"-I{ROOT / 'ros2_ws/src/action_stream_executor/include'}",
            str(
                ROOT
                / "ros2_ws/src/action_stream_executor/src/executor_state_machine.cpp"
            ),
            str(
                ROOT
                / "ros2_ws/src/action_stream_executor/src/differential_replay_probe.cpp"
            ),
            "-o",
            str(executable),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    report = build_report(
        fixture_path=FIXTURE,
        probe_path=executable,
        repository_root=ROOT,
    )
    assert report["common_domain_behavioral_equivalence"] is True
    assert report["common_mismatches"] == []
    assert report["missing_expected_trace_difference_codes"] == []
    assert report["unintended_discrepancies"] == []
    assert report["unresolved_unintended_discrepancies"] == []
    assert report["unintended_discrepancies_found"] == [
        "generation_advance_retained_executable_old_generation_queue"
    ]
    history = report["migration_discrepancy_history"]
    assert len(history) == 1
    assert history[0]["classification"] == "unintended_m7_migration_defect"
    assert history[0]["pre_fix_probe_retained"] is False
    assert history[0]["post_fix_probe_evidence"] == {
        "generation_advance_event_id": "x059",
        "old_generation_actions_invalidated": 30,
        "queue_empty_after_advance": True,
        "first_post_switch_command_event_id": "x063",
        "first_post_switch_command_generation": 2,
        "active_generation": 2,
        "stale_observation_rejection_event_id": "x067",
    }
    assert history[0]["status"] == "resolved"
    assert report["headline_evaluation_unblocked"] is True
    assert report["migration_corrections"] == {
        "observation_generation_orders_invalidation_before_command": True,
        "generation_advanced_event_emitted": True,
        "generation_invalidated_event_count": 30,
        "queue_empty_immediately_after_generation_advance": True,
        "first_post_switch_policy_command_uses_active_generation": True,
        "stale_observation_generation_rejected": True,
    }


def test_fixture_is_plain_json_without_host_specific_paths() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "C:\\" not in serialized
    assert "/home/" not in serialized
