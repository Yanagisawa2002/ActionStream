from __future__ import annotations

from pathlib import Path

import pytest

from action_stream_benchmark.event_recorder_node import (
    _detail_counts,
    expanded_runtime_detail,
    task_success_from_state,
)
from action_stream_benchmark.fault_injector_node import (
    duplicate_delivery_offset_ms,
    load_runtime_fault_trace,
)
from action_stream_benchmark.faults import PROFILE_A, generate_fault_trace as generate_m7
from action_stream_benchmark.faults import write_fault_trace as write_m7
from action_stream_benchmark.m8_faults import M8FaultProfile, generate_fault_trace, write_fault_trace
from action_stream_benchmark.m8_recorder import M8EventRecorder
from action_stream_benchmark.schema import read_jsonl

from _m8_test_support import NEW, OLD, fairness


def test_atomic_recorder_lifecycle_and_nonheadline_split(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    with M8EventRecorder(
        path,
        episode_id="dev-episode",
        seed=7,
        profile_id="profile_1_fixed",
        strategy="aligned_async",
        fairness=fairness(holdout=False),
        split="development",
    ) as recorder:
        recorder.start(generation_id=1)
        recorder.destination_switched(
            step=100,
            old_destination_xyz=OLD,
            new_destination_xyz=NEW,
        )
        recorder.terminate(reason="task_timeout", success=False)
        recorder.finish(success=False, completion_reason="task_timeout")
    rows = read_jsonl(path)
    assert [row["event_index"] for row in rows] == list(range(4))
    assert rows[0]["split"] == "development"
    assert "freeze_sha256" not in rows[0]


def test_incomplete_recorder_never_publishes_partial_log(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    with pytest.raises(RuntimeError, match="closed before episode_end"):
        with M8EventRecorder(
            path,
            episode_id="dev-partial",
            seed=8,
            profile_id="profile_1_fixed",
            strategy="naive_async",
            fairness=fairness(holdout=False),
            split="development",
        ) as recorder:
            recorder.start(generation_id=1)
    assert not path.exists()


def test_legacy_recorder_helpers_preserve_m7_and_decode_m8() -> None:
    state = [0.0] * 44
    state[39] = 1.0
    assert task_success_from_state("M8-G0", state)
    legacy = [0.0] * 10
    legacy[9] = 1.0
    assert task_success_from_state("M7-G0", legacy)
    detail = expanded_runtime_detail(
        "M8-G0", '{"switch_step":100,"scenario_sha256":"abc","event_type":"blocked"}'
    )
    assert detail == {"switch_step": 100, "scenario_sha256": "abc"}
    assert expanded_runtime_detail("M7-G0", '{"switch_step":100}') == {}
    assert _detail_counts("expired=2,duplicates=1") == (2, 1)
    assert _detail_counts(
        '{"insertion_observation_step":12,"expired":2,"duplicates":1}'
    ) == (2, 1)


def test_fault_injector_dispatch_keeps_m7_offset_and_uses_m8_offset(tmp_path: Path) -> None:
    m7_path = tmp_path / "m7.json"
    m7 = generate_m7(PROFILE_A, seed=1, request_count=2)
    write_m7(m7_path, m7)
    loaded_m7 = load_runtime_fault_trace(str(m7_path))
    assert loaded_m7.sha256 == m7.sha256
    assert duplicate_delivery_offset_ms(loaded_m7.entry(0)) == 50

    profile = M8FaultProfile(
        profile_id="test",
        base_latency_ms=100,
        jitter_ms=0,
        drop_probability=0.0,
        extra_delay_probability=0.0,
        extra_delay_ms=0,
        duplicate_probability=1.0,
        duplicate_delivery_offset_ms=17,
        communication_pause_probability=0.0,
        communication_pause_ms=0,
    )
    m8_path = tmp_path / "m8.json"
    m8 = generate_fault_trace(profile, seed=2, request_count=2)
    write_fault_trace(m8_path, m8)
    loaded_m8 = load_runtime_fault_trace(str(m8_path))
    assert loaded_m8.sha256 == m8.sha256
    assert duplicate_delivery_offset_ms(loaded_m8.entry(0)) == 17
