from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from actionstream.m6_conformance import (
    OFFICIAL_RUNTIME_TO_AGGREGATE,
    RUNTIME_IDS,
    OfficialLeRobotAdapter,
    _module_is_below,
    ideal_action,
    load_frozen_manifest,
    load_upstream_bindings,
    run_trace,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "m6_g0.json"
LEROBOT_ROOT = ROOT / ".external" / "lerobot"


@pytest.fixture(scope="module")
def manifest():
    return load_frozen_manifest(MANIFEST_PATH)


@pytest.fixture(scope="module")
def upstream():
    if not (LEROBOT_ROOT / ".git").is_dir():
        pytest.skip("M6 upstream conformance tests require .external/lerobot")
    return load_upstream_bindings(LEROBOT_ROOT)


def _run(manifest, upstream, family: str, runtime: str):
    return run_trace(
        manifest,
        family_id=family,
        seed=manifest["scenario"]["seed_base"],
        runtime=runtime,
        bindings=upstream,
    )


def _stable_result_rows(rows):
    return [
        {key: value for key, value in row.items() if key != "scheduler_overhead_ns"}
        for row in rows
    ]


def test_source_origin_check_accepts_namespace_package_paths(tmp_path: Path) -> None:
    namespace = SimpleNamespace(__file__=None, __path__=[str(tmp_path)])
    assert _module_is_below(namespace, tmp_path)
    assert not _module_is_below(namespace, tmp_path / "different-root")


def test_zero_latency_actionstream_discards_no_valid_prefix(
    manifest,
    upstream,
) -> None:
    records, results, summary, _ = _run(
        manifest,
        upstream,
        "zero",
        "actionstream_aligned",
    )

    assert all(result["discarded_prefix_intended_steps"] == [] for result in results)
    assert not [row for row in records if row.get("discard_reason") == "stale_prefix"]
    assert summary["stale_prefix_actions_inserted"] == 0


def test_delayed_actionstream_removes_exactly_elapsed_prefix(
    manifest,
    upstream,
) -> None:
    records, results, _, _ = _run(
        manifest,
        upstream,
        "quarter_horizon",
        "actionstream_aligned",
    )

    first = results[0]
    assert first["observation_step"] == 0
    assert first["result_arrival_step"] == 3
    assert first["queue_insertion_step"] == 4
    assert first["discarded_prefix_intended_steps"] == [1.0, 2.0, 3.0]
    first_fresh = next(
        row
        for row in records
        if row.get("actual_execution_step") == 4
        and row.get("hold_or_underrun_state") == "fresh"
    )
    assert first_fresh["intended_execution_step"] == 4.0
    assert first_fresh["temporal_index_error"] == 0.0


def test_fake_policy_encodes_ideal_absolute_action_index() -> None:
    for step in (1, 7, 48):
        action = ideal_action(step)
        assert action.shape == (6,)
        assert float(action[0]) == step


def test_adapter_invokes_actual_frozen_upstream_methods(
    manifest,
    upstream,
) -> None:
    _, _, summary, _ = _run(
        manifest,
        upstream,
        "quarter_horizon",
        "lerobot_weighted_average",
    )
    audit = summary["adapter_audit"]
    assert audit["upstream_time_chunk_calls"] == 8
    assert audit["upstream_aggregate_calls"] == 8
    assert audit["upstream_control_action_calls"] > 0

    aggregate_file = Path(
        inspect.getsourcefile(upstream.RobotClient._aggregate_action_queues)
    ).resolve()
    timing_file = Path(
        inspect.getsourcefile(upstream.PolicyServer._time_action_chunk)
    ).resolve()
    aggregate_file.relative_to(LEROBOT_ROOT.resolve())
    timing_file.relative_to(LEROBOT_ROOT.resolve())


def test_official_latest_action_filter_matches_executable_source(
    manifest,
    upstream,
) -> None:
    adapter = OfficialLeRobotAdapter(
        upstream,
        aggregate_name="latest_only",
        environment_dt=manifest["scenario"]["environment_dt_seconds"],
        chunk_horizon=5,
    )
    adapter.client.latest_action = 4
    actions = [ideal_action(step) for step in range(10, 15)]
    timed = adapter.time_action_chunk(
        capture_timestamp=0.0,
        observation_timestep=3,
        actions=actions,
    )
    adapter.aggregate(timed)

    assert [item.get_timestep() for item in adapter.queue_items()] == [5, 6, 7]
    assert [float(item.get_action()[0]) for item in adapter.queue_items()] == [
        12.0,
        13.0,
        14.0,
    ]


def test_late_generation_is_rejected_by_actionstream_and_accepted_upstream(
    manifest,
    upstream,
) -> None:
    _, aligned_results, _, _ = _run(
        manifest,
        upstream,
        "late_after_newer_generation",
        "actionstream_aligned",
    )
    _, official_results, _, _ = _run(
        manifest,
        upstream,
        "late_after_newer_generation",
        "lerobot_weighted_average",
    )
    aligned_late = [row for row in aligned_results if row["request_generation"] == 0]
    official_late = [row for row in official_results if row["request_generation"] == 0]

    assert aligned_late and all(not row["accepted"] for row in aligned_late)
    assert {row["reason"] for row in aligned_late} == {"late_generation"}
    assert official_late and any(row["accepted"] for row in official_late)


def test_out_of_order_case_is_deterministic(manifest, upstream) -> None:
    first = _run(
        manifest,
        upstream,
        "out_of_order",
        "lerobot_conservative",
    )
    second = _run(
        manifest,
        upstream,
        "out_of_order",
        "lerobot_conservative",
    )

    assert first[0] == second[0]
    assert _stable_result_rows(first[1]) == _stable_result_rows(second[1])
    assert first[2]["out_of_order_results_accepted"] > 0


def test_every_runtime_receives_identical_chunks_and_latency_trace(
    manifest,
    upstream,
) -> None:
    input_hashes = set()
    contracts = []
    for runtime in RUNTIME_IDS:
        _, _, summary, contract = _run(
            manifest,
            upstream,
            "bounded_jitter",
            runtime,
        )
        input_hashes.add(summary["input_sha256"])
        contracts.append(contract)

    assert len(input_hashes) == 1
    assert all(contract == contracts[0] for contract in contracts[1:])


def test_metrics_are_exact_on_hand_checked_four_step_timeline(
    manifest,
    upstream,
) -> None:
    hand = copy.deepcopy(manifest)
    hand["scenario"].update(
        {
            "chunk_horizon": 3,
            "execution_steps": 4,
            "request_observation_steps": [0],
            "request_interval_steps": 3,
            "deterministic_trace_count": 1,
        }
    )
    hand["latency_families"] = [
        {
            "id": "hand_checked",
            "kind": "fixed",
            "delay_steps": 1,
            "horizon_ratio": 1 / 3,
        }
    ]

    records, results, summary, _ = run_trace(
        hand,
        family_id="hand_checked",
        seed=1,
        runtime="actionstream_aligned",
        bindings=upstream,
    )

    assert results[0]["discarded_prefix_intended_steps"] == [1.0]
    assert [
        (
            row["actual_execution_step"],
            row["intended_execution_step"],
            row["hold_or_underrun_state"],
        )
        for row in records
        if row["actual_execution_step"] is not None and row["record_kind"] != "underrun"
    ] == [
        (2, 2.0, "fresh"),
        (3, 3.0, "fresh"),
        (4, 3.0, "hold_repeat_last_command"),
    ]
    assert summary["median_temporal_index_error"] == 0.0
    assert summary["p95_temporal_index_error"] == pytest.approx(0.9)
    assert summary["queue_underrun_steps"] == 2
    assert summary["hold_steps"] == 1
    assert summary["stale_prefix_actions_inserted"] == 0


def test_frozen_registry_covers_every_official_distinct_aggregate(
    manifest,
    upstream,
) -> None:
    assert set(upstream.aggregate_functions) == set(
        OFFICIAL_RUNTIME_TO_AGGREGATE.values()
    )
    assert set(manifest["runtimes"]["official_lerobot_act_compatible"]) == set(
        OFFICIAL_RUNTIME_TO_AGGREGATE
    )


def test_scenario_manifest_is_predeclared_and_self_consistent(manifest) -> None:
    scenario = manifest["scenario"]
    assert manifest["frozen_before_benchmark"] is True
    assert scenario["request_interval_steps"] == (
        scenario["chunk_horizon"] * scenario["official_chunk_size_threshold"]
    )
    assert scenario["stochastic_trace_count"] >= 50
    assert json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) == manifest
