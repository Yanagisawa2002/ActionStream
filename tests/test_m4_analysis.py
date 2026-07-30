from __future__ import annotations

import json

import numpy as np
import pytest

from actionstream.m4_analysis import (
    BOOTSTRAP_RESAMPLES,
    build_paired_m3_analysis,
    inspect_historical_replacement_indices,
    paired_metric_effect,
)


def _m3_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for mode in ("sync", "async_naive", "async_aligned"):
        for delay_ms in (0, 200):
            for task_id in (0, 1, 2):
                for episode_index in range(10):
                    steps = 100 + task_id + episode_index
                    if mode == "async_naive":
                        steps += 20
                    elif mode == "async_aligned":
                        steps += 5
                    if delay_ms == 200:
                        steps += 2
                    records.append(
                        {
                            "runtime_mode": mode,
                            "injected_delay_ms": delay_ms,
                            "task_id": task_id,
                            "episode_index": episode_index,
                            "initial_state_index": 2 * episode_index,
                            "seed": 142 + episode_index,
                            "policy_rng_seed": 142 + episode_index,
                            "suite": "libero_object",
                            "model_id": "model",
                            "model_revision_sha": "revision",
                            "success": True,
                            "environment_steps": steps,
                            "wall_clock_episode_seconds": steps / 20,
                            "observation_to_delivery_p50_seconds": 0.1 + delay_ms / 1000,
                            "action_discontinuity_mean_l2": 0.02 + steps / 100_000,
                        }
                    )
    return records


def test_paired_metric_effect_has_requested_sign_and_is_deterministic() -> None:
    first = paired_metric_effect([10, 20, 40], [8, 18, 36])
    second = paired_metric_effect([10, 20, 40], [8, 18, 36])

    assert first == second
    assert first["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    assert first["paired_mean_difference"] == pytest.approx(-8 / 3)
    assert first["reference"]["median"] == 20
    assert first["estimate"]["median"] == 18
    assert first["paired_relative_difference_percent"] == pytest.approx(-13.3333333333)


def test_paired_m3_analysis_includes_all_comparisons_and_task_breakdown() -> None:
    report = build_paired_m3_analysis(
        _m3_records(),
        replacement_audit={"replacement_boundary_indices_available": False},
    )

    comparisons = {
        comparison["comparison_id"]: comparison
        for comparison in report["paired_comparisons"]
    }
    assert len(comparisons) == 7
    aligned_vs_naive = comparisons["async_aligned_minus_async_naive_delay0"]
    assert aligned_vs_naive["overall"]["environment_steps"][
        "paired_mean_difference"
    ] == pytest.approx(-15)
    assert aligned_vs_naive["per_task"]["2"]["environment_steps"][
        "paired_episode_count"
    ] == 10
    degradation = comparisons["async_aligned_delay200_minus_delay0"]
    assert degradation["overall"]["environment_steps"][
        "paired_mean_difference"
    ] == pytest.approx(2)
    assert degradation["overall"]["success"]["paired_mean_difference"] == 0


def test_pair_metadata_mismatch_is_rejected() -> None:
    records = _m3_records()
    records[-1]["seed"] = 999

    with pytest.raises(ValueError, match="Pair metadata mismatch"):
        build_paired_m3_analysis(
            records,
            replacement_audit={"replacement_boundary_indices_available": False},
        )


def test_historical_trace_without_explicit_replacement_index_is_not_reconstructed(
    tmp_path,
) -> None:
    trace_path = tmp_path / "trace.npz"
    events = [
        {
            "observation_control_step": 10,
            "delivery_timestamp": 1.2,
            "merge": {
                "accepted": True,
                "reason": "replaced",
                "age_steps": 4,
                "dropped_prefix_steps": 4,
                "queue_length": 26,
            },
        }
    ]
    np.savez_compressed(
        trace_path,
        actions=np.ones((2, 7), dtype=np.float32),
        dispatch_timestamps=np.asarray([1.0, 1.1]),
        inference_events_json=np.asarray(json.dumps(events)),
    )
    records = [
        {
            "runtime_mode": "async_aligned",
            "injected_delay_ms": 200,
            "action_trace_path": str(trace_path),
        }
    ]

    audit = inspect_historical_replacement_indices(records, episode_paths=[])

    assert audit["accepted_replacement_event_count"] == 1
    assert not audit["replacement_boundary_indices_available"]
    assert audit["no_reverse_engineering_performed"]
    assert audit["replacement_boundary_metrics"]["rotation_geodesic_angle_jump"] is None
