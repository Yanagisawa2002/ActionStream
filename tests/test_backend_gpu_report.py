from __future__ import annotations

from actionstream.backend_gpu_report import (
    build_failure_taxonomy,
    build_main_table,
    build_paired_effects,
    build_secondary_paired_effects,
    evaluate_gates,
)


def _row(
    runtime: str,
    *,
    suite: str = "libero_object",
    success: bool = True,
    steps: int = 100,
) -> dict:
    backend = runtime.startswith("actionstream_backend")
    return {
        "experiment_id": f"actionstream_backend_gpu_xvla_holdout_{suite}_v1",
        "model_key": "xvla",
        "runtime": runtime,
        "delay_profile": "fixed_0950",
        "suite": suite,
        "task_id": {"libero_object": 5, "libero_spatial": 7, "libero_goal": 2}[suite],
        "episode_index": 0,
        "initial_state_index": 40,
        "seed": 2026082140,
        "success": success,
        "environment_steps": steps,
        "wall_clock_episode_seconds": 10.0,
        "inference_calls": 10,
        "inference_completed": 10 if backend else None,
        "peak_cuda_memory_mib": 12000.0,
        "inference_latency_p50_seconds": 0.08,
        "inference_latency_p95_seconds": 0.10,
        "delivery_latency_p50_seconds": 1.03,
        "delivery_latency_p95_seconds": 1.08,
        "queue_depth_p50_steps": 20.0 if backend else None,
        "queue_depth_p95_steps": 28.0 if backend else None,
        "queue_age_p50_steps": 3.0 if backend else None,
        "queue_age_p95_steps": 5.0 if backend else None,
        "dropped_prefix_steps": 6,
        "hold_steps": 0,
        "bounded_hold_steps": 0 if backend else None,
        "depletion_safe_hold_steps": 0 if backend else None,
        "disconnects": 0 if backend else None,
        "recoveries": 0 if backend else None,
        "fallback_activations": 0 if backend else None,
        "fallback_chunks_accepted": 0 if backend else None,
        "chunk_size": 30,
    }


def test_main_table_preserves_unavailable_upstream_queue_age() -> None:
    rows = [
        _row("lerobot_latest_only"),
        _row("actionstream_backend_guarded", steps=90),
    ]

    table = build_main_table(rows)
    upstream = next(row for row in table if row["runtime"] == "lerobot_latest_only")
    guarded = next(
        row for row in table if row["runtime"] == "actionstream_backend_guarded"
    )

    assert upstream["queue_age_p50_steps_median"] is None
    assert upstream["fallback_activations_sum"] is None
    assert guarded["queue_age_p50_steps_median"] == 3.0
    assert guarded["environment_steps_per_second_mean"] == 9.0
    assert guarded["gpu_generated_actions_per_second_p50_median"] == 375.0


def test_paired_effects_use_latest_only_as_reference() -> None:
    rows = [
        _row("lerobot_latest_only", steps=100),
        _row("actionstream_backend_guarded", steps=90),
    ]

    effects = build_paired_effects(rows)

    assert len(effects) == 2
    step_effect = next(
        item for item in effects if item["metric"] == "environment_steps"
    )
    assert step_effect["reference_runtime"] == "lerobot_latest_only"
    assert step_effect["estimate_runtime"] == "actionstream_backend_guarded"
    assert step_effect["paired_mean_difference"] == -10.0


def test_secondary_paired_effects_keep_official_async_separate() -> None:
    rows = [
        _row("lerobot_weighted_average", success=False, steps=200),
        _row("actionstream_backend_aligned", success=True, steps=150),
        _row("lerobot_latest_only", success=True, steps=140),
    ]

    effects = build_secondary_paired_effects(rows)

    assert len(effects) == 2
    success_effect = next(item for item in effects if item["metric"] == "success")
    step_effect = next(
        item for item in effects if item["metric"] == "environment_steps"
    )
    assert success_effect["reference_runtime"] == "lerobot_weighted_average"
    assert success_effect["estimate_runtime"] == "actionstream_backend_aligned"
    assert success_effect["paired_mean_difference"] == 1.0
    assert step_effect["paired_mean_difference"] == -50.0


def test_failure_taxonomy_does_not_turn_missing_telemetry_into_zero_claims() -> None:
    upstream = _row("lerobot_latest_only", success=False, steps=300)
    guarded = _row("actionstream_backend_guarded", success=False, steps=300)
    guarded["depletion_safe_hold_steps"] = 4

    taxonomy = build_failure_taxonomy([upstream, guarded])

    assert {item["category"] for item in taxonomy} == {
        "task_failure_other",
        "queue_depletion_safe_hold",
    }


def test_xvla_canary_gate_requires_all_three_sync_suites_and_recovery() -> None:
    rows = []
    for suite in ("libero_object", "libero_spatial", "libero_goal"):
        sync = _row("sync_hold", suite=suite)
        sync["experiment_id"] = f"actionstream_backend_gpu_xvla_canary_{suite}_v1"
        sync["delay_profile"] = "fixed_0000"
        rows.append(sync)
        for runtime in ("actionstream_backend_aligned", "actionstream_backend_guarded"):
            disconnect = _row(runtime, suite=suite)
            disconnect["experiment_id"] = (
                f"actionstream_backend_gpu_xvla_canary_{suite}_v1"
            )
            disconnect["delay_profile"] = "disconnect_recovery_canary"
            disconnect["status"] = "completed"
            disconnect["disconnects"] = 2
            disconnect["recoveries"] = 2
            rows.append(disconnect)

    gates = evaluate_gates(rows, [])

    assert gates["xvla_sync_zero_delay_three_suite_gate"] is True
    assert gates["xvla_disconnect_recovery_gate"] is True
