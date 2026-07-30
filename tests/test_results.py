from __future__ import annotations

import json

import pytest

from actionstream.results import (
    aggregate_episode_records,
    build_official_manifest,
    reconstruct_all_success_initial_states,
    render_summary_markdown,
)


def test_official_all_success_manifest_and_initial_states(tmp_path) -> None:
    payload = {
        "per_task": [
            {
                "task_group": "libero_object",
                "task_id": task_id,
                "metrics": {"successes": [True] * 10},
            }
            for task_id in (0, 1, 2)
        ],
        "overall": {"eval_s": 123.0},
    }
    source = tmp_path / "eval_info.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    manifest = build_official_manifest(
        source,
        model_revision_sha="sha",
        git_commit="commit",
        command="lerobot-eval ...",
    )
    assert manifest["overall"]["success_count"] == 30
    assert manifest["gate"]["passed"]
    assert manifest["per_task"]["0"]["initial_state_indices"] == list(range(0, 20, 2))


def test_failed_official_episode_makes_init_sequence_ambiguous() -> None:
    with pytest.raises(ValueError, match="terminal/reset provenance"):
        reconstruct_all_success_initial_states([True, False])


def test_episode_aggregation_and_markdown() -> None:
    records = [
        {
            "runtime_mode": mode,
            "injected_delay_ms": 200,
            "success": success,
            "task_id": task_id,
            "environment_steps": 100,
            "wall_clock_episode_seconds": 5,
            "inference_latency_p50_seconds": 0.1,
            "observation_to_delivery_p50_seconds": 0.3,
            "queue_underrun_hold_steps": holds,
            "stale_chunks_discarded": 0,
            "stale_prefix_mean_steps": 4,
            "control_deadline_misses": 1,
            "action_discontinuity_mean_l2": 0.2,
            "peak_cuda_memory_mib": 3500,
        }
        for mode, success, task_id, holds in [
            ("async_naive", True, 0, 2),
            ("async_naive", False, 1, 4),
            ("async_aligned", True, 0, 0),
        ]
    ]
    aggregates = aggregate_episode_records(records)
    naive = next(row for row in aggregates if row["runtime_mode"] == "async_naive")
    assert naive["success_count"] == 1
    assert naive["episode_count"] == 2
    assert naive["mean_hold_steps"] == 3
    markdown = render_summary_markdown(aggregates)
    assert "async_naive" in markdown
    assert "1/2" in markdown
