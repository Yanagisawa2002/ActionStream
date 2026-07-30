from __future__ import annotations

import json

import pytest

from actionstream.results import (
    _resolve_action_trace_path,
    aggregate_episode_records,
    build_custom_parity_manifest,
    build_official_manifest,
    reconstruct_all_success_initial_states,
    render_summary_markdown,
)


def test_action_trace_path_falls_back_to_portable_condition_directory(tmp_path) -> None:
    condition = ("async_aligned", 200)
    condition_dir = tmp_path / "async_aligned_delay200"
    trace = condition_dir / "traces" / "task1_episode8.npz"
    trace.parent.mkdir(parents=True)
    trace.write_bytes(b"trace")

    resolved = _resolve_action_trace_path(
        "/obsolete/workspace/outputs/m3/async_aligned_delay200/traces/task1_episode8.npz",
        condition=condition,
        condition_episode_dirs={condition: condition_dir},
    )

    assert resolved == trace


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
        "overall": {"eval_s": 123.0, "n_episodes": 30, "pc_success": 100.0},
    }
    source = tmp_path / "eval_info.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    manifest = build_official_manifest(
        source,
        model_revision_sha="sha",
        git_commit="commit",
        command="lerobot-eval ...",
        contract_audit={"exit_code": 0, "contract_error_matches": []},
    )
    assert manifest["overall"]["success_count"] == 30
    assert manifest["gate"]["passed"]
    assert manifest["per_task"]["0"]["initial_state_indices"] == list(range(0, 20, 2))


def test_failed_official_episode_makes_init_sequence_ambiguous() -> None:
    with pytest.raises(ValueError, match="terminal/reset provenance"):
        reconstruct_all_success_initial_states([True, False])


def test_custom_parity_manifest_enforces_fixed_protocol(tmp_path) -> None:
    official = {
        "model_revision_sha": "model-sha",
        "overall": {"success_count": 30},
        "gate": {"passed": True},
    }
    official_path = tmp_path / "official.json"
    official_path.write_text(json.dumps(official), encoding="utf-8")
    episode_path = tmp_path / "episodes.jsonl"
    records = []
    for task_id in (0, 1, 2):
        for episode_index, initial_state_index in enumerate(range(0, 20, 2)):
            records.append(
                {
                    "runtime_mode": "sync",
                    "injected_delay_ms": 0,
                    "suite": "libero_object",
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "initial_state_index": initial_state_index,
                    "seed": 142 + episode_index,
                    "realtime_control": False,
                    "policy_rng_seed": 142 + episode_index,
                    "policy_rng_reset_per_episode": True,
                    "environment_steps": 100,
                    "success": True,
                    "git_commit": "commit-sha",
                    "model_revision_sha": "model-sha",
                    "run_id": "run",
                }
            )
    episode_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    manifest = build_custom_parity_manifest(official_path, episode_path)

    assert manifest["gate"]["passed"]
    assert manifest["overall"]["absolute_difference"] == 0
    assert manifest["per_task"]["2"]["success_count"] == 10


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
                "stale_prefix_max_steps": 5,
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
