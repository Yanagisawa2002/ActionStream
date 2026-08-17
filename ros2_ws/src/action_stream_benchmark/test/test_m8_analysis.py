from __future__ import annotations

from pathlib import Path
import shutil

import pytest

import action_stream_benchmark.m8_analysis as analysis_module
from action_stream_benchmark.m8_analysis import (
    _load_summaries,
    _metric_summary,
    _reduction,
    analyze_manifest,
)
from action_stream_benchmark.m8_protocol import sha256_file
from action_stream_benchmark.schema import write_json_atomic, write_jsonl_atomic

from _m8_test_support import cloned_episode


def test_zero_to_zero_is_not_a_material_secondary_reduction() -> None:
    assert _reduction(0.0, 0.0) == 0.0
    assert _reduction(10.0, 5.0) == 0.5


def test_null_distribution_is_reported_as_unavailable_not_zero() -> None:
    assert _metric_summary([None, None]) == {
        "kind": "unavailable",
        "count": 2,
        "observed_count": 0,
        "null_count": 2,
    }


def test_analysis_replays_raw_before_trusting_summary(tmp_path: Path) -> None:
    rows, summary = cloned_episode()
    summary["metrics"]["task_success"] = False
    event_path = tmp_path / "episode.jsonl"
    summary_path = tmp_path / "summary.json"
    write_jsonl_atomic(event_path, rows)
    write_json_atomic(summary_path, summary)
    manifest = {
        "schema_version": 1,
        "milestone": "M8-G0",
        "split": "frozen_holdout",
        "headline_eligible": True,
        "episodes": [
            {
                "event_log_path": event_path.name,
                "summary_path": summary_path.name,
            }
        ],
    }
    manifest_path = tmp_path / "matrix.json"
    write_json_atomic(manifest_path, manifest)
    with pytest.raises(ValueError, match="raw episode replay failed"):
        _load_summaries(manifest_path)


def test_analysis_binds_exact_replay_manifest_bytes(monkeypatch, tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    write_json_atomic(
        matrix_path,
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "split": "frozen_holdout",
            "headline_eligible": True,
        },
    )
    rows = []
    seed = 2026081200
    for profile_id, methods in {
        "profile_0_sanity": ("sync_hold",),
        "profile_1_fixed": ("sync_hold", "naive_async", "aligned_async"),
        "profile_2_faults": ("sync_hold", "naive_async", "aligned_async"),
    }.items():
        for method in methods:
            _events, summary = cloned_episode(
                strategy=method,
                profile_id=profile_id,
                seed=seed,
                episode_id=f"{profile_id}-{method}",
            )
            rows.append(summary)
    monkeypatch.setattr(
        analysis_module,
        "_load_summaries",
        lambda _path: ({"milestone": "M8-G0", "split": "frozen_holdout"}, rows),
    )
    replay = {
        "schema_version": 1,
        "milestone": "M8-G0",
        "split": "frozen_holdout",
        "manifest": matrix_path.name,
        "manifest_sha256": sha256_file(matrix_path),
        "passed": True,
        "fairness_passed": True,
        "freeze_validation_passed": True,
        "seed_validation_passed": True,
        "profile_validation_passed": True,
        "seed_count": 40,
        "episode_count": len(rows),
        "audits": [],
    }
    replay_path = tmp_path / "replay.json"
    write_json_atomic(replay_path, replay)
    result = analyze_manifest(
        matrix_path,
        replay_path=replay_path,
        bootstrap_resamples=1000,
    )
    assert result["manifest"] == matrix_path.name
    assert result["manifest_sha256"] == sha256_file(matrix_path)
    assert result["replay"] == replay_path.name
    assert result["replay_sha256"] == sha256_file(replay_path)
    assert result["profiles"]["profile_1_fixed"]["paired_bootstrap_95_ci_rate"] == [0.0, 0.0]
    assert (
        result["strong_go_gate"]["conditions"][
            "aligned_materially_fewer_obsolete_steps"
        ]
        is False
    )
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    relocated_matrix = relocated / matrix_path.name
    relocated_replay = relocated / replay_path.name
    shutil.copy2(matrix_path, relocated_matrix)
    shutil.copy2(replay_path, relocated_replay)
    relocated_result = analyze_manifest(
        relocated_matrix,
        replay_path=relocated_replay,
        bootstrap_resamples=1000,
    )
    assert relocated_result["classification"] == result["classification"]

    replay["manifest_sha256"] = "0" * 64
    write_json_atomic(replay_path, replay)
    with pytest.raises(ValueError, match="exact matrix manifest bytes"):
        analyze_manifest(matrix_path, replay_path=replay_path, bootstrap_resamples=1000)
