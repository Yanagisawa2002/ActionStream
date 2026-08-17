from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "action_stream_policy"))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "action_stream_benchmark"))

from action_stream_benchmark.analysis import (  # noqa: E402
    REQUIRED_EPISODE_METRICS,
    analyze_manifest,
    paired_bootstrap_ci,
)
from action_stream_benchmark.faults import (  # noqa: E402
    PROFILE_A,
    generate_fault_trace,
)
from action_stream_benchmark.reference_benchmark import (  # noqa: E402
    run_reference_episode,
)
from action_stream_benchmark.replay import (  # noqa: E402
    validate_episode_log,
    validate_manifest,
)
from action_stream_benchmark.schema import (  # noqa: E402
    read_json,
    write_json_atomic,
)


def _matrix(tmp_path: Path) -> Path:
    episodes = []
    for seed in (101, 102):
        trace = generate_fault_trace(PROFILE_A, seed=seed, request_count=64)
        for strategy in ("sync_hold", "naive_async", "aligned_async"):
            event_path = tmp_path / f"{seed}-{strategy}.jsonl"
            summary_path = tmp_path / f"{seed}-{strategy}.json"
            run_reference_episode(
                strategy=strategy,
                trace=trace,
                event_log_path=event_path,
                summary_path=summary_path,
            )
            episodes.append({"event_log": event_path.name, "summary": summary_path.name})
    manifest = tmp_path / "manifest.json"
    write_json_atomic(manifest, {"episodes": episodes})
    return manifest


def test_independent_replay_matches_metrics_and_audits_naive_expiration(
    tmp_path: Path,
) -> None:
    trace = generate_fault_trace(PROFILE_A, seed=33, request_count=64)
    event_path = tmp_path / "naive.jsonl"
    summary_path = tmp_path / "naive.json"
    run_reference_episode(
        strategy="naive_async",
        trace=trace,
        event_log_path=event_path,
        summary_path=summary_path,
    )
    audit = validate_episode_log(event_path, summary_path)
    assert audit["passed"]
    assert audit["metrics_match"]
    assert audit["invariant_violation_counts"]["expired_action_execution"] > 0
    assert audit["naive_expired_execution_is_expected_baseline_behavior"]


def test_replay_detects_summary_tampering(tmp_path: Path) -> None:
    trace = generate_fault_trace(PROFILE_A, seed=34, request_count=64)
    event_path = tmp_path / "aligned.jsonl"
    summary_path = tmp_path / "aligned.json"
    run_reference_episode(
        strategy="aligned_async",
        trace=trace,
        event_log_path=event_path,
        summary_path=summary_path,
    )
    summary = read_json(summary_path)
    summary["metrics"]["simulation_steps"] += 1
    write_json_atomic(summary_path, summary)
    audit = validate_episode_log(event_path, summary_path)
    assert not audit["passed"]
    assert "simulation_steps" in audit["metric_mismatches"]


def test_manifest_validation_and_paired_analysis_preserve_evidence_boundary(
    tmp_path: Path,
) -> None:
    manifest = _matrix(tmp_path)
    audit = validate_manifest(manifest)
    assert audit["passed"]
    assert audit["episode_count"] == 6

    analysis = analyze_manifest(manifest, bootstrap_resamples=1_000)
    profile = analysis["profiles"]["profile_a"]
    assert profile["paired_trial_count"] == 2
    assert profile["strategies"]["aligned_async"]["trials"] == 2
    assert len(profile["raw_paired_outcomes"]) == 2
    assert not analysis["formal_ros_or_isaac_claim_allowed"]

    coverage = analysis["required_metric_coverage"]
    assert set(coverage["episode_summary_metrics"]) == set(REQUIRED_EPISODE_METRICS)
    assert coverage["all_required_metrics_present"]
    assert coverage["all_recorded_metrics_descriptively_summarized"]
    assert coverage["all_executed_action_sources_accounted_for"]

    for strategy in ("sync_hold", "naive_async", "aligned_async"):
        strategy_result = profile["strategies"][strategy]
        descriptive = strategy_result["descriptive_metrics"]
        recorded = coverage["all_recorded_episode_metrics"]
        assert set(descriptive) == set(recorded)
        assert all(metric["count"] == 2 for metric in descriptive.values())
        assert descriptive["task_success"]["kind"] == "boolean"
        assert descriptive["task_success"]["true_count"] == strategy_result["successes"]
        assert descriptive["completion_reason"]["kind"] == "categorical"
        assert sum(descriptive["completion_reason"]["counts"].values()) == 2
        assert descriptive["simulation_steps"]["kind"] == "numeric"

        sources = strategy_result["executed_action_source_generations"]
        assert sources["all_executed_actions_accounted_for"]
        assert sources["events_missing_source_generation"] == 0
        assert sources["missing_event_log_seeds"] == []
        assert sum(sources["generation_counts"].values()) == sources["recorded_executed_actions"]


def test_analysis_rejects_missing_required_metric(tmp_path: Path) -> None:
    manifest = _matrix(tmp_path)
    manifest_payload = read_json(manifest)
    for episode in manifest_payload["episodes"]:
        summary_path = tmp_path / episode["summary"]
        summary = read_json(summary_path)
        del summary["metrics"]["p95_inference_latency_ms"]
        write_json_atomic(summary_path, summary)

    with pytest.raises(ValueError, match="missing required metrics"):
        analyze_manifest(manifest, bootstrap_resamples=1_000)


def test_paired_bootstrap_uses_pair_level_differences() -> None:
    low, high = paired_bootstrap_ci([1.0, 1.0, 1.0, 1.0], resamples=1_000)
    assert low == high == 1.0
