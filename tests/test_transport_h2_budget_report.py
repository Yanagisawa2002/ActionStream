from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/actionstream_transport_h2_budget_holdout"
UNBOUNDED = "actionstream_backend_pipelined_aligned"
BUDGETED = "actionstream_backend_budgeted_pipelined_aligned"


def _summary() -> dict:
    return json.loads((REPORT / "summary.json").read_text(encoding="utf-8"))


def _csv(name: str) -> list[dict[str, str]]:
    with (REPORT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_h2_report_preserves_the_frozen_no_go_verdict() -> None:
    summary = _summary()
    assert summary["verdict"] == "NO_GO"
    failed = {
        key for key, value in summary["gate_results"].items() if not value["pass"]
    }
    assert failed == {"budgeted_success_count_not_below_unbounded"}
    assert summary["runtime_metrics"][UNBOUNDED]["successes"] == 14
    assert summary["runtime_metrics"][BUDGETED]["successes"] == 13


def test_h2_compute_and_supply_effects_match_the_retained_rows() -> None:
    summary = _summary()
    unbounded = summary["runtime_metrics"][UNBOUNDED]
    budgeted = summary["runtime_metrics"][BUDGETED]
    effect = summary["paired_bootstrap"]["aggregate_inference_call_ratio"]
    assert unbounded["inference_calls"] == 1109
    assert budgeted["inference_calls"] == 426
    assert effect["estimate"] == pytest.approx(426 / 1109)
    assert effect["ci_95"][1] < 0.5
    assert budgeted["median_inference_requests_per_second"] > 2.5
    assert budgeted["depletion_fraction"] == 0.0
    assert summary["paired_bootstrap"]["success_difference"]["estimate"] == pytest.approx(
        -1 / 15
    )


def test_h2_report_exposes_family_heterogeneity_and_failures() -> None:
    summary = _summary()
    families = {row["suite"]: row for row in summary["family_table"]}
    assert families["libero_goal"]["unbounded_successes"] == 5
    assert families["libero_goal"]["budgeted_successes"] == 4
    assert families["libero_goal"]["budgeted_call_ratio"] > 0.5
    assert families["libero_object"]["budgeted_successes"] == 4
    assert families["libero_spatial"]["budgeted_successes"] == 5
    failures = _csv("failure_taxonomy.csv")
    assert {row["category"] for row in failures} == {
        "budget_only_failure_paired_regression",
        "shared_policy_task_failure",
    }
    regression = next(
        row
        for row in failures
        if row["category"] == "budget_only_failure_paired_regression"
    )
    assert regression["suite"] == "libero_goal"
    assert regression["initial_state_index"] == "19"


def test_h2_gpu_and_video_provenance_are_reviewable_and_bounded() -> None:
    summary = _summary()
    assert summary["evidence_validation"]["manifest_mismatches"] == []
    assert summary["evidence_validation"]["fresh_process_pid_count"] == 6
    assert summary["non_scored_gpu_canary"]["scored_environment_steps"] == 0
    assert summary["non_scored_gpu_canary"]["record"]["inference_calls"] == 2
    assert len(summary["video_inventory"]) == 6
    assert all(row["decode_status"] == "four_frames_checked" for row in summary["video_inventory"])
    assert all("\\" not in row["path"] and ":" not in row["path"] for row in summary["video_inventory"])
    assert (REPORT / "representative_video_contact_sheet.jpg").stat().st_size < 500_000
    report = (REPORT / "report.md").read_text(encoding="utf-8")
    assert "2.10x" in report
    assert "state 19" in report
    assert "NO-GO" in report
