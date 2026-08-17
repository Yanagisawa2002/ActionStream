from pathlib import Path

import pytest

from scripts.m8_holdout_report import build_report, wilson_interval


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_wilson_interval_handles_boundary_rates() -> None:
    zero_low, zero_high = wilson_interval(0, 60)
    full_low, full_high = wilson_interval(60, 60)

    assert zero_low == 0.0
    assert zero_high == pytest.approx(0.0601718521)
    assert full_low == pytest.approx(0.9398281479)
    assert full_high == pytest.approx(1.0)


def test_checked_in_holdout_presentation_matches_registered_evidence() -> None:
    holdout = REPOSITORY_ROOT / "outputs" / "m8_g0" / "holdout_v3"
    report = build_report(
        matrix_path=holdout / "matrix.json",
        replay_path=holdout / "replay_validation.json",
        analysis_path=holdout / "analysis.json",
        summaries_directory=holdout / "raw" / "summaries",
    )
    groups = {
        (group["profile_id"], group["strategy"]): group for group in report["groups"]
    }

    assert report["official_classification"] == "GO"
    assert report["official_primary_go_passed"] is True
    assert report["official_strong_go_passed"] is False
    assert report["episode_count"] == 420
    assert report["seed_count"] == 60
    assert groups[("profile_1_fixed", "naive_async")]["successes"] == 0
    assert groups[("profile_1_fixed", "aligned_async")]["successes"] == 49
    assert groups[("profile_2_faults", "naive_async")]["successes"] == 0
    assert groups[("profile_2_faults", "aligned_async")]["successes"] == 52
    assert groups[("profile_1_fixed", "aligned_async")]["expired_actions_executed"] == 0
    assert groups[("profile_2_faults", "aligned_async")]["expired_actions_executed"] == 0
