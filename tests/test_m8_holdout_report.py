import hashlib
import json
from pathlib import Path

import pytest

from scripts.m8_holdout_report import wilson_interval


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_wilson_interval_handles_boundary_rates() -> None:
    zero_low, zero_high = wilson_interval(0, 60)
    full_low, full_high = wilson_interval(60, 60)

    assert zero_low == 0.0
    assert zero_high == pytest.approx(0.0601718521)
    assert full_low == pytest.approx(0.9398281479)
    assert full_high == pytest.approx(1.0)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checked_in_holdout_presentation_matches_registered_evidence() -> None:
    holdout = REPOSITORY_ROOT / "outputs" / "m8_g0" / "holdout_v3"
    matrix_path = holdout / "matrix.json"
    replay_path = holdout / "replay_validation.json"
    analysis_path = holdout / "analysis.json"
    matrix = _read(matrix_path)
    replay = _read(replay_path)
    analysis = _read(analysis_path)
    raw_manifest = _read(holdout / "complete_raw.manifest.json")

    # The 420 raw summaries are an explicitly external archive, not clean-clone
    # source fixtures.  The tracked analysis, replay audit, matrix, and archive
    # member manifest are the portable verification boundary.
    summary_members = [
        member
        for member in raw_manifest["members"]
        if "/raw/summaries/" in member["path"]
    ]
    assert len(summary_members) == 420
    assert matrix["expected_episode_count"] == 420
    assert replay["passed"] is True
    assert replay["episode_count"] == 420
    assert replay["seed_count"] == 60
    assert analysis["manifest_sha256"] == _sha256(matrix_path)
    assert analysis["replay_sha256"] == _sha256(replay_path)
    assert analysis["classification"] == "GO"
    assert analysis["primary_go_gate"]["passed"] is True
    assert analysis["strong_go_gate"]["passed"] is False
    assert analysis["episode_count"] == 420

    profiles = analysis["profiles"]
    fixed = profiles["profile_1_fixed"]["strategies"]
    faults = profiles["profile_2_faults"]["strategies"]
    assert fixed["naive_async"]["successes"] == 0
    assert fixed["aligned_async"]["successes"] == 49
    assert faults["naive_async"]["successes"] == 0
    assert faults["aligned_async"]["successes"] == 52
    assert fixed["aligned_async"]["descriptive_metrics"][
        "expired_actions_executed"
    ]["sum"] == 0
    assert faults["aligned_async"]["descriptive_metrics"][
        "expired_actions_executed"
    ]["sum"] == 0
