"""Acceptance must fail on unsafe, incomplete, or unavailable stopping evidence."""

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture
def acceptance(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/engineering"))
    return importlib.import_module("completion_acceptance")


@pytest.fixture
def protocol():
    return json.loads((ROOT / "configs/completion_acceptance_v3.json").read_text())


def valid_episodes(p):
    return [
        dict(
            seed=seed,
            first_truth_control=100,
            first_claim_control=110,
            premature_stop=False,
            missed_completed_event=False,
            no_stop=False,
            confirmation_delay_s=0.5,
            post_stop_stable=True,
        )
        for seed in p["seeds"]
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("premature_stop", True),
        ("missed_completed_event", True),
        ("post_stop_stable", False),
        ("post_stop_stable", None),
        ("confirmation_delay_s", 2.05),
    ],
)
def test_one_failed_stop_cannot_be_averaged_away(acceptance, protocol, field, value):
    rows = valid_episodes(protocol)
    assert acceptance.score_episodes(rows, protocol)["passed"]
    rows[0][field] = value
    assert not acceptance.score_episodes(rows, protocol)["passed"]


def test_all_unknown_cannot_pass_when_events_occur(acceptance, protocol):
    rows = valid_episodes(protocol)
    for row in rows:
        row.update(
            first_claim_control=None,
            missed_completed_event=True,
            no_stop=True,
            confirmation_delay_s=None,
            post_stop_stable=None,
        )
    assert not acceptance.score_episodes(rows, protocol)["passed"]


def test_no_physical_completion_cannot_pass_vacuously(acceptance, protocol):
    rows = valid_episodes(protocol)
    for row in rows:
        row.update(
            first_truth_control=None,
            first_claim_control=None,
            no_stop=True,
            confirmation_delay_s=None,
            post_stop_stable=None,
        )
    result = acceptance.score_episodes(rows, protocol)
    assert result["physically_completed_episodes"] == 0 and not result["passed"]


def test_duplicate_or_missing_seed_fails_coverage(acceptance, protocol):
    rows = valid_episodes(protocol)
    assert not acceptance.score_episodes(rows[:-1], protocol)["passed"]
    rows[0]["seed"] = rows[1]["seed"]
    assert not acceptance.score_episodes(rows, protocol)["passed"]


def test_interval_diagnostic_preserves_completion_then_escape(acceptance):
    assert acceptance.intervals([False, True, True, False, True], [(2, True)]) == [
        dict(start=1, end=2, detected=True),
        dict(start=4, end=4, detected=False),
    ]


def test_new_seeds_exclude_all_consumed_v2_and_development(protocol):
    old = json.loads((ROOT / "configs/completion_v2.json").read_text())
    dev = json.loads(
        (ROOT / "configs/completion_development_controls.json").read_text()
    )
    seeds = protocol["seeds"]
    assert len(seeds) == len(set(seeds)) == 20
    assert all(0 <= s < 2**32 for s in seeds)
    assert not set(seeds) & set(
        old["fresh_holdout"]["seeds"] + dev["train_seeds"] + dev["validation_seeds"]
    )
