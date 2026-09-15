"""Stable task semantics must reject containment without release/support/history."""

import numpy as np
import pytest
import json
from pathlib import Path
from actionstream.completion_labels import instantaneous, StableTruth
from actionstream.llm_vla.temporal_completion import classify


def test_fresh_seeds_are_distinct_and_supported_by_numpy():
    protocol = json.loads(
        (Path(__file__).parents[1] / "configs/completion_v2.json").read_text()
    )
    seeds = protocol["fresh_holdout"]["seeds"]
    assert len(seeds) == len(set(seeds)) == 20
    assert all(0 <= value < 2**32 for value in seeds)
    assert not set(seeds) & {2026091510, 2026091511, 2026091512}


def ready(**values):
    return dict(
        inside=True,
        finger_contact=False,
        basket_contact=True,
        linear_speed=0.0,
        angular_speed=0.0,
        **values,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("inside", False),
        ("finger_contact", True),
        ("basket_contact", False),
        ("linear_speed", 0.031),
        ("angular_speed", 0.301),
        ("linear_speed", float("nan")),
    ],
)
def test_containment_alone_cannot_authorize_completion(field, value):
    facts = ready()
    facts[field] = value
    assert not instantaneous(facts)


def test_half_second_stability_requires_eleven_consecutive_samples():
    state = StableTruth()
    for step in range(10):
        assert not state.update(step, ready())
    assert state.update(10, ready())


def test_drop_and_regrasp_clear_history_immediately():
    state = StableTruth()
    for step in range(11):
        state.update(step, ready())
    bad = ready()
    bad["finger_contact"] = True
    assert not state.update(11, bad)
    for step in range(12, 22):
        assert not state.update(step, ready())
    assert state.update(22, ready())
    dropped = ready()
    dropped["inside"] = False
    assert not state.update(23, dropped)


def test_missing_or_repeated_controls_are_rejected():
    for step in (0, 2):
        state = StableTruth()
        state.update(0, ready())
        with pytest.raises(ValueError):
            state.update(step, ready())


def test_probabilities_require_evidence_for_a_known_class():
    assert classify([0.02, 0.96, 0.02]) == "complete"
    assert classify([0.95, 0.03, 0.02]) == "incomplete"
    assert classify([0.02, 0.02, 0.96]) == "unknown"
    assert classify([0.3, 0.6, 0.1]) == "unknown"
    with pytest.raises(ValueError):
        classify([np.nan, 0, 1])
    with pytest.raises(ValueError):
        classify([0, 0, 0])
