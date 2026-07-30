from __future__ import annotations

import pytest

from actionstream.m4_calibration import (
    choose_pressure_delay,
    derive_candidate_delays,
)


def test_candidate_delays_target_derived_queue_headroom() -> None:
    candidates, extension = derive_candidate_delays(
        control_step_seconds=0.05,
        base_delivery_seconds=0.1,
        queue_headroom_steps=20,
    )

    assert [row["injected_delay_ms"] for row in candidates] == [
        400,
        650,
        900,
        1100,
    ]
    assert [row["target_delivery_age_steps"] for row in candidates] == [
        10,
        15,
        20,
        24,
    ]
    assert extension["injected_delay_ms"] == 1400
    assert extension["target_delivery_age_steps"] == 30


def test_selection_uses_smallest_pressure_delay() -> None:
    metrics = []
    for delay_ms, median_age, holds in (
        (400, 10.0, 0),
        (650, 15.0, 1),
        (950, 21.0, 0),
    ):
        for mode in ("async_naive", "async_aligned"):
            metrics.append(
                {
                    "runtime_mode": mode,
                    "injected_delay_ms": delay_ms,
                    "effective_delivery_age_steps": {"median": median_age},
                    "queue_hold_steps_total": holds if mode == "async_naive" else 0,
                    "fully_stale_chunks_total": 0,
                }
            )

    selected, decisions = choose_pressure_delay(
        metrics,
        queue_headroom_steps=20,
    )

    assert selected == 650
    assert not decisions[0]["qualifies_as_pressure"]
    assert decisions[1]["triggers"]["nonzero_queue_holds"]
    assert decisions[2]["triggers"]["median_age_reaches_headroom"]


def test_selection_requires_both_async_modes() -> None:
    with pytest.raises(ValueError, match="missing a calibration mode"):
        choose_pressure_delay(
            [
                {
                    "runtime_mode": "async_naive",
                    "injected_delay_ms": 400,
                    "effective_delivery_age_steps": {"median": 10.0},
                    "queue_hold_steps_total": 0,
                    "fully_stale_chunks_total": 0,
                }
            ],
            queue_headroom_steps=20,
        )
