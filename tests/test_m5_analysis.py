from __future__ import annotations

import json

import pytest

from actionstream.m5_analysis import (
    ALIGNED_NO_SHIFT,
    ALIGNED_SHIFT,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    GATED_NO_SHIFT,
    GATED_SHIFT,
    build_m5_analysis,
    exact_two_sided_mcnemar_p,
    pair_episode_rows,
    paired_bootstrap_ci,
    read_episode_jsonl,
    validate_episode_rows,
)


def _row(
    condition: str,
    pair_index: int,
    *,
    success: bool,
    stale_duration: float,
) -> dict[str, object]:
    shifted = condition in {ALIGNED_SHIFT, GATED_SHIFT}
    gated = condition in {GATED_SHIFT, GATED_NO_SHIFT}
    seed = 53001 + pair_index
    state = 20 + pair_index
    return {
        "condition": condition,
        "seed": seed,
        "initial_state_index": state,
        "phase": "sealed" if shifted else "no_shift",
        "task_id": 0,
        "task_instruction": "pick up the alphabet soup and place it in the basket",
        "moved_entity": "basket_1",
        "initial_source_entity_pose": {"position": [0.0, 0.0, 0.0]},
        "initial_moved_entity_pose": {"position": [0.0, 0.25, 0.0]},
        "policy_rng_seed": seed,
        "model_id": "lerobot/xvla-libero",
        "model_revision_sha": "frozen-revision",
        "suite": "libero_object",
        "implementation_source_sha256": "source-sha",
        "experiment_config_sha256": "config-sha",
        "seed_manifest_sha256": "manifest-sha",
        "protocol_decision_sha256": "protocol-sha",
        "no_shift_decision_sha256": "no-shift-sha" if shifted else None,
        "injected_delay_ms": 950,
        "controller_frequency_hz": 20.0,
        "chunk_size_steps": 30,
        "replan_interval_steps": 10,
        "nominal_queue_headroom_steps": 20,
        "scheduled_shift_step": 19 if shifted else None,
        "shift_step": 19 if shifted else None,
        "displacement_magnitude_mm": 50 if shifted else 0,
        "requested_displacement_xy_m": [-0.05, 0.0] if shifted else None,
        "achieved_displacement_xyz_m": ([-0.05, 0.0, 0.0] if shifted else None),
        "success": success,
        "stale_action_duration_seconds": stale_duration,
        "stale_action_steps": (int(round(stale_duration * 20.0)) if shifted else 0),
        "environment_steps": 100 + pair_index + (2 if gated else 0),
        "simulated_completion_time_seconds": 5.0
        + pair_index / 20
        + (0.1 if gated else 0),
        "wall_clock_episode_seconds": 4.0 + pair_index / 25 + (0.2 if gated else 0),
        "time_from_shift_to_gate_trigger_seconds": (
            0.02 if shifted and gated else None
        ),
        "time_from_shift_to_first_fresh_action_seconds": (1.2 if shifted else None),
        "time_from_shift_to_queue_clear_seconds": 0.1 if shifted else None,
        "perturbation_valid": True,
        "invalid_reason": None,
        "false_gate_count": 0,
        "scene_gate_trigger_count": int(shifted and gated),
        "queue_invalidation_count": int(shifted and gated),
    }


def _full_go_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_index in range(30):
        aligned_success = pair_index < 18
        gate_success = pair_index < 24
        aligned_duration = 1.0 + pair_index / 100
        rows.extend(
            [
                _row(
                    ALIGNED_SHIFT,
                    pair_index,
                    success=aligned_success,
                    stale_duration=aligned_duration,
                ),
                _row(
                    GATED_SHIFT,
                    pair_index,
                    success=gate_success,
                    stale_duration=0.2 * aligned_duration,
                ),
            ]
        )
    for pair_index in range(10):
        for condition in (ALIGNED_NO_SHIFT, GATED_NO_SHIFT):
            rows.append(
                _row(
                    condition,
                    pair_index,
                    success=True,
                    stale_duration=0.0,
                )
            )
    return rows


def test_exact_two_sided_mcnemar_uses_only_discordant_pairs() -> None:
    assert exact_two_sided_mcnemar_p(0, 0) == 1.0
    assert exact_two_sided_mcnemar_p(0, 6) == pytest.approx(0.03125)
    assert exact_two_sided_mcnemar_p(2, 2) == 1.0

    with pytest.raises(ValueError, match="nonnegative integers"):
        exact_two_sided_mcnemar_p(-1, 2)


def test_paired_bootstrap_is_deterministic_and_has_gate_minus_aligned_sign() -> None:
    first = paired_bootstrap_ci([1.0, 2.0, 4.0], [0.5, 1.0, 3.0])
    second = paired_bootstrap_ci([1.0, 2.0, 4.0], [0.5, 1.0, 3.0])

    assert first == second
    assert first[1] < 0


def test_full_go_statistics_and_classification() -> None:
    report = build_m5_analysis(
        _full_go_rows(),
        m4_baseline_intact=True,
        source_validation_passed=True,
    )

    shift = report["shift_comparison"]
    success = shift["success"]
    stale = shift["stale_action_duration_seconds"]
    assert report["bootstrap"] == {
        "method": "paired nonparametric percentile bootstrap",
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "confidence_level": 0.95,
    }
    assert success["aligned"]["success_count"] == 18
    assert success["gate"]["success_count"] == 24
    assert success["paired_success_difference_gate_minus_aligned"] == pytest.approx(0.2)
    assert success["gate_only_success_count"] == 6
    assert success["aligned_only_success_count"] == 0
    assert success["both_success_count"] == 18
    assert success["both_failure_count"] == 6
    assert success["mcnemar_exact_two_sided_p"] == pytest.approx(0.03125)
    assert stale["mean_reduction_fraction"] == pytest.approx(0.8)
    assert stale["paired_mean_difference"] < 0
    assert stale["paired_mean_reduction"] > 0
    assert stale["paired_mean_reduction_95pct_bootstrap_ci"][0] > 0
    assert report["no_shift_comparison"]["safety_requirements_pass"]
    assert report["classification"]["label"] == "FULL GO"


def test_mechanism_pass_when_success_is_unchanged() -> None:
    rows = _full_go_rows()
    for row in rows:
        if row["condition"] == GATED_SHIFT:
            pair_index = int(row["seed"]) - 53001
            row["success"] = pair_index < 18

    report = build_m5_analysis(
        rows,
        m4_baseline_intact=True,
        source_validation_passed=True,
    )

    assert report["classification"]["label"] == "MECHANISM PASS / OUTCOME INCONCLUSIVE"
    assert (
        report["shift_comparison"]["success"][
            "paired_success_difference_gate_minus_aligned"
        ]
        == 0
    )


@pytest.mark.parametrize(
    "field",
    [
        "false_gate_count",
        "scene_gate_trigger_count",
        "queue_invalidation_count",
        "stale_action_steps",
    ],
)
def test_no_shift_counter_safety_failure_forces_no_go(field: str) -> None:
    rows = _full_go_rows()
    gated_no_shift = next(row for row in rows if row["condition"] == GATED_NO_SHIFT)
    gated_no_shift[field] = 1

    report = build_m5_analysis(
        rows,
        m4_baseline_intact=True,
        source_validation_passed=True,
    )

    safety = report["no_shift_comparison"]["safety_checks"]
    assert not safety[f"zero_{field}"]
    assert report["classification"]["label"] == "NO-GO"


def test_no_shift_success_degradation_over_five_points_forces_no_go() -> None:
    rows = _full_go_rows()
    gated_no_shift = next(row for row in rows if row["condition"] == GATED_NO_SHIFT)
    gated_no_shift["success"] = False

    report = build_m5_analysis(
        rows,
        m4_baseline_intact=True,
        source_validation_passed=True,
    )

    safety = report["no_shift_comparison"]["safety_checks"]
    assert not safety["success_degradation_no_more_than_5pp"]
    assert report["classification"]["label"] == "NO-GO"


def test_invalid_shift_excludes_the_whole_pair_and_preserves_reason() -> None:
    rows = _full_go_rows()
    invalid = next(
        row for row in rows if row["condition"] == GATED_SHIFT and row["seed"] == 53001
    )
    invalid["perturbation_valid"] = False
    invalid["invalid_reason"] = "workspace collision"

    report = build_m5_analysis(
        rows,
        m4_baseline_intact=True,
        source_validation_passed=True,
    )

    assert report["shift_comparison"]["valid_pair_count"] == 29
    assert report["invalid_episode_count"] == 1
    assert report["invalid_episodes"][0]["reason"] == "workspace collision"
    assert report["excluded_pairs"][0]["excluded_episode_count"] == 2
    assert report["classification"]["label"] == "NO-GO"


def test_incomplete_pair_is_reported_instead_of_silently_dropped() -> None:
    rows = _full_go_rows()
    rows.remove(
        next(
            row
            for row in rows
            if row["condition"] == GATED_SHIFT and row["seed"] == 53001
        )
    )
    validate_episode_rows(rows)

    pairing = pair_episode_rows(
        rows,
        reference_condition=ALIGNED_SHIFT,
        estimate_condition=GATED_SHIFT,
        require_valid_perturbation=True,
    )

    assert pairing["valid_pair_count"] == 29
    assert pairing["excluded_pair_count"] == 1
    assert pairing["excluded_pairs"][0]["reasons"] == [
        f"missing condition {GATED_SHIFT}"
    ]


def test_pair_contract_drift_excludes_the_pair_with_exact_reason() -> None:
    rows = _full_go_rows()
    drifted = next(
        row for row in rows if row["condition"] == GATED_SHIFT and row["seed"] == 53001
    )
    drifted["injected_delay_ms"] = 951

    report = build_m5_analysis(
        rows,
        m4_baseline_intact=True,
        source_validation_passed=True,
    )

    assert report["shift_comparison"]["valid_pair_count"] == 29
    reasons = report["excluded_pairs"][0]["reasons"]
    assert reasons == ["pair contract mismatch for injected_delay_ms: 950 != 951"]


def test_calibration_no_go_does_not_require_sealed_rows() -> None:
    report = build_m5_analysis(
        [],
        m4_baseline_intact=True,
        source_validation_passed=True,
        calibration_no_go_reason="No magnitude met the frozen challenge rule",
    )

    assert report["shift_comparison"] is None
    assert report["classification"]["label"] == "NO-GO"
    assert report["classification"]["reasons"] == [
        "Calibration no-go: No magnitude met the frozen challenge rule"
    ]


def test_jsonl_reader_and_schema_errors_are_explicit(tmp_path) -> None:
    path = tmp_path / "episodes.jsonl"
    row = _row(ALIGNED_SHIFT, 0, success=True, stale_duration=1.0)
    path.write_text("\n" + json.dumps(row) + "\n", encoding="utf-8")

    assert read_episode_jsonl([path]) == [row]

    row.pop("success")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_episode_rows([row])
