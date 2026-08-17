"""Paired episode-level statistics for the ActionStream M5-G0 experiment.

This module intentionally stops at machine-readable statistics.  Report and plot
generation belong to the later M5 reporting step.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


BOOTSTRAP_SEED = 20260731
BOOTSTRAP_RESAMPLES = 10_000
SEALED_PAIR_COUNT = 30
NO_SHIFT_PAIR_COUNT = 10

ALIGNED_SHIFT = "aligned_shift"
GATED_SHIFT = "aligned_oracle_pose_gate_shift"
ALIGNED_NO_SHIFT = "aligned_no_shift"
GATED_NO_SHIFT = "aligned_oracle_pose_gate_no_shift"
SHIFT_CONDITIONS = (ALIGNED_SHIFT, GATED_SHIFT)
NO_SHIFT_CONDITIONS = (ALIGNED_NO_SHIFT, GATED_NO_SHIFT)
CONDITIONS = frozenset((*SHIFT_CONDITIONS, *NO_SHIFT_CONDITIONS))

# M5 evaluator/runtime schema assumptions:
# - a pair is identified by both seed and initial_state_index; neither field alone
#   is assumed unique;
# - perturbation_valid applies to shifted episodes.  If either shifted member is
#   invalid, both members are excluded so pairing is never broken;
# - shift-relative timing values may be null when the event was never reached;
# - false_gate_count, scene_gate_trigger_count, queue_invalidation_count, and
#   stale_action_steps are explicit evaluator counters, not reconstructed here.
PAIR_FIELDS = ("seed", "initial_state_index")
PAIR_CONTRACT_FIELDS = (
    "phase",
    "task_id",
    "task_instruction",
    "moved_entity",
    "initial_source_entity_pose",
    "initial_moved_entity_pose",
    "policy_rng_seed",
    "model_id",
    "model_revision_sha",
    "suite",
    "implementation_source_sha256",
    "experiment_config_sha256",
    "seed_manifest_sha256",
    "protocol_decision_sha256",
    "no_shift_decision_sha256",
    "injected_delay_ms",
    "controller_frequency_hz",
    "chunk_size_steps",
    "replan_interval_steps",
    "nominal_queue_headroom_steps",
    "scheduled_shift_step",
    "shift_step",
    "displacement_magnitude_mm",
    "requested_displacement_xy_m",
    "achieved_displacement_xyz_m",
)
REQUIRED_FIELDS = frozenset(
    {
        "condition",
        *PAIR_FIELDS,
        *PAIR_CONTRACT_FIELDS,
        "success",
        "stale_action_duration_seconds",
        "stale_action_steps",
        "environment_steps",
        "simulated_completion_time_seconds",
        "wall_clock_episode_seconds",
        "time_from_shift_to_gate_trigger_seconds",
        "time_from_shift_to_first_fresh_action_seconds",
        "time_from_shift_to_queue_clear_seconds",
        "perturbation_valid",
        "invalid_reason",
        "false_gate_count",
        "scene_gate_trigger_count",
        "queue_invalidation_count",
    }
)
NONNEGATIVE_FLOAT_FIELDS = (
    "stale_action_duration_seconds",
    "simulated_completion_time_seconds",
    "wall_clock_episode_seconds",
)
NULLABLE_NONNEGATIVE_FLOAT_FIELDS = (
    "time_from_shift_to_gate_trigger_seconds",
    "time_from_shift_to_first_fresh_action_seconds",
    "time_from_shift_to_queue_clear_seconds",
)
NONNEGATIVE_INTEGER_FIELDS = (
    "seed",
    "initial_state_index",
    "stale_action_steps",
    "environment_steps",
    "false_gate_count",
    "scene_gate_trigger_count",
    "queue_invalidation_count",
)


def read_episode_jsonl(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    """Read JSON-object episode rows from one or more JSONL files."""

    rows: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{path}:{line_number}: episode row must be a JSON object"
                    )
                rows.append(row)
    return rows


def _is_nonnegative_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def validate_episode_row(row: dict[str, Any], *, row_index: int) -> None:
    """Validate one row without mutating or normalizing source evidence."""

    missing = sorted(REQUIRED_FIELDS.difference(row))
    if missing:
        raise ValueError(f"Row {row_index} is missing required fields: {missing}")
    condition = row["condition"]
    if condition not in CONDITIONS:
        raise ValueError(f"Row {row_index} has unexpected condition {condition!r}")
    if type(row["success"]) is not bool:
        raise ValueError(f"Row {row_index} success must be boolean")
    if type(row["perturbation_valid"]) is not bool:
        raise ValueError(f"Row {row_index} perturbation_valid must be boolean")
    if row["invalid_reason"] is not None and not isinstance(row["invalid_reason"], str):
        raise ValueError(f"Row {row_index} invalid_reason must be a string or null")

    for field in NONNEGATIVE_INTEGER_FIELDS:
        if type(row[field]) is not int or row[field] < 0:
            raise ValueError(f"Row {row_index} {field} must be a nonnegative integer")
    for field in NONNEGATIVE_FLOAT_FIELDS:
        if not _is_nonnegative_finite_number(row[field]):
            raise ValueError(
                f"Row {row_index} {field} must be a finite nonnegative number"
            )
    for field in NULLABLE_NONNEGATIVE_FLOAT_FIELDS:
        value = row[field]
        if value is not None and not _is_nonnegative_finite_number(value):
            raise ValueError(
                f"Row {row_index} {field} must be null or a finite "
                "nonnegative number"
            )

    if (
        condition in SHIFT_CONDITIONS
        and not row["perturbation_valid"]
        and not str(row["invalid_reason"] or "").strip()
    ):
        raise ValueError(
            f"Row {row_index} is an invalid shifted perturbation without a reason"
        )


def validate_episode_rows(rows: Sequence[dict[str, Any]]) -> None:
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {row_index} must be a dictionary")
        validate_episode_row(row, row_index=row_index)


def _pair_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["seed"]), int(row["initial_state_index"])


def _key_payload(key: tuple[int, int]) -> dict[str, int]:
    return {"seed": key[0], "initial_state_index": key[1]}


def pair_episode_rows(
    rows: Sequence[dict[str, Any]],
    *,
    reference_condition: str,
    estimate_condition: str,
    require_valid_perturbation: bool,
) -> dict[str, Any]:
    """Pair two conditions and preserve every exclusion with a reason."""

    if reference_condition == estimate_condition:
        raise ValueError("Pairing requires two distinct conditions")
    if reference_condition not in CONDITIONS or estimate_condition not in CONDITIONS:
        raise ValueError("Pairing conditions must be registered M5 conditions")

    indexed: dict[tuple[int, int, str], dict[str, Any]] = {}
    selected_conditions = {reference_condition, estimate_condition}
    for row in rows:
        if row["condition"] not in selected_conditions:
            continue
        key = (*_pair_key(row), str(row["condition"]))
        if key in indexed:
            raise ValueError(f"Duplicate episode row for {key}")
        indexed[key] = row

    pair_keys = sorted({(key[0], key[1]) for key in indexed})
    valid_pairs: list[dict[str, Any]] = []
    excluded_pairs: list[dict[str, Any]] = []
    invalid_episodes: list[dict[str, Any]] = []
    for key in pair_keys:
        reference = indexed.get((*key, reference_condition))
        estimate = indexed.get((*key, estimate_condition))
        reasons: list[str] = []
        if reference is None:
            reasons.append(f"missing condition {reference_condition}")
        if estimate is None:
            reasons.append(f"missing condition {estimate_condition}")

        if require_valid_perturbation:
            for condition, row in (
                (reference_condition, reference),
                (estimate_condition, estimate),
            ):
                if row is not None and not row["perturbation_valid"]:
                    reason = str(row["invalid_reason"])
                    reasons.append(f"{condition}: invalid perturbation: {reason}")
                    invalid_episodes.append(
                        {
                            **_key_payload(key),
                            "condition": condition,
                            "reason": reason,
                        }
                    )

        if reference is not None and estimate is not None:
            for field in PAIR_CONTRACT_FIELDS:
                if reference[field] != estimate[field]:
                    reasons.append(
                        f"pair contract mismatch for {field}: "
                        f"{reference[field]!r} != {estimate[field]!r}"
                    )

        if reasons:
            excluded_pairs.append(
                {
                    **_key_payload(key),
                    "conditions_present": sorted(
                        condition
                        for condition, row in (
                            (reference_condition, reference),
                            (estimate_condition, estimate),
                        )
                        if row is not None
                    ),
                    "reasons": reasons,
                    "excluded_episode_count": int(reference is not None)
                    + int(estimate is not None),
                }
            )
            continue

        assert reference is not None and estimate is not None
        valid_pairs.append(
            {
                **_key_payload(key),
                "reference": reference,
                "estimate": estimate,
            }
        )

    return {
        "reference_condition": reference_condition,
        "estimate_condition": estimate_condition,
        "valid_pairs": valid_pairs,
        "valid_pair_count": len(valid_pairs),
        "excluded_pairs": excluded_pairs,
        "excluded_pair_count": len(excluded_pairs),
        "invalid_episodes": invalid_episodes,
        "invalid_episode_count": len(invalid_episodes),
    }


def _as_finite_vector(values: Iterable[float], *, name: str) -> np.ndarray:
    vector = np.asarray(list(values), dtype=np.float64)
    if vector.ndim != 1 or not vector.size:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def paired_bootstrap_ci(
    reference: Iterable[float],
    estimate: Iterable[float],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Return a percentile CI for the paired mean ``estimate - reference``."""

    reference_values = _as_finite_vector(reference, name="reference")
    estimate_values = _as_finite_vector(estimate, name="estimate")
    if reference_values.shape != estimate_values.shape:
        raise ValueError("Paired vectors must have the same length")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")

    differences = estimate_values - reference_values
    indices = np.random.default_rng(bootstrap_seed).integers(
        0,
        differences.size,
        size=(bootstrap_resamples, differences.size),
    )
    lower, upper = np.percentile(differences[indices].mean(axis=1), [2.5, 97.5])
    return float(lower), float(upper)


def exact_two_sided_mcnemar_p(
    aligned_only_successes: int,
    gate_only_successes: int,
) -> float:
    """Compute the conventional exact two-sided McNemar binomial p-value."""

    if (
        type(aligned_only_successes) is not int
        or type(gate_only_successes) is not int
        or aligned_only_successes < 0
        or gate_only_successes < 0
    ):
        raise ValueError("Discordant counts must be nonnegative integers")
    discordant = aligned_only_successes + gate_only_successes
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(aligned_only_successes, gate_only_successes) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _summary(values: Iterable[float]) -> dict[str, Any]:
    vector = np.asarray(list(values), dtype=np.float64)
    if vector.ndim != 1 or (vector.size and not np.isfinite(vector).all()):
        raise ValueError("Summary values must be a finite one-dimensional vector")
    if not vector.size:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": int(vector.size),
        "mean": float(vector.mean()),
        "median": float(np.median(vector)),
    }


def paired_numeric_effect(
    reference: Iterable[float],
    estimate: Iterable[float],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Summarize a paired numeric effect using ``estimate - reference``."""

    reference_values = _as_finite_vector(reference, name="reference")
    estimate_values = _as_finite_vector(estimate, name="estimate")
    if reference_values.shape != estimate_values.shape:
        raise ValueError("Paired vectors must have the same length")
    differences = estimate_values - reference_values
    ci = paired_bootstrap_ci(
        reference_values,
        estimate_values,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    return {
        "paired_count": int(differences.size),
        "reference": _summary(reference_values),
        "estimate": _summary(estimate_values),
        "paired_mean_difference": float(differences.mean()),
        "paired_median_difference": float(np.median(differences)),
        "paired_mean_difference_95pct_bootstrap_ci": [ci[0], ci[1]],
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": bootstrap_resamples,
    }


def _empty_paired_effect() -> dict[str, Any]:
    return {
        "paired_count": 0,
        "reference": _summary([]),
        "estimate": _summary([]),
        "paired_mean_difference": None,
        "paired_median_difference": None,
        "paired_mean_difference_95pct_bootstrap_ci": None,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }


def _field_effect(pairs: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    if not pairs:
        return _empty_paired_effect()
    return paired_numeric_effect(
        (float(pair["reference"][field]) for pair in pairs),
        (float(pair["estimate"][field]) for pair in pairs),
    )


def _nullable_field_effect(
    pairs: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    complete = [
        pair
        for pair in pairs
        if pair["reference"][field] is not None and pair["estimate"][field] is not None
    ]
    effect = _field_effect(complete, field)
    return {
        **effect,
        "pairs_without_both_values": len(pairs) - len(complete),
    }


def _success_analysis(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aligned = [bool(pair["reference"]["success"]) for pair in pairs]
    gate = [bool(pair["estimate"]["success"]) for pair in pairs]
    both_success = sum(
        reference_success and estimate_success
        for reference_success, estimate_success in zip(aligned, gate, strict=True)
    )
    aligned_only = sum(
        reference_success and not estimate_success
        for reference_success, estimate_success in zip(aligned, gate, strict=True)
    )
    gate_only = sum(
        not reference_success and estimate_success
        for reference_success, estimate_success in zip(aligned, gate, strict=True)
    )
    both_failure = len(pairs) - both_success - aligned_only - gate_only
    effect = paired_numeric_effect(aligned, gate) if pairs else _empty_paired_effect()
    return {
        "aligned": {
            "success_count": sum(aligned),
            "episode_count": len(aligned),
            "success_rate": (sum(aligned) / len(aligned)) if aligned else None,
        },
        "gate": {
            "success_count": sum(gate),
            "episode_count": len(gate),
            "success_rate": (sum(gate) / len(gate)) if gate else None,
        },
        "paired_success_difference_gate_minus_aligned": effect[
            "paired_mean_difference"
        ],
        "paired_success_difference_95pct_bootstrap_ci": effect[
            "paired_mean_difference_95pct_bootstrap_ci"
        ],
        "gate_only_success_count": gate_only,
        "aligned_only_success_count": aligned_only,
        "both_success_count": both_success,
        "both_failure_count": both_failure,
        "discordant_pair_count": gate_only + aligned_only,
        "mcnemar_exact_two_sided_p": exact_two_sided_mcnemar_p(
            aligned_only,
            gate_only,
        ),
    }


def _stale_duration_analysis(
    pairs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not pairs:
        return {
            **_empty_paired_effect(),
            "paired_mean_reduction": None,
            "paired_median_reduction": None,
            "paired_mean_reduction_95pct_bootstrap_ci": None,
            "mean_reduction_fraction": None,
            "mean_reduction_percent": None,
        }
    aligned = [
        float(pair["reference"]["stale_action_duration_seconds"]) for pair in pairs
    ]
    gate = [float(pair["estimate"]["stale_action_duration_seconds"]) for pair in pairs]
    effect = paired_numeric_effect(aligned, gate)
    reductions = np.asarray(aligned, dtype=np.float64) - np.asarray(
        gate, dtype=np.float64
    )
    reduction_ci = paired_bootstrap_ci(gate, aligned)
    aligned_mean = float(np.mean(aligned))
    reduction_fraction = (
        float(reductions.mean() / aligned_mean) if aligned_mean > 0.0 else None
    )
    return {
        **effect,
        "difference_definition": "gate minus aligned; negative favors gate",
        "paired_mean_reduction": float(reductions.mean()),
        "paired_median_reduction": float(np.median(reductions)),
        "paired_mean_reduction_95pct_bootstrap_ci": [
            reduction_ci[0],
            reduction_ci[1],
        ],
        "reduction_definition": "aligned minus gate; positive favors gate",
        "mean_reduction_fraction": reduction_fraction,
        "mean_reduction_percent": (
            100.0 * reduction_fraction if reduction_fraction is not None else None
        ),
    }


def analyze_shift_comparison(pairing: dict[str, Any]) -> dict[str, Any]:
    """Analyze valid aligned-shift versus gated-shift episode pairs."""

    pairs = pairing["valid_pairs"]
    return {
        "reference_condition": pairing["reference_condition"],
        "estimate_condition": pairing["estimate_condition"],
        "valid_pair_count": len(pairs),
        "success": _success_analysis(pairs),
        "stale_action_duration_seconds": _stale_duration_analysis(pairs),
        "environment_steps": _field_effect(pairs, "environment_steps"),
        "simulated_completion_time_seconds": _field_effect(
            pairs, "simulated_completion_time_seconds"
        ),
        "wall_clock_episode_seconds": _field_effect(
            pairs, "wall_clock_episode_seconds"
        ),
        "time_from_shift_to_gate_trigger_seconds": _nullable_field_effect(
            pairs, "time_from_shift_to_gate_trigger_seconds"
        ),
        "time_from_shift_to_first_fresh_action_seconds": _nullable_field_effect(
            pairs, "time_from_shift_to_first_fresh_action_seconds"
        ),
        "time_from_shift_to_queue_clear_seconds": _nullable_field_effect(
            pairs, "time_from_shift_to_queue_clear_seconds"
        ),
        "excluded_pairs": pairing["excluded_pairs"],
        "invalid_episodes": pairing["invalid_episodes"],
    }


def _counter_summary(
    pairs: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    reference = sum(int(pair["reference"][field]) for pair in pairs)
    estimate = sum(int(pair["estimate"][field]) for pair in pairs)
    return {
        "aligned_total": reference,
        "gate_total": estimate,
        "combined_total": reference + estimate,
    }


def analyze_no_shift_comparison(pairing: dict[str, Any]) -> dict[str, Any]:
    """Analyze no-shift deltas and the predeclared safety checks."""

    pairs = pairing["valid_pairs"]
    success = _success_analysis(pairs)
    false_gates = _counter_summary(pairs, "false_gate_count")
    scene_gate_triggers = _counter_summary(pairs, "scene_gate_trigger_count")
    queue_invalidations = _counter_summary(pairs, "queue_invalidation_count")
    stale_steps = _counter_summary(pairs, "stale_action_steps")
    success_difference = success["paired_success_difference_gate_minus_aligned"]
    checks = {
        "complete_10_pairs": len(pairs) == NO_SHIFT_PAIR_COUNT,
        "zero_false_gate_count": false_gates["combined_total"] == 0,
        "zero_scene_gate_trigger_count": scene_gate_triggers["combined_total"] == 0,
        "zero_queue_invalidation_count": queue_invalidations["combined_total"] == 0,
        "zero_stale_action_steps": stale_steps["combined_total"] == 0,
        "success_degradation_no_more_than_5pp": (
            success_difference is not None and success_difference >= -0.05 - 1e-12
        ),
    }
    return {
        "reference_condition": pairing["reference_condition"],
        "estimate_condition": pairing["estimate_condition"],
        "valid_pair_count": len(pairs),
        "success": success,
        "environment_steps": _field_effect(pairs, "environment_steps"),
        "simulated_completion_time_seconds": _field_effect(
            pairs, "simulated_completion_time_seconds"
        ),
        "wall_clock_episode_seconds": _field_effect(
            pairs, "wall_clock_episode_seconds"
        ),
        "stale_action_duration_seconds": _stale_duration_analysis(pairs),
        "false_gate_count": false_gates,
        "scene_gate_trigger_count": scene_gate_triggers,
        "queue_invalidation_count": queue_invalidations,
        "stale_action_steps": stale_steps,
        "safety_checks": checks,
        "safety_requirements_pass": all(checks.values()),
        "excluded_pairs": pairing["excluded_pairs"],
    }


def classify_m5_result(
    shift: dict[str, Any] | None,
    no_shift: dict[str, Any] | None,
    *,
    m4_baseline_intact: bool,
    source_validation_passed: bool,
    calibration_no_go_reason: str | None = None,
) -> dict[str, Any]:
    """Apply the frozen M5-G0 interpretation rules without post-hoc tuning."""

    if (
        type(m4_baseline_intact) is not bool
        or type(source_validation_passed) is not bool
    ):
        raise ValueError(
            "Baseline and source-validation gates must be explicit booleans"
        )

    success_difference = (
        shift["success"]["paired_success_difference_gate_minus_aligned"]
        if shift is not None
        else None
    )
    mcnemar_p = (
        shift["success"]["mcnemar_exact_two_sided_p"] if shift is not None else None
    )
    stale = shift["stale_action_duration_seconds"] if shift is not None else {}
    reduction_fraction = stale.get("mean_reduction_fraction")
    reduction_ci = stale.get("paired_mean_reduction_95pct_bootstrap_ci")
    stale_reduction_at_least_70pct = (
        reduction_fraction is not None and reduction_fraction >= 0.70 - 1e-12
    )
    reduction_ci_excludes_zero = bool(
        reduction_ci is not None and reduction_ci[0] > 0.0
    )
    success_improves_at_least_15pp = (
        success_difference is not None and success_difference >= 0.15 - 1e-12
    )
    mcnemar_significant = mcnemar_p is not None and mcnemar_p < 0.05
    success_not_regressed_more_than_5pp = (
        success_difference is not None and success_difference >= -0.05 - 1e-12
    )
    success_does_not_significantly_improve = not (
        success_difference is not None
        and success_difference > 0.0
        and mcnemar_significant
    )
    sealed_complete = bool(
        shift is not None and shift["valid_pair_count"] == SEALED_PAIR_COUNT
    )
    no_shift_safe = bool(no_shift is not None and no_shift["safety_requirements_pass"])
    common_evidence_valid = (
        sealed_complete
        and no_shift_safe
        and m4_baseline_intact
        and source_validation_passed
        and calibration_no_go_reason is None
    )

    criteria = {
        "sealed_30_pairs_complete": sealed_complete,
        "stale_duration_reduction_at_least_70pct": stale_reduction_at_least_70pct,
        "stale_duration_reduction_ci_excludes_zero": reduction_ci_excludes_zero,
        "success_improvement_at_least_15pp": success_improves_at_least_15pp,
        "mcnemar_p_below_0_05": mcnemar_significant,
        "success_not_regressed_more_than_5pp": success_not_regressed_more_than_5pp,
        "success_does_not_significantly_improve": (
            success_does_not_significantly_improve
        ),
        "no_shift_safety_requirements_pass": no_shift_safe,
        "m4_baseline_intact": m4_baseline_intact,
        "source_validation_passed": source_validation_passed,
        "calibration_challenge_valid": calibration_no_go_reason is None,
    }

    if calibration_no_go_reason is not None:
        return {
            "label": "NO-GO",
            "criteria": criteria,
            "reasons": [f"Calibration no-go: {calibration_no_go_reason}"],
        }

    full_go = (
        common_evidence_valid
        and stale_reduction_at_least_70pct
        and reduction_ci_excludes_zero
        and success_improves_at_least_15pp
        and mcnemar_significant
    )
    mechanism_pass = (
        common_evidence_valid
        and stale_reduction_at_least_70pct
        and success_not_regressed_more_than_5pp
        and success_does_not_significantly_improve
    )
    if full_go:
        label = "FULL GO"
    elif mechanism_pass:
        label = "MECHANISM PASS / OUTCOME INCONCLUSIVE"
    else:
        label = "NO-GO"

    reasons: list[str] = []
    if calibration_no_go_reason is not None:
        reasons.append(f"Calibration no-go: {calibration_no_go_reason}")
    if not sealed_complete and calibration_no_go_reason is None:
        reasons.append("The sealed evaluation does not contain 30 valid pairs")
    if not no_shift_safe:
        reasons.append("The no-shift safety requirements did not pass")
    if not m4_baseline_intact:
        reasons.append("The existing M4 baseline is not confirmed intact")
    if not source_validation_passed:
        reasons.append("Source/result validation did not pass")
    if (
        label == "NO-GO"
        and calibration_no_go_reason is None
        and not stale_reduction_at_least_70pct
    ):
        reasons.append("Stale-action duration reduction is below 70% or unavailable")
    if (
        label == "NO-GO"
        and success_difference is not None
        and success_difference < -0.05 - 1e-12
    ):
        reasons.append("The gate materially reduces task success by more than 5pp")
    if label == "NO-GO" and not reasons:
        reasons.append("Neither predeclared positive classification is satisfied")

    return {
        "label": label,
        "criteria": criteria,
        "reasons": reasons,
    }


def build_m5_analysis(
    rows: Sequence[dict[str, Any]],
    *,
    m4_baseline_intact: bool,
    source_validation_passed: bool,
    calibration_no_go_reason: str | None = None,
) -> dict[str, Any]:
    """Build the complete M5-G0 statistical payload from episode rows."""

    validate_episode_rows(rows)
    shift_pairing = pair_episode_rows(
        rows,
        reference_condition=ALIGNED_SHIFT,
        estimate_condition=GATED_SHIFT,
        require_valid_perturbation=True,
    )
    no_shift_pairing = pair_episode_rows(
        rows,
        reference_condition=ALIGNED_NO_SHIFT,
        estimate_condition=GATED_NO_SHIFT,
        require_valid_perturbation=False,
    )
    shift = (
        None
        if calibration_no_go_reason is not None and not shift_pairing["valid_pairs"]
        else analyze_shift_comparison(shift_pairing)
    )
    no_shift = (
        analyze_no_shift_comparison(no_shift_pairing)
        if no_shift_pairing["valid_pairs"] or no_shift_pairing["excluded_pairs"]
        else None
    )
    classification = classify_m5_result(
        shift,
        no_shift,
        m4_baseline_intact=m4_baseline_intact,
        source_validation_passed=source_validation_passed,
        calibration_no_go_reason=calibration_no_go_reason,
    )
    excluded_pairs = [
        {
            "comparison": "shift",
            **entry,
        }
        for entry in shift_pairing["excluded_pairs"]
    ] + [
        {
            "comparison": "no_shift",
            **entry,
        }
        for entry in no_shift_pairing["excluded_pairs"]
    ]
    return {
        "schema_version": 1,
        "milestone": "M5-G0",
        "analysis_unit": "paired episode",
        "episode_row_count": len(rows),
        "pairing_fields": list(PAIR_FIELDS),
        "difference_definition": "gate minus aligned unless marked as reduction",
        "bootstrap": {
            "method": "paired nonparametric percentile bootstrap",
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
        },
        "shift_comparison": shift,
        "no_shift_comparison": no_shift,
        "invalid_episodes": shift_pairing["invalid_episodes"],
        "invalid_episode_count": shift_pairing["invalid_episode_count"],
        "excluded_pairs": excluded_pairs,
        "excluded_pair_count": len(excluded_pairs),
        "excluded_episode_count": sum(
            entry["excluded_episode_count"] for entry in excluded_pairs
        ),
        "calibration_no_go_reason": calibration_no_go_reason,
        "classification": classification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_paths", nargs="*", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--validation",
        type=Path,
        help="Passing formal evidence validation used to set acceptance gates.",
    )
    parser.add_argument(
        "--m4-baseline-intact",
        action="store_true",
        help="Confirm that the existing M4 baseline/tests remain valid.",
    )
    parser.add_argument(
        "--source-validation-passed",
        action="store_true",
        help="Confirm result-schema, trace, and source validation.",
    )
    parser.add_argument(
        "--calibration-no-go-reason",
        help="Record a calibration no-go and allow sealed rows to be absent.",
    )
    args = parser.parse_args()
    m4_baseline_intact = args.m4_baseline_intact
    source_validation_passed = args.source_validation_passed
    if args.validation is not None:
        validation = json.loads(args.validation.read_text(encoding="utf-8"))
        if not isinstance(validation, dict):
            raise ValueError("Formal validation must be a JSON object")
        from actionstream.m5_protocol import canonical_sha256

        expected = validation.get("validation_sha256")
        core = {
            key: value
            for key, value in validation.items()
            if key != "validation_sha256"
        }
        if expected != canonical_sha256(core) or validation.get("status") != "pass":
            raise ValueError("Formal evidence validation hash/status is invalid")
        checks = validation.get("checks", {})
        m4_baseline_intact = checks.get("m4_baseline_intact") is True
        source_validation_passed = all(
            checks.get(field) is True
            for field in (
                "repository_tests_and_compile_passed",
                "frozen_source_binding_passed",
                "episode_schema_and_pair_contract_passed",
                "action_request_event_provenance_passed",
            )
        )
    if not args.episode_paths and args.calibration_no_go_reason is None:
        raise ValueError(
            "Episode evidence may be empty only for a calibration hard stop"
        )
    payload = build_m5_analysis(
        read_episode_jsonl(args.episode_paths),
        m4_baseline_intact=m4_baseline_intact,
        source_validation_passed=source_validation_passed,
        calibration_no_go_reason=args.calibration_no_go_reason,
    )
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            if existing != payload:
                raise ValueError(
                    f"Existing statistics differ from recomputed evidence: {args.output}"
                )
            print(f"{args.output} (validated existing)")
        else:
            args.output.write_text(rendered, encoding="utf-8")
            print(args.output)


if __name__ == "__main__":
    main()
