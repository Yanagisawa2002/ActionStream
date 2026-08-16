from __future__ import annotations

import numpy as np

from actionstream.xvla_official_render_bridge_canary import (
    CANDIDATE_CONDITION,
    EXPECTED_ACCEPTANCE,
    EXPECTED_SEEDS,
    REFERENCE_CONDITION,
    evaluate_acceptance,
    summarize_records,
)


def _records(delta: float = 0.01, gripper: float = -1.0) -> list[dict]:
    records = []
    for seed in EXPECTED_SEEDS:
        reference = np.zeros((30, 7), dtype=np.float32)
        reference[:, 6] = -1.0
        candidate = reference.copy()
        candidate[:, :6] += delta
        candidate[:, 6] = gripper
        records.extend(
            [
                {
                    "inference_seed": seed,
                    "condition": REFERENCE_CONDITION,
                    "actions": reference.tolist(),
                },
                {
                    "inference_seed": seed,
                    "condition": CANDIDATE_CONDITION,
                    "actions": candidate.tolist(),
                },
            ]
        )
    return records


def test_bridge_canary_accepts_small_paired_displacement_and_close_gripper() -> None:
    analysis = summarize_records(_records())
    acceptance = evaluate_acceptance(
        analysis, {"all_required": True, **EXPECTED_ACCEPTANCE}
    )
    assert acceptance["all_acceptance_checks_pass"]
    assert acceptance["checks"]["candidate_gripper_negative_fraction_each_seed"][
        "values"
    ] == [1.0, 1.0, 1.0]


def test_bridge_canary_rejects_wrong_gripper_even_with_identical_pose() -> None:
    analysis = summarize_records(_records(delta=0.0, gripper=1.0))
    acceptance = evaluate_acceptance(
        analysis, {"all_required": True, **EXPECTED_ACCEPTANCE}
    )
    assert not acceptance["all_acceptance_checks_pass"]
    assert not acceptance["checks"][
        "candidate_gripper_negative_fraction_each_seed"
    ]["pass"]
