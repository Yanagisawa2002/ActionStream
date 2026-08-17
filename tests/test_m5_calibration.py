from __future__ import annotations

from actionstream.m5_calibration import evaluate_magnitude, select_calibration


def _rows(
    magnitude: int,
    aligned: list[bool],
    gate: list[bool],
    *,
    valid: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (aligned_success, gate_success) in enumerate(
        zip(aligned, gate, strict=True)
    ):
        common = {
            "task_id": 0,
            "seed": 100 + index,
            "initial_state_index": index,
            "displacement_magnitude_mm": magnitude,
            "perturbation_valid": valid,
            "shift_step": 19,
            "achieved_displacement_xyz_m": [-magnitude / 1000.0, 0.0, 0.0],
        }
        rows.extend(
            (
                {**common, "condition": "aligned_shift", "success": aligned_success},
                {
                    **common,
                    "condition": "aligned_oracle_pose_gate_shift",
                    "success": gate_success,
                },
            )
        )
    return rows


def test_calibration_selects_first_qualifying_magnitude() -> None:
    evaluation = evaluate_magnitude(
        _rows(50, [True, True, False, False, False], [True] * 5)
    )
    selection = select_calibration([evaluation], task_id=0)
    assert selection["status"] == "selected"
    assert selection["selected_displacement_magnitude_mm"] == 50


def test_calibration_follows_stronger_rule() -> None:
    first = evaluate_magnitude(_rows(50, [True] * 5, [True] * 5))
    assert select_calibration([first], task_id=0)["status"] == "needs_70_mm"
    stronger = evaluate_magnitude(
        _rows(70, [True, False, False, False, False], [True] * 5)
    )
    selected = select_calibration([first, stronger], task_id=0)
    assert selected["selected_displacement_magnitude_mm"] == 70


def test_calibration_follows_weaker_rule_and_rejects_invalid_shifts() -> None:
    first = evaluate_magnitude(_rows(50, [False] * 5, [True, True, True, False, False]))
    assert select_calibration([first], task_id=0)["status"] == "needs_30_mm"
    weaker = evaluate_magnitude(_rows(30, [False] * 5, [True] * 5, valid=False))
    selection = select_calibration([first, weaker], task_id=0)
    assert selection["status"] == "candidate_no_go"
