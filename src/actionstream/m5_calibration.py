"""Predeclared M5-G0 perturbation-magnitude calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from actionstream.m5_protocol import canonical_sha256, file_sha256


SHIFT_CONDITIONS = (
    "aligned_shift",
    "aligned_oracle_pose_gate_shift",
)


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def evaluate_magnitude(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Calibration magnitude has no episode rows")
    magnitudes = {int(row["displacement_magnitude_mm"]) for row in rows}
    task_ids = {int(row["task_id"]) for row in rows}
    if len(magnitudes) != 1 or len(task_ids) != 1:
        raise ValueError(
            "One calibration evaluation must cover exactly one task and magnitude"
        )

    keyed: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        condition = str(row["condition"])
        if condition not in SHIFT_CONDITIONS:
            raise ValueError(f"Unexpected calibration condition: {condition}")
        key = (int(row["seed"]), int(row["initial_state_index"]), condition)
        if key in keyed:
            raise ValueError(f"Duplicate calibration row: {key}")
        keyed[key] = row

    pair_keys = sorted({(key[0], key[1]) for key in keyed})
    if len(pair_keys) != 5:
        raise ValueError(
            f"Calibration requires exactly five pairs, got {len(pair_keys)}"
        )
    pairs: list[dict[str, Any]] = []
    for seed, state in pair_keys:
        members = {
            condition: keyed.get((seed, state, condition))
            for condition in SHIFT_CONDITIONS
        }
        if any(row is None for row in members.values()):
            raise ValueError(
                f"Incomplete calibration pair for seed/state {(seed, state)}"
            )
        aligned = members["aligned_shift"]
        gate = members["aligned_oracle_pose_gate_shift"]
        assert aligned is not None and gate is not None
        pair = {
            "seed": seed,
            "initial_state_index": state,
            "aligned_success": bool(aligned["success"]),
            "gate_success": bool(gate["success"]),
            "aligned_perturbation_valid": bool(aligned["perturbation_valid"]),
            "gate_perturbation_valid": bool(gate["perturbation_valid"]),
            "gate_only_success": bool(gate["success"]) and not bool(aligned["success"]),
            "aligned_only_success": bool(aligned["success"])
            and not bool(gate["success"]),
            "same_shift_step": aligned.get("shift_step") == gate.get("shift_step"),
            "same_displacement": aligned.get("achieved_displacement_xyz_m")
            == gate.get("achieved_displacement_xyz_m"),
        }
        pairs.append(pair)

    aligned_successes = sum(pair["aligned_success"] for pair in pairs)
    gate_successes = sum(pair["gate_success"] for pair in pairs)
    gate_only = sum(pair["gate_only_success"] for pair in pairs)
    all_valid = all(
        pair["aligned_perturbation_valid"] and pair["gate_perturbation_valid"]
        for pair in pairs
    )
    paired_shift_match = all(
        pair["same_shift_step"] and pair["same_displacement"] for pair in pairs
    )
    qualifies = (
        all_valid
        and paired_shift_match
        and gate_successes >= 4
        and aligned_successes <= 3
        and gate_only >= 2
    )
    return {
        "task_id": next(iter(task_ids)),
        "displacement_magnitude_mm": next(iter(magnitudes)),
        "pair_count": len(pairs),
        "aligned_success_count": aligned_successes,
        "gate_success_count": gate_successes,
        "gate_only_success_count": gate_only,
        "aligned_only_success_count": sum(
            pair["aligned_only_success"] for pair in pairs
        ),
        "all_perturbations_physically_valid": all_valid,
        "paired_shift_parameters_identical": paired_shift_match,
        "qualifies": qualifies,
        "pairs": pairs,
    }


def select_calibration(
    evaluations: list[dict[str, Any]],
    *,
    task_id: int,
) -> dict[str, Any]:
    by_magnitude: dict[int, dict[str, Any]] = {}
    for evaluation in evaluations:
        if int(evaluation["task_id"]) != task_id:
            raise ValueError("Calibration selection mixed task candidates")
        magnitude = int(evaluation["displacement_magnitude_mm"])
        if magnitude in by_magnitude:
            raise ValueError(f"Magnitude {magnitude} mm was tested more than once")
        if magnitude not in {30, 50, 70}:
            raise ValueError(f"Unregistered calibration magnitude: {magnitude}")
        by_magnitude[magnitude] = evaluation
    if 50 not in by_magnitude:
        raise ValueError("The predetermined calibration must begin at 50 mm")

    for magnitude in (50, 70, 30):
        evaluation = by_magnitude.get(magnitude)
        if evaluation is not None and evaluation["qualifies"]:
            return {
                "status": "selected",
                "task_id": task_id,
                "selected_displacement_magnitude_mm": magnitude,
                "tested_magnitudes_mm": list(by_magnitude),
                "evaluations": [by_magnitude[key] for key in by_magnitude],
                "selection_rule": (
                    "first magnitude with 5/5 valid paired shifts, gate >=4/5, "
                    "aligned <=3/5, and >=2 gate-only successes"
                ),
            }

    first = by_magnitude[50]
    if len(by_magnitude) == 1:
        if int(first["gate_success_count"]) < 4:
            return {
                "status": "needs_30_mm",
                "task_id": task_id,
                "next_magnitude_mm": 30,
                "tested_magnitudes_mm": [50],
                "evaluations": [first],
            }
        if (
            int(first["gate_success_count"]) >= 4
            and int(first["aligned_success_count"]) >= 4
        ):
            return {
                "status": "needs_70_mm",
                "task_id": task_id,
                "next_magnitude_mm": 70,
                "tested_magnitudes_mm": [50],
                "evaluations": [first],
            }

    if 70 in by_magnitude and int(by_magnitude[70]["gate_success_count"]) < 4:
        if 30 not in by_magnitude:
            return {
                "status": "needs_30_mm",
                "task_id": task_id,
                "next_magnitude_mm": 30,
                "tested_magnitudes_mm": list(by_magnitude),
                "evaluations": [by_magnitude[key] for key in by_magnitude],
            }

    return {
        "status": "candidate_no_go",
        "task_id": task_id,
        "selected_displacement_magnitude_mm": None,
        "tested_magnitudes_mm": list(by_magnitude),
        "evaluations": [by_magnitude[key] for key in by_magnitude],
        "reason": "No magnitude met the frozen behavioral challenge rule",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_paths", type=Path, nargs="+")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in load_jsonl(args.episode_paths):
        grouped.setdefault(int(row["displacement_magnitude_mm"]), []).append(row)
    evaluations = [evaluate_magnitude(grouped[key]) for key in grouped]
    selection = select_calibration(evaluations, task_id=args.task_id)
    core = {
        "schema_version": 1,
        "milestone": "M5-G0",
        **selection,
        "source_episode_paths": [str(path) for path in args.episode_paths],
        "source_episode_files": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in args.episode_paths
        ],
    }
    payload = {**core, "selection_sha256": canonical_sha256(core)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"Existing calibration decision differs from evidence: {args.output}"
            )
        print(f"{args.output} (validated existing)")
    else:
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
