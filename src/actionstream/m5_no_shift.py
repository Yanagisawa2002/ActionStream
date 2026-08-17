"""Validate the paired M5-G0 no-shift safety and deterministic action trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from actionstream.m5_analysis import (
    ALIGNED_NO_SHIFT,
    GATED_NO_SHIFT,
    analyze_no_shift_comparison,
    pair_episode_rows,
    read_episode_jsonl,
    validate_episode_rows,
)
from actionstream.m5_protocol import canonical_sha256, file_sha256


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                rows.append(row)
    return rows


def _validate_protocol_decision(payload: dict[str, Any]) -> None:
    expected = payload.get("protocol_decision_sha256")
    core = {
        key: value
        for key, value in payload.items()
        if key != "protocol_decision_sha256"
    }
    actual = canonical_sha256(core)
    if expected != actual:
        raise ValueError(
            f"Frozen protocol hash mismatch: expected {expected}, computed {actual}"
        )
    if payload.get("status") not in {"selected", "calibration_no_go"}:
        raise ValueError("No-shift validation requires a final calibration protocol")


def compare_action_traces(
    episode_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare the exact command sent at every no-shift control step."""

    expected: dict[tuple[int, str], int] = {}
    state_by_seed: dict[int, int] = {}
    for row in episode_rows:
        condition = str(row["condition"])
        if condition not in {ALIGNED_NO_SHIFT, GATED_NO_SHIFT}:
            continue
        seed = int(row["seed"])
        key = (seed, condition)
        if key in expected:
            raise ValueError(f"Duplicate no-shift episode: {key}")
        expected[key] = int(row["environment_steps"])
        state = int(row["initial_state_index"])
        previous = state_by_seed.setdefault(seed, state)
        if previous != state:
            raise ValueError(f"Seed {seed} maps to two initial states")

    executed: dict[tuple[int, str], dict[int, np.ndarray]] = {}
    source_kinds: dict[tuple[int, str], dict[int, str]] = {}
    for row in action_rows:
        if bool(row.get("discarded", False)):
            continue
        condition = str(row.get("condition"))
        if condition not in {ALIGNED_NO_SHIFT, GATED_NO_SHIFT}:
            continue
        seed = int(row["seed"])
        step = int(row["queue_execution_step"])
        key = (seed, condition)
        command = np.asarray(row["action"], dtype=np.float32)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError(f"Invalid executed no-shift action at {key}, step {step}")
        bucket = executed.setdefault(key, {})
        if step in bucket:
            raise ValueError(f"Duplicate executed action at {key}, step {step}")
        bucket[step] = command
        source_kinds.setdefault(key, {})[step] = str(row["source_kind"])

    pair_details: list[dict[str, Any]] = []
    all_exact = True
    all_sources_exact = True
    global_max_abs_difference = 0.0
    for seed in sorted(state_by_seed):
        aligned_key = (seed, ALIGNED_NO_SHIFT)
        gate_key = (seed, GATED_NO_SHIFT)
        if aligned_key not in expected or gate_key not in expected:
            raise ValueError(f"Incomplete no-shift episode pair for seed {seed}")
        aligned = executed.get(aligned_key, {})
        gate = executed.get(gate_key, {})
        aligned_expected = expected[aligned_key]
        gate_expected = expected[gate_key]
        if set(aligned) != set(range(aligned_expected)):
            raise ValueError(f"Aligned trace for seed {seed} is incomplete")
        if set(gate) != set(range(gate_expected)):
            raise ValueError(f"Gated trace for seed {seed} is incomplete")

        common_steps = min(aligned_expected, gate_expected)
        first_action_divergence: int | None = None
        first_source_divergence: int | None = None
        pair_max = 0.0
        for step in range(common_steps):
            difference = float(np.max(np.abs(aligned[step] - gate[step])))
            pair_max = max(pair_max, difference)
            if difference != 0.0 and first_action_divergence is None:
                first_action_divergence = step
            if (
                source_kinds[aligned_key][step] != source_kinds[gate_key][step]
                and first_source_divergence is None
            ):
                first_source_divergence = step
        lengths_equal = aligned_expected == gate_expected
        action_exact = lengths_equal and first_action_divergence is None
        source_exact = lengths_equal and first_source_divergence is None
        all_exact = all_exact and action_exact
        all_sources_exact = all_sources_exact and source_exact
        global_max_abs_difference = max(global_max_abs_difference, pair_max)
        pair_details.append(
            {
                "seed": seed,
                "initial_state_index": state_by_seed[seed],
                "aligned_step_count": aligned_expected,
                "gate_step_count": gate_expected,
                "step_counts_equal": lengths_equal,
                "actions_bit_exact": action_exact,
                "action_source_kinds_exact": source_exact,
                "first_action_divergence_step": first_action_divergence,
                "first_source_divergence_step": first_source_divergence,
                "maximum_absolute_action_difference": pair_max,
            }
        )
    return {
        "pair_count": len(pair_details),
        "all_action_traces_bit_exact": all_exact,
        "all_action_source_traces_exact": all_sources_exact,
        "maximum_absolute_action_difference": global_max_abs_difference,
        "pairs": pair_details,
    }


def build_no_shift_decision(
    *,
    protocol_decision_path: Path,
    episode_paths: list[Path],
    action_paths: list[Path],
) -> dict[str, Any]:
    protocol = _read_json(protocol_decision_path)
    _validate_protocol_decision(protocol)
    episode_rows = read_episode_jsonl(episode_paths)
    validate_episode_rows(episode_rows)
    pairing = pair_episode_rows(
        episode_rows,
        reference_condition=ALIGNED_NO_SHIFT,
        estimate_condition=GATED_NO_SHIFT,
        require_valid_perturbation=False,
    )
    analysis = analyze_no_shift_comparison(pairing)
    action_rows = _read_jsonl(action_paths)
    trace = compare_action_traces(episode_rows, action_rows)
    protocol_sha256 = str(protocol["protocol_decision_sha256"])
    source_binding_passed = {
        str(row.get("protocol_decision_sha256")) for row in episode_rows
    } == {protocol_sha256} and {int(row["task_id"]) for row in episode_rows} == {
        int(protocol["no_shift_task_id"])
    }
    checks = {
        "analysis_safety_requirements_pass": bool(analysis["safety_requirements_pass"]),
        "source_binding_passed": source_binding_passed,
        "ten_action_trace_pairs_present": trace["pair_count"] == 10,
        "action_traces_bit_exact": trace["all_action_traces_bit_exact"],
        "action_source_traces_exact": trace["all_action_source_traces_exact"],
    }
    core = {
        "schema_version": 1,
        "milestone": "M5-G0",
        "status": "pass" if all(checks.values()) else "fail",
        "protocol_decision_sha256": protocol_sha256,
        "task_id": int(protocol["no_shift_task_id"]),
        "checks": checks,
        "analysis": analysis,
        "action_trace_comparison": trace,
        "sources": {
            "protocol_decision": {
                "path": str(protocol_decision_path),
                "sha256": file_sha256(protocol_decision_path),
            },
            "episodes": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in episode_paths
            ],
            "actions": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in action_paths
            ],
        },
    }
    return {**core, "no_shift_decision_sha256": canonical_sha256(core)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-decision", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--actions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_no_shift_decision(
        protocol_decision_path=args.protocol_decision,
        episode_paths=args.episodes,
        action_paths=args.actions,
    )
    if args.output.exists():
        existing = _read_json(args.output)
        if existing != payload:
            raise ValueError(
                f"Existing no-shift decision differs from evidence: {args.output}"
            )
        print(f"{args.output} (validated existing)")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
