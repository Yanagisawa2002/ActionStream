"""Freeze the post-calibration M5-G0 protocol before no-shift or sealed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from actionstream.m5_calibration import evaluate_magnitude, select_calibration
from actionstream.m5_protocol import (
    canonical_sha256,
    file_sha256,
    implementation_source_hash,
    validate_disjoint,
    validate_manifest,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _validate_hashed_payload(
    payload: dict[str, Any],
    *,
    hash_field: str,
    role: str,
) -> None:
    expected = payload.get(hash_field)
    core = {key: value for key, value in payload.items() if key != hash_field}
    actual = canonical_sha256(core)
    if expected != actual:
        raise ValueError(
            f"{role} hash mismatch: expected {expected}, computed {actual}"
        )


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


def _candidate_order(config: dict[str, Any]) -> list[int]:
    candidates = sorted(
        config["candidate_tasks"],
        key=lambda candidate: int(candidate["audit_order"]),
    )
    task_ids = [int(candidate["task_id"]) for candidate in candidates]
    if len(task_ids) != 2 or len(set(task_ids)) != 2:
        raise ValueError("M5-G0 requires exactly two pre-audited candidate tasks")
    return task_ids


def build_protocol_decision(
    *,
    repository_root: Path,
    config_path: Path,
    task_audit_path: Path,
    calibration_decision_paths: list[Path],
    seed_manifest_paths: list[Path],
) -> dict[str, Any]:
    config = _read_json(config_path)
    task_audit = _read_json(task_audit_path)
    if config.get("milestone") != "M5-G0" or task_audit.get("milestone") != "M5-G0":
        raise ValueError("Config and task audit must both describe M5-G0")
    candidate_order = _candidate_order(config)
    audited_ids = [
        int(candidate["task_id"])
        for candidate in task_audit.get("candidates_inspected", [])
    ]
    if audited_ids != candidate_order:
        raise ValueError("Task-audit order does not match the frozen two candidates")

    decisions: list[tuple[Path, dict[str, Any]]] = []
    for path in calibration_decision_paths:
        payload = _read_json(path)
        _validate_hashed_payload(
            payload,
            hash_field="selection_sha256",
            role=f"calibration decision {path}",
        )
        if payload.get("status") not in {"selected", "candidate_no_go"}:
            raise ValueError(f"Calibration decision is not final: {path}")
        source_paths = [Path(value) for value in payload["source_episode_paths"]]
        recorded_sources = payload.get("source_episode_files")
        expected_sources = [
            {"path": str(source_path), "sha256": file_sha256(source_path)}
            for source_path in source_paths
        ]
        if recorded_sources != expected_sources:
            raise ValueError(
                f"Calibration decision source hashes do not match evidence: {path}"
            )
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in _read_jsonl(source_paths):
            grouped.setdefault(int(row["displacement_magnitude_mm"]), []).append(row)
        evaluations = [evaluate_magnitude(grouped[key]) for key in grouped]
        recomputed = select_calibration(
            evaluations,
            task_id=int(payload["task_id"]),
        )
        for key, value in recomputed.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"Calibration decision {path} disagrees with its evidence "
                    f"for {key}"
                )
        decisions.append((path, payload))
    if not decisions:
        raise ValueError("At least one final calibration decision is required")
    decision_by_task = {
        int(payload["task_id"]): (path, payload) for path, payload in decisions
    }
    if len(decision_by_task) != len(decisions):
        raise ValueError("Duplicate final calibration decisions for one task")

    selected: tuple[Path, dict[str, Any]] | None = None
    for task_id in candidate_order:
        entry = decision_by_task.get(task_id)
        if entry is None:
            if selected is None:
                raise ValueError(
                    "Calibration decisions must follow candidate audit order without gaps"
                )
            continue
        _, decision = entry
        if decision["status"] == "selected":
            if selected is not None:
                raise ValueError("More than one calibration task was selected")
            selected = entry
            break
        if decision["status"] != "candidate_no_go":
            raise ValueError("Unexpected final calibration status")

    if selected is None:
        if set(decision_by_task) != set(candidate_order):
            raise ValueError("Calibration NO-GO requires both candidate tasks")
        status = "calibration_no_go"
        sealed_permitted = False
        selected_task_id = None
        selected_magnitude = None
        no_shift_task_id = candidate_order[0]
        hard_stop_reason = (
            "Neither of the two pre-audited task candidates produced a valid "
            "behavioral challenge under the frozen magnitude rule"
        )
    else:
        selected_path, selected_payload = selected
        selected_task_id = int(selected_payload["task_id"])
        prior_ids = candidate_order[: candidate_order.index(selected_task_id)]
        if any(
            decision_by_task[task_id][1]["status"] != "candidate_no_go"
            for task_id in prior_ids
        ):
            raise ValueError("A later candidate cannot be selected before prior NO-GO")
        if any(task_id not in decision_by_task for task_id in prior_ids):
            raise ValueError("A later selected candidate requires prior NO-GO evidence")
        expected_decision_ids = {*prior_ids, selected_task_id}
        if set(decision_by_task) != expected_decision_ids:
            raise ValueError(
                "Calibration evidence after the first selected candidate is forbidden"
            )
        selected_magnitude = int(selected_payload["selected_displacement_magnitude_mm"])
        qualifying = [
            evaluation
            for evaluation in selected_payload["evaluations"]
            if int(evaluation["displacement_magnitude_mm"]) == selected_magnitude
            and evaluation["qualifies"] is True
        ]
        if len(qualifying) != 1:
            raise ValueError(
                f"Selected calibration decision {selected_path} lacks one qualifying magnitude"
            )
        status = "selected"
        sealed_permitted = True
        no_shift_task_id = selected_task_id
        hard_stop_reason = None

    manifests: dict[str, dict[str, Any]] = {}
    manifest_sources: list[dict[str, Any]] = []
    manifest_payloads: list[dict[str, Any]] = []
    for path in seed_manifest_paths:
        payload = _read_json(path)
        validate_manifest(payload)
        name = str(payload["manifest_name"])
        if name in manifests:
            raise ValueError(f"Duplicate seed manifest name: {name}")
        manifests[name] = {
            "path": str(path),
            "sha256": str(payload["manifest_sha256"]),
            "pair_count": len(payload["pairs"]),
        }
        manifest_sources.append({"path": str(path), "sha256": file_sha256(path)})
        manifest_payloads.append(payload)
    validate_disjoint(manifest_payloads)
    expected_counts = {
        "calibration": 5,
        "no_shift_regression": 10,
        "sealed_evaluation": 30,
    }
    if set(manifests) != set(expected_counts):
        raise ValueError("All three frozen M5-G0 seed manifests are required")
    if any(
        manifests[name]["pair_count"] != count
        for name, count in expected_counts.items()
    ):
        raise ValueError("One or more seed manifests have the wrong pair count")

    current_source_sha256, source_files = implementation_source_hash(repository_root)
    calibration_rows: list[dict[str, Any]] = []
    calibration_sources: list[dict[str, Any]] = []
    for decision_path, decision in decisions:
        source_paths = [Path(value) for value in decision["source_episode_paths"]]
        calibration_rows.extend(_read_jsonl(source_paths))
        calibration_sources.extend(
            {"path": str(path), "sha256": file_sha256(path)} for path in source_paths
        )
        calibration_sources.append(
            {"path": str(decision_path), "sha256": file_sha256(decision_path)}
        )
    if not calibration_rows:
        raise ValueError("Final calibration decisions have no source episode evidence")
    if {str(row.get("implementation_source_sha256")) for row in calibration_rows} != {
        current_source_sha256
    }:
        raise ValueError(
            "Calibration evidence was not produced by the current frozen implementation"
        )
    if {str(row.get("experiment_config_sha256")) for row in calibration_rows} != {
        file_sha256(config_path)
    }:
        raise ValueError("Calibration evidence config hash does not match")
    if {str(row.get("seed_manifest_sha256")) for row in calibration_rows} != {
        manifests["calibration"]["sha256"]
    }:
        raise ValueError(
            "Calibration evidence does not use the frozen calibration manifest"
        )

    candidate_by_id = {
        int(candidate["task_id"]): candidate for candidate in config["candidate_tasks"]
    }
    selected_candidate = (
        None if selected_task_id is None else dict(candidate_by_id[selected_task_id])
    )
    core = {
        "schema_version": 1,
        "milestone": "M5-G0",
        "status": status,
        "sealed_permitted": sealed_permitted,
        "selected_task_id": selected_task_id,
        "no_shift_task_id": no_shift_task_id,
        "selected_displacement_magnitude_mm": selected_magnitude,
        "selected_candidate": selected_candidate,
        "hard_stop_reason": hard_stop_reason,
        "candidate_order": candidate_order,
        "detector": config["detector"],
        "runtime": config["runtime"],
        "perturbation": {
            "axis_xyz": config["perturbation"]["axis_xyz"],
            "schedule": config["perturbation"]["schedule"],
            "midpoint_formula": config["perturbation"]["midpoint_formula"],
            "workspace_bounds_m": config["perturbation"]["workspace_bounds_m"],
            "displacement_tolerance_m": config["perturbation"][
                "displacement_tolerance_m"
            ],
        },
        "evaluator": {
            "stale_definition": (
                "action.world_epoch_at_observation < current_world_epoch"
            ),
            "world_epoch_runtime_visible": False,
            "detector_input": "ground_truth_entity_pose_only",
        },
        "config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "task_audit": {
            "path": str(task_audit_path),
            "sha256": file_sha256(task_audit_path),
        },
        "seed_manifests": manifests,
        "manifest_sources": manifest_sources,
        "calibration_decisions": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "selection_sha256": payload["selection_sha256"],
                "task_id": int(payload["task_id"]),
                "status": payload["status"],
            }
            for path, payload in decisions
        ],
        "calibration_sources": calibration_sources,
        "implementation_source_sha256": current_source_sha256,
        "implementation_source_files": source_files,
    }
    return {**core, "protocol_decision_sha256": canonical_sha256(core)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-audit", type=Path, required=True)
    parser.add_argument(
        "--calibration-decision",
        dest="calibration_decisions",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--seed-manifest",
        dest="seed_manifests",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    payload = build_protocol_decision(
        repository_root=root,
        config_path=args.config,
        task_audit_path=args.task_audit,
        calibration_decision_paths=args.calibration_decisions,
        seed_manifest_paths=args.seed_manifests,
    )
    if args.output.exists():
        existing = _read_json(args.output)
        if existing != payload:
            raise ValueError(
                f"Existing frozen protocol differs from recomputed evidence: "
                f"{args.output}"
            )
        print(f"{args.output} (validated existing)")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
