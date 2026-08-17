from __future__ import annotations

import json
from pathlib import Path

from actionstream.m5_freeze import build_protocol_decision
from actionstream.m5_protocol import (
    canonical_sha256,
    file_sha256,
    implementation_source_hash,
)


def test_selected_calibration_freezes_exact_source_and_manifests(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/m5_g0.json"
    audit = root / "outputs/m5_g0/audit/task_entity_audit.json"
    manifests = [
        root / "outputs/m5_g0/protocol/calibration_seed_manifest.json",
        root / "outputs/m5_g0/protocol/no_shift_seed_manifest.json",
        root / "outputs/m5_g0/protocol/sealed_seed_manifest.json",
    ]
    calibration_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    source_sha256, _ = implementation_source_hash(root)
    rows_path = tmp_path / "episodes.jsonl"
    rows = []
    for pair_index in range(5):
        common = {
            "task_id": 0,
            "seed": 51001 + pair_index,
            "initial_state_index": pair_index * 2,
            "displacement_magnitude_mm": 50,
            "perturbation_valid": True,
            "shift_step": 19,
            "achieved_displacement_xyz_m": [-0.05, 0.0, 0.0],
            "implementation_source_sha256": source_sha256,
            "experiment_config_sha256": file_sha256(config),
            "seed_manifest_sha256": calibration_manifest["manifest_sha256"],
        }
        rows.extend(
            [
                {
                    **common,
                    "condition": "aligned_shift",
                    "success": pair_index < 2,
                },
                {
                    **common,
                    "condition": "aligned_oracle_pose_gate_shift",
                    "success": True,
                },
            ]
        )
    rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    decision_core = {
        "schema_version": 1,
        "milestone": "M5-G0",
        "status": "selected",
        "task_id": 0,
        "selected_displacement_magnitude_mm": 50,
        "tested_magnitudes_mm": [50],
        "evaluations": [
            {
                "task_id": 0,
                "displacement_magnitude_mm": 50,
                "pair_count": 5,
                "aligned_success_count": 2,
                "gate_success_count": 5,
                "gate_only_success_count": 3,
                "aligned_only_success_count": 0,
                "all_perturbations_physically_valid": True,
                "paired_shift_parameters_identical": True,
                "qualifies": True,
                "pairs": [
                    {
                        "seed": 51001 + pair_index,
                        "initial_state_index": pair_index * 2,
                        "aligned_success": pair_index < 2,
                        "gate_success": True,
                        "aligned_perturbation_valid": True,
                        "gate_perturbation_valid": True,
                        "gate_only_success": pair_index >= 2,
                        "aligned_only_success": False,
                        "same_shift_step": True,
                        "same_displacement": True,
                    }
                    for pair_index in range(5)
                ],
            }
        ],
        "selection_rule": (
            "first magnitude with 5/5 valid paired shifts, gate >=4/5, "
            "aligned <=3/5, and >=2 gate-only successes"
        ),
        "source_episode_paths": [str(rows_path)],
        "source_episode_files": [
            {"path": str(rows_path), "sha256": file_sha256(rows_path)}
        ],
    }
    decision = {
        **decision_core,
        "selection_sha256": canonical_sha256(decision_core),
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    frozen = build_protocol_decision(
        repository_root=root,
        config_path=config,
        task_audit_path=audit,
        calibration_decision_paths=[decision_path],
        seed_manifest_paths=manifests,
    )

    assert frozen["status"] == "selected"
    assert frozen["sealed_permitted"]
    assert frozen["selected_task_id"] == 0
    assert frozen["selected_displacement_magnitude_mm"] == 50
    assert frozen["implementation_source_sha256"] == source_sha256
    assert frozen["seed_manifests"]["sealed_evaluation"]["pair_count"] == 30
