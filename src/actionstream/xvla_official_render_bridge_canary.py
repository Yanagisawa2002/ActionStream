"""Run the frozen reset-scoped X-VLA official-render / Isaac-state canary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from actionstream.official_render_bridge import apply_isaac_state_to_official_renderer
from actionstream.xvla_input_counterfactual import (
    action_descriptors,
    batch_robot_state,
    paired_action_metrics,
    save_rgb_batch,
    sha256_file,
    unbatch_robot_state,
    validate_unbatched_robot_state,
)

REFERENCE_CONDITION = "official_render__official_state"
CANDIDATE_CONDITION = "official_render_bridge__isaac_state"
EXPECTED_SEEDS = (2026081701, 2026081702, 2026081703)
EXPECTED_ACCEPTANCE = {
    "aggregate_mean_full_chunk_rmse_max": 0.25,
    "aggregate_mean_first_action_xyz_l2_max_m": 0.05,
    "aggregate_mean_first_action_7d_l2_max": 0.75,
    "candidate_gripper_negative_fraction_min_each_seed": 0.9,
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _require_sha256(path: Path, expected: str, *, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    if config.get("freeze_status") != "frozen_before_formal_inference":
        raise ValueError("Bridge canary is not marked frozen before inference")
    seeds = tuple(int(item["inference_seed"]) for item in config["paired_schedule"])
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Bridge canary must reuse exact V2 seeds {EXPECTED_SEEDS}")
    expected_conditions = {REFERENCE_CONDITION, CANDIDATE_CONDITION}
    for item in config["paired_schedule"]:
        if set(item["condition_order"]) != expected_conditions:
            raise ValueError("Each paired row must contain exactly reference and bridge")
    acceptance = config.get("acceptance", {})
    for key, expected in EXPECTED_ACCEPTANCE.items():
        if float(acceptance.get(key, float("nan"))) != expected:
            raise ValueError(f"Frozen V2 acceptance changed for {key}")
    if acceptance.get("all_required") is not True:
        raise ValueError("Bridge canary requires every frozen acceptance check")
    return config


def _reset_inference_state(backend: Any, seed: int) -> None:
    backend._set_seed(int(seed))
    backend.policy.reset()
    for pipeline in (
        backend.env_preprocessor,
        backend.preprocessor,
        backend.postprocessor,
        backend.env_postprocessor,
    ):
        pipeline.reset()


def _batched_observation(
    images: Mapping[str, Any], robot_state: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "pixels": {
            "image": np.asarray(images["image"])[None, ...].copy(),
            "image2": np.asarray(images["image2"])[None, ...].copy(),
        },
        "robot_state": batch_robot_state(robot_state),
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["inference_seed"]), {})[
            str(record["condition"])
        ] = record
    per_seed: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        conditions = grouped.get(seed, {})
        if set(conditions) != {REFERENCE_CONDITION, CANDIDATE_CONDITION}:
            raise ValueError(f"Seed {seed} is missing a frozen paired condition")
        reference = np.asarray(conditions[REFERENCE_CONDITION]["actions"])
        candidate = np.asarray(conditions[CANDIDATE_CONDITION]["actions"])
        per_seed.append(
            {
                "inference_seed": seed,
                "paired_metrics": paired_action_metrics(candidate, reference),
                "candidate_gripper_negative_fraction": action_descriptors(candidate)[
                    "gripper_negative_fraction"
                ],
                "reference_gripper_negative_fraction": action_descriptors(reference)[
                    "gripper_negative_fraction"
                ],
            }
        )
    aggregate: dict[str, dict[str, Any]] = {}
    for metric in per_seed[0]["paired_metrics"]:
        values = np.asarray(
            [item["paired_metrics"][metric] for item in per_seed], dtype=np.float64
        )
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
    return {"per_seed": per_seed, "aggregate": aggregate}


def evaluate_acceptance(
    analysis: Mapping[str, Any], acceptance: Mapping[str, Any]
) -> dict[str, Any]:
    aggregate = analysis["aggregate"]
    gripper_values = [
        float(item["candidate_gripper_negative_fraction"])
        for item in analysis["per_seed"]
    ]
    checks = {
        "full_chunk_rmse": {
            "value": float(aggregate["full_chunk_rmse"]["mean"]),
            "operator": "<=",
            "threshold": float(acceptance["aggregate_mean_full_chunk_rmse_max"]),
        },
        "first_action_xyz_l2_m": {
            "value": float(aggregate["first_action_xyz_l2"]["mean"]),
            "operator": "<=",
            "threshold": float(
                acceptance["aggregate_mean_first_action_xyz_l2_max_m"]
            ),
        },
        "first_action_7d_l2": {
            "value": float(aggregate["first_action_7d_l2"]["mean"]),
            "operator": "<=",
            "threshold": float(acceptance["aggregate_mean_first_action_7d_l2_max"]),
        },
        "candidate_gripper_negative_fraction_each_seed": {
            "value": min(gripper_values),
            "operator": ">=",
            "threshold": float(
                acceptance["candidate_gripper_negative_fraction_min_each_seed"]
            ),
            "values": gripper_values,
        },
    }
    for check in checks.values():
        check["pass"] = (
            check["value"] <= check["threshold"]
            if check["operator"] == "<="
            else check["value"] >= check["threshold"]
        )
    return {
        "all_acceptance_checks_pass": all(check["pass"] for check in checks.values()),
        "checks": checks,
    }


def validate_structural_bridge(
    provenance: Mapping[str, Any], structural: Mapping[str, Any]
) -> dict[str, Any]:
    ik = provenance.get("ik")
    if not isinstance(ik, Mapping):
        raise ValueError("Frozen bridge requires EEF IK provenance")
    checks = {
        "pose_mode": provenance.get("pose_mode") == structural["pose_mode"],
        "ik_converged": ik.get("converged") is True,
        "eef_position_error": float(ik["position_error_l2_m"])
        <= float(structural["eef_position_error_l2_max_m"]),
        "eef_orientation_error": float(ik["orientation_error_l2_rad"])
        <= float(structural["eef_orientation_error_l2_max_rad"]),
        "no_physics_step": int(provenance["physics_steps_after_write"]) == 0,
        "gripper_writeback": max(float(v) for v in provenance["write_errors"].values())
        <= float(structural["gripper_write_error_max_abs"]),
    }
    return {"pass": all(checks.values()), "checks": checks}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _report(summary: Mapping[str, Any]) -> str:
    decision = summary["acceptance"]
    aggregate = summary["analysis"]["aggregate"]
    status = "passed" if decision["all_acceptance_checks_pass"] else "failed"
    lines = [
        "# X-VLA official-render / Isaac-state bridge canary",
        "",
        f"Status: **{status} frozen first-chunk canary**.",
        "",
        "This is a reset-scoped offline paired diagnostic, not episode or task-success evidence.",
        "",
        "| Metric | Mean +/- std | Frozen threshold | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    rows = (
        ("Full chunk RMSE", "full_chunk_rmse", "full_chunk_rmse"),
        ("First action XYZ L2 (m)", "first_action_xyz_l2", "first_action_xyz_l2_m"),
        ("First action 7D L2", "first_action_7d_l2", "first_action_7d_l2"),
    )
    for label, metric, check_key in rows:
        item = aggregate[metric]
        check = decision["checks"][check_key]
        lines.append(
            f"| {label} | {item['mean']:.6f} +/- {item['std']:.6f} | "
            f"{check['operator']} {check['threshold']:.6f} | "
            f"{'PASS' if check['pass'] else 'FAIL'} |"
        )
    gripper = decision["checks"]["candidate_gripper_negative_fraction_each_seed"]
    lines.append(
        "| Candidate close fraction, worst seed | "
        f"{gripper['value']:.6f} | >= {gripper['threshold']:.6f} | "
        f"{'PASS' if gripper['pass'] else 'FAIL'} |"
    )
    lines.extend(
        [
            "",
            "The official renderer uses matching reset objects and an EEF-IK-retargeted "
            "canonical Panda; the policy state remains the measured Isaac mapping. Dynamic "
            "object synchronization and closed-loop task capability are not tested here.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--native-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    from actionstream.current_baselines import CurrentLeRobotBackend, load_protocol

    args = _parser().parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"refusing to reuse formal bridge output: {output}")
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    _require_sha256(
        root / config["parent_canary"]["path"],
        config["parent_canary"]["sha256"],
        label="parent V2 canary",
    )
    _require_sha256(
        root / config["bridge_candidate"]["module_path"],
        config["bridge_candidate"]["module_sha256"],
        label="bridge module",
    )
    _require_sha256(
        Path(__file__).resolve(),
        config["bridge_candidate"]["runner_sha256"],
        label="formal bridge runner",
    )
    protocol_path = args.protocol.resolve()
    _require_sha256(
        protocol_path, config["policy"]["protocol_sha256"], label="protocol"
    )
    native_summary_path = args.native_summary.resolve()
    _require_sha256(
        native_summary_path,
        config["native_observation"]["summary_sha256"],
        label="native summary",
    )
    native_summary = json.loads(native_summary_path.read_text(encoding="utf-8"))
    requests = native_summary.get("request_records", [])
    if not requests:
        raise ValueError("Native summary has no policy request state")
    native_state = validate_unbatched_robot_state(requests[0]["robot_state"])
    official = config["official_observation"]
    if native_summary.get("instruction") != official["instruction"]:
        raise ValueError("Native and official instructions differ")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    protocol = load_protocol(protocol_path)
    spec = protocol.models[config["policy"]["key"]]
    backend = CurrentLeRobotBackend(
        spec=spec,
        task_ids=[int(official["task_id"])],
        suite=str(official["suite"]),
        episode_length=800,
        seed=int(official["environment_seed"]),
        device=str(config["policy"]["device"]),
    )
    try:
        reference_observation, _info, instruction = backend.reset_episode(
            task_id=int(official["task_id"]),
            seed=int(official["environment_seed"]),
            initial_state_index=int(official["initial_state_index"]),
        )
        if instruction != official["instruction"]:
            raise ValueError("Frozen official instruction mismatch")
        reference_images = {
            key: np.asarray(reference_observation["pixels"][key])[0].copy()
            for key in ("image", "image2")
        }
        reference_state = unbatch_robot_state(reference_observation["robot_state"])
        bridge = apply_isaac_state_to_official_renderer(
            backend._sub_env(int(official["task_id"])),
            native_state,
            pose_mode=config["bridge_candidate"]["structural_gate"]["pose_mode"],
        )
        structural = validate_structural_bridge(
            bridge.provenance, config["bridge_candidate"]["structural_gate"]
        )
        if not structural["pass"]:
            raise RuntimeError(f"Frozen bridge structural gate failed: {structural}")
        bridge_images = bridge.observation["pixels"]
        image_hashes = {
            "reference_image_sha256": save_rgb_batch(
                reference_images["image"][None, ...], output / "reference_image.png"
            ),
            "reference_image2_sha256": save_rgb_batch(
                reference_images["image2"][None, ...], output / "reference_image2.png"
            ),
            "bridge_image_sha256": save_rgb_batch(
                bridge_images["image"][None, ...], output / "bridge_image.png"
            ),
            "bridge_image2_sha256": save_rgb_batch(
                bridge_images["image2"][None, ...], output / "bridge_image2.png"
            ),
        }
        observations = {
            REFERENCE_CONDITION: _batched_observation(
                reference_images, reference_state
            ),
            CANDIDATE_CONDITION: _batched_observation(bridge_images, native_state),
        }
        records: list[dict[str, Any]] = []
        for schedule in config["paired_schedule"]:
            inference_seed = int(schedule["inference_seed"])
            for condition in schedule["condition_order"]:
                _reset_inference_state(backend, inference_seed)
                inference = backend.infer_action_chunk(
                    copy.deepcopy(observations[condition]), instruction
                )
                actions = np.asarray(inference.actions, dtype=np.float32)
                records.append(
                    {
                        "inference_seed": inference_seed,
                        "condition": condition,
                        "actions": actions.tolist(),
                        "actions_sha256": _sha256_bytes(actions.tobytes(order="C")),
                        "raw_actions_sha256": _sha256_bytes(
                            np.asarray(inference.raw_actions).tobytes(order="C")
                        ),
                        "model_latency_seconds": inference.model_latency_seconds,
                        "raw_shape": list(inference.raw_shape),
                        "raw_dtype": inference.raw_dtype,
                        "descriptors": action_descriptors(actions),
                    }
                )
        analysis = summarize_records(records)
        acceptance = evaluate_acceptance(analysis, config["acceptance"])
        summary = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "status": (
                "passed_frozen_canary"
                if acceptance["all_acceptance_checks_pass"]
                else "failed_frozen_canary"
            ),
            "evidence_class": "reset_scoped_offline_first_chunk_bridge_canary",
            "task_success_evidence": False,
            "config_sha256": sha256_file(config_path),
            "policy": config["policy"],
            "instruction": instruction,
            "native_summary_sha256": sha256_file(native_summary_path),
            "reference_robot_state": reference_state,
            "reference_robot_state_sha256": _canonical_json_sha256(reference_state),
            "isaac_robot_state": native_state,
            "isaac_robot_state_sha256": _canonical_json_sha256(native_state),
            "bridge": bridge.provenance,
            "structural_gate": structural,
            "image_hashes": image_hashes,
            "records": records,
            "analysis": analysis,
            "acceptance": acceptance,
            "peak_cuda_memory_mib": backend.peak_cuda_memory_mib,
            "stop_or_next_gate": (
                config["acceptance"]["on_pass"]
                if acceptance["all_acceptance_checks_pass"]
                else config["acceptance"]["on_fail"]
            ),
            "limitations": [
                "Only one matching reset and first action chunk are evaluated.",
                "Official reset object states are not synchronized after Isaac motion.",
                "Passing is not learned-policy task success or async-runtime evidence.",
            ],
        }
        _write_json(output / "summary.json", summary)
        (output / "report.md").write_text(_report(summary), encoding="utf-8")
        print(
            json.dumps(
                {
                    "event": "completed",
                    "status": summary["status"],
                    "output": str(output),
                    "summary_sha256": sha256_file(output / "summary.json"),
                    "peak_cuda_memory_mib": backend.peak_cuda_memory_mib,
                },
                sort_keys=True,
            )
        )
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
