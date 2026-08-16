from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from actionstream.xvla_input_counterfactual import (
    action_descriptors,
    batch_robot_state,
    paired_action_metrics,
    summarize_records,
    unbatch_robot_state,
    validate_unbatched_robot_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _state() -> dict[str, object]:
    return {
        "eef": {
            "mat": np.eye(3).tolist(),
            "pos": [0.1, 0.2, 0.3],
            "quat": [0.0, 0.0, 0.0, 1.0],
        },
        "gripper": {"qpos": [0.04, -0.04], "qvel": [0.0, 0.0]},
        "joints": {"pos": [0.0] * 7, "vel": [0.0] * 7},
    }


def test_frozen_protocol_has_balanced_four_condition_schedule() -> None:
    config = json.loads(
        (ROOT / "configs" / "xvla_isaac_input_counterfactual_v1.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {item["key"] for item in config["conditions"]}
    assert config["freeze_status"] == "frozen_before_formal_inference"
    assert expected == {
        "official_images__official_state",
        "native_images__official_state",
        "official_images__native_state",
        "native_images__native_state",
    }
    assert len(config["paired_schedule"]) == 3
    assert len({item["inference_seed"] for item in config["paired_schedule"]}) == 3
    assert all(set(item["condition_order"]) == expected for item in config["paired_schedule"])
    assert config["analysis_contract"]["no_post_result_input_tuning"] is True


def test_robot_state_batch_round_trip_is_exact() -> None:
    original = validate_unbatched_robot_state(_state())
    assert unbatch_robot_state(batch_robot_state(original)) == original


def test_robot_state_rejects_wrong_joint_shape() -> None:
    state = _state()
    state["joints"]["pos"] = [0.0] * 6
    with pytest.raises(ValueError, match="shape"):
        validate_unbatched_robot_state(state)


def test_paired_metrics_and_descriptors_are_interpretable() -> None:
    reference = np.zeros((30, 7), dtype=np.float32)
    candidate = reference.copy()
    candidate[:, 0] = 0.03
    candidate[:, 6] = -1.0
    metrics = paired_action_metrics(candidate, reference)
    assert metrics["first_action_xyz_l2"] == pytest.approx(0.03)
    assert metrics["first_action_7d_l2"] == pytest.approx(np.sqrt(1.0009))
    assert metrics["full_chunk_rmse"] == pytest.approx(np.sqrt(1.0009 / 7.0))
    descriptors = action_descriptors(candidate)
    assert descriptors["gripper_negative_fraction"] == 1.0
    assert descriptors["gripper_positive_fraction"] == 0.0


def test_summary_computes_paired_effects_per_seed() -> None:
    records = []
    offsets = {
        "official_images__official_state": 0.0,
        "native_images__official_state": 1.0,
        "official_images__native_state": 2.0,
        "native_images__native_state": 3.0,
    }
    for seed in (17, 18, 19):
        for condition, offset in offsets.items():
            actions = np.full((30, 7), offset, dtype=np.float32)
            records.append(
                {
                    "inference_seed": seed,
                    "condition": condition,
                    "actions": actions.tolist(),
                }
            )
    summary = summarize_records(records)
    aggregate = summary["aggregate"]
    assert aggregate["image_effect_holding_official_state"]["full_chunk_rmse"][
        "mean"
    ] == pytest.approx(1.0)
    assert aggregate["state_effect_holding_official_images"]["full_chunk_rmse"][
        "mean"
    ] == pytest.approx(2.0)
    assert aggregate["joint_shift"]["full_chunk_rmse"]["mean"] == pytest.approx(3.0)


def test_frozen_result_is_complete_and_recomputes_to_reported_effects() -> None:
    result_path = ROOT / "outputs" / "xvla_isaac_input_counterfactual_v1" / "summary.json"
    summary = json.loads(result_path.read_text(encoding="utf-8"))
    config_path = ROOT / "configs" / "xvla_isaac_input_counterfactual_v1.json"
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    assert summary["evidence_class"] == "offline_first_chunk_input_counterfactual"
    assert summary["task_success_evidence"] is False
    assert summary["config_sha256"] == config_sha256
    assert len(summary["records"]) == 12
    assert summary["official_observation"]["image_sha256"] == (
        "5e4c361d237eb2fcf686c4d6a752894658bcfc09a00b8c5b64393e9b4f892ac5"
    )
    assert summary["official_observation"]["image2_sha256"] == (
        "5b3e7f5becfa62667c9cd563a6f7bfcd74d498ae7c2d6b8bc1b83ea6f7824caa"
    )

    recomputed = summarize_records(summary["records"])
    assert recomputed == summary["analysis"]
    aggregate = recomputed["aggregate"]
    image_effect = aggregate["image_effect_holding_official_state"]
    state_effect = aggregate["state_effect_holding_official_images"]
    assert image_effect["full_chunk_rmse"]["mean"] == pytest.approx(0.7608883155)
    assert state_effect["full_chunk_rmse"]["mean"] == pytest.approx(0.0052385180)
    assert image_effect["full_chunk_rmse"]["mean"] > 100 * state_effect[
        "full_chunk_rmse"
    ]["mean"]
    assert image_effect["first_action_xyz_l2"]["mean"] > 8 * state_effect[
        "first_action_xyz_l2"
    ]["mean"]

    for record in summary["records"]:
        if record["image_source"] == "official":
            assert record["descriptors"]["gripper_negative_fraction"] == 1.0
            assert record["descriptors"]["gripper_positive_fraction"] == 0.0
        else:
            assert record["descriptors"]["gripper_negative_fraction"] == 0.0
            assert record["descriptors"]["gripper_positive_fraction"] == 1.0
