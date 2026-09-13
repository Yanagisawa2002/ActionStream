"""False-positive defenses only; these are not native rollout evidence."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from actionstream.native_baseline import (
    episode_outcome,
    native_success,
    require_array,
    validate_config,
    validate_observation,
    validate_prepared_state,
    validate_smoke_receipt,
)


def test_raw_and_processed_actions_reject_bad_shapes_and_nonfinite_values():
    validate_prepared_state({"observation.state": torch.zeros(1, 20)}, 20)
    with pytest.raises(ValueError):
        validate_prepared_state({"observation.state": torch.zeros(1, 8)}, 20)
    require_array(torch.zeros(1, 30, 20), [1, 30, 20])
    require_array(torch.zeros(1, 30, 7), [1, 30, 7])
    for value in (
        torch.zeros(1, 30, 20),
        torch.full((1, 30, 7), float("nan")),
        torch.full((1, 30, 7), float("inf")),
        torch.zeros(1, 7),
    ):
        with pytest.raises(ValueError):
            require_array(value, [1, 30, 7])


def test_missing_camera_or_robot_state_cannot_pass_readiness():
    image = (
        np.arange(360 * 360 * 3, dtype=np.int32)
        .astype(np.uint8)
        .reshape(1, 360, 360, 3)
    )
    observation = {
        "pixels": {"image": image, "image2": image.copy()},
        "robot_state": {
            "eef": {"pos": np.zeros((1, 3)), "quat": np.zeros((1, 4))},
            "gripper": {"qpos": np.zeros((1, 2))},
        },
    }
    validate_observation(observation)
    for section, key in (("pixels", "image2"), ("robot_state", "eef")):
        invalid = copy.deepcopy(observation)
        del invalid[section][key]
        with pytest.raises(KeyError):
            validate_observation(invalid)
    observation["pixels"]["image2"].fill(0)
    with pytest.raises(ValueError):
        validate_observation(observation)


def test_only_authoritative_boolean_success_is_accepted():
    assert (
        native_success({"is_success": np.array([np.bool_(False)], dtype=object)})
        is False
    )
    assert (
        native_success({"is_success": np.array([np.bool_(True)], dtype=object)}) is True
    )
    assert native_success({"is_success": np.array([False]), "reward": 1}) is False
    assert native_success({"is_success": np.array([True])}) is True
    for info in (
        {"reward": 1},
        {"is_success": [float("nan")]},
        {"is_success": [1]},
        {"is_success": np.array(["False"], dtype=object)},
        {"is_success": True},
    ):
        with pytest.raises(ValueError):
            native_success(info)


def test_unrun_interrupted_and_capped_rollouts_are_not_successes():
    def result(steps, success=False, terminated=False):
        return episode_outcome(
            steps=steps,
            success=success,
            terminated=terminated,
            truncated=False,
            cap=300,
        )

    assert result(0)["status"] == "NOT_RUN"
    assert result(0)["success"] is None
    assert result(12)["success"] is None
    assert result(300) == {
        "status": "COMPLETED",
        "success": False,
        "reason": "control_step_limit",
    }
    assert result(12, terminated=True)["success"] is False
    assert result(12, success=True)["success"] is True


def test_development_gate_requires_native_smoke_but_not_task_success():
    receipt = {
        "phase": "smoke",
        "config_sha256": "fixed",
        "model_load": {"status": "PASS"},
        "readiness": {"headless_render": "PASS"},
        "rollout": {
            "status": "COMPLETED",
            "episodes": [
                {"status": "COMPLETED", "control_steps": 300, "success": False}
            ],
        },
    }
    validate_smoke_receipt(receipt, "fixed")
    for section, key, value in (
        ("model_load", "status", "NOT_RUN"),
        ("readiness", "headless_render", "NOT_RUN"),
        ("rollout", "status", "ERROR"),
    ):
        invalid = copy.deepcopy(receipt)
        invalid[section][key] = value
        with pytest.raises(ValueError):
            validate_smoke_receipt(invalid, "fixed")
    with pytest.raises(ValueError):
        validate_smoke_receipt(receipt, "changed")


def test_fixed_development_scope_cannot_expand_or_escape_output_directory():
    config = json.loads(
        (
            Path(__file__).parents[1] / "configs/native_baseline_development.json"
        ).read_text()
    )
    validate_config(config)
    for change in (
        {"max_control_steps": 301},
        {"model_revision": "main"},
        {"raw_action_shape": [1, 30, 7]},
        {"control_mode": "delta"},
        {"development": config["development"] * 2},
    ):
        with pytest.raises(ValueError):
            validate_config({**config, **change})
    config["smoke"][0]["id"] = "../old_result"
    with pytest.raises(ValueError):
        validate_config(config)
