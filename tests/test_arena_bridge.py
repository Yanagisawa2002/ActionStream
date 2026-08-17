from __future__ import annotations

import pytest
import torch

from actionstream.arena_bridge import (
    ArenaBridgeError,
    DroidRelativeIKConfig,
    absolute_targets_to_droid_relative_ik,
    droid_bounded_hold_action,
    snapshot_droid_observation,
)


def _observation(num_envs: int = 2) -> dict:
    return {
        "policy": {
            "eef_pos": torch.tensor([[0.40, 0.00, 0.30]]).repeat(num_envs, 1),
            "eef_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
            "gripper_pos": torch.tensor([[0.0], [1.0]])[:num_envs],
            "joint_pos": torch.zeros((num_envs, 7)),
        },
        "camera_obs": {
            "external_camera_rgb": torch.zeros((num_envs, 8, 12, 3), dtype=torch.uint8),
            "external_camera_2_rgb": torch.ones(
                (num_envs, 8, 12, 3), dtype=torch.uint8
            ),
            "wrist_camera_rgb": torch.full((num_envs, 8, 12, 3), 2, dtype=torch.uint8),
        },
    }


def test_snapshot_requires_real_droid_visual_and_state_contract() -> None:
    observation = _observation()
    snapshot = snapshot_droid_observation(observation)

    assert snapshot.num_envs == 2
    assert set(snapshot.cameras) == {
        "external_camera_rgb",
        "external_camera_2_rgb",
        "wrist_camera_rgb",
    }
    observation["policy"]["eef_pos"].fill_(999)
    assert torch.all(snapshot.policy["eef_pos"][:, 0] == 0.40)
    assert all(tensor.device.type == "cpu" for tensor in snapshot.policy.values())


def test_snapshot_rejects_missing_second_camera() -> None:
    observation = _observation()
    del observation["camera_obs"]["external_camera_2_rgb"]

    with pytest.raises(ArenaBridgeError, match="external_camera_2_rgb"):
        snapshot_droid_observation(observation)


def test_snapshot_rejects_one_tensor_reused_as_two_views() -> None:
    observation = _observation()
    observation["camera_obs"]["external_camera_2_rgb"] = observation["camera_obs"][
        "external_camera_rgb"
    ]

    with pytest.raises(ArenaBridgeError, match="cannot alias"):
        snapshot_droid_observation(observation)


def test_absolute_target_is_converted_at_dispatch_and_bounded() -> None:
    snapshot = snapshot_droid_observation(_observation())
    half_turn_z_wxyz = [0.0, 0.0, 0.0, 1.0]
    targets = torch.tensor(
        [
            [1.40, 0.0, 0.30, *half_turn_z_wxyz, 1.0],
            [0.40, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    config = DroidRelativeIKConfig(
        controller_scale=0.5,
        maximum_translation_per_step_m=0.03,
        maximum_rotation_per_step_rad=0.20,
    )

    action = absolute_targets_to_droid_relative_ik(targets, snapshot, config=config)

    assert action.shape == (2, 7)
    assert action[0, 0].item() == pytest.approx(0.06)
    assert torch.linalg.vector_norm(action[0, 3:6]).item() == pytest.approx(0.40)
    assert action[0, 6].item() == 1.0
    assert torch.allclose(action[1, :6], torch.zeros(6))
    assert action[1, 6].item() == 0.0


def test_hold_is_zero_motion_and_preserves_gripper() -> None:
    snapshot = snapshot_droid_observation(_observation())
    action = droid_bounded_hold_action(snapshot)

    assert torch.allclose(action[:, :6], torch.zeros((2, 6)))
    assert action[:, 6].tolist() == [0.0, 1.0]


def test_target_batch_must_match_vector_env_count() -> None:
    snapshot = snapshot_droid_observation(_observation())
    with pytest.raises(ArenaBridgeError, match=r"\[2,8\]"):
        absolute_targets_to_droid_relative_ik(torch.zeros((1, 8)), snapshot)
