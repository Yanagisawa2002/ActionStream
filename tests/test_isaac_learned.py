from __future__ import annotations

import numpy as np
import pytest

from actionstream.isaac_learned import (
    ISAAC_REFERENCE_EEF_XYZ,
    ISAAC_REFERENCE_JOINT_POS,
    ISAAC_TO_LIBERO_JOINT_POSITION_OFFSET,
    LIBERO_REFERENCE_EEF_MAT,
    LIBERO_REFERENCE_EEF_QUAT_XYZW,
    LIBERO_REFERENCE_JOINT_POS,
    LIBERO_REFERENCE_EEF_XYZ,
    adapter_contract_payload,
    adapter_contract_sha256,
    libero_state_from_isaac,
    map_xvla_chunk_to_isaac,
    validate_worker_request,
)
from actionstream.learned_isaac_smoke import POLICY_OBSERVATION_SETTLE_UPDATES


def test_reference_state_maps_back_to_libero_home() -> None:
    state = libero_state_from_isaac(
        end_effector_xyz=ISAAC_REFERENCE_EEF_XYZ,
        gripper_aperture_m=0.08,
        joint_positions=ISAAC_REFERENCE_JOINT_POS,
        joint_velocities=[0.0] * 7,
    )
    assert state["eef"]["pos"] == pytest.approx(LIBERO_REFERENCE_EEF_XYZ)
    assert np.asarray(state["eef"]["mat"]) == pytest.approx(
        np.asarray(LIBERO_REFERENCE_EEF_MAT)
    )
    assert state["eef"]["quat"] == pytest.approx(LIBERO_REFERENCE_EEF_QUAT_XYZW)
    assert state["joints"]["pos"] == pytest.approx(LIBERO_REFERENCE_JOINT_POS)
    assert state["gripper"]["qpos"] == pytest.approx([0.04, -0.04])
    assert len(adapter_contract_sha256()) == 64


def test_policy_observation_renderer_has_bounded_temporal_settling() -> None:
    assert POLICY_OBSERVATION_SETTLE_UPDATES == 4


def test_dynamic_eef_orientation_is_normalized_and_mapped_to_matrix() -> None:
    state = libero_state_from_isaac(
        end_effector_xyz=ISAAC_REFERENCE_EEF_XYZ,
        end_effector_wxyz=[2.0, 0.0, 0.0, 0.0],
        gripper_aperture_m=0.08,
        joint_positions=ISAAC_REFERENCE_JOINT_POS,
        joint_velocities=[0.0] * 7,
    )
    assert state["eef"]["quat"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert np.asarray(state["eef"]["mat"]) == pytest.approx(np.eye(3))


def test_worker_request_accepts_exactly_one_image_source() -> None:
    state = libero_state_from_isaac(
        end_effector_xyz=ISAAC_REFERENCE_EEF_XYZ,
        gripper_aperture_m=0.08,
        joint_positions=ISAAC_REFERENCE_JOINT_POS,
        joint_velocities=[0.0] * 7,
    )
    validate_worker_request(
        {
            "request_id": 0,
            "instruction": "pick the object",
            "robot_state": state,
            "official_render_bridge": {
                "object_states": {
                    "alphabet_soup_1": {
                        "position_xyz": [0.0, 0.0, 0.0],
                        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }
                }
            },
        }
    )
    with pytest.raises(ValueError, match="cannot combine"):
        validate_worker_request(
            {
                "request_id": 0,
                "instruction": "pick the object",
                "robot_state": state,
                "image": "a.png",
                "image2": "b.png",
                "official_render_bridge": {
                    "object_states": {"alphabet_soup_1": {}}
                },
            }
        )


def test_joint_state_offset_is_explicit_and_additive() -> None:
    assert np.asarray(ISAAC_REFERENCE_JOINT_POS) + np.asarray(
        ISAAC_TO_LIBERO_JOINT_POSITION_OFFSET
    ) == pytest.approx(LIBERO_REFERENCE_JOINT_POS)


def test_action_mapping_inverts_gripper_and_bounds_motion() -> None:
    actions = np.asarray(
        [
            [-0.10, 0.00, 0.25, 0.0, 0.0, 0.0, -1.0],
            [10.00, 10.00, 10.00, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    mapped = map_xvla_chunk_to_isaac(
        actions,
        initial_target_xyz=ISAAC_REFERENCE_EEF_XYZ,
        workspace_xyz=((0.25, 0.70), (-0.35, 0.35), (0.08, 0.60)),
        maximum_translation_per_step_m=0.01,
    )
    assert mapped.commands.shape == (2, 7)
    assert mapped.commands[:, 6].tolist() == [1.0, -1.0]
    assert np.linalg.norm(mapped.commands[0, :3] - ISAAC_REFERENCE_EEF_XYZ) <= 0.0100001
    assert np.linalg.norm(mapped.commands[1, :3] - mapped.commands[0, :3]) <= 0.0100001
    assert mapped.commands[0, 3:6].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert mapped.clipped_workspace_rows == 1
    assert mapped.limited_translation_rows == 2


def test_base_anchored_coordinate_translation_is_explicitly_supported() -> None:
    translation = (0.6, 0.0, 0.0)
    aligned_home = np.asarray(LIBERO_REFERENCE_EEF_XYZ) + np.asarray(translation)
    state = libero_state_from_isaac(
        end_effector_xyz=aligned_home,
        gripper_aperture_m=0.08,
        joint_positions=[0.0] * 7,
        joint_velocities=[0.0] * 7,
        coordinate_translation_xyz=translation,
    )
    assert state["eef"]["pos"] == pytest.approx(LIBERO_REFERENCE_EEF_XYZ)

    mapped = map_xvla_chunk_to_isaac(
        [[*LIBERO_REFERENCE_EEF_XYZ, 0.0, 0.0, 0.0, -1.0]],
        initial_target_xyz=aligned_home,
        workspace_xyz=((0.25, 0.70), (-0.35, 0.35), (0.02, 0.60)),
        maximum_translation_per_step_m=0.01,
        coordinate_translation_xyz=translation,
    )
    assert mapped.commands[0, :3] == pytest.approx(aligned_home)
    contract = adapter_contract_payload(
        coordinate_translation_xyz=translation,
        camera_2_source="audited_libero_eye_in_hand_camera_on_panda_hand",
    )
    assert contract["translation_xyz"] == pytest.approx(translation)
    assert contract["camera_2_source"].startswith("audited_libero")


def test_action_mapping_preserves_absolute_policy_orientation() -> None:
    orientation = (-2.1859326363, -2.2036967278, 0.0628463998)
    mapped = map_xvla_chunk_to_isaac(
        [[*LIBERO_REFERENCE_EEF_XYZ, *orientation, -1.0]],
        initial_target_xyz=(
            np.asarray(LIBERO_REFERENCE_EEF_XYZ) + np.asarray((0.6, 0.0, 0.0))
        ),
        workspace_xyz=((0.25, 0.70), (-0.35, 0.35), (0.02, 0.60)),
        maximum_translation_per_step_m=0.01,
        coordinate_translation_xyz=(0.6, 0.0, 0.0),
    )
    assert mapped.commands[0, 3:6] == pytest.approx(orientation)
    assert "identity_libero_world_axis_angle" in adapter_contract_payload()[
        "orientation_mapping"
    ]


def test_action_mapping_rejects_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        map_xvla_chunk_to_isaac(
            [[0.0, 0.0, float("nan"), 0.0, 0.0, 0.0, -1.0]],
            initial_target_xyz=(0.4, 0.0, 0.4),
            workspace_xyz=((0.25, 0.70), (-0.35, 0.35), (0.08, 0.60)),
            maximum_translation_per_step_m=0.01,
        )
