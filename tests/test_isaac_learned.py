from __future__ import annotations

import math

import numpy as np
import pytest

from actionstream.isaac_learned import (
    ISAAC_REFERENCE_EEF_XYZ,
    LIBERO_REFERENCE_EEF_XYZ,
    adapter_contract_sha256,
    libero_state_from_isaac,
    map_xvla_chunk_to_isaac,
)


def test_reference_state_maps_back_to_libero_home() -> None:
    state = libero_state_from_isaac(
        end_effector_xyz=ISAAC_REFERENCE_EEF_XYZ,
        gripper_aperture_m=0.08,
        joint_positions=[0.0] * 7,
        joint_velocities=[0.0] * 7,
    )
    assert state["eef"]["pos"] == pytest.approx(LIBERO_REFERENCE_EEF_XYZ)
    assert np.asarray(state["eef"]["mat"]).shape == (3, 3)
    assert state["gripper"]["qpos"] == pytest.approx([0.04, -0.04])
    assert len(adapter_contract_sha256()) == 64


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
    assert mapped.commands[0, 3:6].tolist() == pytest.approx([math.pi, 0.0, 0.0])
    assert mapped.clipped_workspace_rows == 1
    assert mapped.limited_translation_rows == 2


def test_action_mapping_rejects_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        map_xvla_chunk_to_isaac(
            [[0.0, 0.0, float("nan"), 0.0, 0.0, 0.0, -1.0]],
            initial_target_xyz=(0.4, 0.0, 0.4),
            workspace_xyz=((0.25, 0.70), (-0.35, 0.35), (0.08, 0.60)),
            maximum_translation_per_step_m=0.01,
        )
