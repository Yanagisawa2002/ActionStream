"""Import-safe X-VLA/LIBERO to native-Isaac development adapter.

This module contains only the explicit coordinate and state contract.  It does
not claim task success: the constants below are a development calibration that
must be frozen separately before any learned-policy holdout is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


LIBERO_REFERENCE_EEF_XYZ = (-0.15255246384174243, -0.00627673, 0.2484475446081703)
ISAAC_REFERENCE_EEF_XYZ = (0.38932982087135315, 0.004671656526625156, 0.4563975930213928)
LIBERO_TO_ISAAC_TRANSLATION_XYZ = tuple(
    isaac - libero
    for isaac, libero in zip(ISAAC_REFERENCE_EEF_XYZ, LIBERO_REFERENCE_EEF_XYZ, strict=True)
)

# The LIBERO absolute-pose checkpoint observes a near-downward Panda wrist at
# reset.  Keep the matrix and quaternion fields exactly as emitted by the
# canonical LeRobot environment: they use environment-specific conventions and
# must not be reconstructed from one another.
LIBERO_REFERENCE_EEF_MAT = (
    (9.82098451e-04, 9.98309689e-01, -5.81102388e-02),
    (9.99999481e-01, -9.64611704e-04, 3.28973525e-04),
    (2.72363641e-04, -5.81105317e-02, -9.98310118e-01),
)
# Standard world-frame rotation values reconstructed from the audited matrix.
# These are separate from the environment-specific ``eef.quat`` observation
# below, which deliberately retains LIBERO's processor convention.
LIBERO_REFERENCE_EEF_AXIS_ANGLE_XYZ = (
    -2.192830311181439,
    -2.1906951432053687,
    0.06340618065030879,
)
LIBERO_REFERENCE_EEF_WORLD_QUATERNION_WXYZ = (
    -0.020660158873573082,
    0.7071521809811441,
    0.7064636239672519,
    -0.020447464040385707,
)
LIBERO_REFERENCE_EEF_QUAT_XYZW = (
    0.9995766444610012,
    -0.000492136737,
    -0.02909065429600386,
    -0.000152263753,
)
# One-reset additive calibration between the two Panda IK solutions after the
# EEFs are position-aligned. Both simulators expose the same seven revolute
# joints in the same order; the explicit offset preserves measured Isaac joint
# deltas while putting the policy state back on its LIBERO training manifold.
LIBERO_REFERENCE_JOINT_POS = (
    -0.0104755626,
    -0.150826342,
    -0.00339447717,
    -2.47076307,
    -0.00127685792,
    2.26179517,
    0.773890658,
)
ISAAC_REFERENCE_JOINT_POS = (
    -0.009856946766376495,
    0.04066438972949982,
    -0.0019815361592918634,
    -2.577049732208252,
    -0.00003342428681207821,
    2.6183884143829346,
    0.77161443233349,
)
ISAAC_TO_LIBERO_JOINT_POSITION_OFFSET = tuple(
    libero - isaac
    for libero, isaac in zip(
        LIBERO_REFERENCE_JOINT_POS, ISAAC_REFERENCE_JOINT_POS, strict=True
    )
)
ISAAC_DOWNWARD_AXIS_ANGLE_XYZ = (math.pi, 0.0, 0.0)


def _finite_vector(value: Sequence[float], *, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector with shape ({length},)")
    return array


def _rotation_matrix_from_xyzw(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = _finite_vector(quaternion, length=4, name="quaternion_xyzw")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("quaternion_xyzw must be nonzero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def adapter_contract_payload(
    *,
    coordinate_translation_xyz: Sequence[float] = LIBERO_TO_ISAAC_TRANSLATION_XYZ,
    camera_2_source: str = "duplicate_external_view_development_smoke_only",
) -> dict[str, Any]:
    translation = _finite_vector(
        coordinate_translation_xyz, length=3, name="coordinate_translation_xyz"
    )
    camera_source = str(camera_2_source).strip()
    if not camera_source:
        raise ValueError("camera_2_source must be non-empty")
    return {
        "schema_version": 1,
        "status": "development_only_not_holdout_frozen",
        "policy_action_space": "LIBERO absolute Cartesian xyz, axis-angle xyz, gripper",
        "isaac_action_space": (
            "absolute Cartesian xyz, policy absolute axis-angle xyz, inverted gripper"
        ),
        "translation_xyz": translation.tolist(),
        "libero_reference_eef_xyz": list(LIBERO_REFERENCE_EEF_XYZ),
        "libero_reference_eef_mat": [list(row) for row in LIBERO_REFERENCE_EEF_MAT],
        "libero_reference_eef_axis_angle_xyz": list(
            LIBERO_REFERENCE_EEF_AXIS_ANGLE_XYZ
        ),
        "libero_reference_eef_world_quaternion_wxyz": list(
            LIBERO_REFERENCE_EEF_WORLD_QUATERNION_WXYZ
        ),
        "libero_reference_eef_quat": list(LIBERO_REFERENCE_EEF_QUAT_XYZW),
        "libero_reference_joint_pos": list(LIBERO_REFERENCE_JOINT_POS),
        "isaac_reference_joint_pos": list(ISAAC_REFERENCE_JOINT_POS),
        "isaac_to_libero_joint_position_offset": list(
            ISAAC_TO_LIBERO_JOINT_POSITION_OFFSET
        ),
        "isaac_reference_eef_xyz": list(ISAAC_REFERENCE_EEF_XYZ),
        "isaac_downward_axis_angle_xyz": list(ISAAC_DOWNWARD_AXIS_ANGLE_XYZ),
        "orientation_mapping": (
            "identity_libero_world_axis_angle_to_isaac_world_axis_angle_"
            "after_base_axis_alignment"
        ),
        "gripper_mapping": "isaac_command=-libero_command",
        "camera_2_source": camera_source,
    }


def adapter_contract_sha256(
    *,
    coordinate_translation_xyz: Sequence[float] = LIBERO_TO_ISAAC_TRANSLATION_XYZ,
    camera_2_source: str = "duplicate_external_view_development_smoke_only",
) -> str:
    encoded = json.dumps(
        adapter_contract_payload(
            coordinate_translation_xyz=coordinate_translation_xyz,
            camera_2_source=camera_2_source,
        ),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def libero_state_from_isaac(
    *,
    end_effector_xyz: Sequence[float],
    gripper_aperture_m: float,
    joint_positions: Sequence[float],
    joint_velocities: Sequence[float],
    end_effector_wxyz: Sequence[float] | None = None,
    coordinate_translation_xyz: Sequence[float] = LIBERO_TO_ISAAC_TRANSLATION_XYZ,
) -> dict[str, Any]:
    """Map measured Franka state into the frozen LIBERO processor schema."""

    eef = _finite_vector(end_effector_xyz, length=3, name="end_effector_xyz")
    translation = _finite_vector(
        coordinate_translation_xyz, length=3, name="coordinate_translation_xyz"
    )
    joints = _finite_vector(joint_positions, length=7, name="joint_positions")
    velocities = _finite_vector(joint_velocities, length=7, name="joint_velocities")
    joint_offset = np.asarray(
        ISAAC_TO_LIBERO_JOINT_POSITION_OFFSET, dtype=np.float64
    )
    aperture = float(gripper_aperture_m)
    if not math.isfinite(aperture) or aperture < 0:
        raise ValueError("gripper_aperture_m must be finite and non-negative")
    half = 0.5 * aperture
    if end_effector_wxyz is None:
        eef_mat = np.asarray(LIBERO_REFERENCE_EEF_MAT, dtype=np.float64)
        eef_quat_xyzw = np.asarray(LIBERO_REFERENCE_EEF_QUAT_XYZW, dtype=np.float64)
    else:
        eef_wxyz = _finite_vector(
            end_effector_wxyz, length=4, name="end_effector_wxyz"
        )
        norm = float(np.linalg.norm(eef_wxyz))
        if norm <= 1e-12:
            raise ValueError("end_effector_wxyz must be nonzero")
        w, x, y, z = eef_wxyz / norm
        eef_quat_xyzw = np.asarray([x, y, z, w], dtype=np.float64)
        eef_mat = _rotation_matrix_from_xyzw(eef_quat_xyzw)
    return {
        "eef": {
            "mat": eef_mat.tolist(),
            "pos": (eef - translation).tolist(),
            "quat": eef_quat_xyzw.tolist(),
        },
        "gripper": {
            "qpos": [half, -half],
            "qvel": [0.0, 0.0],
        },
        "joints": {
            "pos": (joints + joint_offset).tolist(),
            "vel": velocities.tolist(),
        },
    }


@dataclass(frozen=True)
class MappedActionChunk:
    commands: np.ndarray
    clipped_workspace_rows: int
    limited_translation_rows: int


def map_xvla_chunk_to_isaac(
    actions: Sequence[Sequence[float]],
    *,
    initial_target_xyz: Sequence[float],
    workspace_xyz: Sequence[Sequence[float]],
    maximum_translation_per_step_m: float,
    coordinate_translation_xyz: Sequence[float] = LIBERO_TO_ISAAC_TRANSLATION_XYZ,
) -> MappedActionChunk:
    """Apply one fixed, auditable environment adapter to an X-VLA chunk."""

    array = np.asarray(actions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 7 or not np.isfinite(array).all():
        raise ValueError("actions must be a finite [T,7] array")
    previous = _finite_vector(initial_target_xyz, length=3, name="initial_target_xyz")
    bounds = np.asarray(workspace_xyz, dtype=np.float64)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("workspace_xyz must contain three finite increasing bounds")
    limit = float(maximum_translation_per_step_m)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("maximum_translation_per_step_m must be positive")

    translation = _finite_vector(
        coordinate_translation_xyz, length=3, name="coordinate_translation_xyz"
    )
    commands: list[np.ndarray] = []
    workspace_clips = 0
    translation_limits = 0
    for row in array:
        translated = row[:3] + translation
        clipped = np.clip(translated, bounds[:, 0], bounds[:, 1])
        workspace_clips += int(not np.allclose(translated, clipped, atol=0.0, rtol=0.0))
        delta = clipped - previous
        distance = float(np.linalg.norm(delta))
        if distance > limit:
            clipped = previous + delta * (limit / distance)
            translation_limits += 1
        gripper = 1.0 if float(row[6]) <= 0.0 else -1.0
        # The canonical LIBERO action axis-angle reconstructs its measured EEF
        # rotation matrix (0.58 degrees at the audited reset).  LIBERO and this
        # Isaac scene share world-axis directions and differ only by the base
        # translation above, so preserving the absolute orientation is the
        # explicit frame mapping.  Replacing it with a fixed Isaac-downward
        # wrist discards roughly 90 degrees of commanded yaw.
        command = np.asarray([*clipped, *row[3:6], gripper], dtype=np.float64)
        commands.append(command)
        previous = clipped
    return MappedActionChunk(
        commands=np.stack(commands, axis=0),
        clipped_workspace_rows=workspace_clips,
        limited_translation_rows=translation_limits,
    )


def validate_worker_request(value: Mapping[str, Any]) -> None:
    required = {"request_id", "instruction", "robot_state"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"worker request is missing {sorted(missing)}")
    if not str(value["instruction"]).strip():
        raise ValueError("worker instruction must be non-empty")
    render_bridge = value.get("official_render_bridge")
    supplied_images = {"image", "image2"} & set(value)
    if render_bridge is None:
        missing_images = {"image", "image2"} - set(value)
        if missing_images:
            raise ValueError(
                f"worker request is missing {sorted(missing_images)}"
            )
    else:
        if not isinstance(render_bridge, Mapping):
            raise ValueError("worker official_render_bridge must be a mapping")
        if supplied_images:
            raise ValueError(
                "worker request cannot combine supplied images with official_render_bridge"
            )
        object_states = render_bridge.get("object_states")
        if not isinstance(object_states, Mapping) or not object_states:
            raise ValueError(
                "worker official_render_bridge requires non-empty object_states"
            )
