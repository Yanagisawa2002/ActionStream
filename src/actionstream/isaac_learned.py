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
# reset.  Keeping this state reference fixed avoids pretending that the two
# simulators expose identical quaternion conventions.
LIBERO_REFERENCE_EEF_QUAT_XYZW = (
    0.9995766444610012,
    -0.000492136737,
    -0.02909065429600386,
    -0.000152263753,
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


def adapter_contract_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "development_only_not_holdout_frozen",
        "policy_action_space": "LIBERO absolute Cartesian xyz, axis-angle xyz, gripper",
        "isaac_action_space": "absolute Cartesian xyz, fixed downward wrist, inverted gripper",
        "translation_xyz": list(LIBERO_TO_ISAAC_TRANSLATION_XYZ),
        "libero_reference_eef_xyz": list(LIBERO_REFERENCE_EEF_XYZ),
        "isaac_reference_eef_xyz": list(ISAAC_REFERENCE_EEF_XYZ),
        "isaac_downward_axis_angle_xyz": list(ISAAC_DOWNWARD_AXIS_ANGLE_XYZ),
        "orientation_mapping": "fixed_safe_downward_wrist_development_adapter",
        "gripper_mapping": "isaac_command=-libero_command",
        "camera_2_source": "duplicate_external_view_development_smoke_only",
    }


def adapter_contract_sha256() -> str:
    encoded = json.dumps(
        adapter_contract_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def libero_state_from_isaac(
    *,
    end_effector_xyz: Sequence[float],
    gripper_aperture_m: float,
    joint_positions: Sequence[float],
    joint_velocities: Sequence[float],
) -> dict[str, Any]:
    """Map measured Franka state into the frozen LIBERO processor schema."""

    eef = _finite_vector(end_effector_xyz, length=3, name="end_effector_xyz")
    translation = np.asarray(LIBERO_TO_ISAAC_TRANSLATION_XYZ, dtype=np.float64)
    joints = _finite_vector(joint_positions, length=7, name="joint_positions")
    velocities = _finite_vector(joint_velocities, length=7, name="joint_velocities")
    aperture = float(gripper_aperture_m)
    if not math.isfinite(aperture) or aperture < 0:
        raise ValueError("gripper_aperture_m must be finite and non-negative")
    half = 0.5 * aperture
    quaternion = np.asarray(LIBERO_REFERENCE_EEF_QUAT_XYZW, dtype=np.float64)
    return {
        "eef": {
            "mat": _rotation_matrix_from_xyzw(quaternion).tolist(),
            "pos": (eef - translation).tolist(),
            "quat": quaternion.tolist(),
        },
        "gripper": {
            "qpos": [half, -half],
            "qvel": [0.0, 0.0],
        },
        "joints": {
            "pos": joints.tolist(),
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

    translation = np.asarray(LIBERO_TO_ISAAC_TRANSLATION_XYZ, dtype=np.float64)
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
        command = np.asarray(
            [*clipped, *ISAAC_DOWNWARD_AXIS_ANGLE_XYZ, gripper], dtype=np.float64
        )
        commands.append(command)
        previous = clipped
    return MappedActionChunk(
        commands=np.stack(commands, axis=0),
        clipped_workspace_rows=workspace_clips,
        limited_translation_rows=translation_limits,
    )


def validate_worker_request(value: Mapping[str, Any]) -> None:
    required = {"request_id", "instruction", "image", "image2", "robot_state"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"worker request is missing {sorted(missing)}")
    if not str(value["instruction"]).strip():
        raise ValueError("worker instruction must be non-empty")
