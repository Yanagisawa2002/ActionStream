"""Explicit observation and relative-IK contracts for Isaac Lab-Arena.

Arena's DROID differential-IK embodiment consumes relative 7D commands while
the X-VLA/LIBERO checkpoint emits an environment-specific action contract.  The
helpers here make that boundary fail closed: visual/state terms must be present,
simulator tensors are detached before a background worker sees them, and an
absolute end-effector target is converted to a bounded relative-IK command only
on the control thread using the current robot state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


class ArenaBridgeError(ValueError):
    """Raised when the simulator-policy contract is incomplete or inconsistent."""


@dataclass(frozen=True)
class DroidObservationSnapshot:
    """Worker-safe tensors from one vectorized Arena observation."""

    policy: Mapping[str, torch.Tensor]
    cameras: Mapping[str, torch.Tensor]
    num_envs: int


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArenaBridgeError(f"{name} must be a mapping")
    return value


def _tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ArenaBridgeError(f"{name} must be a torch.Tensor")
    if value.ndim < 2 or value.shape[0] < 1:
        raise ArenaBridgeError(
            f"{name} must have a non-empty vector-env batch dimension"
        )
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ArenaBridgeError(f"{name} contains non-finite values")
    return value


def snapshot_droid_observation(
    observation: Mapping[str, Any],
) -> DroidObservationSnapshot:
    """Validate and detach the official Arena DROID camera/state groups.

    Isaac Lab manager environments expose non-concatenated observation groups as
    ``policy`` and ``camera_obs``.  DROID is used instead of the default Franka
    embodiment because its official Arena configuration supplies two external
    cameras plus a wrist camera.  Reusing one image under two feature names is
    intentionally rejected.
    """

    root = _mapping(observation, name="observation")
    policy = _mapping(root.get("policy"), name="observation.policy")
    cameras = _mapping(root.get("camera_obs"), name="observation.camera_obs")

    required_state = {"eef_pos", "eef_quat", "gripper_pos", "joint_pos"}
    missing_state = required_state - set(policy)
    if missing_state:
        raise ArenaBridgeError(f"Arena DROID state is missing {sorted(missing_state)}")
    required_cameras = {
        "external_camera_rgb",
        "external_camera_2_rgb",
        "wrist_camera_rgb",
    }
    missing_cameras = required_cameras - set(cameras)
    if missing_cameras:
        raise ArenaBridgeError(
            f"Arena DROID cameras are missing {sorted(missing_cameras)}"
        )

    state_tensors = {
        key: _tensor(policy[key], name=f"policy.{key}")
        .detach()
        .to(device="cpu")
        .clone()
        for key in required_state
    }
    raw_camera_tensors: dict[str, torch.Tensor] = {}
    camera_tensors: dict[str, torch.Tensor] = {}
    for key in required_cameras:
        image = _tensor(cameras[key], name=f"camera_obs.{key}")
        if image.ndim != 4 or image.shape[-1] not in {3, 4}:
            raise ArenaBridgeError(
                f"camera_obs.{key} must be [N,H,W,3|4], got {tuple(image.shape)}"
            )
        raw_camera_tensors[key] = image
        camera_tensors[key] = image[..., :3].detach().to(device="cpu").clone()

    batch_sizes = {
        int(value.shape[0])
        for value in (*state_tensors.values(), *camera_tensors.values())
    }
    if len(batch_sizes) != 1:
        raise ArenaBridgeError(
            f"Arena observation batch sizes disagree: {sorted(batch_sizes)}"
        )
    num_envs = batch_sizes.pop()

    first = raw_camera_tensors["external_camera_rgb"]
    second = raw_camera_tensors["external_camera_2_rgb"]
    wrist = raw_camera_tensors["wrist_camera_rgb"]
    if first.data_ptr() == second.data_ptr() or first.data_ptr() == wrist.data_ptr():
        raise ArenaBridgeError(
            "camera contract cannot alias one tensor under multiple views"
        )

    return DroidObservationSnapshot(
        policy=state_tensors, cameras=camera_tensors, num_envs=num_envs
    )


def quaternion_wxyz_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized WXYZ quaternions to shortest-path axis angles."""

    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ArenaBridgeError(
            f"quaternion must be [N,4], got {tuple(quaternion.shape)}"
        )
    norm = torch.linalg.vector_norm(quaternion, dim=1, keepdim=True)
    if torch.any(norm <= 1e-8):
        raise ArenaBridgeError("zero-norm quaternion")
    q = quaternion / norm
    q = torch.where(q[:, :1] < 0, -q, q)
    vector = q[:, 1:]
    sin_half = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, q[:, :1].clamp(min=0.0))
    scale = torch.where(
        sin_half > 1e-8, angle / sin_half, torch.full_like(sin_half, 2.0)
    )
    return vector * scale


def _quaternion_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=1)
    rw, rx, ry, rz = right.unbind(dim=1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=1,
    )


@dataclass(frozen=True)
class DroidRelativeIKConfig:
    """Arena 0.2.1 DROID differential-IK command limits."""

    controller_scale: float = 0.5
    maximum_translation_per_step_m: float = 0.03
    maximum_rotation_per_step_rad: float = 0.20
    clip_normalized_action: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.controller_scale,
            self.maximum_translation_per_step_m,
            self.maximum_rotation_per_step_rad,
            self.clip_normalized_action,
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ArenaBridgeError("relative-IK limits must be finite and positive")


def absolute_targets_to_droid_relative_ik(
    targets: torch.Tensor,
    snapshot: DroidObservationSnapshot,
    *,
    config: DroidRelativeIKConfig | None = None,
) -> torch.Tensor:
    """Convert ``[xyz, quat_wxyz, closed_probability]`` to Arena DROID actions.

    The action is computed from the *current* Arena observation at dispatch time,
    so queue delay does not turn a stale precomputed delta into an unrelated
    command.  The returned command is ``[relative_xyz, relative_axis_angle,
    gripper_0_or_1]`` and matches ``droid_differential_ik``.
    """

    cfg = config or DroidRelativeIKConfig()
    values = torch.as_tensor(targets, dtype=torch.float32, device="cpu")
    if values.ndim != 2 or values.shape != (snapshot.num_envs, 8):
        raise ArenaBridgeError(
            f"absolute target must be [{snapshot.num_envs},8], got {tuple(values.shape)}"
        )
    if not torch.isfinite(values).all():
        raise ArenaBridgeError("absolute target contains non-finite values")

    current_pos = snapshot.policy["eef_pos"].to(dtype=torch.float32)
    current_quat = snapshot.policy["eef_quat"].to(dtype=torch.float32)
    if current_pos.shape != (snapshot.num_envs, 3):
        raise ArenaBridgeError(f"eef_pos must be [N,3], got {tuple(current_pos.shape)}")
    if current_quat.shape != (snapshot.num_envs, 4):
        raise ArenaBridgeError(
            f"eef_quat must be [N,4], got {tuple(current_quat.shape)}"
        )

    translation = values[:, :3] - current_pos
    translation_norm = torch.linalg.vector_norm(translation, dim=1, keepdim=True)
    translation_scale = torch.clamp(
        cfg.maximum_translation_per_step_m / translation_norm.clamp(min=1e-12), max=1.0
    )
    translation = translation * translation_scale

    target_quat = values[:, 3:7]
    target_quat = target_quat / torch.linalg.vector_norm(
        target_quat, dim=1, keepdim=True
    ).clamp(min=1e-12)
    current_quat = current_quat / torch.linalg.vector_norm(
        current_quat, dim=1, keepdim=True
    ).clamp(min=1e-12)
    inverse_current = current_quat.clone()
    inverse_current[:, 1:] *= -1
    relative_quat = _quaternion_multiply_wxyz(target_quat, inverse_current)
    rotation = quaternion_wxyz_to_axis_angle(relative_quat)
    rotation_norm = torch.linalg.vector_norm(rotation, dim=1, keepdim=True)
    rotation_scale = torch.clamp(
        cfg.maximum_rotation_per_step_rad / rotation_norm.clamp(min=1e-12), max=1.0
    )
    rotation = rotation * rotation_scale

    arm = torch.cat((translation, rotation), dim=1) / cfg.controller_scale
    arm = arm.clamp(-cfg.clip_normalized_action, cfg.clip_normalized_action)
    gripper = (values[:, 7:8] > 0.5).to(dtype=torch.float32)
    return torch.cat((arm, gripper), dim=1)


def droid_bounded_hold_action(snapshot: DroidObservationSnapshot) -> torch.Tensor:
    """Relative zero-motion action that preserves the observed gripper state."""

    gripper = snapshot.policy["gripper_pos"].to(dtype=torch.float32)
    if gripper.ndim != 2 or gripper.shape[0] != snapshot.num_envs:
        raise ArenaBridgeError("gripper_pos must be a batched tensor")
    closed = (gripper[:, :1] > 0.5).to(dtype=torch.float32)
    return torch.cat(
        (torch.zeros((snapshot.num_envs, 6), dtype=torch.float32), closed), dim=1
    )
