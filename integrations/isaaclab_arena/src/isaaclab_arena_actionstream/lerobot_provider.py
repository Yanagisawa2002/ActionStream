"""Batched X-VLA/SmolVLA provider for the Arena DROID bridge.

The provider deliberately reuses the pinned LeRobot checkpoint and processor
factories.  Arena observations are translated into the checkpoint's LIBERO
feature contract, and processed policy actions are normalized to an internal
absolute target trajectory.  The control-thread policy then converts each
dispatched target to Arena's current-state-relative DROID IK command.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from actionstream.arena_bridge import DroidObservationSnapshot
from actionstream.arena_protocol import load_arena_protocol
from actionstream.isaac_learned import (
    libero_state_from_isaac,
    map_xvla_chunk_to_isaac,
)


def _axis_angle_to_quaternion_wxyz(axis_angle: np.ndarray) -> np.ndarray:
    vector = np.asarray(axis_angle, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    half = angle / 2.0
    xyz = vector * (math.sin(half) / angle)
    return np.asarray([math.cos(half), *xyz], dtype=np.float64)


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    value = np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return value / np.linalg.norm(value)


def _camera_to_uint8(image: torch.Tensor) -> np.ndarray:
    value = image.detach().cpu()
    if value.is_floating_point():
        maximum = float(value.max()) if value.numel() else 0.0
        if maximum <= 1.0:
            value = value * 255.0
        value = value.round().clamp(0, 255).to(torch.uint8)
    else:
        value = value.clamp(0, 255).to(torch.uint8)
    # The established native-Isaac/X-VLA bridge flips the render target
    # vertically before policy inference.  Keep that contract explicit.
    return torch.flip(value, dims=(1,)).numpy().copy()


class LeRobotArenaProvider:
    """One batched learned checkpoint and its exact pinned processors."""

    def __init__(self, config: Mapping[str, Any], model_key: str) -> None:
        protocol = load_arena_protocol(str(config["protocol_path"]))
        if protocol.sha256 != config["protocol_sha256"]:
            raise ValueError("provider protocol hash mismatch")
        model = next(row for row in protocol.models if row["key"] == model_key)
        self.model = model
        self.device = str(config.get("device", "cuda"))
        self.control_frequency_hz = float(
            protocol.raw.get("control_frequency_hz", 20.0)
        )
        self._previous_joint_pos: np.ndarray | None = None
        self._rtc_latency_max_seconds = 0.0
        self._previous_raw_chunk: torch.Tensor | None = None

        from lerobot.configs import PreTrainedConfig
        from lerobot.envs import make_env_config, make_env_pre_post_processors
        from lerobot.policies import make_policy, make_pre_post_processors

        self.env_cfg = make_env_config(
            "libero",
            task="libero_object",
            task_ids=[0],
            control_mode=str(model["control_mode"]),
            episode_length=800,
            max_parallel_tasks=1,
        )
        policy_cfg = PreTrainedConfig.from_pretrained(
            str(model["model_id"]), revision=str(model["revision"])
        )
        policy_cfg.device = self.device
        policy_cfg.pretrained_path = Path(str(model["model_id"]))
        policy_cfg.pretrained_revision = str(model["revision"])
        self.policy = make_policy(
            cfg=policy_cfg,
            env_cfg=self.env_cfg,
            rename_map=dict(model.get("rename_map", {})),
        ).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(model["model_id"]),
            pretrained_revision=str(model["revision"]),
            preprocessor_overrides={
                "device_processor": {"device": str(self.policy.config.device)},
                "rename_observations_processor": {
                    "rename_map": dict(model.get("rename_map", {}))
                },
            },
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=self.env_cfg,
            policy_cfg=policy_cfg,
        )

    @property
    def supports_rtc(self) -> bool:
        capability = getattr(self.policy, "supports_rtc", None)
        return bool(callable(capability) and capability())

    def reset(self) -> None:
        self.policy.reset()
        for pipeline in (
            self.env_preprocessor,
            self.preprocessor,
            self.postprocessor,
            self.env_postprocessor,
        ):
            pipeline.reset()
        self._previous_joint_pos = None
        self._rtc_latency_max_seconds = 0.0
        self._previous_raw_chunk = None

    def _robot_state(self, snapshot: DroidObservationSnapshot) -> dict[str, Any]:
        eef_pos = snapshot.policy["eef_pos"].numpy().astype(np.float64)
        eef_quat = snapshot.policy["eef_quat"].numpy().astype(np.float64)
        joints = snapshot.policy["joint_pos"].numpy().astype(np.float64)
        closed = (
            snapshot.policy["gripper_pos"].numpy().reshape(snapshot.num_envs, -1)[:, 0]
        )
        if self._previous_joint_pos is None:
            velocities = np.zeros_like(joints)
        else:
            velocities = (joints - self._previous_joint_pos) * self.control_frequency_hz
        self._previous_joint_pos = joints.copy()

        rows = [
            libero_state_from_isaac(
                end_effector_xyz=eef_pos[index],
                end_effector_wxyz=eef_quat[index],
                gripper_aperture_m=float(
                    (1.0 - np.clip(closed[index], 0.0, 1.0)) * 0.085
                ),
                joint_positions=joints[index],
                joint_velocities=velocities[index],
            )
            for index in range(snapshot.num_envs)
        ]

        def stack(*keys: str) -> np.ndarray:
            values: list[Any] = []
            for row in rows:
                current: Any = row
                for key in keys:
                    current = current[key]
                values.append(current)
            return np.asarray(values, dtype=np.float64)

        return {
            "eef": {
                "mat": stack("eef", "mat"),
                "pos": stack("eef", "pos"),
                "quat": stack("eef", "quat"),
            },
            "gripper": {
                "qpos": stack("gripper", "qpos"),
                "qvel": stack("gripper", "qvel"),
            },
            "joints": {
                "pos": stack("joints", "pos"),
                "vel": stack("joints", "vel"),
            },
        }

    def _prepare(self, snapshot: DroidObservationSnapshot, task: str) -> dict[str, Any]:
        from lerobot.envs import preprocess_observation

        observation = {
            "pixels": {
                "image": _camera_to_uint8(snapshot.cameras["external_camera_rgb"]),
                "image2": _camera_to_uint8(snapshot.cameras["wrist_camera_rgb"]),
            },
            "robot_state": self._robot_state(snapshot),
        }
        prepared = preprocess_observation(observation)
        prepared["task"] = [task] * snapshot.num_envs
        return self.preprocessor(self.env_preprocessor(prepared))

    def _finalize(self, raw_chunk: torch.Tensor, *, rtc: bool) -> torch.Tensor:
        from lerobot.utils.constants import ACTION

        if raw_chunk.ndim != 3:
            raise ValueError(
                f"policy chunk must be [N,T,D], got {tuple(raw_chunk.shape)}"
            )
        if rtc:
            processed = self.postprocessor(raw_chunk)
            policy_steps = [
                processed[:, index, :] for index in range(processed.shape[1])
            ]
        else:
            policy_steps = [
                self.postprocessor(raw_chunk[:, index, :])
                for index in range(raw_chunk.shape[1])
            ]
        final = [
            self.env_postprocessor({ACTION: step})[ACTION] for step in policy_steps
        ]
        actions = torch.stack(final, dim=1)
        if (
            actions.ndim != 3
            or actions.shape[2] != 7
            or not torch.isfinite(actions).all()
        ):
            raise ValueError(
                f"processed LIBERO chunk must be finite [N,T,7], got {tuple(actions.shape)}"
            )
        return actions

    def _absolute_targets(
        self, actions: np.ndarray, snapshot: DroidObservationSnapshot
    ) -> np.ndarray:
        all_targets: list[np.ndarray] = []
        positions = snapshot.policy["eef_pos"].numpy()
        for env_index in range(snapshot.num_envs):
            mapped = map_xvla_chunk_to_isaac(
                actions[env_index],
                initial_target_xyz=positions[env_index],
                workspace_xyz=((-0.5, 1.2), (-0.8, 0.8), (0.02, 1.5)),
                maximum_translation_per_step_m=0.03,
            ).commands
            targets = np.empty((mapped.shape[0], 8), dtype=np.float32)
            targets[:, :3] = mapped[:, :3]
            targets[:, 3:7] = np.stack(
                [_axis_angle_to_quaternion_wxyz(row) for row in mapped[:, 3:6]], axis=0
            )
            targets[:, 7] = (mapped[:, 6] < 0.0).astype(np.float32)
            all_targets.append(targets)
        return np.stack(all_targets, axis=1)

    def _relative_targets(
        self, actions: np.ndarray, snapshot: DroidObservationSnapshot
    ) -> np.ndarray:
        positions = snapshot.policy["eef_pos"].numpy().astype(np.float64).copy()
        quaternions = snapshot.policy["eef_quat"].numpy().astype(np.float64).copy()
        result = np.empty((actions.shape[1], snapshot.num_envs, 8), dtype=np.float32)
        for step in range(actions.shape[1]):
            for env_index in range(snapshot.num_envs):
                row = actions[env_index, step]
                delta = np.asarray(row[:3], dtype=np.float64)
                norm = float(np.linalg.norm(delta))
                if norm > 0.03:
                    delta *= 0.03 / norm
                positions[env_index] += delta
                rotation = np.asarray(row[3:6], dtype=np.float64)
                rotation_norm = float(np.linalg.norm(rotation))
                if rotation_norm > 0.20:
                    rotation *= 0.20 / rotation_norm
                quaternions[env_index] = _quaternion_multiply_wxyz(
                    _axis_angle_to_quaternion_wxyz(rotation), quaternions[env_index]
                )
                result[step, env_index, :3] = positions[env_index]
                result[step, env_index, 3:7] = quaternions[env_index]
                result[step, env_index, 7] = float(row[6] > 0.0)
        return result

    def predict_targets(
        self,
        snapshot: DroidObservationSnapshot,
        task: str,
        *,
        rtc: bool,
    ) -> torch.Tensor:
        if rtc and not self.supports_rtc:
            raise RuntimeError(f"policy {self.model['key']} does not support RTC")
        batch = self._prepare(snapshot, task)
        kwargs: dict[str, Any] = {}
        if rtc:
            horizon = int(self.model.get("rtc_execution_horizon", 10))
            delay = math.ceil(self._rtc_latency_max_seconds * self.control_frequency_hz)
            prefix = None
            if self._previous_raw_chunk is not None:
                prefix = self._previous_raw_chunk[:, :horizon, :]
            kwargs = {"inference_delay": delay, "prev_chunk_left_over": prefix}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            raw_chunk = self.policy.predict_action_chunk(batch, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        if rtc:
            self._rtc_latency_max_seconds = max(self._rtc_latency_max_seconds, latency)
        self._previous_raw_chunk = raw_chunk.detach()
        processed = self._finalize(raw_chunk, rtc=rtc).detach().cpu().numpy()
        if self.model["control_mode"] == "absolute":
            targets = self._absolute_targets(processed, snapshot)
        elif self.model["control_mode"] == "relative":
            targets = self._relative_targets(processed, snapshot)
        else:
            raise ValueError(f"unsupported control mode: {self.model['control_mode']}")
        return torch.from_numpy(targets)

    def close(self) -> None:
        del self.policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_provider(config: Mapping[str, Any], model_key: str) -> LeRobotArenaProvider:
    return LeRobotArenaProvider(config, model_key)
