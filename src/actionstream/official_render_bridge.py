"""Auditable LIBERO rendering from a mapped native-Isaac robot state.

The bridge deliberately owns rendering only.  Isaac remains the source of the
measured policy state and physics, while the pinned official LIBERO environment
provides the two camera images.  The first implementation is reset-scoped: task
objects remain at the matching official initial state and only the Panda arm
and gripper state are written back before rendering.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _finite_vector(value: Any, length: int, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape ({length},)")
    return array


def validate_unbatched_bridge_state(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the mapped Isaac state used by both policy and renderer."""

    def child(group: str, key: str, length: int) -> list[float]:
        group_value = value.get(group)
        if not isinstance(group_value, Mapping) or key not in group_value:
            raise ValueError(f"bridge state missing {group}.{key}")
        return _finite_vector(
            group_value[key], length, label=f"bridge state {group}.{key}"
        ).tolist()

    eef_mat = np.asarray(value.get("eef", {}).get("mat"), dtype=np.float64)
    if eef_mat.shape != (3, 3) or not np.isfinite(eef_mat).all():
        raise ValueError("bridge state eef.mat must be finite with shape (3,3)")
    return {
        "eef": {
            "mat": eef_mat.tolist(),
            "pos": child("eef", "pos", 3),
            "quat": child("eef", "quat", 4),
        },
        "gripper": {
            "qpos": child("gripper", "qpos", 2),
            "qvel": child("gripper", "qvel", 2),
        },
        "joints": {
            "pos": child("joints", "pos", 7),
            "vel": child("joints", "vel", 7),
        },
    }


@dataclass(frozen=True)
class OfficialRenderBridgeResult:
    observation: dict[str, Any]
    provenance: dict[str, Any]


def _robot_indexes(robot: Any, name: str, length: int) -> np.ndarray:
    if not hasattr(robot, name):
        raise RuntimeError(f"official LIBERO robot lacks {name}")
    indexes = np.asarray(getattr(robot, name), dtype=np.int64)
    if indexes.shape != (length,):
        raise RuntimeError(
            f"official LIBERO robot {name} must have shape ({length},), "
            f"got {indexes.shape}"
        )
    return indexes


def _orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    desired = np.asarray(desired, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if desired.shape != (3, 3) or current.shape != (3, 3):
        raise ValueError("orientation matrices must have shape (3,3)")
    return 0.5 * sum(
        np.cross(current[:, index], desired[:, index]) for index in range(3)
    )


def _damped_least_squares_delta(
    jacobian: np.ndarray,
    error: np.ndarray,
    *,
    damping: float,
) -> np.ndarray:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    if jacobian.ndim != 2 or error.shape != (jacobian.shape[0],):
        raise ValueError("Jacobian and task-space error shapes are inconsistent")
    regularizer = float(damping) ** 2 * np.eye(jacobian.shape[0])
    return jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + regularizer,
        error,
    )


def _solve_official_eef_pose(
    *,
    sim: Any,
    robot: Any,
    arm_qpos: np.ndarray,
    arm_qvel: np.ndarray,
    target_position: np.ndarray,
    target_orientation: np.ndarray,
    maximum_iterations: int = 80,
    position_tolerance_m: float = 1e-4,
    orientation_tolerance_rad: float = 1e-3,
    damping: float = 1e-3,
    maximum_joint_step_rad: float = 0.08,
) -> dict[str, Any]:
    """Retarget an Isaac EEF pose onto the canonical LIBERO Panda."""

    controller = getattr(robot, "controller", None)
    eef_name = getattr(controller, "eef_name", None)
    if not isinstance(eef_name, str) or not eef_name:
        raise RuntimeError("official LIBERO controller has no EEF site name")
    site_id = int(sim.model.site_name2id(eef_name))
    initial_qpos = np.asarray(sim.data.qpos[arm_qpos], dtype=np.float64).copy()
    qpos = initial_qpos.copy()
    joint_ids_value = getattr(robot, "_ref_joint_indexes", None)
    joint_limits: np.ndarray | None = None
    if joint_ids_value is not None:
        joint_ids = np.asarray(joint_ids_value, dtype=np.int64)
        if joint_ids.shape == (7,):
            candidate_limits = np.asarray(sim.model.jnt_range[joint_ids], dtype=np.float64)
            if candidate_limits.shape == (7, 2):
                joint_limits = candidate_limits

    converged = False
    iterations = 0
    position_error = np.full(3, np.inf)
    orientation_error = np.full(3, np.inf)
    for iterations in range(1, int(maximum_iterations) + 1):
        sim.data.qpos[arm_qpos] = qpos
        sim.data.qvel[arm_qvel] = 0.0
        sim.forward()
        current_position = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64)
        current_orientation = np.asarray(
            sim.data.site_xmat[site_id], dtype=np.float64
        ).reshape(3, 3)
        position_error = target_position - current_position
        orientation_error = _orientation_error(target_orientation, current_orientation)
        if (
            np.linalg.norm(position_error) <= position_tolerance_m
            and np.linalg.norm(orientation_error) <= orientation_tolerance_rad
        ):
            converged = True
            break
        jacobian_position = np.asarray(
            sim.data.get_site_jacp(eef_name), dtype=np.float64
        ).reshape(3, -1)[:, arm_qvel]
        jacobian_orientation = np.asarray(
            sim.data.get_site_jacr(eef_name), dtype=np.float64
        ).reshape(3, -1)[:, arm_qvel]
        jacobian = np.vstack((jacobian_position, jacobian_orientation))
        delta = _damped_least_squares_delta(
            jacobian,
            np.concatenate((position_error, orientation_error)),
            damping=damping,
        )
        largest = float(np.max(np.abs(delta)))
        if largest > maximum_joint_step_rad:
            delta *= maximum_joint_step_rad / largest
        qpos = qpos + delta
        if joint_limits is not None:
            qpos = np.clip(qpos, joint_limits[:, 0] + 1e-6, joint_limits[:, 1] - 1e-6)

    sim.data.qpos[arm_qpos] = qpos
    sim.data.qvel[arm_qvel] = 0.0
    sim.forward()
    current_position = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64)
    current_orientation = np.asarray(
        sim.data.site_xmat[site_id], dtype=np.float64
    ).reshape(3, 3)
    position_error = target_position - current_position
    orientation_error = _orientation_error(target_orientation, current_orientation)
    return {
        "method": "damped_least_squares_eef_pose_retargeting",
        "converged": converged,
        "iterations": iterations,
        "initial_arm_qpos": initial_qpos.tolist(),
        "solved_arm_qpos": qpos.tolist(),
        "target_eef_position": target_position.tolist(),
        "rendered_eef_position": current_position.tolist(),
        "position_error_l2_m": float(np.linalg.norm(position_error)),
        "orientation_error_l2_rad": float(np.linalg.norm(orientation_error)),
        "position_tolerance_m": position_tolerance_m,
        "orientation_tolerance_rad": orientation_tolerance_rad,
        "damping": damping,
        "maximum_joint_step_rad": maximum_joint_step_rad,
        "render_arm_qvel": "zeroed because no physics step is taken",
    }


def apply_isaac_state_to_official_renderer(
    sub_env: Any,
    robot_state: Mapping[str, Any],
    *,
    pose_mode: str = "eef_ik",
) -> OfficialRenderBridgeResult:
    """Write one mapped Isaac robot state into a reset official LIBERO env.

    ``sub_env`` is the synchronous LeRobot ``LiberoEnv`` beneath its vector
    wrapper.  The caller must reset it to the frozen task/initial-state first.
    No physics step is taken here: the method writes qpos/qvel, forwards MuJoCo,
    refreshes observables, and returns the two official camera observations.
    """

    state = validate_unbatched_bridge_state(robot_state)
    wrapped = getattr(sub_env, "_env", None)
    if wrapped is None:
        raise RuntimeError("official LIBERO sub-environment has not been reset")
    sim = getattr(wrapped, "sim", None)
    robots = getattr(wrapped, "robots", None)
    inner = getattr(wrapped, "env", None)
    if sim is None or not robots or inner is None:
        raise RuntimeError("official LIBERO rendering internals are unavailable")
    if len(robots) != 1:
        raise RuntimeError(f"expected one official LIBERO robot, got {len(robots)}")
    robot = robots[0]
    arm_qpos = _robot_indexes(robot, "_ref_joint_pos_indexes", 7)
    arm_qvel = _robot_indexes(robot, "_ref_joint_vel_indexes", 7)
    gripper_qpos = _robot_indexes(robot, "_ref_gripper_joint_pos_indexes", 2)
    gripper_qvel = _robot_indexes(robot, "_ref_gripper_joint_vel_indexes", 2)

    if pose_mode not in {"eef_ik", "joint_replay"}:
        raise ValueError(f"unsupported official-render bridge pose mode: {pose_mode}")
    requested_arm_qpos = np.asarray(state["joints"]["pos"], dtype=np.float64)
    requested_arm_qvel = np.asarray(state["joints"]["vel"], dtype=np.float64)
    requested_gripper_qpos = np.asarray(state["gripper"]["qpos"], dtype=np.float64)
    requested_gripper_qvel = np.asarray(state["gripper"]["qvel"], dtype=np.float64)
    ik_provenance: dict[str, Any] | None = None
    if pose_mode == "joint_replay":
        sim.data.qpos[arm_qpos] = requested_arm_qpos
        sim.data.qvel[arm_qvel] = requested_arm_qvel
    else:
        ik_provenance = _solve_official_eef_pose(
            sim=sim,
            robot=robot,
            arm_qpos=arm_qpos,
            arm_qvel=arm_qvel,
            target_position=np.asarray(state["eef"]["pos"], dtype=np.float64),
            target_orientation=np.asarray(state["eef"]["mat"], dtype=np.float64),
        )
    sim.data.qpos[gripper_qpos] = requested_gripper_qpos
    sim.data.qvel[gripper_qvel] = requested_gripper_qvel
    sim.forward()
    inner._update_observables(force=True)
    raw_observation = inner._get_observations()
    formatted = sub_env._format_raw_obs(raw_observation)
    pixels = formatted.get("pixels")
    if not isinstance(pixels, Mapping) or set(pixels) != {"image", "image2"}:
        raise RuntimeError(
            "official bridge requires exactly agentview image and eye-in-hand image2"
        )
    images: dict[str, np.ndarray] = {}
    for key in ("image", "image2"):
        image = np.asarray(pixels[key])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise RuntimeError(
                f"official bridge {key} must be uint8 HxWx3, got {image.shape}/{image.dtype}"
            )
        images[key] = image.copy()

    applied_arm_qpos = np.asarray(sim.data.qpos[arm_qpos], dtype=np.float64).copy()
    applied_arm_qvel = np.asarray(sim.data.qvel[arm_qvel], dtype=np.float64).copy()
    applied_gripper_qpos = np.asarray(
        sim.data.qpos[gripper_qpos], dtype=np.float64
    ).copy()
    applied_gripper_qvel = np.asarray(
        sim.data.qvel[gripper_qvel], dtype=np.float64
    ).copy()
    rendered_eef = _finite_vector(
        raw_observation["robot0_eef_pos"], 3, label="rendered robot0_eef_pos"
    )
    requested_eef = np.asarray(state["eef"]["pos"], dtype=np.float64)
    write_errors = {
        "gripper_qpos_max_abs": float(
            np.max(np.abs(applied_gripper_qpos - requested_gripper_qpos))
        ),
        "gripper_qvel_max_abs": float(
            np.max(np.abs(applied_gripper_qvel - requested_gripper_qvel))
        ),
    }
    if pose_mode == "joint_replay":
        write_errors.update(
            {
                "arm_qpos_max_abs": float(
                    np.max(np.abs(applied_arm_qpos - requested_arm_qpos))
                ),
                "arm_qvel_max_abs": float(
                    np.max(np.abs(applied_arm_qvel - requested_arm_qvel))
                ),
            }
        )
    return OfficialRenderBridgeResult(
        observation={
            "pixels": images,
            "robot_state": copy.deepcopy(state),
        },
        provenance={
            "scope": "matching official reset objects plus Isaac-mapped Panda EEF/gripper state",
            "pose_mode": pose_mode,
            "physics_steps_after_write": 0,
            "arm_qpos_indexes": arm_qpos.tolist(),
            "arm_qvel_indexes": arm_qvel.tolist(),
            "gripper_qpos_indexes": gripper_qpos.tolist(),
            "gripper_qvel_indexes": gripper_qvel.tolist(),
            "requested_robot_state": state,
            "applied_arm_qpos": applied_arm_qpos.tolist(),
            "applied_arm_qvel": applied_arm_qvel.tolist(),
            "applied_gripper_qpos": applied_gripper_qpos.tolist(),
            "applied_gripper_qvel": applied_gripper_qvel.tolist(),
            "rendered_eef_pos": rendered_eef.tolist(),
            "requested_vs_rendered_eef_position_l2_m": float(
                np.linalg.norm(requested_eef - rendered_eef)
            ),
            "ik": ik_provenance,
            "write_errors": write_errors,
            "writeback_exact_at_1e-12": all(error <= 1e-12 for error in write_errors.values()),
        },
    )
