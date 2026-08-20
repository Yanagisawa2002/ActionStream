"""Audit one pinned official LIBERO reset for a native-Isaac proxy task.

This utility is deliberately state-only: it constructs the official LeRobot
LIBERO environment, selects one reset, and records the robot, object, and target
geometry needed to build an auditable native proxy.  It neither loads a policy
nor executes an action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _name(model: Any, kind: str, index: int) -> str | None:
    accessor = getattr(model, f"{kind}_id2name", None)
    if not callable(accessor):
        return None
    value = accessor(int(index))
    return None if value is None else str(value)


def _joint_span(model: Any, joint_name: str, state: str) -> np.ndarray:
    address = getattr(model, f"get_joint_{state}_addr")(joint_name)
    if isinstance(address, tuple):
        return np.arange(int(address[0]), int(address[1]), dtype=np.int64)
    return np.asarray([int(address)], dtype=np.int64)


def _object_record(inner: Any, sim: Any, object_name: str) -> dict[str, Any]:
    objects = getattr(inner, "objects_dict", None)
    if not isinstance(objects, Mapping) or object_name not in objects:
        available = sorted(objects) if isinstance(objects, Mapping) else []
        raise ValueError(f"object {object_name!r} unavailable; available={available}")
    entity = objects[object_name]
    joints = list(getattr(entity, "joints", ()))
    if len(joints) != 1:
        raise ValueError(f"object {object_name!r} must own one joint, got {joints}")
    joint_name = str(joints[0])
    qpos_indexes = _joint_span(sim.model, joint_name, "qpos")
    qvel_indexes = _joint_span(sim.model, joint_name, "qvel")
    contact_geoms: list[dict[str, Any]] = []
    for geom_name in getattr(entity, "contact_geoms", ()):
        geom_id = int(sim.model.geom_name2id(str(geom_name)))
        body_id = int(np.asarray(sim.model.geom_bodyid)[geom_id])
        contact_geoms.append(
            {
                "name": str(geom_name),
                "body_id": body_id,
                "body_name": _name(sim.model, "body", body_id),
                "local_position_xyz": np.asarray(sim.model.geom_pos[geom_id], dtype=float).tolist(),
                "local_orientation_wxyz": np.asarray(sim.model.geom_quat[geom_id], dtype=float).tolist(),
                "size": np.asarray(sim.model.geom_size[geom_id], dtype=float).tolist(),
                "type": int(np.asarray(sim.model.geom_type)[geom_id]),
            }
        )
    root_body_name = str(getattr(entity, "root_body", ""))
    root_body_id = int(sim.model.body_name2id(root_body_name))
    return {
        "name": object_name,
        "joint_name": joint_name,
        "joint_type": int(np.asarray(sim.model.jnt_type)[sim.model.joint_name2id(joint_name)]),
        "qpos_indexes": qpos_indexes.tolist(),
        "qvel_indexes": qvel_indexes.tolist(),
        "qpos": np.asarray(sim.data.qpos[qpos_indexes], dtype=float).tolist(),
        "qvel": np.asarray(sim.data.qvel[qvel_indexes], dtype=float).tolist(),
        "root_body_name": root_body_name,
        "root_body_world_position_xyz": np.asarray(sim.data.body_xpos[root_body_id], dtype=float).tolist(),
        "root_body_world_orientation_wxyz": np.asarray(sim.data.body_xquat[root_body_id], dtype=float).tolist(),
        "root_body_mass_kg": float(np.asarray(sim.model.body_mass)[root_body_id]),
        "contact_geoms": contact_geoms,
    }


def _site_record(sim: Any, site_name: str) -> dict[str, Any]:
    site_id = int(sim.model.site_name2id(site_name))
    body_id = int(np.asarray(sim.model.site_bodyid)[site_id])
    return {
        "name": site_name,
        "body_id": body_id,
        "body_name": _name(sim.model, "body", body_id),
        "world_position_xyz": np.asarray(sim.data.site_xpos[site_id], dtype=float).tolist(),
        "world_orientation_matrix": np.asarray(sim.data.site_xmat[site_id], dtype=float)
        .reshape(3, 3)
        .tolist(),
        "local_position_xyz": np.asarray(sim.model.site_pos[site_id], dtype=float).tolist(),
        "local_orientation_wxyz": np.asarray(sim.model.site_quat[site_id], dtype=float).tolist(),
        "size": np.asarray(sim.model.site_size[site_id], dtype=float).tolist(),
        "type": int(np.asarray(sim.model.site_type)[site_id]),
        "parent_body_world_position_xyz": np.asarray(sim.data.body_xpos[body_id], dtype=float).tolist(),
        "parent_body_world_orientation_wxyz": np.asarray(sim.data.body_xquat[body_id], dtype=float).tolist(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--initial-state-index", type=int, required=True)
    parser.add_argument("--environment-seed", type=int, required=True)
    parser.add_argument("--object-name", action="append", required=True)
    parser.add_argument("--site-name", action="append", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite reset audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lerobot_root = args.lerobot_root.resolve()
    if not (lerobot_root / ".git").exists():
        raise ValueError(f"LeRobot checkout is not a Git repository: {lerobot_root}")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from actionstream.libero_config import ensure_isolated_libero_config

    ensure_isolated_libero_config()
    from lerobot.envs import make_env, make_env_config

    env_cfg = make_env_config(
        "libero",
        task=args.suite,
        task_ids=[args.task_id],
        control_mode="absolute",
        episode_length=300,
        max_parallel_tasks=1,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vector_env = envs[args.suite][args.task_id]
    sub_env = vector_env.envs[0]
    sub_env.init_state_id = int(args.initial_state_index)
    try:
        _observation, _info = vector_env.reset(seed=[int(args.environment_seed)])
        wrapped = getattr(sub_env, "_env", None)
        sim = getattr(wrapped, "sim", None)
        inner = getattr(wrapped, "env", None)
        if wrapped is None or sim is None or inner is None:
            raise RuntimeError("official LIBERO reset internals are unavailable")
        raw = inner._get_observations()
        bddl_value = getattr(inner, "bddl_file_name", None)
        bddl_path = Path(str(bddl_value)).resolve() if bddl_value else None
        result = {
            "schema_version": 1,
            "evidence_class": "development_state_only_official_libero_reset_audit",
            "policy_loaded": False,
            "actions_executed": 0,
            "task_success_claimed": False,
            "suite": args.suite,
            "task_id": int(args.task_id),
            "initial_state_index": int(args.initial_state_index),
            "environment_seed": int(args.environment_seed),
            "task_description": str(sub_env.task_description),
            "lerobot": {
                "root": str(lerobot_root),
                "commit": _git_commit(lerobot_root),
            },
            "bddl": {
                "path": None if bddl_path is None else str(bddl_path),
                "sha256": (
                    _sha256_file(bddl_path)
                    if bddl_path is not None and bddl_path.is_file()
                    else None
                ),
            },
            "robot": {
                "eef_position_xyz": np.asarray(raw["robot0_eef_pos"], dtype=float).tolist(),
                "eef_orientation_wxyz": np.asarray(raw["robot0_eef_quat"], dtype=float).tolist(),
                "joint_position": np.asarray(raw["robot0_joint_pos"], dtype=float).tolist(),
                "gripper_qpos": np.asarray(raw["robot0_gripper_qpos"], dtype=float).tolist(),
            },
            "objects": [
                _object_record(inner, sim, object_name)
                for object_name in args.object_name
            ],
            "sites": [_site_record(sim, site_name) for site_name in args.site_name],
        }
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    finally:
        for task_map in envs.values():
            for env in task_map.values():
                env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
