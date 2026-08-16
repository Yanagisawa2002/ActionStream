"""Render one development-only official-LIBERO / Isaac-state bridge frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from actionstream.official_render_bridge import apply_isaac_state_to_official_renderer


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _save_image(image: np.ndarray, path: Path) -> str:
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
    return _sha256_file(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--environment-seed", type=int, default=2026081601)
    parser.add_argument(
        "--pose-mode",
        choices=("eef_ik", "joint_replay"),
        default="eef_ik",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"refusing to reuse bridge probe output: {output}")
    output.mkdir(parents=True)
    native_summary_path = args.native_summary.resolve()
    native_summary = json.loads(native_summary_path.read_text(encoding="utf-8"))
    records = native_summary.get("request_records", [])
    if not records:
        raise ValueError("native summary contains no request_records")
    native_state = records[0]["robot_state"]

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
        episode_length=800,
        max_parallel_tasks=1,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vector_env = envs[args.suite][args.task_id]
    sub_env = vector_env.envs[0]
    sub_env.init_state_id = args.initial_state_index
    try:
        reference, _ = vector_env.reset(seed=[args.environment_seed])
        bridge = apply_isaac_state_to_official_renderer(
            sub_env, native_state, pose_mode=args.pose_mode
        )
        hashes = {
            "reference_image_sha256": _save_image(
                np.asarray(reference["pixels"]["image"])[0], output / "reference_image.png"
            ),
            "reference_image2_sha256": _save_image(
                np.asarray(reference["pixels"]["image2"])[0], output / "reference_image2.png"
            ),
            "bridge_image_sha256": _save_image(
                bridge.observation["pixels"]["image"], output / "bridge_image.png"
            ),
            "bridge_image2_sha256": _save_image(
                bridge.observation["pixels"]["image2"], output / "bridge_image2.png"
            ),
        }
        summary = {
            "schema_version": 1,
            "evidence_class": "development_render_only_official_libero_isaac_state_bridge_probe",
            "policy_inference_executed": False,
            "task_success_evidence": False,
            "suite": args.suite,
            "task_id": args.task_id,
            "initial_state_index": args.initial_state_index,
            "environment_seed": args.environment_seed,
            "pose_mode": args.pose_mode,
            "native_summary_path": str(native_summary_path),
            "native_summary_sha256": _sha256_file(native_summary_path),
            "bridge": bridge.provenance,
            "image_hashes": hashes,
        }
        _write_json(output / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True))
    finally:
        for task_map in envs.values():
            for env in task_map.values():
                env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
