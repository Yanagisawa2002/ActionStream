"""M0 machine, environment, processor, and one-chunk contract validation."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import HfApi

from actionstream.libero_config import ensure_isolated_libero_config


MODEL_ID = "lerobot/xvla-libero"
MODEL_REVISION = "12e8783e996944f5c97e490d37d4c145484ed70a"
EXPECTED_CHUNK_SIZE = 30
EXPECTED_ACTION_STEPS = 30
EXPECTED_MODEL_ACTION_DIM = 20
EXPECTED_ENV_ACTION_DIM = 7
EXPECTED_LEROBOT_VERSION = "0.6.2"
EXPECTED_LEROBOT_COMMIT = "73e1584473028a2d53ecfc856f5290db84507f90"
EXPECTED_LEROBOT_URL = (
    "https://github.com/Yanagisawa2002/lerobot/archive/"
    f"{EXPECTED_LEROBOT_COMMIT}.tar.gz"
)


def _verify_lerobot_install() -> dict[str, str]:
    distribution = importlib.metadata.distribution("lerobot")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("Pinned LeRobot install is missing direct_url.json provenance")
    direct_url = json.loads(direct_url_text)
    url = str(direct_url.get("url", ""))
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if commit is None and url == EXPECTED_LEROBOT_URL:
        commit = EXPECTED_LEROBOT_COMMIT
    if distribution.version != EXPECTED_LEROBOT_VERSION:
        raise RuntimeError(
            f"Expected LeRobot {EXPECTED_LEROBOT_VERSION}, got {distribution.version}"
        )
    if commit != EXPECTED_LEROBOT_COMMIT:
        raise RuntimeError(f"Expected LeRobot commit {EXPECTED_LEROBOT_COMMIT}, got {commit!r}")
    if url != EXPECTED_LEROBOT_URL:
        raise RuntimeError(f"Expected LeRobot URL {EXPECTED_LEROBOT_URL}, got {url!r}")
    return {
        "version": distribution.version,
        "commit": commit,
        "url": url,
    }


def _shape_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _shape_tree(child) for key, child in value.items()}
    if isinstance(value, torch.Tensor):
        result: dict[str, Any] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
        if value.numel() and (value.is_floating_point() or value.is_complex()):
            result["finite"] = bool(torch.isfinite(value).all().item())
        return result
    array = np.asarray(value)
    result = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.size and np.issubdtype(array.dtype, np.number):
        result["finite"] = bool(np.isfinite(array).all())
    return result


def _feature_dict(features: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {
            "type": str(feature.type),
            "shape": list(feature.shape),
        }
        for key, feature in features.items()
    }


def _cuda_memory() -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "free_mib": free_bytes / 2**20,
        "total_mib": total_bytes / 2**20,
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def _safe_absolute_pose_hold(observation: dict[str, Any], env_postprocessor: Any) -> np.ndarray:
    """Build a one-step pose hold through the official X-VLA/LIBERO converter."""
    from lerobot.utils.constants import ACTION

    eef_pos = torch.as_tensor(observation["robot_state"]["eef"]["pos"], dtype=torch.float32)
    eef_mat = torch.as_tensor(observation["robot_state"]["eef"]["mat"], dtype=torch.float32)
    rotation_6d = torch.cat([eef_mat[:, :, 0], eef_mat[:, :, 1]], dim=-1)

    raw_action = torch.zeros((eef_pos.shape[0], EXPECTED_MODEL_ACTION_DIM), dtype=torch.float32)
    raw_action[:, :3] = eef_pos
    raw_action[:, 3:9] = rotation_6d
    raw_action[:, 9] = -1.0
    final_action = env_postprocessor({ACTION: raw_action})[ACTION]

    if final_action.ndim != 2 or final_action.shape[-1] != EXPECTED_ENV_ACTION_DIM:
        raise RuntimeError(f"Safe pose hold produced invalid shape {tuple(final_action.shape)}")
    if not torch.isfinite(final_action).all():
        raise RuntimeError("Safe pose hold produced a non-finite action")
    if torch.allclose(final_action, torch.zeros_like(final_action)):
        raise RuntimeError("Refusing to step LIBERO with an all-zero absolute command")
    return final_action.cpu().numpy()


def run_preflight(
    output: Path,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    config_file = ensure_isolated_libero_config()

    # Imports below this point may import libero.libero, so the isolated config
    # must already exist and LIBERO_CONFIG_PATH must already be set.
    import lerobot
    from lerobot.configs import PreTrainedConfig
    from lerobot.envs import (
        make_env,
        make_env_config,
        make_env_pre_post_processors,
        preprocess_observation,
    )
    from lerobot.policies import make_policy, make_pre_post_processors

    if platform.system() != "Linux":
        raise RuntimeError(f"LIBERO requires Linux, got {platform.platform()}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"ActionStream requires Python 3.12, got {platform.python_version()}")
    lerobot_install = _verify_lerobot_install()
    if lerobot.__version__ != lerobot_install["version"]:
        raise RuntimeError(
            "LeRobot module/distribution version mismatch: "
            f"module={lerobot.__version__}, distribution={lerobot_install['version']}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise RuntimeError("Headless preflight requires MUJOCO_GL=egl and PYOPENGL_PLATFORM=egl")

    model_info = HfApi().model_info(
        model_id,
        revision=model_revision,
        files_metadata=True,
    )
    resolved_model_revision = model_info.sha
    if not resolved_model_revision:
        raise RuntimeError(f"Could not resolve a revision SHA for {model_id}")
    if resolved_model_revision != model_revision:
        raise RuntimeError(
            f"Checkpoint revision resolved to {resolved_model_revision}, "
            f"expected pinned SHA {model_revision}"
        )

    checkpoint_cfg = PreTrainedConfig.from_pretrained(
        model_id,
        revision=resolved_model_revision,
    )
    checkpoint_snapshot = {
        "type": checkpoint_cfg.type,
        "chunk_size": checkpoint_cfg.chunk_size,
        "n_action_steps": checkpoint_cfg.n_action_steps,
        "action_mode": checkpoint_cfg.action_mode,
        "dtype": str(checkpoint_cfg.dtype),
        "input_features": _feature_dict(checkpoint_cfg.input_features),
        "output_features": _feature_dict(checkpoint_cfg.output_features),
        "num_image_views": checkpoint_cfg.num_image_views,
        "empty_cameras": checkpoint_cfg.empty_cameras,
        "resize_imgs_with_padding": list(checkpoint_cfg.resize_imgs_with_padding),
        "max_state_dim": checkpoint_cfg.max_state_dim,
    }
    if checkpoint_cfg.chunk_size != EXPECTED_CHUNK_SIZE:
        raise RuntimeError(f"Checkpoint chunk_size changed: {checkpoint_cfg.chunk_size}")
    if checkpoint_cfg.n_action_steps != EXPECTED_ACTION_STEPS:
        raise RuntimeError(f"Checkpoint n_action_steps changed: {checkpoint_cfg.n_action_steps}")
    if checkpoint_cfg.action_mode != "ee6d":
        raise RuntimeError(f"Checkpoint action_mode changed: {checkpoint_cfg.action_mode}")

    env_cfg = make_env_config(
        "libero",
        task="libero_object",
        task_ids=[0],
        control_mode="absolute",
        episode_length=800,
        max_parallel_tasks=1,
    )
    policy_cfg = copy.deepcopy(checkpoint_cfg)
    policy_cfg.device = "cuda"
    policy_cfg.pretrained_path = Path(model_id)
    policy_cfg.pretrained_revision = resolved_model_revision

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs["libero_object"][0]

    policy = None
    try:
        observation, reset_info = env.reset(seed=[142])
        raw_observation_contract = _shape_tree(observation)
        safe_hold = _safe_absolute_pose_hold(observation, env_postprocessor)
        next_observation, reward, terminated, truncated, step_info = env.step(safe_hold)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_before_model = _cuda_memory()

        load_started = time.perf_counter()
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={}).eval()
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=model_id,
            pretrained_revision=resolved_model_revision,
            preprocessor_overrides={
                "device_processor": {"device": str(policy.config.device)},
                "rename_observations_processor": {"rename_map": {}},
            },
        )

        prepared_observation = preprocess_observation(next_observation)
        prepared_observation["task"] = list(env.call("task_description"))
        env_processed_observation = env_preprocessor(prepared_observation)
        model_observation = preprocessor(env_processed_observation)

        policy.reset()
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            raw_chunk = policy.predict_action_chunk(model_observation)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started

        chunk_policy_postprocessor_support: dict[str, Any]
        try:
            policy_processed_chunk = postprocessor(raw_chunk)
            chunk_policy_postprocessor_support = {
                "accepted": True,
                "shape": list(policy_processed_chunk.shape),
                "dtype": str(policy_processed_chunk.dtype),
                "device": str(policy_processed_chunk.device),
            }
        except Exception as exc:  # pragma: no cover - records an upstream contract change
            chunk_policy_postprocessor_support = {
                "accepted": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        from actionstream.processors import postprocess_action_chunk_per_timestep

        final_chunk = postprocess_action_chunk_per_timestep(
            raw_chunk,
            policy_postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        )
        if tuple(raw_chunk.shape) != (1, EXPECTED_CHUNK_SIZE, EXPECTED_MODEL_ACTION_DIM):
            raise RuntimeError(f"Unexpected raw chunk shape: {tuple(raw_chunk.shape)}")
        if tuple(final_chunk.shape) != (1, EXPECTED_CHUNK_SIZE, EXPECTED_ENV_ACTION_DIM):
            raise RuntimeError(f"Unexpected final chunk shape: {tuple(final_chunk.shape)}")

        result = {
            "status": "passed",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": {
                "os": platform.platform(),
                "python": platform.python_version(),
                "lerobot": lerobot.__version__,
                "lerobot_install": lerobot_install,
                "torch": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "disk": {
                    "path": "/",
                    "free_bytes": shutil.disk_usage("/").free,
                    "total_bytes": shutil.disk_usage("/").total,
                },
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("hf-libero", "mujoco", "robosuite", "transformers")
                },
                "headless": {
                    "MUJOCO_GL": os.environ["MUJOCO_GL"],
                    "PYOPENGL_PLATFORM": os.environ["PYOPENGL_PLATFORM"],
                },
                "libero_config_file": str(config_file),
                "memory_before_model": memory_before_model,
                "memory_after_inference": _cuda_memory(),
            },
            "checkpoint": {
                "model_id": model_id,
                "revision_sha": resolved_model_revision,
                "config": checkpoint_snapshot,
                "model_load_seconds": model_load_seconds,
            },
            "libero_smoke": {
                "suite": "libero_object",
                "task_id": 0,
                "seed": 142,
                "control_mode": "absolute",
                "episode_length": 800,
                "observation": raw_observation_contract,
                "action_space_shape": list(env.single_action_space.shape),
                "safe_hold_shape": list(safe_hold.shape),
                "safe_hold_all_zero": bool(np.allclose(safe_hold, 0.0)),
                "reward": np.asarray(reward).tolist(),
                "terminated": np.asarray(terminated).tolist(),
                "truncated": np.asarray(truncated).tolist(),
                "reset_info_keys": sorted(reset_info),
                "step_info_keys": sorted(step_info),
            },
            "processor_contract": {
                "ordered_stages": [
                    "preprocess_observation",
                    "task_description_insertion",
                    "xvla_libero_env_preprocessor",
                    "checkpoint_policy_preprocessor",
                    "predict_action_chunk",
                    "checkpoint_policy_postprocessor_per_timestep",
                    "xvla_libero_env_postprocessor_per_timestep",
                ],
                "after_env_preprocessor": _shape_tree(env_processed_observation),
                "model_observation": _shape_tree(model_observation),
                "policy_postprocessor_chunk_probe": chunk_policy_postprocessor_support,
                "raw_chunk": _shape_tree(raw_chunk),
                "inference_latency_seconds": inference_seconds,
                "final_chunk": _shape_tree(final_chunk),
                "final_per_step_shape": list(final_chunk[:, 0, :].shape),
            },
        }

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        env.close()
        del policy
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/preflight/preflight.json"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    args = parser.parse_args()
    result = run_preflight(
        args.output,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
