"""Bounded native X-VLA/LIBERO development acceptance; no async runtime."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time
import traceback

import numpy as np
import torch


MODEL_REVISION = "12e8783e996944f5c97e490d37d4c145484ed70a"
LEROBOT_COMMIT = "73e1584473028a2d53ecfc856f5290db84507f90"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def array_stats(value) -> dict:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not array.size:
        raise ValueError(
            f"Expected nonempty numeric data, got {array.shape}/{array.dtype}"
        )
    finite = np.isfinite(array)
    valid = array[finite]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(finite.all()),
        "min": float(valid.min()) if valid.size else None,
        "max": float(valid.max()) if valid.size else None,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def require_array(value, shape: list[int] | None = None) -> dict:
    stats = array_stats(value)
    if not stats["finite"] or (shape is not None and stats["shape"] != shape):
        raise ValueError(f"Array contract failed: expected={shape}, actual={stats}")
    return stats


def tree_stats(tree: Mapping, prefix: str = "") -> dict:
    result = {}
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            result.update(tree_stats(value, name))
        elif isinstance(value, np.ndarray | torch.Tensor):
            result[name] = require_array(value)
    return result


def validate_config(config: dict) -> None:
    fixed = {
        "purpose": "development_only_native_synchronous_baseline",
        "model_id": "lerobot/xvla-libero",
        "model_revision": MODEL_REVISION,
        "lerobot_commit": LEROBOT_COMMIT,
        "suite": "libero_object",
        "task_id": 5,
        "control_mode": "absolute",
        "control_frequency_hz": 20,
        "max_control_steps": 300,
        "raw_action_shape": [1, 30, 20],
        "processed_action_shape": [1, 30, 7],
        "execute_steps_per_chunk": 30,
        "settle_steps": 10,
        "native_camera_shape": [1, 360, 360, 3],
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"Fixed development contract changed: {key}")
    if (
        len(config.get("smoke", [])) != 1
        or not 1 <= len(config.get("development", [])) <= 3
    ):
        raise ValueError("Require one smoke and at most three development resets")
    episodes = config["smoke"] + config["development"]
    for episode in episodes:
        if not re.fullmatch(r"[a-z0-9_]+", episode["id"]):
            raise ValueError("Episode id must be a simple directory name")
        if (
            not isinstance(episode["initial_state_index"], int)
            or episode["initial_state_index"] < 0
        ):
            raise ValueError("Initial-state index must be a nonnegative integer")
    for key in ("id", "initial_state_index", "seed"):
        values = [row[key] for row in episodes]
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate reset identity: {key}")


def validate_smoke_receipt(receipt: dict, config_sha256: str) -> None:
    episodes = receipt.get("rollout", {}).get("episodes", [])
    if not (
        receipt.get("phase") == "smoke"
        and receipt.get("config_sha256") == config_sha256
        and receipt.get("model_load", {}).get("status") == "PASS"
        and receipt.get("readiness", {}).get("headless_render") == "PASS"
        and receipt.get("rollout", {}).get("status") == "COMPLETED"
        and len(episodes) == 1
        and episodes[0].get("status") == "COMPLETED"
        and episodes[0].get("control_steps", 0) > 0
        and isinstance(episodes[0].get("success"), bool)
    ):
        raise ValueError(
            "Development requires a completed native smoke with the same config"
        )


def validate_observation(observation: dict) -> dict:
    for camera in ("image", "image2"):
        # The fixed LiberoEnv config renders at 360. Checkpoint feature metadata
        # says 256, while native XVLAPolicy._prepare_images resizes to the pinned
        # 224 target. Preserve native observations and that internal transform.
        stats = require_array(observation["pixels"][camera], [1, 360, 360, 3])
        if stats["dtype"] != "uint8" or stats["min"] == stats["max"]:
            raise ValueError(f"Native camera is missing or constant: {camera}")
    state = observation["robot_state"]
    require_array(state["eef"]["pos"], [1, 3])
    require_array(state["eef"]["quat"], [1, 4])
    require_array(state["gripper"]["qpos"], [1, 2])
    return tree_stats(observation)


def native_success(info: dict) -> bool:
    if "is_success" not in info:
        raise ValueError("Missing authoritative native is_success flag")
    values = np.asarray(info["is_success"])
    # Gymnasium may use an object container for numpy.bool_ values. Check the
    # actual scalar type, never coerce numeric/string/NaN values to truthiness.
    if values.shape != (1,) or not isinstance(values[0], (bool, np.bool_)):
        raise ValueError("Native is_success must contain exactly one boolean")
    return bool(values[0])


def validate_prepared_state(prepared: dict, checkpoint_max_state_dim: int) -> dict:
    # The XVLA-specific LIBERO processor constructs ee6d state at max_state_dim.
    # The checkpoint's input feature metadata describes an earlier 8D layout.
    return require_array(prepared["observation.state"], [1, checkpoint_max_state_dim])


def episode_outcome(
    *, steps: int, success: bool, terminated: bool, truncated: bool, cap: int
) -> dict:
    if steps <= 0:
        return {"status": "NOT_RUN", "success": None, "reason": "no_control_step"}
    if success:
        return {"status": "COMPLETED", "success": True, "reason": "native_is_success"}
    if terminated or truncated or steps == cap:
        return {
            "status": "COMPLETED",
            "success": False,
            "reason": "environment_terminated"
            if terminated
            else "environment_truncated"
            if truncated
            else "control_step_limit",
        }
    return {"status": "INCOMPLETE", "success": None, "reason": "interrupted"}


class Journal:
    def __init__(self, path: Path):
        self.handle = path.open("x", encoding="utf-8")

    def emit(self, event: str, **fields):
        self.handle.write(
            json.dumps(
                {"event": event, "monotonic_ns": time.monotonic_ns(), **fields},
                allow_nan=False,
            )
            + "\n"
        )
        self.handle.flush()

    def close(self):
        self.handle.close()


def save_frames(observation: dict, directory: Path, step: int) -> list[dict]:
    from PIL import Image

    frames = []
    for camera in ("image", "image2"):
        array = np.asarray(observation["pixels"][camera])[0]
        target = directory / f"step{step:03d}_{camera}.png"
        Image.fromarray(array).save(target)
        frames.append(
            {
                "path": str(target),
                "sha256": sha256(target),
                "orientation": "unmodified_policy_observation",
            }
        )
    return frames


def verify_egl(directory: Path) -> dict:
    """Render before loading weights; LIBERO cameras are checked separately."""
    import mujoco
    from OpenGL import GL
    from PIL import Image

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 3"/>'
        '<geom type="plane" size="2 2 .1" rgba=".5 .5 .5 1"/>'
        '<geom type="sphere" pos="0 0 .2" size=".2" rgba="1 .1 .1 1"/>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=64, width=64) as renderer:
        renderer.update_scene(data)
        frame = renderer.render()
        stats = require_array(frame, [64, 64, 3])
        if stats["min"] == stats["max"]:
            raise RuntimeError("Headless EGL probe produced a constant image")
        gl = {
            name: (GL.glGetString(key) or b"").decode()
            for name, key in (
                ("vendor", GL.GL_VENDOR),
                ("renderer", GL.GL_RENDERER),
                ("version", GL.GL_VERSION),
            )
        }
    target = directory / "egl_probe.png"
    Image.fromarray(frame).save(target)
    return {
        "status": "PASS",
        "frame": str(target),
        "frame_sha256": sha256(target),
        "image_stats": stats,
        "opengl": gl,
        "mujoco_version": mujoco.__version__,
    }


def verify_assets(config: dict, store: Path) -> dict:
    model = store / "assets/xvla-12e8783"
    files = {}
    for name, expected in config["checkpoint_sha256"].items():
        digest = sha256(model / name)
        if digest != expected:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
        files[name] = digest
    checkpoint = json.loads((model / "config.json").read_text())
    if (
        checkpoint["chunk_size"],
        checkpoint["n_action_steps"],
        checkpoint["action_mode"],
    ) != (30, 30, "ee6d"):
        raise ValueError("Pinned checkpoint action contract differs")
    packages = {
        name: importlib.metadata.version(name) for name in config["required_packages"]
    }
    if packages != config["required_packages"]:
        raise ValueError(f"Package version mismatch: {packages}")
    # Installation receipt is produced by the local provisioner, from the fixed
    # Git archive. Every installed LeRobot Python file must match that receipt.
    import lerobot

    provenance = json.loads((store / "environment_receipt.json").read_text())
    if provenance["lerobot_commit"] != LEROBOT_COMMIT:
        raise ValueError("LeRobot source commit differs")
    package_root = Path(lerobot.__file__).parent
    for relative, expected in provenance["lerobot_python_sha256"].items():
        if sha256(package_root / relative) != expected:
            raise ValueError(f"Installed LeRobot source differs: {relative}")
    bart_root = store / "cache/huggingface/hub/models--facebook--bart-large"
    if (bart_root / "refs/main").read_text().strip() != config["tokenizer_revision"]:
        raise ValueError("Offline tokenizer reference differs")
    for relative, expected in config["tokenizer_sha256"].items():
        if (
            sha256(bart_root / "snapshots" / config["tokenizer_revision"] / relative)
            != expected
        ):
            raise ValueError(f"Offline tokenizer file differs: {relative}")
    asset_receipt = json.loads((store / "libero_assets_receipt.json").read_text())
    if asset_receipt["revision"] != config["libero_assets_revision"]:
        raise ValueError("LIBERO asset revision differs")
    asset_root = Path(asset_receipt["target"]).resolve()
    if not asset_root.is_relative_to(store.resolve()):
        raise ValueError("LIBERO assets must be isolated in the owned store")
    for row in asset_receipt["files"]:
        if sha256(asset_root / row["path"]) != row["sha256"]:
            raise ValueError(f"LIBERO asset differs: {row['path']}")
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("A usable CUDA PyTorch build is required")
    return {
        "status": "ASSETS_AND_CUDA_VERIFIED",
        "headless_render": "NOT_RUN",
        "packages": packages,
        "checkpoint_files": files,
        "checkpoint_config": checkpoint,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "lerobot_path": str(package_root),
        "tokenizer_revision": config["tokenizer_revision"],
        "libero_assets_revision": asset_receipt["revision"],
        "libero_asset_root": str(asset_root),
        "environment_receipt_sha256": sha256(store / "environment_receipt.json"),
    }


def run_episode(backend, config: dict, episode: dict, root: Path) -> dict:
    from actionstream.processors import postprocess_action_chunk_per_timestep
    from libero.libero import benchmark, get_libero_path

    directory = root / episode["id"]
    directory.mkdir()
    journal = Journal(directory / "native_trace.jsonl")
    record = {
        **episode,
        "status": "NOT_RUN",
        "success": None,
        "control_steps": 0,
        "inference_calls": 0,
        "frames": [],
    }
    started = time.perf_counter()
    try:
        task = benchmark.get_benchmark_dict()[config["suite"]]().get_task(
            config["task_id"]
        )
        init_file = (
            Path(get_libero_path("init_states"))
            / task.problem_folder
            / Path(task.init_states_file).name
        )
        bddl_file = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        sub_env = backend._sub_env(config["task_id"])
        index = episode["initial_state_index"]
        if not 0 <= index < len(sub_env._init_states):
            raise ValueError("Development init-state index is unavailable")
        record["task"] = {
            "id": config["task_id"],
            "name": task.name,
            "instruction": task.language,
            "init_file": str(init_file),
            "init_file_sha256": sha256(init_file),
            "init_state": array_stats(sub_env._init_states[index]),
            "bddl_file": str(bddl_file),
            "bddl_sha256": sha256(bddl_file),
        }
        observation, _, instruction = backend.reset_episode(
            task_id=config["task_id"], seed=episode["seed"], initial_state_index=index
        )
        if instruction != config["instruction"]:
            raise ValueError(f"Unexpected native task instruction: {instruction}")
        if (
            backend.controller_frequency_hz(config["task_id"]) != 20
            or sub_env.num_steps_wait != config["settle_steps"]
        ):
            raise ValueError("Native control/settle settings differ")
        if any(robot.controller.use_delta for robot in sub_env._env.env.robots):
            raise ValueError("Native controller did not enter absolute action mode")
        observation_stats = validate_observation(observation)
        record["headless_render"] = "PASS"
        record["image_contract"] = {
            "native_camera_shape": config["native_camera_shape"],
            "checkpoint_resize_imgs_with_padding": backend.policy_cfg.resize_imgs_with_padding,
            "transform": "native XVLAPolicy._prepare_images / resize_with_pad",
        }
        record["settle_steps_excluded"] = sub_env.num_steps_wait
        record["native_env_class"] = (
            f"{type(sub_env).__module__}.{type(sub_env).__name__}"
        )
        record["reset_wall_s"] = time.perf_counter() - started
        record["frames"] += save_frames(observation, directory, 0)
        journal.emit(
            "reset",
            episode=episode,
            task=record["task"],
            observation=observation_stats,
            settle_steps=record["settle_steps_excluded"],
        )
        actions = []
        success = terminated = truncated = False
        model_times, inference_times, step_times = [], [], []
        for step in range(config["max_control_steps"]):
            control_started = time.perf_counter()
            if not actions:
                inference_started = time.perf_counter()
                prepared = backend.prepare_observation(observation, instruction)
                prepared_stats = tree_stats(prepared)
                validate_prepared_state(prepared, backend.policy_cfg.max_state_dim)
                record["inference_calls"] += 1
                journal.emit(
                    "inference_started",
                    ordinal=record["inference_calls"] - 1,
                    source_control_step=step,
                    instruction=instruction,
                    observation=observation_stats,
                    prepared=prepared_stats,
                )
                torch.cuda.synchronize()
                model_started = time.perf_counter()
                with torch.inference_mode():
                    raw = backend.policy.predict_action_chunk(prepared)
                torch.cuda.synchronize()
                model_s = time.perf_counter() - model_started
                raw_stats = require_array(raw, config["raw_action_shape"])
                with torch.inference_mode():
                    processed = postprocess_action_chunk_per_timestep(
                        raw,
                        policy_postprocessor=backend.postprocessor,
                        env_postprocessor=backend.env_postprocessor,
                    )
                final_stats = require_array(processed, config["processed_action_shape"])
                actions = list(
                    processed[0].detach().cpu().numpy().astype(np.float32, copy=True)
                )
                inference_s = time.perf_counter() - inference_started
                model_times.append(model_s)
                inference_times.append(inference_s)
                journal.emit(
                    "inference",
                    ordinal=record["inference_calls"] - 1,
                    source_control_step=step,
                    instruction=instruction,
                    raw_action=raw_stats,
                    final_action=final_stats,
                    model_s=model_s,
                    synchronous_inference_s=inference_s,
                )
            action = actions.pop(0)
            action_stats = require_array(action, [7])
            # Absolute axis-angle coordinates may exceed the wrapper's generic
            # [-1,1] Box; do not clip or change those native action semantics.
            if not -1 <= float(action[6]) <= 1:
                raise ValueError("Native gripper command is outside [-1,1]")
            before_step = time.perf_counter()
            result = backend.step(config["task_id"], action)
            step_s = time.perf_counter() - before_step
            record["control_steps"] = step + 1
            observation = result.observation
            flag = np.asarray(result.info.get("is_success"))
            journal.emit(
                "step_returned",
                control_step=step + 1,
                action=action.tolist(),
                action_stats=action_stats,
                native_step_s=step_s,
                success_container={
                    "shape": list(flag.shape),
                    "dtype": str(flag.dtype),
                    "values_repr": repr(flag.tolist()),
                },
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
            )
            observation_stats = validate_observation(observation)
            success = native_success(result.info)
            terminated, truncated = result.terminated, result.truncated
            step_times.append(step_s)
            journal.emit(
                "step",
                control_step=step + 1,
                action=action.tolist(),
                action_stats=action_stats,
                observation=observation_stats,
                reward=result.reward,
                native_is_success=success,
                terminated=terminated,
                truncated=truncated,
                native_step_s=step_s,
                synchronous_control_s=time.perf_counter() - control_started,
            )
            if (
                step == 0
                or (step + 1) % 50 == 0
                or success
                or terminated
                or truncated
                or step + 1 == config["max_control_steps"]
            ):
                record["frames"] += save_frames(observation, directory, step + 1)
            if success or terminated or truncated:
                break
            remaining = 1 / config["control_frequency_hz"] - (
                time.perf_counter() - control_started
            )
            if remaining > 0:
                time.sleep(remaining)
        record.update(
            episode_outcome(
                steps=record["control_steps"],
                success=success,
                terminated=terminated,
                truncated=truncated,
                cap=config["max_control_steps"],
            )
        )
        record["timing"] = {
            "model_seconds_sum": sum(model_times),
            "model_p50_s": float(np.median(model_times)),
            "synchronous_inference_p50_s": float(np.median(inference_times)),
            "native_step_p50_s": float(np.median(step_times)),
        }
    except BaseException as exc:
        record.update(
            status="ERROR",
            success=None,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        record["wall_s"] = time.perf_counter() - started
        journal.emit("episode_end", **record)
        journal.close()
        write_json(directory / "episode_receipt.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "development"), required=True)
    parser.add_argument("--smoke-receipt", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.phase == "development":
        if args.smoke_receipt is None:
            parser.error("--smoke-receipt is required for development")
        validate_smoke_receipt(
            json.loads(args.smoke_receipt.read_text()), sha256(args.config)
        )
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": 1,
        "phase": args.phase,
        "purpose": config["purpose"],
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(args.config),
        "command": sys.argv,
        "executable": sys.executable,
        "actionstream_sources": {
            name: sha256(Path(__file__).parent / name)
            for name in (
                "native_baseline.py",
                "lerobot_backend.py",
                "processors.py",
                "libero_config.py",
            )
        },
        "readiness": {"status": "NOT_RUN"},
        "model_load": {"status": "NOT_RUN"},
        "rollout": {"status": "NOT_RUN", "episodes": []},
        "task_successes": None,
    }
    backend = None
    started = time.perf_counter()
    active_stage = "readiness"

    def interrupted(signum, frame):
        raise TimeoutError(f"Supervisor terminated native run (signal {signum})")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        receipt[active_stage]["status"] = "RUNNING"
        write_json(args.output / "run_receipt.json", receipt)
        receipt["readiness"] = verify_assets(config, args.store)
        receipt["readiness"]["egl_probe"] = {"status": "RUNNING"}
        write_json(args.output / "run_receipt.json", receipt)
        receipt["readiness"]["egl_probe"] = verify_egl(args.output)
        from actionstream.lerobot_backend import LeRobotBackend

        active_stage = "model_load"
        receipt[active_stage]["status"] = "RUNNING"
        write_json(args.output / "run_receipt.json", receipt)
        loading = time.perf_counter()
        backend = LeRobotBackend(
            task_ids=[5],
            seed=config[args.phase][0]["seed"],
            suite=config["suite"],
            episode_length=300,
            model_id=str(args.store / "assets/xvla-12e8783"),
            model_revision=MODEL_REVISION,
            device="cuda",
        )
        receipt["model_load"] = {
            "status": "PASS",
            "wall_s": time.perf_counter() - loading,
            "class": f"{type(backend.policy).__module__}.{type(backend.policy).__name__}",
            "source_path": inspect.getfile(type(backend.policy)),
            "parameter_count": sum(p.numel() for p in backend.policy.parameters()),
            "device": str(next(backend.policy.parameters()).device),
        }
        receipt["rollout"]["status"] = "RUNNING"
        active_stage = "rollout"
        write_json(args.output / "run_receipt.json", receipt)
        for episode in config[args.phase]:
            result = run_episode(backend, config, episode, args.output)
            receipt["readiness"]["headless_render"] = "PASS"
            receipt["rollout"]["episodes"].append(result)
            write_json(args.output / "run_receipt.json", receipt)
            print(
                json.dumps(
                    {
                        "episode": episode["id"],
                        "status": result["status"],
                        "success": result["success"],
                        "steps": result["control_steps"],
                    }
                ),
                flush=True,
            )
        receipt["rollout"]["status"] = "COMPLETED"
        receipt["task_successes"] = sum(
            row["success"] is True for row in receipt["rollout"]["episodes"]
        )
        return 0
    except BaseException as exc:
        receipt.update(
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        receipt[active_stage]["status"] = "ERROR"
        if active_stage == "rollout":
            recorded = {row["id"] for row in receipt["rollout"]["episodes"]}
            for episode in config[args.phase]:
                path = args.output / episode["id"] / "episode_receipt.json"
                if path.exists() and episode["id"] not in recorded:
                    failed = json.loads(path.read_text())
                    receipt["rollout"]["episodes"].append(failed)
                    if failed.get("headless_render") == "PASS":
                        receipt["readiness"]["headless_render"] = "PASS"
        print(receipt["traceback"], file=sys.stderr, flush=True)
        return 1
    finally:
        if backend is not None:
            receipt["peak_torch_allocated_MiB"] = backend.peak_cuda_memory_mib
            try:
                backend.close()
            except Exception as exc:
                receipt["close_error"] = repr(exc)
        receipt["wall_s"] = time.perf_counter() - started
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "run_receipt.json", receipt)


if __name__ == "__main__":
    raise SystemExit(main())
