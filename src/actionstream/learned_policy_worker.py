"""Persistent JSON-lines LeRobot worker for native-Isaac development runs.

Isaac Sim and current LeRobot intentionally remain in their already-provisioned
Python environments.  This process owns the learned checkpoint and exact
LeRobot processors; the Isaac process owns physics, cameras, safety, and queue
semantics.  Standard output is reserved for machine-readable JSON lines.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch

from actionstream.current_baselines import CurrentLeRobotBackend, load_protocol
from actionstream.isaac_learned import validate_worker_request
from actionstream.official_render_bridge import apply_isaac_state_to_official_renderer


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rgb(path_value: Any) -> tuple[np.ndarray, str]:
    path = Path(str(path_value)).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing non-empty RGB observation: {path}")
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"invalid RGB observation shape: {rgb.shape}")
    return rgb[None, ...], _sha256_file(path)


def _batch_robot_state(value: Mapping[str, Any]) -> dict[str, Any]:
    def array(path: tuple[str, ...], shape: tuple[int, ...]) -> np.ndarray:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                raise ValueError(f"robot_state missing {'.'.join(path)}")
            current = current[key]
        converted = np.asarray(current, dtype=np.float64)
        if converted.shape != shape or not np.isfinite(converted).all():
            raise ValueError(
                f"robot_state {'.'.join(path)} must be finite with shape {shape}"
            )
        return converted[None, ...]

    return {
        "eef": {
            "mat": array(("eef", "mat"), (3, 3)),
            "pos": array(("eef", "pos"), (3,)),
            "quat": array(("eef", "quat"), (4,)),
        },
        "gripper": {
            "qpos": array(("gripper", "qpos"), (2,)),
            "qvel": array(("gripper", "qvel"), (2,)),
        },
        "joints": {
            "pos": array(("joints", "pos"), (7,)),
            "vel": array(("joints", "vel"), (7,)),
        },
    }


def _save_rendered_rgb(
    directory: Path, request_id: int, key: str, image: np.ndarray
) -> tuple[Path, str]:
    path = directory / f"request_{request_id:06d}_{key}.png"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite official render frame: {path}")
    Image.fromarray(image, mode="RGB").save(path)
    return path, _sha256_file(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", default="xvla")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026081601)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=800)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-output-directory", type=Path, default=None)
    return parser


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


def main() -> int:
    args = _parser().parse_args()
    render_output_directory: Path | None = None
    try:
        if args.render_output_directory is not None:
            render_output_directory = args.render_output_directory.expanduser().resolve()
            render_output_directory.mkdir(parents=True, exist_ok=False)
        protocol = load_protocol(args.protocol)
        if args.model not in protocol.models:
            raise ValueError(f"unknown model {args.model!r}")
        spec = protocol.models[args.model]
        # Current LeRobot/LIBERO writes informational messages to stdout.  Keep
        # the IPC stream parseable and preserve those messages in worker stderr.
        with redirect_stdout(sys.stderr):
            backend = CurrentLeRobotBackend(
                spec=spec,
                task_ids=[args.task_id],
                suite=args.suite,
                episode_length=args.episode_length,
                seed=args.seed,
                device=args.device,
            )
            _template, _info, template_instruction = backend.reset_episode(
                task_id=args.task_id,
                seed=args.seed,
                initial_state_index=args.initial_state_index,
            )
        _emit(
            {
                "event": "ready",
                "model_key": spec.key,
                "model_id": spec.model_id,
                "model_revision": spec.revision,
                "chunk_size": spec.chunk_size,
                "supports_rtc": backend.supports_rtc,
                "template_instruction": template_instruction,
                "suite": args.suite,
                "task_id": args.task_id,
                "initial_state_index": args.initial_state_index,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit({"event": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        return 2

    try:
        for line in sys.stdin:
            stripped = line.strip()
            if not stripped:
                continue
            request = json.loads(stripped)
            if request.get("command") == "close":
                _emit({"event": "closed"})
                break
            validate_worker_request(request)
            render_request = request.get("official_render_bridge")
            bridge_provenance: dict[str, Any] | None = None
            image_path: Path
            image2_path: Path
            if render_request is not None:
                if render_output_directory is None:
                    raise RuntimeError(
                        "official render request requires --render-output-directory"
                    )
                with redirect_stdout(sys.stderr):
                    bridge_result = apply_isaac_state_to_official_renderer(
                        backend.official_render_sub_env(args.task_id),
                        request["robot_state"],
                        pose_mode=str(render_request.get("pose_mode", "eef_ik")),
                        object_states=render_request["object_states"],
                    )
                unbatched_image = bridge_result.observation["pixels"]["image"]
                unbatched_image2 = bridge_result.observation["pixels"]["image2"]
                image_path, image_sha256 = _save_rendered_rgb(
                    render_output_directory,
                    int(request["request_id"]),
                    "image",
                    unbatched_image,
                )
                image2_path, image2_sha256 = _save_rendered_rgb(
                    render_output_directory,
                    int(request["request_id"]),
                    "image2",
                    unbatched_image2,
                )
                image = unbatched_image[None, ...]
                image2 = unbatched_image2[None, ...]
                bridge_provenance = bridge_result.provenance
                policy_robot_state = bridge_result.observation["robot_state"]
                observation_source = "dynamic_official_render_bridge"
            else:
                image_path = Path(str(request["image"])).resolve()
                image2_path = Path(str(request["image2"])).resolve()
                image, image_sha256 = _load_rgb(image_path)
                image2, image2_sha256 = _load_rgb(image2_path)
                policy_robot_state = request["robot_state"]
                observation_source = "supplied_rgb_files"
            observation = {
                "pixels": {"image": image, "image2": image2},
                "robot_state": _batch_robot_state(policy_robot_state),
            }
            with redirect_stdout(sys.stderr):
                output = backend.infer_action_chunk(
                    observation,
                    str(request["instruction"]),
                )
            actions_sha256 = hashlib.sha256(
                np.asarray(output.actions, dtype=np.float32).tobytes(order="C")
            ).hexdigest()
            _emit(
                {
                    "event": "inference",
                    "request_id": int(request["request_id"]),
                    "actions": output.actions.tolist(),
                    "actions_sha256": actions_sha256,
                    "raw_shape": list(output.raw_shape),
                    "raw_dtype": output.raw_dtype,
                    "model_latency_seconds": output.model_latency_seconds,
                    "observation_source": observation_source,
                    "image_path": str(image_path),
                    "image2_path": str(image2_path),
                    "image_sha256": image_sha256,
                    "image2_sha256": image2_sha256,
                    "official_render_bridge": bridge_provenance,
                    "peak_cuda_memory_mib": backend.peak_cuda_memory_mib,
                }
            )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit({"event": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        return 3
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
