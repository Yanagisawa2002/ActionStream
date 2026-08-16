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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", default="xvla")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026081601)
    parser.add_argument("--episode-length", type=int, default=800)
    parser.add_argument("--device", default="cuda")
    return parser


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


def main() -> int:
    args = _parser().parse_args()
    try:
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
                initial_state_index=0,
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
            image, image_sha256 = _load_rgb(request["image"])
            image2, image2_sha256 = _load_rgb(request["image2"])
            observation = {
                "pixels": {"image": image, "image2": image2},
                "robot_state": _batch_robot_state(request["robot_state"]),
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
                    "image_sha256": image_sha256,
                    "image2_sha256": image2_sha256,
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
