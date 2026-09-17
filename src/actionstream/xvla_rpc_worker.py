"""Trusted X-VLA/LIBERO worker factory for the ActionStream TCP RPC server.

This worker intentionally reuses the pinned :class:`LeRobotBackend` so remote
inference follows the same official policy and processor chain as the existing
GPU evidence. The server host must have the same checkpoint/LIBERO assets and
EGL environment as the native runners.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from actionstream.lerobot_backend import LeRobotBackend, MODEL_REVISION


class XVLARemoteWorker:
    def __init__(self) -> None:
        store = os.environ.get("ACTIONSTREAM_RPC_STORE")
        if not store:
            raise RuntimeError("ACTIONSTREAM_RPC_STORE is required")
        task_ids_raw = os.environ.get("ACTIONSTREAM_RPC_TASK_IDS", "5")
        task_ids = [int(item.strip()) for item in task_ids_raw.split(",") if item.strip()]
        if not task_ids:
            raise RuntimeError("ACTIONSTREAM_RPC_TASK_IDS must contain at least one id")
        suite = os.environ.get("ACTIONSTREAM_RPC_SUITE", "libero_object")
        device = os.environ.get("ACTIONSTREAM_RPC_DEVICE", "cuda")
        model_path = Path(store) / "assets/xvla-12e8783"
        self.backend = LeRobotBackend(
            task_ids=task_ids,
            seed=int(os.environ.get("ACTIONSTREAM_RPC_SEED", "142")),
            suite=suite,
            episode_length=300,
            model_id=str(model_path),
            model_revision=MODEL_REVISION,
            device=device,
        )

    def __call__(self, observation, task):
        output = self.backend.infer_action_chunk(dict(observation), task)
        return torch.from_numpy(output.actions.copy())

    def reset(self) -> None:
        self.backend.reset_runtime()


def make_xvla_remote_worker() -> XVLARemoteWorker:
    return XVLARemoteWorker()
