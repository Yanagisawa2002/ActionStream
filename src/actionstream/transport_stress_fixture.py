"""Spawn-importable deterministic fixture for transport stress validation."""

from __future__ import annotations

import time

import torch


def create_transport():
    def infer(observation, _task):
        mode = observation.get("mode", "ok")
        if mode == "hang":
            while True:
                time.sleep(1.0)
        if mode == "error":
            raise ConnectionError("fixture transport offline")
        value = float(observation.get("value", 1.0))
        return torch.full((1, 4, 2), value, dtype=torch.float32)

    return infer
