"""Learned chunk-provider boundary used by the Arena policy adapter."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import torch

from actionstream.arena_bridge import DroidObservationSnapshot


class ArenaChunkProvider(Protocol):
    """Batched learned policy that returns absolute target trajectories."""

    @property
    def supports_rtc(self) -> bool: ...

    def reset(self) -> None: ...

    def predict_targets(
        self,
        snapshot: DroidObservationSnapshot,
        task: str,
        *,
        rtc: bool,
    ) -> torch.Tensor:
        """Return finite ``[T,N,8]`` xyz/quaternion/gripper targets."""

    def close(self) -> None: ...


ProviderFactory = Callable[[Mapping[str, Any], str], ArenaChunkProvider]


def resolve_provider_factory(path: str) -> ProviderFactory:
    if ":" not in path:
        raise ValueError("provider_factory must use 'module:function' syntax")
    module_name, attribute = path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError(f"provider factory is not callable: {path}")
    return factory
