"""Official processor composition helpers shared by preflight and runners."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION


def postprocess_action_chunk_per_timestep(
    raw_chunk: Tensor,
    policy_postprocessor: Any,
    env_postprocessor: Any,
) -> Tensor:
    """Apply the exact evaluator output order to each action in a chunk.

    X-VLA emits ``[B,T,20]``. The checkpoint postprocessor preserves arbitrary
    leading dimensions, but the official X-VLA/LIBERO environment postprocessor
    only accepts ``[B,D]`` and converts the first 10 dimensions into a final 7D
    absolute command.
    """
    if raw_chunk.ndim != 3:
        raise ValueError(f"Expected a [B,T,D] action chunk, got {tuple(raw_chunk.shape)}")

    final_steps: list[Tensor] = []
    for step_index in range(raw_chunk.shape[1]):
        action = policy_postprocessor(raw_chunk[:, step_index, :])
        action = env_postprocessor({ACTION: action})[ACTION]
        if action.ndim != 2 or action.shape[-1] != 7:
            raise ValueError(
                f"Official processors produced {tuple(action.shape)} at step {step_index}, "
                "expected [B,7]"
            )
        if not torch.isfinite(action).all():
            raise ValueError(f"Official processors produced a non-finite action at step {step_index}")
        final_steps.append(action)

    return torch.stack(final_steps, dim=1)
