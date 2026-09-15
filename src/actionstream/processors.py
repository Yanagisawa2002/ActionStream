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
        raise ValueError(
            f"Expected a [B,T,D] action chunk, got {tuple(raw_chunk.shape)}"
        )

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
            raise ValueError(
                f"Official processors produced a non-finite action at step {step_index}"
            )
        final_steps.append(action)

    return torch.stack(final_steps, dim=1)


def postprocess_xvla_libero_chunk_batched(
    raw_chunk: Tensor,
    policy_postprocessor: Any,
    env_postprocessor: Any,
) -> Tensor:
    """Batch the pinned stateless X-VLA/LIBERO output processors.

    Flatten batch/time only: every row still passes through both official
    processors, in the same order. Reject other pipelines, whose steps may
    depend on batch size or maintain temporal state.
    """
    from lerobot.processor import DeviceProcessorStep, UnnormalizerProcessorStep
    from lerobot.policies.xvla.processor_xvla import (
        XVLARotation6DToAxisAngleProcessorStep,
    )

    if [type(s) for s in policy_postprocessor.steps] != [
        UnnormalizerProcessorStep,
        DeviceProcessorStep,
    ] or [type(s) for s in env_postprocessor.steps] != [
        XVLARotation6DToAxisAngleProcessorStep
    ]:
        raise ValueError(
            "Batched processing requires the pinned stateless XVLA/LIBERO pipelines"
        )
    if raw_chunk.ndim != 3 or raw_chunk.shape[-1] != 20 or min(raw_chunk.shape[:2]) < 1:
        raise ValueError("Expected a nonempty [B,T,20] XVLA chunk")
    batch, steps, dimensions = raw_chunk.shape
    if not torch.isfinite(raw_chunk).all():
        raise ValueError("Official processors received invalid batched raw actions")
    action = policy_postprocessor(raw_chunk.reshape(batch * steps, dimensions))
    action = env_postprocessor({ACTION: action})[ACTION]
    if action.shape != (batch * steps, 7) or not torch.isfinite(action).all():
        raise ValueError("Official processors produced invalid batched native actions")
    return action.reshape(batch, steps, 7)
