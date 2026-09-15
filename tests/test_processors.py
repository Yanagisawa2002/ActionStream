from __future__ import annotations

import torch
import pytest

from actionstream.processors import (
    postprocess_action_chunk_per_timestep,
    postprocess_xvla_libero_chunk_batched,
)
from lerobot.policies.xvla.processor_xvla import make_xvla_libero_pre_post_processors


def test_official_action_postprocessing_produces_finite_7d_actions() -> None:
    raw_chunk = torch.zeros((1, 2, 20), dtype=torch.float32)
    raw_chunk[:, :, :3] = torch.tensor([0.1, -0.2, 0.3])
    raw_chunk[:, :, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    raw_chunk[:, :, 9] = 1.0
    _, env_postprocessor = make_xvla_libero_pre_post_processors()

    final_chunk = postprocess_action_chunk_per_timestep(
        raw_chunk,
        policy_postprocessor=lambda action: action,
        env_postprocessor=env_postprocessor,
    )

    assert final_chunk.shape == (1, 2, 7)
    assert torch.isfinite(final_chunk).all()
    assert torch.all(final_chunk[:, :, -1] == 1.0)


def pinned_postprocessors():
    from lerobot.processor import (
        DeviceProcessorStep,
        UnnormalizerProcessorStep,
        PolicyProcessorPipeline,
    )
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )

    policy = PolicyProcessorPipeline(
        steps=[
            UnnormalizerProcessorStep(features={}, norm_map={}),
            DeviceProcessorStep(device="cpu"),
        ],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    _, env = make_xvla_libero_pre_post_processors()
    return policy, env


@pytest.mark.parametrize("batch,steps", [(1, 30), (3, 7)])
def test_batched_official_processors_preserve_every_action_and_order(batch, steps):
    policy, env = pinned_postprocessors()
    raw = torch.randn(batch, steps, 20, generator=torch.Generator().manual_seed(781))
    expected = postprocess_action_chunk_per_timestep(raw, policy, env)
    actual = postprocess_xvla_libero_chunk_batched(raw, policy, env)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Time is a leading dimension even for a non-contiguous input view.
    sliced = raw[:, ::2]
    torch.testing.assert_close(
        postprocess_xvla_libero_chunk_batched(sliced, policy, env),
        postprocess_action_chunk_per_timestep(sliced, policy, env),
        rtol=0,
        atol=0,
    )


def test_batched_processing_rejects_unknown_pipelines_and_nonfinite_actions():
    policy, env = pinned_postprocessors()
    with pytest.raises(ValueError, match="invalid batched"):
        postprocess_xvla_libero_chunk_batched(
            torch.full((1, 30, 20), float("nan")), policy, env
        )
    policy.steps = []
    with pytest.raises(ValueError, match="pinned stateless"):
        postprocess_xvla_libero_chunk_batched(torch.zeros(1, 30, 20), policy, env)
