from __future__ import annotations

import torch

from actionstream.processors import postprocess_action_chunk_per_timestep
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
