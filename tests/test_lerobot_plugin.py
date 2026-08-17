from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from actionstream.lerobot_inference import ActionStreamInferenceEngine

import lerobot.rollout.inference as lerobot_inference


if not hasattr(lerobot_inference, "register_inference_engine"):
    pytest.skip(
        "pinned upstream registry patch is not applied", allow_module_level=True
    )

from lerobot.rollout.inference import create_inference_engine  # noqa: E402
from lerobot_policy_actionstream import ActionStreamRolloutInferenceConfig  # noqa: E402


def test_plugin_config_is_registered_and_factory_builds_formal_backend() -> None:
    policy = MagicMock()
    policy.predict_action_chunk = MagicMock()
    policy.config = SimpleNamespace(use_amp=False)
    config = ActionStreamRolloutInferenceConfig(
        inference_timeout_s=1.25,
        bounded_hold_steps=3,
    )

    engine = create_inference_engine(
        config,
        policy=policy,
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["joint.pos"],
        task="mock task",
        fps=30.0,
        device="cpu",
    )

    assert config.type == "actionstream"
    assert isinstance(engine, ActionStreamInferenceEngine)
    assert engine._config.inference_timeout_s == 1.25
    assert engine._config.bounded_hold_steps == 3
