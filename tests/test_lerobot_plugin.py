from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from actionstream.lerobot_inference import ActionStreamInferenceEngine
from actionstream.rpc_transport import TcpInferenceTransport

import lerobot.rollout.inference as lerobot_inference

assert hasattr(lerobot_inference, "register_inference_engine")

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


def test_plugin_builds_tcp_transport_without_changing_core_transport_enum() -> None:
    config = ActionStreamRolloutInferenceConfig(
        transport_mode="tcp",
        tcp_host="127.0.0.1",
        tcp_port=50051,
        tcp_reset_timeout_s=12.5,
        minimum_request_interval_steps=4,
    )

    runtime = config.runtime_config()
    transport = config.build_transport()

    assert runtime.transport_mode == "direct"
    assert runtime.minimum_request_interval_steps == 4
    assert isinstance(transport, TcpInferenceTransport)
    assert transport._reset_timeout_s == 12.5
    assert transport.telemetry().reset_required
    transport.close()
