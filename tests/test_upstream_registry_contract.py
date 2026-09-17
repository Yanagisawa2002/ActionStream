from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

from lerobot.rollout.inference import (
    InferenceEngineConfig,
    create_inference_engine,
    register_inference_engine,
)


@InferenceEngineConfig.register_subclass("actionstream-registry-contract-test")
@dataclass
class _ExternalInferenceConfig(InferenceEngineConfig):
    marker: int = 7


_SENTINEL = object()


@register_inference_engine(_ExternalInferenceConfig)
def _build_external(config, **kwargs):
    assert config.marker == 7
    assert kwargs["task"] == "registry contract"
    return _SENTINEL


def test_external_exact_config_type_dispatches_through_public_registry():
    result = create_inference_engine(
        _ExternalInferenceConfig(),
        policy=MagicMock(),
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=SimpleNamespace(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=[],
        task="registry contract",
        fps=20.0,
        device="cpu",
        shutdown_event=None,
    )
    assert result is _SENTINEL


def test_duplicate_external_builder_registration_is_rejected():
    try:
        decorator = register_inference_engine(_ExternalInferenceConfig)
    except ValueError:
        # The historical pinned patch rejects at registration entry.
        return

    try:
        decorator(lambda **kwargs: None)
    except ValueError:
        return
    raise AssertionError("duplicate inference builder registration was accepted")
