"""Register ActionStream as a third-party ``lerobot-rollout`` backend."""

from __future__ import annotations

from dataclasses import dataclass

from actionstream.lerobot_inference import (
    ActionStreamInferenceConfig,
    ActionStreamInferenceEngine,
)
from lerobot.rollout.inference import InferenceEngineConfig

try:
    from lerobot.rollout.inference import register_inference_engine
except ImportError as exc:  # pragma: no cover - exercised by installation preflight
    raise RuntimeError(
        "This ActionStream plugin needs LeRobot's pluggable inference-engine registry. "
        "Apply upstream/lerobot/0001-feat-rollout-allow-third-party-inference-engines.patch to "
        "the pinned LeRobot revision before running lerobot-rollout."
    ) from exc


@InferenceEngineConfig.register_subclass("actionstream")
@dataclass
class ActionStreamRolloutInferenceConfig(InferenceEngineConfig):
    """CLI-visible configuration for ``--inference.type=actionstream``."""

    inference_timeout_s: float = 5.0
    bounded_hold_steps: int = 2
    retry_backoff_s: float = 0.05
    max_consecutive_failures: int = 10
    join_timeout_s: float = 3.0
    latest_only_fallback: bool = True
    transport_mode: str = "direct"
    process_transport_factory: str | None = None
    process_transport_start_method: str = "spawn"
    process_transport_startup_timeout_s: float = 30.0
    process_transport_terminate_timeout_s: float = 1.0
    telemetry_jsonl_path: str | None = None

    def runtime_config(self) -> ActionStreamInferenceConfig:
        return ActionStreamInferenceConfig(
            inference_timeout_s=self.inference_timeout_s,
            bounded_hold_steps=self.bounded_hold_steps,
            retry_backoff_s=self.retry_backoff_s,
            max_consecutive_failures=self.max_consecutive_failures,
            join_timeout_s=self.join_timeout_s,
            latest_only_fallback=self.latest_only_fallback,
            transport_mode=self.transport_mode,
            process_transport_factory=self.process_transport_factory,
            process_transport_start_method=self.process_transport_start_method,
            process_transport_startup_timeout_s=self.process_transport_startup_timeout_s,
            process_transport_terminate_timeout_s=self.process_transport_terminate_timeout_s,
            telemetry_jsonl_path=self.telemetry_jsonl_path,
        )


@register_inference_engine(ActionStreamRolloutInferenceConfig)
def _build_actionstream_inference_engine(
    config: ActionStreamRolloutInferenceConfig,
    **kwargs,
) -> ActionStreamInferenceEngine:
    robot_wrapper = kwargs["robot_wrapper"]
    return ActionStreamInferenceEngine(
        policy=kwargs["policy"],
        preprocessor=kwargs["preprocessor"],
        postprocessor=kwargs["postprocessor"],
        hw_features=kwargs["hw_features"],
        task=kwargs["task"],
        device=kwargs["device"],
        robot_type=robot_wrapper.robot_type,
        config=config.runtime_config(),
        shutdown_event=kwargs["shutdown_event"],
    )


__all__ = [
    "ActionStreamInferenceEngine",
    "ActionStreamRolloutInferenceConfig",
]
