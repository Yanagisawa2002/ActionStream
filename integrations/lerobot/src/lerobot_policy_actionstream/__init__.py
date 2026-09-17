"""Register ActionStream as a third-party ``lerobot-rollout`` backend."""

from __future__ import annotations

from dataclasses import dataclass

from actionstream.lerobot_inference import (
    ActionStreamInferenceConfig,
    ActionStreamInferenceEngine,
)
from actionstream.rpc_transport import TcpInferenceTransport
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
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 50051
    tcp_connect_timeout_s: float = 3.0
    tcp_control_timeout_s: float = 3.0
    delivery_scheduler_enabled: bool = False
    minimum_request_interval_steps: int = 1
    telemetry_jsonl_path: str | None = None

    def __post_init__(self) -> None:
        if self.transport_mode not in {"direct", "process", "tcp"}:
            raise ValueError("transport_mode must be 'direct', 'process', or 'tcp'")
        if self.transport_mode == "tcp":
            if self.process_transport_factory is not None:
                raise ValueError("tcp transport cannot use process_transport_factory")
            if not self.tcp_host:
                raise ValueError("tcp_host is required for tcp transport")
            if type(self.tcp_port) is not int or not 1 <= self.tcp_port <= 65535:
                raise ValueError("tcp_port must be in [1, 65535]")

    def runtime_config(self) -> ActionStreamInferenceConfig:
        # The core engine already accepts an injected InferenceTransport. TCP is
        # constructed by this plugin and therefore uses the direct core config
        # only as the non-process lifecycle default.
        core_transport_mode = (
            "direct" if self.transport_mode == "tcp" else self.transport_mode
        )
        return ActionStreamInferenceConfig(
            inference_timeout_s=self.inference_timeout_s,
            bounded_hold_steps=self.bounded_hold_steps,
            retry_backoff_s=self.retry_backoff_s,
            max_consecutive_failures=self.max_consecutive_failures,
            join_timeout_s=self.join_timeout_s,
            latest_only_fallback=self.latest_only_fallback,
            transport_mode=core_transport_mode,
            process_transport_factory=(
                self.process_transport_factory
                if core_transport_mode == "process"
                else None
            ),
            process_transport_start_method=self.process_transport_start_method,
            process_transport_startup_timeout_s=self.process_transport_startup_timeout_s,
            process_transport_terminate_timeout_s=self.process_transport_terminate_timeout_s,
            delivery_scheduler_enabled=self.delivery_scheduler_enabled,
            minimum_request_interval_steps=self.minimum_request_interval_steps,
            telemetry_jsonl_path=self.telemetry_jsonl_path,
        )

    def build_transport(self):
        if self.transport_mode != "tcp":
            return None
        return TcpInferenceTransport(
            self.tcp_host,
            self.tcp_port,
            connect_timeout_s=self.tcp_connect_timeout_s,
            control_timeout_s=self.tcp_control_timeout_s,
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
        transport=config.build_transport(),
    )


__all__ = [
    "ActionStreamInferenceEngine",
    "ActionStreamRolloutInferenceConfig",
]
