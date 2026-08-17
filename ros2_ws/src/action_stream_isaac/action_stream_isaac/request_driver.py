"""Production inference-request lifecycle driver for the Isaac adapter.

The module stays importable without ROS 2.  ``create_request_driver_node`` is
called only after ``SimulationApp`` has enabled and updated the Isaac ROS
bridge, which is required by the supported Windows pip environment.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Any, Sequence

from .ros_contract import BOOTSTRAP_OBSERVATION_TOPIC


@dataclass(frozen=True)
class RequestDriverConfig:
    episode_id: str
    generation_id: int
    request_interval_steps: int = 10
    expected_horizon: int = 30

    def validate(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if self.generation_id < 0:
            raise ValueError("generation_id must be non-negative")
        if not 0 < self.request_interval_steps < self.expected_horizon:
            raise ValueError("request_interval_steps must lie in [1, expected_horizon-1]")
        if self.expected_horizon <= 0:
            raise ValueError("expected_horizon must be positive")


def should_request_observation(observation_step: int, request_interval_steps: int) -> bool:
    """Return whether a nonterminal observation opens a policy request."""

    if observation_step < 0:
        raise ValueError("observation_step must be non-negative")
    if request_interval_steps <= 0:
        raise ValueError("request_interval_steps must be positive")
    return observation_step == 0 or observation_step % request_interval_steps == 0


def success_from_task_state(task_state: Sequence[float]) -> bool:
    """Read the frozen success flag at task_state index 9."""

    values = tuple(float(value) for value in task_state)
    if len(values) != 12 or not all(math.isfinite(value) for value in values):
        raise ValueError("task_state must contain exactly 12 finite values")
    return values[9] >= 0.5


def create_request_driver_node(config: RequestDriverConfig) -> Any:
    """Create an in-process ROS node after the Isaac bridge is initialized."""

    config.validate()
    try:
        from action_stream_msgs.msg import (
            EpisodeControl,
            InferenceRequest,
            Observation,
            RuntimeEvent,
        )
        from action_stream_policy.scripted_policy import CHUNK_HORIZON
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:  # pragma: no cover - exercised only in Isaac/ROS
        raise RuntimeError(
            "Isaac request driver requires post-bridge rclpy, action_stream_msgs, "
            "and action_stream_policy"
        ) from exc

    if config.expected_horizon != CHUNK_HORIZON:
        raise RuntimeError(
            "request driver horizon drift: "
            f"configured={config.expected_horizon}, policy={CHUNK_HORIZON}"
        )

    class IsaacRequestDriverNode(Node):
        def __init__(self) -> None:
            super().__init__("action_stream_isaac_request_driver")
            reliable = QoSProfile(depth=256, reliability=ReliabilityPolicy.RELIABLE)
            lifecycle = QoSProfile(
                depth=16,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._request_publisher = self.create_publisher(
                InferenceRequest, "/action_stream/inference_request", reliable
            )
            self._control_publisher = self.create_publisher(
                EpisodeControl, "/action_stream/episode_control", lifecycle
            )
            self._observation_subscription = self.create_subscription(
                Observation,
                "/action_stream/observation",
                self._on_observation,
                reliable,
            )
            self._bootstrap_observation_subscription = self.create_subscription(
                Observation,
                BOOTSTRAP_OBSERVATION_TOPIC,
                self._on_observation,
                reliable,
            )
            self._runtime_event_subscription = self.create_subscription(
                RuntimeEvent,
                "/action_stream/events",
                self._on_runtime_event,
                reliable,
            )
            self._control_subscription = self.create_subscription(
                EpisodeControl,
                "/action_stream/episode_control",
                self._on_control,
                lifecycle,
            )
            self._next_request_id = 1
            self._seen_observation_steps: set[int] = set()
            self._terminal_request_sent = False
            self._terminal_status_seen = False
            self._terminal_success = False
            self._terminal_step: int | None = None
            self._bootstrap_ready = False

        @property
        def bootstrap_ready(self) -> bool:
            """Return whether the executor accepted the first delayed chunk."""

            return self._bootstrap_ready

        @property
        def complete(self) -> bool:
            return self._terminal_request_sent and self._terminal_status_seen

        @property
        def terminal_success(self) -> bool:
            return self._terminal_success

        @property
        def terminal_step(self) -> int | None:
            return self._terminal_step

        @property
        def request_count(self) -> int:
            return self._next_request_id - 1

        def _on_observation(self, message: Any) -> None:
            if str(message.episode_id) != config.episode_id:
                return
            step = int(message.observation_step)
            if step in self._seen_observation_steps:
                return
            self._seen_observation_steps.add(step)
            if bool(message.terminated):
                self._terminal_step = step
                self._terminal_success = success_from_task_state(message.task_state)
                self._publish_terminate(message)
                return
            if should_request_observation(step, config.request_interval_steps):
                self._publish_request(message)

        def _publish_request(self, observation: Any) -> None:
            message = InferenceRequest()
            message.sim_stamp = copy.deepcopy(observation.sim_stamp)
            message.request_steady_time_ns = time.monotonic_ns()
            message.request_wall_time_ns = time.time_ns()
            message.episode_id = config.episode_id
            message.request_id = self._next_request_id
            message.generation_id = config.generation_id
            message.source_observation_step = int(observation.observation_step)
            message.source_sim_stamp = copy.deepcopy(observation.sim_stamp)
            message.source_observation_steady_time_ns = int(observation.steady_time_ns)
            message.expected_horizon = config.expected_horizon
            message.observation = copy.deepcopy(observation)
            self._request_publisher.publish(message)
            self._next_request_id += 1

        def _publish_terminate(self, observation: Any) -> None:
            if self._terminal_request_sent:
                return
            message = EpisodeControl()
            message.message_kind = EpisodeControl.REQUEST
            message.command = EpisodeControl.TERMINATE
            message.sim_stamp = copy.deepcopy(observation.sim_stamp)
            message.steady_time_ns = time.monotonic_ns()
            message.wall_time_ns = time.time_ns()
            message.episode_id = config.episode_id
            message.generation_id = config.generation_id
            message.active = False
            message.terminated = False
            message.success = self._terminal_success
            self._terminal_request_sent = True
            self._control_publisher.publish(message)

        def _on_control(self, message: Any) -> None:
            if (
                self._terminal_request_sent
                and int(message.message_kind) == EpisodeControl.STATUS
                and int(message.command) == EpisodeControl.TERMINATE
                and str(message.episode_id) == config.episode_id
                and bool(message.terminated)
            ):
                self._terminal_status_seen = True

        def _on_runtime_event(self, message: Any) -> None:
            if (
                str(message.episode_id) == config.episode_id
                and str(message.event_type) == "queue_updated"
                and int(message.request_id) == 1
                and int(message.generation_id) == config.generation_id
            ):
                self._bootstrap_ready = True

    return IsaacRequestDriverNode()
