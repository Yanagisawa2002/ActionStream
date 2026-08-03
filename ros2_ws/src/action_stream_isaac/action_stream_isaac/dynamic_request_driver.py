"""Generation-aware M8-G0 inference-request lifecycle driver.

The scheduling core is pure Python and importable without ROS 2.  The ROS
factory is invoked only after Isaac's ROS bridge has initialized.  A
destination-switch generation advance always opens a fresh request on that
observation, even when the normal cadence would not.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import time
from typing import Any, Final, Sequence

from .dynamic_task import (
    ACTION_HORIZON,
    DISTURBED_GENERATION_ID,
    DYNAMIC_TASK_STATE_SIZE,
    INITIAL_GENERATION_ID,
    REQUEST_INTERVAL_STEPS,
    unpack_dynamic_task_state,
)
from .ros_contract import BOOTSTRAP_OBSERVATION_TOPIC


SUCCESS_INDEX: Final = 39


@dataclass(frozen=True, slots=True)
class DynamicRequestDriverConfig:
    episode_id: str
    strategy: str
    request_interval_steps: int = REQUEST_INTERVAL_STEPS
    expected_horizon: int = ACTION_HORIZON

    def validate(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if not self.strategy:
            raise ValueError("strategy must be non-empty")
        if not 0 < self.request_interval_steps < self.expected_horizon:
            raise ValueError(
                "request_interval_steps must lie in [1, expected_horizon-1]"
            )
        if self.expected_horizon != ACTION_HORIZON:
            raise ValueError(f"expected_horizon must remain frozen at {ACTION_HORIZON}")


@dataclass(frozen=True, slots=True)
class DynamicRequestDecision:
    request: bool
    scheduled: bool
    forced_generation_advance: bool
    retry_after_failure: bool
    previous_generation_id: int | None
    generation_id: int
    observation_step: int


def generation_from_task_state(task_state: Sequence[float]) -> int:
    """Return the strictly validated generation encoded at index 24."""

    return int(unpack_dynamic_task_state(task_state)["generation_id"])


def success_from_dynamic_task_state(task_state: Sequence[float]) -> bool:
    """Return the strictly validated success flag encoded at index 39."""

    values = tuple(task_state)
    if len(values) != DYNAMIC_TASK_STATE_SIZE:
        raise ValueError(
            f"dynamic task_state must contain exactly {DYNAMIC_TASK_STATE_SIZE} values"
        )
    return bool(unpack_dynamic_task_state(values)["success"])


def sync_resolution_for_runtime_event(
    event_type: str, reason: str
) -> bool | None:
    """Return retry intent for a C++-resolved sync request, else ``None``.

    A fault-injector ``response_dropped`` event does not cancel the registered
    request in the C++ state machine, so resolving it driver-side would create
    a hidden overlapping request.  Only a valid queue update or the C++
    ``fully_expired`` terminal response state releases the sync slot.
    """

    if event_type == "queue_updated":
        return False
    if event_type == "chunk_rejected" and reason == "fully_expired":
        return True
    return None


def is_bootstrap_request_accepted(
    event_type: str, request_id: int, generation_id: int
) -> bool:
    return (
        event_type == "request_registered"
        and request_id == 1
        and generation_id == INITIAL_GENERATION_ID
    )


def is_executor_lifecycle_ready(event_type: str, generation_id: int) -> bool:
    """Recognize the external C++ executor's start/reset acknowledgement."""

    return (
        event_type in {"episode_started", "episode_reset"}
        and generation_id == INITIAL_GENERATION_ID
    )


class DynamicRequestScheduler:
    """Stateful pure scheduler shared by tests and the ROS node."""

    def __init__(
        self,
        request_interval_steps: int = REQUEST_INTERVAL_STEPS,
        *,
        strategy: str = "aligned_async",
    ) -> None:
        if request_interval_steps <= 0:
            raise ValueError("request_interval_steps must be positive")
        if strategy not in {"sync_hold", "naive_async", "aligned_async"}:
            raise ValueError("unknown request scheduling strategy")
        self.request_interval_steps = request_interval_steps
        self.strategy = strategy
        self._last_generation_id: int | None = None
        self._last_observation_step: int | None = None
        self._seen: set[tuple[int, int]] = set()
        self._sync_inflight_request_id: int | None = None
        self._sync_inflight_generation_id: int | None = None
        self._sync_retry_needed = False

    @property
    def sync_request_in_flight(self) -> bool:
        return self.strategy == "sync_hold" and self._sync_inflight_request_id is not None

    def mark_request_published(self, *, request_id: int, generation_id: int) -> None:
        if request_id <= 0:
            raise ValueError("request_id must be positive")
        if self.strategy != "sync_hold":
            return
        if self._sync_inflight_request_id != 0:
            raise ValueError("sync request publication was not reserved by observe()")
        self._sync_inflight_request_id = request_id
        self._sync_inflight_generation_id = generation_id
        self._sync_retry_needed = False

    def resolve_request(self, *, request_id: int, retry: bool) -> bool:
        """Resolve only the current sync request; ignore superseded responses."""

        if type(retry) is not bool:
            raise ValueError("retry must be exactly bool")
        if self.strategy != "sync_hold":
            return False
        if request_id != self._sync_inflight_request_id:
            return False
        self._sync_inflight_request_id = None
        self._sync_inflight_generation_id = None
        self._sync_retry_needed = retry
        return True

    def observe(
        self, *, observation_step: int, generation_id: int, terminated: bool = False
    ) -> DynamicRequestDecision:
        if observation_step < 0:
            raise ValueError("observation_step must be non-negative")
        if generation_id not in {
            INITIAL_GENERATION_ID,
            DISTURBED_GENERATION_ID,
        }:
            raise ValueError("generation_id must be the frozen initial or disturbed ID")
        if type(terminated) is not bool:
            raise ValueError("terminated must be exactly bool")
        previous = self._last_generation_id
        if previous is not None and generation_id < previous:
            raise ValueError(
                f"generation regressed from {previous} to {generation_id}"
            )
        if (
            self._last_observation_step is not None
            and observation_step < self._last_observation_step
        ):
            raise ValueError(
                "observation step regressed from "
                f"{self._last_observation_step} to {observation_step}"
            )
        key = (observation_step, generation_id)
        duplicate = key in self._seen
        scheduled = (
            observation_step == 0
            or observation_step % self.request_interval_steps == 0
        )
        forced = previous is not None and generation_id > previous
        retry_after_failure = self.strategy == "sync_hold" and self._sync_retry_needed
        sync_blocked = (
            self.strategy == "sync_hold"
            and self._sync_inflight_request_id is not None
            and not forced
        )
        request = (
            not terminated
            and not duplicate
            and (scheduled or forced or retry_after_failure)
            and not sync_blocked
        )
        if request and self.strategy == "sync_hold":
            # Reserve synchronously so another observation cannot overlap in
            # the small interval before the ROS request ID is assigned.
            self._sync_inflight_request_id = 0
            self._sync_inflight_generation_id = generation_id
            self._sync_retry_needed = False
        self._seen.add(key)
        self._last_generation_id = generation_id
        self._last_observation_step = observation_step
        return DynamicRequestDecision(
            request=request,
            scheduled=scheduled,
            forced_generation_advance=forced and request,
            retry_after_failure=retry_after_failure and request,
            previous_generation_id=previous,
            generation_id=generation_id,
            observation_step=observation_step,
        )


def create_dynamic_request_driver_node(
    config: DynamicRequestDriverConfig,
) -> Any:
    """Create an in-process ROS node after the Isaac bridge is initialized."""

    config.validate()
    try:
        from action_stream_msgs.msg import (
            EpisodeControl,
            InferenceRequest,
            Observation,
            RuntimeEvent,
        )
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:  # pragma: no cover - exercised in Isaac/ROS
        raise RuntimeError(
            "dynamic request driver requires post-bridge rclpy and "
            "action_stream_msgs"
        ) from exc

    class DynamicRequestDriverNode(Node):
        def __init__(self) -> None:
            super().__init__("action_stream_dynamic_request_driver")
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
            self._event_publisher = self.create_publisher(
                RuntimeEvent, "/action_stream/events", reliable
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
            self._scheduler = DynamicRequestScheduler(
                config.request_interval_steps,
                strategy=config.strategy,
            )
            self._next_request_id = 1
            self._terminal_request_sent = False
            self._terminal_status_seen = False
            self._terminal_success = False
            self._terminal_step: int | None = None
            self._terminal_generation_id = INITIAL_GENERATION_ID
            self._executor_episode_ready = False
            self._bootstrap_ready = False

        @property
        def executor_episode_ready(self) -> bool:
            return self._executor_episode_ready

        @property
        def bootstrap_ready(self) -> bool:
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
            try:
                task_generation = generation_from_task_state(message.task_state)
                message_generation = int(message.generation_id)
                if message_generation != task_generation:
                    raise ValueError(
                        "Observation.generation_id does not match task_state generation"
                    )
                decision = self._scheduler.observe(
                    observation_step=int(message.observation_step),
                    generation_id=message_generation,
                    terminated=bool(message.terminated),
                )
            except (TypeError, ValueError) as exc:
                self.get_logger().error(f"Rejected dynamic observation: {exc}")
                return
            if bool(message.terminated):
                self._terminal_step = int(message.observation_step)
                self._terminal_generation_id = message_generation
                self._terminal_success = success_from_dynamic_task_state(
                    message.task_state
                )
                self._publish_terminate(message)
                return
            if decision.request:
                request_id = self._publish_request(message, message_generation)
                if decision.forced_generation_advance:
                    self._publish_forced_event(message, decision, request_id)

        def _publish_request(self, observation: Any, generation_id: int) -> int:
            request_id = self._next_request_id
            message = InferenceRequest()
            message.sim_stamp = copy.deepcopy(observation.sim_stamp)
            message.request_steady_time_ns = time.monotonic_ns()
            message.request_wall_time_ns = time.time_ns()
            message.episode_id = config.episode_id
            message.request_id = request_id
            message.generation_id = generation_id
            message.source_observation_step = int(observation.observation_step)
            message.source_sim_stamp = copy.deepcopy(observation.sim_stamp)
            message.source_observation_steady_time_ns = int(observation.steady_time_ns)
            message.expected_horizon = config.expected_horizon
            message.observation = copy.deepcopy(observation)
            self._request_publisher.publish(message)
            self._scheduler.mark_request_published(
                request_id=request_id,
                generation_id=generation_id,
            )
            self._next_request_id += 1
            return request_id

        def _publish_forced_event(
            self,
            observation: Any,
            decision: DynamicRequestDecision,
            request_id: int,
        ) -> None:
            event = RuntimeEvent()
            event.sim_stamp = copy.deepcopy(observation.sim_stamp)
            event.steady_time_ns = time.monotonic_ns()
            event.wall_time_ns = time.time_ns()
            event.event_type = "disturbance_request_forced"
            event.reason = "generation_advanced"
            event.strategy = config.strategy
            event.episode_id = config.episode_id
            event.request_id = request_id
            event.generation_id = decision.generation_id
            event.active_generation_id = decision.generation_id
            event.source_observation_step = decision.observation_step
            event.action_count = config.expected_horizon
            event.detail = json.dumps(
                {
                    "previous_generation_id": decision.previous_generation_id,
                    "generation_id": decision.generation_id,
                    "scheduled_boundary": decision.scheduled,
                    "retry_after_failure": decision.retry_after_failure,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            self._event_publisher.publish(event)

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
            message.generation_id = self._terminal_generation_id
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
            if str(message.episode_id) != config.episode_id:
                return
            event_type = str(message.event_type)
            request_id = int(message.request_id)
            generation_id = int(message.generation_id)
            if is_executor_lifecycle_ready(event_type, generation_id):
                self._executor_episode_ready = True
            retry = sync_resolution_for_runtime_event(
                event_type, str(message.reason)
            )
            if retry is not None:
                self._scheduler.resolve_request(request_id=request_id, retry=retry)
            if is_bootstrap_request_accepted(
                event_type, request_id, generation_id
            ):
                self._bootstrap_ready = True

    return DynamicRequestDriverNode()
