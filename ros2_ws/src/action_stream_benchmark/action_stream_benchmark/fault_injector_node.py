"""ROS 2 deterministic pre-generated latency/fault injector."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import heapq
import json
import time
from typing import Any, Sequence

from .faults import FaultTrace, load_fault_trace
from .schema import read_json


def load_runtime_fault_trace(path: str) -> Any:
    """Dispatch M8 traces without changing legacy M7 deserialization."""

    payload = read_json(path)
    if payload.get("milestone") == "M8-G0":
        from .m8_faults import load_fault_trace as load_m8_fault_trace

        return load_m8_fault_trace(path)
    return load_fault_trace(path)


def duplicate_delivery_offset_ms(entry: Any) -> int:
    """M7's accepted implicit offset remains exactly 50 ms."""

    value = getattr(entry, "duplicate_delivery_offset_ms", 50)
    if type(value) is not int or value < 0:
        raise ValueError("duplicate delivery offset must be a non-negative integer")
    return value


@dataclass(order=True, slots=True)
class _ScheduledChunk:
    due_steady_ns: int
    serial: int
    message: Any = field(compare=False)
    request_ordinal: int = field(compare=False)
    duplicate: bool = field(compare=False)
    latency_ms: int = field(compare=False)


def create_fault_injector_node(*, parameter_overrides: Sequence[Any] | None = None) -> Any:
    """Create the ROS node with imports deferred until factory invocation.

    ``parameter_overrides`` is forwarded to :class:`rclpy.node.Node`; callers
    embedding the node should pass a sequence of ``rclpy.parameter.Parameter``
    instances after initializing their ROS context.
    """

    try:
        from action_stream_msgs.msg import ActionChunk, RuntimeEvent
        from rclpy.clock import Clock, ClockType
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
    except ImportError as exc:  # pragma: no cover - requires ROS image
        raise RuntimeError("fault_injector_node requires ROS 2 and action_stream_msgs") from exc

    class FaultInjectorNode(Node):
        def __init__(self) -> None:
            super().__init__(
                "action_stream_fault_injector",
                parameter_overrides=list(parameter_overrides or ()),
            )
            self.declare_parameter("trace_file", "")
            self.declare_parameter("raw_chunk_topic", "/action_stream/raw_action_chunk")
            self.declare_parameter("action_chunk_topic", "/action_stream/action_chunk")
            self.declare_parameter("event_topic", "/action_stream/events")
            trace_file = str(self.get_parameter("trace_file").value)
            if not trace_file:
                raise ValueError("trace_file is required; online random faults are forbidden")
            self._trace: FaultTrace = load_runtime_fault_trace(trace_file)
            self._scheduled: list[_ScheduledChunk] = []
            self._serial = 0
            qos = QoSProfile(depth=256, reliability=ReliabilityPolicy.RELIABLE)
            self._publisher = self.create_publisher(
                ActionChunk, str(self.get_parameter("action_chunk_topic").value), qos
            )
            self._event_publisher = self.create_publisher(
                RuntimeEvent, str(self.get_parameter("event_topic").value), qos
            )
            self._subscription = self.create_subscription(
                ActionChunk,
                str(self.get_parameter("raw_chunk_topic").value),
                self._on_chunk,
                qos,
            )
            self._timer = self.create_timer(
                0.002,
                self._deliver_due,
                clock=Clock(clock_type=ClockType.STEADY_TIME),
            )
            self.get_logger().info(
                f"Loaded frozen trace {self._trace.sha256} "
                f"({self._trace.profile.profile_id}, seed={self._trace.seed})"
            )

        def _event(self, message, event_type: str, reason: str, detail: dict[str, Any]) -> None:
            event = RuntimeEvent()
            event.sim_stamp = self.get_clock().now().to_msg()
            event.steady_time_ns = time.monotonic_ns()
            event.wall_time_ns = time.time_ns()
            event.event_type = event_type
            event.reason = reason
            event.strategy = "fault_injector"
            event.episode_id = message.episode_id
            event.request_id = message.request_id
            event.generation_id = message.generation_id
            event.source_observation_step = message.source_observation_step
            event.action_count = len(message.actions)
            event.detail = json.dumps(detail, sort_keys=True, separators=(",", ":"))
            self._event_publisher.publish(event)

        def _schedule(self, message, ordinal: int, latency_ms: int, duplicate: bool) -> None:
            self._serial += 1
            heapq.heappush(
                self._scheduled,
                _ScheduledChunk(
                    due_steady_ns=time.monotonic_ns() + latency_ms * 1_000_000,
                    serial=self._serial,
                    message=copy.deepcopy(message),
                    request_ordinal=ordinal,
                    duplicate=duplicate,
                    latency_ms=latency_ms,
                ),
            )

        def _on_chunk(self, message) -> None:
            if message.request_id == 0:
                self.get_logger().error("request_id=0 cannot index a frozen fault trace")
                return
            ordinal = int(message.request_id) - 1
            try:
                fault = self._trace.entry(ordinal)
            except IndexError as exc:
                self.get_logger().error(str(exc))
                return
            detail = {
                "request_ordinal": ordinal,
                "trace_sha256": self._trace.sha256,
                "latency_ms": fault.total_delivery_delay_ms,
                "jitter_ms": fault.jitter_ms,
                "extra_delay_ms": fault.extra_delay_ms,
                "communication_pause_ms": fault.communication_pause_ms,
            }
            if fault.dropped:
                self._event(message, "response_dropped", "frozen_trace_drop", detail)
                return
            self._schedule(message, ordinal, fault.total_delivery_delay_ms, False)
            self._event(message, "chunk_scheduled", "frozen_trace_delay", detail)
            if fault.duplicate_count:
                duplicate_offset = duplicate_delivery_offset_ms(fault)
                duplicate_latency = fault.total_delivery_delay_ms + duplicate_offset
                self._schedule(message, ordinal, duplicate_latency, True)

        def _deliver_due(self) -> None:
            now = time.monotonic_ns()
            while self._scheduled and self._scheduled[0].due_steady_ns <= now:
                item = heapq.heappop(self._scheduled)
                message = item.message
                stamp = self.get_clock().now().to_msg()
                message.sim_stamp = stamp
                message.steady_time_ns = now
                message.wall_time_ns = time.time_ns()
                message.response_publish_sim_stamp = stamp
                message.response_publish_steady_time_ns = now
                message.response_publish_wall_time_ns = time.time_ns()
                self._publisher.publish(message)
                self._event(
                    message,
                    "chunk_delivered",
                    "duplicate" if item.duplicate else "original",
                    {
                        "request_ordinal": item.request_ordinal,
                        "trace_sha256": self._trace.sha256,
                        "latency_ms": item.latency_ms,
                        "duplicate": item.duplicate,
                    },
                )

    return FaultInjectorNode()


def main(args: list[str] | None = None) -> None:
    try:
        import rclpy
    except ImportError as exc:  # pragma: no cover - requires ROS image
        raise RuntimeError("fault_injector_node requires ROS 2 and action_stream_msgs") from exc

    rclpy.init(args=args)
    node = create_fault_injector_node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
