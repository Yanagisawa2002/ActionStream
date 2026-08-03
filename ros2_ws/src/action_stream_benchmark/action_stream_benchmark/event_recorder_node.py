"""Structured atomic JSONL recorder for a ROS/C++ benchmark episode."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Sequence

from .ros_utils import combine_nanoseconds
from .schema import AtomicJsonlLog, MILESTONE, SCHEMA_VERSION


def _detail_counts(detail: str) -> tuple[int, int]:
    expired = 0
    duplicates = 0
    for item in detail.split(","):
        key, separator, value = item.partition("=")
        if not separator or not value.isdigit():
            continue
        if key == "expired":
            expired = int(value)
        elif key == "duplicates":
            duplicates = int(value)
    return expired, duplicates


def create_event_recorder_node(*, parameter_overrides: Sequence[Any] | None = None) -> Any:
    """Create the ROS node with imports deferred until factory invocation.

    ``parameter_overrides`` is forwarded to :class:`rclpy.node.Node`; callers
    embedding the node should pass a sequence of ``rclpy.parameter.Parameter``
    instances after initializing their ROS context.  The caller must invoke
    ``close()`` before destroying an embedded recorder so its atomic log is
    committed.
    """

    try:
        from action_stream_msgs.msg import (
            ActionChunk,
            EpisodeControl,
            ExecutorDiagnostics,
            InferenceRequest,
            Observation,
            RobotCommand,
            RuntimeEvent,
        )
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:  # pragma: no cover - requires ROS image
        raise RuntimeError("event_recorder_node requires ROS 2 and action_stream_msgs") from exc

    class EventRecorderNode(Node):
        def __init__(self) -> None:
            super().__init__(
                "action_stream_event_recorder",
                parameter_overrides=list(parameter_overrides or ()),
            )
            self.declare_parameter("output_path", "")
            self.declare_parameter("strategy", "")
            self.declare_parameter("profile_id", "")
            self.declare_parameter("seed", 0)
            self.declare_parameter("trace_sha256", "")
            self.declare_parameter("evidence_class", "ros_cpp_test_plant")
            self.declare_parameter("observation_topic", "/action_stream/observation")
            self.declare_parameter("request_topic", "/action_stream/inference_request")
            self.declare_parameter("action_chunk_topic", "/action_stream/action_chunk")
            self.declare_parameter("command_topic", "/action_stream/robot_command")
            self.declare_parameter("diagnostics_topic", "/action_stream/diagnostics")
            self.declare_parameter("event_topic", "/action_stream/events")
            self.declare_parameter("episode_control_topic", "/action_stream/episode_control")
            output_path = str(self.get_parameter("output_path").value)
            if not output_path:
                raise ValueError("output_path is required")
            self._strategy = str(self.get_parameter("strategy").value)
            self._profile_id = str(self.get_parameter("profile_id").value)
            self._seed = int(self.get_parameter("seed").value)
            self._trace_sha256 = str(self.get_parameter("trace_sha256").value)
            self._evidence_class = str(self.get_parameter("evidence_class").value)
            self._log = AtomicJsonlLog(Path(output_path), overwrite=True)
            self._log.__enter__()
            self._closed = False
            self._index = 0
            self._lock = threading.Lock()
            self._episode_id = ""
            self._request_starts: dict[int, int] = {}
            self._seen_responses: set[int] = set()
            self._last_response_request_id = 0

            qos = QoSProfile(depth=512, reliability=ReliabilityPolicy.RELIABLE)
            lifecycle_qos = QoSProfile(
                depth=16,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(
                Observation,
                str(self.get_parameter("observation_topic").value),
                self._on_observation,
                qos,
            )
            self.create_subscription(
                InferenceRequest,
                str(self.get_parameter("request_topic").value),
                self._on_request,
                qos,
            )
            self.create_subscription(
                ActionChunk,
                str(self.get_parameter("action_chunk_topic").value),
                self._on_chunk,
                qos,
            )
            self.create_subscription(
                RobotCommand,
                str(self.get_parameter("command_topic").value),
                self._on_command,
                qos,
            )
            self.create_subscription(
                ExecutorDiagnostics,
                str(self.get_parameter("diagnostics_topic").value),
                self._on_diagnostics,
                qos,
            )
            self.create_subscription(
                RuntimeEvent,
                str(self.get_parameter("event_topic").value),
                self._on_runtime_event,
                qos,
            )
            self.create_subscription(
                EpisodeControl,
                str(self.get_parameter("episode_control_topic").value),
                self._on_episode_control,
                lifecycle_qos,
            )

        def _append(
            self,
            event_type: str,
            *,
            sim_time_ns: int,
            steady_time_ns: int,
            wall_time_ns: int,
            episode_id: str = "",
            **payload: Any,
        ) -> None:
            with self._lock:
                if self._closed:
                    return
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "milestone": MILESTONE,
                    "event_index": self._index,
                    "event_type": event_type,
                    "profile_id": self._profile_id,
                    "seed": self._seed,
                    "strategy": self._strategy,
                    "episode_id": episode_id or self._episode_id,
                    "sim_time_ns": sim_time_ns,
                    "wall_time_ns": wall_time_ns,
                    "steady_time_ns": steady_time_ns,
                    **payload,
                }
                self._log.append(row)
                self._index += 1

        @staticmethod
        def _sim(message) -> int:
            return combine_nanoseconds(int(message.sim_stamp.sec), int(message.sim_stamp.nanosec))

        def _on_episode_control(self, message) -> None:
            if int(message.message_kind) != EpisodeControl.REQUEST:
                return
            if int(message.command) == EpisodeControl.START:
                self._episode_id = message.episode_id
                self._append(
                    "episode_start",
                    sim_time_ns=self._sim(message),
                    steady_time_ns=int(message.steady_time_ns),
                    wall_time_ns=int(message.wall_time_ns),
                    episode_id=message.episode_id,
                    generation_id=int(message.generation_id),
                    trace_sha256=self._trace_sha256,
                    evidence_class=self._evidence_class,
                )
            elif int(message.command) == EpisodeControl.TERMINATE:
                self._append(
                    "episode_end",
                    sim_time_ns=self._sim(message),
                    steady_time_ns=int(message.steady_time_ns),
                    wall_time_ns=int(message.wall_time_ns),
                    episode_id=message.episode_id,
                    success=bool(message.success),
                    completion_reason=("lift_success" if message.success else "step_limit"),
                )

        def _on_observation(self, message) -> None:
            self._append(
                "observation",
                sim_time_ns=self._sim(message),
                steady_time_ns=int(message.steady_time_ns),
                wall_time_ns=int(message.wall_time_ns),
                episode_id=message.episode_id,
                sim_step=int(message.observation_step),
                observation_step=int(message.observation_step),
                task_id=message.task_id,
                robot_state=list(message.robot_state),
                task_state=list(message.task_state),
                terminated=bool(message.terminated),
                success=(len(message.task_state) > 9 and message.task_state[9] >= 0.5),
            )

        def _on_request(self, message) -> None:
            request_id = int(message.request_id)
            self._request_starts[request_id] = int(message.request_steady_time_ns)
            self._append(
                "inference_request",
                sim_time_ns=self._sim(message),
                steady_time_ns=int(message.request_steady_time_ns),
                wall_time_ns=int(message.request_wall_time_ns),
                episode_id=message.episode_id,
                sim_step=int(message.source_observation_step),
                request_id=request_id,
                request_ordinal=request_id - 1,
                generation_id=int(message.generation_id),
                source_observation_step=int(message.source_observation_step),
                expected_horizon=int(message.expected_horizon),
                trace_sha256=self._trace_sha256,
            )

        def _on_chunk(self, message) -> None:
            request_id = int(message.request_id)
            duplicate = request_id in self._seen_responses
            out_of_order = request_id < self._last_response_request_id
            self._seen_responses.add(request_id)
            self._last_response_request_id = max(self._last_response_request_id, request_id)
            start = self._request_starts.get(request_id, int(message.steady_time_ns))
            latency_ms = max(0.0, (int(message.steady_time_ns) - start) / 1e6)
            self._append(
                "chunk_arrived",
                sim_time_ns=self._sim(message),
                steady_time_ns=int(message.steady_time_ns),
                wall_time_ns=int(message.wall_time_ns),
                episode_id=message.episode_id,
                sim_step=int(message.source_observation_step),
                request_id=request_id,
                request_ordinal=request_id - 1,
                generation_id=int(message.generation_id),
                source_observation_step=int(message.source_observation_step),
                latency_ms=latency_ms,
                duplicate=duplicate,
                out_of_order=out_of_order,
                actions=[
                    {
                        "target_step": int(action.target_step),
                        "command": list(action.command),
                    }
                    for action in message.actions
                ],
            )
            if self._strategy == "sync_hold" and not duplicate:
                self._append(
                    "sync_wait",
                    sim_time_ns=self._sim(message),
                    steady_time_ns=int(message.steady_time_ns),
                    wall_time_ns=int(message.wall_time_ns),
                    episode_id=message.episode_id,
                    duration_ns=max(0, int(message.steady_time_ns) - start),
                    counts_as_hold=True,
                    reason="inference_pending",
                )

        def _on_command(self, message) -> None:
            self._append(
                "command_executed",
                sim_time_ns=self._sim(message),
                steady_time_ns=int(message.steady_time_ns),
                wall_time_ns=int(message.wall_time_ns),
                episode_id=message.episode_id,
                sim_step=int(message.actual_target_step),
                actual_target_step=int(message.actual_target_step),
                source_request_id=int(message.source_request_id),
                source_generation_id=int(message.source_generation_id),
                source_observation_step=int(message.source_observation_step),
                source_target_step=int(message.source_target_step),
                command=list(message.command),
                hold=bool(message.hold),
                reason=message.reason,
            )

        def _on_diagnostics(self, message) -> None:
            self._append(
                "diagnostics",
                sim_time_ns=self._sim(message),
                steady_time_ns=int(message.steady_time_ns),
                wall_time_ns=int(message.wall_time_ns),
                episode_id=message.episode_id,
                sim_step=int(message.latest_observation_step),
                queue_length=int(message.queue_length),
                active_generation_id=int(message.active_generation_id),
                latest_executed_target_step=int(message.latest_executed_target_step),
                accepted_chunks=int(message.accepted_chunks),
                rejected_chunks=int(message.rejected_chunks),
                expired_actions_removed=int(message.expired_actions_removed),
                duplicate_actions_removed=int(message.duplicate_actions_removed),
                queue_rebuilds=int(message.queue_rebuilds),
                deadline_misses=int(message.deadline_misses),
                total_hold_duration_ns=int(message.total_hold_duration_ns),
            )

        def _on_runtime_event(self, message) -> None:
            event_type = message.event_type
            payload: dict[str, Any] = {
                "reason": message.reason,
                "request_id": int(message.request_id),
                "generation_id": int(message.generation_id),
                "source_observation_step": int(message.source_observation_step),
                "source_target_step": int(message.source_target_step),
                "actual_target_step": int(message.actual_target_step),
                "active_generation_id": int(message.active_generation_id),
                "queue_length_before": int(message.queue_length_before),
                "queue_length_after": int(message.queue_length_after),
                "action_count": int(message.action_count),
                "detail": message.detail,
            }
            if event_type == "queue_updated":
                expired, duplicates = _detail_counts(message.detail)
                payload["expired_actions_removed"] = expired
                payload["duplicate_target_actions_removed"] = duplicates
            if event_type in {"chunk_scheduled", "chunk_delivered", "response_dropped"}:
                try:
                    detail = json.loads(message.detail)
                except (TypeError, ValueError):
                    detail = {}
                payload.update(detail)
            self._append(
                event_type,
                sim_time_ns=self._sim(message),
                steady_time_ns=int(message.steady_time_ns),
                wall_time_ns=int(message.wall_time_ns),
                episode_id=message.episode_id,
                sim_step=int(message.actual_target_step or message.source_observation_step),
                **payload,
            )

        def close(self) -> None:
            with self._lock:
                if not self._closed:
                    self._log.commit()
                    self._closed = True

    return EventRecorderNode()


def main(args: list[str] | None = None) -> None:
    try:
        import rclpy
    except ImportError as exc:  # pragma: no cover - requires ROS image
        raise RuntimeError("event_recorder_node requires ROS 2 and action_stream_msgs") from exc

    rclpy.init(args=args)
    node = create_event_recorder_node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
