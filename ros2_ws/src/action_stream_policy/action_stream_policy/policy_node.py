"""ROS 2 adapter for :class:`ReachLiftScriptedPolicy`."""

from __future__ import annotations

import json
import time
from typing import Any, Sequence

from .scripted_policy import (
    CHUNK_HORIZON,
    PolicyObservation,
    PolicyRequestData,
    ReachLiftScriptedPolicy,
)


def _set_common_event_fields(event, request, *, reason: str, detail: dict[str, object]) -> None:
    event.event_type = "policy_inference_completed"
    event.reason = reason
    event.strategy = "policy"
    event.episode_id = request.episode_id
    event.request_id = request.request_id
    event.generation_id = request.generation_id
    event.source_observation_step = request.source_observation_step
    event.action_count = CHUNK_HORIZON
    event.detail = json.dumps(detail, sort_keys=True, separators=(",", ":"))


def create_scripted_policy_node(*, parameter_overrides: Sequence[Any] | None = None) -> Any:
    """Create the ROS node without importing ROS until this factory is called.

    ``parameter_overrides`` is forwarded to :class:`rclpy.node.Node`; callers
    embedding the node should pass a sequence of ``rclpy.parameter.Parameter``
    instances after initializing their ROS context.
    """

    try:
        from action_stream_msgs.msg import (
            ActionChunk,
            InferenceRequest,
            RuntimeEvent,
            TargetAction,
        )
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
    except ImportError as exc:  # pragma: no cover - exercised in ROS images
        raise RuntimeError("scripted_policy_node requires ROS 2 and action_stream_msgs") from exc

    class ScriptedPolicyNode(Node):
        def __init__(self) -> None:
            super().__init__(
                "action_stream_scripted_policy",
                parameter_overrides=list(parameter_overrides or ()),
            )
            self.declare_parameter("request_topic", "/action_stream/inference_request")
            self.declare_parameter("raw_chunk_topic", "/action_stream/raw_action_chunk")
            self.declare_parameter("event_topic", "/action_stream/events")
            qos = QoSProfile(depth=64, reliability=ReliabilityPolicy.RELIABLE)
            self._policy = ReachLiftScriptedPolicy()
            self._publisher = self.create_publisher(
                ActionChunk,
                str(self.get_parameter("raw_chunk_topic").value),
                qos,
            )
            self._event_publisher = self.create_publisher(
                RuntimeEvent,
                str(self.get_parameter("event_topic").value),
                qos,
            )
            self._subscription = self.create_subscription(
                InferenceRequest,
                str(self.get_parameter("request_topic").value),
                self._on_request,
                qos,
            )

        def _on_request(self, message) -> None:
            started_steady_ns = time.monotonic_ns()
            try:
                request = PolicyRequestData(
                    episode_id=message.episode_id,
                    request_id=int(message.request_id),
                    generation_id=int(message.generation_id),
                    source_observation_step=int(message.source_observation_step),
                    expected_horizon=int(message.expected_horizon),
                    observation=PolicyObservation(
                        episode_id=message.observation.episode_id,
                        observation_step=int(message.observation.observation_step),
                        task_id=message.observation.task_id,
                        robot_state=tuple(message.observation.robot_state),
                        task_state=tuple(message.observation.task_state),
                        terminated=bool(message.observation.terminated),
                    ),
                )
                result = self._policy.predict(request)
            except (TypeError, ValueError) as exc:
                self.get_logger().error(f"Rejected inference request: {exc}")
                return

            completed_steady_ns = time.monotonic_ns()
            completed_wall_ns = time.time_ns()
            completed_sim_stamp = self.get_clock().now().to_msg()
            chunk = ActionChunk()
            chunk.sim_stamp = completed_sim_stamp
            chunk.steady_time_ns = completed_steady_ns
            chunk.wall_time_ns = completed_wall_ns
            chunk.episode_id = result.episode_id
            chunk.request_id = result.request_id
            chunk.generation_id = result.generation_id
            chunk.source_observation_step = result.source_observation_step
            chunk.source_sim_stamp = message.source_sim_stamp
            chunk.source_observation_steady_time_ns = message.source_observation_steady_time_ns
            chunk.action_dimension = result.action_dimension
            for item in result.actions:
                action = TargetAction()
                action.target_step = item.target_step
                action.command = list(item.command)
                chunk.actions.append(action)
            chunk.inference_complete_sim_stamp = completed_sim_stamp
            chunk.inference_complete_steady_time_ns = completed_steady_ns
            chunk.inference_complete_wall_time_ns = completed_wall_ns
            chunk.response_publish_sim_stamp = completed_sim_stamp
            chunk.response_publish_steady_time_ns = time.monotonic_ns()
            chunk.response_publish_wall_time_ns = time.time_ns()
            self._publisher.publish(chunk)

            event = RuntimeEvent()
            event.sim_stamp = completed_sim_stamp
            event.steady_time_ns = completed_steady_ns
            event.wall_time_ns = completed_wall_ns
            _set_common_event_fields(
                event,
                message,
                reason="scripted_chunk",
                detail={
                    "horizon": CHUNK_HORIZON,
                    "inference_duration_ns": completed_steady_ns - started_steady_ns,
                    "first_target_step": result.actions[0].target_step,
                    "last_target_step": result.actions[-1].target_step,
                },
            )
            self._event_publisher.publish(event)

    return ScriptedPolicyNode()


def main(args: list[str] | None = None) -> None:
    try:
        import rclpy
    except ImportError as exc:  # pragma: no cover - exercised in ROS images
        raise RuntimeError("scripted_policy_node requires ROS 2 and action_stream_msgs") from exc

    rclpy.init(args=args)
    node = create_scripted_policy_node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
