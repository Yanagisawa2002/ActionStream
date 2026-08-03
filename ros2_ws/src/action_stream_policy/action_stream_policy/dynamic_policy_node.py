"""ROS 2 adapter for the deterministic M8 dynamic pick/place policy."""

from __future__ import annotations

import json
import time
from typing import Any, Sequence

from .dynamic_pick_place import (
    CHUNK_HORIZON,
    PHASE_CODES,
    DynamicPickPlacePolicy,
    DynamicPolicyConfig,
    DynamicPolicyObservation,
    DynamicPolicyRequestData,
)


def _set_dynamic_event_fields(
    event: Any,
    request: Any,
    *,
    detail: dict[str, object],
) -> None:
    event.event_type = "policy_inference_completed"
    event.reason = "dynamic_pick_place_chunk"
    event.strategy = "policy"
    event.episode_id = request.episode_id
    event.request_id = request.request_id
    event.generation_id = request.generation_id
    event.active_generation_id = request.generation_id
    event.source_observation_step = request.source_observation_step
    event.action_count = CHUNK_HORIZON
    event.detail = json.dumps(detail, sort_keys=True, separators=(",", ":"))


def create_dynamic_policy_node(
    *, parameter_overrides: Sequence[Any] | None = None
) -> Any:
    """Create the ROS node without importing ROS until this factory is called."""

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
        raise RuntimeError(
            "dynamic_policy_node requires ROS 2 and action_stream_msgs"
        ) from exc

    class DynamicPolicyNode(Node):
        def __init__(self) -> None:
            super().__init__(
                "action_stream_dynamic_policy",
                parameter_overrides=list(parameter_overrides or ()),
            )
            self.declare_parameter("request_topic", "/action_stream/inference_request")
            self.declare_parameter("raw_chunk_topic", "/action_stream/raw_action_chunk")
            self.declare_parameter("event_topic", "/action_stream/events")
            defaults = DynamicPolicyConfig()
            self.declare_parameter("approach_height_m", defaults.approach_height_m)
            self.declare_parameter("grasp_hand_offset_m", defaults.grasp_hand_offset_m)
            self.declare_parameter("carry_hand_height_m", defaults.carry_hand_height_m)
            self.declare_parameter(
                "recovery_hover_offset_m", defaults.recovery_hover_offset_m
            )
            self.declare_parameter(
                "maximum_translation_per_step_m",
                defaults.maximum_translation_per_step_m,
            )
            self.declare_parameter("workspace_x_m", list(defaults.workspace_x_m))
            self.declare_parameter("workspace_y_m", list(defaults.workspace_y_m))
            self.declare_parameter("workspace_z_m", list(defaults.workspace_z_m))
            self.declare_parameter(
                "downward_axis_angle_xyz", list(defaults.downward_axis_angle_xyz)
            )
            policy_config = DynamicPolicyConfig(
                approach_height_m=float(self.get_parameter("approach_height_m").value),
                grasp_hand_offset_m=float(
                    self.get_parameter("grasp_hand_offset_m").value
                ),
                carry_hand_height_m=float(
                    self.get_parameter("carry_hand_height_m").value
                ),
                recovery_hover_offset_m=float(
                    self.get_parameter("recovery_hover_offset_m").value
                ),
                maximum_translation_per_step_m=float(
                    self.get_parameter("maximum_translation_per_step_m").value
                ),
                workspace_x_m=tuple(self.get_parameter("workspace_x_m").value),
                workspace_y_m=tuple(self.get_parameter("workspace_y_m").value),
                workspace_z_m=tuple(self.get_parameter("workspace_z_m").value),
                downward_axis_angle_xyz=tuple(
                    self.get_parameter("downward_axis_angle_xyz").value
                ),
            )
            qos = QoSProfile(depth=64, reliability=ReliabilityPolicy.RELIABLE)
            self._policy = DynamicPickPlacePolicy(policy_config)
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

        def _on_request(self, message: Any) -> None:
            started_steady_ns = time.monotonic_ns()
            try:
                request = DynamicPolicyRequestData(
                    episode_id=message.episode_id,
                    request_id=int(message.request_id),
                    generation_id=int(message.generation_id),
                    source_observation_step=int(message.source_observation_step),
                    expected_horizon=int(message.expected_horizon),
                    observation=DynamicPolicyObservation(
                        episode_id=message.observation.episode_id,
                        observation_step=int(message.observation.observation_step),
                        generation_id=int(message.observation.generation_id),
                        task_id=message.observation.task_id,
                        robot_state=tuple(message.observation.robot_state),
                        task_state=tuple(message.observation.task_state),
                        terminated=bool(message.observation.terminated),
                    ),
                )
                result = self._policy.predict(request)
            except (TypeError, ValueError) as exc:
                self.get_logger().error(f"Rejected dynamic inference request: {exc}")
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
            chunk.source_observation_steady_time_ns = (
                message.source_observation_steady_time_ns
            )
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

            state = request.observation.task_state
            phase_code = int(state[25])
            if phase_code not in PHASE_CODES:  # defensive; dataclass already validates
                self.get_logger().error(f"Unexpected phase after inference: {phase_code}")
                return
            event = RuntimeEvent()
            event.sim_stamp = completed_sim_stamp
            event.steady_time_ns = completed_steady_ns
            event.wall_time_ns = completed_wall_ns
            _set_dynamic_event_fields(
                event,
                message,
                detail={
                    "phase_code": phase_code,
                    "disturbance_switched": state[23] == 1.0,
                    "grasped": state[27] == 1.0,
                    "active_destination_xyz": list(state[19:22]),
                    "horizon": CHUNK_HORIZON,
                    "inference_duration_ns": completed_steady_ns
                    - started_steady_ns,
                    "first_target_step": result.actions[0].target_step,
                    "last_target_step": result.actions[-1].target_step,
                    "first_target_xyz": list(result.actions[0].command[0:3]),
                    "last_target_xyz": list(result.actions[-1].command[0:3]),
                },
            )
            self._event_publisher.publish(event)

    return DynamicPolicyNode()


def main(args: list[str] | None = None) -> None:
    try:
        import rclpy
    except ImportError as exc:  # pragma: no cover - exercised in ROS images
        raise RuntimeError(
            "dynamic_policy_node requires ROS 2 and action_stream_msgs"
        ) from exc

    rclpy.init(args=args)
    node = create_dynamic_policy_node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
