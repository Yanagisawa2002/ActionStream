"""ROS 2 deterministic reach-and-lift plant and episode request driver."""

from __future__ import annotations

import copy
import time

from action_stream_policy.scripted_policy import CHUNK_HORIZON, TASK_ID

from .plant import CONTROL_FREQUENCY_HZ, ReachLiftPlant
from .ros_utils import assign_time


def main(args: list[str] | None = None) -> None:
    try:
        import rclpy
        from action_stream_msgs.msg import (
            ActionChunk,
            EpisodeControl,
            InferenceRequest,
            Observation,
            RobotCommand,
        )
        from rclpy.clock import Clock, ClockType
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock as ClockMessage
    except ImportError as exc:  # pragma: no cover - requires ROS image
        raise RuntimeError("test_plant_node requires ROS 2 and action_stream_msgs") from exc

    class TestPlantNode(Node):
        def __init__(self) -> None:
            super().__init__("action_stream_test_plant")
            self.declare_parameter("strategy", "aligned_async")
            self.declare_parameter("seed", 0)
            self.declare_parameter("episode_id", "")
            self.declare_parameter("max_steps", 180)
            self.declare_parameter("request_interval_steps", 10)
            self.declare_parameter("sync_retry_timeout_seconds", 3.0)
            self.declare_parameter("startup_delay_seconds", 0.5)
            self.declare_parameter("shutdown_grace_seconds", 0.5)
            self.declare_parameter("observation_topic", "/action_stream/observation")
            self.declare_parameter("request_topic", "/action_stream/inference_request")
            self.declare_parameter("action_chunk_topic", "/action_stream/action_chunk")
            self.declare_parameter("command_topic", "/action_stream/robot_command")
            self.declare_parameter("episode_control_topic", "/action_stream/episode_control")
            self.declare_parameter("clock_topic", "/clock")

            self._strategy = str(self.get_parameter("strategy").value)
            if self._strategy not in {"sync_hold", "naive_async", "aligned_async"}:
                raise ValueError(f"unknown strategy: {self._strategy}")
            self._seed = int(self.get_parameter("seed").value)
            episode_parameter = str(self.get_parameter("episode_id").value)
            self._episode_id = episode_parameter or f"m7-ros-{self._seed}-{self._strategy}"
            self._request_interval = int(self.get_parameter("request_interval_steps").value)
            if not 0 < self._request_interval < CHUNK_HORIZON:
                raise ValueError("request_interval_steps must lie in [1,H-1]")
            self._retry_timeout_ns = int(
                float(self.get_parameter("sync_retry_timeout_seconds").value) * 1e9
            )
            self._startup_delay_ns = int(
                float(self.get_parameter("startup_delay_seconds").value) * 1e9
            )
            self._shutdown_grace_ns = int(
                float(self.get_parameter("shutdown_grace_seconds").value) * 1e9
            )
            self._plant = ReachLiftPlant(max_steps=int(self.get_parameter("max_steps").value))
            self._plant.reset(self._episode_id, seed=self._seed)
            self._generation_id = 0
            self._next_request_id = 1
            self._pending_sync_request_id: int | None = None
            self._pending_since_ns = 0
            self._bootstrap_complete = False
            self._started = False
            self._terminated = False
            self._shutdown_at_ns: int | None = None
            self._latest_command: RobotCommand | None = None
            self._last_observation: Observation | None = None
            self._created_ns = time.monotonic_ns()

            qos = QoSProfile(depth=256, reliability=ReliabilityPolicy.RELIABLE)
            lifecycle_qos = QoSProfile(
                depth=16,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._observation_publisher = self.create_publisher(
                Observation, str(self.get_parameter("observation_topic").value), qos
            )
            self._request_publisher = self.create_publisher(
                InferenceRequest, str(self.get_parameter("request_topic").value), qos
            )
            self._control_publisher = self.create_publisher(
                EpisodeControl,
                str(self.get_parameter("episode_control_topic").value),
                lifecycle_qos,
            )
            self._clock_publisher = self.create_publisher(
                ClockMessage, str(self.get_parameter("clock_topic").value), 10
            )
            self._command_subscription = self.create_subscription(
                RobotCommand,
                str(self.get_parameter("command_topic").value),
                self._on_command,
                qos,
            )
            self._chunk_subscription = self.create_subscription(
                ActionChunk,
                str(self.get_parameter("action_chunk_topic").value),
                self._on_chunk,
                qos,
            )
            self._timer = self.create_timer(
                1.0 / CONTROL_FREQUENCY_HZ,
                self._on_tick,
                clock=Clock(clock_type=ClockType.STEADY_TIME),
            )
            self._publish_clock()

        def _sim_time_ns(self) -> int:
            return int(self._plant.step_count * 1_000_000_000 / CONTROL_FREQUENCY_HZ)

        def _publish_clock(self) -> None:
            message = ClockMessage()
            assign_time(message.clock, self._sim_time_ns())
            self._clock_publisher.publish(message)

        def _control(self, command: int, *, success: bool = False) -> None:
            message = EpisodeControl()
            message.message_kind = EpisodeControl.REQUEST
            message.command = command
            assign_time(message.sim_stamp, self._sim_time_ns())
            message.steady_time_ns = time.monotonic_ns()
            message.wall_time_ns = time.time_ns()
            message.episode_id = self._episode_id
            message.generation_id = self._generation_id
            # REQUEST carries a command, not an echoed lifecycle status.  The
            # sole payload exception is the terminal task outcome consumed by
            # the executor; active/terminated remain STATUS-only fields.
            message.active = False
            message.terminated = False
            message.success = success if command == EpisodeControl.TERMINATE else False
            self._control_publisher.publish(message)

        def _observation_message(self) -> Observation:
            state = self._plant.observation()
            message = Observation()
            assign_time(message.sim_stamp, self._sim_time_ns())
            message.steady_time_ns = time.monotonic_ns()
            message.wall_time_ns = time.time_ns()
            message.episode_id = state.episode_id
            message.observation_step = state.observation_step
            message.task_id = TASK_ID
            message.robot_state = list(state.robot_state)
            message.task_state = list(state.task_state)
            message.terminated = state.terminated
            return message

        def _publish_observation(self) -> None:
            message = self._observation_message()
            self._last_observation = message
            self._observation_publisher.publish(message)

        def _submit_request(self) -> None:
            observation = self._observation_message()
            request = InferenceRequest()
            request.sim_stamp = copy.deepcopy(observation.sim_stamp)
            request.request_steady_time_ns = time.monotonic_ns()
            request.request_wall_time_ns = time.time_ns()
            request.episode_id = self._episode_id
            request.request_id = self._next_request_id
            request.generation_id = self._generation_id
            request.source_observation_step = observation.observation_step
            request.source_sim_stamp = copy.deepcopy(observation.sim_stamp)
            request.source_observation_steady_time_ns = observation.steady_time_ns
            request.expected_horizon = CHUNK_HORIZON
            request.observation = observation
            self._request_publisher.publish(request)
            self._pending_sync_request_id = request.request_id
            self._pending_since_ns = time.monotonic_ns()
            self._next_request_id += 1

        def _on_command(self, message) -> None:
            if message.episode_id != self._episode_id or self._terminated:
                return
            expected = self._plant.step_count + 1
            if int(message.actual_target_step) != expected:
                self.get_logger().warning(
                    "Ignoring command target=" f"{message.actual_target_step}; expected={expected}"
                )
                return
            self._latest_command = message

        def _on_chunk(self, message) -> None:
            if message.episode_id != self._episode_id:
                return
            if self._pending_sync_request_id == int(message.request_id):
                self._pending_sync_request_id = None
                if not self._bootstrap_complete:
                    self._bootstrap_complete = True
                    # Let the executor ingest the chunk before the first observation.
                    self._latest_command = None

        def _start(self) -> None:
            self._started = True
            self._control(EpisodeControl.START)
            self._submit_request()
            self.get_logger().info(
                f"Started {self._episode_id}: strategy={self._strategy} " f"seed={self._seed}"
            )

        def _terminate(self) -> None:
            if self._terminated:
                return
            self._terminated = True
            self._publish_clock()
            self._publish_observation()
            self._control(EpisodeControl.TERMINATE, success=self._plant.success)
            self._shutdown_at_ns = time.monotonic_ns() + self._shutdown_grace_ns
            self.get_logger().info(
                f"Episode complete: success={self._plant.success} "
                f"reason={self._plant.termination_reason} "
                f"steps={self._plant.step_count}"
            )

        def _on_tick(self) -> None:
            now = time.monotonic_ns()
            if self._shutdown_at_ns is not None and now >= self._shutdown_at_ns:
                rclpy.shutdown()
                return
            if not self._started:
                if now - self._created_ns >= self._startup_delay_ns:
                    self._start()
                return
            if self._terminated:
                return
            if self._pending_sync_request_id is not None:
                if now - self._pending_since_ns >= self._retry_timeout_ns:
                    self.get_logger().warning(
                        f"Retrying timed-out request {self._pending_sync_request_id}"
                    )
                    self._submit_request()
                return
            if not self._bootstrap_complete:
                return

            # The observation that releases the bootstrap/rebuilt queue is sent
            # only after its response has arrived, preserving O+1 labeling.
            if self._last_observation is None or (
                int(self._last_observation.observation_step) != self._plant.step_count
            ):
                self._publish_clock()
                self._publish_observation()
                return
            if self._latest_command is None:
                return

            command = self._latest_command
            self._latest_command = None
            result = self._plant.step(tuple(command.command))
            self._publish_clock()
            if result.terminated:
                self._terminate()
                return

            step = self._plant.step_count
            if self._strategy == "sync_hold" and step % CHUNK_HORIZON == 0:
                self._submit_request()
                return
            self._publish_observation()
            if self._strategy != "sync_hold" and step > 0 and step % self._request_interval == 0:
                self._submit_request()
                # Async requests do not pause plant progress.
                self._pending_sync_request_id = None

    rclpy.init(args=args)
    node = TestPlantNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
