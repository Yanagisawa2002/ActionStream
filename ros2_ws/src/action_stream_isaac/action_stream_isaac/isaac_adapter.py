"""Isaac Sim 6.0.1 Franka reach-and-lift bridge for ActionStream.

The module is intentionally safe to import without ROS 2 or Isaac Sim.  The
capability probe runs before either runtime is imported, and a failed probe
terminates this executable.  It never starts the deterministic ROS test plant.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Final, Sequence

from .capability_probe import probe
from .ros_contract import (
    BOOTSTRAP_OBSERVATION_TOPIC,
    CLOCK_TOPIC,
    EPISODE_CONTROL_TOPIC,
    OBSERVATION_TOPIC,
    ROBOT_COMMAND_TOPIC,
    load_and_validate_message_types,
    set_ros_time,
)
from .request_driver import RequestDriverConfig, create_request_driver_node
from .task_logic import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    TASK_ID,
    TaskThresholds,
    axis_angle_to_quaternion_wxyz,
    deterministic_spawn_position,
    evaluate_task,
    gripper_command_to_finger_positions,
    pack_robot_state,
    pack_task_state,
    quaternion_wxyz_to_axis_angle,
    split_sim_time,
    validate_robot_command_for_step,
)

PHYSICS_FREQUENCY_HZ: Final = 60
CONTROL_FREQUENCY_HZ: Final = 20
PHYSICS_STEPS_PER_CONTROL: Final = PHYSICS_FREQUENCY_HZ // CONTROL_FREQUENCY_HZ
PHYSICS_DT_SECONDS: Final = 1.0 / PHYSICS_FREQUENCY_HZ
CONTROL_DT_SECONDS: Final = 1.0 / CONTROL_FREQUENCY_HZ

FRANKA_PATH: Final = "/World/Franka"
OBJECT_PATH: Final = "/World/LiftObject"
TARGET_PATH: Final = "/World/LiftTarget"
OBJECT_SIDE_M: Final = 0.05
LIFT_TARGET_OFFSET_M: Final = 0.18
HOME_JOINT_POSITIONS: Final = (
    0.0,
    -0.785398,
    0.0,
    -2.356194,
    0.0,
    1.570796,
    0.785398,
    0.04,
    0.04,
)


@dataclass(frozen=True)
class PendingCommand:
    actual_target_step: int
    command: tuple[float, ...]
    hold: bool
    source_request_id: int
    source_generation_id: int
    source_observation_step: int
    source_target_step: int
    reason: str


@dataclass(frozen=True)
class ControlRequest:
    command: int
    episode_id: str
    generation_id: int
    success: bool


@dataclass(frozen=True)
class InKitPolicyConfig:
    episode_id: str
    fault_trace_file: Path
    event_log_path: Path
    strategy: str
    profile_id: str
    seed: int
    trace_sha256: str
    request_interval_steps: int


@dataclass(frozen=True)
class VideoCaptureConfig:
    """Validated parameters for a bounded Isaac viewport MP4 capture."""

    output_path: Path
    fps: int
    width: int
    height: int
    finalize_timeout_seconds: float


def _flat_floats(value: Any, *, length: int, name: str) -> tuple[float, ...]:
    """Convert a singleton experimental-API tensor to a finite flat tuple."""

    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    try:
        converted = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Isaac returned an invalid {name}: {value!r}") from exc
    if len(converted) != length or not all(math.isfinite(item) for item in converted):
        raise RuntimeError(
            f"Isaac returned an invalid {name}: expected {length} finite values, "
            f"got {converted!r}"
        )
    return converted


class IsaacFrankaScene:
    """Small scene built only from Isaac Sim 6 experimental/current APIs."""

    def __init__(self, simulation_app: Any) -> None:
        # These imports are legal only after SimulationApp has initialized Kit.
        import numpy as np
        import omni.timeline
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.objects import Cube, GroundPlane
        from isaacsim.core.experimental.prims import (
            GeomPrim,
            RigidPrim,
            XformPrim,
        )
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager

        app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples")
        simulation_app.update()
        from isaacsim.robot.experimental.manipulators.examples.franka import Franka

        required_franka_api = (
            "set_end_effector_pose",
            "get_current_state",
            "reset_to_default_pose",
            "set_gripper_position",
        )
        missing = tuple(name for name in required_franka_api if not hasattr(Franka, name))
        if missing:
            raise RuntimeError(
                "Installed Isaac Franka experimental API is incompatible; missing " f"{missing!r}"
            )

        self._app = simulation_app
        self._np = np
        self._RenderingManager = RenderingManager
        self._SimulationManager = SimulationManager
        self._app_utils = app_utils
        self._timeline = omni.timeline.get_timeline_interface()

        stage_utils.create_new_stage()
        GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, 0.0])

        self._franka = Franka(robot_path=FRANKA_PATH, create_robot=True)
        self._articulation = self._franka

        Cube(paths=OBJECT_PATH, positions=[0.45, 0.0, 0.026], sizes=OBJECT_SIDE_M)
        self._object = RigidPrim(paths=OBJECT_PATH)
        GeomPrim(paths=OBJECT_PATH, apply_collision_apis=True)
        Cube(paths=TARGET_PATH, positions=[0.45, 0.0, 0.206], sizes=0.025)
        self._target_marker = XformPrim(TARGET_PATH)

        # Let Kit finish loading the Franka USD reference before physics setup.
        # The installed 6.0.1 experimental examples perform the same update
        # between scene creation and SimulationManager setup.
        self._app.update()

        # setup_simulation is the current coherent-rate entry point in 6.0.1.
        SimulationManager.setup_simulation(dt=PHYSICS_DT_SECONDS, device="cpu")
        RenderingManager.set_dt(PHYSICS_DT_SECONDS)
        self._app.update()
        app_utils.play()
        # Physics tensor views are initialized on the first Kit updates after
        # play.  Calling SimulationManager.step before these updates leaves the
        # Franka articulation view invalid.
        self._app.update()
        self._app.update()
        # Keep the timeline initialized but paused so each control tick has
        # exactly the three explicit SimulationManager physics steps below.
        self._timeline.pause()
        self._app.update()
        for _ in range(5):
            self._raw_step()

        self._joint_space = tuple(self._articulation.dof_names)
        if len(self._joint_space) != 9:
            raise RuntimeError(
                "Official Franka asset contract changed: expected 9 DOFs, "
                f"got {len(self._joint_space)} ({self._joint_space!r})"
            )
        try:
            self._finger_indices = (
                self._joint_space.index("panda_finger_joint1"),
                self._joint_space.index("panda_finger_joint2"),
            )
        except ValueError as exc:
            raise RuntimeError(
                "Official Franka asset lacks panda_finger_joint1/2; refusing an "
                "ambiguous gripper mapping"
            ) from exc

        self._target_xyz = (0.45, 0.0, 0.35)
        self._target_wxyz = (1.0, 0.0, 0.0, 0.0)
        self._gripper_command = GRIPPER_OPEN
        self.reset("adapter-bootstrap")

    @property
    def gripper_command(self) -> float:
        return self._gripper_command

    def _raw_step(self) -> None:
        self._SimulationManager.step()
        self._RenderingManager.render()
        self._app.update()

    def reset(self, episode_id: str) -> tuple[float, float, float]:
        """Deterministically reset robot, object, controller, and target marker."""

        spawn = deterministic_spawn_position(episode_id)
        self._franka.reset_to_default_pose()
        self._articulation.set_dof_positions(HOME_JOINT_POSITIONS)
        self._articulation.set_dof_velocities((0.0,) * len(HOME_JOINT_POSITIONS))
        self._articulation.set_dof_position_targets(HOME_JOINT_POSITIONS)
        self._object.set_world_poses(
            positions=spawn,
            orientations=(1.0, 0.0, 0.0, 0.0),
        )
        self._object.set_velocities(
            linear_velocities=(0.0, 0.0, 0.0),
            angular_velocities=(0.0, 0.0, 0.0),
        )
        self._target_marker.set_world_poses(
            positions=(spawn[0], spawn[1], spawn[2] + LIFT_TARGET_OFFSET_M),
            orientations=(1.0, 0.0, 0.0, 0.0),
        )
        self._gripper_command = GRIPPER_OPEN
        for _ in range(5):
            self._raw_step()

        self._target_xyz, self._target_wxyz = self.end_effector_pose()
        self._apply_finger_target()
        return self.object_pose()[0]

    def set_command(self, command: Sequence[float]) -> None:
        self._target_xyz = tuple(float(value) for value in command[:3])
        self._target_wxyz = axis_angle_to_quaternion_wxyz(command[3:6])
        self._gripper_command = float(command[6])

    def _apply_finger_target(self) -> None:
        finger_positions = gripper_command_to_finger_positions(self._gripper_command)
        self._franka.set_gripper_position(
            self._np.asarray(finger_positions, dtype=self._np.float64)
        )

    def step(self) -> None:
        self._franka.set_end_effector_pose(
            position=self._np.asarray(self._target_xyz, dtype=self._np.float64),
            orientation=self._np.asarray(self._target_wxyz, dtype=self._np.float64),
            ik_method="damped-least-squares",
        )
        self._apply_finger_target()
        self._raw_step()

    def sim_time_seconds(self) -> float:
        # Physics time, not timeline or wall time.  NVIDIA defines it as
        # completed physics steps divided by the configured steps per second.
        steps = int(self._SimulationManager.get_num_physics_steps())
        return steps / float(PHYSICS_FREQUENCY_HZ)

    def end_effector_pose(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        _dof_positions, positions, orientations = self._franka.get_current_state()
        return (
            _flat_floats(positions, length=3, name="end-effector position"),
            _flat_floats(orientations, length=4, name="end-effector orientation"),
        )

    def object_pose(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        positions, orientations = self._object.get_world_poses()
        return (
            _flat_floats(positions, length=3, name="object position"),
            _flat_floats(orientations, length=4, name="object orientation"),
        )

    def close(self) -> None:
        self._app_utils.stop()


class ActionStreamIsaacBridge:
    """Native rclpy bridge around the current Isaac scene APIs."""

    def __init__(
        self,
        scene: IsaacFrankaScene,
        *,
        command_timeout_seconds: float,
        defer_initial_observation: bool = False,
    ) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )

        self._rclpy = rclpy
        self._scene = scene
        self._types = load_and_validate_message_types()
        self.node = Node("action_stream_isaac_adapter")
        self._thresholds = TaskThresholds()
        self._command_timeout_seconds = command_timeout_seconds
        self._defer_initial_observation = defer_initial_observation

        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        lifecycle_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # Match rclcpp::ClockQoS on ROS 2 Jazzy.  Jazzy's rclpy does not export
        # the later ``qos_profile_clock`` convenience constant.
        clock_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._observation_publisher = self.node.create_publisher(
            self._types.Observation, OBSERVATION_TOPIC, reliable
        )
        self._bootstrap_observation_publisher = self.node.create_publisher(
            self._types.Observation, BOOTSTRAP_OBSERVATION_TOPIC, reliable
        )
        self._clock_publisher = self.node.create_publisher(
            self._types.Clock, CLOCK_TOPIC, clock_qos
        )
        self._control_publisher = self.node.create_publisher(
            self._types.EpisodeControl, EPISODE_CONTROL_TOPIC, lifecycle_qos
        )
        self._command_subscription = self.node.create_subscription(
            self._types.RobotCommand,
            ROBOT_COMMAND_TOPIC,
            self._on_robot_command,
            reliable,
        )
        self._control_subscription = self.node.create_subscription(
            self._types.EpisodeControl,
            EPISODE_CONTROL_TOPIC,
            self._on_episode_control,
            lifecycle_qos,
        )

        self._control_requests: deque[ControlRequest] = deque()
        self._pending_command: PendingCommand | None = None
        self._active_episode_id: str | None = None
        self._generation_id = 0
        self._observation_step = 0
        self._initial_object_z = 0.0
        self._lift_target_xyz = (0.0, 0.0, 0.0)
        self._success_streak = 0
        self._command_deadline_wall: float | None = None
        self._rejected_commands = 0
        self._last_terminal_episode_id: str | None = None
        self._deferred_initial_observation: Any | None = None

    def _stamp(self, message: Any) -> None:
        sec, nanosec = split_sim_time(self._scene.sim_time_seconds())
        set_ros_time(message.sim_stamp, sec=sec, nanosec=nanosec)
        message.steady_time_ns = time.monotonic_ns()
        message.wall_time_ns = time.time_ns()

    def _on_episode_control(self, message: Any) -> None:
        # Status messages share the lifecycle topic.  The explicit discriminator
        # prevents an acknowledgement from becoming a second lifecycle request.
        if message.message_kind == self._types.EpisodeControl.STATUS:
            return
        if message.message_kind != self._types.EpisodeControl.REQUEST:
            self.node.get_logger().error(
                "episode_control_rejected reason=unknown_message_kind "
                f"message_kind={message.message_kind}"
            )
            return
        if bool(message.active) or bool(message.terminated):
            self.node.get_logger().error(
                "episode_control_rejected reason=request_has_status_flags"
            )
            return
        if not message.episode_id:
            self.node.get_logger().error("episode_control_rejected reason=empty_episode_id")
            return
        if message.command not in {
            self._types.EpisodeControl.START,
            self._types.EpisodeControl.RESET,
            self._types.EpisodeControl.TERMINATE,
        }:
            self.node.get_logger().error(
                f"episode_control_rejected reason=unknown_command command={message.command}"
            )
            return
        if message.command in {
            self._types.EpisodeControl.START,
            self._types.EpisodeControl.RESET,
        } and bool(message.success):
            self.node.get_logger().error(
                "episode_control_rejected reason=nonterminal_request_has_success"
            )
            return
        self._control_requests.append(
            ControlRequest(
                command=int(message.command),
                episode_id=str(message.episode_id),
                generation_id=int(message.generation_id),
                success=bool(message.success),
            )
        )

    def _on_robot_command(self, message: Any) -> None:
        pending_step = (
            self._pending_command.actual_target_step if self._pending_command is not None else None
        )
        validation = validate_robot_command_for_step(
            active_episode_id=self._active_episode_id,
            latest_observation_step=self._observation_step,
            pending_target_step=pending_step,
            message_episode_id=str(message.episode_id),
            actual_target_step=int(message.actual_target_step),
            command=message.command,
            hold=bool(message.hold),
        )
        if not validation.accepted or validation.command is None:
            self._rejected_commands += 1
            self.node.get_logger().warning(
                "robot_command_rejected "
                f"reason={validation.reason} episode={message.episode_id!r} "
                f"actual_target_step={message.actual_target_step} "
                f"latest_observation_step={self._observation_step}"
            )
            return
        self._pending_command = PendingCommand(
            actual_target_step=int(message.actual_target_step),
            command=validation.command,
            hold=bool(message.hold),
            source_request_id=int(message.source_request_id),
            source_generation_id=int(message.source_generation_id),
            source_observation_step=int(message.source_observation_step),
            source_target_step=int(message.source_target_step),
            reason=str(message.reason),
        )
        self._command_deadline_wall = None

    def publish_auto_start(self, episode_id: str) -> None:
        if not episode_id:
            return
        message = self._types.EpisodeControl()
        message.message_kind = self._types.EpisodeControl.REQUEST
        message.command = self._types.EpisodeControl.START
        self._stamp(message)
        message.episode_id = episode_id
        message.generation_id = 1
        message.active = False
        message.terminated = False
        message.success = False
        self._control_publisher.publish(message)

    def _publish_control_status(
        self,
        *,
        command: int,
        active: bool,
        terminated: bool,
        success: bool,
        episode_id: str | None = None,
    ) -> None:
        message = self._types.EpisodeControl()
        message.message_kind = self._types.EpisodeControl.STATUS
        message.command = command
        self._stamp(message)
        message.episode_id = episode_id or self._active_episode_id or ""
        message.generation_id = self._generation_id
        message.active = active
        message.terminated = terminated
        message.success = success
        self._control_publisher.publish(message)

    def _publish_clock(self) -> None:
        message = self._types.Clock()
        sec, nanosec = split_sim_time(self._scene.sim_time_seconds())
        set_ros_time(message.clock, sec=sec, nanosec=nanosec)
        self._clock_publisher.publish(message)

    def _publish_observation(self, *, defer_for_bootstrap: bool = False) -> bool:
        if self._active_episode_id is None:
            return False
        end_effector_xyz, end_effector_wxyz = self._scene.end_effector_pose()
        object_xyz, _object_wxyz = self._scene.object_pose()
        evaluation = evaluate_task(
            end_effector_xyz=end_effector_xyz,
            object_xyz=object_xyz,
            initial_object_z=self._initial_object_z,
            episode_step=self._observation_step,
            previous_success_streak=self._success_streak,
            thresholds=self._thresholds,
        )
        self._success_streak = evaluation.success_streak_steps
        grasped = (
            self._scene.gripper_command == GRIPPER_CLOSED
            and evaluation.end_effector_object_distance_m <= self._thresholds.max_grasp_distance_m
        )
        robot_state = pack_robot_state(
            end_effector_xyz=end_effector_xyz,
            end_effector_axis_angle_xyz=quaternion_wxyz_to_axis_angle(end_effector_wxyz),
            gripper_command=self._scene.gripper_command,
        )
        task_state = pack_task_state(
            object_xyz=object_xyz,
            lift_target_xyz=self._lift_target_xyz,
            grasped=grasped,
            initial_object_z=self._initial_object_z,
            evaluation=evaluation,
            episode_step=self._observation_step,
        )

        message = self._types.Observation()
        self._stamp(message)
        message.episode_id = self._active_episode_id
        message.observation_step = self._observation_step
        message.task_id = TASK_ID
        message.robot_state = list(robot_state)
        message.task_state = list(task_state)
        message.terminated = evaluation.terminated
        if defer_for_bootstrap:
            if evaluation.terminated:
                raise RuntimeError("initial Isaac observation unexpectedly terminated")
            self._deferred_initial_observation = message
            self._bootstrap_observation_publisher.publish(message)
            return False
        self._observation_publisher.publish(message)
        if not evaluation.terminated:
            self._command_deadline_wall = time.monotonic() + self._command_timeout_seconds
        else:
            self.node.get_logger().info(
                "episode_terminal "
                f"episode={self._active_episode_id!r} "
                f"step={self._observation_step} success={evaluation.success} "
                f"reason={evaluation.termination_reason}"
            )
            self._publish_control_status(
                command=self._types.EpisodeControl.TERMINATE,
                active=False,
                terminated=True,
                success=evaluation.success,
            )
            self._active_episode_id = None
            self._pending_command = None
            self._command_deadline_wall = None
            self._last_terminal_episode_id = message.episode_id
        return evaluation.terminated

    def process_control_requests(self) -> None:
        while self._control_requests:
            request = self._control_requests.popleft()
            if request.command == self._types.EpisodeControl.START:
                if self._active_episode_id is not None:
                    self.node.get_logger().warning(
                        "episode_control_rejected reason=episode_already_active "
                        f"active_episode={self._active_episode_id!r}"
                    )
                    continue
                self._begin_episode(request)
            elif request.command == self._types.EpisodeControl.RESET:
                self._begin_episode(request)
            else:
                if (
                    self._active_episode_id is None
                    and request.episode_id == self._last_terminal_episode_id
                ):
                    # The request driver publishes the auditable terminal
                    # REQUEST after consuming Isaac's terminal Observation.
                    self._publish_control_status(
                        command=self._types.EpisodeControl.TERMINATE,
                        active=False,
                        terminated=True,
                        success=request.success,
                        episode_id=request.episode_id,
                    )
                    continue
                if request.episode_id != self._active_episode_id:
                    self.node.get_logger().warning(
                        "episode_control_rejected reason=previous_episode "
                        f"episode={request.episode_id!r}"
                    )
                    continue
                self._publish_control_status(
                    command=self._types.EpisodeControl.TERMINATE,
                    active=False,
                    terminated=True,
                    success=request.success,
                )
                self._last_terminal_episode_id = request.episode_id
                self._active_episode_id = None
                self._pending_command = None
                self._command_deadline_wall = None

    def _begin_episode(self, request: ControlRequest) -> None:
        object_xyz = self._scene.reset(request.episode_id)
        self._active_episode_id = request.episode_id
        self._generation_id = request.generation_id
        self._observation_step = 0
        self._pending_command = None
        self._deferred_initial_observation = None
        self._initial_object_z = object_xyz[2]
        self._lift_target_xyz = (
            object_xyz[0],
            object_xyz[1],
            object_xyz[2] + LIFT_TARGET_OFFSET_M,
        )
        self._success_streak = 0
        self._last_terminal_episode_id = None
        self._publish_clock()
        self._publish_control_status(
            command=request.command,
            active=True,
            terminated=False,
            success=False,
        )
        self._publish_observation(defer_for_bootstrap=self._defer_initial_observation)
        self.node.get_logger().info(
            "episode_started "
            f"episode={request.episode_id!r} generation={request.generation_id} "
            f"task={TASK_ID} control_hz={CONTROL_FREQUENCY_HZ}"
        )

    def step_if_ready(self) -> bool:
        if self._active_episode_id is None or self._pending_command is None:
            return False
        started = time.monotonic()
        command = self._pending_command
        self._pending_command = None
        if not command.hold:
            self._scene.set_command(command.command)
        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            self._scene.step()
            self._publish_clock()
        remaining = CONTROL_DT_SECONDS - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)
        self._observation_step = command.actual_target_step
        self._publish_observation()
        return True

    @property
    def initial_observation_is_deferred(self) -> bool:
        return self._deferred_initial_observation is not None

    def release_initial_observation(self) -> None:
        """Release O=0 only after the executor accepts the bootstrap chunk."""

        message = self._deferred_initial_observation
        if message is None:
            return
        self._deferred_initial_observation = None
        self._observation_publisher.publish(message)
        self._command_deadline_wall = time.monotonic() + self._command_timeout_seconds
        self.node.get_logger().info(
            "bootstrap_observation_released "
            f"episode={message.episode_id!r} step={message.observation_step}"
        )

    def command_wait_timed_out(self) -> bool:
        return (
            self._active_episode_id is not None
            and self._pending_command is None
            and self._command_deadline_wall is not None
            and time.monotonic() >= self._command_deadline_wall
        )

    def episode_is_terminal(self, episode_id: str) -> bool:
        return bool(episode_id) and self._last_terminal_episode_id == episode_id

    def close(self) -> None:
        self.node.destroy_node()


class InKitPolicyRuntime:
    """Post-Kit Python policy, deterministic fault, recorder, and request graph."""

    def __init__(self, bridge_node: Any, config: InKitPolicyConfig) -> None:
        from action_stream_benchmark.event_recorder_node import (
            create_event_recorder_node,
        )
        from action_stream_benchmark.fault_injector_node import (
            create_fault_injector_node,
        )
        from action_stream_policy.policy_node import create_scripted_policy_node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter

        def parameters(**values: object) -> list[Any]:
            return [Parameter(name, value=value) for name, value in values.items()]

        self._executor: Any | None = SingleThreadedExecutor()
        self._nodes: list[Any] = []
        self._recorder: Any | None = None
        self.request_driver: Any | None = None
        try:
            policy = create_scripted_policy_node(
                parameter_overrides=parameters(
                    use_sim_time=True,
                    request_topic="/action_stream/inference_request",
                    raw_chunk_topic="/action_stream/raw_action_chunk",
                    event_topic="/action_stream/events",
                )
            )
            self._nodes.append(policy)
            injector = create_fault_injector_node(
                parameter_overrides=parameters(
                    use_sim_time=True,
                    trace_file=str(config.fault_trace_file),
                    raw_chunk_topic="/action_stream/raw_action_chunk",
                    action_chunk_topic="/action_stream/action_chunk",
                    event_topic="/action_stream/events",
                )
            )
            self._nodes.append(injector)
            recorder = create_event_recorder_node(
                parameter_overrides=parameters(
                    use_sim_time=True,
                    output_path=str(config.event_log_path),
                    strategy=config.strategy,
                    profile_id=config.profile_id,
                    seed=config.seed,
                    trace_sha256=config.trace_sha256,
                    evidence_class="ros_cpp_isaac_sim",
                    observation_topic=OBSERVATION_TOPIC,
                    request_topic="/action_stream/inference_request",
                    action_chunk_topic="/action_stream/action_chunk",
                    command_topic=ROBOT_COMMAND_TOPIC,
                    diagnostics_topic="/action_stream/diagnostics",
                    event_topic="/action_stream/events",
                    episode_control_topic=EPISODE_CONTROL_TOPIC,
                )
            )
            self._nodes.append(recorder)
            self._recorder = recorder
            request_driver = create_request_driver_node(
                RequestDriverConfig(
                    episode_id=config.episode_id,
                    generation_id=1,
                    request_interval_steps=config.request_interval_steps,
                    expected_horizon=30,
                )
            )
            self._nodes.append(request_driver)
            self.request_driver = request_driver
            for node in [bridge_node, *self._nodes]:
                if not self._executor.add_node(node):
                    raise RuntimeError(
                        f"failed to add post-Kit node {node.get_name()!r} to executor"
                    )
        except Exception:
            self.close()
            raise

    @property
    def bootstrap_ready(self) -> bool:
        return bool(self.request_driver is not None and self.request_driver.bootstrap_ready)

    @property
    def complete(self) -> bool:
        return bool(self.request_driver is not None and self.request_driver.complete)

    @property
    def request_count(self) -> int:
        if self.request_driver is None:
            return 0
        return int(self.request_driver.request_count)

    @property
    def terminal_success(self) -> bool:
        return bool(self.request_driver is not None and self.request_driver.terminal_success)

    @property
    def terminal_step(self) -> int | None:
        if self.request_driver is None:
            return None
        value = self.request_driver.terminal_step
        return None if value is None else int(value)

    def spin_once(self, *, timeout_sec: float) -> None:
        if self._executor is None:
            raise RuntimeError("post-Kit policy executor is closed")
        self._executor.spin_once(timeout_sec=timeout_sec)

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if self._recorder is not None:
            self._recorder.close()
        for node in reversed(self._nodes):
            node.destroy_node()
        self._nodes = []
        self._recorder = None
        self.request_driver = None


class IsaacViewportVideoCapture:
    """Passive 20 Hz viewport frames encoded by NVIDIA's bundled APIs."""

    _EXTENSIONS: Final = (
        "omni.kit.renderer.capture",
        "omni.kit.viewport.utility",
        "omni.videoencoding",
    )

    def __init__(self, simulation_app: Any, config: VideoCaptureConfig) -> None:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        for extension in self._EXTENSIONS:
            manager.set_extension_enabled_immediate(extension, True)
            simulation_app.update()
            if not manager.is_extension_enabled(extension):
                raise RuntimeError(f"Isaac capture extension {extension!r} could not be enabled")

        import asyncio
        from omni.kit.viewport.utility import (
            capture_viewport_to_file,
            get_active_viewport,
        )
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        from pxr import Gf, UsdLux

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("Isaac has no active viewport for MP4 capture")
        stage = viewport.stage
        if stage is None:
            raise RuntimeError("Isaac viewport has no USD stage for MP4 capture")
        dome_light = UsdLux.DomeLight.Define(stage, "/World/M7VideoDomeLight")
        dome_light.CreateIntensityAttr(1000.0)
        dome_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        if config.output_path.exists():
            raise RuntimeError(f"refusing to overwrite video capture: {config.output_path}")
        frame_directory = config.output_path.parent / f"{config.output_path.stem}_frames"
        if frame_directory.exists():
            raise RuntimeError(f"refusing to reuse stale capture frames: {frame_directory}")
        frame_directory.mkdir()

        self._asyncio = asyncio
        self._capture_viewport_to_file = capture_viewport_to_file
        self._config = config
        self._frame_directory = frame_directory
        self._frame_paths: list[Path] = []
        self._wait_tasks: list[Any] = []
        self._viewport = viewport
        self._original_resolution = tuple(viewport.resolution)
        self._resolution_restored = False
        viewport.resolution = (config.width, config.height)
        simulation_app.update()
        camera = ViewportCameraState(viewport=viewport)
        camera.set_position_world(Gf.Vec3d(1.6, 1.4, 1.1), rotate=False)
        camera.set_target_world(Gf.Vec3d(0.2, 0.0, 0.45), rotate=True)
        # Let the camera transition settle before the first episode frame.
        simulation_app.update()
        simulation_app.update()

    def capture_frame(self, simulation_app: Any) -> None:
        """Schedule exactly one LDR PNG after a completed 20 Hz control tick."""

        frame_path = self._frame_directory / f"frame_{len(self._frame_paths):06d}.png"
        helper = self._capture_viewport_to_file(
            self._viewport,
            file_path=str(frame_path),
            is_hdr=False,
        )
        task = self._asyncio.ensure_future(helper.wait_for_result(completion_frames=0))
        self._frame_paths.append(frame_path)
        self._wait_tasks.append(task)
        # Service the scheduled viewport capture without advancing physics.
        simulation_app.update()

    def _restore_resolution(self) -> None:
        if not self._resolution_restored:
            self._viewport.resolution = self._original_resolution
            self._resolution_restored = True

    def finish(self, simulation_app: Any) -> Path:
        """Wait for frames, encode them synchronously, and verify the MP4."""

        if not self._frame_paths:
            raise RuntimeError("Isaac viewport capture recorded zero control frames")
        deadline = time.monotonic() + self._config.finalize_timeout_seconds
        while True:
            for task in self._wait_tasks:
                if task.cancelled():
                    raise RuntimeError("Isaac viewport frame capture was cancelled")
                if task.done() and task.exception() is not None:
                    raise RuntimeError("Isaac viewport frame capture failed") from task.exception()
            pending = any(not task.done() for task in self._wait_tasks)
            missing = any(
                not path.is_file() or path.stat().st_size <= 0 for path in self._frame_paths
            )
            if not pending and not missing:
                break
            if not simulation_app.is_running():
                raise RuntimeError("Isaac stopped before viewport frame capture finalized")
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Isaac viewport frames did not finalize within "
                    f"{self._config.finalize_timeout_seconds:.3f}s"
                )
            simulation_app.update()

        self._restore_resolution()
        import omni.kit.renderer_capture
        from video_encoding import encode_image_file_sequence

        omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        pattern = str(self._frame_directory / "frame_%06d.png")
        encoded = encode_image_file_sequence(
            pattern,
            0,
            self._config.fps,
            str(self._config.output_path),
            False,
        )
        expected = self._config.output_path
        if not encoded:
            raise RuntimeError(f"Isaac video encoder rejected {len(self._frame_paths)} frames")
        if not expected.is_file() or expected.stat().st_size <= 0:
            raise RuntimeError(f"Isaac capture produced no non-empty MP4 at {expected}")
        for frame_path in self._frame_paths:
            frame_path.unlink()
        self._frame_directory.rmdir()
        return expected

    def cancel(self) -> None:
        """Cancel only local waiters and restore the viewport on shutdown."""

        for task in self._wait_tasks:
            if not task.done():
                task.cancel()
        self._restore_resolution()


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", type=_parse_bool, default=True)
    parser.add_argument("--auto-start-episode", default="")
    parser.add_argument("--exit-after-auto-episode", type=_parse_bool, default=False)
    parser.add_argument("--command-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--inkit-policy-run", type=_parse_bool, default=False)
    parser.add_argument("--fault-trace-file", type=Path)
    parser.add_argument("--event-log-path", type=Path)
    parser.add_argument("--strategy", choices=("aligned_async",), default="aligned_async")
    parser.add_argument("--profile-id", default="profile_a")
    parser.add_argument("--seed", type=int, default=2026080300)
    parser.add_argument("--request-interval-steps", type=int, default=10)
    parser.add_argument("--startup-discovery-seconds", type=float, default=1.0)
    parser.add_argument("--bootstrap-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--terminal-drain-seconds", type=float, default=0.5)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-finalize-timeout-seconds", type=float, default=120.0)
    return parser


def _load_policy_config(args: Any) -> InKitPolicyConfig | None:
    if not args.inkit_policy_run:
        return None
    if not args.auto_start_episode or not args.exit_after_auto_episode:
        raise SystemExit(
            "--inkit-policy-run requires --auto-start-episode and "
            "--exit-after-auto-episode true"
        )
    if args.fault_trace_file is None or args.event_log_path is None:
        raise SystemExit("--inkit-policy-run requires --fault-trace-file and --event-log-path")
    if not 0 < args.request_interval_steps < 30:
        raise SystemExit("--request-interval-steps must lie in [1,29]")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    for name in (
        "startup_discovery_seconds",
        "bootstrap_timeout_seconds",
        "terminal_drain_seconds",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.bootstrap_timeout_seconds == 0.0:
        raise SystemExit("--bootstrap-timeout-seconds must be positive")

    trace_file = args.fault_trace_file.resolve()
    event_log_path = args.event_log_path.resolve()
    if not trace_file.is_file():
        raise SystemExit(f"frozen fault trace does not exist: {trace_file}")
    if event_log_path.exists():
        raise SystemExit(f"refusing to overwrite event log: {event_log_path}")
    try:
        from action_stream_benchmark.faults import load_fault_trace

        trace = load_fault_trace(trace_file)
    except (ImportError, OSError, ValueError) as exc:
        raise SystemExit(f"invalid frozen fault trace {trace_file}: {exc}") from exc
    if trace.profile.profile_id != args.profile_id or trace.seed != args.seed:
        raise SystemExit(
            "fault trace metadata mismatch: "
            f"trace=({trace.profile.profile_id}, seed={trace.seed}), "
            f"requested=({args.profile_id}, seed={args.seed})"
        )
    return InKitPolicyConfig(
        episode_id=args.auto_start_episode,
        fault_trace_file=trace_file,
        event_log_path=event_log_path,
        strategy=args.strategy,
        profile_id=args.profile_id,
        seed=args.seed,
        trace_sha256=trace.sha256,
        request_interval_steps=args.request_interval_steps,
    )


def _load_video_capture_config(args: Any) -> VideoCaptureConfig | None:
    if args.video_output is None:
        return None
    if not args.auto_start_episode or not args.exit_after_auto_episode:
        raise SystemExit(
            "--video-output requires --auto-start-episode and " "--exit-after-auto-episode true"
        )
    output_path = args.video_output.resolve()
    if output_path.suffix.lower() != ".mp4":
        raise SystemExit("--video-output must end in .mp4")
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite video capture: {output_path}")
    frame_directory = output_path.parent / f"{output_path.stem}_frames"
    if frame_directory.exists():
        raise SystemExit(f"refusing to reuse stale capture frames: {frame_directory}")
    if not 1 <= args.video_fps <= 120:
        raise SystemExit("--video-fps must lie in [1,120]")
    if not 64 <= args.video_width <= 4096 or not 64 <= args.video_height <= 4096:
        raise SystemExit("--video-width/--video-height are outside the supported bounds")
    if (
        not math.isfinite(args.video_finalize_timeout_seconds)
        or args.video_finalize_timeout_seconds <= 0.0
    ):
        raise SystemExit("--video-finalize-timeout-seconds must be finite and positive")
    return VideoCaptureConfig(
        output_path=output_path,
        fps=args.video_fps,
        width=args.video_width,
        height=args.video_height,
        finalize_timeout_seconds=args.video_finalize_timeout_seconds,
    )


def _enable_ros_bridge(simulation_app: Any) -> None:
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    extension = "isaacsim.ros2.bridge"
    manager.set_extension_enabled_immediate(extension, True)
    simulation_app.update()
    if not manager.is_extension_enabled(extension):
        raise RuntimeError(
            "Isaac ROS 2 bridge extension isaacsim.ros2.bridge could not be enabled"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command_timeout_seconds <= 0.0 or not math.isfinite(args.command_timeout_seconds):
        raise SystemExit("--command-timeout-seconds must be finite and positive")
    if args.exit_after_auto_episode and not args.auto_start_episode:
        raise SystemExit("--exit-after-auto-episode requires --auto-start-episode")
    policy_config = _load_policy_config(args)
    video_config = _load_video_capture_config(args)

    # The standalone capability probe validates rclpy in a child process.  The
    # adapter cannot do that on the supported Windows pip path because the
    # Isaac bridge loads its bundled ROS DLLs only after SimulationApp starts.
    # Defer the ROS/message imports to the explicit post-bridge validation
    # below; do not preload a private DLL or silently fall back to a test plant.
    report = probe(require_ros_imports=False)
    if not report.ready:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True), file=sys.stderr)
        print(
            "FATAL: Isaac adapter capability probe failed; refusing to import Isaac or "
            "start simulation. No deterministic test-plant fallback was invoked.",
            file=sys.stderr,
        )
        return 2

    simulation_app: Any | None = None
    scene: IsaacFrankaScene | None = None
    bridge: ActionStreamIsaacBridge | None = None
    policy_runtime: InKitPolicyRuntime | None = None
    video_capture: IsaacViewportVideoCapture | None = None
    rclpy: Any | None = None
    exit_code = 0
    try:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": bool(args.headless)})
        _enable_ros_bridge(simulation_app)

        import rclpy as rclpy_module

        rclpy = rclpy_module
        rclpy.init(args=None)
        scene = IsaacFrankaScene(simulation_app)
        bridge = ActionStreamIsaacBridge(
            scene,
            command_timeout_seconds=args.command_timeout_seconds,
            defer_initial_observation=policy_config is not None,
        )
        if policy_config is not None:
            policy_runtime = InKitPolicyRuntime(bridge.node, policy_config)
            discovery_deadline = time.monotonic() + args.startup_discovery_seconds
            while (
                simulation_app.is_running()
                and rclpy.ok()
                and time.monotonic() < discovery_deadline
            ):
                policy_runtime.spin_once(timeout_sec=0.01)
        if video_config is not None:
            video_capture = IsaacViewportVideoCapture(simulation_app, video_config)
        bridge.publish_auto_start(args.auto_start_episode)
        bootstrap_deadline: float | None = None
        terminal_drain_deadline: float | None = None
        while simulation_app.is_running() and rclpy.ok():
            if policy_runtime is None:
                rclpy.spin_once(bridge.node, timeout_sec=0.01)
            else:
                policy_runtime.spin_once(timeout_sec=0.01)
            bridge.process_control_requests()
            if policy_runtime is not None and bridge.initial_observation_is_deferred:
                if bootstrap_deadline is None:
                    bootstrap_deadline = time.monotonic() + args.bootstrap_timeout_seconds
                if policy_runtime.bootstrap_ready:
                    bridge.release_initial_observation()
                    bootstrap_deadline = None
                elif time.monotonic() >= bootstrap_deadline:
                    raise RuntimeError(
                        "Executor did not accept the delayed bootstrap chunk within "
                        f"{args.bootstrap_timeout_seconds:.3f}s; trace="
                        f"{policy_config.trace_sha256 if policy_config else 'unavailable'}"
                    )
            stepped = bridge.step_if_ready()
            if stepped and video_capture is not None:
                video_capture.capture_frame(simulation_app)
            if (
                policy_runtime is not None
                and policy_runtime.complete
                and terminal_drain_deadline is None
            ):
                terminal_drain_deadline = time.monotonic() + args.terminal_drain_seconds
                bridge.node.get_logger().info(
                    "policy_episode_complete "
                    f"episode={args.auto_start_episode!r} "
                    f"step={policy_runtime.terminal_step} "
                    f"success={policy_runtime.terminal_success} "
                    f"requests={policy_runtime.request_count} "
                    f"trace={policy_config.trace_sha256 if policy_config else 'unavailable'}"
                )
            if terminal_drain_deadline is not None and time.monotonic() >= terminal_drain_deadline:
                break
            if (
                policy_runtime is None
                and args.exit_after_auto_episode
                and bridge.episode_is_terminal(args.auto_start_episode)
            ):
                break
            if bridge.command_wait_timed_out():
                raise RuntimeError(
                    "No RobotCommand arrived for the next observation step within "
                    f"{args.command_timeout_seconds:.3f}s; refusing to fabricate a hold "
                    "or substitute the deterministic test plant"
                )
        if video_capture is not None:
            output_path = video_capture.finish(simulation_app)
            print(f"Isaac viewport capture complete: {output_path}")
        exit_code = 0
    except Exception as exc:
        print(f"FATAL: Isaac adapter stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        exit_code = 1
    finally:
        if video_capture is not None:
            video_capture.cancel()
        if policy_runtime is not None:
            policy_runtime.close()
        if bridge is not None:
            bridge.close()
        if scene is not None:
            scene.close()
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()
        if simulation_app is not None:
            # Isaac's fast-shutdown path terminates the process from close().
            # Passing the result is the supported way to preserve a nonzero
            # adapter failure status instead of replacing it with zero.
            simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
