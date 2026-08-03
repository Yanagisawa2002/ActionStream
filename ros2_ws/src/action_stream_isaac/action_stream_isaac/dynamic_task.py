"""Pure M8-G0 dynamic-target pick-and-place task semantics.

The module intentionally has no ROS 2 or Isaac Sim imports.  It is the single
source of truth for deterministic scenario generation, task-state encoding,
phase transitions, success, and failure taxonomy.  Native Isaac code supplies
measured poses, velocities, contact/grasp state, and safety flags.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Final, Sequence


MILESTONE: Final = "M8-G0"
TASK_ID: Final = "dynamic_target_pick_place_v1"
CONTROL_FREQUENCY_HZ: Final = 20
ACTION_HORIZON: Final = 30
ACTION_DIMENSION: Final = 7
REQUEST_INTERVAL_STEPS: Final = 10
INITIAL_GENERATION_ID: Final = 1
DISTURBED_GENERATION_ID: Final = 2

PHASE_APPROACH_ABOVE: Final = 0
PHASE_DESCEND: Final = 1
PHASE_CLOSE: Final = 2
PHASE_LIFT: Final = 3
PHASE_TRANSPORT: Final = 4
PHASE_LOWER: Final = 5
PHASE_RELEASE: Final = 6
PHASE_RETRACT: Final = 7
PHASE_STABILIZE: Final = 8
PHASE_SUCCESS: Final = 9
PHASE_TERMINATED: Final = 10

PHASE_NAMES: Final = {
    PHASE_APPROACH_ABOVE: "approach_above",
    PHASE_DESCEND: "descend",
    PHASE_CLOSE: "close",
    PHASE_LIFT: "lift",
    PHASE_TRANSPORT: "transport",
    PHASE_LOWER: "lower",
    PHASE_RELEASE: "release",
    PHASE_RETRACT: "retract",
    PHASE_STABILIZE: "stabilize",
    PHASE_SUCCESS: "success",
    PHASE_TERMINATED: "terminated",
}

FAILURE_FAILED_APPROACH: Final = "failed_approach"
FAILURE_GRASP_MISS: Final = "grasp_miss"
FAILURE_UNSTABLE_GRASP: Final = "unstable_grasp"
FAILURE_DROPPED_OBJECT: Final = "dropped_object"
FAILURE_MOVED_TOWARD_OBSOLETE: Final = "moved_toward_obsolete_destination"
FAILURE_RELEASED_AT_OBSOLETE: Final = "released_at_obsolete_destination"
FAILURE_RECOVERY_TIMEOUT: Final = "destination_switch_recovery_timeout"
FAILURE_SWITCH_PRECONDITION_MISSED: Final = "switch_precondition_missed"
FAILURE_COLLISION: Final = "collision"
FAILURE_JOINT_OR_WORKSPACE_LIMIT: Final = "joint_or_workspace_limit"
FAILURE_TASK_TIMEOUT: Final = "task_timeout"
FAILURE_INVALID_COMMAND: Final = "invalid_command"
FAILURE_SIMULATOR_OR_TRANSPORT: Final = "simulator_or_transport_failure"

FAILURE_REASONS: Final = (
    FAILURE_FAILED_APPROACH,
    FAILURE_GRASP_MISS,
    FAILURE_UNSTABLE_GRASP,
    FAILURE_DROPPED_OBJECT,
    FAILURE_MOVED_TOWARD_OBSOLETE,
    FAILURE_RELEASED_AT_OBSOLETE,
    FAILURE_RECOVERY_TIMEOUT,
    FAILURE_SWITCH_PRECONDITION_MISSED,
    FAILURE_COLLISION,
    FAILURE_JOINT_OR_WORKSPACE_LIMIT,
    FAILURE_TASK_TIMEOUT,
    FAILURE_INVALID_COMMAND,
    FAILURE_SIMULATOR_OR_TRANSPORT,
)

# Observation.task_state layout.  Boolean values are encoded as exactly 0/1.
DYNAMIC_TASK_STATE_LAYOUT: Final = (
    "object_position_xyz[0:3]",
    "object_orientation_wxyz[3:7]",
    "object_linear_velocity_xyz[7:10]",
    "object_angular_velocity_xyz[10:13]",
    "original_destination_xyz[13:16]",
    "final_destination_xyz[16:19]",
    "active_destination_xyz[19:22]",
    "active_destination_id[22]",
    "disturbance_switched[23]",
    "generation_id[24]",
    "phase_code[25]",
    "phase_step[26]",
    "grasped[27]",
    "gripper_aperture_m[28]",
    "initial_object_z[29]",
    "lift_height_m[30]",
    "final_destination_xy_error_m[31]",
    "obsolete_destination_xy_error_m[32]",
    "success_streak_steps[33]",
    "correct_destination_placement[34]",
    "obsolete_destination_placement[35]",
    "collision[36]",
    "joint_or_workspace_limit[37]",
    "grasp_ever[38]",
    "success[39]",
    "terminated[40]",
    "episode_step[41]",
    "switch_step[42]",
    "max_disallowed_contact_force_n[43]",
)
DYNAMIC_TASK_STATE_SIZE: Final = 44


def _finite_vector(name: str, values: Sequence[float], length: int) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if len(converted) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(converted)}")
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    left = _finite_vector("left position", a, 3)
    right = _finite_vector("right position", b, 3)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(left, right, strict=True)))


def _xy_distance(a: Sequence[float], b: Sequence[float]) -> float:
    left = _finite_vector("left position", a, 3)
    right = _finite_vector("right position", b, 3)
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _norm(values: Sequence[float]) -> float:
    vector = tuple(float(value) for value in values)
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("norm input must be a non-empty finite vector")
    return math.sqrt(sum(value * value for value in vector))


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be exactly bool")
    return value


def _strict_binary(name: str, value: float) -> bool:
    if value not in {0.0, 1.0}:
        raise ValueError(f"dynamic task_state {name} must be exactly 0 or 1")
    return value == 1.0


def _nonnegative_integer(name: str, value: float) -> int:
    converted = int(value)
    if float(converted) != value or converted < 0:
        raise ValueError(f"dynamic task_state {name} must be a non-negative integer")
    return converted


@dataclass(frozen=True, slots=True)
class DynamicTaskConfig:
    """Frozen physical/task contract used before profile calibration."""

    object_x_range_m: tuple[float, float] = (0.42, 0.48)
    object_y_range_m: tuple[float, float] = (-0.04, 0.04)
    cube_side_m: float = 0.0515
    zone_a_xy_m: tuple[float, float] = (0.48, -0.22)
    zone_b_xy_m: tuple[float, float] = (0.48, 0.22)
    switch_steps: tuple[int, ...] = (90, 100, 110)
    max_steps: int = 320
    placement_tolerance_m: float = 0.04
    placement_height_tolerance_m: float = 0.035
    stable_placement_steps: int = 20
    stable_linear_speed_mps: float = 0.03
    stable_angular_speed_rps: float = 0.50
    collision_threshold_n: float = 40.0
    approach_height_m: float = 0.20
    grasp_hand_offset_m: float = 0.10
    carry_hand_height_m: float = 0.32
    position_reached_tolerance_m: float = 0.035
    grasp_close_min_steps: int = 5
    grasp_close_timeout_steps: int = 30
    approach_timeout_steps: int = 55
    descend_timeout_steps: int = 45
    lift_timeout_steps: int = 60
    lower_timeout_steps: int = 60
    release_min_steps: int = 5
    release_timeout_steps: int = 30
    retract_timeout_steps: int = 45
    lost_grasp_failure_steps: int = 5
    lift_clearance_m: float = 0.10
    dropped_object_lift_m: float = 0.02
    gripper_open_aperture_m: float = 0.07
    workspace_radius_m: float = 1.25
    floor_failure_z_m: float = -0.02

    @property
    def object_center_z_m(self) -> float:
        return 0.5 * self.cube_side_m

    def validate(self) -> None:
        for name, pair in (
            ("object_x_range_m", self.object_x_range_m),
            ("object_y_range_m", self.object_y_range_m),
        ):
            values = _finite_vector(name, pair, 2)
            if values[0] > values[1]:
                raise ValueError(f"{name} lower bound exceeds upper bound")
        _finite_vector("zone_a_xy_m", self.zone_a_xy_m, 2)
        _finite_vector("zone_b_xy_m", self.zone_b_xy_m, 2)
        if not self.switch_steps or any(
            step <= 0 or step % REQUEST_INTERVAL_STEPS != 0 for step in self.switch_steps
        ):
            raise ValueError("switch_steps must be positive request-boundary steps")
        if max(self.switch_steps) >= self.max_steps:
            raise ValueError("switch_steps must precede max_steps")
        positive_numbers = (
            self.cube_side_m,
            self.placement_tolerance_m,
            self.placement_height_tolerance_m,
            self.stable_linear_speed_mps,
            self.stable_angular_speed_rps,
            self.collision_threshold_n,
            self.approach_height_m,
            self.grasp_hand_offset_m,
            self.carry_hand_height_m,
            self.position_reached_tolerance_m,
            self.lift_clearance_m,
            self.workspace_radius_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_numbers):
            raise ValueError("task distance/rate thresholds must be finite and positive")
        counters = (
            self.max_steps,
            self.stable_placement_steps,
            self.grasp_close_min_steps,
            self.grasp_close_timeout_steps,
            self.approach_timeout_steps,
            self.descend_timeout_steps,
            self.lift_timeout_steps,
            self.lower_timeout_steps,
            self.release_min_steps,
            self.release_timeout_steps,
            self.retract_timeout_steps,
            self.lost_grasp_failure_steps,
        )
        if any(value <= 0 for value in counters):
            raise ValueError("task step thresholds must be positive")
        if self.grasp_close_min_steps >= self.grasp_close_timeout_steps:
            raise ValueError("grasp close minimum must precede its timeout")
        if self.release_min_steps >= self.release_timeout_steps:
            raise ValueError("release minimum must precede its timeout")
        if not math.isfinite(self.floor_failure_z_m):
            raise ValueError("floor_failure_z_m must be finite")


@dataclass(frozen=True, slots=True)
class DynamicScenario:
    seed: int
    object_xyz: tuple[float, float, float]
    object_wxyz: tuple[float, float, float, float]
    zone_a_xyz: tuple[float, float, float]
    zone_b_xyz: tuple[float, float, float]
    original_destination_xyz: tuple[float, float, float]
    final_destination_xyz: tuple[float, float, float]
    original_zone_label: str
    final_zone_label: str
    switch_step: int


def scenario_payload(scenario: DynamicScenario) -> dict[str, Any]:
    """Return the complete JSON-safe physical scenario identity.

    This is the canonical payload shared by the native reset and benchmark
    trace.  Binding both destination labels and both physical zone poses makes
    a paired-run hash sensitive to a direction or scene-layout change.
    """

    if scenario.seed < 0 or scenario.switch_step <= 0:
        raise ValueError("scenario seed and switch_step must be non-negative/positive")
    if {scenario.original_zone_label, scenario.final_zone_label} != {"A", "B"}:
        raise ValueError("scenario destinations must be a permutation of zones A and B")
    fields = {
        "object_position_xyz": _finite_vector("object_xyz", scenario.object_xyz, 3),
        "object_orientation_wxyz": _finite_vector(
            "object_wxyz", scenario.object_wxyz, 4
        ),
        "zone_a_xyz": _finite_vector("zone_a_xyz", scenario.zone_a_xyz, 3),
        "zone_b_xyz": _finite_vector("zone_b_xyz", scenario.zone_b_xyz, 3),
        "original_destination_xyz": _finite_vector(
            "original_destination_xyz", scenario.original_destination_xyz, 3
        ),
        "final_destination_xyz": _finite_vector(
            "final_destination_xyz", scenario.final_destination_xyz, 3
        ),
    }
    if fields["original_destination_xyz"] == fields["final_destination_xyz"]:
        raise ValueError("scenario destination switch must change the physical target")
    expected_by_label = {"A": fields["zone_a_xyz"], "B": fields["zone_b_xyz"]}
    if fields["original_destination_xyz"] != expected_by_label[scenario.original_zone_label]:
        raise ValueError("original destination label and pose disagree")
    if fields["final_destination_xyz"] != expected_by_label[scenario.final_zone_label]:
        raise ValueError("final destination label and pose disagree")
    return {
        "schema_version": 1,
        "milestone": MILESTONE,
        "task_id": TASK_ID,
        "disturbance_type": "destination_switch",
        "seed": int(scenario.seed),
        **{name: list(value) for name, value in fields.items()},
        "original_zone_label": scenario.original_zone_label,
        "final_zone_label": scenario.final_zone_label,
        "original_destination_id": 0,
        "final_destination_id": 1,
        "switch_step": int(scenario.switch_step),
        "initial_generation_id": INITIAL_GENERATION_ID,
        "disturbed_generation_id": DISTURBED_GENERATION_ID,
    }


def scenario_sha256(scenario: DynamicScenario) -> str:
    """Hash :func:`scenario_payload` using sorted compact UTF-8 JSON."""

    encoded = json.dumps(
        scenario_payload(scenario), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest_unit(digest: bytes, offset: int) -> float:
    integer = int.from_bytes(digest[offset : offset + 8], "big")
    return integer / float((1 << 64) - 1)


def scenario_for_seed(
    seed: int, config: DynamicTaskConfig | None = None
) -> DynamicScenario:
    """Map a non-negative seed to one immutable paired Isaac scenario."""

    task = config or DynamicTaskConfig()
    task.validate()
    if seed < 0:
        raise ValueError("seed must be non-negative")
    digest = hashlib.sha256(f"{MILESTONE}:{seed}".encode("ascii")).digest()
    x_fraction = _digest_unit(digest, 0)
    y_fraction = _digest_unit(digest, 8)
    x = task.object_x_range_m[0] + x_fraction * (
        task.object_x_range_m[1] - task.object_x_range_m[0]
    )
    y = task.object_y_range_m[0] + y_fraction * (
        task.object_y_range_m[1] - task.object_y_range_m[0]
    )
    z = task.object_center_z_m
    zone_a = (task.zone_a_xy_m[0], task.zone_a_xy_m[1], z)
    zone_b = (task.zone_b_xy_m[0], task.zone_b_xy_m[1], z)
    # Seed-controlled direction avoids a one-direction-only recovery result.
    if seed % 2 == 0:
        original, final = zone_a, zone_b
        original_label, final_label = "A", "B"
    else:
        original, final = zone_b, zone_a
        original_label, final_label = "B", "A"
    switch_index = int.from_bytes(digest[16:20], "big") % len(task.switch_steps)
    return DynamicScenario(
        seed=seed,
        object_xyz=(x, y, z),
        object_wxyz=(1.0, 0.0, 0.0, 0.0),
        zone_a_xyz=zone_a,
        zone_b_xyz=zone_b,
        original_destination_xyz=original,
        final_destination_xyz=final,
        original_zone_label=original_label,
        final_zone_label=final_label,
        switch_step=task.switch_steps[switch_index],
    )


@dataclass(frozen=True, slots=True)
class DynamicTaskMeasurement:
    episode_step: int
    end_effector_xyz: tuple[float, float, float]
    object_xyz: tuple[float, float, float]
    object_wxyz: tuple[float, float, float, float]
    object_linear_velocity_xyz: tuple[float, float, float]
    object_angular_velocity_xyz: tuple[float, float, float]
    gripper_aperture_m: float
    physically_grasped: bool
    collision: bool = False
    joint_or_workspace_limit: bool = False
    invalid_command: bool = False
    max_disallowed_contact_force_n: float = 0.0

    def validated(self) -> DynamicTaskMeasurement:
        if self.episode_step < 0:
            raise ValueError("episode_step must be non-negative")
        for name, values, length in (
            ("end_effector_xyz", self.end_effector_xyz, 3),
            ("object_xyz", self.object_xyz, 3),
            ("object_wxyz", self.object_wxyz, 4),
            ("object_linear_velocity_xyz", self.object_linear_velocity_xyz, 3),
            ("object_angular_velocity_xyz", self.object_angular_velocity_xyz, 3),
        ):
            _finite_vector(name, values, length)
        if not math.isfinite(self.gripper_aperture_m) or self.gripper_aperture_m < 0.0:
            raise ValueError("gripper_aperture_m must be finite and non-negative")
        if (
            not math.isfinite(self.max_disallowed_contact_force_n)
            or self.max_disallowed_contact_force_n < 0.0
        ):
            raise ValueError("max_disallowed_contact_force_n must be finite and non-negative")
        for name in (
            "physically_grasped",
            "collision",
            "joint_or_workspace_limit",
            "invalid_command",
        ):
            _require_bool(name, getattr(self, name))
        return self


@dataclass(frozen=True, slots=True)
class DynamicTaskEvaluation:
    phase_code: int
    phase: str
    previous_phase_code: int
    phase_changed: bool
    phase_step: int
    disturbance_switched: bool
    destination_switched_now: bool
    switch_precondition_met: bool
    generation_id: int
    active_destination_id: int
    active_destination_xyz: tuple[float, float, float]
    lift_height_m: float
    final_destination_xy_error_m: float
    obsolete_destination_xy_error_m: float
    grasped: bool
    grasp_ever: bool
    correct_destination_placement: bool
    obsolete_destination_placement: bool
    success_streak_steps: int
    success: bool
    terminated: bool
    termination_reason: str


class DynamicTaskMachine:
    """Deterministic observation-driven physical task state machine."""

    def __init__(
        self, scenario: DynamicScenario, config: DynamicTaskConfig | None = None
    ) -> None:
        self.scenario = scenario
        self.config = config or DynamicTaskConfig()
        self.config.validate()
        self.phase_code = PHASE_APPROACH_ABOVE
        self.phase_step = 0
        self.disturbance_switched = False
        self.generation_id = INITIAL_GENERATION_ID
        self.grasp_ever = False
        self.success_streak_steps = 0
        self._obsolete_streak_steps = 0
        self._lost_grasp_steps = 0
        self._last_episode_step = -1
        self._terminated = False
        self._success = False
        self._termination_reason = ""

    @property
    def active_destination_xyz(self) -> tuple[float, float, float]:
        if self.disturbance_switched:
            return self.scenario.final_destination_xyz
        return self.scenario.original_destination_xyz

    def _transition(self, phase_code: int) -> None:
        if phase_code not in PHASE_NAMES:
            raise ValueError(f"unknown phase code: {phase_code}")
        self.phase_code = phase_code
        self.phase_step = 0

    def _terminate(self, reason: str, *, success: bool = False) -> None:
        if success:
            self.phase_code = PHASE_SUCCESS
            self._termination_reason = "correct_destination_stable_placement"
        else:
            if reason not in FAILURE_REASONS:
                raise ValueError(f"unknown failure reason: {reason}")
            self.phase_code = PHASE_TERMINATED
            self._termination_reason = reason
        self.phase_step = 0
        self._terminated = True
        self._success = success

    def update(self, measurement: DynamicTaskMeasurement) -> DynamicTaskEvaluation:
        item = measurement.validated()
        if item.episode_step != self._last_episode_step + 1:
            raise ValueError(
                "episode steps must be contiguous: "
                f"previous={self._last_episode_step}, current={item.episode_step}"
            )
        self._last_episode_step = item.episode_step
        previous_phase = self.phase_code
        switched_now = False
        obj = item.object_xyz
        eef = item.end_effector_xyz
        lift_height = obj[2] - self.scenario.object_xyz[2]
        final_error = _xy_distance(obj, self.scenario.final_destination_xyz)
        obsolete_error = _xy_distance(obj, self.scenario.original_destination_xyz)
        linear_speed = _norm(item.object_linear_velocity_xyz)
        angular_speed = _norm(item.object_angular_velocity_xyz)
        released = item.gripper_aperture_m >= self.config.gripper_open_aperture_m
        correct_placement = (
            released
            and final_error <= self.config.placement_tolerance_m
            and abs(obj[2] - self.scenario.final_destination_xyz[2])
            <= self.config.placement_height_tolerance_m
        )
        obsolete_placement = (
            released
            and obsolete_error <= self.config.placement_tolerance_m
            and abs(obj[2] - self.scenario.original_destination_xyz[2])
            <= self.config.placement_height_tolerance_m
        )
        stable = (
            linear_speed <= self.config.stable_linear_speed_mps
            and angular_speed <= self.config.stable_angular_speed_rps
        )

        if not self._terminated:
            self.grasp_ever = self.grasp_ever or item.physically_grasped
            switch_precondition = self.grasp_ever and self.phase_code in {
                PHASE_LIFT,
                PHASE_TRANSPORT,
                PHASE_LOWER,
            }
            if (
                not self.disturbance_switched
                and item.episode_step >= self.scenario.switch_step
            ):
                self.disturbance_switched = True
                self.generation_id = DISTURBED_GENERATION_ID
                switched_now = True
            else:
                switch_precondition = False
            if self.grasp_ever and self.phase_code in {
                PHASE_LIFT,
                PHASE_TRANSPORT,
                PHASE_LOWER,
            }:
                self._lost_grasp_steps = (
                    0 if item.physically_grasped else self._lost_grasp_steps + 1
                )
            else:
                self._lost_grasp_steps = 0

            object_radius = math.hypot(obj[0], obj[1])
            if item.invalid_command:
                self._terminate(FAILURE_INVALID_COMMAND)
            elif item.collision:
                self._terminate(FAILURE_COLLISION)
            elif item.joint_or_workspace_limit:
                self._terminate(FAILURE_JOINT_OR_WORKSPACE_LIMIT)
            elif obj[2] < self.config.floor_failure_z_m or object_radius > self.config.workspace_radius_m:
                self._terminate(FAILURE_JOINT_OR_WORKSPACE_LIMIT)
            elif switched_now and not switch_precondition:
                self._terminate(FAILURE_SWITCH_PRECONDITION_MISSED)
            elif (
                self._lost_grasp_steps >= self.config.lost_grasp_failure_steps
                and lift_height >= self.config.dropped_object_lift_m
            ):
                self._terminate(FAILURE_DROPPED_OBJECT)
            elif self.phase_code == PHASE_APPROACH_ABOVE:
                goal = (obj[0], obj[1], obj[2] + self.config.approach_height_m)
                if _distance(eef, goal) <= self.config.position_reached_tolerance_m:
                    self._transition(PHASE_DESCEND)
                elif self.phase_step >= self.config.approach_timeout_steps:
                    self._terminate(FAILURE_FAILED_APPROACH)
            elif self.phase_code == PHASE_DESCEND:
                goal = (obj[0], obj[1], obj[2] + self.config.grasp_hand_offset_m)
                if _distance(eef, goal) <= self.config.position_reached_tolerance_m:
                    self._transition(PHASE_CLOSE)
                elif self.phase_step >= self.config.descend_timeout_steps:
                    self._terminate(FAILURE_FAILED_APPROACH)
            elif self.phase_code == PHASE_CLOSE:
                if item.physically_grasped and self.phase_step >= self.config.grasp_close_min_steps:
                    self._transition(PHASE_LIFT)
                elif self.phase_step >= self.config.grasp_close_timeout_steps:
                    self._terminate(FAILURE_GRASP_MISS)
            elif self.phase_code == PHASE_LIFT:
                if item.physically_grasped and lift_height >= self.config.lift_clearance_m:
                    self._transition(PHASE_TRANSPORT)
                elif self.phase_step >= self.config.lift_timeout_steps:
                    self._terminate(FAILURE_UNSTABLE_GRASP)
            elif self.phase_code == PHASE_TRANSPORT:
                carry_goal = (
                    self.active_destination_xyz[0],
                    self.active_destination_xyz[1],
                    self.config.carry_hand_height_m,
                )
                if (
                    self.disturbance_switched
                    and item.physically_grasped
                    and _distance(eef, carry_goal)
                    <= self.config.position_reached_tolerance_m
                ):
                    self._transition(PHASE_LOWER)
            elif self.phase_code == PHASE_LOWER:
                lower_goal = (
                    self.active_destination_xyz[0],
                    self.active_destination_xyz[1],
                    self.active_destination_xyz[2] + self.config.grasp_hand_offset_m,
                )
                if _distance(eef, lower_goal) <= self.config.position_reached_tolerance_m:
                    self._transition(PHASE_RELEASE)
                elif self.phase_step >= self.config.lower_timeout_steps:
                    self._terminate(FAILURE_RECOVERY_TIMEOUT)
            elif self.phase_code == PHASE_RELEASE:
                if released and self.phase_step >= self.config.release_min_steps:
                    self._transition(PHASE_RETRACT)
                elif self.phase_step >= self.config.release_timeout_steps:
                    self._terminate(FAILURE_UNSTABLE_GRASP)
            elif self.phase_code == PHASE_RETRACT:
                retract_goal = (
                    self.active_destination_xyz[0],
                    self.active_destination_xyz[1],
                    self.active_destination_xyz[2] + self.config.approach_height_m,
                )
                if _distance(eef, retract_goal) <= self.config.position_reached_tolerance_m:
                    self._transition(PHASE_STABILIZE)
                elif self.phase_step >= self.config.retract_timeout_steps:
                    self._terminate(FAILURE_RECOVERY_TIMEOUT)
            elif self.phase_code == PHASE_STABILIZE:
                self.success_streak_steps = (
                    self.success_streak_steps + 1 if correct_placement and stable else 0
                )
                self._obsolete_streak_steps = (
                    self._obsolete_streak_steps + 1 if obsolete_placement and stable else 0
                )
                if self.success_streak_steps >= self.config.stable_placement_steps:
                    self._terminate("", success=True)
                elif self._obsolete_streak_steps >= self.config.stable_placement_steps:
                    self._terminate(FAILURE_RELEASED_AT_OBSOLETE)

            if not self._terminated and item.episode_step >= self.config.max_steps:
                if obsolete_placement:
                    self._terminate(FAILURE_RELEASED_AT_OBSOLETE)
                elif self.disturbance_switched and self.grasp_ever:
                    self._terminate(FAILURE_RECOVERY_TIMEOUT)
                elif not self.grasp_ever and self.phase_code in {
                    PHASE_CLOSE,
                    PHASE_LIFT,
                }:
                    self._terminate(FAILURE_GRASP_MISS)
                elif not self.grasp_ever:
                    self._terminate(FAILURE_FAILED_APPROACH)
                else:
                    self._terminate(FAILURE_TASK_TIMEOUT)
        else:
            switch_precondition = False

        if self.phase_code == previous_phase and not self._terminated:
            self.phase_step += 1
        phase_changed = self.phase_code != previous_phase
        return DynamicTaskEvaluation(
            phase_code=self.phase_code,
            phase=PHASE_NAMES[self.phase_code],
            previous_phase_code=previous_phase,
            phase_changed=phase_changed,
            phase_step=self.phase_step,
            disturbance_switched=self.disturbance_switched,
            destination_switched_now=switched_now,
            switch_precondition_met=switch_precondition,
            generation_id=self.generation_id,
            active_destination_id=1 if self.disturbance_switched else 0,
            active_destination_xyz=self.active_destination_xyz,
            lift_height_m=lift_height,
            final_destination_xy_error_m=final_error,
            obsolete_destination_xy_error_m=obsolete_error,
            grasped=item.physically_grasped,
            grasp_ever=self.grasp_ever,
            correct_destination_placement=correct_placement,
            obsolete_destination_placement=obsolete_placement,
            success_streak_steps=self.success_streak_steps,
            success=self._success,
            terminated=self._terminated,
            termination_reason=self._termination_reason,
        )


def pack_dynamic_task_state(
    *,
    scenario: DynamicScenario,
    measurement: DynamicTaskMeasurement,
    evaluation: DynamicTaskEvaluation,
) -> tuple[float, ...]:
    """Encode the stable 44-value task observation contract."""

    item = measurement.validated()
    for name in (
        "disturbance_switched",
        "destination_switched_now",
        "switch_precondition_met",
        "grasped",
        "grasp_ever",
        "correct_destination_placement",
        "obsolete_destination_placement",
        "success",
        "terminated",
    ):
        _require_bool(name, getattr(evaluation, name))
    state = (
        *_finite_vector("object_xyz", item.object_xyz, 3),
        *_finite_vector("object_wxyz", item.object_wxyz, 4),
        *_finite_vector("object_linear_velocity_xyz", item.object_linear_velocity_xyz, 3),
        *_finite_vector("object_angular_velocity_xyz", item.object_angular_velocity_xyz, 3),
        *scenario.original_destination_xyz,
        *scenario.final_destination_xyz,
        *evaluation.active_destination_xyz,
        float(evaluation.active_destination_id),
        1.0 if evaluation.disturbance_switched else 0.0,
        float(evaluation.generation_id),
        float(evaluation.phase_code),
        float(evaluation.phase_step),
        1.0 if evaluation.grasped else 0.0,
        float(item.gripper_aperture_m),
        float(scenario.object_xyz[2]),
        float(evaluation.lift_height_m),
        float(evaluation.final_destination_xy_error_m),
        float(evaluation.obsolete_destination_xy_error_m),
        float(evaluation.success_streak_steps),
        1.0 if evaluation.correct_destination_placement else 0.0,
        1.0 if evaluation.obsolete_destination_placement else 0.0,
        1.0 if item.collision else 0.0,
        1.0 if item.joint_or_workspace_limit else 0.0,
        1.0 if evaluation.grasp_ever else 0.0,
        1.0 if evaluation.success else 0.0,
        1.0 if evaluation.terminated else 0.0,
        float(item.episode_step),
        float(scenario.switch_step),
        float(item.max_disallowed_contact_force_n),
    )
    if len(state) != DYNAMIC_TASK_STATE_SIZE or not all(math.isfinite(v) for v in state):
        raise RuntimeError("internal dynamic task-state packing error")
    return state


def unpack_dynamic_task_state(values: Sequence[float]) -> dict[str, object]:
    """Validate and expose named fields for tests, replay, and audits."""

    state = _finite_vector("dynamic task_state", values, DYNAMIC_TASK_STATE_SIZE)
    generation = int(state[24])
    phase = int(state[25])
    if float(generation) != state[24] or generation not in {
        INITIAL_GENERATION_ID,
        DISTURBED_GENERATION_ID,
    }:
        raise ValueError("dynamic task_state contains an invalid generation")
    if float(phase) != state[25] or phase not in PHASE_NAMES:
        raise ValueError("dynamic task_state contains an invalid phase")
    active_destination_id = _strict_binary("active_destination_id", state[22])
    disturbance_switched = _strict_binary("disturbance_switched", state[23])
    boolean_fields = {
        "grasped": _strict_binary("grasped", state[27]),
        "correct_destination_placement": _strict_binary(
            "correct_destination_placement", state[34]
        ),
        "obsolete_destination_placement": _strict_binary(
            "obsolete_destination_placement", state[35]
        ),
        "collision": _strict_binary("collision", state[36]),
        "joint_or_workspace_limit": _strict_binary(
            "joint_or_workspace_limit", state[37]
        ),
        "grasp_ever": _strict_binary("grasp_ever", state[38]),
        "success": _strict_binary("success", state[39]),
        "terminated": _strict_binary("terminated", state[40]),
    }
    phase_step = _nonnegative_integer("phase_step", state[26])
    success_streak_steps = _nonnegative_integer("success_streak_steps", state[33])
    episode_step = _nonnegative_integer("episode_step", state[41])
    switch_step = _nonnegative_integer("switch_step", state[42])
    if active_destination_id != disturbance_switched:
        raise ValueError("active_destination_id must match disturbance_switched")
    if disturbance_switched != (generation == DISTURBED_GENERATION_ID):
        raise ValueError("disturbance state and generation are inconsistent")
    return {
        "object_xyz": state[0:3],
        "object_wxyz": state[3:7],
        "object_linear_velocity_xyz": state[7:10],
        "object_angular_velocity_xyz": state[10:13],
        "original_destination_xyz": state[13:16],
        "final_destination_xyz": state[16:19],
        "active_destination_xyz": state[19:22],
        "active_destination_id": int(active_destination_id),
        "disturbance_switched": disturbance_switched,
        "generation_id": generation,
        "phase_code": phase,
        "phase_step": phase_step,
        "grasped": boolean_fields["grasped"],
        "gripper_aperture_m": state[28],
        "initial_object_z": state[29],
        "lift_height_m": state[30],
        "final_destination_xy_error_m": state[31],
        "obsolete_destination_xy_error_m": state[32],
        "success_streak_steps": success_streak_steps,
        "correct_destination_placement": boolean_fields[
            "correct_destination_placement"
        ],
        "obsolete_destination_placement": boolean_fields[
            "obsolete_destination_placement"
        ],
        "collision": boolean_fields["collision"],
        "joint_or_workspace_limit": boolean_fields["joint_or_workspace_limit"],
        "grasp_ever": boolean_fields["grasp_ever"],
        "success": boolean_fields["success"],
        "terminated": boolean_fields["terminated"],
        "episode_step": episode_step,
        "switch_step": switch_step,
        "max_disallowed_contact_force_n": state[43],
    }
