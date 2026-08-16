"""Persistent native-Isaac M8 dynamic-recovery batch adapter.

The module is safe to import without ROS 2 or Isaac Sim.  ``--validate-only``
performs pure manifest/configuration validation and is explicitly ineligible as
native evidence.  A real run starts exactly one SimulationApp, keeps its
physics clock monotonic, resets the same official Franka scene for every
episode, and recreates only the per-episode policy/fault/recorder/request graph.
There is no deterministic-plant fallback.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable, Final, Mapping, Sequence

from action_stream_policy.dynamic_pick_place import DynamicPolicyConfig

from .capability_probe import probe
from .dynamic_request_driver import (
    DynamicRequestDriverConfig,
    create_dynamic_request_driver_node,
)
from .dynamic_task import (
    ACTION_HORIZON,
    CONTROL_FREQUENCY_HZ,
    DYNAMIC_TASK_STATE_LAYOUT,
    DYNAMIC_TASK_STATE_SIZE,
    FAILURE_REASONS,
    INITIAL_GENERATION_ID,
    MILESTONE,
    PHASE_NAMES,
    TASK_ID,
    DynamicScenario,
    DynamicTaskConfig,
    DynamicTaskEvaluation,
    DynamicTaskMachine,
    DynamicTaskMeasurement,
    pack_dynamic_task_state,
    scenario_for_seed,
    scenario_payload,
    scenario_sha256,
)
from .ros_contract import (
    BOOTSTRAP_OBSERVATION_TOPIC,
    CLOCK_TOPIC,
    EPISODE_CONTROL_TOPIC,
    OBSERVATION_TOPIC,
    ROBOT_COMMAND_TOPIC,
    load_and_validate_message_types,
    set_ros_time,
)
from .task_logic import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    axis_angle_to_quaternion_wxyz,
    gripper_command_to_finger_positions,
    pack_robot_state,
    quaternion_wxyz_to_axis_angle,
    split_sim_time,
    validate_robot_command_for_step,
)

PHYSICS_FREQUENCY_HZ: Final = 60
PHYSICS_STEPS_PER_CONTROL: Final = PHYSICS_FREQUENCY_HZ // CONTROL_FREQUENCY_HZ
PHYSICS_DT_SECONDS: Final = 1.0 / PHYSICS_FREQUENCY_HZ
CONTROL_DT_SECONDS: Final = 1.0 / CONTROL_FREQUENCY_HZ
ROS_CALLBACK_DRAIN_LIMIT: Final = 64

FRANKA_PATH: Final = "/World/Franka"
OBJECT_PATH: Final = "/World/DynamicObject"
ZONE_A_PATH: Final = "/World/ZoneA"
ZONE_B_PATH: Final = "/World/ZoneB"
ACTIVE_TARGET_PATH: Final = "/World/ActiveDestination"
GROUND_PATH: Final = "/World/GroundPlane"
GROUND_COLLISION_PATH: Final = f"{GROUND_PATH}/collisionPlane"
DISALLOWED_CONTACT_FILTER_PATHS: Final = (OBJECT_PATH, GROUND_COLLISION_PATH)
LEFT_FINGER_PATH: Final = f"{FRANKA_PATH}/panda_leftfinger"
RIGHT_FINGER_PATH: Final = f"{FRANKA_PATH}/panda_rightfinger"
DISALLOWED_LINK_PATHS: Final = tuple(
    f"{FRANKA_PATH}/panda_link{index}" for index in range(1, 8)
) + (f"{FRANKA_PATH}/panda_hand",)
HOME_JOINT_POSITIONS: Final = (
    0.012,
    -0.568,
    0.0,
    -2.811,
    0.0,
    3.037,
    0.741,
    0.04,
    0.04,
)
STRATEGIES: Final = ("sync_hold", "naive_async", "aligned_async")
SPLITS: Final = ("baseline_gate", "development", "frozen_holdout")
PROFILE_STRATEGIES: Final = {
    "profile_0_sanity": ("sync_hold",),
    "profile_1_fixed": STRATEGIES,
    "profile_2_faults": STRATEGIES,
}
NATIVE_EVIDENCE_CLASS: Final = "ros_cpp_isaac_sim"
VALIDATION_EVIDENCE_CLASS: Final = "config_validation_only"

_TASK_PROTOCOL_KEYS: Final = frozenset(
    {
        "task_id",
        "primary_disturbance",
        "object_x_range_m",
        "object_y_range_m",
        "cube_side_m",
        "zone_a_xy_m",
        "zone_b_xy_m",
        "switch_steps",
        "placement_tolerance_m",
        "placement_height_tolerance_m",
        "stable_placement_steps",
        "stable_linear_speed_mps",
        "stable_angular_speed_rps",
        "lift_clearance_m",
        "collision_threshold_n",
        "success",
        "failure_reasons",
    }
)
_CONTROLLER_PROTOCOL_KEYS: Final = frozenset(
    {
        "type",
        "approach_height_m",
        "grasp_hand_offset_m",
        "carry_hand_height_m",
        "recovery_hover_offset_m",
        "maximum_translation_per_step_m",
        "maximum_translation_scope",
        "workspace_x_m",
        "workspace_y_m",
        "workspace_z_m",
        "downward_axis_angle_xyz",
    }
)
_CALIBRATABLE_TASK_FIELDS: Final = frozenset(
    {
        "object_x_range_m",
        "object_y_range_m",
        "zone_a_xy_m",
        "zone_b_xy_m",
        "switch_steps",
        "lift_clearance_m",
    }
)
_CALIBRATABLE_CONTROLLER_FIELDS: Final = frozenset(
    {
        "approach_height_m",
        "grasp_hand_offset_m",
        "carry_hand_height_m",
        "recovery_hover_offset_m",
        "maximum_translation_per_step_m",
        "workspace_x_m",
        "workspace_y_m",
        "workspace_z_m",
    }
)
_FROZEN_NONCALIBRATABLE_TASK_FIELDS: Final[Mapping[str, Any]] = {
    "position_reached_tolerance_m": 0.035,
    "grasp_close_min_steps": 5,
    "grasp_close_timeout_steps": 30,
    "approach_timeout_steps": 55,
    "descend_timeout_steps": 45,
    "lift_timeout_steps": 60,
    "lower_timeout_steps": 60,
    "release_min_steps": 5,
    "release_timeout_steps": 30,
    "retract_timeout_steps": 45,
    "lost_grasp_failure_steps": 5,
    "dropped_object_lift_m": 0.02,
    "gripper_open_aperture_m": 0.07,
    "workspace_radius_m": 1.25,
    "floor_failure_z_m": -0.02,
}


def _exact_protocol_section(
    protocol: Mapping[str, Any], name: str, expected_keys: frozenset[str]
) -> Mapping[str, Any]:
    value = protocol.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"protocol.{name} must be an object")
    actual = set(value)
    if actual != expected_keys:
        raise ValueError(
            f"protocol.{name} field drift: "
            f"missing={sorted(expected_keys - actual)}, unknown={sorted(actual - expected_keys)}"
        )
    return value


def _protocol_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _protocol_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a JSON integer")
    return value


def _protocol_float_tuple(value: Any, *, name: str, length: int) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise ValueError(f"{name} must contain exactly {length} JSON numbers")
    return tuple(
        _protocol_float(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _protocol_int_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{name} must be a non-empty integer array")
    return tuple(
        _protocol_int(item, name=f"{name}[{index}]") for index, item in enumerate(value)
    )


def _validate_runtime_calibration_bounds(
    task_config: DynamicTaskConfig,
    policy_config: DynamicPolicyConfig,
) -> None:
    """Repeat the public candidate bounds at the native runtime boundary."""

    workspace_x = policy_config.workspace_x_m
    workspace_y = policy_config.workspace_y_m
    workspace_z = policy_config.workspace_z_m
    if not 0.15 <= workspace_x[0] < workspace_x[1] <= 0.85:
        raise ValueError("controller.workspace_x_m is outside the bounded workspace")
    if not -0.50 <= workspace_y[0] < workspace_y[1] <= 0.50:
        raise ValueError("controller.workspace_y_m is outside the bounded workspace")
    if not 0.02 <= workspace_z[0] < workspace_z[1] <= 0.80:
        raise ValueError("controller.workspace_z_m is outside the bounded workspace")

    bounded_policy_scalars = {
        "approach_height_m": (0.05, workspace_z[1] - workspace_z[0]),
        "grasp_hand_offset_m": (0.02, 0.20),
        "carry_hand_height_m": workspace_z,
        "recovery_hover_offset_m": (0.02, 0.25),
        "maximum_translation_per_step_m": (0.001, 0.02),
    }
    for name, (lower, upper) in bounded_policy_scalars.items():
        value = getattr(policy_config, name)
        if not lower <= value <= upper:
            raise ValueError(
                f"controller.{name} is outside the bounded calibration range"
            )
    if not 0.05 <= task_config.lift_clearance_m <= 0.25:
        raise ValueError(
            "task.lift_clearance_m is outside the bounded calibration range"
        )
    if len(set(task_config.switch_steps)) != len(task_config.switch_steps):
        raise ValueError("task.switch_steps must contain unique request boundaries")


def runtime_configs_from_protocol(
    protocol_payload: Mapping[str, Any],
) -> tuple[DynamicTaskConfig, DynamicPolicyConfig]:
    """Construct every native task/policy value from one validated protocol.

    Calibratable fields must be explicit in the protocol. Internal task-machine
    guards are deliberately non-calibratable and are repeated here so a changed
    dataclass default or a newly added field fails closed before Isaac starts.
    """

    task = _exact_protocol_section(protocol_payload, "task", _TASK_PROTOCOL_KEYS)
    controller = _exact_protocol_section(
        protocol_payload, "controller", _CONTROLLER_PROTOCOL_KEYS
    )
    runtime = protocol_payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("protocol.runtime must be an object")
    if (
        task["task_id"] != TASK_ID
        or task["primary_disturbance"] != "destination_switch"
    ):
        raise ValueError(
            "protocol task identity does not match the native dynamic task"
        )
    failure_reasons = task["failure_reasons"]
    if (
        not isinstance(failure_reasons, Sequence)
        or isinstance(failure_reasons, (str, bytes))
        or tuple(failure_reasons) != FAILURE_REASONS
    ):
        raise ValueError(
            "protocol failure taxonomy differs from the native dynamic task"
        )
    if not isinstance(task["success"], str) or not task["success"].strip():
        raise ValueError("protocol.task.success must be a non-empty contract string")
    if controller["type"] != "deterministic_observation_conditioned_cartesian_waypoint":
        raise ValueError("protocol controller type differs from the native policy")
    if controller["maximum_translation_scope"] != (
        "adjacent rows in the planned chunk; expired-prefix selection is measured "
        "separately at execution"
    ):
        raise ValueError("protocol controller translation scope drifted")

    policy_values: dict[str, Any] = {
        "approach_height_m": _protocol_float(
            controller["approach_height_m"], name="controller.approach_height_m"
        ),
        "grasp_hand_offset_m": _protocol_float(
            controller["grasp_hand_offset_m"], name="controller.grasp_hand_offset_m"
        ),
        "carry_hand_height_m": _protocol_float(
            controller["carry_hand_height_m"], name="controller.carry_hand_height_m"
        ),
        "recovery_hover_offset_m": _protocol_float(
            controller["recovery_hover_offset_m"],
            name="controller.recovery_hover_offset_m",
        ),
        "maximum_translation_per_step_m": _protocol_float(
            controller["maximum_translation_per_step_m"],
            name="controller.maximum_translation_per_step_m",
        ),
        "workspace_x_m": _protocol_float_tuple(
            controller["workspace_x_m"], name="controller.workspace_x_m", length=2
        ),
        "workspace_y_m": _protocol_float_tuple(
            controller["workspace_y_m"], name="controller.workspace_y_m", length=2
        ),
        "workspace_z_m": _protocol_float_tuple(
            controller["workspace_z_m"], name="controller.workspace_z_m", length=2
        ),
        "downward_axis_angle_xyz": _protocol_float_tuple(
            controller["downward_axis_angle_xyz"],
            name="controller.downward_axis_angle_xyz",
            length=3,
        ),
    }
    policy_defaults = asdict(DynamicPolicyConfig())
    if set(policy_defaults) != set(policy_values):
        raise RuntimeError(
            "DynamicPolicyConfig field set drifted from the protocol-owned runtime contract"
        )
    policy_config = DynamicPolicyConfig(**policy_values)
    policy_config.validate()

    task_values: dict[str, Any] = {
        "object_x_range_m": _protocol_float_tuple(
            task["object_x_range_m"], name="task.object_x_range_m", length=2
        ),
        "object_y_range_m": _protocol_float_tuple(
            task["object_y_range_m"], name="task.object_y_range_m", length=2
        ),
        "cube_side_m": _protocol_float(task["cube_side_m"], name="task.cube_side_m"),
        "zone_a_xy_m": _protocol_float_tuple(
            task["zone_a_xy_m"], name="task.zone_a_xy_m", length=2
        ),
        "zone_b_xy_m": _protocol_float_tuple(
            task["zone_b_xy_m"], name="task.zone_b_xy_m", length=2
        ),
        "switch_steps": _protocol_int_tuple(
            task["switch_steps"], name="task.switch_steps"
        ),
        "max_steps": _protocol_int(
            runtime.get("maximum_episode_steps"), name="runtime.maximum_episode_steps"
        ),
        "placement_tolerance_m": _protocol_float(
            task["placement_tolerance_m"], name="task.placement_tolerance_m"
        ),
        "placement_height_tolerance_m": _protocol_float(
            task["placement_height_tolerance_m"],
            name="task.placement_height_tolerance_m",
        ),
        "stable_placement_steps": _protocol_int(
            task["stable_placement_steps"], name="task.stable_placement_steps"
        ),
        "stable_linear_speed_mps": _protocol_float(
            task["stable_linear_speed_mps"], name="task.stable_linear_speed_mps"
        ),
        "stable_angular_speed_rps": _protocol_float(
            task["stable_angular_speed_rps"], name="task.stable_angular_speed_rps"
        ),
        "collision_threshold_n": _protocol_float(
            task["collision_threshold_n"], name="task.collision_threshold_n"
        ),
        "approach_height_m": policy_config.approach_height_m,
        "grasp_hand_offset_m": policy_config.grasp_hand_offset_m,
        "carry_hand_height_m": policy_config.carry_hand_height_m,
        "lift_clearance_m": _protocol_float(
            task["lift_clearance_m"], name="task.lift_clearance_m"
        ),
        **_FROZEN_NONCALIBRATABLE_TASK_FIELDS,
    }
    task_defaults = asdict(DynamicTaskConfig())
    if set(task_defaults) != set(task_values):
        raise RuntimeError(
            "DynamicTaskConfig field set drifted from the explicit runtime contract"
        )
    fixed_drift = {
        name: {"expected": expected, "actual": task_defaults.get(name)}
        for name, expected in _FROZEN_NONCALIBRATABLE_TASK_FIELDS.items()
        if task_defaults.get(name) != expected
    }
    if fixed_drift:
        raise RuntimeError(
            f"non-calibratable DynamicTaskConfig default drift: {fixed_drift}"
        )
    task_config = DynamicTaskConfig(**task_values)
    task_config.validate()

    for name, bounds in (
        ("object_x_range_m", policy_config.workspace_x_m),
        ("object_y_range_m", policy_config.workspace_y_m),
    ):
        values = getattr(task_config, name)
        if (
            values[0] >= values[1]
            or not bounds[0] <= values[0] < values[1] <= bounds[1]
        ):
            raise ValueError(
                f"task.{name} must be strictly inside the policy workspace"
            )
    for name in ("zone_a_xy_m", "zone_b_xy_m"):
        point = getattr(task_config, name)
        if not (
            policy_config.workspace_x_m[0] <= point[0] <= policy_config.workspace_x_m[1]
            and policy_config.workspace_y_m[0]
            <= point[1]
            <= policy_config.workspace_y_m[1]
        ):
            raise ValueError(f"task.{name} must be inside the policy workspace")
    if task_config.zone_a_xy_m == task_config.zone_b_xy_m:
        raise ValueError("native destination zones must remain distinct")
    _validate_runtime_calibration_bounds(task_config, policy_config)
    return task_config, policy_config


def _policy_parameter_values(config: DynamicPolicyConfig) -> dict[str, object]:
    """Return ROS-safe parameter values for the exact validated policy config."""

    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in asdict(config).items()
    }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_path(base: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _portable_reference(path: Path, *, base: Path) -> str:
    """Return one forward-slash relative reference for recorded metadata."""

    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {name} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one JSON object: {path}")
    return payload


def _validate_embedded_hash(
    payload: Mapping[str, Any], *, field: str, name: str
) -> str:
    embedded = payload.get(field)
    if not isinstance(embedded, str) or len(embedded) != 64:
        raise ValueError(f"{name} has no valid {field}")
    core = dict(payload)
    core.pop(field, None)
    actual = _canonical_sha256(core)
    if embedded != actual:
        raise ValueError(
            f"{name} {field} mismatch: embedded={embedded}, actual={actual}"
        )
    return embedded


@dataclass(frozen=True, slots=True)
class MatrixEpisodeSpec:
    episode_id: str
    seed: int
    strategy: str
    profile_id: str
    fault_trace_file: Path
    event_log_path: Path
    summary_path: Path
    scenario_sha256: str
    trace_sha256: str


@dataclass(frozen=True, slots=True)
class DynamicMatrixManifest:
    source_path: Path
    split: str
    episodes: tuple[MatrixEpisodeSpec, ...]
    provenance_manifest: Path
    provenance_sha256: str
    protocol_path: Path
    protocol_payload: Mapping[str, Any]
    protocol_sha256: str
    task_config: DynamicTaskConfig
    policy_config: DynamicPolicyConfig
    freeze_sha256: str | None
    frozen_before_first_holdout_result: bool
    holdout_freeze_status: str
    persistent_isaac_process_required: bool

    @property
    def strategy(self) -> str:
        return self.episodes[0].strategy

    @property
    def profile_id(self) -> str:
        return self.episodes[0].profile_id


def load_matrix_manifest(path: Path | str) -> DynamicMatrixManifest:
    """Load and strictly validate a persistent native-run matrix manifest."""

    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"matrix manifest does not exist: {source}")
    payload = _json_object(source, name="matrix manifest")
    if payload.get("schema_version") != 1 or payload.get("milestone") != MILESTONE:
        raise ValueError(
            "matrix manifest must use schema_version=1 and milestone=M8-G0"
        )
    split = str(payload.get("split", ""))
    if split not in SPLITS:
        raise ValueError(f"matrix manifest split must be one of {SPLITS}")
    persistent = payload.get("persistent_isaac_process_required")
    if persistent is not True:
        raise ValueError("persistent_isaac_process_required must be exactly true")
    frozen = payload.get("frozen_before_first_holdout_result")
    if type(frozen) is not bool:
        raise ValueError("frozen_before_first_holdout_result must be exactly bool")
    status = str(payload.get("holdout_freeze_status", ""))
    base = source.parent

    matrix_protocol_sha = payload.get("protocol_sha256")
    if not isinstance(matrix_protocol_sha, str) or len(matrix_protocol_sha) != 64:
        raise ValueError("matrix manifest protocol_sha256 is invalid")
    freeze_sha256: str | None = None
    resolved_frozen_inputs: dict[str, Path] = {}
    if split == "frozen_holdout":
        if not frozen or status != "frozen":
            raise ValueError(
                "frozen_holdout requires frozen_before_first_holdout_result=true "
                "and holdout_freeze_status=frozen"
            )
        provenance = _resolved_path(
            base, payload.get("freeze_manifest"), name="freeze_manifest"
        )
        if not provenance.is_file():
            raise ValueError(f"freeze manifest does not exist: {provenance}")
        provenance_payload = _json_object(provenance, name="freeze manifest")
        provenance_sha = _validate_embedded_hash(
            provenance_payload, field="freeze_sha256", name="freeze manifest"
        )
        freeze_sha256 = provenance_sha
        if payload.get("freeze_sha256") != provenance_sha:
            raise ValueError("matrix freeze_sha256 does not match freeze manifest")
        frozen_inputs = provenance_payload.get("inputs")
        if not isinstance(frozen_inputs, list) or not frozen_inputs:
            raise ValueError("freeze manifest has no frozen inputs")
        for item in frozen_inputs:
            if not isinstance(item, Mapping):
                raise ValueError("freeze manifest input must be an object")
            role = str(item.get("role", ""))
            path_value = str(item.get("path", ""))
            sha256 = str(item.get("sha256", ""))
            size_bytes = item.get("size_bytes")
            if (
                not role
                or role in resolved_frozen_inputs
                or not path_value
                or len(sha256) != 64
                or type(size_bytes) is not int
                or size_bytes < 0
            ):
                raise ValueError(
                    "freeze manifest input identity is invalid or duplicated"
                )
            relative = Path(path_value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"freeze input path is not repository-relative: {path_value}"
                )
            candidates = (
                (provenance.parent / relative).resolve(),
                *((candidate / relative).resolve() for candidate in source.parents),
            )
            resolved = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.is_file()
                    and candidate.stat().st_size == size_bytes
                    and _sha256_file(candidate) == sha256
                ),
                None,
            )
            if resolved is None:
                raise ValueError(f"live frozen input is missing or drifted: {role}")
            resolved_frozen_inputs[role] = resolved
        protocol_record = next(
            (
                item
                for item in frozen_inputs
                if isinstance(item, Mapping) and item.get("role") == "protocol"
            ),
            None,
        )
        if protocol_record is None or not protocol_record.get("path"):
            raise ValueError("freeze manifest does not bind a protocol input")
        protocol_path = resolved_frozen_inputs["protocol"]
    else:
        if frozen or status != "pending":
            raise ValueError(
                "baseline/development requires frozen_before_first_holdout_result=false "
                "and holdout_freeze_status=pending"
            )
        provenance = _resolved_path(
            base, payload.get("candidate_manifest"), name="candidate_manifest"
        )
        if not provenance.is_file():
            raise ValueError(f"candidate manifest does not exist: {provenance}")
        provenance_payload = _json_object(provenance, name="candidate manifest")
        provenance_sha = _canonical_sha256(provenance_payload)
        protocol_path = provenance

    from action_stream_benchmark.m8_protocol import (
        CALIBRATABLE_CONTRACT_PATHS,
        load_protocol,
    )
    from action_stream_benchmark.schema import canonical_sha256

    protocol_payload = load_protocol(protocol_path)
    protocol_sha = canonical_sha256(protocol_payload)
    if protocol_sha != matrix_protocol_sha:
        raise ValueError(
            "matrix protocol_sha256 does not match the validated protocol: "
            f"matrix={matrix_protocol_sha}, actual={protocol_sha}"
        )
    declared_runtime_calibration_paths = {
        path
        for path in CALIBRATABLE_CONTRACT_PATHS
        if path.startswith(("/protocol/task/", "/protocol/controller/"))
    }
    bound_runtime_calibration_paths = {
        *(f"/protocol/task/{name}" for name in _CALIBRATABLE_TASK_FIELDS),
        *(f"/protocol/controller/{name}" for name in _CALIBRATABLE_CONTROLLER_FIELDS),
    }
    if declared_runtime_calibration_paths != bound_runtime_calibration_paths:
        raise RuntimeError(
            "native runtime calibratable-field binding drift: "
            f"declared={sorted(declared_runtime_calibration_paths)}, "
            f"bound={sorted(bound_runtime_calibration_paths)}"
        )
    task_config, policy_config = runtime_configs_from_protocol(protocol_payload)

    from action_stream_benchmark.m8_faults import (
        declared_fault_profile,
        fault_profile_binding_mismatches,
        fault_trace_from_mapping,
    )

    repository_root = next(
        (
            candidate
            for candidate in (source.parent, *source.parents)
            if (candidate / ".git").exists()
        ),
        None,
    )
    if repository_root is None:
        raise ValueError("matrix manifest is not inside an ActionStream repository")
    raw_profile_records = payload.get("profiles")
    if not isinstance(raw_profile_records, Mapping) or not raw_profile_records:
        raise ValueError("matrix manifest has no declared profile records")
    protocol_profiles = protocol_payload.get("profiles")
    if not isinstance(protocol_profiles, Mapping):
        raise ValueError("validated protocol has no profile declarations")
    declared_profiles: dict[str, Any] = {}
    for raw_profile_id, raw_record in raw_profile_records.items():
        profile_id = str(raw_profile_id)
        expected_strategies = PROFILE_STRATEGIES.get(profile_id)
        if expected_strategies is None:
            raise ValueError(f"matrix declares an unknown fault profile: {profile_id}")
        if not isinstance(raw_record, Mapping) or set(raw_record) != {"path", "sha256"}:
            raise ValueError(f"matrix profile record field drift: {profile_id}")
        profile_path_value = raw_record.get("path")
        if not isinstance(profile_path_value, str) or not profile_path_value:
            raise ValueError(f"matrix profile path is invalid: {profile_id}")
        if Path(profile_path_value).is_absolute():
            raise ValueError(f"matrix profile path must be portable: {profile_id}")
        profile_path = (base / profile_path_value).resolve()
        try:
            profile_path.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(
                f"matrix profile escapes the repository: {profile_id}"
            ) from exc
        if not profile_path.is_file():
            raise ValueError(f"declared matrix profile does not exist: {profile_path}")
        profile_sha = _sha256_file(profile_path)
        if raw_record.get("sha256") != profile_sha:
            raise ValueError(f"matrix profile byte hash mismatch: {profile_id}")

        protocol_record = protocol_profiles.get(profile_id)
        if not isinstance(protocol_record, Mapping) or set(protocol_record) != {
            "strategies",
            "profile_file",
            "profile_sha256",
        }:
            raise ValueError(f"protocol profile record field drift: {profile_id}")
        if tuple(protocol_record.get("strategies", ())) != expected_strategies:
            raise ValueError(f"protocol profile strategy set drift: {profile_id}")
        protocol_relative = Path(str(protocol_record.get("profile_file", "")))
        if (
            not protocol_record.get("profile_file")
            or protocol_relative.is_absolute()
            or ".." in protocol_relative.parts
        ):
            raise ValueError(f"protocol profile path is invalid: {profile_id}")
        protocol_profile_path = (repository_root / protocol_relative).resolve()
        if (
            protocol_profile_path != profile_path
            or protocol_record.get("profile_sha256") != profile_sha
        ):
            raise ValueError(
                f"matrix profile bytes differ from the validated protocol: {profile_id}"
            )
        if split == "frozen_holdout":
            frozen_profile_path = resolved_frozen_inputs.get(f"profile:{profile_id}")
            if frozen_profile_path != profile_path:
                raise ValueError(
                    f"matrix profile does not match frozen role profile:{profile_id}"
                )
        declared_profiles[profile_id] = declared_fault_profile(
            _json_object(profile_path, name=f"declared profile {profile_id}"),
            expected_profile_id=profile_id,
            expected_strategies=expected_strategies,
        )

    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("matrix manifest episodes must be a non-empty array")
    episodes: list[MatrixEpisodeSpec] = []
    seen_ids: set[str] = set()
    seen_outputs: set[Path] = set()
    for index, raw in enumerate(raw_episodes):
        if not isinstance(raw, dict):
            raise ValueError(f"episodes[{index}] must be an object")
        episode_id = str(raw.get("episode_id", ""))
        if not episode_id or episode_id in seen_ids:
            raise ValueError(f"episodes[{index}] has empty or duplicate episode_id")
        seen_ids.add(episode_id)
        seed = raw.get("seed")
        if type(seed) is not int or seed < 0:
            raise ValueError(f"episodes[{index}].seed must be a non-negative integer")
        strategy = str(raw.get("strategy", ""))
        if strategy not in STRATEGIES:
            raise ValueError(f"episodes[{index}].strategy must be one of {STRATEGIES}")
        profile_id = str(raw.get("profile_id", ""))
        if not profile_id:
            raise ValueError(f"episodes[{index}].profile_id must be non-empty")
        trace_path = _resolved_path(
            base,
            raw.get("fault_trace_file"),
            name=f"episodes[{index}].fault_trace_file",
        )
        if not trace_path.is_file():
            raise ValueError(f"frozen fault trace does not exist: {trace_path}")
        trace_payload = _json_object(trace_path, name="fault trace")
        trace = fault_trace_from_mapping(trace_payload, source=str(trace_path))
        trace_sha = trace.sha256
        if raw.get("fault_trace_sha256") != trace_sha:
            raise ValueError(
                f"matrix fault_trace_sha256 mismatch for episode {episode_id}"
            )
        if trace.profile.profile_id != profile_id or trace.seed != seed:
            raise ValueError(
                f"fault trace profile/seed mismatch for episode {episode_id}"
            )
        declared_profile = declared_profiles.get(profile_id)
        if declared_profile is None:
            raise ValueError(f"episode uses an undeclared fault profile: {profile_id}")
        profile_mismatches = fault_profile_binding_mismatches(
            trace.profile,
            declared_profile,
        )
        if profile_mismatches:
            raise ValueError(
                f"fault trace profile differs from declared bytes for {episode_id}: "
                f"{profile_mismatches}"
            )
        if strategy not in PROFILE_STRATEGIES[profile_id]:
            raise ValueError(
                f"episode strategy is not declared by profile {profile_id}: {strategy}"
            )
        event_path = _resolved_path(
            base, raw.get("event_log_path"), name=f"episodes[{index}].event_log_path"
        )
        summary_path = _resolved_path(
            base, raw.get("summary_path"), name=f"episodes[{index}].summary_path"
        )
        for output in (event_path, summary_path):
            if output in seen_outputs:
                raise ValueError(f"duplicate matrix output path: {output}")
            if output.exists():
                raise ValueError(f"refusing to overwrite matrix output: {output}")
            seen_outputs.add(output)
        expected_scenario_sha = scenario_sha256(scenario_for_seed(seed, task_config))
        if raw.get("scenario_sha256") != expected_scenario_sha:
            raise ValueError(
                f"scenario_sha256 mismatch for {episode_id}: "
                f"expected {expected_scenario_sha}"
            )
        episodes.append(
            MatrixEpisodeSpec(
                episode_id=episode_id,
                seed=seed,
                strategy=strategy,
                profile_id=profile_id,
                fault_trace_file=trace_path,
                event_log_path=event_path,
                summary_path=summary_path,
                scenario_sha256=expected_scenario_sha,
                trace_sha256=trace_sha,
            )
        )
    if len({episode.strategy for episode in episodes}) != 1:
        raise ValueError("one persistent matrix must use exactly one executor strategy")
    if len({episode.profile_id for episode in episodes}) != 1:
        raise ValueError("one persistent matrix must use exactly one fault profile")
    episode_profile_ids = {episode.profile_id for episode in episodes}
    if set(declared_profiles) != episode_profile_ids:
        raise ValueError(
            "persistent matrix profile records differ from episode profiles: "
            f"declared={sorted(declared_profiles)}, episodes={sorted(episode_profile_ids)}"
        )
    if payload.get("batch_profile_id") != episodes[0].profile_id:
        raise ValueError(
            "batch_profile_id does not match the persistent episode profile"
        )
    if payload.get("batch_strategy") != episodes[0].strategy:
        raise ValueError(
            "batch_strategy does not match the persistent episode strategy"
        )
    if payload.get("expected_episode_count") != len(episodes):
        raise ValueError("expected_episode_count does not match persistent episodes")
    return DynamicMatrixManifest(
        source_path=source,
        split=split,
        episodes=tuple(episodes),
        provenance_manifest=provenance,
        provenance_sha256=provenance_sha,
        protocol_path=protocol_path,
        protocol_payload=protocol_payload,
        protocol_sha256=protocol_sha,
        task_config=task_config,
        policy_config=policy_config,
        freeze_sha256=freeze_sha256,
        frozen_before_first_holdout_result=frozen,
        holdout_freeze_status=status,
        persistent_isaac_process_required=True,
    )


def validation_report(manifest: DynamicMatrixManifest) -> dict[str, Any]:
    """Return a report that cannot be confused with executed Isaac evidence."""

    return {
        "schema_version": 1,
        "milestone": MILESTONE,
        "validation_passed": True,
        "native_execution_performed": False,
        "headline_evidence_eligible": False,
        "evidence_class": VALIDATION_EVIDENCE_CLASS,
        "manifest": str(manifest.source_path),
        "split": manifest.split,
        "strategy": manifest.strategy,
        "profile_id": manifest.profile_id,
        "episode_count": len(manifest.episodes),
        "persistent_isaac_process_required": True,
        "outputs_created": False,
        "runtime_config_binding_passed": True,
        "task_config": asdict(manifest.task_config),
        "policy_config": asdict(manifest.policy_config),
    }


def _flat_finite(value: Any, *, length: int, name: str) -> tuple[float, ...]:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()

    flattened: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            flattened.append(float(item))

    try:
        visit(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Isaac returned invalid {name}: {value!r}") from exc
    converted = tuple(flattened)
    if len(converted) != length or not all(math.isfinite(item) for item in converted):
        raise RuntimeError(
            f"Isaac returned invalid {name}: expected {length} finite values, "
            f"got {converted!r}"
        )
    return converted


def _joint_limit_vectors(
    value: Any, *, dof_count: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Decode the Isaac 6 experimental articulation limit contract.

    ``Articulation.get_dof_limits()`` returns ``(lower, upper)`` Warp arrays,
    not one interleaved ``[lower_0, upper_0, ...]`` matrix.  Decode the two
    arrays independently so a container of device arrays is never coerced to
    ``float`` and the lower/upper association remains explicit.
    """

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RuntimeError(
            "Isaac returned invalid DOF limits: expected separate lower/upper arrays"
        )
    lower = _flat_finite(value[0], length=dof_count, name="DOF lower limits")
    upper = _flat_finite(value[1], length=dof_count, name="DOF upper limits")
    if any(low >= high for low, high in zip(lower, upper, strict=True)):
        raise RuntimeError(
            "Isaac returned invalid DOF limits: lower must be below upper"
        )
    return lower, upper


def _maximum_vector_norm(value: Any, *, name: str) -> float:
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        import numpy as np

        array = np.asarray(value, dtype=np.float64)
        if array.size == 0:
            return 0.0
        if array.shape[-1] != 3 or not np.all(np.isfinite(array)):
            raise ValueError
        return float(np.max(np.linalg.norm(array.reshape((-1, 3)), axis=1)))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Isaac returned invalid {name}") from exc


@dataclass(frozen=True, slots=True)
class DynamicSceneMeasurement:
    end_effector_xyz: tuple[float, float, float]
    end_effector_wxyz: tuple[float, float, float, float]
    object_xyz: tuple[float, float, float]
    object_wxyz: tuple[float, float, float, float]
    object_linear_velocity_xyz: tuple[float, float, float]
    object_angular_velocity_xyz: tuple[float, float, float]
    gripper_aperture_m: float
    physically_grasped: bool
    collision: bool
    joint_or_workspace_limit: bool
    max_disallowed_contact_force_n: float


class DynamicIsaacScene:
    """Official Isaac 6 Franka scene with measured grasp/contact semantics."""

    def __init__(
        self,
        simulation_app: Any,
        *,
        task_config: DynamicTaskConfig,
        policy_config: DynamicPolicyConfig,
    ) -> None:
        # All Isaac imports intentionally occur only after SimulationApp starts.
        import numpy as np
        import omni.timeline
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.objects import Cube, GroundPlane
        from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager

        self.config = task_config
        self.config.validate()
        self.policy_config = policy_config
        self.policy_config.validate()
        app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples")
        simulation_app.update()
        from isaacsim.robot.experimental.manipulators.examples.franka import Franka

        required = (
            "set_end_effector_pose",
            "get_current_state",
            "get_downward_orientation",
            "reset_to_default_pose",
            "set_gripper_position",
            "get_dof_limits",
            "get_dof_positions",
            "get_dof_velocities",
        )
        missing = tuple(name for name in required if not hasattr(Franka, name))
        if missing:
            raise RuntimeError(
                f"Installed Isaac Franka experimental API is missing {missing!r}"
            )

        self._app = simulation_app
        self._np = np
        self._RenderingManager = RenderingManager
        self._SimulationManager = SimulationManager
        self._app_utils = app_utils
        self._timeline = omni.timeline.get_timeline_interface()

        stage_utils.create_new_stage()
        GroundPlane(GROUND_PATH, positions=[0.0, 0.0, 0.0])
        self._franka = Franka(robot_path=FRANKA_PATH, create_robot=True)
        self._articulation = self._franka

        cube = Cube(
            paths=OBJECT_PATH,
            positions=[0.45, 0.0, self.config.object_center_z_m],
            orientations=[1.0, 0.0, 0.0, 0.0],
            sizes=1.0,
            scales=[self.config.cube_side_m] * 3,
            colors="blue",
        )
        GeomPrim(paths=cube.paths, apply_collision_apis=True)
        self._object = RigidPrim(paths=cube.paths)

        marker_z = 0.003
        zone_scale = [2.0 * self.config.placement_tolerance_m] * 2 + [0.004]
        Cube(
            paths=ZONE_A_PATH,
            positions=[*self.config.zone_a_xy_m, marker_z],
            sizes=1.0,
            scales=zone_scale,
            colors="red",
        )
        Cube(
            paths=ZONE_B_PATH,
            positions=[*self.config.zone_b_xy_m, marker_z],
            sizes=1.0,
            scales=zone_scale,
            colors="green",
        )
        Cube(
            paths=ACTIVE_TARGET_PATH,
            positions=[*self.config.zone_a_xy_m, marker_z + 0.004],
            sizes=1.0,
            scales=[0.025, 0.025, 0.008],
            colors="yellow",
        )
        self._active_marker = XformPrim(ACTIVE_TARGET_PATH)

        # Resolve the official robot USD reference before wrapping its links.
        self._app.update()
        self._finger_contacts = RigidPrim(
            paths=[LEFT_FINGER_PATH, RIGHT_FINGER_PATH],
            contact_filter_paths=[OBJECT_PATH],
            max_contact_count=64,
        )
        self._disallowed_contacts = RigidPrim(
            paths=list(DISALLOWED_LINK_PATHS),
            # Isaac 6 GroundPlane is an Xform root; its collidable child is
            # ``collisionPlane``.  Filtering the root silently misses ground
            # contacts in the tensor contact view.
            contact_filter_paths=list(DISALLOWED_CONTACT_FILTER_PATHS),
            max_contact_count=256,
        )

        # ``contact_filter_paths`` configures the tensor contact view, but
        # Isaac Sim 6 does not implicitly apply PhysxContactReportAPI to the
        # monitored rigid bodies.  Enable reporting before physics starts so
        # the views can resolve the finger/link prims on the first ready event.
        self._finger_contacts.set_enabled_contact_tracking([True], threshold=0.0)
        self._disallowed_contacts.set_enabled_contact_tracking([True], threshold=0.0)

        SimulationManager.setup_simulation(dt=PHYSICS_DT_SECONDS, device="cpu")
        RenderingManager.set_dt(PHYSICS_DT_SECONDS)
        self._app.update()
        app_utils.play()
        self._app.update()
        self._app.update()
        self._timeline.pause()
        self._app.update()
        for _ in range(5):
            self._raw_step()

        self._joint_names = tuple(self._articulation.dof_names)
        if len(self._joint_names) != 9:
            raise RuntimeError(
                "Official Franka asset contract changed: expected 9 DOFs, got "
                f"{len(self._joint_names)} ({self._joint_names!r})"
            )
        self._finger_indices = (
            self._joint_names.index("panda_finger_joint1"),
            self._joint_names.index("panda_finger_joint2"),
        )
        self._target_xyz = (0.45, 0.0, 0.40)
        self._target_wxyz = tuple(
            float(x) for x in self._franka.get_downward_orientation()
        )
        self._gripper_command = GRIPPER_OPEN
        self._scenario = scenario_for_seed(0, self.config)
        self.reset(self._scenario)

    @property
    def gripper_command(self) -> float:
        return self._gripper_command

    @property
    def current_command(self) -> tuple[float, ...]:
        return (
            *self._target_xyz,
            *quaternion_wxyz_to_axis_angle(self._target_wxyz),
            self._gripper_command,
        )

    def _raw_step(self) -> None:
        self._SimulationManager.step()
        self._RenderingManager.render()
        self._app.update()

    def sim_time_seconds(self) -> float:
        steps = int(self._SimulationManager.get_num_physics_steps())
        return steps / float(PHYSICS_FREQUENCY_HZ)

    def reset(self, scenario: DynamicScenario) -> DynamicSceneMeasurement:
        """Reset in place while retaining the app, stage, and monotonic clock."""

        if scenario_sha256(scenario) != scenario_sha256(
            scenario_for_seed(scenario.seed, self.config)
        ):
            raise ValueError("scene reset scenario is not canonical for its seed")
        self._scenario = scenario
        self._franka.reset_to_default_pose()
        self._articulation.set_dof_positions(HOME_JOINT_POSITIONS)
        self._articulation.set_dof_velocities((0.0,) * len(HOME_JOINT_POSITIONS))
        self._articulation.set_dof_position_targets(HOME_JOINT_POSITIONS)
        self._object.set_world_poses(
            positions=scenario.object_xyz,
            orientations=scenario.object_wxyz,
        )
        self._object.set_velocities(
            linear_velocities=(0.0, 0.0, 0.0),
            angular_velocities=(0.0, 0.0, 0.0),
        )
        self.set_active_destination(scenario.original_destination_xyz)
        self._gripper_command = GRIPPER_OPEN
        for _ in range(8):
            self._raw_step()
        self._target_xyz, self._target_wxyz = self.end_effector_pose()
        self._apply_finger_target()
        return self.measure()

    def reset_state_payload(
        self,
        scenario: DynamicScenario,
        measurement: DynamicSceneMeasurement,
    ) -> dict[str, Any]:
        """Return the exact native reset facts bound into paired fairness."""

        if scenario != self._scenario:
            raise ValueError("reset-state scenario is not the scene's active scenario")
        joint_positions = _flat_finite(
            self._articulation.get_dof_positions(),
            length=9,
            name="reset DOF positions",
        )
        joint_velocities = _flat_finite(
            self._articulation.get_dof_velocities(),
            length=9,
            name="reset DOF velocities",
        )
        return {
            "seed": scenario.seed,
            "robot_joint_positions": list(joint_positions),
            "robot_joint_velocities": list(joint_velocities),
            "end_effector_position_xyz": list(measurement.end_effector_xyz),
            "end_effector_orientation_wxyz": list(measurement.end_effector_wxyz),
            "object_position_xyz": list(measurement.object_xyz),
            "object_orientation_wxyz": list(measurement.object_wxyz),
            "zone_a_xyz": list(scenario.zone_a_xyz),
            "zone_b_xyz": list(scenario.zone_b_xyz),
            "physics_dt_seconds": PHYSICS_DT_SECONDS,
            "rendering_dt_seconds": PHYSICS_DT_SECONDS,
            "stage_units_in_meters": 1.0,
            "gravity_xyz": [0.0, 0.0, -9.81],
        }

    def set_active_destination(self, destination_xyz: Sequence[float]) -> None:
        destination = tuple(float(value) for value in destination_xyz)
        if len(destination) != 3 or not all(
            math.isfinite(value) for value in destination
        ):
            raise ValueError("active destination must contain three finite values")
        self._active_marker.set_world_poses(
            positions=(destination[0], destination[1], 0.007),
            orientations=(1.0, 0.0, 0.0, 0.0),
        )

    def set_command(self, command: Sequence[float]) -> None:
        values = tuple(float(value) for value in command)
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise ValueError(
                "dynamic Cartesian command must contain seven finite values"
            )
        if values[6] not in {GRIPPER_OPEN, GRIPPER_CLOSED}:
            raise ValueError("dynamic gripper command must be exactly +1 or -1")
        for value, bounds, axis in zip(
            values[:3],
            (
                self.policy_config.workspace_x_m,
                self.policy_config.workspace_y_m,
                self.policy_config.workspace_z_m,
            ),
            "xyz",
            strict=True,
        ):
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(
                    f"dynamic Cartesian {axis} command {value} outside {bounds}"
                )
        self._target_xyz = values[:3]
        self._target_wxyz = axis_angle_to_quaternion_wxyz(values[3:6])
        self._gripper_command = values[6]

    def _apply_finger_target(self) -> None:
        fingers = gripper_command_to_finger_positions(self._gripper_command)
        self._franka.set_gripper_position(
            self._np.asarray(fingers, dtype=self._np.float64)
        )

    def step(self) -> None:
        self._franka.set_end_effector_pose(
            position=self._np.asarray(self._target_xyz, dtype=self._np.float64),
            orientation=self._np.asarray(self._target_wxyz, dtype=self._np.float64),
            ik_method="damped-least-squares",
        )
        self._apply_finger_target()
        self._raw_step()

    def end_effector_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        _dof_positions, positions, orientations = self._franka.get_current_state()
        return (
            _flat_finite(positions, length=3, name="end-effector position"),
            _flat_finite(orientations, length=4, name="end-effector orientation"),
        )

    def arm_joint_state(
        self,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return measured seven-DOF arm state for learned-policy observations."""

        positions = _flat_finite(
            self._articulation.get_dof_positions(), length=9, name="DOF positions"
        )
        velocities = _flat_finite(
            self._articulation.get_dof_velocities(), length=9, name="DOF velocities"
        )
        return positions[:7], velocities[:7]

    def _joint_limit_reached(self, eef_xyz: Sequence[float]) -> bool:
        positions = _flat_finite(
            self._articulation.get_dof_positions(), length=9, name="DOF positions"
        )
        lower_limits, upper_limits = _joint_limit_vectors(
            self._articulation.get_dof_limits(), dof_count=9
        )
        for index, value in enumerate(positions[:7]):
            lower, upper = lower_limits[index], upper_limits[index]
            if value <= lower + 1e-3 or value >= upper - 1e-3:
                return True
        return any(
            not bounds[0] <= value <= bounds[1]
            for value, bounds in zip(
                eef_xyz,
                (
                    self.policy_config.workspace_x_m,
                    self.policy_config.workspace_y_m,
                    self.policy_config.workspace_z_m,
                ),
                strict=True,
            )
        )

    def measure(self) -> DynamicSceneMeasurement:
        eef_xyz, eef_wxyz = self.end_effector_pose()
        object_positions, object_orientations = self._object.get_world_poses()
        linear, angular = self._object.get_velocities()
        object_xyz = _flat_finite(object_positions, length=3, name="object position")
        object_wxyz = _flat_finite(
            object_orientations, length=4, name="object orientation"
        )
        linear_xyz = _flat_finite(linear, length=3, name="object linear velocity")
        angular_xyz = _flat_finite(angular, length=3, name="object angular velocity")
        joints = _flat_finite(
            self._articulation.get_dof_positions(), length=9, name="DOF positions"
        )
        aperture = sum(joints[index] for index in self._finger_indices)
        finger_forces = self._finger_contacts.get_contact_force_matrix(
            dt=PHYSICS_DT_SECONDS
        )
        if hasattr(finger_forces, "numpy"):
            finger_forces = finger_forces.numpy()
        force_array = self._np.asarray(finger_forces, dtype=self._np.float64).reshape(
            (2, -1, 3)
        )
        each_finger_force = self._np.max(
            self._np.linalg.norm(force_array, axis=2), axis=1
        )
        xy_error = math.hypot(eef_xyz[0] - object_xyz[0], eef_xyz[1] - object_xyz[1])
        vertical_offset = eef_xyz[2] - object_xyz[2]
        contact_grasp = bool(self._np.all(each_finger_force >= 0.05))
        geometry_grasp = (
            xy_error <= 0.75 * self.config.cube_side_m
            and abs(vertical_offset - self.config.grasp_hand_offset_m) <= 0.055
            and aperture <= self.config.gripper_open_aperture_m
        )
        disallowed_force = _maximum_vector_norm(
            self._disallowed_contacts.get_contact_force_matrix(dt=PHYSICS_DT_SECONDS),
            name="disallowed contact force matrix",
        )
        return DynamicSceneMeasurement(
            end_effector_xyz=eef_xyz,
            end_effector_wxyz=eef_wxyz,
            object_xyz=object_xyz,
            object_wxyz=object_wxyz,
            object_linear_velocity_xyz=linear_xyz,
            object_angular_velocity_xyz=angular_xyz,
            gripper_aperture_m=aperture,
            physically_grasped=contact_grasp and geometry_grasp,
            collision=disallowed_force >= self.config.collision_threshold_n,
            joint_or_workspace_limit=self._joint_limit_reached(eef_xyz),
            max_disallowed_contact_force_n=disallowed_force,
        )

    def close(self) -> None:
        self._app_utils.stop()


@dataclass(frozen=True, slots=True)
class PendingDynamicCommand:
    actual_target_step: int
    command: tuple[float, ...]
    hold: bool
    invalid_command: bool
    source_request_id: int
    source_generation_id: int
    source_observation_step: int
    source_target_step: int
    reason: str


@dataclass(frozen=True, slots=True)
class DynamicControlRequest:
    command: int
    episode_id: str
    generation_id: int
    success: bool


@dataclass(frozen=True, slots=True)
class DynamicEpisodeResult:
    episode_id: str
    success: bool
    termination_reason: str
    terminal_step: int
    generation_id: int
    scenario_sha256: str
    maximum_disallowed_contact_force_n: float


def _command_motion_audit_payload(
    *,
    requested_command: Sequence[float],
    previous_scene_target: Sequence[float],
    applied_scene_target: Sequence[float],
    end_effector_before_xyz: Sequence[float],
    end_effector_after_xyz: Sequence[float],
    hold: bool,
    invalid_command: bool,
    planned_row_translation_limit_m: float,
) -> dict[str, Any]:
    """Describe target discontinuity and realized motion for one control step."""

    return {
        "requested_target_xyz": list(requested_command[:3]),
        "previous_scene_target_xyz": list(previous_scene_target[:3]),
        "applied_scene_target_xyz": list(applied_scene_target[:3]),
        "requested_target_delta_m": math.dist(
            requested_command[:3], previous_scene_target[:3]
        ),
        "applied_target_delta_m": math.dist(
            applied_scene_target[:3], previous_scene_target[:3]
        ),
        "measured_end_effector_delta_m": math.dist(
            end_effector_after_xyz, end_effector_before_xyz
        ),
        "hold": bool(hold),
        "invalid_command": bool(invalid_command),
        "planned_row_translation_limit_m": float(planned_row_translation_limit_m),
        "planned_row_limit_scope": "adjacent_planned_rows_only",
    }


class DynamicIsaacBridge:
    """Persistent rclpy bridge from the C++ executor to native Isaac physics."""

    def __init__(
        self,
        scene: DynamicIsaacScene,
        *,
        command_timeout_seconds: float,
        planned_row_translation_limit_m: float,
        defer_initial_observation: bool = True,
    ) -> None:
        import rclpy
        from action_stream_msgs.msg import RuntimeEvent
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
        self._RuntimeEvent = RuntimeEvent
        self.node = Node("action_stream_dynamic_isaac_adapter")
        self._command_timeout_seconds = command_timeout_seconds
        self._planned_row_translation_limit_m = planned_row_translation_limit_m
        self._defer_initial_observation = defer_initial_observation

        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=512,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        lifecycle = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=16,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
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
            self._types.EpisodeControl, EPISODE_CONTROL_TOPIC, lifecycle
        )
        self._event_publisher = self.node.create_publisher(
            RuntimeEvent, "/action_stream/events", reliable
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
            lifecycle,
        )

        self._control_requests: deque[DynamicControlRequest] = deque()
        self._pending_command: PendingDynamicCommand | None = None
        self._configured_episode: MatrixEpisodeSpec | None = None
        self._matrix: DynamicMatrixManifest | None = None
        self._active_episode_id: str | None = None
        self._scenario: DynamicScenario | None = None
        self._task_machine: DynamicTaskMachine | None = None
        self._last_measurement: DynamicTaskMeasurement | None = None
        self._last_evaluation: DynamicTaskEvaluation | None = None
        self._generation_id = INITIAL_GENERATION_ID
        self._observation_step = 0
        self._command_deadline_wall: float | None = None
        self._deferred_initial_observation: Any | None = None
        self._bootstrap_observation_published = False
        self._last_terminal_episode_id: str | None = None
        self._last_result: DynamicEpisodeResult | None = None
        self._maximum_contact_force_n = 0.0

    def configure_episode(
        self, spec: MatrixEpisodeSpec, matrix: DynamicMatrixManifest
    ) -> None:
        if self._active_episode_id is not None:
            raise RuntimeError("cannot reconfigure while a dynamic episode is active")
        if spec not in matrix.episodes:
            raise ValueError("episode specification is not part of the matrix")
        self._configured_episode = spec
        self._matrix = matrix
        self._last_result = None

    def _stamp(self, message: Any) -> None:
        sec, nanosec = split_sim_time(self._scene.sim_time_seconds())
        set_ros_time(message.sim_stamp, sec=sec, nanosec=nanosec)
        message.steady_time_ns = time.monotonic_ns()
        message.wall_time_ns = time.time_ns()

    def _publish_event(
        self,
        event_type: str,
        reason: str,
        detail: Mapping[str, Any],
        *,
        request_id: int = 0,
        generation_id: int | None = None,
        source_observation_step: int = 0,
        source_target_step: int = 0,
        actual_target_step: int = 0,
        action_count: int = 0,
    ) -> None:
        event = self._RuntimeEvent()
        self._stamp(event)
        event.event_type = event_type
        event.reason = reason
        event.strategy = (
            self._configured_episode.strategy if self._configured_episode else "adapter"
        )
        event.episode_id = self._active_episode_id or (
            self._configured_episode.episode_id if self._configured_episode else ""
        )
        event.request_id = request_id
        event.generation_id = (
            self._generation_id if generation_id is None else generation_id
        )
        event.active_generation_id = self._generation_id
        event.source_observation_step = source_observation_step
        event.source_target_step = source_target_step
        event.actual_target_step = actual_target_step
        event.action_count = action_count
        event.detail = json.dumps(detail, sort_keys=True, separators=(",", ":"))
        self._event_publisher.publish(event)

    def _on_episode_control(self, message: Any) -> None:
        if message.message_kind == self._types.EpisodeControl.STATUS:
            return
        if message.message_kind != self._types.EpisodeControl.REQUEST:
            self.node.get_logger().error(
                "episode_control_rejected reason=unknown_message_kind"
            )
            return
        if bool(message.active) or bool(message.terminated) or not message.episode_id:
            self.node.get_logger().error(
                "episode_control_rejected reason=invalid_request_status_or_episode"
            )
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
        self._control_requests.append(
            DynamicControlRequest(
                command=int(message.command),
                episode_id=str(message.episode_id),
                generation_id=int(message.generation_id),
                success=bool(message.success),
            )
        )

    def _on_robot_command(self, message: Any) -> None:
        pending_step = (
            self._pending_command.actual_target_step
            if self._pending_command is not None
            else None
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
            self.node.get_logger().warning(
                "robot_command_rejected "
                f"reason={validation.reason} episode={message.episode_id!r} "
                f"target={message.actual_target_step} observation={self._observation_step}"
            )
            return
        invalid_command = False
        command = validation.command
        if not bool(message.hold):
            try:
                # Validate without mutating; set_command is repeated at execution.
                previous = self._scene.current_command
                self._scene.set_command(command)
                self._scene.set_command(previous)
            except (TypeError, ValueError):
                invalid_command = True
                command = self._scene.current_command
        self._pending_command = PendingDynamicCommand(
            actual_target_step=int(message.actual_target_step),
            command=command,
            hold=bool(message.hold) or invalid_command,
            invalid_command=invalid_command,
            source_request_id=int(message.source_request_id),
            source_generation_id=int(message.source_generation_id),
            source_observation_step=int(message.source_observation_step),
            source_target_step=int(message.source_target_step),
            reason=str(message.reason),
        )
        self._command_deadline_wall = None

        scenario = self._scenario
        switched = bool(
            self._last_evaluation and self._last_evaluation.disturbance_switched
        )
        obsolete = False
        if scenario is not None and switched and not bool(message.hold):
            old_distance = math.dist(command[:3], scenario.original_destination_xyz)
            final_distance = math.dist(command[:3], scenario.final_destination_xyz)
            obsolete = old_distance < final_distance
        self._publish_event(
            "obsolete_command_classified",
            "geometry_distance_comparison",
            {
                "actual_target_step": int(message.actual_target_step),
                "source_request_id": int(message.source_request_id),
                "source_generation_id": int(message.source_generation_id),
                "command": list(command),
                "disturbance_switched": switched,
                "obsolete_destination_command": obsolete,
                "invalid_command": invalid_command,
            },
            request_id=int(message.source_request_id),
            generation_id=int(message.source_generation_id),
            source_observation_step=int(message.source_observation_step),
            source_target_step=int(message.source_target_step),
            actual_target_step=int(message.actual_target_step),
            action_count=1,
        )

    def publish_episode_request(self, *, reset: bool) -> None:
        spec = self._configured_episode
        if spec is None:
            raise RuntimeError("configure_episode must precede lifecycle request")
        message = self._types.EpisodeControl()
        message.message_kind = self._types.EpisodeControl.REQUEST
        message.command = (
            self._types.EpisodeControl.RESET
            if reset
            else self._types.EpisodeControl.START
        )
        self._stamp(message)
        message.episode_id = spec.episode_id
        message.generation_id = INITIAL_GENERATION_ID
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

    def _dynamic_measurement(self, *, invalid_command: bool) -> DynamicTaskMeasurement:
        measured = self._scene.measure()
        self._maximum_contact_force_n = max(
            self._maximum_contact_force_n,
            measured.max_disallowed_contact_force_n,
        )
        return DynamicTaskMeasurement(
            episode_step=self._observation_step,
            end_effector_xyz=measured.end_effector_xyz,
            object_xyz=measured.object_xyz,
            object_wxyz=measured.object_wxyz,
            object_linear_velocity_xyz=measured.object_linear_velocity_xyz,
            object_angular_velocity_xyz=measured.object_angular_velocity_xyz,
            gripper_aperture_m=measured.gripper_aperture_m,
            physically_grasped=measured.physically_grasped,
            collision=measured.collision,
            joint_or_workspace_limit=measured.joint_or_workspace_limit,
            invalid_command=invalid_command,
            max_disallowed_contact_force_n=measured.max_disallowed_contact_force_n,
        )

    def _publish_observation(
        self,
        *,
        invalid_command: bool = False,
        defer_for_bootstrap: bool = False,
    ) -> bool:
        if (
            self._active_episode_id is None
            or self._task_machine is None
            or self._scenario is None
        ):
            return False
        scene_measurement = self._scene.measure()
        item = DynamicTaskMeasurement(
            episode_step=self._observation_step,
            end_effector_xyz=scene_measurement.end_effector_xyz,
            object_xyz=scene_measurement.object_xyz,
            object_wxyz=scene_measurement.object_wxyz,
            object_linear_velocity_xyz=scene_measurement.object_linear_velocity_xyz,
            object_angular_velocity_xyz=scene_measurement.object_angular_velocity_xyz,
            gripper_aperture_m=scene_measurement.gripper_aperture_m,
            physically_grasped=scene_measurement.physically_grasped,
            collision=scene_measurement.collision,
            joint_or_workspace_limit=scene_measurement.joint_or_workspace_limit,
            invalid_command=invalid_command,
            max_disallowed_contact_force_n=scene_measurement.max_disallowed_contact_force_n,
        )
        self._maximum_contact_force_n = max(
            self._maximum_contact_force_n, item.max_disallowed_contact_force_n
        )
        evaluation = self._task_machine.update(item)
        self._last_measurement = item
        self._last_evaluation = evaluation
        self._generation_id = evaluation.generation_id

        if evaluation.destination_switched_now:
            self._scene.set_active_destination(evaluation.active_destination_xyz)
            self._publish_event(
                "destination_switched",
                "seeded_disturbance_boundary",
                {
                    "switch_step": self._observation_step,
                    "old_destination_id": 0,
                    "new_destination_id": 1,
                    "old_destination_xyz": list(
                        self._scenario.original_destination_xyz
                    ),
                    "new_destination_xyz": list(self._scenario.final_destination_xyz),
                    "generation_before": INITIAL_GENERATION_ID,
                    "generation_after": evaluation.generation_id,
                    "switch_precondition_met": evaluation.switch_precondition_met,
                },
                generation_id=evaluation.generation_id,
                source_observation_step=self._observation_step,
            )
        if evaluation.phase_changed:
            self._publish_event(
                "task_phase_changed",
                "physical_state_transition",
                {
                    "from_phase_code": evaluation.previous_phase_code,
                    "from_phase": PHASE_NAMES[evaluation.previous_phase_code],
                    "to_phase_code": evaluation.phase_code,
                    "to_phase": evaluation.phase,
                    "episode_step": self._observation_step,
                },
                source_observation_step=self._observation_step,
            )
        if item.collision:
            self._publish_event(
                "collision_detected",
                "disallowed_contact_force_threshold",
                {
                    "episode_step": self._observation_step,
                    "force_n": item.max_disallowed_contact_force_n,
                    "threshold_n": self._scene.config.collision_threshold_n,
                    "filter_paths": list(DISALLOWED_CONTACT_FILTER_PATHS),
                    "sensor_links": list(DISALLOWED_LINK_PATHS),
                },
                source_observation_step=self._observation_step,
            )

        robot_state = pack_robot_state(
            end_effector_xyz=scene_measurement.end_effector_xyz,
            end_effector_axis_angle_xyz=quaternion_wxyz_to_axis_angle(
                scene_measurement.end_effector_wxyz
            ),
            gripper_command=self._scene.gripper_command,
        )
        task_state = pack_dynamic_task_state(
            scenario=self._scenario,
            measurement=item,
            evaluation=evaluation,
        )
        message = self._types.Observation()
        self._stamp(message)
        message.episode_id = self._active_episode_id
        message.observation_step = self._observation_step
        message.generation_id = evaluation.generation_id
        message.task_id = TASK_ID
        message.robot_state = list(robot_state)
        message.task_state = list(task_state)
        message.terminated = evaluation.terminated
        if defer_for_bootstrap:
            if evaluation.terminated:
                raise RuntimeError(
                    "initial dynamic observation unexpectedly terminated"
                )
            self._deferred_initial_observation = message
            self._bootstrap_observation_published = False
            return False
        self._observation_publisher.publish(message)

        if not evaluation.terminated:
            self._command_deadline_wall = (
                time.monotonic() + self._command_timeout_seconds
            )
            return False

        terminal_detail = {
            "success": evaluation.success,
            "termination_reason": evaluation.termination_reason,
            "phase_code": evaluation.phase_code,
            "phase": evaluation.phase,
            "terminal_step": self._observation_step,
            "generation_id": evaluation.generation_id,
            "final_destination_xy_error_m": evaluation.final_destination_xy_error_m,
            "obsolete_destination_xy_error_m": evaluation.obsolete_destination_xy_error_m,
            "success_streak_steps": evaluation.success_streak_steps,
            "max_disallowed_contact_force_n": self._maximum_contact_force_n,
        }
        self._publish_event(
            "task_terminated",
            evaluation.termination_reason,
            terminal_detail,
            source_observation_step=self._observation_step,
        )
        self._publish_event(
            "episode_end",
            evaluation.termination_reason,
            terminal_detail,
            source_observation_step=self._observation_step,
        )
        self._publish_control_status(
            command=self._types.EpisodeControl.TERMINATE,
            active=False,
            terminated=True,
            success=evaluation.success,
        )
        episode_id = self._active_episode_id
        self._last_terminal_episode_id = episode_id
        self._last_result = DynamicEpisodeResult(
            episode_id=episode_id,
            success=evaluation.success,
            termination_reason=evaluation.termination_reason,
            terminal_step=self._observation_step,
            generation_id=evaluation.generation_id,
            scenario_sha256=scenario_sha256(self._scenario),
            maximum_disallowed_contact_force_n=self._maximum_contact_force_n,
        )
        self._active_episode_id = None
        self._pending_command = None
        self._command_deadline_wall = None
        return True

    def process_control_requests(self) -> None:
        while self._control_requests:
            request = self._control_requests.popleft()
            if request.command in {
                self._types.EpisodeControl.START,
                self._types.EpisodeControl.RESET,
            }:
                if self._active_episode_id is not None:
                    self.node.get_logger().warning(
                        "episode_control_rejected reason=episode_already_active"
                    )
                    continue
                self._begin_episode(request)
                continue
            if (
                self._active_episode_id is None
                and request.episode_id == self._last_terminal_episode_id
            ):
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
                    "episode_control_rejected reason=previous_episode"
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

    def _begin_episode(self, request: DynamicControlRequest) -> None:
        spec = self._configured_episode
        matrix = self._matrix
        if spec is None or matrix is None or request.episode_id != spec.episode_id:
            raise RuntimeError(
                "lifecycle episode does not match configured matrix episode"
            )
        if request.generation_id != INITIAL_GENERATION_ID:
            raise RuntimeError("dynamic episode must start at generation 1")
        scenario = scenario_for_seed(spec.seed, self._scene.config)
        if scenario_sha256(scenario) != spec.scenario_sha256:
            raise RuntimeError("configured native reset scenario hash drifted")
        measured_reset = self._scene.reset(scenario)
        reset_state = self._scene.reset_state_payload(scenario, measured_reset)
        from action_stream_benchmark.m8_protocol import build_episode_fairness

        fairness = build_episode_fairness(
            protocol_payload=matrix.protocol_payload,
            scenario_payload=scenario_payload(scenario),
            reset_state_payload=reset_state,
            fault_trace_sha256=spec.trace_sha256,
            freeze_sha256=matrix.freeze_sha256,
        )
        if fairness["scenario_sha256"] != spec.scenario_sha256:
            raise RuntimeError("native reset fairness scenario hash drifted")
        self._active_episode_id = spec.episode_id
        self._scenario = scenario
        self._task_machine = DynamicTaskMachine(scenario, self._scene.config)
        self._generation_id = INITIAL_GENERATION_ID
        self._observation_step = 0
        self._pending_command = None
        self._deferred_initial_observation = None
        self._bootstrap_observation_published = False
        self._maximum_contact_force_n = measured_reset.max_disallowed_contact_force_n
        self._last_terminal_episode_id = None
        self._publish_clock()
        self._publish_control_status(
            command=request.command,
            active=True,
            terminated=False,
            success=False,
        )
        task_thresholds = asdict(self._scene.config)
        common = {
            **fairness,
            "scenario": scenario_payload(scenario),
            "reset_state": reset_state,
            "split": matrix.split,
            "profile_id": spec.profile_id,
            "strategy": spec.strategy,
            "provenance_manifest": _portable_reference(
                matrix.provenance_manifest,
                base=matrix.source_path.parent,
            ),
            "provenance_sha256": matrix.provenance_sha256,
            "frozen_before_first_holdout_result": (
                matrix.frozen_before_first_holdout_result
            ),
            "holdout_freeze_status": matrix.holdout_freeze_status,
            "paired_fairness_contract": {
                "same_policy": True,
                "same_scenario_hash_required": True,
                "same_fault_trace_hash_required": True,
                "same_control_frequency_hz": CONTROL_FREQUENCY_HZ,
                "same_action_horizon": ACTION_HORIZON,
            },
            "command_motion_audit_required": True,
            "command_motion_audit_contract": {
                "requested_target_delta": "requested xyz versus previous native scene target",
                "applied_target_delta": "applied xyz versus previous native scene target",
                "measured_end_effector_delta": "physical xyz change over the control step",
                "planned_row_limit_scope": "adjacent_planned_rows_only",
            },
            "task_thresholds": task_thresholds,
            "controller_parameters": asdict(matrix.policy_config),
            "task_state_size": DYNAMIC_TASK_STATE_SIZE,
            "task_state_layout": list(DYNAMIC_TASK_STATE_LAYOUT),
            "native_evidence_class": NATIVE_EVIDENCE_CLASS,
        }
        self._publish_event("episode_start", "native_isaac_reset", common)
        self._publish_event(
            "simulator_reset",
            "seeded_native_scene_reset",
            {
                **common,
                "measured_object_xyz": list(measured_reset.object_xyz),
                "measured_object_wxyz": list(measured_reset.object_wxyz),
                "measured_gripper_aperture_m": measured_reset.gripper_aperture_m,
                "simulation_time_seconds": self._scene.sim_time_seconds(),
            },
        )
        self._publish_observation(defer_for_bootstrap=self._defer_initial_observation)

    def step_if_ready(self) -> bool:
        if self._active_episode_id is None or self._pending_command is None:
            return False
        started = time.monotonic()
        command = self._pending_command
        self._pending_command = None
        previous_scene_target = self._scene.current_command
        end_effector_before_xyz, _ = self._scene.end_effector_pose()
        if not command.hold:
            self._scene.set_command(command.command)
        applied_scene_target = self._scene.current_command
        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            self._scene.step()
            self._publish_clock()
        remaining = CONTROL_DT_SECONDS - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)
        end_effector_after_xyz, _ = self._scene.end_effector_pose()
        self._publish_event(
            "command_motion_audit",
            "native_target_and_measured_delta",
            {
                **_command_motion_audit_payload(
                    requested_command=command.command,
                    previous_scene_target=previous_scene_target,
                    applied_scene_target=applied_scene_target,
                    end_effector_before_xyz=end_effector_before_xyz,
                    end_effector_after_xyz=end_effector_after_xyz,
                    hold=command.hold,
                    invalid_command=command.invalid_command,
                    planned_row_translation_limit_m=(
                        self._planned_row_translation_limit_m
                    ),
                ),
                "actual_target_step": command.actual_target_step,
                "source_request_id": command.source_request_id,
                "source_generation_id": command.source_generation_id,
                "source_observation_step": command.source_observation_step,
                "source_target_step": command.source_target_step,
            },
            request_id=command.source_request_id,
            generation_id=command.source_generation_id,
            source_observation_step=command.source_observation_step,
            source_target_step=command.source_target_step,
            actual_target_step=command.actual_target_step,
        )
        self._observation_step = command.actual_target_step
        self._publish_observation(invalid_command=command.invalid_command)
        return True

    @property
    def initial_observation_is_deferred(self) -> bool:
        return self._deferred_initial_observation is not None

    @property
    def bootstrap_observation_published(self) -> bool:
        return self._bootstrap_observation_published

    def publish_bootstrap_observation(self) -> None:
        message = self._deferred_initial_observation
        if message is None:
            raise RuntimeError("no deferred initial observation is available")
        if self._bootstrap_observation_published:
            return
        self._bootstrap_observation_publisher.publish(message)
        self._bootstrap_observation_published = True
        self.node.get_logger().info(
            "dynamic_bootstrap_observation_published_after_executor_ack "
            f"episode={message.episode_id!r} step={message.observation_step}"
        )

    def release_initial_observation(self) -> None:
        message = self._deferred_initial_observation
        if message is None:
            return
        if not self._bootstrap_observation_published:
            raise RuntimeError(
                "cannot release initial observation before bootstrap publication"
            )
        self._deferred_initial_observation = None
        self._observation_publisher.publish(message)
        self._command_deadline_wall = time.monotonic() + self._command_timeout_seconds
        self.node.get_logger().info(
            "dynamic_bootstrap_observation_released "
            f"episode={message.episode_id!r} step={message.observation_step}"
        )

    def command_wait_timed_out(self) -> bool:
        return (
            self._active_episode_id is not None
            and self._pending_command is None
            and self._command_deadline_wall is not None
            and time.monotonic() >= self._command_deadline_wall
        )

    @property
    def last_result(self) -> DynamicEpisodeResult | None:
        return self._last_result

    def close(self) -> None:
        self.node.destroy_node()


class DynamicInKitEpisodeRuntime:
    """Per-episode policy, M8 fault, recorder, and request graph."""

    def __init__(
        self,
        bridge_node: Any,
        spec: MatrixEpisodeSpec,
        matrix: DynamicMatrixManifest,
    ) -> None:
        from action_stream_benchmark.fault_injector_node import (
            create_fault_injector_node,
        )
        from action_stream_isaac.dynamic_event_recorder import (
            DynamicRecorderConfig,
            create_dynamic_event_recorder_node,
        )
        from action_stream_policy.dynamic_policy_node import create_dynamic_policy_node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter

        def parameters(**values: object) -> list[Any]:
            return [Parameter(name, value=value) for name, value in values.items()]

        spec.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._executor: Any | None = SingleThreadedExecutor()
        self._bridge_node = bridge_node
        self._nodes: list[Any] = []
        self._recorder: Any | None = None
        self.request_driver: Any | None = None
        try:
            policy = create_dynamic_policy_node(
                parameter_overrides=parameters(
                    use_sim_time=True,
                    request_topic="/action_stream/inference_request",
                    raw_chunk_topic="/action_stream/raw_action_chunk",
                    event_topic="/action_stream/events",
                    **_policy_parameter_values(matrix.policy_config),
                )
            )
            self._nodes.append(policy)
            injector = create_fault_injector_node(
                parameter_overrides=parameters(
                    use_sim_time=True,
                    trace_file=str(spec.fault_trace_file),
                    raw_chunk_topic="/action_stream/raw_action_chunk",
                    action_chunk_topic="/action_stream/action_chunk",
                    event_topic="/action_stream/events",
                )
            )
            self._nodes.append(injector)
            recorder = create_dynamic_event_recorder_node(
                DynamicRecorderConfig(
                    output_path=spec.event_log_path,
                    episode_id=spec.episode_id,
                    seed=spec.seed,
                    profile_id=spec.profile_id,
                    strategy=spec.strategy,
                    split=matrix.split,
                    expected_scenario_sha256=spec.scenario_sha256,
                    expected_fault_trace_sha256=spec.trace_sha256,
                )
            )
            self._nodes.append(recorder)
            self._recorder = recorder
            request_driver = create_dynamic_request_driver_node(
                DynamicRequestDriverConfig(
                    episode_id=spec.episode_id,
                    strategy=spec.strategy,
                )
            )
            self._nodes.append(request_driver)
            self.request_driver = request_driver
            for node in [bridge_node, *self._nodes]:
                if not self._executor.add_node(node):
                    raise RuntimeError(
                        f"failed to add dynamic node {node.get_name()!r} to executor"
                    )
        except Exception:
            self.close()
            raise

    @property
    def bootstrap_ready(self) -> bool:
        return bool(self.request_driver and self.request_driver.bootstrap_ready)

    @property
    def executor_episode_ready(self) -> bool:
        return bool(self.request_driver and self.request_driver.executor_episode_ready)

    @property
    def complete(self) -> bool:
        return bool(self.request_driver and self.request_driver.complete)

    @property
    def request_count(self) -> int:
        return int(self.request_driver.request_count) if self.request_driver else 0

    @property
    def terminal_success(self) -> bool:
        return bool(self.request_driver and self.request_driver.terminal_success)

    @property
    def terminal_step(self) -> int | None:
        if self.request_driver is None or self.request_driver.terminal_step is None:
            return None
        return int(self.request_driver.terminal_step)

    def spin_once(self, *, timeout_sec: float) -> None:
        if self._executor is None:
            raise RuntimeError("dynamic in-Kit executor is closed")
        self._executor.spin_once(timeout_sec=timeout_sec)

    def spin_pending_callbacks(self, *, initial_timeout_sec: float) -> None:
        """Wait once, then drain the bounded in-process ROS callback backlog.

        A native control step publishes several events plus an observation. A
        single callback per 20 Hz cycle therefore falls behind even when the
        policy itself is immediate, making fresh requests appear artificially
        delayed. The public rclpy executor API has no ``spin_some`` result, so
        use a fixed nonblocking drain bound after the initial wait.
        """

        if self._executor is None:
            raise RuntimeError("dynamic in-Kit executor is closed")
        self._executor.spin_once(timeout_sec=initial_timeout_sec)
        for _ in range(ROS_CALLBACK_DRAIN_LIMIT - 1):
            self._executor.spin_once(timeout_sec=0.0)

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        close_error: BaseException | None = None
        try:
            if self._recorder is not None:
                self._recorder.close()
        except BaseException as exc:
            close_error = exc
        finally:
            if executor is not None:
                for node in [self._bridge_node, *self._nodes]:
                    executor.remove_node(node)
                executor.shutdown(timeout_sec=2.0)
            for node in reversed(self._nodes):
                node.destroy_node()
            self._nodes = []
            self._recorder = None
            self.request_driver = None
        if close_error is not None:
            raise close_error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise RuntimeError(f"refusing to reuse stale summary temporary: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _finalize_episode_summary(
    *, matrix: DynamicMatrixManifest, spec: MatrixEpisodeSpec
) -> dict[str, Any]:
    """Recompute the only successful summary from committed raw JSONL rows."""

    if not spec.event_log_path.is_file() or spec.event_log_path.stat().st_size <= 0:
        raise RuntimeError(
            f"native episode produced no raw event log: {spec.event_log_path}"
        )
    from action_stream_benchmark.m8_replay import (
        FAIRNESS_FIELDS,
        HOLDOUT_FAIRNESS_FIELDS,
        write_summary_from_rows,
    )
    from action_stream_benchmark.schema import read_jsonl

    rows = read_jsonl(spec.event_log_path)
    starts = [row for row in rows if row.get("event_type") == "episode_start"]
    if len(starts) != 1:
        raise RuntimeError("committed M8 raw log does not contain exactly one start")
    required = (
        HOLDOUT_FAIRNESS_FIELDS if matrix.split == "frozen_holdout" else FAIRNESS_FIELDS
    )
    missing = [name for name in required if name not in starts[0]]
    if missing:
        raise RuntimeError(f"committed M8 raw fairness fields missing: {missing}")
    fairness = {name: starts[0][name] for name in required}
    if fairness["scenario_sha256"] != spec.scenario_sha256:
        raise RuntimeError("committed raw scenario hash differs from matrix")
    if fairness["fault_trace_sha256"] != spec.trace_sha256:
        raise RuntimeError("committed raw fault trace hash differs from matrix")
    if fairness["protocol_sha256"] != matrix.protocol_sha256:
        raise RuntimeError("committed raw protocol hash differs from matrix")
    if (
        matrix.freeze_sha256 is not None
        and fairness.get("freeze_sha256") != matrix.freeze_sha256
    ):
        raise RuntimeError("committed raw freeze hash differs from matrix")
    return write_summary_from_rows(
        rows,
        spec.summary_path,
        profile_id=spec.profile_id,
        seed=spec.seed,
        strategy=spec.strategy,
        episode_id=spec.episode_id,
        fairness=fairness,
        split=matrix.split,
    )


def _failure_summary(
    *,
    matrix: DynamicMatrixManifest,
    spec: MatrixEpisodeSpec,
    exc: BaseException,
    started_wall_ns: int,
    cleanup_errors: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    event_exists = spec.event_log_path.is_file()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "milestone": MILESTONE,
        "evidence_class": NATIVE_EVIDENCE_CLASS,
        "native_execution_performed": True,
        "native_execution_complete": False,
        "headline_evidence_eligible": False,
        "split": matrix.split,
        "episode_id": spec.episode_id,
        "seed": spec.seed,
        "strategy": spec.strategy,
        "profile_id": spec.profile_id,
        "scenario_sha256": spec.scenario_sha256,
        "trace_sha256": spec.trace_sha256,
        "success": False,
        "termination_reason": "simulator_or_transport_failure",
        "infrastructure_failure": f"{type(exc).__name__}: {exc}",
        "event_log_path": str(spec.event_log_path) if event_exists else None,
        "event_log_sha256": _sha256_file(spec.event_log_path) if event_exists else None,
        "started_wall_time_ns": started_wall_ns,
        "completed_wall_time_ns": time.time_ns(),
        "persistent_isaac_process": True,
    }
    if cleanup_errors:
        payload["cleanup_errors"] = [dict(error) for error in cleanup_errors]
    return payload


def _cleanup_error(stage: str, exc: BaseException) -> dict[str, str]:
    try:
        message = str(exc)
    except BaseException:  # pragma: no cover - defensive against hostile exceptions
        message = "<exception string conversion failed>"
    return {
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": message,
    }


def _capture_cleanup_error(
    stage: str, action: Callable[[], object]
) -> dict[str, str] | None:
    try:
        action()
    except BaseException as exc:
        return _cleanup_error(stage, exc)
    return None


def _record_episode_failure(
    *,
    runtime: Any | None,
    matrix: DynamicMatrixManifest,
    spec: MatrixEpisodeSpec,
    exc: BaseException,
    started_wall_ns: int,
) -> list[dict[str, str]]:
    """Close one failed episode and record it without raising a cleanup error."""

    cleanup_errors: list[dict[str, str]] = []
    if runtime is not None:
        close_error = _capture_cleanup_error(
            "episode_runtime_close", lambda: runtime.close()
        )
        if close_error is not None:
            cleanup_errors.append(close_error)

    def write_failure_summary() -> None:
        if spec.summary_path.exists():
            return
        _write_json_atomic(
            spec.summary_path,
            _failure_summary(
                matrix=matrix,
                spec=spec,
                exc=exc,
                started_wall_ns=started_wall_ns,
                cleanup_errors=cleanup_errors,
            ),
        )

    summary_error = _capture_cleanup_error(
        "failure_summary_write", write_failure_summary
    )
    if summary_error is not None:
        cleanup_errors.append(summary_error)
    return cleanup_errors


def _emit_cleanup_errors(cleanup_errors: Sequence[Mapping[str, str]]) -> None:
    for error in cleanup_errors:
        try:
            print(
                "WARNING: cleanup_error "
                + json.dumps(dict(error), sort_keys=True, separators=(",", ":")),
                file=sys.stderr,
            )
        except BaseException:
            # Cleanup reporting must not become a new failure source.
            pass


def _cleanup_batch_resources(
    *,
    video_capture: Any | None,
    runtime: Any | None,
    bridge: Any | None,
    scene: Any | None,
    rclpy: Any | None,
    simulation_app: Any | None,
    exit_code: int,
) -> tuple[int, list[dict[str, str]]]:
    """Attempt every batch cleanup step in order and return a deterministic status."""

    cleanup_errors: list[dict[str, str]] = []

    def attempt(stage: str, action: Callable[[], object]) -> None:
        error = _capture_cleanup_error(stage, action)
        if error is not None:
            cleanup_errors.append(error)

    if video_capture is not None:
        attempt("video_capture_cancel", lambda: video_capture.cancel())
    if runtime is not None:
        attempt("episode_runtime_close", lambda: runtime.close())
    if bridge is not None:
        attempt("isaac_bridge_close", lambda: bridge.close())
    if scene is not None:
        attempt("isaac_scene_close", lambda: scene.close())
    if rclpy is not None:

        def shutdown_ros() -> None:
            if rclpy.ok():
                rclpy.shutdown()

        attempt("rclpy_shutdown", shutdown_ros)

    if cleanup_errors and exit_code == 0:
        exit_code = 1
    if simulation_app is not None:
        attempt(
            "simulation_app_close",
            lambda: simulation_app.close(exit_code=exit_code, skip_cleanup=True),
        )
    if cleanup_errors and exit_code == 0:
        exit_code = 1
    return exit_code, cleanup_errors


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--headless", type=_parse_bool, default=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--expected-strategy", choices=STRATEGIES, default=None)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--startup-discovery-seconds", type=float, default=1.0)
    parser.add_argument("--bootstrap-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--command-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--terminal-drain-seconds", type=float, default=0.5)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-finalize-timeout-seconds", type=float, default=120.0)
    return parser


def _validate_runtime_args(args: Any, matrix: DynamicMatrixManifest) -> int:
    if args.max_episodes < 0:
        raise ValueError("--max-episodes must be non-negative")
    selected_count = (
        len(matrix.episodes)
        if args.max_episodes == 0
        else min(args.max_episodes, len(matrix.episodes))
    )
    if selected_count <= 0:
        raise ValueError("runtime selected zero episodes")
    if matrix.split == "frozen_holdout" and selected_count != len(matrix.episodes):
        raise ValueError("--max-episodes cannot truncate a frozen holdout matrix")
    for name in (
        "startup_discovery_seconds",
        "bootstrap_timeout_seconds",
        "command_timeout_seconds",
        "terminal_drain_seconds",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )
    if args.bootstrap_timeout_seconds <= 0.0 or args.command_timeout_seconds <= 0.0:
        raise ValueError("bootstrap and command timeouts must be positive")
    if args.expected_strategy is not None and args.expected_strategy != matrix.strategy:
        raise ValueError(
            "external executor strategy mismatch: "
            f"expected={args.expected_strategy}, matrix={matrix.strategy}"
        )
    if not args.validate_only and args.expected_strategy is None:
        raise ValueError(
            "real native execution requires --expected-strategy so the external "
            "C++ executor cannot be mislabeled"
        )
    if args.video_output is not None:
        if selected_count != 1:
            raise ValueError("--video-output requires exactly one selected episode")
        output = args.video_output.resolve()
        if output.suffix.lower() != ".mp4":
            raise ValueError("--video-output must end in .mp4")
        if output.exists():
            raise ValueError(f"refusing to overwrite video output: {output}")
        partial = output.with_name(f".{output.stem}.partial.mp4")
        if partial.exists():
            raise ValueError(f"refusing to reuse partial video output: {partial}")
        if not 1 <= args.video_fps <= 120:
            raise ValueError("--video-fps must lie in [1,120]")
        if not 64 <= args.video_width <= 4096 or not 64 <= args.video_height <= 4096:
            raise ValueError("video dimensions are outside [64,4096]")
        if (
            not math.isfinite(args.video_finalize_timeout_seconds)
            or args.video_finalize_timeout_seconds <= 0.0
        ):
            raise ValueError("video finalize timeout must be finite and positive")
    return selected_count


def _enable_ros_bridge(simulation_app: Any) -> None:
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    extension = "isaacsim.ros2.bridge"
    manager.set_extension_enabled_immediate(extension, True)
    simulation_app.update()
    if not manager.is_extension_enabled(extension):
        raise RuntimeError(f"Isaac extension {extension} could not be enabled")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        matrix = load_matrix_manifest(args.matrix_manifest)
        selected_count = _validate_runtime_args(args, matrix)
    except ValueError as exc:
        raise SystemExit(f"invalid dynamic matrix/configuration: {exc}") from exc
    if args.validate_only:
        report = validation_report(matrix)
        report["selected_episode_count"] = selected_count
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    report = probe(require_ros_imports=False)
    if not report.ready:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True), file=sys.stderr)
        print(
            "FATAL: native M8 Isaac preflight failed; no test-plant fallback was used",
            file=sys.stderr,
        )
        return 2

    selected = matrix.episodes[:selected_count]
    simulation_app: Any | None = None
    scene: DynamicIsaacScene | None = None
    bridge: DynamicIsaacBridge | None = None
    runtime: DynamicInKitEpisodeRuntime | None = None
    video_capture: Any | None = None
    video_final_output: Path | None = None
    video_partial_output: Path | None = None
    rclpy: Any | None = None
    exit_code = 0
    try:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": bool(args.headless)})
        _enable_ros_bridge(simulation_app)
        import rclpy as rclpy_module

        rclpy = rclpy_module
        rclpy.init(args=None)
        scene = DynamicIsaacScene(
            simulation_app,
            task_config=matrix.task_config,
            policy_config=matrix.policy_config,
        )
        bridge = DynamicIsaacBridge(
            scene,
            command_timeout_seconds=args.command_timeout_seconds,
            planned_row_translation_limit_m=(
                matrix.policy_config.maximum_translation_per_step_m
            ),
            defer_initial_observation=True,
        )
        if args.video_output is not None:
            from .isaac_adapter import IsaacViewportVideoCapture, VideoCaptureConfig

            video_final_output = args.video_output.resolve()
            video_partial_output = video_final_output.with_name(
                f".{video_final_output.stem}.partial.mp4"
            )
            video_capture = IsaacViewportVideoCapture(
                simulation_app,
                VideoCaptureConfig(
                    output_path=video_partial_output,
                    fps=args.video_fps,
                    width=args.video_width,
                    height=args.video_height,
                    finalize_timeout_seconds=args.video_finalize_timeout_seconds,
                ),
            )

        for index, spec in enumerate(selected):
            started_wall_ns = time.time_ns()
            runtime = None
            try:
                bridge.configure_episode(spec, matrix)
                runtime = DynamicInKitEpisodeRuntime(bridge.node, spec, matrix)
                discovery_deadline = time.monotonic() + args.startup_discovery_seconds
                while (
                    simulation_app.is_running()
                    and rclpy.ok()
                    and time.monotonic() < discovery_deadline
                ):
                    runtime.spin_pending_callbacks(initial_timeout_sec=0.01)
                bridge.publish_episode_request(reset=index > 0)
                executor_ack_deadline: float | None = None
                bootstrap_request_deadline: float | None = None
                terminal_drain_deadline: float | None = None
                while simulation_app.is_running() and rclpy.ok():
                    runtime.spin_pending_callbacks(initial_timeout_sec=0.01)
                    bridge.process_control_requests()
                    if bridge.initial_observation_is_deferred:
                        if not runtime.executor_episode_ready:
                            if executor_ack_deadline is None:
                                executor_ack_deadline = (
                                    time.monotonic() + args.bootstrap_timeout_seconds
                                )
                            elif time.monotonic() >= executor_ack_deadline:
                                raise RuntimeError(
                                    "C++ executor did not acknowledge episode START/RESET "
                                    f"within {args.bootstrap_timeout_seconds:.3f}s"
                                )
                        else:
                            executor_ack_deadline = None
                            if not bridge.bootstrap_observation_published:
                                bridge.publish_bootstrap_observation()
                                bootstrap_request_deadline = (
                                    time.monotonic() + args.bootstrap_timeout_seconds
                                )
                        if runtime.bootstrap_ready:
                            bridge.release_initial_observation()
                            bootstrap_request_deadline = None
                        elif (
                            bootstrap_request_deadline is not None
                            and time.monotonic() >= bootstrap_request_deadline
                        ):
                            raise RuntimeError(
                                "executor did not register the generation-1 bootstrap "
                                f"request within {args.bootstrap_timeout_seconds:.3f}s"
                            )
                    stepped = bridge.step_if_ready()
                    if stepped and video_capture is not None:
                        video_capture.capture_frame(simulation_app)
                    if runtime.complete and terminal_drain_deadline is None:
                        terminal_drain_deadline = (
                            time.monotonic() + args.terminal_drain_seconds
                        )
                    if (
                        terminal_drain_deadline is not None
                        and time.monotonic() >= terminal_drain_deadline
                    ):
                        break
                    if bridge.command_wait_timed_out():
                        raise RuntimeError(
                            "no RobotCommand arrived before the command timeout; "
                            "refusing to fabricate progress"
                        )
                if not simulation_app.is_running() or not rclpy.ok():
                    raise RuntimeError("Isaac or ROS stopped before episode completion")
                result = bridge.last_result
                if result is None or result.episode_id != spec.episode_id:
                    raise RuntimeError(
                        "dynamic episode ended without a terminal task result"
                    )
                runtime.close()
                runtime = None
                summary = _finalize_episode_summary(matrix=matrix, spec=spec)
                if video_capture is not None:
                    captured_partial = video_capture.finish(simulation_app)
                    video_capture = None
                    if (
                        video_final_output is None
                        or video_partial_output is None
                        or captured_partial != video_partial_output
                        or not captured_partial.is_file()
                        or captured_partial.stat().st_size <= 0
                    ):
                        raise RuntimeError(
                            "native viewport capture failed MP4 validation"
                        )
                    captured_partial.replace(video_final_output)
                print(
                    "native_dynamic_episode_complete "
                    f"episode={spec.episode_id} "
                    f"success={summary['metrics']['task_success']} "
                    f"reason={summary['metrics']['completion_reason']} "
                    f"step={summary['metrics']['completion_steps']} "
                    f"summary={spec.summary_path}"
                )
            except Exception as exc:
                cleanup_errors = _record_episode_failure(
                    runtime=runtime,
                    matrix=matrix,
                    spec=spec,
                    exc=exc,
                    started_wall_ns=started_wall_ns,
                )
                runtime = None
                _emit_cleanup_errors(cleanup_errors)
                raise
        exit_code = 0
    except Exception as exc:
        print(
            f"FATAL: dynamic Isaac batch stopped: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        exit_code = 1
    finally:
        exit_code, cleanup_errors = _cleanup_batch_resources(
            video_capture=video_capture,
            runtime=runtime,
            bridge=bridge,
            scene=scene,
            rclpy=rclpy,
            simulation_app=simulation_app,
            exit_code=exit_code,
        )
        _emit_cleanup_errors(cleanup_errors)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
