"""Frozen protocol and provenance helpers for ActionStream M8-G0.

This module is deliberately separate from :mod:`action_stream_benchmark.schema`.
The latter is part of the accepted M7-G0 artifact hash contract and must keep
emitting ``M7-G0``.  Only generic atomic JSON and canonical hashing helpers are
reused here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import hashlib
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Final

from .schema import canonical_sha256, read_json, write_json_atomic


M8_SCHEMA_VERSION: Final = 1
M8_MILESTONE: Final = "M8-G0"
CONTROL_FREQUENCY_HZ: Final = 20.0
ACTION_HORIZON: Final = 30
ACTION_DIMENSION: Final = 7
REQUEST_INTERVAL_STEPS: Final = 10
SAFE_HOLD_COMMAND: Final = (
    0.307015,
    0.0,
    0.589907,
    3.141592653589793,
    0.0,
    0.0,
    1.0,
)
STRATEGIES: Final = ("sync_hold", "naive_async", "aligned_async")
PROFILE_STRATEGIES: Final = {
    "profile_0_sanity": ("sync_hold",),
    "profile_1_fixed": STRATEGIES,
    "profile_2_faults": STRATEGIES,
}
NATIVE_ISAAC_EVIDENCE_CLASS: Final = "ros_cpp_isaac_sim"
NATIVE_RUNNER_SOURCE_PATHS: Final = (
    "scripts/m8_run_isaac.ps1",
    "scripts/m8_run_isaac.sh",
)
LINUX_NATIVE_RUNNER_SOURCE_PATH: Final = "scripts/m8_run_isaac.sh"
LINUX_NATIVE_RUNNER_SUPPORT_SOURCE_PATH: Final = (
    "scripts/m8_linux_runner_support.py"
)
LINUX_NATIVE_RUNNER_SUPPORT_FIELDS: Final = (
    "runner_support_source",
    "runner_support_evidence",
    "runner_support_evidence_sha256",
    "runner_support_sha256",
)
LINUX_NATIVE_RUNNER_SUPPORT_BASELINE_FIELDS: Final = (
    "runner_support_source",
    "runner_support_evidence_sha256",
    "runner_support_sha256",
)
ISAAC_WORKSPACE_REPOSITORY_URL: Final = (
    "https://github.com/isaac-sim/IsaacSim-ros_workspaces.git"
)
ISAAC_WORKSPACE_COMMIT: Final = "dd3eeede7912755996a18f4884285d9f50843f79"
ISAAC_WORKSPACE_RELATIVE_PATH: Final = "jazzy_ws"
OFFICIAL_LINUX_PIXI_VERSION: Final = "0.75.0"
OFFICIAL_LINUX_PIXI_VERSION_OUTPUT: Final = "pixi 0.75.0"
OFFICIAL_LINUX_PIXI_EXECUTABLE_SIZE_BYTES: Final = 77_311_024
OFFICIAL_LINUX_PIXI_EXECUTABLE_SHA256: Final = (
    "4383aed18b2d5569cf34a19638daf954aa4415cc87ad3a9da9f34059cc4a004c"
)
OFFICIAL_WORKSPACE_FILE_SPECS: Final = {
    "pixi.toml": {
        "canonical_lf_size_bytes": 5_467,
        "canonical_lf_sha256": (
            "b4e7a34c264e88f19ba6bfb3c7a72ee46b0843f3e6eb7e75619dc0ebb87b313d"
        ),
        "exact_crlf_size_bytes": 5_618,
        "exact_crlf_sha256": (
            "9649bf57644781a1fe0203ed6b80828ccb42ea07555475080e5d11a9b0c3e1ae"
        ),
    },
    "pixi.lock": {
        "canonical_lf_size_bytes": 1_491_808,
        "canonical_lf_sha256": (
            "ba8e59eef962cbf49a1ff06ff947ed5eaa4547d018389b048771a1e7d8bb890d"
        ),
        "exact_crlf_size_bytes": 1_533_045,
        "exact_crlf_sha256": (
            "2c2f9097b129847735b5abb805045a22076a731e2138c8192caac95d09a2866e"
        ),
    },
}
EXPECTED_RUNTIME_VERSIONS: Final = {
    "python_version": "3.12.13",
    "isaacsim": "6.0.1.0",
    "isaacsim-app": "6.0.1.0",
    "isaacsim-core": "6.0.1.0",
    "isaacsim-robot": "6.0.1.0",
    "isaacsim-ros2": "6.0.1.0",
    "rclpy": "7.1.9",
    "rosgraph-msgs": "2.0.3",
    "ros_distribution": "jazzy",
    "rmw_implementation": "rmw_zenoh_cpp",
    "rmw_zenoh_cpp": "0.2.9",
}
NATIVE_ENVIRONMENT_HASH_FIELDS: Final = (
    "external_environment_evidence_sha256",
    "pixi_manifest_evidence_sha256",
    "pixi_lock_evidence_sha256",
)
NATIVE_ENVIRONMENT_LEDGER_FIELDS: Final = (
    *NATIVE_ENVIRONMENT_HASH_FIELDS,
    "isaac_workspace_commit",
    "pixi_version",
    "python_version",
    "isaacsim_version",
    "ros_distribution",
    "rmw_implementation",
    "rmw_zenoh_cpp_version",
    "selected_gpu_uuid",
    "selected_gpu_name",
    "selected_gpu_driver_version",
    "selected_gpu_memory_total_mib",
)
STRONG_OBSOLETE_STEP_REDUCTION_THRESHOLD: Final = 0.20
PAIRED_RESET_FAIRNESS_CONTRACT: Final = {
    "raw_reset_state_sha256_required": True,
    "canonical_digest_is_diagnostic_when_numeric_tolerances_pass": True,
    "canonical_grids": {
        "robot_joint_positions_rad": 1e-5,
        "robot_joint_velocities_rad_s": 1e-4,
        "end_effector_position_m": 1e-5,
        "object_position_m": 1e-5,
        "quaternion_component": 1e-6,
    },
    "pairwise_tolerances": {
        "robot_joint_position_max_abs_rad": 1e-5,
        "robot_joint_velocity_max_abs_rad_s": 1e-4,
        "position_axis_max_abs_m": 2e-5,
        "position_l2_m": 3e-5,
        "quaternion_geodesic_rad": 1e-4,
        "quaternion_norm_abs_error": 1e-6,
    },
}
CALIBRATABLE_CONTRACT_PATHS: Final = frozenset(
    {
        "/protocol/controller/approach_height_m",
        "/protocol/controller/grasp_hand_offset_m",
        "/protocol/controller/carry_hand_height_m",
        "/protocol/controller/recovery_hover_offset_m",
        "/protocol/controller/maximum_translation_per_step_m",
        "/protocol/controller/workspace_x_m",
        "/protocol/controller/workspace_y_m",
        "/protocol/controller/workspace_z_m",
        "/protocol/task/object_x_range_m",
        "/protocol/task/object_y_range_m",
        "/protocol/task/zone_a_xy_m",
        "/protocol/task/zone_b_xy_m",
        "/protocol/task/switch_steps",
        "/protocol/task/lift_clearance_m",
        "/protocol/profiles/profile_1_fixed/profile_file",
        "/protocol/profiles/profile_1_fixed/profile_sha256",
        "/protocol/profiles/profile_2_faults/profile_file",
        "/protocol/profiles/profile_2_faults/profile_sha256",
        "/profiles/profile_1_fixed/profile/base_latency_ms",
        "/profiles/profile_1_fixed/profile/candidate_status",
        "/profiles/profile_2_faults/profile/base_latency_ms",
        "/profiles/profile_2_faults/profile/jitter_ms",
        "/profiles/profile_2_faults/profile/drop_probability",
        "/profiles/profile_2_faults/profile/extra_delay_probability",
        "/profiles/profile_2_faults/profile/extra_delay_ms",
        "/profiles/profile_2_faults/profile/duplicate_probability",
        "/profiles/profile_2_faults/profile/duplicate_delivery_offset_ms",
        "/profiles/profile_2_faults/profile/communication_pause_probability",
        "/profiles/profile_2_faults/profile/communication_pause_ms",
        "/profiles/profile_2_faults/profile/candidate_status",
    }
)
RUNTIME_BEHAVIORAL_CALIBRATION_PATHS: Final = frozenset(
    {
        "/protocol/controller/approach_height_m",
        "/protocol/controller/grasp_hand_offset_m",
        "/protocol/controller/carry_hand_height_m",
        "/protocol/controller/recovery_hover_offset_m",
        "/protocol/controller/maximum_translation_per_step_m",
        "/protocol/controller/workspace_x_m",
        "/protocol/controller/workspace_y_m",
        "/protocol/controller/workspace_z_m",
        "/protocol/task/object_x_range_m",
        "/protocol/task/object_y_range_m",
        "/protocol/task/zone_a_xy_m",
        "/protocol/task/zone_b_xy_m",
        "/protocol/task/switch_steps",
        "/protocol/task/lift_clearance_m",
        "/profiles/profile_1_fixed/profile/base_latency_ms",
        "/profiles/profile_2_faults/profile/base_latency_ms",
        "/profiles/profile_2_faults/profile/jitter_ms",
        "/profiles/profile_2_faults/profile/drop_probability",
        "/profiles/profile_2_faults/profile/extra_delay_probability",
        "/profiles/profile_2_faults/profile/extra_delay_ms",
        "/profiles/profile_2_faults/profile/duplicate_probability",
        "/profiles/profile_2_faults/profile/duplicate_delivery_offset_ms",
        "/profiles/profile_2_faults/profile/communication_pause_probability",
        "/profiles/profile_2_faults/profile/communication_pause_ms",
    }
)
REQUIRED_FREEZE_SOURCE_PATHS: Final = (
    "ros2_ws/src/action_stream_executor/CMakeLists.txt",
    "ros2_ws/src/action_stream_executor/package.xml",
    "ros2_ws/src/action_stream_executor/include/action_stream_executor/executor_state_machine.hpp",
    "ros2_ws/src/action_stream_executor/include/action_stream_executor/executor_node.hpp",
    "ros2_ws/src/action_stream_executor/src/differential_replay_probe.cpp",
    "ros2_ws/src/action_stream_executor/src/executor_node.cpp",
    "ros2_ws/src/action_stream_executor/src/executor_state_machine.cpp",
    "ros2_ws/src/action_stream_executor/src/main.cpp",
    "ros2_ws/src/action_stream_isaac/package.xml",
    "ros2_ws/src/action_stream_isaac/resource/action_stream_isaac",
    "ros2_ws/src/action_stream_isaac/setup.cfg",
    "ros2_ws/src/action_stream_isaac/setup.py",
    "ros2_ws/src/action_stream_isaac/action_stream_isaac/__init__.py",
    "ros2_ws/src/action_stream_isaac/action_stream_isaac/capability_probe.py",
    "ros2_ws/src/action_stream_isaac/action_stream_isaac/dynamic_event_recorder.py",
    "ros2_ws/src/action_stream_isaac/action_stream_isaac/dynamic_task.py",
    "ros2_ws/src/action_stream_isaac/action_stream_isaac/dynamic_isaac_adapter.py",
    "ros2_ws/src/action_stream_isaac/action_stream_isaac/dynamic_request_driver.py",
    "ros2_ws/src/action_stream_isaac/action_stream_isaac/ros_contract.py",
    "ros2_ws/src/action_stream_isaac/action_stream_isaac/task_logic.py",
    "ros2_ws/src/action_stream_isaac/launch/franka_dynamic_recovery.launch.py",
    "ros2_ws/src/action_stream_policy/package.xml",
    "ros2_ws/src/action_stream_policy/resource/action_stream_policy",
    "ros2_ws/src/action_stream_policy/setup.cfg",
    "ros2_ws/src/action_stream_policy/setup.py",
    "ros2_ws/src/action_stream_policy/action_stream_policy/__init__.py",
    "ros2_ws/src/action_stream_policy/action_stream_policy/dynamic_pick_place.py",
    "ros2_ws/src/action_stream_policy/action_stream_policy/dynamic_policy_node.py",
    "ros2_ws/src/action_stream_policy/action_stream_policy/scripted_policy.py",
    "ros2_ws/src/action_stream_benchmark/package.xml",
    "ros2_ws/src/action_stream_benchmark/resource/action_stream_benchmark",
    "ros2_ws/src/action_stream_benchmark/setup.cfg",
    "ros2_ws/src/action_stream_benchmark/setup.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/__init__.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/fault_injector_node.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/faults.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/m8_analysis.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/m8_archive.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/m8_cli.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/m8_faults.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/m8_figures.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/m8_matrix.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/m8_protocol.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/m8_recorder.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/m8_replay.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/m8_report.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/m8_scenario.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/plant.py",
    "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/schema.py",
    "ros2_ws/src/action_stream_msgs/CMakeLists.txt",
    "ros2_ws/src/action_stream_msgs/package.xml",
    "ros2_ws/src/action_stream_msgs/msg/ActionChunk.msg",
    "ros2_ws/src/action_stream_msgs/msg/EpisodeControl.msg",
    "ros2_ws/src/action_stream_msgs/msg/ExecutorDiagnostics.msg",
    "ros2_ws/src/action_stream_msgs/msg/InferenceRequest.msg",
    "ros2_ws/src/action_stream_msgs/msg/Observation.msg",
    "ros2_ws/src/action_stream_msgs/msg/RobotCommand.msg",
    "ros2_ws/src/action_stream_msgs/msg/RuntimeEvent.msg",
    "ros2_ws/src/action_stream_msgs/msg/TargetAction.msg",
    *NATIVE_RUNNER_SOURCE_PATHS,
    LINUX_NATIVE_RUNNER_SUPPORT_SOURCE_PATH,
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path_parts(value: object, *, name: str) -> tuple[str, ...]:
    text = value if isinstance(value, str) else ""
    normalized = text.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(text)
    if (
        not text
        or Path(text).is_absolute()
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
    ):
        raise ValueError(f"{name} must be a portable relative path")
    return tuple(posix.parts)


def _receipt_relative_file(
    value: object,
    *,
    receipt_file: Path,
    repository_root: Path,
    name: str,
    allow_parent: bool,
) -> Path:
    try:
        parts = _portable_path_parts(value, name=name)
    except ValueError as exc:
        raise ValueError(f"{name} must be receipt-relative") from exc
    if not allow_parent and ".." in parts:
        raise ValueError(f"{name} must stay inside its receipt directory")
    candidate = (receipt_file.parent / Path(*parts)).resolve()
    try:
        candidate.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{name} escapes the repository") from exc
    if not candidate.is_file():
        raise ValueError(f"{name} does not exist: {candidate}")
    return candidate


def _repository_relative_file(
    value: object,
    *,
    repository_root: Path,
    name: str,
) -> tuple[str, Path]:
    parts = _portable_path_parts(value, name=name)
    if ".." in parts:
        raise ValueError(f"{name} escapes the repository")
    portable = PurePosixPath(*parts).as_posix()
    candidate = (repository_root / Path(*parts)).resolve()
    try:
        candidate.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{name} escapes the repository") from exc
    if not candidate.is_file():
        raise ValueError(f"{name} does not exist: {candidate}")
    return portable, candidate


def _is_informational_absolute_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return PurePosixPath(value.replace("\\", "/")).is_absolute() or PureWindowsPath(
        value
    ).is_absolute()


def load_seed_file(path: Path | str) -> tuple[int, ...]:
    payload = read_json(path)
    if payload.get("milestone") != M8_MILESTONE:
        raise ValueError(f"{path}: expected milestone {M8_MILESTONE}")
    seeds = tuple(int(seed) for seed in payload.get("seeds", ()))
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: seeds must be unique non-negative integers")
    return seeds


def validate_seed_splits(
    *,
    baseline: Sequence[int],
    development: Sequence[int],
    holdout: Sequence[int],
) -> None:
    groups = {
        "baseline": tuple(int(seed) for seed in baseline),
        "development": tuple(int(seed) for seed in development),
        "holdout": tuple(int(seed) for seed in holdout),
    }
    if len(groups["baseline"]) < 20:
        raise ValueError("simulator baseline gate requires at least 20 seeds")
    if not 1 <= len(groups["development"]) <= 12:
        raise ValueError("development candidates require 1 to 12 seeds")
    if len(groups["holdout"]) < 40:
        raise ValueError("frozen holdout requires at least 40 seeds")
    for name, seeds in groups.items():
        if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"{name} seeds must be unique and non-negative")
    for first, second in (
        ("baseline", "development"),
        ("baseline", "holdout"),
        ("development", "holdout"),
    ):
        overlap = set(groups[first]) & set(groups[second])
        if overlap:
            raise ValueError(f"{first}/{second} seed overlap: {sorted(overlap)}")


def validate_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    protocol = dict(payload)
    if protocol.get("schema_version") != M8_SCHEMA_VERSION:
        raise ValueError("unsupported M8 protocol schema version")
    if protocol.get("milestone") != M8_MILESTONE:
        raise ValueError(f"protocol milestone must be {M8_MILESTONE}")
    runtime = protocol.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("protocol.runtime must be an object")
    required_runtime = {
        "control_frequency_hz": CONTROL_FREQUENCY_HZ,
        "chunk_horizon": ACTION_HORIZON,
        "action_dimension": ACTION_DIMENSION,
        "request_interval_steps": REQUEST_INTERVAL_STEPS,
    }
    drift = {
        name: {"expected": expected, "actual": runtime.get(name)}
        for name, expected in required_runtime.items()
        if runtime.get(name) != expected
    }
    if drift:
        raise ValueError(f"frozen ActionStream runtime drift: {drift}")
    safe_hold = runtime.get("safe_hold_command")
    if (
        not isinstance(safe_hold, Sequence)
        or isinstance(safe_hold, (str, bytes))
        or tuple(safe_hold) != SAFE_HOLD_COMMAND
    ):
        raise ValueError(
            "safe_hold_command must remain the exact measured native-reset command "
            f"{list(SAFE_HOLD_COMMAND)}"
        )
    if int(runtime.get("maximum_episode_steps", 0)) <= 0:
        raise ValueError("maximum_episode_steps must be positive")

    task = protocol.get("task")
    if not isinstance(task, Mapping):
        raise ValueError("protocol.task must be an object")
    if task.get("task_id") != "dynamic_target_pick_place_v1":
        raise ValueError("M8 task_id must be dynamic_target_pick_place_v1")
    tolerance = float(task.get("placement_tolerance_m", 0.0))
    if not 0.03 <= tolerance <= 0.05:
        raise ValueError("placement tolerance must remain in [0.03,0.05] m")
    if int(task.get("stable_placement_steps", 0)) < 20:
        raise ValueError("stable placement requires at least 20 control steps")
    if task.get("primary_disturbance") != "destination_switch":
        raise ValueError("primary disturbance must remain destination_switch")
    controller = protocol.get("controller")
    if not isinstance(controller, Mapping):
        raise ValueError("protocol.controller must be an object")
    if controller.get("maximum_translation_scope") != (
        "adjacent rows in the planned chunk; expired-prefix selection is measured "
        "separately at execution"
    ):
        raise ValueError("controller translation limit scope must remain explicit")

    profiles = protocol.get("profiles")
    if not isinstance(profiles, Mapping) or set(profiles) != set(PROFILE_STRATEGIES):
        raise ValueError(f"profiles must be exactly {sorted(PROFILE_STRATEGIES)}")
    for profile_id, strategies in PROFILE_STRATEGIES.items():
        profile_record = profiles[profile_id]
        if not isinstance(profile_record, Mapping):
            raise ValueError(f"protocol profile {profile_id} must be an object")
        configured = tuple(profile_record.get("strategies", ()))
        if configured != strategies:
            raise ValueError(
                f"{profile_id} strategies must be {list(strategies)}, got {list(configured)}"
            )
        profile_file = Path(str(profile_record.get("profile_file", "")))
        profile_sha256 = profile_record.get("profile_sha256")
        if (
            not profile_record.get("profile_file")
            or profile_file.is_absolute()
            or ".." in profile_file.parts
            or not isinstance(profile_sha256, str)
            or len(profile_sha256) != 64
            or any(character not in "0123456789abcdef" for character in profile_sha256)
        ):
            raise ValueError(
                f"protocol profile {profile_id} must bind a portable file and lowercase SHA-256"
            )

    gate = protocol.get("primary_go_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("primary_go_gate must be an object")
    if float(gate.get("minimum_success_difference_percentage_points", -1.0)) != 15.0:
        raise ValueError("M8 primary success threshold must remain 15 percentage points")
    if float(gate.get("minimum_profile_0_success_rate", -1.0)) != 0.90:
        raise ValueError("Profile 0 ceiling threshold must remain 0.90")
    if protocol.get("headline_evidence_class") != NATIVE_ISAAC_EVIDENCE_CLASS:
        raise ValueError("M8 headline evidence must require native Isaac Sim")
    strong_gate = protocol.get("strong_go_gate")
    if not isinstance(strong_gate, Mapping) or float(
        strong_gate.get(
            "minimum_aligned_obsolete_control_step_reduction_vs_naive", -1.0
        )
    ) != STRONG_OBSOLETE_STEP_REDUCTION_THRESHOLD:
        raise ValueError("strong GO obsolete-control-step reduction must remain 20%")
    if protocol.get("paired_reset_fairness") != PAIRED_RESET_FAIRNESS_CONTRACT:
        raise ValueError(
            "paired-reset fairness grids and tolerances must remain exactly preregistered"
        )
    return protocol


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _repository_artifact(
    repository_root: Path,
    value: Any,
    *,
    role: str,
) -> Path:
    relative = Path(str(value or ""))
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{role} must be a portable repository-relative path")
    target = (repository_root / relative).resolve()
    try:
        target.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{role} escapes the repository") from exc
    if not target.is_file():
        raise ValueError(f"{role} does not exist: {target}")
    return target


def _numeric_range(value: Any, *, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{name} must be a two-value range")
    lower, upper = (float(child) for child in value)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(f"{name} must be finite and strictly increasing")
    return lower, upper


def _validate_bounded_calibration_values(protocol: Mapping[str, Any]) -> None:
    controller = protocol["controller"]
    task = protocol["task"]
    workspace_x = _numeric_range(controller.get("workspace_x_m"), name="workspace_x_m")
    workspace_y = _numeric_range(controller.get("workspace_y_m"), name="workspace_y_m")
    workspace_z = _numeric_range(controller.get("workspace_z_m"), name="workspace_z_m")
    if not 0.15 <= workspace_x[0] < workspace_x[1] <= 0.85:
        raise ValueError("workspace_x_m is outside the bounded Franka task workspace")
    if not -0.50 <= workspace_y[0] < workspace_y[1] <= 0.50:
        raise ValueError("workspace_y_m is outside the bounded Franka task workspace")
    if not 0.02 <= workspace_z[0] < workspace_z[1] <= 0.80:
        raise ValueError("workspace_z_m is outside the bounded Franka task workspace")
    object_x = _numeric_range(task.get("object_x_range_m"), name="object_x_range_m")
    object_y = _numeric_range(task.get("object_y_range_m"), name="object_y_range_m")
    if not workspace_x[0] <= object_x[0] < object_x[1] <= workspace_x[1]:
        raise ValueError("object_x_range_m must remain inside the controller workspace")
    if not workspace_y[0] <= object_y[0] < object_y[1] <= workspace_y[1]:
        raise ValueError("object_y_range_m must remain inside the controller workspace")
    cube_side_m = float(task.get("cube_side_m", math.nan))
    if not math.isfinite(cube_side_m) or cube_side_m <= 0.0:
        raise ValueError("task.cube_side_m must remain finite and positive")
    for name in ("zone_a_xy_m", "zone_b_xy_m"):
        value = task.get(name)
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
            or not all(math.isfinite(float(child)) for child in value)
            or not workspace_x[0] <= float(value[0]) <= workspace_x[1]
            or not workspace_y[0] <= float(value[1]) <= workspace_y[1]
        ):
            raise ValueError(f"{name} must remain a finite point inside the workspace")
    if tuple(task["zone_a_xy_m"]) == tuple(task["zone_b_xy_m"]):
        raise ValueError("destination zones must remain distinct")
    switch_steps = task.get("switch_steps")
    maximum_steps = int(protocol["runtime"]["maximum_episode_steps"])
    if (
        not isinstance(switch_steps, Sequence)
        or isinstance(switch_steps, (str, bytes))
        or not switch_steps
        or any(
            type(step) is not int
            or not 1 <= step < maximum_steps
            or step % REQUEST_INTERVAL_STEPS != 0
            for step in switch_steps
        )
        or len(set(switch_steps)) != len(switch_steps)
    ):
        raise ValueError(
            "switch_steps must remain unique request-boundary steps inside the episode"
        )
    bounded_scalars = {
        "approach_height_m": (0.05, workspace_z[1] - workspace_z[0]),
        "grasp_hand_offset_m": (0.02, 0.20),
        "carry_hand_height_m": (workspace_z[0], workspace_z[1]),
        "recovery_hover_offset_m": (0.02, 0.25),
        "maximum_translation_per_step_m": (0.001, 0.02),
    }
    for name, (lower, upper) in bounded_scalars.items():
        value = float(controller.get(name, math.nan))
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"controller.{name} is outside the bounded calibration range")
    lift_clearance = float(task.get("lift_clearance_m", math.nan))
    if not math.isfinite(lift_clearance) or not 0.05 <= lift_clearance <= 0.25:
        raise ValueError("task.lift_clearance_m is outside the bounded calibration range")


def _validate_profile_payload(profile_id: str, payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != M8_SCHEMA_VERSION or payload.get("milestone") != M8_MILESTONE:
        raise ValueError(f"{profile_id} profile schema or milestone mismatch")
    profile = payload.get("profile")
    if not isinstance(profile, Mapping) or profile.get("profile_id") != profile_id:
        raise ValueError(f"{profile_id} profile identity mismatch")
    if tuple(profile.get("strategies", ())) != PROFILE_STRATEGIES[profile_id]:
        raise ValueError(f"{profile_id} strategy set drift")
    integer_names = (
        "base_latency_ms",
        "jitter_ms",
        "extra_delay_ms",
        "duplicate_delivery_offset_ms",
        "communication_pause_ms",
    )
    probability_names = (
        "drop_probability",
        "extra_delay_probability",
        "duplicate_probability",
        "communication_pause_probability",
    )
    if any(type(profile.get(name)) is not int or profile[name] < 0 for name in integer_names):
        raise ValueError(f"{profile_id} latency values must be non-negative integers")
    if any(
        isinstance(profile.get(name), bool)
        or not isinstance(profile.get(name), (int, float))
        or not math.isfinite(float(profile[name]))
        or not 0.0 <= float(profile[name]) <= 1.0
        for name in probability_names
    ):
        raise ValueError(f"{profile_id} probabilities must lie in [0,1]")
    if not isinstance(profile.get("candidate_status"), str) or not profile["candidate_status"]:
        raise ValueError(f"{profile_id} candidate_status must be non-empty")
    if profile_id == "profile_0_sanity":
        if any(profile[name] != 0 for name in (*integer_names, *probability_names)):
            raise ValueError("Profile 0 cannot be changed by development calibration")
    elif profile_id == "profile_1_fixed":
        if not 800 <= profile["base_latency_ms"] <= 950:
            raise ValueError("Profile 1 latency must remain in [800,950] ms")
        if any(
            profile[name] != 0
            for name in (
                "jitter_ms",
                "drop_probability",
                "extra_delay_probability",
                "extra_delay_ms",
                "duplicate_probability",
                "duplicate_delivery_offset_ms",
                "communication_pause_probability",
                "communication_pause_ms",
            )
        ):
            raise ValueError("Profile 1 must remain a fixed-latency no-fault profile")
    else:
        if not 800 <= profile["base_latency_ms"] <= 950:
            raise ValueError("Profile 2 base latency must remain in [800,950] ms")
        if not 150 <= profile["jitter_ms"] <= 250:
            raise ValueError("Profile 2 jitter must remain in [150,250] ms")
        if not 0.02 <= float(profile["drop_probability"]) <= 0.05:
            raise ValueError("Profile 2 drop probability must remain in [0.02,0.05]")
        if not 0.08 <= float(profile["extra_delay_probability"]) <= 0.15:
            raise ValueError("Profile 2 extra-delay probability must remain in [0.08,0.15]")
        if profile["extra_delay_ms"] <= 0:
            raise ValueError("Profile 2 additional delay must remain positive")
        if profile["extra_delay_ms"] > 2000:
            raise ValueError("Profile 2 additional delay must remain at most 2000 ms")
        if not 0.01 <= float(profile["duplicate_probability"]) <= 0.03:
            raise ValueError("Profile 2 duplicate probability must remain in [0.01,0.03]")
        if profile["duplicate_delivery_offset_ms"] <= 0:
            raise ValueError("Profile 2 duplicate delivery offset must remain positive")
        if profile["duplicate_delivery_offset_ms"] > 250:
            raise ValueError("Profile 2 duplicate delivery offset must remain at most 250 ms")
        if not 0.0 <= float(profile["communication_pause_probability"]) <= 0.05:
            raise ValueError("Profile 2 communication-pause probability must remain in [0,0.05]")
        if profile["communication_pause_ms"] > 500:
            raise ValueError("Profile 2 communication pauses must remain at most 500 ms")


def _bound_profile_payloads(
    protocol_payload: Mapping[str, Any],
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    protocol = validate_protocol(protocol_payload)
    _validate_bounded_calibration_values(protocol)
    payloads: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for profile_id, record in protocol["profiles"].items():
        profile_path = _repository_artifact(
            repository_root,
            record["profile_file"],
            role=f"{profile_id} bound profile",
        )
        if sha256_file(profile_path) != record["profile_sha256"]:
            raise ValueError(f"{profile_id} profile hash does not match its protocol binding")
        payload = read_json(profile_path)
        _validate_profile_payload(profile_id, payload)
        payloads[profile_id] = payload
        paths[profile_id] = profile_path
    if payloads["profile_2_faults"]["profile"]["base_latency_ms"] != payloads[
        "profile_1_fixed"
    ]["profile"]["base_latency_ms"]:
        raise ValueError("Profile 2 must reuse the selected Profile 1 base latency")
    return payloads, paths


def _calibration_contract(
    protocol_payload: Mapping[str, Any],
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    protocol = validate_protocol(protocol_payload)
    profile_payloads, profile_paths = _bound_profile_payloads(
        protocol,
        repository_root=repository_root,
    )
    behavioral_protocol = deepcopy(protocol)
    behavioral_protocol.pop("protocol_status", None)
    behavioral_protocol.pop("holdout_freeze_status", None)
    return {
        "protocol": behavioral_protocol,
        "profiles": profile_payloads,
    }, profile_paths


def validate_candidate_contract(
    protocol_payload: Mapping[str, Any],
    *,
    repository_root: Path | str,
) -> dict[str, Any]:
    """Fail closed on the complete pre-development candidate contract.

    Matrix creation calls this before it creates any output directory.  In
    particular, this validates all three byte-bound profile files together so
    Profile 2 cannot silently drift from the selected Profile 1 base latency.
    """

    root = Path(repository_root).resolve()
    contract, profile_paths = _calibration_contract(
        protocol_payload,
        repository_root=root,
    )
    return {
        "contract_sha256": canonical_sha256(contract),
        "profile_sha256": {
            profile_id: sha256_file(profile_path)
            for profile_id, profile_path in sorted(profile_paths.items())
        },
    }


def _contract_diff(
    before: Any,
    after: Any,
    *,
    path: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        if set(before) != set(after):
            raise ValueError(f"calibration cannot add or remove contract fields at {path or '/'}")
        before_diff: dict[str, Any] = {}
        after_diff: dict[str, Any] = {}
        for key in sorted(before):
            child_before, child_after = _contract_diff(
                before[key],
                after[key],
                path=f"{path}/{key}",
            )
            before_diff.update(child_before)
            after_diff.update(child_after)
        return before_diff, after_diff
    if before == after:
        return {}, {}
    return {path: before}, {path: after}


def _require_runtime_behavioral_calibration(
    changed_paths: Iterable[str],
    *,
    change_id: str,
) -> None:
    if not set(changed_paths) & RUNTIME_BEHAVIORAL_CALIBRATION_PATHS:
        raise ValueError(
            f"{change_id} changes only profile binding/status metadata; "
            "at least one runtime behavioral calibration value must change"
        )


def validate_calibration_ledger(
    payload: Mapping[str, Any],
    *,
    require_closed: bool = False,
) -> dict[str, Any]:
    ledger = dict(payload)
    if (
        ledger.get("schema_version") != M8_SCHEMA_VERSION
        or ledger.get("milestone") != M8_MILESTONE
    ):
        raise ValueError("calibration ledger schema or milestone mismatch")
    maximum = int(ledger.get("maximum_bounded_calibration_changes", -1))
    changes = ledger.get("bounded_calibration_changes")
    if maximum != 2 or not isinstance(changes, list) or len(changes) > maximum:
        raise ValueError("M8 permits at most two documented bounded calibration changes")
    for index, change in enumerate(changes):
        if not isinstance(change, Mapping):
            raise ValueError(f"calibration change {index} must be an object")
        missing = [
            name
            for name in (
                "change_id",
                "reason",
                "before_candidate_id",
                "after_candidate_id",
                "before_contract_sha256",
                "after_contract_sha256",
                "before",
                "after",
                "development_evidence",
            )
            if name not in change or change.get(name) in (None, "", {}, [])
        ]
        if missing:
            raise ValueError(f"calibration change {index} missing fields: {missing}")
        expected_before = f"candidate_{index}"
        expected_after = f"candidate_{index + 1}"
        if (
            change.get("change_id") != f"calibration_{index + 1}"
            or change.get("before_candidate_id") != expected_before
            or change.get("after_candidate_id") != expected_after
        ):
            raise ValueError("calibration changes must form the ordered candidate_0 chain")
        before = change.get("before")
        after = change.get("after")
        if (
            not isinstance(before, Mapping)
            or not isinstance(after, Mapping)
            or set(before) != set(after)
            or not before
            or any(not isinstance(path, str) or not path.startswith("/") for path in before)
            or any(before[path] == after[path] for path in before)
        ):
            raise ValueError(
                f"calibration change {index} must record the same changed paths before/after"
            )
        if not _is_sha256(change.get("before_contract_sha256")) or not _is_sha256(
            change.get("after_contract_sha256")
        ):
            raise ValueError(f"calibration change {index} contract hashes are malformed")
        evidence = change.get("development_evidence")
        if not isinstance(evidence, Mapping) or (
            evidence.get("candidate_id") != expected_before
            or not _is_sha256(evidence.get("matrix_sha256"))
            or not _is_sha256(evidence.get("replay_sha256"))
        ):
            raise ValueError(
                f"calibration change {index} must cite replayed evidence for {expected_before}"
            )
    if require_closed:
        baseline = ledger.get("baseline_gate")
        development = ledger.get("development")
        if ledger.get("status") != "closed_before_holdout":
            raise ValueError("calibration ledger must be closed before holdout freeze")
        if not isinstance(baseline, Mapping) or not (
            baseline.get("native_isaac_physics") is True
            and baseline.get("status") == "passed"
            and baseline.get("replay_validated") is True
            and int(baseline.get("trial_count", 0)) >= 20
            and int(baseline.get("success_count", 0)) >= 18
            and int(baseline.get("success_count", 0))
            / int(baseline.get("trial_count", 1))
            >= float(baseline.get("required_success_rate", 0.90))
            and isinstance(baseline.get("replay_sha256"), str)
            and len(baseline.get("replay_sha256")) == 64
            and isinstance(baseline.get("matrix_sha256"), str)
            and len(baseline.get("matrix_sha256")) == 64
            and baseline.get("profile_id") == "profile_0_sanity"
            and baseline.get("strategy") == "sync_hold"
            and baseline.get("evidence_class") == NATIVE_ISAAC_EVIDENCE_CLASS
            and baseline.get("runner_source") in NATIVE_RUNNER_SOURCE_PATHS
            and all(
                isinstance(baseline.get(name), str) and len(baseline.get(name)) == 64
                for name in (
                    "seed_file_sha256",
                    "profile_sha256",
                    "protocol_sha256",
                    "baseline_candidate_protocol_sha256",
                    "behavioral_protocol_contract_sha256",
                    "completion_receipt_sha256",
                    "preflight_receipt_sha256",
                    "source_manifest_sha256",
                    "runner_sha256",
                    "runner_evidence_sha256",
                    "installed_dynamic_adapter_sha256",
                    "installed_dynamic_adapter_evidence_sha256",
                    "executor_sha256",
                    "executor_evidence_sha256",
                    "router_sha256",
                    "process_log_archive_sha256",
                    "process_log_manifest_sha256",
                    *NATIVE_ENVIRONMENT_HASH_FIELDS,
                )
            )
            and int(baseline.get("source_file_count", 0)) > 0
        ):
            raise ValueError("native-Isaac baseline gate has not been validated at >=90%")
        if (
            baseline.get("runner_source") != LINUX_NATIVE_RUNNER_SOURCE_PATH
            or baseline.get("isaac_workspace_commit") != ISAAC_WORKSPACE_COMMIT
            or baseline.get("pixi_version") != OFFICIAL_LINUX_PIXI_VERSION
            or baseline.get("python_version") != EXPECTED_RUNTIME_VERSIONS["python_version"]
            or baseline.get("isaacsim_version") != EXPECTED_RUNTIME_VERSIONS["isaacsim"]
            or baseline.get("ros_distribution")
            != EXPECTED_RUNTIME_VERSIONS["ros_distribution"]
            or baseline.get("rmw_implementation")
            != EXPECTED_RUNTIME_VERSIONS["rmw_implementation"]
            or baseline.get("rmw_zenoh_cpp_version")
            != EXPECTED_RUNTIME_VERSIONS["rmw_zenoh_cpp"]
            or not isinstance(baseline.get("selected_gpu_uuid"), str)
            or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", baseline["selected_gpu_uuid"])
            or not isinstance(baseline.get("selected_gpu_name"), str)
            or not baseline["selected_gpu_name"].strip()
            or not isinstance(baseline.get("selected_gpu_driver_version"), str)
            or not baseline["selected_gpu_driver_version"].strip()
            or type(baseline.get("selected_gpu_memory_total_mib")) is not int
            or baseline["selected_gpu_memory_total_mib"] <= 0
        ):
            raise ValueError("native-Isaac baseline environment/GPU proof is invalid")
        baseline_support_fields = {
            str(name) for name in baseline if str(name).startswith("runner_support_")
        }
        if baseline.get("runner_source") == LINUX_NATIVE_RUNNER_SOURCE_PATH:
            if (
                baseline_support_fields
                != set(LINUX_NATIVE_RUNNER_SUPPORT_BASELINE_FIELDS)
                or baseline.get("runner_support_source")
                != LINUX_NATIVE_RUNNER_SUPPORT_SOURCE_PATH
                or not _is_sha256(baseline.get("runner_support_sha256"))
                or baseline.get("runner_support_sha256")
                != baseline.get("runner_support_evidence_sha256")
            ):
                raise ValueError(
                    "Linux native baseline does not bind exact runner-support evidence"
                )
        elif baseline_support_fields:
            raise ValueError(
                "non-Linux native baseline must not declare runner-support evidence"
            )
        if not isinstance(development, Mapping) or development.get("status") != "complete":
            raise ValueError("development calibration must be complete before freeze")
        candidates = development.get("candidates")
        expected_ids = [f"candidate_{index}" for index in range(len(changes) + 1)]
        if (
            development.get("headline_eligible") is not False
            or int(development.get("seed_count_per_candidate", 0)) not in range(1, 13)
            or not isinstance(candidates, list)
            or len(candidates) != len(expected_ids)
            or any(not isinstance(candidate, Mapping) for candidate in candidates)
            or [candidate.get("candidate_id") for candidate in candidates] != expected_ids
            or development.get("selected_candidate_id") not in expected_ids
        ):
            raise ValueError(
                "development must document candidate_0 plus one candidate per bounded change"
            )
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"development candidate {index} must be an object")
            missing = [
                name
                for name in (
                    "candidate_id",
                    "protocol_path",
                    "protocol_file_sha256",
                    "protocol_sha256",
                    "contract_sha256",
                    "matrix_path",
                    "matrix_sha256",
                    "replay_path",
                    "replay_sha256",
                    "seed_count",
                    "replay_validated",
                    "headline_eligible",
                )
                if name not in candidate
            ]
            if missing:
                raise ValueError(f"development candidate {index} missing fields: {missing}")
            if any(
                not _is_sha256(candidate.get(name))
                for name in (
                    "protocol_file_sha256",
                    "protocol_sha256",
                    "contract_sha256",
                    "matrix_sha256",
                    "replay_sha256",
                )
            ):
                raise ValueError(f"development candidate {index} has malformed hashes")
            if (
                Path(str(candidate.get("protocol_path", ""))).is_absolute()
                or Path(str(candidate.get("matrix_path", ""))).is_absolute()
                or Path(str(candidate.get("replay_path", ""))).is_absolute()
                or not candidate.get("protocol_path")
                or not candidate.get("matrix_path")
                or not candidate.get("replay_path")
                or type(candidate.get("seed_count")) is not int
                or not 1 <= candidate["seed_count"] <= 12
                or candidate.get("replay_validated") is not True
                or candidate.get("headline_eligible") is not False
            ):
                raise ValueError(f"development candidate {index} evidence is malformed")
        candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
        for index, change in enumerate(changes):
            evidence = change["development_evidence"]
            source = candidate_by_id[f"candidate_{index}"]
            if evidence != {
                "candidate_id": source["candidate_id"],
                "matrix_sha256": source["matrix_sha256"],
                "replay_sha256": source["replay_sha256"],
            }:
                raise ValueError(
                    f"calibration change {index} evidence does not match its candidate record"
                )
        if ledger.get("holdout_freeze_authorized") is not True:
            raise ValueError("calibration ledger does not authorize holdout freeze")
        if ledger.get("first_holdout_started") is not False:
            raise ValueError("freeze must be created before the first holdout episode")
    return ledger


def validate_development_calibration_lifecycle(
    *,
    repository_root: Path | str,
    baseline_candidate_protocol_path: Path | str,
    finalized_protocol_payload: Mapping[str, Any],
    ledger_payload: Mapping[str, Any],
    expected_development_seed_path: Path | str | None = None,
    verify_replay: bool = False,
) -> dict[str, Any]:
    """Validate the exact candidate chain and its non-headline evidence.

    Candidate files and latency-profile files are immutable snapshots.  A change
    is accepted only when its recorded before/after map exactly equals the
    structural diff between adjacent snapshots and every changed leaf is in the
    user-authorized bounded calibration allowlist.
    """

    root = Path(repository_root).resolve()
    ledger = validate_calibration_ledger(ledger_payload, require_closed=True)
    baseline_candidate_file = Path(baseline_candidate_protocol_path).resolve()
    try:
        baseline_candidate_file.relative_to(root)
    except ValueError as exc:
        raise ValueError("baseline candidate protocol must be inside the repository") from exc
    if not baseline_candidate_file.is_file():
        raise ValueError("baseline candidate protocol does not exist")
    expected_seed_file = (
        None
        if expected_development_seed_path is None
        else Path(expected_development_seed_path).resolve()
    )
    if expected_seed_file is not None and not expected_seed_file.is_file():
        raise ValueError("development seed file does not exist")

    candidates = ledger["development"]["candidates"]
    candidate_states: dict[str, dict[str, Any]] = {}
    candidate_protocol_paths: dict[str, Path] = {}
    candidate_inputs: list[tuple[str, Path]] = []
    candidate_summaries: list[dict[str, Any]] = []
    for index, record in enumerate(candidates):
        candidate_id = f"candidate_{index}"
        protocol_file = _repository_artifact(
            root,
            record["protocol_path"],
            role=f"{candidate_id} protocol",
        )
        if index == 0 and protocol_file != baseline_candidate_file:
            raise ValueError("candidate_0 must be the immutable baseline candidate protocol")
        protocol = load_protocol(protocol_file)
        if protocol.get("holdout_freeze_status") != "not_yet_frozen":
            raise ValueError(f"{candidate_id} must remain a pre-holdout candidate")
        contract, profile_paths = _calibration_contract(protocol, repository_root=root)
        contract_sha256 = canonical_sha256(contract)
        protocol_sha256 = canonical_sha256(protocol)
        protocol_file_sha256 = sha256_file(protocol_file)
        if (
            record["protocol_file_sha256"] != protocol_file_sha256
            or record["protocol_sha256"] != protocol_sha256
            or record["contract_sha256"] != contract_sha256
        ):
            raise ValueError(f"{candidate_id} protocol or contract hash mismatch")

        matrix_file = _repository_artifact(
            root,
            record["matrix_path"],
            role=f"{candidate_id} development matrix",
        )
        replay_file = _repository_artifact(
            root,
            record["replay_path"],
            role=f"{candidate_id} development replay",
        )
        if (
            record["matrix_sha256"] != sha256_file(matrix_file)
            or record["replay_sha256"] != sha256_file(replay_file)
        ):
            raise ValueError(f"{candidate_id} development artifact hash mismatch")
        matrix = read_json(matrix_file)
        candidate_reference = Path(str(matrix.get("candidate_manifest", "")))
        if not matrix.get("candidate_manifest") or candidate_reference.is_absolute():
            raise ValueError(f"{candidate_id} matrix candidate reference is not portable")
        referenced_candidate = (matrix_file.parent / candidate_reference).resolve()
        if (
            matrix.get("schema_version") != M8_SCHEMA_VERSION
            or matrix.get("milestone") != M8_MILESTONE
            or matrix.get("split") != "development"
            or matrix.get("headline_eligible") is not False
            or referenced_candidate != protocol_file
            or matrix.get("protocol_sha256") != protocol_sha256
        ):
            raise ValueError(f"{candidate_id} matrix does not bind its candidate protocol")
        matrix_seed = Path(str(matrix.get("seed_file", "")))
        if not matrix.get("seed_file") or matrix_seed.is_absolute():
            raise ValueError(f"{candidate_id} matrix seed reference is not portable")
        matrix_seed = (matrix_file.parent / matrix_seed).resolve()
        seeds = load_seed_file(matrix_seed)
        if (
            len(seeds) != record["seed_count"]
            or len(seeds) != ledger["development"]["seed_count_per_candidate"]
            or matrix.get("seed_count") != len(seeds)
            or matrix.get("seed_file_sha256") != sha256_file(matrix_seed)
            or (expected_seed_file is not None and matrix_seed != expected_seed_file)
        ):
            raise ValueError(f"{candidate_id} does not bind the frozen development seeds")
        matrix_profiles = matrix.get("profiles")
        if not isinstance(matrix_profiles, Mapping) or set(matrix_profiles) != {
            "profile_1_fixed",
            "profile_2_faults",
        }:
            raise ValueError(f"{candidate_id} must evaluate both development profiles")
        for profile_id in matrix_profiles:
            profile_record = matrix_profiles[profile_id]
            profile_reference = Path(str(profile_record.get("path", "")))
            if not profile_record.get("path") or profile_reference.is_absolute():
                raise ValueError(f"{candidate_id} {profile_id} matrix path is not portable")
            referenced_profile = (matrix_file.parent / profile_reference).resolve()
            if (
                referenced_profile != profile_paths[profile_id]
                or profile_record.get("sha256") != sha256_file(referenced_profile)
            ):
                raise ValueError(f"{candidate_id} matrix profile binding mismatch: {profile_id}")

        recorded_replay = read_json(replay_file)
        if (
            recorded_replay.get("schema_version") != M8_SCHEMA_VERSION
            or recorded_replay.get("milestone") != M8_MILESTONE
            or recorded_replay.get("split") != "development"
            or recorded_replay.get("manifest_sha256") != sha256_file(matrix_file)
            or recorded_replay.get("passed") is not True
            or int(recorded_replay.get("seed_count", -1)) != len(seeds)
            or int(recorded_replay.get("episode_count", -1))
            != int(matrix.get("expected_episode_count", -2))
            or int(recorded_replay.get("episode_audits_passed", -1))
            != int(recorded_replay.get("episode_count", -2))
        ):
            raise ValueError(f"{candidate_id} recorded development replay is incomplete")
        if verify_replay:
            # Local import avoids a module cycle: m8_replay imports protocol helpers.
            from .m8_replay import validate_manifest

            fresh_replay = validate_manifest(matrix_file)
            if canonical_sha256(fresh_replay) != canonical_sha256(recorded_replay):
                raise ValueError(
                    f"{candidate_id} replay artifact does not match fresh raw replay"
                )

        candidate_states[candidate_id] = contract
        candidate_protocol_paths[candidate_id] = protocol_file
        candidate_inputs.extend(
            (
                (f"development:{candidate_id}:protocol", protocol_file),
                (f"development:{candidate_id}:matrix", matrix_file),
                (f"development:{candidate_id}:replay", replay_file),
                *(
                    (f"development:{candidate_id}:profile:{profile_id}", profile_path)
                    for profile_id, profile_path in sorted(profile_paths.items())
                ),
            )
        )
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "protocol_sha256": protocol_sha256,
                "contract_sha256": contract_sha256,
                "matrix_sha256": sha256_file(matrix_file),
                "replay_sha256": sha256_file(replay_file),
                "seed_count": len(seeds),
            }
        )

    for index, change in enumerate(ledger["bounded_calibration_changes"]):
        before_id = f"candidate_{index}"
        after_id = f"candidate_{index + 1}"
        before_state = candidate_states[before_id]
        after_state = candidate_states[after_id]
        before_diff, after_diff = _contract_diff(before_state, after_state)
        if not before_diff:
            raise ValueError(f"calibration_{index + 1} does not change the candidate contract")
        unauthorized = sorted(set(before_diff) - CALIBRATABLE_CONTRACT_PATHS)
        if unauthorized:
            raise ValueError(
                f"calibration_{index + 1} changes unauthorized contract paths: {unauthorized}"
            )
        _require_runtime_behavioral_calibration(
            before_diff,
            change_id=f"calibration_{index + 1}",
        )
        if dict(change["before"]) != before_diff or dict(change["after"]) != after_diff:
            raise ValueError(
                f"calibration_{index + 1} before/after values do not equal the candidate diff"
            )
        if (
            change["before_contract_sha256"] != canonical_sha256(before_state)
            or change["after_contract_sha256"] != canonical_sha256(after_state)
        ):
            raise ValueError(f"calibration_{index + 1} contract hash chain mismatch")

    selected_id = ledger["development"]["selected_candidate_id"]
    finalized_contract, finalized_profile_paths = _calibration_contract(
        finalized_protocol_payload,
        repository_root=root,
    )
    if canonical_sha256(finalized_contract) != canonical_sha256(candidate_states[selected_id]):
        raise ValueError(
            "finalized behavioral protocol contracts differ from the selected "
            "development candidate"
        )
    selected_record = next(
        record for record in candidates if record["candidate_id"] == selected_id
    )
    return {
        "selected_candidate_id": selected_id,
        "selected_candidate_protocol_path": candidate_protocol_paths[selected_id]
        .relative_to(root)
        .as_posix(),
        "selected_candidate_protocol_file_sha256": sha256_file(
            candidate_protocol_paths[selected_id]
        ),
        "selected_contract_sha256": canonical_sha256(finalized_contract),
        "selected_protocol_sha256": selected_record["protocol_sha256"],
        "candidate_count": len(candidates),
        "bounded_calibration_change_count": len(ledger["bounded_calibration_changes"]),
        "candidates": candidate_summaries,
        "input_role_paths": candidate_inputs,
        "selected_profile_paths": finalized_profile_paths,
    }


def ros_source_manifest(repository_root: Path | str) -> dict[str, Any]:
    """Reproduce the exact source-tree fingerprint emitted by m8_run_isaac.ps1."""

    root = Path(repository_root).resolve()
    source_root = root / "ros2_ws" / "src"
    if not source_root.is_dir():
        raise ValueError(f"ROS source root does not exist: {source_root}")
    excluded_directories = {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "install",
        "log",
    }

    def stable_source_file(path: Path) -> bool:
        relative_parts = path.relative_to(source_root).parts
        directory_parts = {part.casefold() for part in relative_parts[:-1]}
        if directory_parts & excluded_directories:
            return False
        if any(part.casefold().endswith(".egg-info") for part in relative_parts[:-1]):
            return False
        return path.suffix.casefold() not in {".pyc", ".pyo"}

    files = [
        path
        for path in source_root.rglob("*")
        if path.is_file() and stable_source_file(path)
    ]
    files.sort(key=lambda path: str(path).casefold())
    records = [
        f"{path.relative_to(root).as_posix()}|{sha256_file(path)}"
        for path in files
    ]
    digest = hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()
    return {
        "source_root": source_root,
        "source_file_count": len(records),
        "source_manifest_sha256": digest,
    }


def _validated_formal_gpu_identity(preflight: Mapping[str, Any]) -> dict[str, Any]:
    threshold = preflight.get("gpu_memory_refusal_threshold_mib")
    selected_index = preflight.get("selected_gpu_index")
    selected_uuid = preflight.get("selected_gpu_uuid")
    selected_identity = preflight.get("selected_gpu_identity")
    if (
        type(threshold) is not int
        or threshold <= 0
        or type(selected_index) is not int
        or selected_index < 0
        or not isinstance(selected_uuid, str)
        or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", selected_uuid)
        or not isinstance(selected_identity, Mapping)
    ):
        raise ValueError("native preflight selected GPU identity is malformed")
    identities: list[dict[str, Any]] = []
    for field, expected_phase in (
        ("initial_gpu_preflight", "initial_pre_build"),
        ("post_build_gpu_preflight", "post_build_pre_launch"),
    ):
        snapshot = preflight.get(field)
        if not isinstance(snapshot, Mapping):
            raise ValueError(f"native preflight {field} is missing")
        inventory = snapshot.get("gpu_inventory")
        if (
            snapshot.get("phase") != expected_phase
            or snapshot.get("passed") is not True
            or snapshot.get("selected_gpu_index") != selected_index
            or not isinstance(inventory, list)
            or len(inventory) != 1
            or not isinstance(inventory[0], Mapping)
        ):
            raise ValueError("native preflight GPU snapshot identity is malformed")
        row = dict(inventory[0])
        if (
            row.get("index") != selected_index
            or row.get("uuid") != selected_uuid
            or not isinstance(row.get("name"), str)
            or not row["name"].strip()
            or not isinstance(row.get("driver_version"), str)
            or not row["driver_version"].strip()
            or type(row.get("memory_total_mib")) is not int
            or row["memory_total_mib"] <= 0
            or type(row.get("memory_used_mib")) is not int
            or not 0 <= row["memory_used_mib"] <= threshold
            or type(row.get("utilization_gpu_percent")) is not int
            or not 0 <= row["utilization_gpu_percent"] <= 100
        ):
            raise ValueError("native preflight GPU snapshot row is invalid")
        for name in (
            "reported_compute_processes",
            "actionable_compute_processes",
            "blocking_compute_processes",
            "unknown_memory_compute_processes",
            "occupied_gpus",
        ):
            if snapshot.get(name) != []:
                raise ValueError(f"native preflight GPU snapshot {name} is not empty")
        identities.append(
            {
                key: row[key]
                for key in (
                    "index",
                    "name",
                    "uuid",
                    "driver_version",
                    "memory_total_mib",
                )
            }
        )
    if identities[0] != identities[1] or dict(selected_identity) != identities[1]:
        raise ValueError("native preflight selected GPU identity is not cross-linked")
    if (
        preflight.get("gpu_inventory")
        != preflight["post_build_gpu_preflight"]["gpu_inventory"]
        or preflight.get("reported_compute_processes") != []
        or preflight.get("preexisting_compute_processes") != []
    ):
        raise ValueError("native preflight top-level GPU state is not cross-linked")
    return identities[1]


def _validated_workspace_file_evidence(
    record: Any,
    *,
    filename: str,
    environment_file: Path,
    repository_root: Path,
) -> tuple[Path, str]:
    if not isinstance(record, Mapping):
        raise ValueError(f"external environment {filename} record is malformed")
    spec = OFFICIAL_WORKSPACE_FILE_SPECS[filename]
    if (
        not _is_informational_absolute_path(record.get("source_path"))
        or record.get("canonical_lf_size_bytes") != spec["canonical_lf_size_bytes"]
        or record.get("canonical_lf_sha256") != spec["canonical_lf_sha256"]
        or record.get("byte_form") not in {"lf", "crlf"}
    ):
        raise ValueError(f"external environment {filename} canonical record is invalid")
    evidence = _receipt_relative_file(
        record.get("evidence"),
        receipt_file=environment_file,
        repository_root=repository_root,
        name=f"external environment {filename} evidence",
        allow_parent=False,
    )
    data = evidence.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if record.get("size_bytes") != len(data) or record.get("sha256") != digest:
        raise ValueError(f"external environment {filename} evidence binding mismatch")
    if record["byte_form"] == "lf":
        if (
            len(data) != spec["canonical_lf_size_bytes"]
            or digest != spec["canonical_lf_sha256"]
        ):
            raise ValueError(f"external environment {filename} is not the exact LF blob")
    else:
        if (
            len(data) != spec["exact_crlf_size_bytes"]
            or digest != spec["exact_crlf_sha256"]
            or b"\r" in data.replace(b"\r\n", b"")
        ):
            raise ValueError(
                f"external environment {filename} is not the exact all-CRLF form"
            )
        normalized = data.replace(b"\r\n", b"\n")
        if (
            len(normalized) != spec["canonical_lf_size_bytes"]
            or hashlib.sha256(normalized).hexdigest()
            != spec["canonical_lf_sha256"]
        ):
            raise ValueError(
                f"external environment {filename} does not normalize to its Git blob"
            )
    return evidence, digest


def _validated_external_environment(
    preflight: Mapping[str, Any],
    *,
    preflight_file: Path,
    repository_root: Path,
) -> dict[str, Any]:
    environment_file = _receipt_relative_file(
        preflight.get("external_environment_evidence"),
        receipt_file=preflight_file,
        repository_root=repository_root,
        name="external environment evidence",
        allow_parent=False,
    )
    environment_hash = sha256_file(environment_file)
    if preflight.get("external_environment_evidence_sha256") != environment_hash:
        raise ValueError("native preflight external environment evidence hash mismatch")
    environment = read_json(environment_file)
    if not isinstance(environment, Mapping) or (
        environment.get("schema_version") != M8_SCHEMA_VERSION
        or environment.get("milestone") != M8_MILESTONE
        or environment.get("evidence_kind") != "native_external_environment"
        or environment.get("repository_url") != ISAAC_WORKSPACE_REPOSITORY_URL
        or environment.get("workspace_commit") != ISAAC_WORKSPACE_COMMIT
        or environment.get("workspace_relative_path")
        != ISAAC_WORKSPACE_RELATIVE_PATH
        or environment.get("tracked_manifest_lock_clean") is not True
        or not _is_informational_absolute_path(environment.get("repository_root"))
    ):
        raise ValueError("native external environment repository proof is invalid")
    workspace_files = environment.get("workspace_files")
    if not isinstance(workspace_files, Mapping) or set(workspace_files) != set(
        OFFICIAL_WORKSPACE_FILE_SPECS
    ):
        raise ValueError("native external environment workspace file inventory is invalid")
    manifest_evidence, manifest_hash = _validated_workspace_file_evidence(
        workspace_files["pixi.toml"],
        filename="pixi.toml",
        environment_file=environment_file,
        repository_root=repository_root,
    )
    lock_evidence, lock_hash = _validated_workspace_file_evidence(
        workspace_files["pixi.lock"],
        filename="pixi.lock",
        environment_file=environment_file,
        repository_root=repository_root,
    )
    if workspace_files["pixi.toml"].get("byte_form") != workspace_files[
        "pixi.lock"
    ].get("byte_form"):
        raise ValueError(
            "native external environment pixi.toml and pixi.lock must use the "
            "same exact newline form"
        )
    pixi = environment.get("pixi")
    if not isinstance(pixi, Mapping) or (
        not _is_informational_absolute_path(pixi.get("executable"))
        or pixi.get("executable_size_bytes")
        != OFFICIAL_LINUX_PIXI_EXECUTABLE_SIZE_BYTES
        or pixi.get("executable_sha256") != OFFICIAL_LINUX_PIXI_EXECUTABLE_SHA256
        or pixi.get("version") != OFFICIAL_LINUX_PIXI_VERSION
        or pixi.get("version_output") != OFFICIAL_LINUX_PIXI_VERSION_OUTPUT
    ):
        raise ValueError("native external environment Pixi proof is invalid")
    runtime = environment.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("native external environment runtime proof is malformed")
    packages = runtime.get("packages")
    if not isinstance(packages, Mapping):
        raise ValueError("native external environment package proof is malformed")
    for name in (
        "isaacsim",
        "isaacsim-app",
        "isaacsim-core",
        "isaacsim-robot",
        "isaacsim-ros2",
        "rclpy",
        "rosgraph-msgs",
    ):
        if packages.get(name) != EXPECTED_RUNTIME_VERSIONS[name]:
            raise ValueError(f"native external environment {name} version is invalid")
    for name in (
        "python_version",
        "ros_distribution",
        "rmw_implementation",
        "rmw_zenoh_cpp",
    ):
        if runtime.get(name) != EXPECTED_RUNTIME_VERSIONS[name]:
            raise ValueError(f"native external environment {name} is invalid")
    if (
        runtime.get("platform_system") != "Linux"
        or runtime.get("platform_machine") not in {"x86_64", "AMD64"}
        or not _is_informational_absolute_path(runtime.get("python_executable"))
        or not _is_informational_absolute_path(runtime.get("sys_prefix"))
    ):
        raise ValueError("native external environment platform proof is invalid")
    return {
        "external_environment_evidence_path": environment_file,
        "external_environment_evidence_sha256": environment_hash,
        "pixi_manifest_evidence_path": manifest_evidence,
        "pixi_manifest_evidence_sha256": manifest_hash,
        "pixi_lock_evidence_path": lock_evidence,
        "pixi_lock_evidence_sha256": lock_hash,
        "environment": dict(environment),
        "runtime": dict(runtime),
    }


def validate_native_completion_receipt_portability(
    *,
    completion_receipt_path: Path | str,
    repository_root: Path | str,
) -> dict[str, Any]:
    """Validate portable source evidence without dereferencing a rental install.

    Successful native receipts keep absolute runtime paths only as informational
    strings.  Revalidation instead consumes immutable copies stored next to the
    preflight receipt and binds them to repository source bytes.
    """

    root = Path(repository_root).resolve()
    completion_file = Path(completion_receipt_path).resolve()
    try:
        completion_file.relative_to(root)
    except ValueError as exc:
        raise ValueError("completion receipt must be inside the repository") from exc
    if not completion_file.is_file():
        raise ValueError(f"completion receipt does not exist: {completion_file}")
    completion = read_json(completion_file)
    if (
        completion.get("schema_version") != M8_SCHEMA_VERSION
        or completion.get("milestone") != M8_MILESTONE
        or completion.get("receipt_kind") != "native_completion"
        or completion.get("status") != "complete"
    ):
        raise ValueError("receipt is not a successful native completion")

    preflight_file = _receipt_relative_file(
        completion.get("preflight_receipt"),
        receipt_file=completion_file,
        repository_root=root,
        name="preflight receipt",
        allow_parent=True,
    )
    if completion.get("preflight_receipt_sha256") != sha256_file(preflight_file):
        raise ValueError("native completion preflight receipt hash mismatch")
    preflight = read_json(preflight_file)
    if (
        preflight.get("schema_version") != M8_SCHEMA_VERSION
        or preflight.get("milestone") != M8_MILESTONE
        or preflight.get("receipt_kind") != "native_preflight"
        or preflight.get("operator_authorized_native_gpu_run") is not True
    ):
        raise ValueError("referenced receipt is not a successful native preflight")

    source = ros_source_manifest(root)
    source_parts = _portable_path_parts(
        preflight.get("source_root"), name="native preflight source root"
    )
    if ".." in source_parts:
        raise ValueError("native preflight source root escapes the repository")
    recorded_source_root = (root / Path(*source_parts)).resolve()
    if (
        recorded_source_root != source["source_root"]
        or int(preflight.get("source_file_count", -1)) != source["source_file_count"]
        or preflight.get("source_manifest_sha256") != source["source_manifest_sha256"]
    ):
        raise ValueError("native preflight source manifest does not match the current ROS tree")

    if not _is_informational_absolute_path(preflight.get("installed_dynamic_adapter")):
        raise ValueError("installed adapter runtime path must be absolute and informational")
    adapter_source = (
        root
        / "ros2_ws/src/action_stream_isaac/action_stream_isaac/dynamic_isaac_adapter.py"
    )
    if not adapter_source.is_file():
        raise ValueError("repository dynamic adapter source is missing")
    adapter_evidence = _receipt_relative_file(
        preflight.get("installed_dynamic_adapter_evidence"),
        receipt_file=preflight_file,
        repository_root=root,
        name="installed adapter evidence",
        allow_parent=False,
    )
    adapter_hash = sha256_file(adapter_evidence)
    if (
        not _is_sha256(preflight.get("installed_dynamic_adapter_evidence_sha256"))
        or preflight.get("installed_dynamic_adapter_evidence_sha256") != adapter_hash
        or preflight.get("installed_dynamic_adapter_sha256") != adapter_hash
        or sha256_file(adapter_source) != adapter_hash
    ):
        raise ValueError("native preflight portable installed adapter evidence mismatch")

    if not _is_informational_absolute_path(preflight.get("executor_path")):
        raise ValueError("installed executor runtime path must be absolute and informational")
    executor_evidence = _receipt_relative_file(
        preflight.get("executor_evidence"),
        receipt_file=preflight_file,
        repository_root=root,
        name="installed executor evidence",
        allow_parent=False,
    )
    executor_hash = sha256_file(executor_evidence)
    if (
        not _is_sha256(preflight.get("executor_evidence_sha256"))
        or preflight.get("executor_evidence_sha256") != executor_hash
        or preflight.get("executor_sha256") != executor_hash
    ):
        raise ValueError("native preflight portable installed executor evidence mismatch")
    if (
        not _is_informational_absolute_path(preflight.get("router_path"))
        or not _is_sha256(preflight.get("router_sha256"))
    ):
        raise ValueError("native preflight Zenoh router provenance is malformed")

    runner_relative, runner_source = _repository_relative_file(
        preflight.get("runner_source"),
        repository_root=root,
        name="native runner source",
    )
    if runner_relative not in NATIVE_RUNNER_SOURCE_PATHS:
        raise ValueError("native runner source is not an allowed M8 runner")
    if runner_relative != LINUX_NATIVE_RUNNER_SOURCE_PATH:
        raise ValueError(
            "formal native portability requires the Linux runner with exact "
            "external-environment and selected-GPU evidence"
        )
    runtime_runner = preflight.get("runner")
    if runtime_runner is not None and not _is_informational_absolute_path(runtime_runner):
        raise ValueError("native runner runtime path must be absolute and informational")
    runner_evidence = _receipt_relative_file(
        preflight.get("runner_evidence"),
        receipt_file=preflight_file,
        repository_root=root,
        name="native runner evidence",
        allow_parent=False,
    )
    runner_hash = sha256_file(runner_evidence)
    if (
        not _is_sha256(preflight.get("runner_evidence_sha256"))
        or preflight.get("runner_evidence_sha256") != runner_hash
        or preflight.get("runner_sha256") != runner_hash
        or sha256_file(runner_source) != runner_hash
    ):
        raise ValueError("native preflight portable runner evidence mismatch")

    runner_support_source: str | None = None
    runner_support_evidence: Path | None = None
    runner_support_hash: str | None = None
    support_fields_present = {
        str(name) for name in preflight if str(name).startswith("runner_support_")
    }
    if runner_relative == LINUX_NATIVE_RUNNER_SOURCE_PATH:
        if support_fields_present != set(LINUX_NATIVE_RUNNER_SUPPORT_FIELDS):
            raise ValueError(
                "Linux native preflight must bind every runner-support evidence field"
            )
        support_relative, support_source = _repository_relative_file(
            preflight.get("runner_support_source"),
            repository_root=root,
            name="Linux native runner support source",
        )
        if support_relative != LINUX_NATIVE_RUNNER_SUPPORT_SOURCE_PATH:
            raise ValueError("Linux native preflight binds an unexpected runner support")
        runner_support_evidence = _receipt_relative_file(
            preflight.get("runner_support_evidence"),
            receipt_file=preflight_file,
            repository_root=root,
            name="Linux native runner support evidence",
            allow_parent=False,
        )
        runner_support_hash = sha256_file(runner_support_evidence)
        if (
            not _is_sha256(preflight.get("runner_support_evidence_sha256"))
            or preflight.get("runner_support_evidence_sha256")
            != runner_support_hash
            or preflight.get("runner_support_sha256") != runner_support_hash
            or sha256_file(support_source) != runner_support_hash
        ):
            raise ValueError(
                "native preflight portable Linux runner-support evidence mismatch"
            )
        runner_support_source = support_relative
    elif support_fields_present:
        raise ValueError(
            "non-Linux native preflight must not declare runner-support evidence"
        )

    external_environment = _validated_external_environment(
        preflight,
        preflight_file=preflight_file,
        repository_root=root,
    )
    selected_gpu = _validated_formal_gpu_identity(preflight)

    return {
        **external_environment,
        "completion_receipt_path": completion_file,
        "completion_receipt_sha256": sha256_file(completion_file),
        "preflight_receipt_path": preflight_file,
        "preflight_receipt_sha256": sha256_file(preflight_file),
        "source_file_count": source["source_file_count"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "installed_dynamic_adapter_evidence_path": adapter_evidence,
        "installed_dynamic_adapter_evidence_sha256": adapter_hash,
        "installed_dynamic_adapter_sha256": adapter_hash,
        "executor_evidence_path": executor_evidence,
        "executor_evidence_sha256": executor_hash,
        "executor_sha256": executor_hash,
        "router_sha256": preflight["router_sha256"],
        "runner_source": runner_relative,
        "runner_evidence_path": runner_evidence,
        "runner_evidence_sha256": runner_hash,
        "runner_sha256": runner_hash,
        "runner_support_source": runner_support_source,
        "runner_support_evidence_path": runner_support_evidence,
        "runner_support_evidence_sha256": runner_support_hash,
        "runner_support_sha256": runner_support_hash,
        "selected_gpu_identity": selected_gpu,
    }


def validate_native_completion_receipt(
    *,
    completion_receipt_path: Path | str,
    matrix_path: Path | str,
    replay_path: Path | str,
    repository_root: Path | str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    portable = validate_native_completion_receipt_portability(
        completion_receipt_path=completion_receipt_path,
        repository_root=root,
    )
    completion_file = portable["completion_receipt_path"]
    matrix_file = Path(matrix_path).resolve()
    replay_file = Path(replay_path).resolve()
    for artifact_name, artifact_path in (
        ("baseline matrix", matrix_file),
        ("baseline replay", replay_file),
    ):
        try:
            artifact_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{artifact_name} must be inside the repository") from exc
    completion = read_json(completion_file)
    recorded_replay = _receipt_relative_file(
        completion.get("replay_validation"),
        receipt_file=completion_file,
        repository_root=root,
        name="receipt replay",
        allow_parent=True,
    )
    if (
        recorded_replay != replay_file
        or completion.get("replay_validation_sha256") != sha256_file(replay_file)
    ):
        raise ValueError("native completion receipt does not bind the baseline replay")
    preflight_file = portable["preflight_receipt_path"]
    preflight = read_json(preflight_file)
    if preflight.get("preexisting_compute_processes") not in ([], ()):
        raise ValueError("referenced receipt is not a successful native preflight")
    matrix_payload = read_json(matrix_file)
    matrix_split = matrix_payload.get("split")
    if matrix_split not in {"baseline_gate", "development", "frozen_holdout"}:
        raise ValueError("native completion matrix split is invalid")
    expected_frozen_validation = matrix_split == "frozen_holdout"
    if preflight.get("frozen_live_inputs_validated") is not expected_frozen_validation:
        raise ValueError(
            "native preflight frozen-live-input validation does not match the matrix split"
        )
    batch_records = matrix_payload.get("batch_manifests")
    if (
        not isinstance(batch_records, list)
        or not batch_records
        or type(matrix_payload.get("persistent_batch_count")) is not int
        or matrix_payload["persistent_batch_count"] != len(batch_records)
    ):
        raise ValueError("baseline matrix has no exact persistent batch inventory")
    expected_batch_validation_logs: list[str] = []
    expected_batch_process_logs: list[str] = []
    seen_batch_paths: set[Path] = set()
    for index, record in enumerate(batch_records):
        if not isinstance(record, Mapping) or not record.get("path"):
            raise ValueError(f"baseline batch record {index} is malformed")
        relative_batch = Path(str(record["path"]))
        if relative_batch.is_absolute() or ".." in relative_batch.parts:
            raise ValueError(f"baseline batch record {index} path is not portable")
        batch_file = (matrix_file.parent / relative_batch).resolve()
        try:
            batch_file.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"baseline batch record {index} escapes the repository") from exc
        if not batch_file.is_file() or batch_file in seen_batch_paths:
            raise ValueError(f"baseline batch record {index} is missing or duplicated")
        seen_batch_paths.add(batch_file)
        log_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", batch_file.stem)
        expected_batch_validation_logs.append(f"{log_stem}.validate_only.log")
        expected_batch_process_logs.extend(
            f"{log_stem}.{process}.{stream}.log"
            for process in ("executor", "adapter")
            for stream in ("stdout", "stderr")
        )
    if len(set(expected_batch_validation_logs)) != len(expected_batch_validation_logs):
        raise ValueError("baseline batch validation log names are not unique")
    recorded_validation_logs = preflight.get("current_source_batch_validation_logs")
    if (
        preflight.get("current_source_batch_validation_passed") is not True
        or type(preflight.get("current_source_batch_validation_count")) is not int
        or preflight["current_source_batch_validation_count"] != len(batch_records)
        or not isinstance(recorded_validation_logs, list)
        or recorded_validation_logs != expected_batch_validation_logs
    ):
        raise ValueError(
            "native preflight does not prove exact current-source batch validation"
        )
    gpu_threshold = preflight.get("gpu_memory_refusal_threshold_mib")
    gpu_inventory = preflight.get("gpu_inventory")
    if (
        not isinstance(gpu_threshold, int)
        or isinstance(gpu_threshold, bool)
        or gpu_threshold <= 0
        or not isinstance(gpu_inventory, list)
        or not gpu_inventory
    ):
        raise ValueError("native preflight GPU inventory/threshold is missing or malformed")
    seen_gpu_indices: set[int] = set()
    seen_gpu_uuids: set[str] = set()
    for gpu in gpu_inventory:
        if not isinstance(gpu, Mapping):
            raise ValueError("native preflight GPU inventory row is malformed")
        index = gpu.get("index")
        total = gpu.get("memory_total_mib")
        used = gpu.get("memory_used_mib")
        utilization = gpu.get("utilization_gpu_percent")
        uuid = gpu.get("uuid")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index in seen_gpu_indices
            or not isinstance(total, int)
            or isinstance(total, bool)
            or total <= 0
            or not isinstance(used, int)
            or isinstance(used, bool)
            or used < 0
            or used > total
            or used > gpu_threshold
            or not isinstance(utilization, int)
            or isinstance(utilization, bool)
            or not 0 <= utilization <= 100
            or not isinstance(uuid, str)
            or not uuid.strip()
            or uuid in seen_gpu_uuids
            or not isinstance(gpu.get("name"), str)
            or not gpu["name"].strip()
            or not isinstance(gpu.get("driver_version"), str)
            or not gpu["driver_version"].strip()
        ):
            raise ValueError("native preflight GPU inventory row is invalid or over threshold")
        seen_gpu_indices.add(index)
        seen_gpu_uuids.add(uuid)
    reported_compute = preflight.get("reported_compute_processes")
    if not isinstance(reported_compute, list):
        raise ValueError("native preflight compute-process inventory is malformed")
    for process in reported_compute:
        if not isinstance(process, Mapping):
            raise ValueError("native preflight compute-process row is malformed")
        used_memory = process.get("used_memory_mib")
        actionable = process.get("actionable_compute_allocation")
        if (
            not isinstance(process.get("gpu_uuid"), str)
            or not process["gpu_uuid"].strip()
            or not isinstance(process.get("process_id"), int)
            or isinstance(process.get("process_id"), bool)
            or process["process_id"] <= 0
            or not isinstance(process.get("process_name"), str)
            or not process["process_name"].strip()
            or (
                used_memory is not None
                and (
                    not isinstance(used_memory, int)
                    or isinstance(used_memory, bool)
                    or used_memory < 0
                )
            )
            or not isinstance(actionable, bool)
            or actionable != (isinstance(used_memory, int) and used_memory > 0)
            or actionable
        ):
            raise ValueError("native preflight reports competing compute use")
    recorded_matrix_value = Path(str(preflight.get("suite_manifest", "")))
    if recorded_matrix_value.is_absolute() or not preflight.get("suite_manifest"):
        raise ValueError("native preflight suite path must be repository-relative")
    recorded_matrix = (root / recorded_matrix_value).resolve()
    if (
        recorded_matrix != matrix_file
        or preflight.get("suite_manifest_sha256") != sha256_file(matrix_file)
    ):
        raise ValueError("native preflight receipt does not bind the baseline matrix")
    log_archive = _receipt_relative_file(
        completion.get("process_log_archive"),
        receipt_file=completion_file,
        repository_root=root,
        name="native process-log archive",
        allow_parent=True,
    )
    log_manifest = _receipt_relative_file(
        completion.get("process_log_manifest"),
        receipt_file=completion_file,
        repository_root=root,
        name="native process-log manifest",
        allow_parent=True,
    )
    if (
        completion.get("process_log_archive_sha256") != sha256_file(log_archive)
        or completion.get("process_log_manifest_sha256") != sha256_file(log_manifest)
    ):
        raise ValueError("native completion process-log archive binding mismatch")
    # Local import avoids a module cycle: m8_archive imports protocol constants.
    from .m8_archive import validate_archive

    log_archive_audit = validate_archive(log_archive, log_manifest)
    if log_archive_audit.get("passed") is not True:
        raise ValueError(
            "native process-log archive failed exact validation: "
            f"{log_archive_audit.get('errors', [])}"
        )
    log_manifest_payload = read_json(log_manifest)
    log_members = log_manifest_payload.get("members")
    if not isinstance(log_members, list) or not log_members:
        raise ValueError("native process-log archive manifest has no members")
    logged_names = {
        str(record.get("path", ""))
        for record in log_members
        if isinstance(record, Mapping)
    }
    if (
        len(logged_names) != len(log_members)
        or any(not name or Path(name).name != name or not name.endswith(".log") for name in logged_names)
        or isinstance(completion.get("process_log_member_count"), bool)
        or completion.get("process_log_member_count") != len(logged_names)
    ):
        raise ValueError("native process-log archive member inventory is malformed")
    planned_records = preflight.get("planned_process_logs")
    if (
        not isinstance(planned_records, list)
        or not planned_records
        or any(
            not isinstance(value, str)
            or not value
            or "\\" in value
            or Path(value).name != value
            or not value.endswith(".log")
            for value in planned_records
        )
        or len(set(planned_records)) != len(planned_records)
    ):
        raise ValueError("native preflight planned process-log inventory is malformed")
    planned_names = set(planned_records)
    if not planned_names or not planned_names.issubset(logged_names):
        raise ValueError("native process-log archive is missing planned process logs")
    required_process_logs = {
        "colcon_build.log",
        "replay_validate.log",
        "router.stdout.log",
        "router.stderr.log",
        *expected_batch_validation_logs,
        *expected_batch_process_logs,
    }
    if expected_frozen_validation:
        required_process_logs.update({"freeze_validate.log", "analysis_figures.log"})
    if not required_process_logs.issubset(planned_names) or not (
        required_process_logs.issubset(logged_names)
    ):
        raise ValueError(
            "native process-log archive does not bind every required runner log"
        )
    expected_validation_names = set(expected_batch_validation_logs)
    if not expected_validation_names.issubset(planned_names) or not (
        expected_validation_names.issubset(logged_names)
    ):
        raise ValueError(
            "native process-log archive does not bind every batch validation log"
        )
    return {
        **portable,
        "process_log_archive_path": log_archive,
        "process_log_archive_sha256": sha256_file(log_archive),
        "process_log_manifest_path": log_manifest,
        "process_log_manifest_sha256": sha256_file(log_manifest),
    }


def validate_baseline_gate_artifacts(
    *,
    matrix_path: Path | str,
    replay_path: Path | str,
    baseline_seed_path: Path | str,
    profile_path: Path | str,
    protocol_payload: Mapping[str, Any],
    completion_receipt_path: Path | str,
    repository_root: Path | str,
    ledger_payload: Mapping[str, Any] | None = None,
    require_closed_lifecycle: bool = True,
) -> dict[str, Any]:
    """Replay and bind the completed native Profile-0 gate.

    The pre-development recording command uses ``require_closed_lifecycle=False``.
    Freeze validation keeps the default and additionally verifies the selected
    development candidate, finalized protocol, and closed ledger.
    """

    root = Path(repository_root).resolve()
    matrix_file = Path(matrix_path).resolve()
    replay_file = Path(replay_path).resolve()
    if not matrix_file.is_file() or not replay_file.is_file():
        raise ValueError("baseline matrix and replay artifacts must both exist")
    matrix = read_json(matrix_file)
    protocol = validate_protocol(protocol_payload)
    if (
        matrix.get("schema_version") != M8_SCHEMA_VERSION
        or matrix.get("milestone") != M8_MILESTONE
        or matrix.get("split") != "baseline_gate"
        or matrix.get("evidence_class") != NATIVE_ISAAC_EVIDENCE_CLASS
        or matrix.get("headline_eligible") is not False
    ):
        raise ValueError("baseline matrix is not native non-headline M8 baseline evidence")
    candidate_value = matrix.get("candidate_manifest")
    candidate_relative = Path(str(candidate_value or ""))
    if not candidate_value or candidate_relative.is_absolute():
        raise ValueError("baseline matrix must reference a portable immutable candidate protocol")
    candidate_file = (matrix_file.parent / candidate_relative).resolve()
    if not candidate_file.is_file():
        raise ValueError("baseline candidate protocol artifact does not exist")
    candidate = load_protocol(candidate_file)
    validate_candidate_contract(candidate, repository_root=root)
    if (
        candidate.get("protocol_status")
        != "preregistered_candidate_pending_native_baseline_and_development"
        or candidate.get("holdout_freeze_status") != "not_yet_frozen"
    ):
        raise ValueError("baseline candidate protocol status is invalid")
    candidate_sha256 = canonical_sha256(candidate)
    if require_closed_lifecycle:
        if (
            protocol.get("protocol_status") != "ready_for_holdout_freeze"
            or protocol.get("holdout_freeze_status") != "ready_to_freeze"
        ):
            raise ValueError("baseline-to-freeze protocol status transition is invalid")
    elif canonical_sha256(protocol) != candidate_sha256:
        raise ValueError("pre-development baseline recording requires its exact candidate protocol")
    if matrix.get("protocol_sha256") != candidate_sha256:
        raise ValueError("baseline matrix does not bind its immutable candidate protocol bytes")
    profiles = matrix.get("profiles")
    if not isinstance(profiles, Mapping) or set(profiles) != {"profile_0_sanity"}:
        raise ValueError("baseline matrix must contain exactly Profile 0")
    seed_file = Path(str(matrix.get("seed_file", "")))
    if seed_file.is_absolute() or not matrix.get("seed_file"):
        raise ValueError("baseline matrix seed path must be portable")
    seed_file = (matrix_file.parent / seed_file).resolve()
    expected_seed_file = Path(baseline_seed_path).resolve()
    if (
        seed_file != expected_seed_file
        or not seed_file.is_file()
        or matrix.get("seed_file_sha256") != sha256_file(expected_seed_file)
    ):
        raise ValueError("baseline matrix does not bind the frozen baseline seed file")
    profile_record = profiles["profile_0_sanity"]
    if not isinstance(profile_record, Mapping) or not profile_record.get("path"):
        raise ValueError("baseline matrix Profile-0 record is malformed")
    matrix_profile = Path(str(profile_record["path"]))
    if matrix_profile.is_absolute():
        raise ValueError("baseline matrix Profile-0 path must be portable")
    matrix_profile = (matrix_file.parent / matrix_profile).resolve()
    expected_profile = Path(profile_path).resolve()
    if (
        matrix_profile != expected_profile
        or not matrix_profile.is_file()
        or profile_record.get("sha256") != sha256_file(expected_profile)
    ):
        raise ValueError("baseline matrix does not bind the frozen Profile-0 config")
    episodes = matrix.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < 20:
        raise ValueError("baseline matrix must contain at least 20 completed trials")
    if int(matrix.get("expected_episode_count", -1)) != len(episodes):
        raise ValueError("baseline matrix expected episode count does not match its trials")
    seeds = [int(entry.get("seed", -1)) for entry in episodes if isinstance(entry, Mapping)]
    if (
        len(seeds) != len(episodes)
        or len(set(seeds)) != len(episodes)
        or any(seed < 0 for seed in seeds)
        or any(
            entry.get("profile_id") != "profile_0_sanity"
            or entry.get("strategy") != "sync_hold"
            for entry in episodes
        )
    ):
        raise ValueError("baseline trials must be one Profile-0 sync episode per unique seed")
    if tuple(seeds) != load_seed_file(expected_seed_file):
        raise ValueError("baseline trial seeds do not exactly match the frozen baseline seed list")

    # Local import avoids a module cycle: m8_replay imports protocol helpers.
    from .m8_replay import validate_manifest

    fresh = validate_manifest(matrix_file)
    recorded = read_json(replay_file)
    matrix_sha256 = sha256_file(matrix_file)
    replay_sha256 = sha256_file(replay_file)
    if fresh.get("manifest_sha256") != matrix_sha256 or fresh.get("split") != "baseline_gate":
        raise ValueError("fresh baseline replay did not bind the exact matrix bytes")
    if canonical_sha256(recorded) != canonical_sha256(fresh):
        raise ValueError("baseline replay artifact does not match a fresh raw-evidence replay")
    receipt = validate_native_completion_receipt(
        completion_receipt_path=completion_receipt_path,
        matrix_path=matrix_file,
        replay_path=replay_file,
        repository_root=repository_root,
    )
    audits = fresh.get("audits")
    if (
        fresh.get("passed") is not True
        or fresh.get("provenance_validation_passed") is not True
        or fresh.get("seed_validation_passed") is not True
        or fresh.get("profile_validation_passed") is not True
        or fresh.get("fairness_passed") is not True
        or int(fresh.get("episode_count", -1)) != len(episodes)
        or int(fresh.get("episode_audits_passed", -1)) != len(episodes)
        or int(fresh.get("seed_count", -1)) != len(seeds)
        or fresh.get("seed_file_sha256") != sha256_file(expected_seed_file)
        or not isinstance(audits, list)
        or len(audits) != len(episodes)
        or any(
            audit.get("passed") is not True or audit.get("strategy") != "sync_hold"
            for audit in audits
        )
    ):
        raise ValueError("baseline raw replay is incomplete or contains a failed audit")
    success_count = sum(
        audit.get("recomputed_metrics", {}).get("task_success") is True for audit in audits
    )
    trial_count = len(audits)
    observed_rate = success_count / trial_count
    if success_count < 18 or observed_rate < 0.90:
        raise ValueError("native baseline gate requires at least 18/20 replay-clean successes")

    try:
        candidate_relative_to_root = candidate_file.relative_to(root).as_posix()
        matrix_relative_to_root = matrix_file.relative_to(root).as_posix()
        replay_relative_to_root = replay_file.relative_to(root).as_posix()
        seed_relative_to_root = expected_seed_file.relative_to(root).as_posix()
        profile_relative_to_root = expected_profile.relative_to(root).as_posix()
        completion_relative_to_root = Path(completion_receipt_path).resolve().relative_to(
            root
        ).as_posix()
    except ValueError as exc:
        raise ValueError("baseline evidence must remain inside the repository") from exc
    environment = receipt["environment"]
    runtime = receipt["runtime"]
    runtime_packages = runtime["packages"]
    selected_gpu = receipt["selected_gpu_identity"]
    evidence_summary = {
        "native_isaac_physics": True,
        "status": "passed",
        "matrix_path": matrix_relative_to_root,
        "replay_path": replay_relative_to_root,
        "seed_file_path": seed_relative_to_root,
        "profile_path": profile_relative_to_root,
        "completion_receipt_path": completion_relative_to_root,
        "baseline_candidate_protocol_path": candidate_relative_to_root,
        "baseline_candidate_protocol_file_sha256": sha256_file(candidate_file),
        "matrix_sha256": matrix_sha256,
        "replay_sha256": replay_sha256,
        "seed_count": len(seeds),
        "trial_count": trial_count,
        "success_count": success_count,
        "required_success_rate": 0.90,
        "observed_success_rate": observed_rate,
        "replay_validated": True,
        "profile_id": "profile_0_sanity",
        "strategy": "sync_hold",
        "evidence_class": NATIVE_ISAAC_EVIDENCE_CLASS,
        "seed_file_sha256": sha256_file(expected_seed_file),
        "profile_sha256": sha256_file(expected_profile),
        "baseline_candidate_protocol_sha256": candidate_sha256,
        "completion_receipt_sha256": receipt["completion_receipt_sha256"],
        "preflight_receipt_sha256": receipt["preflight_receipt_sha256"],
        "source_file_count": receipt["source_file_count"],
        "source_manifest_sha256": receipt["source_manifest_sha256"],
        "runner_source": receipt["runner_source"],
        "runner_sha256": receipt["runner_sha256"],
        "runner_evidence_sha256": receipt["runner_evidence_sha256"],
        "external_environment_evidence_sha256": receipt[
            "external_environment_evidence_sha256"
        ],
        "pixi_manifest_evidence_sha256": receipt[
            "pixi_manifest_evidence_sha256"
        ],
        "pixi_lock_evidence_sha256": receipt["pixi_lock_evidence_sha256"],
        "isaac_workspace_commit": environment["workspace_commit"],
        "pixi_version": environment["pixi"]["version"],
        "python_version": runtime["python_version"],
        "isaacsim_version": runtime_packages["isaacsim"],
        "ros_distribution": runtime["ros_distribution"],
        "rmw_implementation": runtime["rmw_implementation"],
        "rmw_zenoh_cpp_version": runtime["rmw_zenoh_cpp"],
        "selected_gpu_uuid": selected_gpu["uuid"],
        "selected_gpu_name": selected_gpu["name"],
        "selected_gpu_driver_version": selected_gpu["driver_version"],
        "selected_gpu_memory_total_mib": selected_gpu["memory_total_mib"],
        "installed_dynamic_adapter_sha256": receipt[
            "installed_dynamic_adapter_sha256"
        ],
        "installed_dynamic_adapter_evidence_sha256": receipt[
            "installed_dynamic_adapter_evidence_sha256"
        ],
        "executor_sha256": receipt["executor_sha256"],
        "executor_evidence_sha256": receipt["executor_evidence_sha256"],
        "router_sha256": receipt["router_sha256"],
        "process_log_archive_sha256": receipt["process_log_archive_sha256"],
        "process_log_manifest_sha256": receipt["process_log_manifest_sha256"],
    }
    if receipt["runner_support_source"] is not None:
        evidence_summary.update(
            {
                "runner_support_source": receipt["runner_support_source"],
                "runner_support_sha256": receipt["runner_support_sha256"],
                "runner_support_evidence_sha256": receipt[
                    "runner_support_evidence_sha256"
                ],
            }
        )
    if not require_closed_lifecycle:
        return evidence_summary

    if ledger_payload is None:
        raise ValueError("closed baseline validation requires a calibration ledger")

    lifecycle = validate_development_calibration_lifecycle(
        repository_root=repository_root,
        baseline_candidate_protocol_path=candidate_file,
        finalized_protocol_payload=protocol,
        ledger_payload=ledger_payload,
        verify_replay=False,
    )
    ledger = validate_calibration_ledger(ledger_payload, require_closed=True)
    baseline = ledger["baseline_gate"]
    expected_ledger = {
        **{
            name: evidence_summary[name]
            for name in (
                "native_isaac_physics",
                "status",
                "seed_count",
                "trial_count",
                "success_count",
                "required_success_rate",
                "replay_validated",
                "replay_sha256",
                "matrix_sha256",
                "profile_id",
                "strategy",
                "evidence_class",
                "seed_file_sha256",
                "profile_sha256",
                "baseline_candidate_protocol_sha256",
                "completion_receipt_sha256",
                "preflight_receipt_sha256",
                "source_file_count",
                "source_manifest_sha256",
                "runner_source",
                "runner_sha256",
                "runner_evidence_sha256",
                "installed_dynamic_adapter_sha256",
                "installed_dynamic_adapter_evidence_sha256",
                "executor_sha256",
                "executor_evidence_sha256",
                "router_sha256",
                "process_log_archive_sha256",
                "process_log_manifest_sha256",
            )
        },
        "protocol_sha256": canonical_sha256(protocol),
        "behavioral_protocol_contract_sha256": lifecycle[
            "selected_contract_sha256"
        ],
        "selected_development_candidate_id": lifecycle["selected_candidate_id"],
        "bounded_calibration_change_count": lifecycle[
            "bounded_calibration_change_count"
        ],
    }
    for name in LINUX_NATIVE_RUNNER_SUPPORT_BASELINE_FIELDS:
        if name in evidence_summary:
            expected_ledger[name] = evidence_summary[name]
    for name in NATIVE_ENVIRONMENT_LEDGER_FIELDS:
        expected_ledger[name] = evidence_summary[name]
    drift = {
        name: {"expected": expected, "actual": baseline.get(name)}
        for name, expected in expected_ledger.items()
        if baseline.get(name) != expected
    }
    try:
        recorded_rate = float(baseline.get("observed_success_rate"))
    except (TypeError, ValueError):
        recorded_rate = math.nan
    if not math.isclose(recorded_rate, observed_rate, rel_tol=0.0, abs_tol=1e-12):
        drift["observed_success_rate"] = {
            "expected": observed_rate,
            "actual": baseline.get("observed_success_rate"),
        }
    if drift:
        raise ValueError(f"baseline calibration ledger does not match replayed evidence: {drift}")
    return {
        **evidence_summary,
        "protocol_sha256": canonical_sha256(protocol),
        "behavioral_protocol_contract_sha256": lifecycle[
            "selected_contract_sha256"
        ],
        "selected_development_candidate_id": lifecycle["selected_candidate_id"],
        "bounded_calibration_change_count": lifecycle[
            "bounded_calibration_change_count"
        ],
    }


def load_protocol(path: Path | str) -> dict[str, Any]:
    return validate_protocol(read_json(path))


def _repository_file_argument(
    repository_root: Path,
    value: Path | str,
    *,
    role: str,
) -> tuple[Path, str]:
    target = Path(value).resolve()
    try:
        relative = target.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{role} must be inside the repository") from exc
    if not target.is_file():
        raise ValueError(f"{role} does not exist: {target}")
    return target, relative


def _write_calibration_ledger_update(
    ledger_path: Path,
    *,
    original: Mapping[str, Any],
    updated: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace a ledger atomically after a best-effort compare-and-swap check."""

    current = read_json(ledger_path)
    if canonical_sha256(current) != canonical_sha256(original):
        raise RuntimeError("calibration ledger changed while the update was being validated")
    payload = dict(updated)
    write_json_atomic(ledger_path, payload)
    return payload


def _validate_recorded_baseline_stage(
    *,
    repository_root: Path,
    ledger_payload: Mapping[str, Any],
) -> dict[str, Any]:
    ledger = validate_calibration_ledger(ledger_payload)
    baseline = ledger.get("baseline_gate")
    if not isinstance(baseline, Mapping) or baseline.get("status") != "passed":
        raise ValueError("native baseline must be recorded before development")
    required_paths = {
        "matrix_path": "baseline matrix",
        "replay_path": "baseline replay",
        "seed_file_path": "baseline seeds",
        "profile_path": "baseline Profile 0",
        "completion_receipt_path": "baseline completion receipt",
        "baseline_candidate_protocol_path": "baseline candidate protocol",
    }
    resolved: dict[str, Path] = {}
    for field, role in required_paths.items():
        resolved[field] = _repository_artifact(
            repository_root,
            baseline.get(field),
            role=role,
        )
    candidate = load_protocol(resolved["baseline_candidate_protocol_path"])
    evidence = validate_baseline_gate_artifacts(
        matrix_path=resolved["matrix_path"],
        replay_path=resolved["replay_path"],
        baseline_seed_path=resolved["seed_file_path"],
        profile_path=resolved["profile_path"],
        protocol_payload=candidate,
        completion_receipt_path=resolved["completion_receipt_path"],
        repository_root=repository_root,
        require_closed_lifecycle=False,
    )
    drift = {
        name: {"expected": expected, "actual": baseline.get(name)}
        for name, expected in evidence.items()
        if baseline.get(name) != expected
    }
    if drift:
        raise ValueError(f"recorded native baseline evidence has drifted: {drift}")
    return evidence


def record_native_baseline_gate(
    *,
    repository_root: Path | str,
    ledger_path: Path | str,
    matrix_path: Path | str,
    replay_path: Path | str,
    baseline_seed_path: Path | str,
    profile_path: Path | str,
    completion_receipt_path: Path | str,
) -> dict[str, Any]:
    """Record one completed replay-clean >=18/20 native baseline atomically."""

    root = Path(repository_root).resolve()
    ledger_file, _ = _repository_file_argument(
        root,
        ledger_path,
        role="calibration ledger",
    )
    original = validate_calibration_ledger(read_json(ledger_file))
    baseline = original.get("baseline_gate")
    development = original.get("development")
    if (
        original.get("status") != "open_before_native_baseline"
        or not isinstance(baseline, Mapping)
        or baseline.get("status") != "not_run"
        or baseline.get("native_isaac_physics") is not False
        or baseline.get("replay_validated") is not False
        or original.get("bounded_calibration_changes") != []
        or not isinstance(development, Mapping)
        or development.get("status") != "not_run"
        or development.get("candidates") != []
        or development.get("selected_candidate_id") is not None
        or original.get("holdout_freeze_authorized") is not False
        or original.get("first_holdout_started") is not False
    ):
        raise ValueError("baseline can only be recorded into a pristine open ledger")

    matrix_file, _ = _repository_file_argument(root, matrix_path, role="baseline matrix")
    matrix = read_json(matrix_file)
    candidate_reference = Path(str(matrix.get("candidate_manifest", "")))
    if not matrix.get("candidate_manifest") or candidate_reference.is_absolute():
        raise ValueError("baseline matrix candidate protocol reference must be portable")
    candidate_file = (matrix_file.parent / candidate_reference).resolve()
    candidate, _ = _repository_file_argument(
        root,
        candidate_file,
        role="baseline candidate protocol",
    )
    evidence = validate_baseline_gate_artifacts(
        matrix_path=matrix_file,
        replay_path=replay_path,
        baseline_seed_path=baseline_seed_path,
        profile_path=profile_path,
        protocol_payload=load_protocol(candidate),
        completion_receipt_path=completion_receipt_path,
        repository_root=root,
        require_closed_lifecycle=False,
    )

    updated = deepcopy(original)
    updated["status"] = "baseline_passed_development_open"
    updated["baseline_gate"].update(evidence)
    # These values name the selected/final contract and are deliberately not
    # knowable until ledger-close.
    for field in (
        "protocol_sha256",
        "behavioral_protocol_contract_sha256",
        "selected_development_candidate_id",
        "bounded_calibration_change_count",
    ):
        updated["baseline_gate"][field] = None
    updated["development"]["status"] = "open"
    validate_calibration_ledger(updated)
    return _write_calibration_ledger_update(
        ledger_file,
        original=original,
        updated=updated,
    )


def _proposed_closed_calibration_ledger(
    *,
    repository_root: Path,
    ledger_payload: Mapping[str, Any],
    selected_candidate_id: str,
    finalized_protocol_payload: Mapping[str, Any],
) -> dict[str, Any]:
    proposed = deepcopy(dict(ledger_payload))
    finalized = validate_protocol(finalized_protocol_payload)
    contract, _ = _calibration_contract(finalized, repository_root=repository_root)
    proposed["status"] = "closed_before_holdout"
    proposed["development"]["status"] = "complete"
    proposed["development"]["selected_candidate_id"] = selected_candidate_id
    proposed["holdout_freeze_authorized"] = True
    proposed["first_holdout_started"] = False
    proposed["baseline_gate"].update(
        {
            "protocol_sha256": canonical_sha256(finalized),
            "behavioral_protocol_contract_sha256": canonical_sha256(contract),
            "selected_development_candidate_id": selected_candidate_id,
            "bounded_calibration_change_count": len(
                proposed["bounded_calibration_changes"]
            ),
        }
    )
    return proposed


def record_development_candidate(
    *,
    repository_root: Path | str,
    ledger_path: Path | str,
    candidate_id: str,
    protocol_path: Path | str,
    matrix_path: Path | str,
    replay_path: Path | str,
    development_seed_path: Path | str,
    reason: str | None = None,
    verify_replay: bool = True,
) -> dict[str, Any]:
    """Append one immutable replayed development candidate and exact diff."""

    root = Path(repository_root).resolve()
    ledger_file, _ = _repository_file_argument(root, ledger_path, role="calibration ledger")
    original = validate_calibration_ledger(read_json(ledger_file))
    if original.get("status") not in {
        "baseline_passed_development_open",
        "development_open",
    }:
        raise ValueError("development candidates require a recorded baseline and open ledger")
    _validate_recorded_baseline_stage(
        repository_root=root,
        ledger_payload=original,
    )
    development = original.get("development")
    if not isinstance(development, Mapping) or development.get("status") != "open":
        raise ValueError("development ledger is not open")
    candidates = development.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("development candidates must be a list")
    if len(candidates) >= 3 or len(original["bounded_calibration_changes"]) >= 2:
        raise ValueError("M8 permits candidate_0 plus at most candidate_1 and candidate_2")
    expected_id = f"candidate_{len(candidates)}"
    if candidate_id != expected_id:
        raise ValueError(f"next immutable development candidate must be {expected_id}")

    protocol_file, protocol_relative = _repository_file_argument(
        root,
        protocol_path,
        role=f"{candidate_id} protocol",
    )
    matrix_file, matrix_relative = _repository_file_argument(
        root,
        matrix_path,
        role=f"{candidate_id} development matrix",
    )
    replay_file, replay_relative = _repository_file_argument(
        root,
        replay_path,
        role=f"{candidate_id} development replay",
    )
    seed_file, _ = _repository_file_argument(
        root,
        development_seed_path,
        role="development seeds",
    )
    protocol = load_protocol(protocol_file)
    if (
        protocol.get("protocol_status")
        != "preregistered_candidate_pending_native_baseline_and_development"
        or protocol.get("holdout_freeze_status") != "not_yet_frozen"
    ):
        raise ValueError(f"{candidate_id} must remain an explicitly pending candidate")
    contract, _ = _calibration_contract(protocol, repository_root=root)
    seed_count = len(load_seed_file(seed_file))
    if seed_count != int(development.get("seed_count_per_candidate", -1)):
        raise ValueError("candidate does not use the ledger's frozen development seed count")

    prior_paths = {
        record[field]
        for record in candidates
        if isinstance(record, Mapping)
        for field in ("protocol_path", "matrix_path", "replay_path")
    }
    if {protocol_relative, matrix_relative, replay_relative} & prior_paths:
        raise ValueError("each development candidate must use immutable distinct artifacts")
    baseline_candidate_path = str(
        original["baseline_gate"].get("baseline_candidate_protocol_path", "")
    )
    if candidate_id == "candidate_0":
        if protocol_relative != baseline_candidate_path:
            raise ValueError("candidate_0 must reuse the immutable baseline candidate protocol")
        if reason not in (None, ""):
            raise ValueError("candidate_0 does not represent a calibration change")
    elif not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{candidate_id} requires a non-empty calibration reason")

    record = {
        "candidate_id": candidate_id,
        "protocol_path": protocol_relative,
        "protocol_file_sha256": sha256_file(protocol_file),
        "protocol_sha256": canonical_sha256(protocol),
        "contract_sha256": canonical_sha256(contract),
        "matrix_path": matrix_relative,
        "matrix_sha256": sha256_file(matrix_file),
        "replay_path": replay_relative,
        "replay_sha256": sha256_file(replay_file),
        "seed_count": seed_count,
        "replay_validated": True,
        "headline_eligible": False,
    }
    updated = deepcopy(original)
    updated["status"] = "development_open"
    updated["development"]["candidates"].append(record)
    if candidate_id != "candidate_0":
        before_record = candidates[-1]
        before_file = _repository_artifact(
            root,
            before_record["protocol_path"],
            role=f"candidate_{len(candidates) - 1} protocol",
        )
        before_contract, _ = _calibration_contract(
            load_protocol(before_file),
            repository_root=root,
        )
        before_diff, after_diff = _contract_diff(before_contract, contract)
        if not before_diff:
            raise ValueError(f"{candidate_id} does not change the candidate contract")
        unauthorized = sorted(set(before_diff) - CALIBRATABLE_CONTRACT_PATHS)
        if unauthorized:
            raise ValueError(
                f"{candidate_id} changes unauthorized contract paths: {unauthorized}"
            )
        _require_runtime_behavioral_calibration(
            before_diff,
            change_id=candidate_id,
        )
        change_index = len(candidates)
        updated["bounded_calibration_changes"].append(
            {
                "change_id": f"calibration_{change_index}",
                "reason": reason.strip(),
                "before_candidate_id": f"candidate_{change_index - 1}",
                "after_candidate_id": candidate_id,
                "before_contract_sha256": canonical_sha256(before_contract),
                "after_contract_sha256": canonical_sha256(contract),
                "before": before_diff,
                "after": after_diff,
                "development_evidence": {
                    "candidate_id": before_record["candidate_id"],
                    "matrix_sha256": before_record["matrix_sha256"],
                    "replay_sha256": before_record["replay_sha256"],
                },
            }
        )

    # Reuse the closed lifecycle validator against a temporary proposal.  This
    # independently replays every candidate and validates all matrix, seed,
    # profile, contract-diff, and hash bindings before the open ledger changes.
    temporary_final = deepcopy(protocol)
    temporary_final["protocol_status"] = "ready_for_holdout_freeze"
    temporary_final["holdout_freeze_status"] = "ready_to_freeze"
    validation_ledger = _proposed_closed_calibration_ledger(
        repository_root=root,
        ledger_payload=updated,
        selected_candidate_id=candidate_id,
        finalized_protocol_payload=temporary_final,
    )
    validate_development_calibration_lifecycle(
        repository_root=root,
        baseline_candidate_protocol_path=root / baseline_candidate_path,
        finalized_protocol_payload=temporary_final,
        ledger_payload=validation_ledger,
        expected_development_seed_path=seed_file,
        verify_replay=verify_replay,
    )
    validate_calibration_ledger(updated)
    return _write_calibration_ledger_update(
        ledger_file,
        original=original,
        updated=updated,
    )


def close_calibration_ledger(
    *,
    repository_root: Path | str,
    ledger_path: Path | str,
    selected_candidate_id: str,
    finalized_protocol_path: Path | str,
    development_seed_path: Path | str,
    verify_replay: bool = True,
) -> dict[str, Any]:
    """Select the final candidate and irreversibly close the ledger for freeze."""

    root = Path(repository_root).resolve()
    ledger_file, _ = _repository_file_argument(root, ledger_path, role="calibration ledger")
    original = validate_calibration_ledger(read_json(ledger_file))
    if original.get("status") != "development_open":
        raise ValueError("ledger-close requires an open ledger with recorded development")
    _validate_recorded_baseline_stage(
        repository_root=root,
        ledger_payload=original,
    )
    candidates = original.get("development", {}).get("candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or selected_candidate_id not in {
            record.get("candidate_id")
            for record in candidates
            if isinstance(record, Mapping)
        }
    ):
        raise ValueError("selected candidate must name a recorded development candidate")
    finalized_file, _ = _repository_file_argument(
        root,
        finalized_protocol_path,
        role="finalized protocol",
    )
    seed_file, _ = _repository_file_argument(
        root,
        development_seed_path,
        role="development seeds",
    )
    finalized = load_protocol(finalized_file)
    if (
        finalized.get("protocol_status") != "ready_for_holdout_freeze"
        or finalized.get("holdout_freeze_status") != "ready_to_freeze"
    ):
        raise ValueError("finalized protocol must be explicitly ready for holdout freeze")
    updated = _proposed_closed_calibration_ledger(
        repository_root=root,
        ledger_payload=original,
        selected_candidate_id=selected_candidate_id,
        finalized_protocol_payload=finalized,
    )
    lifecycle = validate_development_calibration_lifecycle(
        repository_root=root,
        baseline_candidate_protocol_path=root
        / str(original["baseline_gate"]["baseline_candidate_protocol_path"]),
        finalized_protocol_payload=finalized,
        ledger_payload=updated,
        expected_development_seed_path=seed_file,
        verify_replay=verify_replay,
    )
    if lifecycle["selected_candidate_id"] != selected_candidate_id:
        raise ValueError("selected development candidate validation mismatch")
    validate_calibration_ledger(updated, require_closed=True)
    return _write_calibration_ledger_update(
        ledger_file,
        original=original,
        updated=updated,
    )


def protocol_component_hashes(protocol_payload: Mapping[str, Any]) -> dict[str, str]:
    protocol = validate_protocol(protocol_payload)
    runtime = protocol["runtime"]
    task = protocol["task"]
    controller = protocol.get("controller")
    if not isinstance(controller, Mapping):
        raise ValueError("protocol.controller must be an object")
    safety = {
        "runtime": {
            name: runtime[name]
            for name in (
                "control_frequency_hz",
                "chunk_horizon",
                "action_dimension",
                "request_interval_steps",
                "maximum_episode_steps",
                "safe_hold_command",
            )
        },
        "task": {
            name: task[name]
            for name in (
                "placement_tolerance_m",
                "placement_height_tolerance_m",
                "stable_placement_steps",
                "stable_linear_speed_mps",
                "stable_angular_speed_rps",
                "lift_clearance_m",
                "collision_threshold_n",
            )
        },
        "controller": {
            name: controller[name]
            for name in (
                "maximum_translation_per_step_m",
                "workspace_x_m",
                "workspace_y_m",
                "workspace_z_m",
            )
        },
    }
    return {
        "protocol_sha256": canonical_sha256(protocol),
        "controller_sha256": canonical_sha256(controller),
        "task_contract_sha256": canonical_sha256(task),
        "safety_limits_sha256": canonical_sha256(safety),
    }


def _finite_reset_vector(
    reset: Mapping[str, Any],
    field: str,
    *,
    length: int,
) -> tuple[float, ...]:
    raw = reset.get(field)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"reset-state {field} must be a numeric vector")
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reset-state {field} must be a numeric vector") from exc
    if len(values) != length or not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"reset-state {field} must contain exactly {length} finite values"
        )
    return values


def _canonical_unit_quaternion(
    reset: Mapping[str, Any],
    field: str,
) -> tuple[float, float, float, float]:
    values = _finite_reset_vector(reset, field, length=4)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        raise ValueError(f"reset-state {field} quaternion has zero norm")
    normalized = tuple(value / norm for value in values)
    first_nonzero = next((value for value in normalized if value != 0.0), 0.0)
    if first_nonzero < 0.0:
        normalized = tuple(-value for value in normalized)
    return normalized  # type: ignore[return-value]


def paired_reset_canonical_payload(
    reset_state_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a sign-invariant, quantized reset digest payload.

    This digest is diagnostic across separately started Kit batches.  Exact raw
    reset bytes keep their own SHA-256, while direct numeric tolerances remain
    authoritative for paired physical fairness because values can straddle a
    quantization boundary.
    """

    reset = dict(reset_state_payload)
    reset.pop("reset_state_sha256", None)
    reset.pop("paired_reset_canonical_sha256", None)
    grids = PAIRED_RESET_FAIRNESS_CONTRACT["canonical_grids"]

    def quantized(values: Sequence[float], grid_name: str) -> list[int]:
        grid = float(grids[grid_name])
        return [round(float(value) / grid) for value in values]

    exact_vectors = {
        field: list(_finite_reset_vector(reset, field, length=length))
        for field, length in (
            ("zone_a_xyz", 3),
            ("zone_b_xyz", 3),
            ("gravity_xyz", 3),
        )
    }
    exact_scalars: dict[str, float] = {}
    for field in (
        "physics_dt_seconds",
        "rendering_dt_seconds",
        "stage_units_in_meters",
    ):
        try:
            value = float(reset[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"reset-state {field} must be finite") from exc
        if not math.isfinite(value):
            raise ValueError(f"reset-state {field} must be finite")
        exact_scalars[field] = value
    try:
        seed = int(reset["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("reset-state seed must be an integer") from exc
    if seed < 0 or isinstance(reset.get("seed"), bool):
        raise ValueError("reset-state seed must be a non-negative integer")

    return {
        "seed": seed,
        **exact_vectors,
        **exact_scalars,
        "robot_joint_positions_bins": quantized(
            _finite_reset_vector(reset, "robot_joint_positions", length=9),
            "robot_joint_positions_rad",
        ),
        "robot_joint_velocities_bins": quantized(
            _finite_reset_vector(reset, "robot_joint_velocities", length=9),
            "robot_joint_velocities_rad_s",
        ),
        "end_effector_position_bins": quantized(
            _finite_reset_vector(reset, "end_effector_position_xyz", length=3),
            "end_effector_position_m",
        ),
        "object_position_bins": quantized(
            _finite_reset_vector(reset, "object_position_xyz", length=3),
            "object_position_m",
        ),
        "end_effector_orientation_bins": quantized(
            _canonical_unit_quaternion(reset, "end_effector_orientation_wxyz"),
            "quaternion_component",
        ),
        "object_orientation_bins": quantized(
            _canonical_unit_quaternion(reset, "object_orientation_wxyz"),
            "quaternion_component",
        ),
    }


def build_episode_fairness(
    *,
    protocol_payload: Mapping[str, Any],
    scenario_payload: Mapping[str, Any],
    reset_state_payload: Mapping[str, Any],
    fault_trace_sha256: str,
    freeze_sha256: str | None = None,
) -> dict[str, Any]:
    if len(fault_trace_sha256) != 64:
        raise ValueError("fault_trace_sha256 must be a 64-character digest")
    scenario = dict(scenario_payload)
    embedded_scenario = scenario.pop("scenario_sha256", None)
    scenario_sha = canonical_sha256(scenario)
    if embedded_scenario is not None and embedded_scenario != scenario_sha:
        raise ValueError("scenario payload self-hash mismatch")
    required_scenario = (
        "seed",
        "object_position_xyz",
        "original_destination_xyz",
        "final_destination_xyz",
        "switch_step",
    )
    missing_scenario = [name for name in required_scenario if name not in scenario]
    if missing_scenario:
        raise ValueError(f"scenario fairness fields missing: {missing_scenario}")
    reset = dict(reset_state_payload)
    embedded_reset = reset.pop("reset_state_sha256", None)
    embedded_paired_reset = reset.pop("paired_reset_canonical_sha256", None)
    required_reset = (
        "seed",
        "robot_joint_positions",
        "robot_joint_velocities",
        "end_effector_position_xyz",
        "end_effector_orientation_wxyz",
        "object_position_xyz",
        "object_orientation_wxyz",
        "zone_a_xyz",
        "zone_b_xyz",
        "physics_dt_seconds",
        "rendering_dt_seconds",
        "stage_units_in_meters",
        "gravity_xyz",
    )
    missing_reset = [name for name in required_reset if name not in reset]
    if missing_reset:
        raise ValueError(f"reset-state fields missing: {missing_reset}")
    if int(reset["seed"]) != int(scenario["seed"]):
        raise ValueError("reset-state seed does not match scenario seed")
    reset_sha = canonical_sha256(reset)
    if embedded_reset is not None and embedded_reset != reset_sha:
        raise ValueError("reset-state payload self-hash mismatch")
    paired_reset_sha = canonical_sha256(paired_reset_canonical_payload(reset))
    if embedded_paired_reset is not None and embedded_paired_reset != paired_reset_sha:
        raise ValueError("paired reset-state canonical hash mismatch")
    fairness = {
        "fault_trace_sha256": fault_trace_sha256,
        "scenario_sha256": scenario_sha,
        "reset_state_sha256": reset_sha,
        "paired_reset_canonical_sha256": paired_reset_sha,
        "switch_step": int(scenario["switch_step"]),
        "object_position_xyz": list(scenario["object_position_xyz"]),
        "original_destination_xyz": list(scenario["original_destination_xyz"]),
        "final_destination_xyz": list(scenario["final_destination_xyz"]),
        **protocol_component_hashes(protocol_payload),
    }
    if freeze_sha256 is not None:
        if len(freeze_sha256) != 64:
            raise ValueError("freeze_sha256 must be a 64-character digest")
        fairness["freeze_sha256"] = freeze_sha256
    return fairness


def _frozen_input(path: Path, *, repository_root: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ValueError(f"frozen input escapes repository: {resolved}") from exc
    return {
        "role": role,
        "path": relative.as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _baseline_artifact_role_paths(
    matrix_path: Path,
    *,
    repository_root: Path,
) -> list[tuple[str, Path]]:
    matrix = read_json(matrix_path)
    resolved: set[Path] = set()
    for entry in matrix.get("episodes", ()):
        if not isinstance(entry, Mapping):
            raise ValueError("baseline matrix contains a malformed episode record")
        fields = (
            ("archive",) if entry.get("archive_member") else ("event_log_path", "summary_path")
        ) + ("fault_trace_file", "scenario_file")
        for field in fields:
            value = entry.get(field, matrix.get(field))
            if not value:
                raise ValueError(f"baseline episode is missing {field}")
            relative = Path(str(value))
            if relative.is_absolute():
                raise ValueError(f"baseline artifact path must be portable: {value}")
            target = (matrix_path.parent / relative).resolve()
            try:
                target.relative_to(repository_root)
            except ValueError as exc:
                raise ValueError(f"baseline artifact escapes repository: {target}") from exc
            if not target.is_file():
                raise ValueError(f"baseline artifact does not exist: {target}")
            resolved.add(target)
    return [
        (
            f"baseline_artifact:{path.relative_to(repository_root).as_posix()}",
            path,
        )
        for path in sorted(resolved, key=lambda item: item.as_posix())
    ]


def build_freeze_manifest(
    *,
    repository_root: Path | str,
    protocol_path: Path | str,
    baseline_seed_path: Path | str,
    development_seed_path: Path | str,
    holdout_seed_path: Path | str,
    calibration_ledger_path: Path | str,
    baseline_matrix_path: Path | str,
    baseline_replay_path: Path | str,
    baseline_completion_receipt_path: Path | str,
    profile_paths: Iterable[Path | str],
    additional_inputs: Iterable[Path | str] = (),
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_file)
    if protocol.get("protocol_status") != "ready_for_holdout_freeze":
        raise ValueError("protocol must be finalized after development before holdout freeze")
    if protocol.get("holdout_freeze_status") != "ready_to_freeze":
        raise ValueError("holdout freeze status must be ready_to_freeze")
    baseline_file = Path(baseline_seed_path).resolve()
    development_file = Path(development_seed_path).resolve()
    holdout_file = Path(holdout_seed_path).resolve()
    profile_files = tuple(Path(path).resolve() for path in profile_paths)
    if len(profile_files) != 3:
        raise ValueError("exactly three frozen M8 profile files are required")
    profile_by_id = {
        str(read_json(path).get("profile", {}).get("profile_id")): path
        for path in profile_files
    }
    if set(profile_by_id) != set(PROFILE_STRATEGIES):
        raise ValueError(f"profile files must cover {sorted(PROFILE_STRATEGIES)}")
    ledger_file = Path(calibration_ledger_path).resolve()
    ledger = validate_calibration_ledger(read_json(ledger_file), require_closed=True)
    baseline_matrix_file = Path(baseline_matrix_path).resolve()
    baseline_replay_file = Path(baseline_replay_path).resolve()
    baseline_completion_receipt_file = Path(baseline_completion_receipt_path).resolve()
    baseline_gate = validate_baseline_gate_artifacts(
        matrix_path=baseline_matrix_file,
        replay_path=baseline_replay_file,
        baseline_seed_path=baseline_file,
        profile_path=profile_by_id["profile_0_sanity"],
        protocol_payload=protocol,
        completion_receipt_path=baseline_completion_receipt_file,
        repository_root=root,
        ledger_payload=ledger,
    )
    baseline_matrix_payload = read_json(baseline_matrix_file)
    candidate_value = baseline_matrix_payload.get("candidate_manifest")
    candidate_relative = Path(str(candidate_value or ""))
    if not candidate_value or candidate_relative.is_absolute():
        raise ValueError("baseline matrix candidate protocol path must be portable")
    baseline_candidate_protocol_file = (baseline_matrix_file.parent / candidate_relative).resolve()
    development_lifecycle = validate_development_calibration_lifecycle(
        repository_root=root,
        baseline_candidate_protocol_path=baseline_candidate_protocol_file,
        finalized_protocol_payload=protocol,
        ledger_payload=ledger,
        expected_development_seed_path=development_file,
        verify_replay=True,
    )
    if {
        profile_id: path.resolve() for profile_id, path in profile_by_id.items()
    } != {
        profile_id: path.resolve()
        for profile_id, path in development_lifecycle["selected_profile_paths"].items()
    }:
        raise ValueError("freeze profile inputs do not equal the selected development candidate")
    completion_payload = read_json(baseline_completion_receipt_file)
    receipt_portability = validate_native_completion_receipt_portability(
        completion_receipt_path=baseline_completion_receipt_file,
        repository_root=root,
    )
    baseline_preflight_receipt_file = receipt_portability["preflight_receipt_path"]
    baseline_adapter_evidence_file = receipt_portability[
        "installed_dynamic_adapter_evidence_path"
    ]
    baseline_executor_evidence_file = receipt_portability["executor_evidence_path"]
    baseline_runner_evidence_file = receipt_portability["runner_evidence_path"]
    baseline_runner_support_evidence_file = receipt_portability[
        "runner_support_evidence_path"
    ]
    baseline_external_environment_file = receipt_portability[
        "external_environment_evidence_path"
    ]
    baseline_pixi_manifest_evidence_file = receipt_portability[
        "pixi_manifest_evidence_path"
    ]
    baseline_pixi_lock_evidence_file = receipt_portability["pixi_lock_evidence_path"]
    baseline_process_log_archive_file = _receipt_relative_file(
        completion_payload.get("process_log_archive"),
        receipt_file=baseline_completion_receipt_file,
        repository_root=root,
        name="baseline process-log archive",
        allow_parent=True,
    )
    baseline_process_log_manifest_file = _receipt_relative_file(
        completion_payload.get("process_log_manifest"),
        receipt_file=baseline_completion_receipt_file,
        repository_root=root,
        name="baseline process-log manifest",
        allow_parent=True,
    )
    baseline_artifacts = _baseline_artifact_role_paths(
        baseline_matrix_file,
        repository_root=root,
    )
    validate_seed_splits(
        baseline=load_seed_file(baseline_file),
        development=load_seed_file(development_file),
        holdout=load_seed_file(holdout_file),
    )
    required_sources = tuple((root / relative).resolve() for relative in REQUIRED_FREEZE_SOURCE_PATHS)
    role_paths: list[tuple[str, Path]] = [
        ("protocol", protocol_file),
        ("baseline_seeds", baseline_file),
        ("development_seeds", development_file),
        ("holdout_seeds", holdout_file),
        ("calibration_ledger", ledger_file),
        ("baseline_matrix", baseline_matrix_file),
        ("baseline_replay", baseline_replay_file),
        ("baseline_candidate_protocol", baseline_candidate_protocol_file),
        ("baseline_completion_receipt", baseline_completion_receipt_file),
        ("baseline_preflight_receipt", baseline_preflight_receipt_file),
        ("baseline_installed_dynamic_adapter_evidence", baseline_adapter_evidence_file),
        ("baseline_installed_executor_evidence", baseline_executor_evidence_file),
        ("baseline_runner_evidence", baseline_runner_evidence_file),
        ("baseline_external_environment", baseline_external_environment_file),
        ("baseline_pixi_manifest_evidence", baseline_pixi_manifest_evidence_file),
        ("baseline_pixi_lock_evidence", baseline_pixi_lock_evidence_file),
        ("baseline_process_log_archive", baseline_process_log_archive_file),
        ("baseline_process_log_manifest", baseline_process_log_manifest_file),
    ]
    if baseline_runner_support_evidence_file is not None:
        role_paths.append(
            (
                "baseline_runner_support_evidence",
                baseline_runner_support_evidence_file,
            )
        )
    for profile_file in profile_files:
        profile_id = str(read_json(profile_file).get("profile", {}).get("profile_id"))
        role_paths.append((f"profile:{profile_id}", profile_file))
    development_role_paths: list[tuple[str, Path]] = []
    occupied_paths = {path.resolve() for _role, path in role_paths}
    for role, path in development_lifecycle["input_role_paths"]:
        resolved = path.resolve()
        if resolved in occupied_paths:
            continue
        development_role_paths.append((role, resolved))
        occupied_paths.add(resolved)
    role_paths.extend(development_role_paths)
    selected_candidate_protocol_path = (
        root / development_lifecycle["selected_candidate_protocol_path"]
    ).resolve()
    selected_candidate_protocol_role = next(
        (
            role
            for role, path in role_paths
            if path.resolve() == selected_candidate_protocol_path
        ),
        None,
    )
    if selected_candidate_protocol_role is None:
        raise ValueError("selected development candidate protocol has no freeze input role")
    role_paths.extend(
        (f"source:{relative}", source)
        for relative, source in zip(REQUIRED_FREEZE_SOURCE_PATHS, required_sources, strict=True)
    )
    role_paths.extend(baseline_artifacts)
    role_paths.extend(
        (f"additional:{index}", Path(value).resolve())
        for index, value in enumerate(additional_inputs)
    )
    roles = [role for role, _path in role_paths]
    paths = [path.resolve() for _role, path in role_paths]
    if len(set(roles)) != len(roles) or len(set(paths)) != len(paths):
        raise ValueError("freeze inputs must have unique roles and file paths")
    records = [
        _frozen_input(path, repository_root=root, role=role)
        for role, path in role_paths
    ]
    required_roles = [
        "protocol",
        "baseline_seeds",
        "development_seeds",
        "holdout_seeds",
        "calibration_ledger",
        "baseline_matrix",
        "baseline_replay",
        "baseline_candidate_protocol",
        "baseline_completion_receipt",
        "baseline_preflight_receipt",
        "baseline_installed_dynamic_adapter_evidence",
        "baseline_installed_executor_evidence",
        "baseline_runner_evidence",
        "baseline_external_environment",
        "baseline_pixi_manifest_evidence",
        "baseline_pixi_lock_evidence",
        *(
            ("baseline_runner_support_evidence",)
            if baseline_runner_support_evidence_file is not None
            else ()
        ),
        "baseline_process_log_archive",
        "baseline_process_log_manifest",
        *(f"profile:{profile_id}" for profile_id in sorted(PROFILE_STRATEGIES)),
        *(role for role, _path in development_role_paths),
        *(f"source:{relative}" for relative in REQUIRED_FREEZE_SOURCE_PATHS),
        *(role for role, _path in baseline_artifacts),
    ]
    manifest = {
        "schema_version": M8_SCHEMA_VERSION,
        "milestone": M8_MILESTONE,
        "frozen_before_first_holdout_result": True,
        "headline_evidence_class": NATIVE_ISAAC_EVIDENCE_CLASS,
        "protocol_sha256": canonical_sha256(protocol),
        "baseline_gate": baseline_gate,
        "development_calibration": {
            **{
                name: value
                for name, value in development_lifecycle.items()
                if name not in {"input_role_paths", "selected_profile_paths"}
            },
            "selected_candidate_protocol_role": selected_candidate_protocol_role,
        },
        "required_source_paths": list(REQUIRED_FREEZE_SOURCE_PATHS),
        "required_roles": required_roles,
        "inputs": sorted(records, key=lambda item: item["path"]),
    }
    manifest["freeze_sha256"] = canonical_sha256(manifest)
    return manifest


def write_freeze_manifest(path: Path | str, manifest: Mapping[str, Any]) -> None:
    destination = Path(path)
    payload = dict(manifest)
    expected = payload.pop("freeze_sha256", None)
    if expected != canonical_sha256(payload):
        raise ValueError("freeze manifest self-hash mismatch")
    payload["freeze_sha256"] = expected
    if destination.exists():
        existing = read_json(destination)
        if canonical_sha256(existing) != canonical_sha256(payload):
            raise RuntimeError(f"refusing to change frozen protocol manifest: {destination}")
        return
    write_json_atomic(destination, payload)


def validate_freeze_manifest(
    path: Path | str,
    *,
    repository_root: Path | str,
) -> dict[str, Any]:
    manifest = read_json(path)
    embedded = manifest.get("freeze_sha256")
    core = dict(manifest)
    core.pop("freeze_sha256", None)
    errors: list[str] = []
    if embedded != canonical_sha256(core):
        errors.append("freeze_sha256_mismatch")
    if manifest.get("frozen_before_first_holdout_result") is not True:
        errors.append("frozen_before_holdout_declaration_missing")
    if manifest.get("headline_evidence_class") != NATIVE_ISAAC_EVIDENCE_CLASS:
        errors.append("headline_evidence_class_mismatch")
    root = Path(repository_root).resolve()
    for record in manifest.get("inputs", []):
        target = (root / str(record["path"])).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            errors.append(f"input_escapes_repository:{record['path']}")
            continue
        if not target.is_file():
            errors.append(f"input_missing:{record['path']}")
        elif sha256_file(target) != record.get("sha256"):
            errors.append(f"input_hash_mismatch:{record['path']}")
    recorded_paths = {str(record.get("path")) for record in manifest.get("inputs", [])}
    recorded_roles = [str(record.get("role")) for record in manifest.get("inputs", [])]
    if len(recorded_roles) != len(set(recorded_roles)):
        errors.append("duplicate_input_role")
    records_by_role = {
        str(record.get("role")): record for record in manifest.get("inputs", [])
    }
    baseline_artifact_roles: list[str] = []
    baseline_matrix_record = records_by_role.get("baseline_matrix")
    baseline_replay_record = records_by_role.get("baseline_replay")
    baseline_candidate_record = records_by_role.get("baseline_candidate_protocol")
    baseline_completion_record = records_by_role.get("baseline_completion_receipt")
    baseline_preflight_record = records_by_role.get("baseline_preflight_receipt")
    baseline_adapter_evidence_record = records_by_role.get(
        "baseline_installed_dynamic_adapter_evidence"
    )
    baseline_executor_evidence_record = records_by_role.get(
        "baseline_installed_executor_evidence"
    )
    baseline_runner_evidence_record = records_by_role.get("baseline_runner_evidence")
    baseline_runner_support_evidence_record = records_by_role.get(
        "baseline_runner_support_evidence"
    )
    baseline_external_environment_record = records_by_role.get(
        "baseline_external_environment"
    )
    baseline_pixi_manifest_evidence_record = records_by_role.get(
        "baseline_pixi_manifest_evidence"
    )
    baseline_pixi_lock_evidence_record = records_by_role.get(
        "baseline_pixi_lock_evidence"
    )
    baseline_gate_payload = manifest.get("baseline_gate")
    baseline_runner_support_required = (
        isinstance(baseline_gate_payload, Mapping)
        and baseline_gate_payload.get("runner_source")
        == LINUX_NATIVE_RUNNER_SOURCE_PATH
    )
    if (baseline_runner_support_evidence_record is not None) != (
        baseline_runner_support_required
    ):
        errors.append("baseline_runner_support_evidence_role_presence_mismatch")
    baseline_process_log_archive_record = records_by_role.get(
        "baseline_process_log_archive"
    )
    baseline_process_log_manifest_record = records_by_role.get(
        "baseline_process_log_manifest"
    )
    ledger_record = records_by_role.get("calibration_ledger")
    protocol_record = records_by_role.get("protocol")
    baseline_seed_record = records_by_role.get("baseline_seeds")
    development_seed_record = records_by_role.get("development_seeds")
    profile0_record = records_by_role.get("profile:profile_0_sanity")
    baseline_matrix_file: Path | None = None
    baseline_replay_file: Path | None = None
    baseline_completion_file: Path | None = None
    ledger_file: Path | None = None
    protocol_file: Path | None = None
    baseline_seed_file: Path | None = None
    development_seed_file: Path | None = None
    profile0_file: Path | None = None
    if baseline_matrix_record is not None:
        baseline_matrix_file = (root / str(baseline_matrix_record.get("path", ""))).resolve()
        try:
            baseline_artifact_roles = [
                role
                for role, _path in _baseline_artifact_role_paths(
                    baseline_matrix_file,
                    repository_root=root,
                )
            ]
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"baseline_artifact_binding_invalid:{exc}")
    if baseline_replay_record is not None:
        baseline_replay_file = (root / str(baseline_replay_record.get("path", ""))).resolve()
    if baseline_completion_record is not None:
        baseline_completion_file = (
            root / str(baseline_completion_record.get("path", ""))
        ).resolve()
    if ledger_record is not None:
        ledger_file = (root / str(ledger_record.get("path", ""))).resolve()
    if protocol_record is not None:
        protocol_file = (root / str(protocol_record.get("path", ""))).resolve()
    if baseline_seed_record is not None:
        baseline_seed_file = (root / str(baseline_seed_record.get("path", ""))).resolve()
    if development_seed_record is not None:
        development_seed_file = (
            root / str(development_seed_record.get("path", ""))
        ).resolve()
    if profile0_record is not None:
        profile0_file = (root / str(profile0_record.get("path", ""))).resolve()
    development_roles: list[str] = []
    if (
        baseline_matrix_file is not None
        and baseline_matrix_file.is_file()
        and protocol_file is not None
        and protocol_file.is_file()
        and ledger_file is not None
        and ledger_file.is_file()
        and development_seed_file is not None
        and development_seed_file.is_file()
    ):
        try:
            baseline_matrix_payload = read_json(baseline_matrix_file)
            candidate_value = baseline_matrix_payload.get("candidate_manifest")
            candidate_relative = Path(str(candidate_value or ""))
            if not candidate_value or candidate_relative.is_absolute():
                raise ValueError("baseline matrix candidate protocol path must be portable")
            baseline_candidate_file = (
                baseline_matrix_file.parent / candidate_relative
            ).resolve()
            development_lifecycle = validate_development_calibration_lifecycle(
                repository_root=root,
                baseline_candidate_protocol_path=baseline_candidate_file,
                finalized_protocol_payload=read_json(protocol_file),
                ledger_payload=read_json(ledger_file),
                expected_development_seed_path=development_seed_file,
                verify_replay=True,
            )
            selected_profile_paths = development_lifecycle["selected_profile_paths"]
            for profile_id, selected_path in selected_profile_paths.items():
                profile_record = records_by_role.get(f"profile:{profile_id}")
                recorded_path = (
                    None
                    if profile_record is None
                    else (root / str(profile_record.get("path", ""))).resolve()
                )
                if recorded_path != selected_path.resolve():
                    errors.append(
                        f"selected_development_profile_role_mismatch:{profile_id}"
                    )
            base_role_names = {
                "protocol",
                "baseline_seeds",
                "development_seeds",
                "holdout_seeds",
                "calibration_ledger",
                "baseline_matrix",
                "baseline_replay",
                "baseline_candidate_protocol",
                "baseline_completion_receipt",
                "baseline_preflight_receipt",
                "baseline_installed_dynamic_adapter_evidence",
                "baseline_installed_executor_evidence",
                "baseline_runner_evidence",
                "baseline_external_environment",
                "baseline_pixi_manifest_evidence",
                "baseline_pixi_lock_evidence",
                *(
                    ("baseline_runner_support_evidence",)
                    if baseline_runner_support_required
                    else ()
                ),
                "baseline_process_log_archive",
                "baseline_process_log_manifest",
                *(f"profile:{profile_id}" for profile_id in PROFILE_STRATEGIES),
            }
            occupied_paths = {
                (root / str(record.get("path", ""))).resolve()
                for role, record in records_by_role.items()
                if role in base_role_names
            }
            for role, input_path in development_lifecycle["input_role_paths"]:
                resolved = input_path.resolve()
                if resolved in occupied_paths:
                    continue
                development_roles.append(role)
                occupied_paths.add(resolved)
            selected_protocol_path = (
                root / development_lifecycle["selected_candidate_protocol_path"]
            ).resolve()
            selected_protocol_role = next(
                (
                    role
                    for role, record in records_by_role.items()
                    if role in {"baseline_candidate_protocol", *development_roles}
                    and (root / str(record.get("path", ""))).resolve()
                    == selected_protocol_path
                ),
                None,
            )
            expected_development = {
                **{
                    name: value
                    for name, value in development_lifecycle.items()
                    if name not in {"input_role_paths", "selected_profile_paths"}
                },
                "selected_candidate_protocol_role": selected_protocol_role,
            }
            if (
                selected_protocol_role is None
                or manifest.get("development_calibration") != expected_development
            ):
                errors.append("development_calibration_summary_mismatch")
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"development_calibration_validation_failed:{exc}")
    expected_roles = [
        "protocol",
        "baseline_seeds",
        "development_seeds",
        "holdout_seeds",
        "calibration_ledger",
        "baseline_matrix",
        "baseline_replay",
        "baseline_candidate_protocol",
        "baseline_completion_receipt",
        "baseline_preflight_receipt",
        "baseline_installed_dynamic_adapter_evidence",
        "baseline_installed_executor_evidence",
        "baseline_runner_evidence",
        "baseline_external_environment",
        "baseline_pixi_manifest_evidence",
        "baseline_pixi_lock_evidence",
        *(
            ("baseline_runner_support_evidence",)
            if baseline_runner_support_required
            else ()
        ),
        "baseline_process_log_archive",
        "baseline_process_log_manifest",
        *(f"profile:{profile_id}" for profile_id in sorted(PROFILE_STRATEGIES)),
        *development_roles,
        *(f"source:{relative}" for relative in REQUIRED_FREEZE_SOURCE_PATHS),
        *baseline_artifact_roles,
    ]
    if list(manifest.get("required_roles", ())) != expected_roles:
        errors.append("required_role_declaration_mismatch")
    for role in expected_roles:
        if role not in recorded_roles:
            errors.append(f"required_role_missing:{role}")
    declared_required = tuple(manifest.get("required_source_paths", ()))
    if declared_required != REQUIRED_FREEZE_SOURCE_PATHS:
        errors.append("required_source_declaration_mismatch")
    for required in REQUIRED_FREEZE_SOURCE_PATHS:
        if required not in recorded_paths:
            errors.append(f"required_source_missing:{required}")
    if (
        baseline_matrix_file is not None
        and baseline_replay_file is not None
        and baseline_completion_file is not None
        and ledger_file is not None
        and protocol_file is not None
        and baseline_seed_file is not None
        and profile0_file is not None
        and baseline_matrix_file.is_file()
        and baseline_replay_file.is_file()
        and baseline_completion_file.is_file()
        and ledger_file.is_file()
        and protocol_file.is_file()
        and baseline_seed_file.is_file()
        and profile0_file.is_file()
    ):
        try:
            replayed_gate = validate_baseline_gate_artifacts(
                matrix_path=baseline_matrix_file,
                replay_path=baseline_replay_file,
                baseline_seed_path=baseline_seed_file,
                profile_path=profile0_file,
                protocol_payload=read_json(protocol_file),
                completion_receipt_path=baseline_completion_file,
                repository_root=root,
                ledger_payload=read_json(ledger_file),
            )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"baseline_gate_validation_failed:{exc}")
        else:
            if replayed_gate != manifest.get("baseline_gate"):
                errors.append("baseline_gate_summary_mismatch")
    if baseline_matrix_file is not None and baseline_matrix_file.is_file():
        baseline_matrix_payload = read_json(baseline_matrix_file)
        candidate_value = baseline_matrix_payload.get("candidate_manifest")
        candidate_path = Path(str(candidate_value or ""))
        if not candidate_value or candidate_path.is_absolute():
            errors.append("baseline_candidate_protocol_path_invalid")
        else:
            candidate_path = (baseline_matrix_file.parent / candidate_path).resolve()
            recorded_candidate = (
                None
                if baseline_candidate_record is None
                else (root / str(baseline_candidate_record.get("path", ""))).resolve()
            )
            if recorded_candidate != candidate_path:
                errors.append("baseline_candidate_protocol_role_mismatch")
    if baseline_completion_file is not None and baseline_completion_file.is_file():
        try:
            receipt_portability = validate_native_completion_receipt_portability(
                completion_receipt_path=baseline_completion_file,
                repository_root=root,
            )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"baseline_receipt_portability_invalid:{exc}")
        else:
            for expected_path, record, error_name in (
                (
                    receipt_portability["preflight_receipt_path"],
                    baseline_preflight_record,
                    "baseline_preflight_receipt_role_mismatch",
                ),
                (
                    receipt_portability["installed_dynamic_adapter_evidence_path"],
                    baseline_adapter_evidence_record,
                    "baseline_installed_dynamic_adapter_evidence_role_mismatch",
                ),
                (
                    receipt_portability["executor_evidence_path"],
                    baseline_executor_evidence_record,
                    "baseline_installed_executor_evidence_role_mismatch",
                ),
                (
                    receipt_portability["runner_evidence_path"],
                    baseline_runner_evidence_record,
                    "baseline_runner_evidence_role_mismatch",
                ),
                (
                    receipt_portability["external_environment_evidence_path"],
                    baseline_external_environment_record,
                    "baseline_external_environment_role_mismatch",
                ),
                (
                    receipt_portability["pixi_manifest_evidence_path"],
                    baseline_pixi_manifest_evidence_record,
                    "baseline_pixi_manifest_evidence_role_mismatch",
                ),
                (
                    receipt_portability["pixi_lock_evidence_path"],
                    baseline_pixi_lock_evidence_record,
                    "baseline_pixi_lock_evidence_role_mismatch",
                ),
                *(
                    (
                        (
                            receipt_portability["runner_support_evidence_path"],
                            baseline_runner_support_evidence_record,
                            "baseline_runner_support_evidence_role_mismatch",
                        ),
                    )
                    if receipt_portability["runner_support_evidence_path"]
                    is not None
                    else ()
                ),
            ):
                recorded_path = (
                    None
                    if record is None
                    else (root / str(record.get("path", ""))).resolve()
                )
                if recorded_path != expected_path:
                    errors.append(error_name)
        completion_payload = read_json(baseline_completion_file)
        for field, record, error_name in (
            (
                "process_log_archive",
                baseline_process_log_archive_record,
                "baseline_process_log_archive_role_mismatch",
            ),
            (
                "process_log_manifest",
                baseline_process_log_manifest_record,
                "baseline_process_log_manifest_role_mismatch",
            ),
        ):
            try:
                expected_path = _receipt_relative_file(
                    completion_payload.get(field),
                    receipt_file=baseline_completion_file,
                    repository_root=root,
                    name=field.replace("_", " "),
                    allow_parent=True,
                )
            except (OSError, TypeError, ValueError) as exc:
                errors.append(f"{error_name}:invalid_receipt_path:{exc}")
                continue
            recorded_path = (
                None
                if record is None
                else (root / str(record.get("path", ""))).resolve()
            )
            if recorded_path != expected_path:
                errors.append(error_name)
    return {"passed": not errors, "errors": errors, "manifest": manifest}
