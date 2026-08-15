from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import shutil

import pytest

from action_stream_policy.dynamic_pick_place import DynamicPolicyConfig

from action_stream_isaac.dynamic_isaac_adapter import (
    DynamicIsaacScene,
    DynamicInKitEpisodeRuntime,
    DISALLOWED_CONTACT_FILTER_PATHS,
    GROUND_COLLISION_PATH,
    GROUND_PATH,
    ROS_CALLBACK_DRAIN_LIMIT,
    _command_motion_audit_payload,
    _cleanup_batch_resources,
    _finalize_episode_summary,
    _joint_limit_vectors,
    _portable_reference,
    _policy_parameter_values,
    _record_episode_failure,
    _validate_runtime_args,
    load_matrix_manifest,
    main,
    runtime_configs_from_protocol,
    validation_report,
)
from action_stream_isaac.dynamic_task import scenario_for_seed, scenario_sha256


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_isaac_6_joint_limits_are_separate_lower_upper_arrays() -> None:
    lower = (-2.0,) * 9
    upper = (2.0,) * 9
    decoded_lower, decoded_upper = _joint_limit_vectors((lower, upper), dof_count=9)
    assert decoded_lower == lower
    assert decoded_upper == upper

    scene = object.__new__(DynamicIsaacScene)
    scene.policy_config = DynamicPolicyConfig()
    scene._articulation = SimpleNamespace(
        get_dof_positions=lambda: (0.0,) * 9,
        get_dof_limits=lambda: (lower, upper),
    )
    assert scene._joint_limit_reached((0.45, 0.0, 0.30)) is False
    scene._articulation.get_dof_positions = lambda: (-2.0,) + (0.0,) * 8
    assert scene._joint_limit_reached((0.45, 0.0, 0.30)) is True

    with pytest.raises(RuntimeError, match="separate lower/upper"):
        _joint_limit_vectors(tuple(zip(lower, upper)), dof_count=9)


def test_scene_uses_protocol_policy_workspace_for_commands_and_limits() -> None:
    scene = object.__new__(DynamicIsaacScene)
    scene.policy_config = DynamicPolicyConfig(
        workspace_x_m=(0.30, 0.65),
        workspace_y_m=(-0.25, 0.25),
        workspace_z_m=(0.10, 0.55),
    )
    scene._articulation = SimpleNamespace(
        get_dof_positions=lambda: (0.0,) * 9,
        get_dof_limits=lambda: ((-2.0,) * 9, (2.0,) * 9),
    )

    scene.set_command((0.31, 0.24, 0.54, 3.14, 0.0, 0.0, 1.0))
    assert scene.current_command[:3] == (0.31, 0.24, 0.54)
    assert scene._joint_limit_reached((0.31, 0.24, 0.54)) is False
    assert scene._joint_limit_reached((0.29, 0.24, 0.54)) is True
    with pytest.raises(ValueError, match="outside"):
        scene.set_command((0.29, 0.24, 0.54, 3.14, 0.0, 0.0, 1.0))


def test_isaac_6_ground_contact_filter_targets_collision_child() -> None:
    assert GROUND_PATH == "/World/GroundPlane"
    assert GROUND_COLLISION_PATH == "/World/GroundPlane/collisionPlane"
    assert DISALLOWED_CONTACT_FILTER_PATHS == (
        "/World/DynamicObject",
        "/World/GroundPlane/collisionPlane",
    )


def test_scene_enables_contact_reporting_before_physics_setup() -> None:
    source = inspect.getsource(DynamicIsaacScene.__init__)
    finger_enable = (
        "self._finger_contacts.set_enabled_contact_tracking([True], threshold=0.0)"
    )
    disallowed_enable = (
        "self._disallowed_contacts.set_enabled_contact_tracking([True], threshold=0.0)"
    )
    setup = "SimulationManager.setup_simulation"

    assert finger_enable in source
    assert disallowed_enable in source
    assert source.index(finger_enable) < source.index(setup)
    assert source.index(disallowed_enable) < source.index(setup)


def test_command_motion_audit_distinguishes_requested_and_applied_hold_target() -> None:
    payload = _command_motion_audit_payload(
        requested_command=(0.48, 0.18, 0.30, 3.14, 0.0, 0.0, 1.0),
        previous_scene_target=(0.31, 0.0, 0.59, 3.14, 0.0, 0.0, 1.0),
        applied_scene_target=(0.31, 0.0, 0.59, 3.14, 0.0, 0.0, 1.0),
        end_effector_before_xyz=(0.31, 0.0, 0.59),
        end_effector_after_xyz=(0.311, 0.0, 0.589),
        hold=True,
        invalid_command=False,
        planned_row_translation_limit_m=0.01,
    )

    assert payload["requested_target_delta_m"] > 0.3
    assert payload["applied_target_delta_m"] == 0.0
    assert payload["measured_end_effector_delta_m"] == pytest.approx(2**0.5 * 0.001)
    assert payload["planned_row_limit_scope"] == "adjacent_planned_rows_only"


def test_recorded_provenance_reference_is_portable(tmp_path: Path) -> None:
    matrix_directory = tmp_path / "outputs" / "m8"
    protocol = tmp_path / "configs" / "m8_g0.json"
    assert (
        _portable_reference(protocol, base=matrix_directory)
        == "../../configs/m8_g0.json"
    )


def _matrix(
    tmp_path: Path,
    *,
    split: str = "development",
    strategies: tuple[str, ...] = ("aligned_async",),
    profile_id: str = "profile_1_fixed",
) -> Path:
    (tmp_path / ".git").mkdir(parents=True, exist_ok=True)
    candidate = tmp_path / "candidate.json"
    repository_root = Path(__file__).resolve().parents[4]
    candidate_payload = json.loads(
        (repository_root / "configs" / "m8_g0.json").read_text(encoding="utf-8")
    )
    profile_relative = Path(candidate_payload["profiles"][profile_id]["profile_file"])
    profile_path = tmp_path / profile_relative
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repository_root / profile_relative, profile_path)
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    declared_profile = profile_payload["profile"]
    _write_json(candidate, candidate_payload)
    protocol_sha256 = _canonical_sha256(candidate_payload)
    freeze_core = {
        "schema_version": 1,
        "milestone": "M8-G0",
        "protocol_sha256": protocol_sha256,
        "inputs": [
            {
                "role": "protocol",
                "path": candidate.name,
                "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "size_bytes": candidate.stat().st_size,
            },
            {
                "role": f"profile:{profile_id}",
                "path": profile_relative.as_posix(),
                "sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                "size_bytes": profile_path.stat().st_size,
            },
        ],
    }
    freeze = {**freeze_core, "freeze_sha256": _canonical_sha256(freeze_core)}
    freeze_path = tmp_path / "freeze.json"
    _write_json(freeze_path, freeze)

    seed = 2026081000
    trace_core = {
        "schema_version": 1,
        "milestone": "M8-G0",
        "profile": {
            **{
                name: declared_profile[name]
                for name in (
                    "profile_id",
                    "base_latency_ms",
                    "jitter_ms",
                    "drop_probability",
                    "extra_delay_probability",
                    "extra_delay_ms",
                    "duplicate_probability",
                    "duplicate_delivery_offset_ms",
                    "communication_pause_probability",
                    "communication_pause_ms",
                )
            },
            "frozen": True,
        },
        "seed": seed,
        "entries": [],
    }
    trace = {**trace_core, "trace_sha256": _canonical_sha256(trace_core)}
    trace_path = tmp_path / "trace.json"
    _write_json(trace_path, trace)

    episodes = []
    for strategy in strategies:
        episodes.append(
            {
                "episode_id": f"episode-{strategy}",
                "seed": seed,
                "strategy": strategy,
                "profile_id": profile_id,
                "fault_trace_file": trace_path.name,
                "fault_trace_sha256": trace["trace_sha256"],
                "event_log_path": f"events/{strategy}.jsonl",
                "summary_path": f"summaries/{strategy}.json",
                "scenario_sha256": scenario_sha256(scenario_for_seed(seed)),
                "extra_provenance_key": "accepted",
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "milestone": "M8-G0",
        "split": split,
        "protocol_sha256": protocol_sha256,
        "persistent_isaac_process_required": True,
        "profiles": {
            profile_id: {
                "path": profile_relative.as_posix(),
                "sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            }
        },
        "batch_profile_id": profile_id,
        "batch_strategy": strategies[0],
        "expected_episode_count": len(episodes),
        "episodes": episodes,
    }
    if split == "frozen_holdout":
        payload.update(
            {
                "freeze_manifest": freeze_path.name,
                "freeze_sha256": freeze["freeze_sha256"],
                "holdout_freeze_status": "frozen",
                "frozen_before_first_holdout_result": True,
            }
        )
    else:
        payload.update(
            {
                "candidate_manifest": candidate.name,
                "holdout_freeze_status": "pending",
                "frozen_before_first_holdout_result": False,
            }
        )
    path = tmp_path / "matrix.json"
    _write_json(path, payload)
    return path


def _runtime_args(**overrides: object) -> SimpleNamespace:
    values = {
        "max_episodes": 0,
        "startup_discovery_seconds": 0.0,
        "bootstrap_timeout_seconds": 15.0,
        "command_timeout_seconds": 5.0,
        "terminal_drain_seconds": 0.0,
        "expected_strategy": "aligned_async",
        "validate_only": False,
        "video_output": None,
        "video_fps": 20,
        "video_width": 1280,
        "video_height": 720,
        "video_finalize_timeout_seconds": 120.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_loader_accepts_relative_candidate_batch_and_marks_validation_non_evidence(
    tmp_path: Path,
) -> None:
    matrix = load_matrix_manifest(_matrix(tmp_path))
    assert matrix.strategy == "aligned_async"
    assert matrix.split == "development"
    assert matrix.episodes[0].fault_trace_file == (tmp_path / "trace.json").resolve()
    report = validation_report(matrix)
    assert report["validation_passed"] is True
    assert report["native_execution_performed"] is False
    assert report["headline_evidence_eligible"] is False
    assert report["evidence_class"] == "config_validation_only"
    assert report["runtime_config_binding_passed"] is True
    assert report["task_config"]["max_steps"] == 320
    assert report["policy_config"]["maximum_translation_per_step_m"] == 0.01


def _rewrite_candidate_protocol(matrix_path: Path, mutate) -> dict[str, object]:
    matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    candidate_path = matrix_path.parent / str(matrix_payload["candidate_manifest"])
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    mutate(candidate)
    _write_json(candidate_path, candidate)
    matrix_payload["protocol_sha256"] = _canonical_sha256(candidate)
    _write_json(matrix_path, matrix_payload)
    return candidate


def _refresh_matrix_scenarios(matrix_path: Path, candidate: dict[str, object]) -> None:
    task_config, _policy_config = runtime_configs_from_protocol(candidate)
    matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    for episode in matrix_payload["episodes"]:
        episode["scenario_sha256"] = scenario_sha256(
            scenario_for_seed(episode["seed"], task_config)
        )
    _write_json(matrix_path, matrix_payload)


def _rewrite_trace_profile(
    matrix_path: Path,
    *,
    field: str,
    value: object,
) -> str:
    matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    trace_path = matrix_path.parent / matrix_payload["episodes"][0]["fault_trace_file"]
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_payload["profile"][field] = value
    trace_core = dict(trace_payload)
    trace_core.pop("trace_sha256", None)
    trace_sha = _canonical_sha256(trace_core)
    trace_payload["trace_sha256"] = trace_sha
    _write_json(trace_path, trace_payload)
    for episode in matrix_payload["episodes"]:
        episode["fault_trace_sha256"] = trace_sha
    _write_json(matrix_path, matrix_payload)
    return trace_sha


def test_nondefault_protocol_values_construct_exact_task_and_policy_configs(
    tmp_path: Path,
) -> None:
    path = _matrix(tmp_path)

    def calibrate(candidate: dict[str, object]) -> None:
        task = candidate["task"]
        controller = candidate["controller"]
        assert isinstance(task, dict)
        assert isinstance(controller, dict)
        task.update(
            {
                "object_x_range_m": [0.40, 0.46],
                "object_y_range_m": [-0.06, 0.02],
                "zone_a_xy_m": [0.50, -0.24],
                "zone_b_xy_m": [0.50, 0.24],
                "switch_steps": [80, 100, 120],
                "lift_clearance_m": 0.12,
            }
        )
        controller.update(
            {
                "approach_height_m": 0.18,
                "grasp_hand_offset_m": 0.09,
                "carry_hand_height_m": 0.30,
                "recovery_hover_offset_m": 0.11,
                "maximum_translation_per_step_m": 0.008,
                "workspace_x_m": [0.20, 0.75],
                "workspace_y_m": [-0.40, 0.40],
                "workspace_z_m": [0.06, 0.65],
            }
        )

    candidate = _rewrite_candidate_protocol(path, calibrate)
    _refresh_matrix_scenarios(path, candidate)
    direct_task, direct_policy = runtime_configs_from_protocol(candidate)
    matrix = load_matrix_manifest(path)

    assert matrix.task_config == direct_task
    assert matrix.policy_config == direct_policy
    assert matrix.task_config.object_x_range_m == (0.40, 0.46)
    assert matrix.task_config.object_y_range_m == (-0.06, 0.02)
    assert matrix.task_config.zone_a_xy_m == (0.50, -0.24)
    assert matrix.task_config.zone_b_xy_m == (0.50, 0.24)
    assert matrix.task_config.switch_steps == (80, 100, 120)
    assert matrix.task_config.lift_clearance_m == 0.12
    assert matrix.task_config.approach_height_m == 0.18
    assert matrix.task_config.grasp_hand_offset_m == 0.09
    assert matrix.task_config.carry_hand_height_m == 0.30
    assert matrix.policy_config.recovery_hover_offset_m == 0.11
    assert matrix.policy_config.maximum_translation_per_step_m == 0.008
    assert _policy_parameter_values(matrix.policy_config) == {
        "approach_height_m": 0.18,
        "grasp_hand_offset_m": 0.09,
        "carry_hand_height_m": 0.30,
        "recovery_hover_offset_m": 0.11,
        "maximum_translation_per_step_m": 0.008,
        "workspace_x_m": [0.2, 0.75],
        "workspace_y_m": [-0.4, 0.4],
        "workspace_z_m": [0.06, 0.65],
        "downward_axis_angle_xyz": [3.141592653589793, 0.0, 0.0],
    }


@pytest.mark.parametrize("section", ["task", "controller"])
@pytest.mark.parametrize("field_change", ["missing", "unknown"])
def test_protocol_runtime_binding_rejects_field_drift(
    tmp_path: Path, section: str, field_change: str
) -> None:
    path = _matrix(tmp_path)

    def drift(candidate: dict[str, object]) -> None:
        values = candidate[section]
        assert isinstance(values, dict)
        if field_change == "missing":
            values.pop(
                "lift_clearance_m" if section == "task" else "recovery_hover_offset_m"
            )
        else:
            values["unbound_native_gain"] = 1.0

    _rewrite_candidate_protocol(path, drift)
    with pytest.raises(ValueError, match=rf"protocol\.{section} field drift"):
        load_matrix_manifest(path)


def test_protocol_runtime_binding_rejects_out_of_bounds_candidate_value(
    tmp_path: Path,
) -> None:
    path = _matrix(tmp_path)

    def drift(candidate: dict[str, object]) -> None:
        controller = candidate["controller"]
        assert isinstance(controller, dict)
        controller["maximum_translation_per_step_m"] = 0.021

    _rewrite_candidate_protocol(path, drift)
    with pytest.raises(ValueError, match="bounded calibration range"):
        load_matrix_manifest(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_latency_ms", 900),
        ("jitter_ms", 1),
        ("drop_probability", 0.1),
        ("extra_delay_probability", 0.1),
        ("extra_delay_ms", 1),
        ("duplicate_probability", 0.1),
        ("duplicate_delivery_offset_ms", 1),
        ("communication_pause_probability", 0.1),
        ("communication_pause_ms", 1),
    ],
)
def test_loader_rejects_self_consistent_trace_profile_divergence(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = _matrix(tmp_path)
    _rewrite_trace_profile(path, field=field, value=value)

    with pytest.raises(ValueError, match="differs from declared bytes"):
        load_matrix_manifest(path)


def test_loader_accepts_profile_and_trace_with_same_nondefault_declared_latency(
    tmp_path: Path,
) -> None:
    path = _matrix(tmp_path)
    matrix_payload = json.loads(path.read_text(encoding="utf-8"))
    profile_record = matrix_payload["profiles"]["profile_1_fixed"]
    profile_path = path.parent / profile_record["path"]
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_payload["profile"]["base_latency_ms"] = 900
    _write_json(profile_path, profile_payload)
    profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    profile_record["sha256"] = profile_sha

    candidate_path = path.parent / matrix_payload["candidate_manifest"]
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["profiles"]["profile_1_fixed"]["profile_sha256"] = profile_sha
    _write_json(candidate_path, candidate)
    matrix_payload["protocol_sha256"] = _canonical_sha256(candidate)
    _write_json(path, matrix_payload)
    _rewrite_trace_profile(path, field="base_latency_ms", value=900)

    matrix = load_matrix_manifest(path)
    assert matrix.profile_id == "profile_1_fixed"


def test_loader_rejects_profile0_faults_and_undeclared_strategy(tmp_path: Path) -> None:
    path = _matrix(
        tmp_path / "faults",
        profile_id="profile_0_sanity",
        strategies=("sync_hold",),
    )
    _rewrite_trace_profile(path, field="drop_probability", value=0.1)
    with pytest.raises(ValueError, match="differs from declared bytes"):
        load_matrix_manifest(path)

    path = _matrix(
        tmp_path / "strategy",
        profile_id="profile_0_sanity",
        strategies=("naive_async",),
    )
    with pytest.raises(ValueError, match="not declared by profile"):
        load_matrix_manifest(path)


def test_loader_accepts_only_consistent_persistent_batches(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one executor strategy"):
        load_matrix_manifest(
            _matrix(tmp_path, strategies=("naive_async", "aligned_async"))
        )


def test_frozen_holdout_requires_matching_freeze_and_cannot_be_truncated(
    tmp_path: Path,
) -> None:
    path = _matrix(tmp_path, split="frozen_holdout")
    payload = json.loads(path.read_text(encoding="utf-8"))
    second = dict(payload["episodes"][0])
    second["episode_id"] = "episode-aligned-2"
    second["event_log_path"] = "events/aligned-2.jsonl"
    second["summary_path"] = "summaries/aligned-2.json"
    payload["episodes"].append(second)
    payload["expected_episode_count"] = len(payload["episodes"])
    _write_json(path, payload)
    matrix = load_matrix_manifest(path)
    assert matrix.frozen_before_first_holdout_result
    with pytest.raises(ValueError, match="cannot truncate"):
        _validate_runtime_args(_runtime_args(max_episodes=1), matrix)


def test_loader_rejects_scenario_hash_drift_and_existing_output(tmp_path: Path) -> None:
    path = _matrix(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["episodes"][0]["scenario_sha256"] = "0" * 64
    _write_json(path, payload)
    with pytest.raises(ValueError, match="scenario_sha256 mismatch"):
        load_matrix_manifest(path)

    path.unlink()
    path = _matrix(tmp_path)
    output = tmp_path / "events" / "aligned_async.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        load_matrix_manifest(path)


def test_validation_only_never_imports_isaac_or_creates_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _matrix(tmp_path)
    before = set(sys.modules)
    assert main(["--matrix-manifest", str(path), "--validate-only"]) == 0
    newly_loaded = set(sys.modules) - before
    assert "isaacsim" not in newly_loaded
    assert "rclpy" not in newly_loaded
    report = json.loads(capsys.readouterr().out)
    assert report["outputs_created"] is False
    assert not (tmp_path / "events").exists()
    assert not (tmp_path / "summaries").exists()


def test_adapter_module_import_is_ros_and_isaac_safe() -> None:
    before = set(sys.modules)
    importlib.reload(sys.modules["action_stream_isaac.dynamic_isaac_adapter"])
    newly_loaded = set(sys.modules) - before
    assert "isaacsim" not in newly_loaded
    assert "rclpy" not in newly_loaded


def test_runtime_wires_single_m8_recorder_and_raw_derived_summary() -> None:
    runtime_source = inspect.getsource(DynamicInKitEpisodeRuntime)
    assert "create_dynamic_event_recorder_node" in runtime_source
    assert "DynamicRecorderConfig" in runtime_source
    assert "matrix.split" in runtime_source
    assert "create_event_recorder_node" not in runtime_source
    finalizer_source = inspect.getsource(_finalize_episode_summary)
    assert "read_jsonl" in finalizer_source
    assert "write_summary_from_rows" in finalizer_source
    assert "result.success" not in finalizer_source


def test_runtime_waits_once_then_drains_ros_callback_backlog() -> None:
    class Executor:
        def __init__(self) -> None:
            self.timeouts: list[float] = []

        def spin_once(self, *, timeout_sec: float) -> None:
            self.timeouts.append(timeout_sec)

    runtime = object.__new__(DynamicInKitEpisodeRuntime)
    runtime._executor = Executor()

    runtime.spin_pending_callbacks(initial_timeout_sec=0.01)

    assert runtime._executor.timeouts == [0.01] + [0.0] * (
        ROS_CALLBACK_DRAIN_LIMIT - 1
    )
    assert "spin_pending_callbacks" in inspect.getsource(main)


def test_episode_failure_preserves_primary_and_records_runtime_close_error(
    tmp_path: Path,
) -> None:
    matrix = load_matrix_manifest(_matrix(tmp_path))
    spec = matrix.episodes[0]

    class FailingRuntime:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("recorder close failed")

    runtime = FailingRuntime()
    primary = ValueError("bootstrap request timed out")

    def fail_episode() -> None:
        try:
            raise primary
        except Exception as exc:
            errors = _record_episode_failure(
                runtime=runtime,
                matrix=matrix,
                spec=spec,
                exc=exc,
                started_wall_ns=123,
            )
            assert errors == [
                {
                    "stage": "episode_runtime_close",
                    "exception_type": "RuntimeError",
                    "message": "recorder close failed",
                }
            ]
            raise

    with pytest.raises(ValueError, match="bootstrap request timed out") as caught:
        fail_episode()
    assert caught.value is primary
    assert runtime.close_calls == 1

    summary = json.loads(spec.summary_path.read_text(encoding="utf-8"))
    assert summary["infrastructure_failure"] == (
        "ValueError: bootstrap request timed out"
    )
    assert summary["cleanup_errors"] == [
        {
            "stage": "episode_runtime_close",
            "exception_type": "RuntimeError",
            "message": "recorder close failed",
        }
    ]


def test_failure_summary_write_error_does_not_mask_primary(tmp_path: Path) -> None:
    matrix = load_matrix_manifest(_matrix(tmp_path))
    spec = matrix.episodes[0]
    spec.summary_path.parent.mkdir(parents=True, exist_ok=True)
    stale_temporary = spec.summary_path.with_name(f".{spec.summary_path.name}.tmp")
    stale_temporary.write_text("stale", encoding="utf-8")
    primary = LookupError("transport stopped")

    def fail_episode() -> None:
        try:
            raise primary
        except Exception as exc:
            errors = _record_episode_failure(
                runtime=None,
                matrix=matrix,
                spec=spec,
                exc=exc,
                started_wall_ns=456,
            )
            assert len(errors) == 1
            assert errors[0]["stage"] == "failure_summary_write"
            assert errors[0]["exception_type"] == "RuntimeError"
            assert "stale summary temporary" in errors[0]["message"]
            raise

    with pytest.raises(LookupError, match="transport stopped") as caught:
        fail_episode()
    assert caught.value is primary
    assert not spec.summary_path.exists()


def test_batch_cleanup_attempts_every_stage_and_returns_failure_status() -> None:
    calls: list[object] = []

    class Video:
        def cancel(self) -> None:
            calls.append("video")
            raise RuntimeError("cancel failed")

    class Closable:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            calls.append(self.name)

    class Ros:
        def ok(self) -> bool:
            calls.append("ros_ok")
            return True

        def shutdown(self) -> None:
            calls.append("ros_shutdown")

    class App:
        def close(self, *, exit_code: int, skip_cleanup: bool) -> None:
            calls.append(("app", exit_code, skip_cleanup))

    exit_code, errors = _cleanup_batch_resources(
        video_capture=Video(),
        runtime=Closable("runtime"),
        bridge=Closable("bridge"),
        scene=Closable("scene"),
        rclpy=Ros(),
        simulation_app=App(),
        exit_code=0,
    )

    assert exit_code == 1
    assert calls == [
        "video",
        "runtime",
        "bridge",
        "scene",
        "ros_ok",
        "ros_shutdown",
        ("app", 1, True),
    ]
    assert errors == [
        {
            "stage": "video_capture_cancel",
            "exception_type": "RuntimeError",
            "message": "cancel failed",
        }
    ]


def test_dynamic_launch_entrypoint_and_native_runner_are_single_owner() -> None:
    root = Path(__file__).resolve().parents[4]
    setup_source = (
        root / "ros2_ws" / "src" / "action_stream_isaac" / "setup.py"
    ).read_text(encoding="utf-8")
    assert (
        "dynamic_isaac_adapter = action_stream_isaac.dynamic_isaac_adapter:main"
        in setup_source
    )
    launch_source = (
        root
        / "ros2_ws"
        / "src"
        / "action_stream_isaac"
        / "launch"
        / "franka_dynamic_recovery.launch.py"
    ).read_text(encoding="utf-8")
    assert launch_source.count('executable="dynamic_isaac_adapter"') == 1
    assert launch_source.count('executable="action_stream_executor_node"') == 1
    assert '"sync_periodic_replan": True' in launch_source
    assert "rmw_zenohd" in launch_source
    assert 'executable="dynamic_policy_node"' not in launch_source
    runner_source = (root / "scripts" / "m8_run_isaac.ps1").read_text(encoding="utf-8")
    assert "colcon build --base-paths '$escapedSource'" in runner_source
    assert "python -m action_stream_isaac.dynamic_isaac_adapter" in runner_source
    assert "ros2 launch" not in runner_source
    assert "sync_periodic_replan:=true" in runner_source
    assert "freeze-validate" in runner_source
    assert "AuthorizeNativeGpuRun" in runner_source
    assert "gpu_memory_refusal_threshold_mib" in runner_source
    assert "RedirectStandardOutput" in runner_source
    assert "preflight_receipt.json" in runner_source
    assert "completion_receipt.json" in runner_source
    assert "actionable_compute_allocation" in runner_source
    assert "$null -ne $usedMemory -and $usedMemory -gt 0" in runner_source
    linux_runner_source = (root / "scripts" / "m8_run_isaac.sh").read_text(
        encoding="utf-8"
    )
    assert "colcon build" in linux_runner_source
    assert "exec python -m action_stream_isaac.dynamic_isaac_adapter" in linux_runner_source
    assert "ros2 launch" not in linux_runner_source
    assert "sync_periodic_replan:=true" in linux_runner_source
    assert "freeze-validate" in linux_runner_source
    assert "--authorize-native-gpu-run" in linux_runner_source
    assert "preflight_receipt.json" in linux_runner_source
    assert "completion_receipt.json" in linux_runner_source
    assert "external_environment.json" in linux_runner_source
    assert linux_runner_source.count("run --frozen --manifest-path") == 4
    linux_support_source = (
        root / "scripts" / "m8_linux_runner_support.py"
    ).read_text(encoding="utf-8")
    assert "gpu_memory_refusal_threshold_mib" in linux_support_source
    demo_source = (root / "scripts" / "m8_record_demo.ps1").read_text(encoding="utf-8")
    assert "python -m action_stream_isaac.dynamic_isaac_adapter" in demo_source
    assert "--video-output" in demo_source
    assert "ros2 launch" not in demo_source
    assert "rmw_zenohd.exe" in demo_source
    assert "AuthorizeNativeGpuRun" in demo_source
    assert "demo_receipt.json" in demo_source
    assert "actionable_compute_allocation" in demo_source
