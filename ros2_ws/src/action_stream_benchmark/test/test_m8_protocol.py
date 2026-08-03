from __future__ import annotations

import json
from pathlib import Path

import pytest

import action_stream_benchmark.m8_replay as replay_module
from action_stream_benchmark.m8_archive import build_archive
from action_stream_benchmark.m8_matrix import build_matrix_manifest
from action_stream_benchmark.m8_protocol import (
    REQUIRED_FREEZE_SOURCE_PATHS,
    SAFE_HOLD_COMMAND,
    STRONG_OBSOLETE_STEP_REDUCTION_THRESHOLD,
    build_episode_fairness,
    build_freeze_manifest,
    close_calibration_ledger,
    load_protocol,
    load_seed_file,
    protocol_component_hashes,
    record_development_candidate,
    record_native_baseline_gate,
    ros_source_manifest,
    sha256_file,
    validate_baseline_gate_artifacts,
    validate_calibration_ledger,
    validate_development_calibration_lifecycle,
    validate_protocol,
    validate_seed_splits,
)
from action_stream_benchmark.m8_scenario import generate_scenario
from action_stream_benchmark.schema import canonical_sha256, read_json, write_json_atomic


ROOT = Path(__file__).resolve().parents[4]


def test_candidate_protocol_seed_contract_and_component_hashes() -> None:
    protocol = load_protocol(ROOT / "configs/m8_g0.json")
    baseline = load_seed_file(ROOT / "configs/m8_baseline_gate_seeds.json")
    development = load_seed_file(ROOT / "configs/m8_development_seeds.json")
    holdout = load_seed_file(ROOT / "configs/m8_holdout_seeds.json")
    validate_seed_splits(baseline=baseline, development=development, holdout=holdout)
    assert len(baseline) == 20
    assert len(development) == 12
    assert len(holdout) == 60
    development_manifest = json.loads(
        (ROOT / "configs/m8_development_seeds.json").read_text(encoding="utf-8")
    )
    assert development_manifest["maximum_candidates"] == 3
    assert development_manifest["maximum_bounded_calibration_changes"] == 2
    assert protocol["protocol_status"].startswith("preregistered_candidate")
    hashes = protocol_component_hashes(protocol)
    assert set(hashes) == {
        "protocol_sha256",
        "controller_sha256",
        "task_contract_sha256",
        "safety_limits_sha256",
    }
    assert all(len(value) == 64 for value in hashes.values())
    assert tuple(protocol["runtime"]["safe_hold_command"]) == SAFE_HOLD_COMMAND
    assert (
        protocol["strong_go_gate"][
            "minimum_aligned_obsolete_control_step_reduction_vs_naive"
        ]
        == STRONG_OBSOLETE_STEP_REDUCTION_THRESHOLD
    )


def test_freeze_covers_native_runner_code_build_and_message_dependencies() -> None:
    required = set(REQUIRED_FREEZE_SOURCE_PATHS)
    expected = {
        "ros2_ws/src/action_stream_executor/CMakeLists.txt",
        "ros2_ws/src/action_stream_executor/include/action_stream_executor/executor_node.hpp",
        "ros2_ws/src/action_stream_executor/src/main.cpp",
        "ros2_ws/src/action_stream_isaac/action_stream_isaac/ros_contract.py",
        "ros2_ws/src/action_stream_benchmark/action_stream_benchmark/m8_scenario.py",
        "ros2_ws/src/action_stream_msgs/CMakeLists.txt",
        "ros2_ws/src/action_stream_msgs/msg/TargetAction.msg",
        "ros2_ws/src/action_stream_msgs/msg/ExecutorDiagnostics.msg",
        "ros2_ws/src/action_stream_isaac/setup.py",
        "ros2_ws/src/action_stream_policy/setup.py",
        "ros2_ws/src/action_stream_benchmark/setup.py",
        "scripts/m8_run_isaac.ps1",
    }
    assert expected <= required
    assert len(required) == len(REQUIRED_FREEZE_SOURCE_PATHS)
    assert all((ROOT / relative).is_file() for relative in REQUIRED_FREEZE_SOURCE_PATHS)


def test_safety_hash_binds_exact_safe_hold_command() -> None:
    protocol = load_protocol(ROOT / "configs/m8_g0.json")
    baseline_hash = protocol_component_hashes(protocol)["safety_limits_sha256"]
    drifted = json.loads(json.dumps(protocol))
    drifted["runtime"]["safe_hold_command"][0] += 0.001
    with pytest.raises(ValueError, match="safe_hold_command"):
        protocol_component_hashes(drifted)
    assert protocol_component_hashes(protocol)["safety_limits_sha256"] == baseline_hash


def test_protocol_preregisters_strict_paired_reset_tolerances() -> None:
    protocol = load_protocol(ROOT / "configs/m8_g0.json")
    drifted = json.loads(json.dumps(protocol))
    drifted["paired_reset_fairness"]["pairwise_tolerances"][
        "position_l2_m"
    ] = 0.001

    with pytest.raises(ValueError, match="paired-reset fairness"):
        validate_protocol(drifted)


def test_episode_fairness_binds_physical_reset() -> None:
    protocol = load_protocol(ROOT / "configs/m8_g0.json")
    scenario = generate_scenario(2026081200).to_dict()
    reset = {
        "seed": 2026081200,
        "robot_joint_positions": [0.0] * 9,
        "robot_joint_velocities": [0.0] * 9,
        "end_effector_position_xyz": [0.4, 0.0, 0.4],
        "end_effector_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "object_position_xyz": scenario["object_position_xyz"],
        "object_orientation_wxyz": scenario["object_orientation_wxyz"],
        "zone_a_xyz": scenario["zone_a_xyz"],
        "zone_b_xyz": scenario["zone_b_xyz"],
        "physics_dt_seconds": 0.01,
        "rendering_dt_seconds": 0.05,
        "stage_units_in_meters": 1.0,
        "gravity_xyz": [0.0, 0.0, -9.81],
    }
    first = build_episode_fairness(
        protocol_payload=protocol,
        scenario_payload=scenario,
        reset_state_payload=reset,
        fault_trace_sha256="a" * 64,
    )
    moved = dict(reset)
    moved["robot_joint_positions"] = [0.01] + [0.0] * 8
    second = build_episode_fairness(
        protocol_payload=protocol,
        scenario_payload=scenario,
        reset_state_payload=moved,
        fault_trace_sha256="a" * 64,
    )
    assert first["reset_state_sha256"] != second["reset_state_sha256"]
    assert first["scenario_sha256"] == scenario["scenario_sha256"]


def test_holdout_freeze_refuses_open_candidate_ledger() -> None:
    with pytest.raises(ValueError, match="finalized after development"):
        build_freeze_manifest(
            repository_root=ROOT,
            protocol_path=ROOT / "configs/m8_g0.json",
            baseline_seed_path=ROOT / "configs/m8_baseline_gate_seeds.json",
            development_seed_path=ROOT / "configs/m8_development_seeds.json",
            holdout_seed_path=ROOT / "configs/m8_holdout_seeds.json",
            calibration_ledger_path=ROOT / "configs/m8_calibration_ledger.json",
            baseline_matrix_path=ROOT / "outputs/m8_g0/baseline_gate/matrix_suite.json",
            baseline_replay_path=ROOT / "outputs/m8_g0/baseline_gate/replay_validation.json",
            baseline_completion_receipt_path=ROOT / "outputs/m8_g0/baseline_gate/completion_receipt.json",
            profile_paths=[
                ROOT / "ros2_ws/src/action_stream_benchmark/config/m8_profile_0_sanity.json",
                ROOT / "ros2_ws/src/action_stream_benchmark/config/m8_profile_1_fixed.json",
                ROOT / "ros2_ws/src/action_stream_benchmark/config/m8_profile_2_faults.json",
            ],
        )
    ledger = json.loads((ROOT / "configs/m8_calibration_ledger.json").read_text())
    assert validate_calibration_ledger(ledger)["status"] == "open_before_native_baseline"


def test_split_aware_persistent_matrix_is_portable(tmp_path: Path) -> None:
    profile0 = ROOT / "ros2_ws/src/action_stream_benchmark/config/m8_profile_0_sanity.json"
    output = tmp_path / "baseline"
    manifest_path = output / "matrix_suite.json"
    matrix = build_matrix_manifest(
        repository_root=ROOT,
        freeze_manifest_path=None,
        candidate_protocol_path=ROOT / "configs/m8_g0.json",
        seed_path=ROOT / "configs/m8_baseline_gate_seeds.json",
        profile_paths=[profile0],
        output_root=output,
        manifest_path=manifest_path,
        split="baseline_gate",
        request_count=32,
    )
    assert matrix["expected_episode_count"] == 20
    assert matrix["persistent_batch_count"] == 1
    assert matrix["native_results_status_at_creation"] == "not_run"
    assert "native_results_status" not in matrix
    assert matrix["batch_manifests"][0]["strategy"] == "sync_hold"
    serialized = manifest_path.read_text(encoding="utf-8")
    assert "C:\\" not in serialized and "/home/" not in serialized
    assert all(not Path(record["path"]).is_absolute() for record in matrix["profiles"].values())

    dev_output = tmp_path / "development"
    development = build_matrix_manifest(
        repository_root=ROOT,
        freeze_manifest_path=None,
        candidate_protocol_path=ROOT / "configs/m8_g0.json",
        seed_path=ROOT / "configs/m8_development_seeds.json",
        profile_paths=[ROOT / "ros2_ws/src/action_stream_benchmark/config/m8_profile_1_fixed.json"],
        output_root=dev_output,
        manifest_path=dev_output / "matrix_suite.json",
        split="development",
        request_count=32,
    )
    assert development["expected_episode_count"] == 36
    assert development["persistent_batch_count"] == 3

    demo_seed_path = tmp_path / "demo_seed.json"
    write_json_atomic(
        demo_seed_path,
        {"schema_version": 1, "milestone": "M8-G0", "seeds": [2026081100]},
    )
    demo_output = tmp_path / "demo"
    demo = build_matrix_manifest(
        repository_root=ROOT,
        freeze_manifest_path=None,
        candidate_protocol_path=ROOT / "configs/m8_g0.json",
        seed_path=demo_seed_path,
        profile_paths=[
            ROOT / "ros2_ws/src/action_stream_benchmark/config/m8_profile_1_fixed.json"
        ],
        output_root=demo_output / "raw",
        manifest_path=demo_output / "matrix.json",
        split="development",
        request_count=32,
        strategy_filter=["aligned_async"],
    )
    assert demo["expected_episode_count"] == 1
    assert demo["persistent_batch_count"] == 1
    assert demo["development_strategy_filter"] == ["aligned_async"]
    assert demo["episodes"][0]["strategy"] == "aligned_async"

    with pytest.raises(ValueError, match="only for non-headline development"):
        build_matrix_manifest(
            repository_root=ROOT,
            freeze_manifest_path=None,
            candidate_protocol_path=ROOT / "configs/m8_g0.json",
            seed_path=ROOT / "configs/m8_baseline_gate_seeds.json",
            profile_paths=[profile0],
            output_root=tmp_path / "invalid-filter" / "raw",
            manifest_path=tmp_path / "invalid-filter" / "matrix.json",
            split="baseline_gate",
            request_count=32,
            strategy_filter=["sync_hold"],
        )


def _baseline_gate_fixture(tmp_path: Path) -> tuple:
    repository_root = tmp_path
    adapter_path = (
        repository_root
        / "ros2_ws/src/action_stream_isaac/action_stream_isaac/dynamic_isaac_adapter.py"
    )
    adapter_path.parent.mkdir(parents=True)
    adapter_path.write_text("# installed adapter source\n", encoding="utf-8")
    installed_adapter_path = (
        repository_root
        / "native_install/action_stream_isaac/action_stream_isaac/dynamic_isaac_adapter.py"
    )
    installed_adapter_path.parent.mkdir(parents=True)
    installed_adapter_path.write_bytes(adapter_path.read_bytes())
    runner_path = repository_root / "scripts/m8_run_isaac.ps1"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("# native runner\n", encoding="utf-8")
    matrix_path = tmp_path / "baseline_matrix.json"
    seed_path = tmp_path / "baseline_seeds.json"
    write_json_atomic(
        seed_path,
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "split": "simulator_baseline_gate",
            "seeds": list(range(20)),
        },
    )
    profile_directory = tmp_path / "profiles"
    profile_directory.mkdir()
    profile_payloads = {}
    profile_paths = {}
    for profile_id in (
        "profile_0_sanity",
        "profile_1_fixed",
        "profile_2_faults",
    ):
        source = (
            ROOT
            / "ros2_ws/src/action_stream_benchmark/config"
            / f"m8_{profile_id}.json"
        )
        destination = profile_directory / source.name
        payload = json.loads(source.read_text(encoding="utf-8"))
        write_json_atomic(destination, payload)
        profile_payloads[profile_id] = payload
        profile_paths[profile_id] = destination
    profile_path = profile_paths["profile_0_sanity"]
    candidate = json.loads((ROOT / "configs/m8_g0.json").read_text(encoding="utf-8"))
    for profile_id, candidate_profile in candidate["profiles"].items():
        candidate_profile["profile_file"] = profile_paths[profile_id].relative_to(
            repository_root
        ).as_posix()
        candidate_profile["profile_sha256"] = sha256_file(profile_paths[profile_id])
    candidate = validate_protocol(candidate)
    candidate_path = tmp_path / "baseline_candidate_protocol.json"
    write_json_atomic(candidate_path, candidate)
    protocol = json.loads(json.dumps(candidate))
    protocol["protocol_status"] = "ready_for_holdout_freeze"
    protocol["holdout_freeze_status"] = "ready_to_freeze"
    episodes = [
        {
            "episode_id": f"baseline-{seed}",
            "profile_id": "profile_0_sanity",
            "strategy": "sync_hold",
            "seed": seed,
        }
        for seed in range(20)
    ]
    batch_path = tmp_path / "batch_profile_0_sanity_sync_hold.json"
    write_json_atomic(
        batch_path,
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "split": "baseline_gate",
            "batch_profile_id": "profile_0_sanity",
            "batch_strategy": "sync_hold",
            "expected_episode_count": 20,
            "episodes": episodes,
        },
    )
    write_json_atomic(
        matrix_path,
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "split": "baseline_gate",
            "evidence_class": "ros_cpp_isaac_sim",
            "headline_eligible": False,
            "candidate_manifest": candidate_path.name,
            "protocol_sha256": protocol_component_hashes(candidate)["protocol_sha256"],
            "seed_file": seed_path.name,
            "seed_file_sha256": sha256_file(seed_path),
            "profiles": {
                "profile_0_sanity": {
                    "path": profile_path.relative_to(matrix_path.parent).as_posix(),
                    "sha256": sha256_file(profile_path),
                }
            },
            "expected_episode_count": 20,
            "persistent_batch_count": 1,
            "batch_manifests": [
                {
                    "profile_id": "profile_0_sanity",
                    "strategy": "sync_hold",
                    "path": batch_path.name,
                    "episode_count": 20,
                }
            ],
            "episodes": episodes,
        },
    )
    audits = [
        {
            "passed": True,
            "strategy": "sync_hold",
            "recomputed_metrics": {"task_success": index < 18},
        }
        for index in range(20)
    ]
    fresh = {
        "schema_version": 1,
        "milestone": "M8-G0",
        "manifest": str(matrix_path.resolve()),
        "manifest_sha256": sha256_file(matrix_path),
        "split": "baseline_gate",
        "episode_count": 20,
        "episode_audits_passed": 20,
        "fairness_passed": True,
        "provenance_validation_passed": True,
        "seed_validation_passed": True,
        "seed_count": 20,
        "seed_file_sha256": sha256_file(seed_path),
        "profile_validation_passed": True,
        "audits": audits,
        "passed": True,
    }
    replay_path = tmp_path / "baseline_replay.json"
    write_json_atomic(replay_path, fresh)
    run_log_dir = tmp_path / "native_run_logs/run-001"
    run_log_dir.mkdir(parents=True)
    process_log = run_log_dir / "router.stdout.log"
    process_log.write_text("native process completed\n", encoding="utf-8")
    validation_log = run_log_dir / f"{batch_path.stem}.validate_only.log"
    validation_log.write_text(
        '{"validation_passed":true,"native_execution_performed":false}\n',
        encoding="utf-8",
    )
    process_log_archive = run_log_dir / "process_logs.tar.gz"
    process_log_manifest = run_log_dir / "process_logs.manifest.json"
    archive_payload = build_archive(
        root=run_log_dir,
        members=[process_log.name, validation_log.name],
        archive_path=process_log_archive,
        manifest_path=process_log_manifest,
    )
    process_log.unlink()
    validation_log.unlink()
    source = ros_source_manifest(repository_root)
    preflight_path = run_log_dir / "preflight_receipt.json"
    write_json_atomic(
        preflight_path,
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "receipt_kind": "native_preflight",
            "operator_authorized_native_gpu_run": True,
            "gpu_memory_refusal_threshold_mib": 4096,
            "gpu_inventory": [
                {
                    "index": 0,
                    "name": "Test GPU",
                    "uuid": "GPU-test-0",
                    "driver_version": "999.0",
                    "memory_total_mib": 24576,
                    "memory_used_mib": 512,
                    "utilization_gpu_percent": 2,
                }
            ],
            "reported_compute_processes": [],
            "preexisting_compute_processes": [],
            "suite_manifest": matrix_path.relative_to(repository_root).as_posix(),
            "suite_manifest_sha256": sha256_file(matrix_path),
            "source_root": "ros2_ws/src",
            "source_file_count": source["source_file_count"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "installed_dynamic_adapter": str(installed_adapter_path.resolve()),
            "installed_dynamic_adapter_sha256": sha256_file(adapter_path),
            "runner_sha256": sha256_file(runner_path),
            "current_source_batch_validation_passed": True,
            "current_source_batch_validation_count": 1,
            "current_source_batch_validation_logs": [validation_log.name],
            "planned_process_logs": [process_log.name, validation_log.name],
        },
    )
    completion_path = run_log_dir / "completion_receipt.json"
    write_json_atomic(
        completion_path,
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "receipt_kind": "native_completion",
            "status": "complete",
            "preflight_receipt": preflight_path.name,
            "preflight_receipt_sha256": sha256_file(preflight_path),
            "replay_validation": "../../baseline_replay.json",
            "replay_validation_sha256": sha256_file(replay_path),
            "analysis": None,
            "analysis_sha256": None,
            "process_log_archive": process_log_archive.name,
            "process_log_archive_sha256": sha256_file(process_log_archive),
            "process_log_manifest": process_log_manifest.name,
            "process_log_manifest_sha256": sha256_file(process_log_manifest),
            "process_log_member_count": len(archive_payload["members"]),
        },
    )
    development_seed_path = tmp_path / "development_seeds.json"
    write_json_atomic(
        development_seed_path,
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "split": "development",
            "seeds": list(range(100, 112)),
        },
    )
    development_directory = tmp_path / "development/candidate_0"
    development_directory.mkdir(parents=True)
    development_matrix_path = development_directory / "matrix.json"
    development_matrix = {
        "schema_version": 1,
        "milestone": "M8-G0",
        "split": "development",
        "evidence_class": "ros_cpp_isaac_sim",
        "headline_eligible": False,
        "candidate_manifest": "../../baseline_candidate_protocol.json",
        "protocol_sha256": canonical_sha256(candidate),
        "seed_file": "../../development_seeds.json",
        "seed_file_sha256": sha256_file(development_seed_path),
        "seed_count": 12,
        "profiles": {
            profile_id: {
                "path": f"../../profiles/{profile_paths[profile_id].name}",
                "sha256": sha256_file(profile_paths[profile_id]),
            }
            for profile_id in ("profile_1_fixed", "profile_2_faults")
        },
        "expected_episode_count": 72,
        "episodes": [],
    }
    write_json_atomic(development_matrix_path, development_matrix)
    development_replay_path = development_directory / "replay_validation.json"
    development_replay = {
        "schema_version": 1,
        "milestone": "M8-G0",
        "manifest": str(development_matrix_path.resolve()),
        "manifest_sha256": sha256_file(development_matrix_path),
        "split": "development",
        "episode_count": 72,
        "episode_audits_passed": 72,
        "seed_count": 12,
        "passed": True,
    }
    write_json_atomic(development_replay_path, development_replay)
    behavioral_protocol = dict(protocol)
    behavioral_protocol.pop("protocol_status")
    behavioral_protocol.pop("holdout_freeze_status")
    behavioral_contract = {
        "protocol": behavioral_protocol,
        "profiles": profile_payloads,
    }
    candidate_record = {
        "candidate_id": "candidate_0",
        "protocol_path": candidate_path.relative_to(repository_root).as_posix(),
        "protocol_file_sha256": sha256_file(candidate_path),
        "protocol_sha256": canonical_sha256(candidate),
        "contract_sha256": canonical_sha256(behavioral_contract),
        "matrix_path": development_matrix_path.relative_to(repository_root).as_posix(),
        "matrix_sha256": sha256_file(development_matrix_path),
        "replay_path": development_replay_path.relative_to(repository_root).as_posix(),
        "replay_sha256": sha256_file(development_replay_path),
        "seed_count": 12,
        "replay_validated": True,
        "headline_eligible": False,
    }
    ledger = {
        "schema_version": 1,
        "milestone": "M8-G0",
        "status": "closed_before_holdout",
        "maximum_bounded_calibration_changes": 2,
        "bounded_calibration_changes": [],
        "baseline_gate": {
            "native_isaac_physics": True,
            "status": "passed",
            "profile_id": "profile_0_sanity",
            "strategy": "sync_hold",
            "evidence_class": "ros_cpp_isaac_sim",
            "seed_count": 20,
            "trial_count": 20,
            "success_count": 18,
            "required_success_rate": 0.9,
            "observed_success_rate": 0.9,
            "replay_validated": True,
            "replay_sha256": sha256_file(replay_path),
            "matrix_sha256": sha256_file(matrix_path),
            "seed_file_sha256": sha256_file(seed_path),
            "profile_sha256": sha256_file(profile_path),
            "protocol_sha256": protocol_component_hashes(protocol)["protocol_sha256"],
            "baseline_candidate_protocol_sha256": protocol_component_hashes(candidate)[
                "protocol_sha256"
            ],
            "behavioral_protocol_contract_sha256": canonical_sha256(behavioral_contract),
            "selected_development_candidate_id": "candidate_0",
            "bounded_calibration_change_count": 0,
            "completion_receipt_sha256": sha256_file(completion_path),
            "preflight_receipt_sha256": sha256_file(preflight_path),
            "source_file_count": source["source_file_count"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "runner_sha256": sha256_file(runner_path),
            "installed_dynamic_adapter_sha256": sha256_file(adapter_path),
            "process_log_archive_sha256": sha256_file(process_log_archive),
            "process_log_manifest_sha256": sha256_file(process_log_manifest),
        },
        "development": {
            "status": "complete",
            "seed_count_per_candidate": 12,
            "headline_eligible": False,
            "selected_candidate_id": "candidate_0",
            "candidates": [candidate_record],
        },
        "holdout_freeze_authorized": True,
        "first_holdout_started": False,
    }
    return (
        matrix_path,
        replay_path,
        fresh,
        ledger,
        seed_path,
        profile_path,
        protocol,
        completion_path,
        repository_root,
        adapter_path,
    )


def _fixture_contract(protocol: dict, repository_root: Path) -> dict:
    behavioral = json.loads(json.dumps(protocol))
    behavioral.pop("protocol_status")
    behavioral.pop("holdout_freeze_status")
    profiles = {
        profile_id: json.loads(
            (repository_root / record["profile_file"]).read_text(encoding="utf-8")
        )
        for profile_id, record in protocol["profiles"].items()
    }
    return {"protocol": behavioral, "profiles": profiles}


def _add_calibrated_candidate(
    *,
    repository_root: Path,
    ledger: dict,
    path: str,
    value,
) -> tuple[dict, dict]:
    candidate0_record = ledger["development"]["candidates"][0]
    candidate0_path = repository_root / candidate0_record["protocol_path"]
    candidate0 = json.loads(candidate0_path.read_text(encoding="utf-8"))
    candidate1 = json.loads(json.dumps(candidate0))
    target = candidate1
    parts = path.strip("/").split("/")
    assert parts[0] == "protocol"
    for part in parts[1:-1]:
        target = target[part]
    before_value = target[parts[-1]]
    target[parts[-1]] = value
    candidate1 = validate_protocol(candidate1)
    candidate1_path = repository_root / "candidate_1_protocol.json"
    write_json_atomic(candidate1_path, candidate1)

    matrix0_path = repository_root / candidate0_record["matrix_path"]
    matrix1_directory = repository_root / "development/candidate_1"
    matrix1_directory.mkdir(parents=True)
    matrix1_path = matrix1_directory / "matrix.json"
    matrix1 = json.loads(matrix0_path.read_text(encoding="utf-8"))
    matrix1["candidate_manifest"] = "../../candidate_1_protocol.json"
    matrix1["protocol_sha256"] = canonical_sha256(candidate1)
    write_json_atomic(matrix1_path, matrix1)
    replay1_path = matrix1_directory / "replay_validation.json"
    replay1 = json.loads(
        (repository_root / candidate0_record["replay_path"]).read_text(encoding="utf-8")
    )
    replay1["manifest"] = str(matrix1_path.resolve())
    replay1["manifest_sha256"] = sha256_file(matrix1_path)
    write_json_atomic(replay1_path, replay1)

    contract0 = _fixture_contract(candidate0, repository_root)
    contract1 = _fixture_contract(candidate1, repository_root)
    candidate1_record = {
        "candidate_id": "candidate_1",
        "protocol_path": candidate1_path.relative_to(repository_root).as_posix(),
        "protocol_file_sha256": sha256_file(candidate1_path),
        "protocol_sha256": canonical_sha256(candidate1),
        "contract_sha256": canonical_sha256(contract1),
        "matrix_path": matrix1_path.relative_to(repository_root).as_posix(),
        "matrix_sha256": sha256_file(matrix1_path),
        "replay_path": replay1_path.relative_to(repository_root).as_posix(),
        "replay_sha256": sha256_file(replay1_path),
        "seed_count": 12,
        "replay_validated": True,
        "headline_eligible": False,
    }
    ledger["bounded_calibration_changes"] = [
        {
            "change_id": "calibration_1",
            "reason": "bounded development calibration test",
            "before_candidate_id": "candidate_0",
            "after_candidate_id": "candidate_1",
            "before_contract_sha256": canonical_sha256(contract0),
            "after_contract_sha256": canonical_sha256(contract1),
            "before": {path: before_value},
            "after": {path: value},
            "development_evidence": {
                "candidate_id": "candidate_0",
                "matrix_sha256": candidate0_record["matrix_sha256"],
                "replay_sha256": candidate0_record["replay_sha256"],
            },
        }
    ]
    ledger["development"]["candidates"].append(candidate1_record)
    ledger["development"]["selected_candidate_id"] = "candidate_1"
    finalized = json.loads(json.dumps(candidate1))
    finalized["protocol_status"] = "ready_for_holdout_freeze"
    finalized["holdout_freeze_status"] = "ready_to_freeze"
    return finalized, ledger


def test_documented_bounded_calibration_selects_candidate_1(tmp_path: Path) -> None:
    fixture = _baseline_gate_fixture(tmp_path)
    ledger = fixture[3]
    repository_root = fixture[8]
    finalized, ledger = _add_calibrated_candidate(
        repository_root=repository_root,
        ledger=ledger,
        path="/protocol/controller/maximum_translation_per_step_m",
        value=0.012,
    )

    result = validate_development_calibration_lifecycle(
        repository_root=repository_root,
        baseline_candidate_protocol_path=repository_root
        / "baseline_candidate_protocol.json",
        finalized_protocol_payload=finalized,
        ledger_payload=ledger,
        expected_development_seed_path=repository_root / "development_seeds.json",
    )

    assert result["selected_candidate_id"] == "candidate_1"
    assert result["bounded_calibration_change_count"] == 1
    assert result["candidate_count"] == 2


def test_documented_latency_calibration_binds_immutable_profile_candidates(
    tmp_path: Path,
) -> None:
    fixture = _baseline_gate_fixture(tmp_path)
    ledger = fixture[3]
    repository_root = fixture[8]
    candidate0_record = ledger["development"]["candidates"][0]
    candidate0 = json.loads(
        (repository_root / candidate0_record["protocol_path"]).read_text(
            encoding="utf-8"
        )
    )
    candidate1 = json.loads(json.dumps(candidate0))
    calibrated_directory = repository_root / "profiles/candidate_1"
    calibrated_directory.mkdir(parents=True)
    changed_paths = {}
    for profile_id in ("profile_1_fixed", "profile_2_faults"):
        source_record = candidate0["profiles"][profile_id]
        source_path = repository_root / source_record["profile_file"]
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        payload["profile"]["base_latency_ms"] = 900
        destination = calibrated_directory / source_path.name
        write_json_atomic(destination, payload)
        candidate1["profiles"][profile_id]["profile_file"] = destination.relative_to(
            repository_root
        ).as_posix()
        candidate1["profiles"][profile_id]["profile_sha256"] = sha256_file(destination)
        changed_paths.update(
            {
                f"/protocol/profiles/{profile_id}/profile_file": (
                    source_record["profile_file"],
                    candidate1["profiles"][profile_id]["profile_file"],
                ),
                f"/protocol/profiles/{profile_id}/profile_sha256": (
                    source_record["profile_sha256"],
                    candidate1["profiles"][profile_id]["profile_sha256"],
                ),
                f"/profiles/{profile_id}/profile/base_latency_ms": (850, 900),
            }
        )
    candidate1 = validate_protocol(candidate1)
    candidate1_path = repository_root / "candidate_1_protocol.json"
    write_json_atomic(candidate1_path, candidate1)

    matrix0 = json.loads(
        (repository_root / candidate0_record["matrix_path"]).read_text(encoding="utf-8")
    )
    matrix1_directory = repository_root / "development/candidate_1"
    matrix1_directory.mkdir(parents=True)
    matrix1_path = matrix1_directory / "matrix.json"
    matrix0["candidate_manifest"] = "../../candidate_1_protocol.json"
    matrix0["protocol_sha256"] = canonical_sha256(candidate1)
    for profile_id in ("profile_1_fixed", "profile_2_faults"):
        profile_path = repository_root / candidate1["profiles"][profile_id]["profile_file"]
        matrix0["profiles"][profile_id] = {
            "path": f"../../profiles/candidate_1/{profile_path.name}",
            "sha256": sha256_file(profile_path),
        }
    write_json_atomic(matrix1_path, matrix0)
    replay1_path = matrix1_directory / "replay_validation.json"
    replay1 = json.loads(
        (repository_root / candidate0_record["replay_path"]).read_text(encoding="utf-8")
    )
    replay1["manifest"] = str(matrix1_path.resolve())
    replay1["manifest_sha256"] = sha256_file(matrix1_path)
    write_json_atomic(replay1_path, replay1)

    contract0 = _fixture_contract(candidate0, repository_root)
    contract1 = _fixture_contract(candidate1, repository_root)
    candidate1_record = {
        "candidate_id": "candidate_1",
        "protocol_path": candidate1_path.relative_to(repository_root).as_posix(),
        "protocol_file_sha256": sha256_file(candidate1_path),
        "protocol_sha256": canonical_sha256(candidate1),
        "contract_sha256": canonical_sha256(contract1),
        "matrix_path": matrix1_path.relative_to(repository_root).as_posix(),
        "matrix_sha256": sha256_file(matrix1_path),
        "replay_path": replay1_path.relative_to(repository_root).as_posix(),
        "replay_sha256": sha256_file(replay1_path),
        "seed_count": 12,
        "replay_validated": True,
        "headline_eligible": False,
    }
    ledger["bounded_calibration_changes"] = [
        {
            "change_id": "calibration_1",
            "reason": "choose the fixed latency using development seeds",
            "before_candidate_id": "candidate_0",
            "after_candidate_id": "candidate_1",
            "before_contract_sha256": canonical_sha256(contract0),
            "after_contract_sha256": canonical_sha256(contract1),
            "before": {path: values[0] for path, values in changed_paths.items()},
            "after": {path: values[1] for path, values in changed_paths.items()},
            "development_evidence": {
                "candidate_id": "candidate_0",
                "matrix_sha256": candidate0_record["matrix_sha256"],
                "replay_sha256": candidate0_record["replay_sha256"],
            },
        }
    ]
    ledger["development"]["candidates"].append(candidate1_record)
    ledger["development"]["selected_candidate_id"] = "candidate_1"
    finalized = json.loads(json.dumps(candidate1))
    finalized["protocol_status"] = "ready_for_holdout_freeze"
    finalized["holdout_freeze_status"] = "ready_to_freeze"

    result = validate_development_calibration_lifecycle(
        repository_root=repository_root,
        baseline_candidate_protocol_path=repository_root
        / "baseline_candidate_protocol.json",
        finalized_protocol_payload=finalized,
        ledger_payload=ledger,
    )

    assert result["selected_candidate_id"] == "candidate_1"
    assert result["selected_contract_sha256"] == canonical_sha256(contract1)


def test_calibration_lifecycle_rejects_undocumented_and_unauthorized_drift(
    tmp_path: Path,
) -> None:
    fixture = _baseline_gate_fixture(tmp_path)
    ledger = fixture[3]
    repository_root = fixture[8]
    finalized, ledger = _add_calibrated_candidate(
        repository_root=repository_root,
        ledger=ledger,
        path="/protocol/task/placement_tolerance_m",
        value=0.045,
    )

    with pytest.raises(ValueError, match="unauthorized contract paths"):
        validate_development_calibration_lifecycle(
            repository_root=repository_root,
            baseline_candidate_protocol_path=repository_root
            / "baseline_candidate_protocol.json",
            finalized_protocol_payload=finalized,
            ledger_payload=ledger,
        )

    fixture = _baseline_gate_fixture(tmp_path / "tampered-diff")
    ledger = fixture[3]
    repository_root = fixture[8]
    finalized, ledger = _add_calibrated_candidate(
        repository_root=repository_root,
        ledger=ledger,
        path="/protocol/controller/maximum_translation_per_step_m",
        value=0.012,
    )
    ledger["bounded_calibration_changes"][0]["before"][
        "/protocol/controller/maximum_translation_per_step_m"
    ] = 0.009
    with pytest.raises(ValueError, match="before/after values do not equal"):
        validate_development_calibration_lifecycle(
            repository_root=repository_root,
            baseline_candidate_protocol_path=repository_root
            / "baseline_candidate_protocol.json",
            finalized_protocol_payload=finalized,
            ledger_payload=ledger,
        )


def test_baseline_gate_replays_raw_and_binds_exact_ledger(monkeypatch, tmp_path: Path) -> None:
    matrix_path, replay_path, fresh, ledger, seed_path, profile_path, protocol, completion_path, repository_root, _adapter_path = _baseline_gate_fixture(tmp_path)
    monkeypatch.setattr(replay_module, "validate_manifest", lambda _path: fresh)
    result = validate_baseline_gate_artifacts(
        matrix_path=matrix_path,
        replay_path=replay_path,
        baseline_seed_path=seed_path,
        profile_path=profile_path,
        protocol_payload=protocol,
        completion_receipt_path=completion_path,
        repository_root=repository_root,
        ledger_payload=ledger,
    )
    assert result["success_count"] == 18
    assert result["trial_count"] == 20
    assert result["matrix_sha256"] == sha256_file(matrix_path)
    assert result["replay_sha256"] == sha256_file(replay_path)
    assert result["process_log_archive_sha256"] == ledger["baseline_gate"][
        "process_log_archive_sha256"
    ]
    assert result["process_log_manifest_sha256"] == ledger["baseline_gate"][
        "process_log_manifest_sha256"
    ]
    assert not list(completion_path.parent.glob("*.log"))


@pytest.mark.parametrize("tamper_kind", ["archive", "manifest"])
def test_baseline_gate_rejects_process_log_archive_tampering(
    monkeypatch, tmp_path: Path, tamper_kind: str
) -> None:
    (
        matrix_path,
        replay_path,
        fresh,
        ledger,
        seed_path,
        profile_path,
        protocol,
        completion_path,
        repository_root,
        _adapter_path,
    ) = _baseline_gate_fixture(tmp_path)
    monkeypatch.setattr(replay_module, "validate_manifest", lambda _path: fresh)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if tamper_kind == "archive":
        archive_path = completion_path.parent / completion["process_log_archive"]
        archive_path.write_bytes(archive_path.read_bytes() + b"tamper")
        completion["process_log_archive_sha256"] = sha256_file(archive_path)
    else:
        manifest_path = completion_path.parent / completion["process_log_manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["members"][0]["sha256"] = "0" * 64
        write_json_atomic(manifest_path, manifest)
        completion["process_log_manifest_sha256"] = sha256_file(manifest_path)
    write_json_atomic(completion_path, completion)

    with pytest.raises(ValueError, match="process-log archive failed exact validation"):
        validate_baseline_gate_artifacts(
            matrix_path=matrix_path,
            replay_path=replay_path,
            baseline_seed_path=seed_path,
            profile_path=profile_path,
            protocol_payload=protocol,
            completion_receipt_path=completion_path,
            repository_root=repository_root,
            ledger_payload=ledger,
        )


def test_baseline_gate_rejects_replay_ledger_and_raw_tampering(monkeypatch, tmp_path: Path) -> None:
    matrix_path, replay_path, fresh, ledger, seed_path, profile_path, protocol, completion_path, repository_root, _adapter_path = _baseline_gate_fixture(tmp_path)
    monkeypatch.setattr(replay_module, "validate_manifest", lambda _path: fresh)

    drifted_ledger = json.loads(json.dumps(ledger))
    drifted_ledger["baseline_gate"]["replay_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="does not match replayed evidence"):
        validate_baseline_gate_artifacts(
            matrix_path=matrix_path,
            replay_path=replay_path,
            baseline_seed_path=seed_path,
            profile_path=profile_path,
            protocol_payload=protocol,
            completion_receipt_path=completion_path,
            repository_root=repository_root,
            ledger_payload=drifted_ledger,
        )

    recorded = json.loads(replay_path.read_text(encoding="utf-8"))
    recorded["passed"] = False
    write_json_atomic(replay_path, recorded)
    with pytest.raises(ValueError, match="fresh raw-evidence replay"):
        validate_baseline_gate_artifacts(
            matrix_path=matrix_path,
            replay_path=replay_path,
            baseline_seed_path=seed_path,
            profile_path=profile_path,
            protocol_payload=protocol,
            completion_receipt_path=completion_path,
            repository_root=repository_root,
            ledger_payload=ledger,
        )

    write_json_atomic(replay_path, fresh)
    raw_tampered = json.loads(json.dumps(fresh))
    raw_tampered["audits"][0]["passed"] = False
    monkeypatch.setattr(replay_module, "validate_manifest", lambda _path: raw_tampered)
    with pytest.raises(ValueError, match="fresh raw-evidence replay"):
        validate_baseline_gate_artifacts(
            matrix_path=matrix_path,
            replay_path=replay_path,
            baseline_seed_path=seed_path,
            profile_path=profile_path,
            protocol_payload=protocol,
            completion_receipt_path=completion_path,
            repository_root=repository_root,
            ledger_payload=ledger,
        )


def test_baseline_gate_rejects_below_18_of_20(monkeypatch, tmp_path: Path) -> None:
    matrix_path, replay_path, fresh, ledger, seed_path, profile_path, protocol, completion_path, repository_root, _adapter_path = _baseline_gate_fixture(tmp_path)
    fresh["audits"][17]["recomputed_metrics"]["task_success"] = False
    write_json_atomic(replay_path, fresh)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["replay_validation_sha256"] = sha256_file(replay_path)
    write_json_atomic(completion_path, completion)
    ledger["baseline_gate"].update(
        {
            "success_count": 17,
            "observed_success_rate": 0.85,
            "replay_sha256": sha256_file(replay_path),
            "completion_receipt_sha256": sha256_file(completion_path),
        }
    )
    monkeypatch.setattr(replay_module, "validate_manifest", lambda _path: fresh)
    with pytest.raises(ValueError, match="at least 18/20"):
        validate_baseline_gate_artifacts(
            matrix_path=matrix_path,
            replay_path=replay_path,
            baseline_seed_path=seed_path,
            profile_path=profile_path,
            protocol_payload=protocol,
            completion_receipt_path=completion_path,
            repository_root=repository_root,
            ledger_payload=ledger,
        )


def test_ros_source_manifest_ignores_transient_tool_caches(tmp_path: Path) -> None:
    source_root = tmp_path / "ros2_ws/src/example_package"
    source_root.mkdir(parents=True)
    (source_root / "stable_source.py").write_text("VALUE = 1\n", encoding="utf-8")
    expected = ros_source_manifest(tmp_path)

    transient_files = (
        source_root / ".pytest_cache/v/cache/nodeids",
        source_root / ".ruff_cache/content.bin",
        source_root / "__pycache__/stable_source.cpython-312.pyc",
        source_root / "example_package.egg-info/PKG-INFO",
        source_root / "top_level.pyc",
        source_root / "top_level.pyo",
    )
    for transient in transient_files:
        transient.parent.mkdir(parents=True, exist_ok=True)
        transient.write_bytes(b"transient cache bytes")

    assert ros_source_manifest(tmp_path) == expected


def test_baseline_gate_rejects_behavioral_protocol_drift(monkeypatch, tmp_path: Path) -> None:
    (
        matrix_path,
        replay_path,
        fresh,
        ledger,
        seed_path,
        profile_path,
        protocol,
        completion_path,
        repository_root,
        _adapter_path,
    ) = _baseline_gate_fixture(tmp_path)
    monkeypatch.setattr(replay_module, "validate_manifest", lambda _path: fresh)
    drifted_protocol = json.loads(json.dumps(protocol))
    drifted_protocol["controller"]["maximum_translation_per_step_m"] = 0.02

    with pytest.raises(ValueError, match="behavioral protocol contracts differ"):
        validate_baseline_gate_artifacts(
            matrix_path=matrix_path,
            replay_path=replay_path,
            baseline_seed_path=seed_path,
            profile_path=profile_path,
            protocol_payload=drifted_protocol,
            completion_receipt_path=completion_path,
            repository_root=repository_root,
            ledger_payload=ledger,
        )


def test_baseline_gate_rejects_source_and_absolute_receipt_tampering(
    monkeypatch, tmp_path: Path
) -> None:
    (
        matrix_path,
        replay_path,
        fresh,
        ledger,
        seed_path,
        profile_path,
        protocol,
        completion_path,
        repository_root,
        adapter_path,
    ) = _baseline_gate_fixture(tmp_path)
    monkeypatch.setattr(replay_module, "validate_manifest", lambda _path: fresh)
    adapter_path.write_text("# stable source was tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source manifest does not match"):
        validate_baseline_gate_artifacts(
            matrix_path=matrix_path,
            replay_path=replay_path,
            baseline_seed_path=seed_path,
            profile_path=profile_path,
            protocol_payload=protocol,
            completion_receipt_path=completion_path,
            repository_root=repository_root,
            ledger_payload=ledger,
        )

    # A fresh fixture isolates receipt portability from source-tree tampering.
    (
        matrix_path,
        replay_path,
        fresh,
        ledger,
        seed_path,
        profile_path,
        protocol,
        completion_path,
        repository_root,
        _adapter_path,
    ) = _baseline_gate_fixture(tmp_path / "absolute-receipt")
    monkeypatch.setattr(replay_module, "validate_manifest", lambda _path: fresh)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["replay_validation"] = str(replay_path.resolve())
    write_json_atomic(completion_path, completion)
    with pytest.raises(ValueError, match="receipt replay must be receipt-relative"):
        validate_baseline_gate_artifacts(
            matrix_path=matrix_path,
            replay_path=replay_path,
            baseline_seed_path=seed_path,
            profile_path=profile_path,
            protocol_payload=protocol,
            completion_receipt_path=completion_path,
            repository_root=repository_root,
            ledger_payload=ledger,
        )


@pytest.mark.parametrize("inventory_kind", ["empty", "over_threshold"])
def test_baseline_gate_rejects_invalid_gpu_preflight_inventory(
    monkeypatch, tmp_path: Path, inventory_kind: str
) -> None:
    (
        matrix_path,
        replay_path,
        fresh,
        ledger,
        seed_path,
        profile_path,
        protocol,
        completion_path,
        repository_root,
        _adapter_path,
    ) = _baseline_gate_fixture(tmp_path)
    monkeypatch.setattr(replay_module, "validate_manifest", lambda _path: fresh)
    preflight_path = completion_path.parent / "preflight_receipt.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if inventory_kind == "empty":
        preflight["gpu_inventory"] = []
    else:
        preflight["gpu_inventory"][0]["memory_used_mib"] = 4097
    write_json_atomic(preflight_path, preflight)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["preflight_receipt_sha256"] = sha256_file(preflight_path)
    write_json_atomic(completion_path, completion)

    with pytest.raises(ValueError, match="GPU inventory"):
        validate_baseline_gate_artifacts(
            matrix_path=matrix_path,
            replay_path=replay_path,
            baseline_seed_path=seed_path,
            profile_path=profile_path,
            protocol_payload=protocol,
            completion_receipt_path=completion_path,
            repository_root=repository_root,
            ledger_payload=ledger,
        )


@pytest.mark.parametrize(
    "tamper_kind",
    ("missing_passed", "zero_count", "missing_log", "wrong_log"),
)
def test_baseline_gate_requires_exact_current_source_batch_validation(
    monkeypatch,
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    (
        matrix_path,
        replay_path,
        fresh,
        ledger,
        seed_path,
        profile_path,
        protocol,
        completion_path,
        repository_root,
        _adapter_path,
    ) = _baseline_gate_fixture(tmp_path)
    monkeypatch.setattr(replay_module, "validate_manifest", lambda _path: fresh)
    preflight_path = completion_path.parent / "preflight_receipt.json"
    preflight = read_json(preflight_path)
    if tamper_kind == "missing_passed":
        preflight.pop("current_source_batch_validation_passed")
    elif tamper_kind == "zero_count":
        preflight["current_source_batch_validation_count"] = 0
    elif tamper_kind == "missing_log":
        preflight["current_source_batch_validation_logs"] = []
    else:
        preflight["current_source_batch_validation_logs"] = [
            "unbound.validate_only.log"
        ]
    write_json_atomic(preflight_path, preflight)
    completion = read_json(completion_path)
    completion["preflight_receipt_sha256"] = sha256_file(preflight_path)
    write_json_atomic(completion_path, completion)

    with pytest.raises(ValueError, match="exact current-source batch validation"):
        validate_baseline_gate_artifacts(
            matrix_path=matrix_path,
            replay_path=replay_path,
            baseline_seed_path=seed_path,
            profile_path=profile_path,
            protocol_payload=protocol,
            completion_receipt_path=completion_path,
            repository_root=repository_root,
            ledger_payload=ledger,
        )


def test_baseline_gate_binds_batch_validation_logs_into_process_archive(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (
        matrix_path,
        replay_path,
        fresh,
        ledger,
        seed_path,
        profile_path,
        protocol,
        completion_path,
        repository_root,
        _adapter_path,
    ) = _baseline_gate_fixture(tmp_path)
    monkeypatch.setattr(replay_module, "validate_manifest", lambda _path: fresh)
    preflight_path = completion_path.parent / "preflight_receipt.json"
    preflight = read_json(preflight_path)
    preflight["planned_process_logs"] = ["router.stdout.log"]
    write_json_atomic(preflight_path, preflight)
    completion = read_json(completion_path)
    completion["preflight_receipt_sha256"] = sha256_file(preflight_path)
    write_json_atomic(completion_path, completion)

    with pytest.raises(ValueError, match="bind every batch validation log"):
        validate_baseline_gate_artifacts(
            matrix_path=matrix_path,
            replay_path=replay_path,
            baseline_seed_path=seed_path,
            profile_path=profile_path,
            protocol_payload=protocol,
            completion_receipt_path=completion_path,
            repository_root=repository_root,
            ledger_payload=ledger,
        )


@pytest.mark.parametrize("tamper_kind", ["installed_adapter", "competing_compute"])
def test_baseline_gate_rejects_native_preflight_provenance_tampering(
    monkeypatch, tmp_path: Path, tamper_kind: str
) -> None:
    (
        matrix_path,
        replay_path,
        fresh,
        ledger,
        seed_path,
        profile_path,
        protocol,
        completion_path,
        repository_root,
        _adapter_path,
    ) = _baseline_gate_fixture(tmp_path)
    monkeypatch.setattr(replay_module, "validate_manifest", lambda _path: fresh)
    preflight_path = completion_path.parent / "preflight_receipt.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    expected_error = "installed adapter or runner hash mismatch"
    if tamper_kind == "installed_adapter":
        Path(preflight["installed_dynamic_adapter"]).write_text(
            "# installed bytes changed after preflight\n", encoding="utf-8"
        )
    else:
        preflight["reported_compute_processes"] = [
            {
                "gpu_uuid": "GPU-test-0",
                "process_id": 1234,
                "process_name": "competing-job.exe",
                "used_memory_mib": 256,
                "actionable_compute_allocation": True,
            }
        ]
        write_json_atomic(preflight_path, preflight)
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["preflight_receipt_sha256"] = sha256_file(preflight_path)
        write_json_atomic(completion_path, completion)
        expected_error = "competing compute use"

    with pytest.raises(ValueError, match=expected_error):
        validate_baseline_gate_artifacts(
            matrix_path=matrix_path,
            replay_path=replay_path,
            baseline_seed_path=seed_path,
            profile_path=profile_path,
            protocol_payload=protocol,
            completion_receipt_path=completion_path,
            repository_root=repository_root,
            ledger_payload=ledger,
        )


def _write_open_calibration_ledger(repository_root: Path) -> Path:
    ledger_path = repository_root / "configs/m8_calibration_ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        ledger_path,
        json.loads((ROOT / "configs/m8_calibration_ledger.json").read_text()),
    )
    return ledger_path


def _install_staged_replay_validator(
    monkeypatch,
    *,
    baseline_matrix_path: Path,
    baseline_replay: dict,
) -> None:
    def validate(path: Path | str) -> dict:
        matrix_path = Path(path).resolve()
        if matrix_path == baseline_matrix_path.resolve():
            return json.loads(json.dumps(baseline_replay))
        replay_path = matrix_path.parent / "replay_validation.json"
        return json.loads(replay_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(replay_module, "validate_manifest", validate)


def _write_metadata_only_candidate_artifacts(
    *,
    repository_root: Path,
    candidate0_ledger: dict,
    change_candidate_status: bool,
) -> tuple[Path, Path, Path]:
    candidate0_record = candidate0_ledger["development"]["candidates"][0]
    candidate0 = read_json(repository_root / candidate0_record["protocol_path"])
    candidate1 = json.loads(json.dumps(candidate0))
    source_profile = repository_root / candidate0["profiles"]["profile_1_fixed"][
        "profile_file"
    ]
    copied_profile = (
        repository_root
        / "profiles/candidate_1_metadata"
        / source_profile.name
    )
    copied_profile.parent.mkdir(parents=True)
    if change_candidate_status:
        profile_payload = read_json(source_profile)
        profile_payload["profile"]["candidate_status"] = "metadata_only_candidate"
        write_json_atomic(copied_profile, profile_payload)
    else:
        copied_profile.write_bytes(source_profile.read_bytes())
    candidate1["profiles"]["profile_1_fixed"]["profile_file"] = (
        copied_profile.relative_to(repository_root).as_posix()
    )
    candidate1["profiles"]["profile_1_fixed"]["profile_sha256"] = sha256_file(
        copied_profile
    )
    candidate1 = validate_protocol(candidate1)
    protocol_path = repository_root / "candidate_1_metadata_protocol.json"
    write_json_atomic(protocol_path, candidate1)

    matrix0 = read_json(repository_root / candidate0_record["matrix_path"])
    matrix_directory = repository_root / "development/candidate_1_metadata"
    matrix_directory.mkdir(parents=True)
    matrix_path = matrix_directory / "matrix.json"
    matrix0["candidate_manifest"] = "../../candidate_1_metadata_protocol.json"
    matrix0["protocol_sha256"] = canonical_sha256(candidate1)
    matrix0["profiles"]["profile_1_fixed"] = {
        "path": f"../../profiles/candidate_1_metadata/{copied_profile.name}",
        "sha256": sha256_file(copied_profile),
    }
    write_json_atomic(matrix_path, matrix0)

    replay = read_json(repository_root / candidate0_record["replay_path"])
    replay_path = matrix_directory / "replay_validation.json"
    replay["manifest"] = str(matrix_path.resolve())
    replay["manifest_sha256"] = sha256_file(matrix_path)
    write_json_atomic(replay_path, replay)
    return protocol_path, matrix_path, replay_path


@pytest.mark.parametrize("metadata_kind", ("profile_relocation", "candidate_status"))
def test_candidate_record_rejects_metadata_only_calibration(
    monkeypatch,
    tmp_path: Path,
    metadata_kind: str,
) -> None:
    (
        baseline_matrix,
        baseline_replay_path,
        fresh_baseline_replay,
        _closed_fixture_ledger,
        baseline_seeds,
        profile0,
        _fixture_finalized,
        completion_receipt,
        repository_root,
        _adapter_path,
    ) = _baseline_gate_fixture(tmp_path)
    ledger_path = _write_open_calibration_ledger(repository_root)
    _install_staged_replay_validator(
        monkeypatch,
        baseline_matrix_path=baseline_matrix,
        baseline_replay=fresh_baseline_replay,
    )
    record_native_baseline_gate(
        repository_root=repository_root,
        ledger_path=ledger_path,
        matrix_path=baseline_matrix,
        replay_path=baseline_replay_path,
        baseline_seed_path=baseline_seeds,
        profile_path=profile0,
        completion_receipt_path=completion_receipt,
    )
    candidate0_ledger = record_development_candidate(
        repository_root=repository_root,
        ledger_path=ledger_path,
        candidate_id="candidate_0",
        protocol_path=repository_root / "baseline_candidate_protocol.json",
        matrix_path=repository_root / "development/candidate_0/matrix.json",
        replay_path=repository_root
        / "development/candidate_0/replay_validation.json",
        development_seed_path=repository_root / "development_seeds.json",
    )
    protocol_path, matrix_path, replay_path = _write_metadata_only_candidate_artifacts(
        repository_root=repository_root,
        candidate0_ledger=candidate0_ledger,
        change_candidate_status=metadata_kind == "candidate_status",
    )
    unchanged_ledger = ledger_path.read_bytes()

    with pytest.raises(ValueError, match="runtime behavioral calibration value"):
        record_development_candidate(
            repository_root=repository_root,
            ledger_path=ledger_path,
            candidate_id="candidate_1",
            protocol_path=protocol_path,
            matrix_path=matrix_path,
            replay_path=replay_path,
            development_seed_path=repository_root / "development_seeds.json",
            reason="metadata relocation is not a behavioral calibration",
        )
    assert ledger_path.read_bytes() == unchanged_ledger


def test_supported_calibration_lifecycle_records_derives_and_closes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (
        baseline_matrix,
        baseline_replay_path,
        fresh_baseline_replay,
        closed_fixture_ledger,
        baseline_seeds,
        profile0,
        _fixture_finalized,
        completion_receipt,
        repository_root,
        _adapter_path,
    ) = _baseline_gate_fixture(tmp_path)
    ledger_path = _write_open_calibration_ledger(repository_root)
    _install_staged_replay_validator(
        monkeypatch,
        baseline_matrix_path=baseline_matrix,
        baseline_replay=fresh_baseline_replay,
    )

    baseline_ledger = record_native_baseline_gate(
        repository_root=repository_root,
        ledger_path=ledger_path,
        matrix_path=baseline_matrix,
        replay_path=baseline_replay_path,
        baseline_seed_path=baseline_seeds,
        profile_path=profile0,
        completion_receipt_path=completion_receipt,
    )
    assert baseline_ledger["status"] == "baseline_passed_development_open"
    assert baseline_ledger["baseline_gate"]["success_count"] == 18
    assert baseline_ledger["baseline_gate"]["trial_count"] == 20
    assert baseline_ledger["baseline_gate"]["protocol_sha256"] is None
    assert baseline_ledger["development"]["candidates"] == []
    assert read_json(ledger_path) == baseline_ledger

    candidate0_ledger = record_development_candidate(
        repository_root=repository_root,
        ledger_path=ledger_path,
        candidate_id="candidate_0",
        protocol_path=repository_root / "baseline_candidate_protocol.json",
        matrix_path=repository_root / "development/candidate_0/matrix.json",
        replay_path=repository_root
        / "development/candidate_0/replay_validation.json",
        development_seed_path=repository_root / "development_seeds.json",
    )
    assert candidate0_ledger["status"] == "development_open"
    assert [
        record["candidate_id"]
        for record in candidate0_ledger["development"]["candidates"]
    ] == ["candidate_0"]
    assert candidate0_ledger["bounded_calibration_changes"] == []

    finalized_candidate1, _ = _add_calibrated_candidate(
        repository_root=repository_root,
        ledger=json.loads(json.dumps(closed_fixture_ledger)),
        path="/protocol/controller/maximum_translation_per_step_m",
        value=0.012,
    )
    candidate1_ledger = record_development_candidate(
        repository_root=repository_root,
        ledger_path=ledger_path,
        candidate_id="candidate_1",
        protocol_path=repository_root / "candidate_1_protocol.json",
        matrix_path=repository_root / "development/candidate_1/matrix.json",
        replay_path=repository_root
        / "development/candidate_1/replay_validation.json",
        development_seed_path=repository_root / "development_seeds.json",
        reason="bounded translation step selected on development replay",
    )
    change = candidate1_ledger["bounded_calibration_changes"][0]
    changed_path = "/protocol/controller/maximum_translation_per_step_m"
    assert change["before"] == {changed_path: 0.01}
    assert change["after"] == {changed_path: 0.012}
    assert change["development_evidence"] == {
        "candidate_id": "candidate_0",
        "matrix_sha256": candidate1_ledger["development"]["candidates"][0][
            "matrix_sha256"
        ],
        "replay_sha256": candidate1_ledger["development"]["candidates"][0][
            "replay_sha256"
        ],
    }

    finalized_path = repository_root / "configs/m8_g0_frozen.json"
    write_json_atomic(finalized_path, finalized_candidate1)
    closed = close_calibration_ledger(
        repository_root=repository_root,
        ledger_path=ledger_path,
        selected_candidate_id="candidate_1",
        finalized_protocol_path=finalized_path,
        development_seed_path=repository_root / "development_seeds.json",
    )
    assert closed["status"] == "closed_before_holdout"
    assert closed["development"]["status"] == "complete"
    assert closed["development"]["selected_candidate_id"] == "candidate_1"
    assert closed["baseline_gate"]["selected_development_candidate_id"] == "candidate_1"
    assert closed["baseline_gate"]["bounded_calibration_change_count"] == 1
    assert closed["holdout_freeze_authorized"] is True
    validate_calibration_ledger(closed, require_closed=True)
    validate_development_calibration_lifecycle(
        repository_root=repository_root,
        baseline_candidate_protocol_path=repository_root
        / "baseline_candidate_protocol.json",
        finalized_protocol_payload=finalized_candidate1,
        ledger_payload=closed,
        expected_development_seed_path=repository_root / "development_seeds.json",
        verify_replay=True,
    )

    closed_bytes = ledger_path.read_bytes()
    with pytest.raises(ValueError, match="requires an open ledger"):
        close_calibration_ledger(
            repository_root=repository_root,
            ledger_path=ledger_path,
            selected_candidate_id="candidate_0",
            finalized_protocol_path=finalized_path,
            development_seed_path=repository_root / "development_seeds.json",
        )
    assert ledger_path.read_bytes() == closed_bytes


def test_failed_native_baseline_record_preserves_open_ledger(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (
        baseline_matrix,
        baseline_replay_path,
        fresh_baseline_replay,
        _closed_fixture_ledger,
        baseline_seeds,
        profile0,
        _fixture_finalized,
        completion_receipt,
        repository_root,
        _adapter_path,
    ) = _baseline_gate_fixture(tmp_path)
    ledger_path = _write_open_calibration_ledger(repository_root)
    failed_replay = json.loads(json.dumps(fresh_baseline_replay))
    failed_replay["audits"][17]["recomputed_metrics"]["task_success"] = False
    write_json_atomic(baseline_replay_path, failed_replay)
    completion = read_json(completion_receipt)
    completion["replay_validation_sha256"] = sha256_file(baseline_replay_path)
    write_json_atomic(completion_receipt, completion)
    _install_staged_replay_validator(
        monkeypatch,
        baseline_matrix_path=baseline_matrix,
        baseline_replay=failed_replay,
    )
    original_bytes = ledger_path.read_bytes()

    with pytest.raises(ValueError, match="at least 18/20"):
        record_native_baseline_gate(
            repository_root=repository_root,
            ledger_path=ledger_path,
            matrix_path=baseline_matrix,
            replay_path=baseline_replay_path,
            baseline_seed_path=baseline_seeds,
            profile_path=profile0,
            completion_receipt_path=completion_receipt,
        )

    assert ledger_path.read_bytes() == original_bytes


def _write_candidate_matrix_repository(
    repository_root: Path,
) -> tuple[Path, dict[str, Path], Path]:
    profile_directory = repository_root / "profiles"
    profile_directory.mkdir(parents=True)
    profile_paths: dict[str, Path] = {}
    for profile_id in (
        "profile_0_sanity",
        "profile_1_fixed",
        "profile_2_faults",
    ):
        source = (
            ROOT
            / "ros2_ws/src/action_stream_benchmark/config"
            / f"m8_{profile_id}.json"
        )
        destination = profile_directory / source.name
        write_json_atomic(destination, read_json(source))
        profile_paths[profile_id] = destination
    protocol = read_json(ROOT / "configs/m8_g0.json")
    for profile_id, record in protocol["profiles"].items():
        record["profile_file"] = profile_paths[profile_id].relative_to(
            repository_root
        ).as_posix()
        record["profile_sha256"] = sha256_file(profile_paths[profile_id])
    protocol_path = repository_root / "configs/candidate.json"
    protocol_path.parent.mkdir(parents=True)
    write_json_atomic(protocol_path, protocol)
    seed_path = repository_root / "configs/development_seed.json"
    write_json_atomic(
        seed_path,
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "split": "development",
            "seeds": [2026081200],
        },
    )
    return protocol_path, profile_paths, seed_path


def test_candidate_matrix_scenarios_use_exact_protocol_task_contract(
    tmp_path: Path,
) -> None:
    protocol_path, profile_paths, seed_path = _write_candidate_matrix_repository(
        tmp_path
    )
    protocol = read_json(protocol_path)
    protocol["task"].update(
        {
            "object_x_range_m": [0.50, 0.52],
            "object_y_range_m": [0.10, 0.12],
            "cube_side_m": 0.06,
            "zone_a_xy_m": [0.55, -0.25],
            "zone_b_xy_m": [0.56, 0.25],
            "switch_steps": [80],
        }
    )
    write_json_atomic(protocol_path, protocol)
    output_root = tmp_path / "outputs/development/raw"
    matrix_path = tmp_path / "outputs/development/matrix.json"

    matrix = build_matrix_manifest(
        repository_root=tmp_path,
        freeze_manifest_path=None,
        candidate_protocol_path=protocol_path,
        seed_path=seed_path,
        profile_paths=[profile_paths["profile_1_fixed"]],
        output_root=output_root,
        manifest_path=matrix_path,
        split="development",
        request_count=32,
        strategy_filter=["aligned_async"],
    )

    scenario = read_json(
        matrix_path.parent / matrix["episodes"][0]["scenario_file"]
    )
    assert 0.50 <= scenario["object_position_xyz"][0] <= 0.52
    assert 0.10 <= scenario["object_position_xyz"][1] <= 0.12
    assert scenario["object_position_xyz"][2] == pytest.approx(0.03)
    assert scenario["zone_a_xyz"] == pytest.approx([0.55, -0.25, 0.03])
    assert scenario["zone_b_xyz"] == pytest.approx([0.56, 0.25, 0.03])
    assert scenario["switch_step"] == 80


def test_candidate_matrix_preflights_cross_profile_latency_before_outputs(
    tmp_path: Path,
) -> None:
    protocol_path, profile_paths, seed_path = _write_candidate_matrix_repository(
        tmp_path
    )
    profile1 = read_json(profile_paths["profile_1_fixed"])
    profile1["profile"]["base_latency_ms"] = 900
    write_json_atomic(profile_paths["profile_1_fixed"], profile1)
    protocol = read_json(protocol_path)
    protocol["profiles"]["profile_1_fixed"]["profile_sha256"] = sha256_file(
        profile_paths["profile_1_fixed"]
    )
    write_json_atomic(protocol_path, protocol)
    output_root = tmp_path / "outputs/invalid/raw"

    with pytest.raises(ValueError, match="Profile 2 must reuse"):
        build_matrix_manifest(
            repository_root=tmp_path,
            freeze_manifest_path=None,
            candidate_protocol_path=protocol_path,
            seed_path=seed_path,
            profile_paths=[profile_paths["profile_1_fixed"]],
            output_root=output_root,
            manifest_path=tmp_path / "outputs/invalid/matrix.json",
            split="development",
            request_count=32,
        )

    assert not output_root.exists()
