from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import action_stream_benchmark.m8_protocol as protocol_module
import action_stream_benchmark.m8_report as report_module
from action_stream_benchmark.m8_archive import build_archive
from action_stream_benchmark.m8_cli import run
from action_stream_benchmark.m8_protocol import (
    PROFILE_STRATEGIES,
    ros_source_manifest,
    sha256_file,
)
from action_stream_benchmark.m8_report import (
    generate_technical_report,
    write_unavailable_report,
)
from action_stream_benchmark.schema import read_json, write_json_atomic


TEST_PIXI_TOML_BYTES = b"[workspace]\nname = 'report-test'\n"
TEST_PIXI_LOCK_BYTES = b"version: 7\n"


def _test_workspace_specs() -> dict[str, dict[str, int | str]]:
    def record(data: bytes) -> dict[str, int | str]:
        crlf = data.replace(b"\n", b"\r\n")
        return {
            "canonical_lf_size_bytes": len(data),
            "canonical_lf_sha256": hashlib.sha256(data).hexdigest(),
            "exact_crlf_size_bytes": len(crlf),
            "exact_crlf_sha256": hashlib.sha256(crlf).hexdigest(),
        }

    return {
        "pixi.toml": record(TEST_PIXI_TOML_BYTES),
        "pixi.lock": record(TEST_PIXI_LOCK_BYTES),
    }


def _formal_external_environment_fixture(receipt_directory: Path) -> Path:
    manifest_evidence = receipt_directory / "isaac_workspace.pixi.toml"
    lock_evidence = receipt_directory / "isaac_workspace.pixi.lock"
    manifest_evidence.write_bytes(TEST_PIXI_TOML_BYTES)
    lock_evidence.write_bytes(TEST_PIXI_LOCK_BYTES)
    workspace_files = {}
    for filename, evidence in (
        ("pixi.toml", manifest_evidence),
        ("pixi.lock", lock_evidence),
    ):
        spec = _test_workspace_specs()[filename]
        workspace_files[filename] = {
            "source_path": (
                f"/deleted-rental/IsaacSim-ros_workspaces/jazzy_ws/{filename}"
            ),
            "evidence": evidence.name,
            "size_bytes": evidence.stat().st_size,
            "sha256": sha256_file(evidence),
            "canonical_lf_size_bytes": spec["canonical_lf_size_bytes"],
            "canonical_lf_sha256": spec["canonical_lf_sha256"],
            "byte_form": "lf",
        }
    environment = receipt_directory / "external_environment.json"
    write_json_atomic(
        environment,
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "evidence_kind": "native_external_environment",
            "created_utc": "2026-08-04T00:00:00Z",
            "repository_root": "/deleted-rental/IsaacSim-ros_workspaces",
            "repository_url": protocol_module.ISAAC_WORKSPACE_REPOSITORY_URL,
            "workspace_commit": protocol_module.ISAAC_WORKSPACE_COMMIT,
            "workspace_relative_path": protocol_module.ISAAC_WORKSPACE_RELATIVE_PATH,
            "tracked_manifest_lock_clean": True,
            "workspace_files": workspace_files,
            "pixi": {
                "executable": "/deleted-rental/bin/pixi",
                "executable_size_bytes": (
                    protocol_module.OFFICIAL_LINUX_PIXI_EXECUTABLE_SIZE_BYTES
                ),
                "executable_sha256": (
                    protocol_module.OFFICIAL_LINUX_PIXI_EXECUTABLE_SHA256
                ),
                "version": protocol_module.OFFICIAL_LINUX_PIXI_VERSION,
                "version_output": protocol_module.OFFICIAL_LINUX_PIXI_VERSION_OUTPUT,
            },
            "runtime": {
                "platform_system": "Linux",
                "platform_machine": "x86_64",
                "python_version": protocol_module.EXPECTED_RUNTIME_VERSIONS[
                    "python_version"
                ],
                "python_executable": "/deleted-rental/.pixi/envs/default/bin/python",
                "sys_prefix": "/deleted-rental/.pixi/envs/default",
                "packages": {
                    name: protocol_module.EXPECTED_RUNTIME_VERSIONS[name]
                    for name in (
                        "isaacsim",
                        "isaacsim-app",
                        "isaacsim-core",
                        "isaacsim-robot",
                        "isaacsim-ros2",
                        "rclpy",
                        "rosgraph-msgs",
                    )
                },
                "ros_distribution": protocol_module.EXPECTED_RUNTIME_VERSIONS[
                    "ros_distribution"
                ],
                "rmw_implementation": protocol_module.EXPECTED_RUNTIME_VERSIONS[
                    "rmw_implementation"
                ],
                "rmw_zenoh_cpp": protocol_module.EXPECTED_RUNTIME_VERSIONS[
                    "rmw_zenoh_cpp"
                ],
            },
        },
    )
    return environment


def _formal_gpu_snapshot(phase: str) -> dict:
    return {
        "phase": phase,
        "captured_utc": "2026-08-04T00:00:00Z",
        "selected_gpu_index": 0,
        "gpu_inventory": [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 5090",
                "uuid": "GPU-report-test-0",
                "driver_version": "580.105.08",
                "memory_total_mib": 32607,
                "memory_used_mib": 0,
                "utilization_gpu_percent": 0,
            }
        ],
        "reported_compute_processes": [],
        "actionable_compute_processes": [],
        "blocking_compute_processes": [],
        "unknown_memory_compute_processes": [],
        "occupied_gpus": [],
        "passed": True,
    }


def test_unavailable_path_never_synthesizes_native_metrics(tmp_path: Path) -> None:
    output = tmp_path / "unavailable.json"
    report = write_unavailable_report(
        output,
        stage="native_isaac_preflight",
        reason="Isaac Sim runtime not installed",
    )
    assert report["classification"] == "NO-GO"
    assert report["headline_evidence_generated"] is False
    assert report["metrics"] is None
    assert report["no_test_plant_substitution"] is True
    assert read_json(output) == report


def _write(root: Path, relative: str, payload: dict) -> Path:
    path = root / relative
    write_json_atomic(path, payload)
    return path


def _file_record(path: Path, *, base: Path, format_name: str | None = None) -> dict:
    record = {
        "path": path.relative_to(base).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if format_name is not None:
        record["format"] = format_name
    return record


def _report_fixture(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, Path]:
    monkeypatch.setattr(
        protocol_module, "OFFICIAL_WORKSPACE_FILE_SPECS", _test_workspace_specs()
    )
    (root / ".git").mkdir(parents=True)
    protocol = _write(
        root,
        "configs/m8_g0.json",
        {
            "runtime": {
                "control_frequency_hz": 20,
                "chunk_horizon": 30,
                "request_interval_steps": 10,
                "action_representation": "absolute Cartesian pose",
            },
            "task": {
                "task_id": "dynamic_target_pick_place_v1",
                "primary_disturbance": "destination_switch",
                "placement_tolerance_m": 0.04,
                "stable_placement_steps": 20,
            },
            "controller": {
                "type": "deterministic_observation_conditioned_cartesian_waypoint",
            },
        },
    )
    profile_paths = {}
    for profile_id in PROFILE_STRATEGIES:
        profile_paths[profile_id] = _write(
            root,
            f"configs/{profile_id}.json",
            {"profile_id": profile_id},
        )

    freeze_sha256 = "f" * 64
    freeze = _write(
        root,
        "outputs/freeze.json",
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "freeze_sha256": freeze_sha256,
            "inputs": [
                {
                    "role": "protocol",
                    "path": protocol.relative_to(root).as_posix(),
                    "sha256": sha256_file(protocol),
                },
                *[
                    {
                        "role": f"profile:{profile_id}",
                        "path": path.relative_to(root).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for profile_id, path in profile_paths.items()
                ],
            ],
        },
    )
    seed_sha256 = "e" * 64
    batch = _write(
        root,
        "outputs/batch_profile_0_sanity_sync_hold.json",
        {"schema_version": 1, "milestone": "M8-G0"},
    )
    matrix = _write(
        root,
        "outputs/matrix.json",
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "split": "frozen_holdout",
            "headline_eligible": True,
            "freeze_manifest": freeze.relative_to(root).as_posix(),
            "freeze_sha256": freeze_sha256,
            "seed_file_sha256": seed_sha256,
            "persistent_batch_count": 1,
            "batch_manifests": [
                {"path": batch.relative_to(root / "outputs").as_posix()}
            ],
        },
    )

    metric_summary = {"kind": "numeric", "count": 1, "null_count": 0, "mean": 1.0, "median": 1.0}
    unavailable_summary = {
        "kind": "unavailable",
        "count": 1,
        "observed_count": 0,
        "null_count": 1,
    }
    strategy_row = {
        "successes": 1,
        "trials": 1,
        "success_rate": 1.0,
        "median_completion_steps": 100.0,
        "mean_completion_steps": 100.0,
        "median_wall_clock_seconds": 5.0,
        "mean_wall_clock_seconds": 5.0,
        "median_penalized_recovery_latency_steps": 10.0,
        "mean_obsolete_destination_command_steps": 1.0,
        "mean_hold_control_seconds": 0.5,
        "descriptive_metrics": {
            "mean_inference_latency_ms": dict(unavailable_summary),
            "p95_inference_latency_ms": dict(unavailable_summary),
            "mean_action_age_steps": dict(metric_summary),
            "p95_action_age_steps": dict(metric_summary),
        },
    }
    profiles = {}
    for profile_id, methods in PROFILE_STRATEGIES.items():
        profile = {
            "paired_trial_count": 1,
            "strategies": {method: dict(strategy_row) for method in methods},
        }
        if len(methods) == 3:
            profile.update(
                {
                    "aligned_minus_naive_percentage_points": 0.0,
                    "paired_bootstrap_95_ci_percentage_points": [0.0, 0.0],
                    "raw_paired_outcomes": [{"seed": 1}],
                }
            )
        profiles[profile_id] = profile

    replay = _write(
        root,
        "outputs/replay.json",
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "split": "frozen_holdout",
            "passed": True,
            "manifest": matrix.relative_to(root).as_posix(),
            "manifest_sha256": sha256_file(matrix),
            "episode_audits_passed": 7,
            "episode_count": 7,
            "fairness_passed": True,
            "freeze_validation_passed": True,
            "provenance_validation_passed": True,
            "seed_validation_passed": True,
            "profile_validation_passed": True,
            "seed_count": 1,
            "seed_file_sha256": seed_sha256,
            "freeze_audit": {"manifest": {"freeze_sha256": freeze_sha256}},
            "audits": [],
        },
    )
    analysis = _write(
        root,
        "outputs/analysis.json",
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "evidence_class": "ros_cpp_isaac_sim",
            "classification": "NO-GO",
            "episode_count": 7,
            "manifest": matrix.relative_to(root).as_posix(),
            "manifest_sha256": sha256_file(matrix),
            "replay": replay.relative_to(root).as_posix(),
            "replay_sha256": sha256_file(replay),
            "freeze_manifest": freeze.relative_to(root).as_posix(),
            "freeze_sha256": freeze_sha256,
            "holdout_seed_file_sha256": seed_sha256,
            "profiles": profiles,
            "primary_go_gate": {
                "conditions": {"paired_fairness_passes": True},
                "aligned_forbidden_execution_count": 0,
                "passed": False,
            },
            "strong_go_gate": {"eligible": False, "passed": False},
            "bootstrap": {"resamples": 20_000, "confidence": 0.95},
        },
    )

    starting = _write(
        root,
        "outputs/starting.json",
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "repository": {
                "next_unused_milestone": "M8-G0",
                "branch": "main",
                "head": "1" * 40,
                "worktree_clean": True,
                "upstream": "origin/main",
                "ahead": 0,
                "behind": 0,
            },
            "windows": {
                "edition": "Windows 11",
                "build": "1",
                "python": "3.12",
                "pytorch": "2.6",
                "cuda_toolkit": "12.4",
            },
            "gpu": {"name": "NVIDIA GeForce RTX 4090", "driver": "1"},
            "docker": {"version": "1"},
            "isaac_ros_environment": {
                "isaac_sim": "6.0",
                "ros_distribution": "jazzy",
                "pixi": "0.75",
            },
            "canonical_components": {"executor": "src/executor.cpp"},
        },
    )
    differential = _write(
        root,
        "outputs/differential.json",
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "common_domain_behavioral_equivalence": True,
            "headline_evaluation_unblocked": True,
            "unresolved_unintended_discrepancies": [],
            "migration_discrepancy_history": [
                {
                    "discrepancy_id": "D1",
                    "fix_summary": "normalized termination",
                    "status": "repaired",
                }
            ],
            "fixture": {"common_event_count": 10},
        },
    )
    cpu = _write(
        root,
        "outputs/cpu.json",
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "repository_python": {"passed": 10, "failed": 0, "skipped": 1},
            "native_windows_ros": {
                "build": {"result": "passed"},
                "test": {"leaf_test_cases": 10, "errors": 0, "failures": 0},
            },
            "phase_0_replay": {"headline_evaluation_unblocked": True},
        },
    )
    runtime_source = root / "src/runtime.cpp"
    runtime_source.parent.mkdir(parents=True)
    runtime_source.write_text("runtime", encoding="utf-8")
    native_runtime = _write(
        root,
        "outputs/native_runtime.json",
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "native_episode_executed": False,
            "verified_contracts": [{"contract": "runtime"}],
            "source_sha256": {
                runtime_source.relative_to(root).as_posix(): sha256_file(runtime_source)
            },
        },
    )
    ledger = _write(
        root,
        "outputs/ledger.json",
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "bounded_calibration_changes": [],
            "baseline_gate": {
                "success_count": 19,
                "trial_count": 20,
                "observed_success_rate": 0.95,
                "replay_validated": True,
                "required_success_rate": 0.9,
            },
            "development": {
                "selected_candidate_id": "candidate_0",
                "candidates": [{"candidate_id": "candidate_0", "seed_count": 1}],
            },
        },
    )
    adapter_source = (
        root
        / "ros2_ws/src/action_stream_isaac/action_stream_isaac/dynamic_isaac_adapter.py"
    )
    adapter_source.parent.mkdir(parents=True)
    adapter_source.write_text("# portable adapter source\n", encoding="utf-8")
    runner_source = root / "scripts/m8_run_isaac.sh"
    runner_source.parent.mkdir(parents=True)
    runner_source.write_text("# portable runner source\n", encoding="utf-8")
    runner_support_source = root / "scripts/m8_linux_runner_support.py"
    runner_support_source.write_text(
        "# portable runner support source\n", encoding="utf-8"
    )
    receipt_directory = root / "outputs/native_receipt"
    receipt_directory.mkdir(parents=True)
    adapter_evidence = receipt_directory / "installed_dynamic_adapter.py"
    adapter_evidence.write_bytes(adapter_source.read_bytes())
    executor_evidence = receipt_directory / "action_stream_executor_node"
    executor_evidence.write_bytes(b"compiled executor evidence\n")
    runner_evidence = receipt_directory / "m8_run_isaac.runner.sh"
    runner_evidence.write_bytes(runner_source.read_bytes())
    runner_support_evidence = receipt_directory / "m8_linux_runner_support.py"
    runner_support_evidence.write_bytes(runner_support_source.read_bytes())
    external_environment = _formal_external_environment_fixture(receipt_directory)
    initial_gpu = _formal_gpu_snapshot("initial_pre_build")
    post_gpu = _formal_gpu_snapshot("post_build_pre_launch")
    selected_gpu_identity = {
        key: post_gpu["gpu_inventory"][0][key]
        for key in ("index", "name", "uuid", "driver_version", "memory_total_mib")
    }
    source = ros_source_manifest(root)
    validation_log_name = "batch_profile_0_sanity_sync_hold.validate_only.log"
    freeze_validation_log_name = "freeze_validate.log"
    process_log_names = [
        "colcon_build.log",
        "replay_validate.log",
        "analysis_figures.log",
        "router.stdout.log",
        "router.stderr.log",
        "batch_profile_0_sanity_sync_hold.executor.stdout.log",
        "batch_profile_0_sanity_sync_hold.executor.stderr.log",
        "batch_profile_0_sanity_sync_hold.adapter.stdout.log",
        "batch_profile_0_sanity_sync_hold.adapter.stderr.log",
        validation_log_name,
        freeze_validation_log_name,
    ]
    for name in process_log_names:
        (receipt_directory / name).write_text("runner log\n", encoding="utf-8")
    process_log_archive = receipt_directory / "process_logs.tar.gz"
    process_log_manifest = receipt_directory / "process_logs.manifest.json"
    build_archive(
        root=receipt_directory,
        members=[receipt_directory / name for name in process_log_names],
        archive_path=process_log_archive,
        manifest_path=process_log_manifest,
    )
    preflight = _write(
        root,
        "outputs/native_receipt/preflight.json",
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "receipt_kind": "native_preflight",
            "operator_authorized_native_gpu_run": True,
            "gpu_memory_refusal_threshold_mib": 4096,
            "gpu_inventory": post_gpu["gpu_inventory"],
            "reported_compute_processes": [],
            "preexisting_compute_processes": [],
            "initial_gpu_preflight": initial_gpu,
            "post_build_gpu_preflight": post_gpu,
            "selected_gpu_index": selected_gpu_identity["index"],
            "selected_gpu_uuid": selected_gpu_identity["uuid"],
            "selected_gpu_identity": selected_gpu_identity,
            "source_root": "ros2_ws/src",
            "source_file_count": source["source_file_count"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "suite_manifest": matrix.relative_to(root).as_posix(),
            "suite_manifest_sha256": sha256_file(matrix),
            "current_source_batch_validation_passed": True,
            "current_source_batch_validation_count": 1,
            "current_source_batch_validation_logs": [validation_log_name],
            "frozen_live_inputs_validated": True,
            "planned_process_logs": process_log_names,
            "external_environment_evidence": external_environment.name,
            "external_environment_evidence_sha256": sha256_file(
                external_environment
            ),
            "installed_dynamic_adapter": (
                "/deleted-rental/install/action_stream_isaac/"
                "action_stream_isaac/dynamic_isaac_adapter.py"
            ),
            "installed_dynamic_adapter_evidence": adapter_evidence.name,
            "installed_dynamic_adapter_evidence_sha256": sha256_file(
                adapter_evidence
            ),
            "installed_dynamic_adapter_sha256": sha256_file(adapter_source),
            "executor_path": "/deleted-rental/install/action_stream_executor_node",
            "executor_evidence": executor_evidence.name,
            "executor_evidence_sha256": sha256_file(executor_evidence),
            "executor_sha256": sha256_file(executor_evidence),
            "router_path": "/deleted-rental/.pixi/envs/default/bin/rmw_zenohd",
            "router_sha256": "a" * 64,
            "runner": "/deleted-rental/checkout/scripts/m8_run_isaac.sh",
            "runner_source": "scripts/m8_run_isaac.sh",
            "runner_evidence": runner_evidence.name,
            "runner_evidence_sha256": sha256_file(runner_evidence),
            "runner_sha256": sha256_file(runner_source),
            "runner_support_source": "scripts/m8_linux_runner_support.py",
            "runner_support_evidence": runner_support_evidence.name,
            "runner_support_evidence_sha256": sha256_file(
                runner_support_evidence
            ),
            "runner_support_sha256": sha256_file(runner_support_source),
        },
    )
    completion = _write(
        root,
        "outputs/completion.json",
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "receipt_kind": "native_completion",
            "status": "complete",
            "preflight_receipt": preflight.relative_to(root / "outputs").as_posix(),
            "preflight_receipt_sha256": sha256_file(preflight),
            "analysis_sha256": sha256_file(analysis),
            "analysis": analysis.relative_to(root / "outputs").as_posix(),
            "replay_validation": replay.relative_to(root / "outputs").as_posix(),
            "replay_validation_sha256": sha256_file(replay),
            "process_log_archive": process_log_archive.relative_to(
                root / "outputs"
            ).as_posix(),
            "process_log_archive_sha256": sha256_file(process_log_archive),
            "process_log_manifest": process_log_manifest.relative_to(
                root / "outputs"
            ).as_posix(),
            "process_log_manifest_sha256": sha256_file(process_log_manifest),
            "process_log_member_count": len(process_log_names),
        },
    )

    figure_directory = root / "outputs/figures"
    figure_directory.mkdir(parents=True)
    figure_rows = []
    for figure_id in report_module.REQUIRED_FIGURE_IDS:
        files = []
        for suffix in ("pdf", "png"):
            path = figure_directory / f"{figure_id}.{suffix}"
            path.write_bytes(f"{figure_id}-{suffix}".encode())
            files.append(_file_record(path, base=figure_directory, format_name=suffix))
        figure_rows.append({"figure_id": figure_id, "files": files})
    latex = figure_directory / "m8_figures.tex"
    latex.write_text("% figures", encoding="utf-8")
    figure_manifest = _write(
        root,
        "outputs/figures/figure_manifest.json",
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "evidence_class": "ros_cpp_isaac_sim",
            "source_analysis_sha256": sha256_file(analysis),
            "classification": "NO-GO",
            "figures": figure_rows,
            "latex_includes": _file_record(latex, base=figure_directory),
        },
    )

    monkeypatch.setattr(report_module, "validate_calibration_ledger", lambda *_a, **_k: {})
    monkeypatch.setattr(
        report_module,
        "validate_freeze_manifest",
        lambda *_a, **_k: {"passed": True, "errors": [], "manifest": read_json(freeze)},
    )
    return {
        "repository_root": root,
        "analysis_path": analysis,
        "replay_path": replay,
        "starting_audit_path": starting,
        "differential_report_path": differential,
        "calibration_ledger_path": ledger,
        "freeze_manifest_path": freeze,
        "cpu_validation_path": cpu,
        "native_runtime_audit_path": native_runtime,
        "completion_receipt_path": completion,
        "figure_manifest_path": figure_manifest,
        "output_path": root / "outputs/report.md",
    }


def test_technical_report_binds_evidence_and_reports_unavailable_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _report_fixture(monkeypatch, tmp_path / "repo")
    report = generate_technical_report(**paths)

    assert "## Starting commit and environment" in report
    assert "## Phase 0: M4-to-M7 differential replay" in report
    assert "## CPU/native validation and simulator baseline gate" in report
    assert "## Frozen calibration, profiles, and seeds" in report
    assert "## Exact frozen-holdout results" in report
    assert "## Exact reproduction commands" in report
    assert "not real-robot validation" in report
    assert "unavailable" in report
    assert "Holdout Matrix" in report
    assert "Starting audit GPU only (not the native holdout runtime)" in report
    assert "NVIDIA GeForce RTX 4090" in report
    assert "NVIDIA GeForce RTX 5090" in report
    assert "GPU-report-test-0" in report
    assert protocol_module.ISAAC_WORKSPACE_COMMIT in report
    assert "Pixi 0.75.0" in report
    assert "bash scripts/m8_run_isaac.sh" in report

    cli_output = paths["repository_root"] / "outputs/report-from-cli.md"
    assert (
        run(
            [
                "report",
                "--repository-root",
                str(paths["repository_root"]),
                "--analysis",
                str(paths["analysis_path"]),
                "--replay",
                str(paths["replay_path"]),
                "--starting-audit",
                str(paths["starting_audit_path"]),
                "--differential-report",
                str(paths["differential_report_path"]),
                "--calibration-ledger",
                str(paths["calibration_ledger_path"]),
                "--freeze-manifest",
                str(paths["freeze_manifest_path"]),
                "--cpu-validation",
                str(paths["cpu_validation_path"]),
                "--native-runtime-audit",
                str(paths["native_runtime_audit_path"]),
                "--completion-receipt",
                str(paths["completion_receipt_path"]),
                "--figure-manifest",
                str(paths["figure_manifest_path"]),
                "--output",
                str(cli_output),
            ]
        )
        == 0
    )
    assert cli_output.read_text(encoding="utf-8") == report


def test_technical_report_fails_closed_on_replay_or_matrix_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _report_fixture(monkeypatch, tmp_path / "repo")
    replay = read_json(paths["replay_path"])
    replay["episode_count"] = 8
    write_json_atomic(paths["replay_path"], replay)
    with pytest.raises(ValueError, match="cryptographically bound"):
        generate_technical_report(**paths)

    paths = _report_fixture(monkeypatch, tmp_path / "repo2")
    matrix = read_json(paths["repository_root"] / "outputs/matrix.json")
    matrix["headline_eligible"] = False
    write_json_atomic(paths["repository_root"] / "outputs/matrix.json", matrix)
    with pytest.raises(ValueError, match="cryptographically bound"):
        generate_technical_report(**paths)


def test_technical_report_fails_closed_on_portable_runner_evidence_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _report_fixture(monkeypatch, tmp_path / "repo")
    evidence = (
        paths["repository_root"]
        / "outputs/native_receipt/m8_run_isaac.runner.sh"
    )
    evidence.write_text("# tampered after native run\n", encoding="utf-8")

    with pytest.raises(ValueError, match="portable runner evidence mismatch"):
        generate_technical_report(**paths)


def test_technical_report_rejects_unrelated_native_completion_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _report_fixture(monkeypatch, tmp_path / "repo")
    root = paths["repository_root"]
    matrix_path = root / "outputs/matrix.json"
    unrelated_matrix = root / "outputs/unrelated_matrix.json"
    unrelated_matrix.write_bytes(matrix_path.read_bytes())

    preflight_path = root / "outputs/native_receipt/preflight.json"
    preflight = read_json(preflight_path)
    preflight["suite_manifest"] = unrelated_matrix.relative_to(root).as_posix()
    preflight["suite_manifest_sha256"] = sha256_file(unrelated_matrix)
    write_json_atomic(preflight_path, preflight)

    completion = read_json(paths["completion_receipt_path"])
    completion["preflight_receipt_sha256"] = sha256_file(preflight_path)
    write_json_atomic(paths["completion_receipt_path"], completion)

    with pytest.raises(ValueError, match="does not bind the baseline matrix"):
        generate_technical_report(**paths)


def test_technical_report_rejects_unrelated_analysis_path_with_same_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _report_fixture(monkeypatch, tmp_path / "repo")
    root = paths["repository_root"]
    unrelated_analysis = root / "outputs/unrelated_analysis.json"
    unrelated_analysis.write_bytes(paths["analysis_path"].read_bytes())
    completion = read_json(paths["completion_receipt_path"])
    completion["analysis"] = unrelated_analysis.relative_to(
        paths["completion_receipt_path"].parent
    ).as_posix()
    completion["analysis_sha256"] = sha256_file(unrelated_analysis)
    write_json_atomic(paths["completion_receipt_path"], completion)

    with pytest.raises(ValueError, match="does not bind this analysis/replay pair"):
        generate_technical_report(**paths)
