from __future__ import annotations

from pathlib import Path

import pytest

import action_stream_benchmark.m8_report as report_module
from action_stream_benchmark.m8_cli import run
from action_stream_benchmark.m8_protocol import PROFILE_STRATEGIES, sha256_file
from action_stream_benchmark.m8_report import (
    generate_technical_report,
    write_unavailable_report,
)
from action_stream_benchmark.schema import read_json, write_json_atomic


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
            "gpu": {"name": "GPU", "driver": "1"},
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
    completion = _write(
        root,
        "outputs/completion.json",
        {
            "schema_version": 1,
            "milestone": "M8-G0",
            "receipt_kind": "native_completion",
            "status": "complete",
            "analysis_sha256": sha256_file(analysis),
            "replay_validation_sha256": sha256_file(replay),
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
