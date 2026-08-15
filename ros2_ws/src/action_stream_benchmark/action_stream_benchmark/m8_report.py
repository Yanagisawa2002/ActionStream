"""Honest, hash-bound M8 result and unavailable/not-run reporting paths."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from .m8_archive import validate_archive, validate_archive_matrix_binding
from .m8_protocol import (
    M8_MILESTONE,
    M8_SCHEMA_VERSION,
    NATIVE_ISAAC_EVIDENCE_CLASS,
    PROFILE_STRATEGIES,
    sha256_file,
    validate_calibration_ledger,
    validate_freeze_manifest,
    validate_native_completion_receipt,
)
from .schema import read_json, write_json_atomic


REQUIRED_FIGURE_IDS = {
    "profile_1_success",
    "profile_1_recovery_latency",
    "profile_1_obsolete_commands",
}


def write_unavailable_report(
    output_path: Path | str,
    *,
    reason: str,
    stage: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not reason.strip() or not stage.strip():
        raise ValueError("unavailable report requires a concrete stage and reason")
    report = {
        "schema_version": M8_SCHEMA_VERSION,
        "milestone": M8_MILESTONE,
        "native_results_status": "unavailable_not_run",
        "native_execution_completed": False,
        "headline_evidence_generated": False,
        "headline_eligible": False,
        "evidence_class_required": NATIVE_ISAAC_EVIDENCE_CLASS,
        "classification": "NO-GO",
        "stage": stage,
        "reason": reason,
        "metrics": None,
        "no_test_plant_substitution": True,
        "details": dict(details or {}),
    }
    write_json_atomic(output_path, report)
    return report


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _require_m8(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("milestone") != M8_MILESTONE:
        raise ValueError(f"{label} is not an M8-G0 artifact")


def _portable_reference(path: Path | str, *, repository_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"report artifact is outside the repository: {resolved}") from exc


def _artifact_record(path: Path | str, *, repository_root: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": _portable_reference(resolved, repository_root=repository_root),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _resolve_repository_reference(value: object, *, repository_root: Path, label: str) -> Path:
    text = str(value)
    windows = PureWindowsPath(text)
    pure = PurePosixPath(text.replace("\\", "/"))
    if (
        not text
        or Path(text).is_absolute()
        or windows.is_absolute()
        or windows.drive
        or pure.is_absolute()
        or ".." in pure.parts
    ):
        raise ValueError(f"{label} must be a repository-relative path")
    resolved = (repository_root / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository") from exc
    return resolved


def _resolve_receipt_reference(
    value: object,
    *,
    receipt_path: Path,
    repository_root: Path,
    label: str,
) -> Path:
    text = str(value)
    windows = PureWindowsPath(text)
    pure = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or Path(text).is_absolute()
        or windows.is_absolute()
        or windows.drive
        or pure.is_absolute()
    ):
        raise ValueError(f"{label} must be a portable receipt-relative path")
    resolved = (receipt_path.parent / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _validate_file_record(
    record: Mapping[str, Any],
    *,
    base_directory: Path,
    label: str,
) -> Path:
    text = str(record.get("path", ""))
    pure = PurePosixPath(text.replace("\\", "/"))
    windows = PureWindowsPath(text)
    if (
        not text
        or Path(text).is_absolute()
        or pure.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in pure.parts
    ):
        raise ValueError(f"{label} path must be portable and relative")
    resolved = (base_directory / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(base_directory.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} path escapes its artifact directory") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    expected_hash = record.get("sha256")
    if not _is_sha256(expected_hash) or sha256_file(resolved) != expected_hash:
        raise ValueError(f"{label} SHA-256 mismatch")
    if "size_bytes" in record and resolved.stat().st_size != record.get("size_bytes"):
        raise ValueError(f"{label} size mismatch")
    return resolved


def _validate_starting_audit(payload: Mapping[str, Any]) -> None:
    _require_m8(payload, label="starting audit")
    repository = payload.get("repository")
    if not isinstance(repository, Mapping):
        raise ValueError("starting audit lacks repository state")
    if (
        repository.get("next_unused_milestone") != M8_MILESTONE
        or not isinstance(repository.get("branch"), str)
        or not isinstance(repository.get("head"), str)
        or len(str(repository.get("head"))) != 40
        or repository.get("worktree_clean") is not True
    ):
        raise ValueError("starting audit does not prove a clean M8 starting state")
    for section in ("windows", "gpu", "docker", "isaac_ros_environment", "canonical_components"):
        if not isinstance(payload.get(section), Mapping) or not payload.get(section):
            raise ValueError(f"starting audit lacks {section}")


def _validate_differential(payload: Mapping[str, Any]) -> None:
    _require_m8(payload, label="Phase-0 differential report")
    unresolved = payload.get("unresolved_unintended_discrepancies")
    if (
        payload.get("common_domain_behavioral_equivalence") is not True
        or payload.get("headline_evaluation_unblocked") is not True
        or unresolved != []
    ):
        raise ValueError("Phase-0 differential report has an unresolved headline blocker")
    if not isinstance(payload.get("migration_discrepancy_history"), list):
        raise ValueError("Phase-0 differential report lacks migration discrepancy history")


def _validate_cpu_audit(payload: Mapping[str, Any]) -> None:
    _require_m8(payload, label="CPU/native validation")
    repository_python = payload.get("repository_python")
    native_windows = payload.get("native_windows_ros")
    phase_zero = payload.get("phase_0_replay")
    if not all(isinstance(item, Mapping) for item in (repository_python, native_windows, phase_zero)):
        raise ValueError("CPU/native validation lacks required validation blocks")
    if (
        int(repository_python.get("passed", 0)) <= 0
        or int(repository_python.get("failed", -1)) != 0
        or not isinstance(native_windows.get("build"), Mapping)
        or native_windows["build"].get("result") != "passed"
        or not isinstance(native_windows.get("test"), Mapping)
        or int(native_windows["test"].get("errors", -1)) != 0
        or int(native_windows["test"].get("failures", -1)) != 0
        or phase_zero.get("headline_evaluation_unblocked") is not True
    ):
        raise ValueError("CPU/native validation contains a failed required gate")


def _validate_native_runtime_audit(
    payload: Mapping[str, Any], *, repository_root: Path
) -> None:
    _require_m8(payload, label="native runtime contract audit")
    if payload.get("native_episode_executed") is not False:
        raise ValueError("native runtime contract audit must not masquerade as episode evidence")
    contracts = payload.get("verified_contracts")
    hashes = payload.get("source_sha256")
    if not isinstance(contracts, list) or not contracts or not isinstance(hashes, Mapping):
        raise ValueError("native runtime contract audit is incomplete")
    for relative, expected in hashes.items():
        source = _resolve_repository_reference(
            relative,
            repository_root=repository_root,
            label="native runtime source",
        )
        if not source.is_file() or not _is_sha256(expected) or sha256_file(source) != expected:
            raise ValueError(f"native runtime source binding is stale: {relative}")


def _validate_analysis_replay_binding(
    analysis: Mapping[str, Any],
    replay: Mapping[str, Any],
    *,
    replay_path: Path,
    repository_root: Path,
) -> Path:
    _require_m8(analysis, label="analysis")
    _require_m8(replay, label="replay")
    if analysis.get("evidence_class") != NATIVE_ISAAC_EVIDENCE_CLASS:
        raise ValueError("analysis is not native-Isaac evidence")
    if replay.get("split") != "frozen_holdout" or replay.get("passed") is not True:
        raise ValueError("replay is not a passing frozen holdout audit")
    if any(
        replay.get(field) is not True
        for field in (
            "fairness_passed",
            "freeze_validation_passed",
            "provenance_validation_passed",
            "seed_validation_passed",
            "profile_validation_passed",
        )
    ):
        raise ValueError("replay lacks a required semantic, fairness, or provenance pass")
    if analysis.get("manifest") != replay.get("manifest"):
        raise ValueError("analysis and replay are not cryptographically bound to one matrix/run")
    manifest_path = _resolve_repository_reference(
        analysis.get("manifest"),
        repository_root=repository_root,
        label="holdout matrix",
    )
    if (
        not manifest_path.is_file()
        or analysis.get("manifest_sha256") != replay.get("manifest_sha256")
        or analysis.get("manifest_sha256") != sha256_file(manifest_path)
        or analysis.get("replay_sha256") != sha256_file(replay_path)
        or _resolve_repository_reference(
            analysis.get("replay"),
            repository_root=repository_root,
            label="analysis replay",
        )
        != replay_path
    ):
        raise ValueError("analysis and replay are not cryptographically bound to one matrix/run")
    matrix = read_json(manifest_path)
    _require_m8(matrix, label="holdout matrix")
    if (
        matrix.get("split") != "frozen_holdout"
        or matrix.get("headline_eligible") is not True
        or matrix.get("freeze_sha256") != analysis.get("freeze_sha256")
        or matrix.get("seed_file_sha256") != replay.get("seed_file_sha256")
        or matrix.get("seed_file_sha256") != analysis.get("holdout_seed_file_sha256")
    ):
        raise ValueError("matrix, replay, and analysis do not describe one frozen holdout")
    if (
        int(analysis.get("episode_count", -1)) != int(replay.get("episode_count", -2))
        or int(replay.get("episode_audits_passed", -1)) != int(replay.get("episode_count", -2))
    ):
        raise ValueError("analysis/replay episode counts do not prove a complete holdout")
    profiles = analysis.get("profiles")
    if not isinstance(profiles, Mapping) or set(profiles) != set(PROFILE_STRATEGIES):
        raise ValueError("analysis does not contain every frozen M8 profile")
    return manifest_path


def _validate_completion_receipt(
    payload: Mapping[str, Any],
    *,
    completion_path: Path,
    matrix_path: Path,
    analysis_path: Path,
    replay_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    receipt_audit = validate_native_completion_receipt(
        completion_receipt_path=completion_path,
        matrix_path=matrix_path,
        replay_path=replay_path,
        repository_root=repository_root,
    )
    _require_m8(payload, label="holdout completion receipt")
    if payload.get("receipt_kind") != "native_completion" or payload.get("status") != "complete":
        raise ValueError("holdout completion receipt is not complete")
    if (
        _resolve_receipt_reference(
            payload.get("analysis"),
            receipt_path=completion_path,
            repository_root=repository_root,
            label="holdout completion analysis",
        )
        != analysis_path
        or payload.get("analysis_sha256") != sha256_file(analysis_path)
        or payload.get("replay_validation_sha256") != sha256_file(replay_path)
    ):
        raise ValueError("holdout completion receipt does not bind this analysis/replay pair")
    return receipt_audit


def _validate_figure_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    analysis_path: Path,
    classification: object,
) -> None:
    _require_m8(payload, label="figure manifest")
    if (
        payload.get("evidence_class") != NATIVE_ISAAC_EVIDENCE_CLASS
        or payload.get("source_analysis_sha256") != sha256_file(analysis_path)
        or payload.get("classification") != classification
    ):
        raise ValueError("figure manifest does not bind the analyzed native holdout")
    figures = payload.get("figures")
    if not isinstance(figures, list):
        raise ValueError("figure manifest lacks figures")
    ids = [str(figure.get("figure_id")) for figure in figures if isinstance(figure, Mapping)]
    if len(ids) != len(set(ids)) or not REQUIRED_FIGURE_IDS.issubset(ids):
        raise ValueError("figure manifest lacks the three required analytical figures")
    for figure in figures:
        if not isinstance(figure, Mapping) or not isinstance(figure.get("files"), list):
            raise ValueError("figure manifest contains a malformed figure")
        formats = set()
        for record in figure["files"]:
            if not isinstance(record, Mapping):
                raise ValueError("figure file record is malformed")
            _validate_file_record(
                record,
                base_directory=manifest_path.parent,
                label=f"figure {figure.get('figure_id')}",
            )
            formats.add(record.get("format"))
        if not {"pdf", "png"}.issubset(formats):
            raise ValueError(f"figure {figure.get('figure_id')} lacks PDF/PNG outputs")
    latex = payload.get("latex_includes")
    if not isinstance(latex, Mapping):
        raise ValueError("figure manifest lacks LaTeX includes")
    _validate_file_record(
        latex,
        base_directory=manifest_path.parent,
        label="figure LaTeX includes",
    )
    timeline = payload.get("timeline")
    if timeline is not None:
        if not isinstance(timeline, Mapping):
            raise ValueError("figure timeline record is malformed")
        _validate_file_record(
            timeline,
            base_directory=manifest_path.parent,
            label="representative timeline",
        )
        required_roles = timeline.get("causal_roles")
        if not isinstance(required_roles, Mapping) or not all(required_roles.values()):
            raise ValueError("representative timeline does not prove every required causal role")


def _audit_payloads(replay: Mapping[str, Any]) -> dict[tuple[Any, ...], tuple[Any, Any]]:
    result: dict[tuple[Any, ...], tuple[Any, Any]] = {}
    for audit in replay.get("audits", []):
        if not isinstance(audit, Mapping):
            raise ValueError("replay audit record is malformed")
        key = (
            audit.get("episode_id"),
            audit.get("profile_id"),
            audit.get("seed"),
            audit.get("strategy"),
        )
        if key in result:
            raise ValueError("replay contains a duplicate episode audit identity")
        result[key] = (
            audit.get("recomputed_metrics"),
            audit.get("invariant_violation_counts"),
        )
    return result


def _validate_archive_bundle(
    *,
    archive_path: Path,
    archive_manifest_path: Path,
    archive_matrix_path: Path,
    archive_replay_path: Path,
    direct_replay: Mapping[str, Any],
    direct_manifest_sha256: object,
) -> None:
    archive_audit = validate_archive(archive_path, archive_manifest_path)
    if not archive_audit["passed"]:
        raise ValueError(f"complete raw archive failed validation: {archive_audit['errors']}")
    binding = validate_archive_matrix_binding(archive_matrix_path)
    if binding is None:
        raise ValueError("archive matrix is not archive-backed")
    archive_matrix = read_json(archive_matrix_path)
    if archive_matrix.get("source_matrix_sha256") != direct_manifest_sha256:
        raise ValueError("archive matrix does not derive from the analyzed direct matrix")
    archived_replay = read_json(archive_replay_path)
    _require_m8(archived_replay, label="archive-backed replay")
    if (
        archived_replay.get("passed") is not True
        or archived_replay.get("manifest_sha256") != sha256_file(archive_matrix_path)
        or _audit_payloads(archived_replay) != _audit_payloads(direct_replay)
    ):
        raise ValueError("archive-backed replay differs from the direct validated replay")


def _validate_demo_receipt(
    payload: Mapping[str, Any],
    *,
    repository_root: Path,
    freeze_path: Path,
    freeze_sha256: object,
) -> None:
    _require_m8(payload, label="demo receipt")
    if (
        payload.get("receipt_kind") != "native_demo"
        or payload.get("status") != "complete"
        or payload.get("headline_eligible") is not False
        or payload.get("benchmark_evidence") is not False
        or payload.get("freeze_validation_passed") is not True
        or payload.get("freeze_manifest_sha256") != sha256_file(freeze_path)
        or payload.get("freeze_sha256") != freeze_sha256
    ):
        raise ValueError("demo receipt violates its frozen non-headline evidence boundary")
    for section, hash_name in (("video", "sha256"), ("independent_replay", "sha256")):
        record = payload.get(section)
        if not isinstance(record, Mapping) or not _is_sha256(record.get(hash_name)):
            raise ValueError(f"demo receipt lacks {section} binding")
        target = _resolve_repository_reference(
            record.get("path"),
            repository_root=repository_root,
            label=f"demo {section}",
        )
        if not target.is_file() or sha256_file(target) != record[hash_name]:
            raise ValueError(f"demo {section} hash mismatch")


def _format_optional(value: object, *, digits: int = 3) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _artifact_line(label: str, record: Mapping[str, Any]) -> str:
    return f"- {label}: `{record['path']}` (SHA-256 `{record['sha256']}`)."


def generate_technical_report(
    *,
    repository_root: Path | str,
    analysis_path: Path | str,
    replay_path: Path | str,
    starting_audit_path: Path | str,
    differential_report_path: Path | str,
    calibration_ledger_path: Path | str,
    freeze_manifest_path: Path | str,
    cpu_validation_path: Path | str,
    native_runtime_audit_path: Path | str,
    completion_receipt_path: Path | str,
    figure_manifest_path: Path | str,
    output_path: Path | str,
    archive_path: Path | str | None = None,
    archive_manifest_path: Path | str | None = None,
    archive_matrix_path: Path | str | None = None,
    archive_replay_path: Path | str | None = None,
    demo_receipt_path: Path | str | None = None,
) -> str:
    root = Path(repository_root).resolve()
    if not (root / ".git").exists():
        raise ValueError("repository_root must identify the ActionStream checkout")

    paths = {
        "analysis": Path(analysis_path).resolve(),
        "replay": Path(replay_path).resolve(),
        "starting audit": Path(starting_audit_path).resolve(),
        "Phase-0 differential": Path(differential_report_path).resolve(),
        "calibration ledger": Path(calibration_ledger_path).resolve(),
        "freeze manifest": Path(freeze_manifest_path).resolve(),
        "CPU/native validation": Path(cpu_validation_path).resolve(),
        "native runtime audit": Path(native_runtime_audit_path).resolve(),
        "holdout completion receipt": Path(completion_receipt_path).resolve(),
        "figure manifest": Path(figure_manifest_path).resolve(),
    }
    artifacts = {
        label: _artifact_record(path, repository_root=root) for label, path in paths.items()
    }
    analysis = read_json(paths["analysis"])
    replay = read_json(paths["replay"])
    starting = read_json(paths["starting audit"])
    differential = read_json(paths["Phase-0 differential"])
    ledger = read_json(paths["calibration ledger"])
    freeze = read_json(paths["freeze manifest"])
    cpu = read_json(paths["CPU/native validation"])
    native_runtime = read_json(paths["native runtime audit"])
    completion = read_json(paths["holdout completion receipt"])
    figures = read_json(paths["figure manifest"])

    _validate_starting_audit(starting)
    _validate_differential(differential)
    _validate_cpu_audit(cpu)
    _validate_native_runtime_audit(native_runtime, repository_root=root)
    validate_calibration_ledger(ledger, require_closed=True)
    freeze_audit = validate_freeze_manifest(paths["freeze manifest"], repository_root=root)
    if not freeze_audit["passed"]:
        raise ValueError(f"freeze validation failed: {freeze_audit['errors']}")
    holdout_matrix_path = _validate_analysis_replay_binding(
        analysis,
        replay,
        replay_path=paths["replay"],
        repository_root=root,
    )
    artifacts["holdout matrix"] = _artifact_record(
        holdout_matrix_path,
        repository_root=root,
    )
    if (
        analysis.get("freeze_sha256") != freeze.get("freeze_sha256")
        or _resolve_repository_reference(
            analysis.get("freeze_manifest"),
            repository_root=root,
            label="analysis freeze manifest",
        )
        != paths["freeze manifest"]
        or replay.get("freeze_audit", {}).get("manifest", {}).get("freeze_sha256")
        != freeze.get("freeze_sha256")
    ):
        raise ValueError("analysis/replay do not bind the supplied immutable freeze")
    completion_portability = _validate_completion_receipt(
        completion,
        completion_path=paths["holdout completion receipt"],
        matrix_path=holdout_matrix_path,
        analysis_path=paths["analysis"],
        replay_path=paths["replay"],
        repository_root=root,
    )
    artifacts["holdout preflight receipt"] = _artifact_record(
        completion_portability["preflight_receipt_path"],
        repository_root=root,
    )
    artifacts["holdout installed-adapter evidence"] = _artifact_record(
        completion_portability["installed_dynamic_adapter_evidence_path"],
        repository_root=root,
    )
    artifacts["holdout installed-executor evidence"] = _artifact_record(
        completion_portability["executor_evidence_path"],
        repository_root=root,
    )
    artifacts["holdout runner evidence"] = _artifact_record(
        completion_portability["runner_evidence_path"],
        repository_root=root,
    )
    if completion_portability["runner_support_evidence_path"] is not None:
        artifacts["holdout runner-support evidence"] = _artifact_record(
            completion_portability["runner_support_evidence_path"],
            repository_root=root,
        )
    artifacts["holdout external-environment evidence"] = _artifact_record(
        completion_portability["external_environment_evidence_path"],
        repository_root=root,
    )
    artifacts["holdout pixi.toml evidence"] = _artifact_record(
        completion_portability["pixi_manifest_evidence_path"],
        repository_root=root,
    )
    artifacts["holdout pixi.lock evidence"] = _artifact_record(
        completion_portability["pixi_lock_evidence_path"],
        repository_root=root,
    )
    _validate_figure_manifest(
        figures,
        manifest_path=paths["figure manifest"],
        analysis_path=paths["analysis"],
        classification=analysis.get("classification"),
    )

    archive_values = (archive_path, archive_manifest_path, archive_matrix_path, archive_replay_path)
    if any(value is not None for value in archive_values) and not all(
        value is not None for value in archive_values
    ):
        raise ValueError("archive reporting requires archive, sidecar, archive matrix, and replay")
    archive_available = all(value is not None for value in archive_values)
    if archive_available:
        assert archive_path is not None
        assert archive_manifest_path is not None
        assert archive_matrix_path is not None
        assert archive_replay_path is not None
        archive_paths = {
            "complete raw archive": Path(archive_path).resolve(),
            "archive manifest": Path(archive_manifest_path).resolve(),
            "archive-backed matrix": Path(archive_matrix_path).resolve(),
            "archive-backed replay": Path(archive_replay_path).resolve(),
        }
        artifacts.update(
            {
                label: _artifact_record(path, repository_root=root)
                for label, path in archive_paths.items()
            }
        )
        _validate_archive_bundle(
            archive_path=archive_paths["complete raw archive"],
            archive_manifest_path=archive_paths["archive manifest"],
            archive_matrix_path=archive_paths["archive-backed matrix"],
            archive_replay_path=archive_paths["archive-backed replay"],
            direct_replay=replay,
            direct_manifest_sha256=analysis.get("manifest_sha256"),
        )

    demo_available = demo_receipt_path is not None
    demo: Mapping[str, Any] | None = None
    if demo_available:
        demo_path = Path(demo_receipt_path).resolve()
        artifacts["demo receipt"] = _artifact_record(demo_path, repository_root=root)
        demo = read_json(demo_path)
        _validate_demo_receipt(
            demo,
            repository_root=root,
            freeze_path=paths["freeze manifest"],
            freeze_sha256=freeze.get("freeze_sha256"),
        )

    profiles = analysis["profiles"]
    primary_gate = analysis.get("primary_go_gate", {})
    strong_gate = analysis.get("strong_go_gate", {})
    repository = starting["repository"]
    baseline = ledger["baseline_gate"]
    development = ledger["development"]
    selected_id = development["selected_candidate_id"]
    selected = next(
        candidate for candidate in development["candidates"] if candidate["candidate_id"] == selected_id
    )
    freeze_inputs = {
        str(record["role"]): record
        for record in freeze.get("inputs", [])
        if isinstance(record, Mapping) and isinstance(record.get("role"), str)
    }
    protocol_record = freeze_inputs.get("protocol")
    if protocol_record is None:
        raise ValueError("freeze manifest lacks the finalized protocol role")
    protocol_path = _resolve_repository_reference(
        protocol_record.get("path"),
        repository_root=root,
        label="frozen protocol",
    )
    if sha256_file(protocol_path) != protocol_record.get("sha256"):
        raise ValueError("frozen protocol file hash mismatch")
    protocol = read_json(protocol_path)
    task = protocol["task"]
    controller = protocol["controller"]
    runtime = protocol["runtime"]
    native_environment = completion_portability["environment"]
    native_runtime = completion_portability["runtime"]
    native_gpu = completion_portability["selected_gpu_identity"]

    lines = [
        "# ActionStream M8-G0 dynamic-recovery report",
        "",
        "## Decision and evidence boundary",
        "",
        f"Classification: **{analysis['classification']}**",
        "",
        f"Evidence class: `{NATIVE_ISAAC_EVIDENCE_CLASS}`. Native frozen-holdout episodes: "
        f"{analysis['episode_count']}.",
        f"Replay: {replay['episode_audits_passed']}/{replay['episode_count']} episodes passed; "
        f"paired fairness={str(replay['fairness_passed']).lower()}; frozen provenance="
        f"{str(replay['freeze_validation_passed']).lower()}.",
        "Development results and the illustrative video are excluded from every headline result.",
        "",
        "## Starting commit and environment",
        "",
        f"The clean starting checkout was `{repository['branch']}` at `{repository['head']}` "
        f"tracking `{repository['upstream']}` ({repository['ahead']} ahead, "
        f"{repository['behind']} behind).",
        f"Starting audit host only (not the native holdout runtime): Windows "
        f"{starting['windows']['edition']} build {starting['windows']['build']}; "
        f"Python {starting['windows']['python']}; PyTorch {starting['windows']['pytorch']}; "
        f"CUDA toolkit {starting['windows']['cuda_toolkit']}.",
        f"Starting audit GPU only (not the native holdout runtime): "
        f"`{starting['gpu']['name']}` with driver {starting['gpu']['driver']}.",
        f"Native holdout runtime: GPU `{native_gpu['name']}` UUID "
        f"`{native_gpu['uuid']}` with {native_gpu['memory_total_mib']} MiB and driver "
        f"{native_gpu['driver_version']}; IsaacSim-ros_workspaces commit "
        f"`{native_environment['workspace_commit']}`; Pixi "
        f"{native_environment['pixi']['version']}; Python "
        f"{native_runtime['python_version']}; Isaac Sim "
        f"{native_runtime['packages']['isaacsim']}; ROS "
        f"{native_runtime['ros_distribution']} with "
        f"{native_runtime['rmw_implementation']} "
        f"{native_runtime['rmw_zenoh_cpp']}.",
        "",
        "## Reused components and frozen ActionStream semantics",
        "",
    ]
    for name, component in starting["canonical_components"].items():
        lines.append(f"- {name.replace('_', ' ')}: `{component}`.")
    lines.extend(
        [
            "",
            f"The frozen runtime is {runtime['control_frequency_hz']} Hz with a "
            f"{runtime['chunk_horizon']}-step horizon, {runtime['request_interval_steps']}-step "
            f"request interval, and `{runtime['action_representation']}` actions. Generation IDs "
            "remain invalidation epochs; aligned replacement remains atomic.",
            "",
            "## Phase 0: M4-to-M7 differential replay",
            "",
            f"Common-domain behavioral equivalence: "
            f"**{str(differential['common_domain_behavioral_equivalence']).lower()}** over "
            f"{differential['fixture']['common_event_count']} common events. Unresolved unintended "
            f"discrepancies: {len(differential['unresolved_unintended_discrepancies'])}.",
        ]
    )
    discrepancy_history = differential["migration_discrepancy_history"]
    if discrepancy_history:
        lines.append("Migration discrepancy found and repaired before benchmark execution:")
        for item in discrepancy_history:
            lines.append(
                f"- `{item['discrepancy_id']}`: {item['fix_summary']} Status: `{item['status']}`."
            )
    else:
        lines.append("No migration discrepancy was found.")
    lines.extend(
        [
            "",
            "## Native task, controller, and observation dependence",
            "",
            f"Task `{task['task_id']}` uses the physical `{task['primary_disturbance']}` disturbance. "
            f"Success requires release inside the final destination with {task['placement_tolerance_m']} m "
            f"planar tolerance and {task['stable_placement_steps']} stable steps.",
            f"The deterministic `{controller['type']}` controller consumes the latest end-effector, "
            "object, grasp, phase, active-destination, and disturbance state; bounded policy tests are "
            "part of the hash-bound CPU/native validation suite. No neural policy was trained.",
            "",
            "## CPU/native validation and simulator baseline gate",
            "",
            f"Repository Python tests: {cpu['repository_python']['passed']} passed, "
            f"{cpu['repository_python']['failed']} failed, {cpu['repository_python']['skipped']} skipped.",
            f"Native Windows ROS: {cpu['native_windows_ros']['test']['leaf_test_cases']} leaf tests, "
            f"{cpu['native_windows_ros']['test']['failures']} failures, "
            f"{cpu['native_windows_ros']['test']['errors']} errors.",
            f"Native Isaac Profile-0 baseline: {baseline['success_count']}/{baseline['trial_count']} "
            f"success ({float(baseline['observed_success_rate']):.1%}); replay validated="
            f"{str(baseline['replay_validated']).lower()}; required ceiling="
            f"{float(baseline['required_success_rate']):.1%}.",
            "",
            "## Frozen calibration, profiles, and seeds",
            "",
            f"Selected development candidate: `{selected_id}` after "
            f"{len(ledger['bounded_calibration_changes'])} bounded calibration change(s). Development "
            f"remains non-headline with {selected['seed_count']} seeds.",
            f"Frozen holdout seed count: {replay['seed_count']}; holdout seed SHA-256 "
            f"`{replay['seed_file_sha256']}`; freeze SHA-256 `{freeze['freeze_sha256']}`.",
        ]
    )
    for profile_id in PROFILE_STRATEGIES:
        record = freeze_inputs.get(f"profile:{profile_id}")
        if record is None:
            raise ValueError(f"freeze manifest lacks {profile_id}")
        lines.append(
            f"- `{profile_id}`: `{record['path']}` (SHA-256 `{record['sha256']}`)."
        )

    lines.extend(["", "## Preregistered gates", "", "| Gate condition | Passed |", "|---|---:|"])
    for name, passed in primary_gate.get("conditions", {}).items():
        lines.append(f"| {name.replace('_', ' ')} | {'yes' if passed else 'no'} |")
    lines.extend(
        [
            f"| primary gate overall | {'yes' if primary_gate.get('passed') else 'no'} |",
            f"| strong gate eligible | {'yes' if strong_gate.get('eligible') else 'no'} |",
            f"| strong gate overall | {'yes' if strong_gate.get('passed') else 'no'} |",
            "",
            "## Exact frozen-holdout results",
        ]
    )
    for profile_id, methods in PROFILE_STRATEGIES.items():
        profile = profiles[profile_id]
        lines.extend(
            [
                "",
                f"### `{profile_id}` ({profile['paired_trial_count']} paired seeds)",
                "",
                "| Method | Success | Rate | Median steps | Mean steps | Median wall (s) | "
                "Mean wall (s) | Median penalized recovery | Mean obsolete commands | Mean hold (s) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in methods:
            row = profile["strategies"][method]
            lines.append(
                f"| `{method}` | {row['successes']}/{row['trials']} | "
                f"{row['success_rate']:.1%} | {row['median_completion_steps']:.1f} | "
                f"{row['mean_completion_steps']:.1f} | {row['median_wall_clock_seconds']:.3f} | "
                f"{row['mean_wall_clock_seconds']:.3f} | "
                f"{row['median_penalized_recovery_latency_steps']:.1f} | "
                f"{row['mean_obsolete_destination_command_steps']:.2f} | "
                f"{row['mean_hold_control_seconds']:.3f} |"
            )
        if "aligned_minus_naive_percentage_points" in profile:
            lines.extend(
                [
                    "",
                    f"Aligned minus naive: {profile['aligned_minus_naive_percentage_points']:.1f} "
                    "percentage points; paired 95% CI "
                    f"[{profile['paired_bootstrap_95_ci_percentage_points'][0]:.1f}, "
                    f"{profile['paired_bootstrap_95_ci_percentage_points'][1]:.1f}] percentage "
                    f"points over {len(profile['raw_paired_outcomes'])} exact paired outcomes.",
                ]
            )

    primary_strategies = profiles["profile_1_fixed"]["strategies"]
    lines.extend(
        [
            "",
            "## Recovery, obsolete-action, and unavailable metric semantics",
            "",
            "| Method | Mean inference latency (ms) | P95 inference latency (ms) | "
            "Mean action age (steps) | P95 action age (steps) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in PROFILE_STRATEGIES["profile_1_fixed"]:
        descriptive = primary_strategies[method]["descriptive_metrics"]
        lines.append(
            f"| `{method}` | {_format_optional(descriptive['mean_inference_latency_ms'].get('mean'))} | "
            f"{_format_optional(descriptive['p95_inference_latency_ms'].get('median'))} | "
            f"{_format_optional(descriptive['mean_action_age_steps'].get('mean'))} | "
            f"{_format_optional(descriptive['p95_action_age_steps'].get('median'))} |"
        )
    lines.extend(
        [
            "",
            "A distribution with no observations is reported as `unavailable`, never as zero. The "
            "analysis artifact retains every required per-method descriptive metric, including "
            "collision/timeout rates, failure-reason counts, simulation time, obsolete motion, "
            "pre-switch command exposure, expired/duplicate/stale handling, request/response counts, "
            "queue rebuilds, and deadline misses.",
            "",
            "## Semantic, replay, and fairness checks",
            "",
            f"- Aligned forbidden executions: "
            f"{primary_gate.get('aligned_forbidden_execution_count', 'unavailable')}.",
            f"- Seed validation: {str(replay.get('seed_validation_passed')).lower()}; profile "
            f"validation: {str(replay.get('profile_validation_passed')).lower()}.",
            f"- Replay provenance: {str(replay.get('provenance_validation_passed')).lower()}; "
            f"paired fairness: {str(replay.get('fairness_passed')).lower()}.",
            f"- Bootstrap: {analysis['bootstrap']['resamples']} paired nonparametric resamples at "
            f"{analysis['bootstrap']['confidence']:.0%} confidence.",
            "",
            "## Figures, archive, and illustrative demo",
            "",
            f"Three analytical figure families are hash-bound by `{artifacts['figure manifest']['path']}`.",
            (
                "The complete raw archive and archive-backed replay are present and exactly reproduce "
                "the direct replay metrics."
                if archive_available
                else "Complete raw archive: unavailable at report generation; no archive success is claimed."
            ),
            (
                f"A replay-clean non-headline demo is present for episode "
                f"`{demo['episode']['episode_id']}`; it is illustrative, not benchmark evidence."
                if demo is not None
                else "Illustrative demo: unavailable/not recorded; no video claim is made."
            ),
            "",
            "## Exact reproduction commands",
            "",
            "```bash",
            "python -m action_stream_benchmark.m8_cli freeze-validate --repository-root . \\",
            "  --manifest outputs/m8_g0/protocol/freeze_manifest.json",
            "bash scripts/m8_run_isaac.sh \\",
            "  --matrix-suite-manifest outputs/m8_g0/holdout/matrix.json \\",
            "  --isaac-workspace \"$isaac_ws\" --pixi-exe \"$pixi\" \\",
            "  --gpu-index 0 --headless true --authorize-native-gpu-run",
            "python -m action_stream_benchmark.m8_cli validate \\",
            "  --manifest outputs/m8_g0/holdout/matrix.archive.json \\",
            "  --output outputs/m8_g0/holdout/replay_validation.archive.json",
            "```",
            "",
            "The native runner builds the current checkout, validates live frozen inputs before "
            "execution, and creates replay, analysis, and the three analytical figures. Archive and "
            "demo commands are separate because neither is headline evidence.",
            "",
            "## Limitations",
            "",
            "This is native Isaac Sim evidence with a deterministic scripted controller, not "
            "real-robot validation or production readiness. It covers one Franka destination-switch "
            "task and two asynchronous profiles. In Profile 2, a dropped synchronous response remains "
            "in flight until terminal drain and may produce an episode-long hold. Video is illustrative "
            "and never benchmark evidence. The M7-G0 static-plant saturation result remains a separate "
            "honest NO-GO and is not reinterpreted.",
            "",
            "## Bound artifacts",
            "",
        ]
    )
    for label, record in artifacts.items():
        lines.append(_artifact_line(label.title(), record))
    lines.extend(
        [
            "",
            "## Release-time repository status",
            "",
            "The final M8 commit hash, exact changed-file list, and clean-worktree confirmation are "
            "release-time metadata and are unavailable while this report is being built for inclusion "
            "in that commit. They must be supplied by the final handoff; no value is synthesized here.",
            "",
            "## Resume-ready result",
            "",
            f"- Built and replay-validated a native Isaac Sim Franka destination-switch benchmark "
            f"across {analysis['episode_count']} frozen holdout episodes; Profile-1 aligned versus "
            f"naive success differed by {profiles['profile_1_fixed']['aligned_minus_naive_percentage_points']:.1f} "
            f"percentage points (paired 95% CI "
            f"[{profiles['profile_1_fixed']['paired_bootstrap_95_ci_percentage_points'][0]:.1f}, "
            f"{profiles['profile_1_fixed']['paired_bootstrap_95_ci_percentage_points'][1]:.1f}]), "
            f"yielding **{analysis['classification']}**; this is simulation evidence, not real-robot "
            "validation.",
            "",
        ]
    )
    text = "\n".join(lines)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return text
