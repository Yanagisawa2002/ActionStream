"""Command-line evidence pipeline for the frozen M8-G0 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .m8_analysis import analyze_manifest
from .m8_archive import build_archive, build_archive_matrix, validate_archive
from .m8_figures import generate_analysis_figures, generate_episode_timeline
from .m8_matrix import build_matrix_manifest
from .m8_protocol import (
    build_freeze_manifest,
    close_calibration_ledger,
    load_protocol,
    record_development_candidate,
    record_native_baseline_gate,
    validate_freeze_manifest,
    write_freeze_manifest,
)
from .m8_replay import validate_manifest
from .m8_report import generate_technical_report, write_unavailable_report
from .schema import write_json_atomic


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="action-stream-m8",
        description=(
            "Record calibration, freeze, validate, analyze, archive, and plot "
            "M8-G0 evidence."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    protocol = commands.add_parser("protocol-validate", help="validate the frozen protocol")
    protocol.add_argument("--protocol", type=Path, required=True)

    baseline_record = commands.add_parser(
        "baseline-record",
        help="validate and atomically record the completed native >=18/20 baseline",
    )
    baseline_record.add_argument("--repository-root", type=Path, required=True)
    baseline_record.add_argument("--calibration-ledger", type=Path, required=True)
    baseline_record.add_argument("--baseline-seeds", type=Path, required=True)
    baseline_record.add_argument("--profile", type=Path, required=True)
    baseline_record.add_argument("--matrix", type=Path, required=True)
    baseline_record.add_argument("--replay", type=Path, required=True)
    baseline_record.add_argument("--completion-receipt", type=Path, required=True)

    candidate_record = commands.add_parser(
        "candidate-record",
        help="replay and atomically append one immutable development candidate",
    )
    candidate_record.add_argument("--repository-root", type=Path, required=True)
    candidate_record.add_argument("--calibration-ledger", type=Path, required=True)
    candidate_record.add_argument(
        "--candidate-id",
        choices=("candidate_0", "candidate_1", "candidate_2"),
        required=True,
    )
    candidate_record.add_argument("--protocol", type=Path, required=True)
    candidate_record.add_argument("--matrix", type=Path, required=True)
    candidate_record.add_argument("--replay", type=Path, required=True)
    candidate_record.add_argument("--development-seeds", type=Path, required=True)
    candidate_record.add_argument(
        "--reason",
        help="required for candidate_1/2; exact diffs are derived, never entered by hand",
    )

    ledger_close = commands.add_parser(
        "ledger-close",
        help="select a recorded candidate and atomically authorize holdout freeze",
    )
    ledger_close.add_argument("--repository-root", type=Path, required=True)
    ledger_close.add_argument("--calibration-ledger", type=Path, required=True)
    ledger_close.add_argument(
        "--selected-candidate",
        choices=("candidate_0", "candidate_1", "candidate_2"),
        required=True,
    )
    ledger_close.add_argument("--protocol", type=Path, required=True)
    ledger_close.add_argument("--development-seeds", type=Path, required=True)

    freeze = commands.add_parser("freeze", help="create or verify the immutable protocol manifest")
    freeze.add_argument("--repository-root", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--baseline-seeds", type=Path, required=True)
    freeze.add_argument("--development-seeds", type=Path, required=True)
    freeze.add_argument("--holdout-seeds", type=Path, required=True)
    freeze.add_argument("--calibration-ledger", type=Path, required=True)
    freeze.add_argument("--baseline-matrix", type=Path, required=True)
    freeze.add_argument("--baseline-replay", type=Path, required=True)
    freeze.add_argument("--baseline-completion-receipt", type=Path, required=True)
    freeze.add_argument("--profile", type=Path, action="append", required=True)
    freeze.add_argument("--additional-input", type=Path, action="append", default=[])
    freeze.add_argument("--output", type=Path, required=True)

    freeze_validate = commands.add_parser("freeze-validate", help="verify all frozen input hashes")
    freeze_validate.add_argument("--repository-root", type=Path, required=True)
    freeze_validate.add_argument("--manifest", type=Path, required=True)

    matrix = commands.add_parser(
        "matrix-create",
        help="create one persistent-Isaac batch manifest and shared paired traces",
    )
    matrix.add_argument("--repository-root", type=Path, required=True)
    matrix.add_argument("--freeze-manifest", type=Path)
    matrix.add_argument("--candidate-protocol", type=Path)
    matrix.add_argument("--seeds", type=Path, required=True)
    matrix.add_argument("--profile", type=Path, action="append", required=True)
    matrix.add_argument("--split", required=True)
    matrix.add_argument("--output-root", type=Path, required=True)
    matrix.add_argument("--manifest", type=Path, required=True)
    matrix.add_argument("--request-count", type=int, default=512)
    matrix.add_argument(
        "--strategy",
        action="append",
        default=[],
        help="development-only method filter; repeat for more than one method",
    )

    replay = commands.add_parser("validate", help="independently replay a native-Isaac matrix")
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)

    analysis = commands.add_parser("analyze", help="analyze a replay-validated paired matrix")
    analysis.add_argument("--manifest", type=Path, required=True)
    analysis.add_argument("--replay", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--bootstrap-resamples", type=int, default=20_000)

    figures = commands.add_parser(
        "figures",
        help="generate analytical PDF/PNG figures and an optional causal SVG timeline",
    )
    figures.add_argument("--analysis", type=Path, required=True)
    figures.add_argument("--output-dir", type=Path, required=True)
    figures.add_argument("--timeline-log", type=Path)
    figures.add_argument("--timeline-output", type=Path)

    archive = commands.add_parser("archive", help="build a deterministic complete-evidence archive")
    archive.add_argument("--root", type=Path, required=True)
    archive.add_argument("--member", type=Path, action="append", default=[])
    archive.add_argument(
        "--member-list",
        type=Path,
        help="UTF-8 file containing one root-relative evidence member per line",
    )
    archive.add_argument("--archive", type=Path, required=True)
    archive.add_argument("--manifest", type=Path, required=True)

    archive_validate = commands.add_parser("archive-validate", help="verify archive bytes and every member")
    archive_validate.add_argument("--archive", type=Path, required=True)
    archive_validate.add_argument("--manifest", type=Path, required=True)

    archive_matrix = commands.add_parser(
        "archive-matrix",
        help="derive a portable full-raw archive-backed matrix without changing the direct matrix",
    )
    archive_matrix.add_argument("--source-matrix", type=Path, required=True)
    archive_matrix.add_argument("--archive", type=Path, required=True)
    archive_matrix.add_argument("--archive-manifest", type=Path, required=True)
    archive_matrix.add_argument("--output", type=Path, required=True)

    report = commands.add_parser("report", help="write a report from validated native analysis")
    report.add_argument("--repository-root", type=Path, required=True)
    report.add_argument("--analysis", type=Path, required=True)
    report.add_argument("--replay", type=Path, required=True)
    report.add_argument("--starting-audit", type=Path, required=True)
    report.add_argument("--differential-report", type=Path, required=True)
    report.add_argument("--calibration-ledger", type=Path, required=True)
    report.add_argument("--freeze-manifest", type=Path, required=True)
    report.add_argument("--cpu-validation", type=Path, required=True)
    report.add_argument("--native-runtime-audit", type=Path, required=True)
    report.add_argument("--completion-receipt", type=Path, required=True)
    report.add_argument("--figure-manifest", type=Path, required=True)
    report.add_argument("--archive", type=Path)
    report.add_argument("--archive-manifest", type=Path)
    report.add_argument("--archive-matrix", type=Path)
    report.add_argument("--archive-replay", type=Path)
    report.add_argument("--demo-receipt", type=Path)
    report.add_argument("--output", type=Path, required=True)

    unavailable = commands.add_parser(
        "report-unavailable",
        help="record an explicit native unavailable/not-run NO-GO without metrics",
    )
    unavailable.add_argument("--stage", required=True)
    unavailable.add_argument("--reason", required=True)
    unavailable.add_argument("--output", type=Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "protocol-validate":
        _emit(load_protocol(args.protocol))
        return 0
    if args.command == "baseline-record":
        result = record_native_baseline_gate(
            repository_root=args.repository_root,
            ledger_path=args.calibration_ledger,
            matrix_path=args.matrix,
            replay_path=args.replay,
            baseline_seed_path=args.baseline_seeds,
            profile_path=args.profile,
            completion_receipt_path=args.completion_receipt,
        )
        _emit(result)
        return 0
    if args.command == "candidate-record":
        result = record_development_candidate(
            repository_root=args.repository_root,
            ledger_path=args.calibration_ledger,
            candidate_id=args.candidate_id,
            protocol_path=args.protocol,
            matrix_path=args.matrix,
            replay_path=args.replay,
            development_seed_path=args.development_seeds,
            reason=args.reason,
        )
        _emit(result)
        return 0
    if args.command == "ledger-close":
        result = close_calibration_ledger(
            repository_root=args.repository_root,
            ledger_path=args.calibration_ledger,
            selected_candidate_id=args.selected_candidate,
            finalized_protocol_path=args.protocol,
            development_seed_path=args.development_seeds,
        )
        _emit(result)
        return 0
    if args.command == "freeze":
        manifest = build_freeze_manifest(
            repository_root=args.repository_root,
            protocol_path=args.protocol,
            baseline_seed_path=args.baseline_seeds,
            development_seed_path=args.development_seeds,
            holdout_seed_path=args.holdout_seeds,
            calibration_ledger_path=args.calibration_ledger,
            baseline_matrix_path=args.baseline_matrix,
            baseline_replay_path=args.baseline_replay,
            baseline_completion_receipt_path=args.baseline_completion_receipt,
            profile_paths=args.profile,
            additional_inputs=args.additional_input,
        )
        write_freeze_manifest(args.output, manifest)
        _emit(manifest)
        return 0
    if args.command == "freeze-validate":
        result = validate_freeze_manifest(
            args.manifest,
            repository_root=args.repository_root,
        )
        _emit(result)
        return 0 if result["passed"] else 1
    if args.command == "matrix-create":
        result = build_matrix_manifest(
            repository_root=args.repository_root,
            freeze_manifest_path=args.freeze_manifest,
            candidate_protocol_path=args.candidate_protocol,
            seed_path=args.seeds,
            profile_paths=args.profile,
            output_root=args.output_root,
            manifest_path=args.manifest,
            split=args.split,
            request_count=args.request_count,
            strategy_filter=args.strategy,
        )
        _emit(result)
        return 0
    if args.command == "validate":
        result = validate_manifest(args.manifest, output_path=args.output)
        _emit(result)
        return 0 if result["passed"] else 1
    if args.command == "analyze":
        result = analyze_manifest(
            args.manifest,
            replay_path=args.replay,
            output_path=args.output,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        _emit(result)
        return 0
    if args.command == "figures":
        result = generate_analysis_figures(args.analysis, args.output_dir)
        if args.timeline_log:
            timeline_output = args.timeline_output or (args.output_dir / "representative_timeline.svg")
            result["timeline"] = generate_episode_timeline(args.timeline_log, timeline_output)
            write_json_atomic(args.output_dir / "figure_manifest.json", result)
        _emit(result)
        return 0
    if args.command == "archive":
        members = list(args.member)
        if args.member_list:
            members.extend(
                Path(line.strip())
                for line in args.member_list.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        result = build_archive(
            root=args.root,
            members=members,
            archive_path=args.archive,
            manifest_path=args.manifest,
        )
        _emit(result)
        return 0
    if args.command == "archive-validate":
        result = validate_archive(args.archive, args.manifest)
        _emit(result)
        return 0 if result["passed"] else 1
    if args.command == "archive-matrix":
        result = build_archive_matrix(
            source_matrix_path=args.source_matrix,
            archive_path=args.archive,
            archive_manifest_path=args.archive_manifest,
            output_path=args.output,
        )
        _emit(result)
        return 0
    if args.command == "report":
        text = generate_technical_report(
            repository_root=args.repository_root,
            analysis_path=args.analysis,
            replay_path=args.replay,
            starting_audit_path=args.starting_audit,
            differential_report_path=args.differential_report,
            calibration_ledger_path=args.calibration_ledger,
            freeze_manifest_path=args.freeze_manifest,
            cpu_validation_path=args.cpu_validation,
            native_runtime_audit_path=args.native_runtime_audit,
            completion_receipt_path=args.completion_receipt,
            figure_manifest_path=args.figure_manifest,
            archive_path=args.archive,
            archive_manifest_path=args.archive_manifest,
            archive_matrix_path=args.archive_matrix,
            archive_replay_path=args.archive_replay,
            demo_receipt_path=args.demo_receipt,
            output_path=args.output,
        )
        _emit({"output": str(args.output), "character_count": len(text)})
        return 0
    if args.command == "report-unavailable":
        result = write_unavailable_report(
            args.output,
            stage=args.stage,
            reason=args.reason,
        )
        _emit(result)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
