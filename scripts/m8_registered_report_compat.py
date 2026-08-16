#!/usr/bin/env python3
"""Generate the registered M8 report with one narrow path-compatibility shim.

The frozen matrix writer records ``freeze_manifest`` relative to the matrix
directory (for example ``../protocol_v3/freeze_manifest.json``), while the
registered report reader historically accepted only repository-relative
references.  Mutating the canonical matrix/analysis or the frozen report
source would invalidate their hashes.  This wrapper therefore teaches the
reader only that one already-bound reference form at runtime and records the
scope in a companion receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from action_stream_benchmark import m8_report as report_module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolve_matrix_relative_freeze(
    value: object,
    *,
    matrix_path: Path,
    repository_root: Path,
    supplied_freeze_path: Path,
) -> Path:
    text = str(value)
    pure = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        not text
        or "\\" in text
        or Path(text).is_absolute()
        or windows.is_absolute()
        or windows.drive
        or pure.is_absolute()
    ):
        raise ValueError("analysis freeze manifest is not a portable matrix-relative path")
    resolved = (matrix_path.parent / Path(*pure.parts)).resolve()
    root = repository_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("analysis freeze manifest escapes the repository") from exc
    if resolved != supplied_freeze_path.resolve():
        raise ValueError("analysis freeze manifest does not identify the supplied freeze")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def audit_payloads_by_normalized_source(
    replay: dict[str, Any],
) -> dict[tuple[str, str], tuple[Any, Any]]:
    """Index replay audits by their portable raw-event source identity.

    Current replay records do not carry the optional episode/profile/seed
    fields used by the historical report helper.  Direct records name the raw
    event path, while archive-backed records prefix that same path with
    ``archive.tar.gz!``.  Removing only that archive prefix yields the exact
    stable identity already hash-validated by the archive manifest.
    """

    result: dict[tuple[str, str], tuple[Any, Any]] = {}
    for audit in replay.get("audits", []):
        if not isinstance(audit, dict):
            raise ValueError("replay audit record is malformed")
        source = str(audit.get("source", "")).replace("\\", "/")
        normalized_source = source.rsplit("!", 1)[-1]
        pure = PurePosixPath(normalized_source)
        strategy = str(audit.get("strategy", ""))
        if (
            not normalized_source
            or not strategy
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.suffix != ".jsonl"
        ):
            raise ValueError("replay audit lacks a portable raw-event source identity")
        key = (pure.as_posix(), strategy)
        if key in result:
            raise ValueError("replay contains a duplicate normalized source identity")
        result[key] = (
            audit.get("recomputed_metrics"),
            audit.get("invariant_violation_counts"),
        )
    return result


def rewrite_fixed_reproduction_paths(
    report_path: Path,
    replacements: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Replace stale fixed example paths, requiring one unambiguous hit each."""

    text = report_path.read_text(encoding="utf-8")
    records = []
    for old, new in replacements:
        count = text.count(old)
        if count != 1:
            raise ValueError(
                f"expected exactly one registered-report reproduction path {old!r}, found {count}"
            )
        text = text.replace(old, new)
        records.append({"from": old, "to": new, "replacement_count": count})
    report_path.write_text(text, encoding="utf-8", newline="\n")
    return records


def generate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve()
    analysis_path = args.analysis.resolve()
    analysis = read_json(analysis_path)
    matrix_path = report_module._resolve_repository_reference(
        analysis.get("manifest"),
        repository_root=root,
        label="holdout matrix",
    )
    supplied_freeze = args.freeze_manifest.resolve()
    raw_freeze_reference = analysis.get("freeze_manifest")

    original_resolver = report_module._resolve_repository_reference
    original_audit_payloads = report_module._audit_payloads
    compatibility_applied = False
    try:
        ordinary_freeze = original_resolver(
            raw_freeze_reference,
            repository_root=root,
            label="analysis freeze manifest",
        )
    except ValueError:
        ordinary_freeze = None

    if ordinary_freeze == supplied_freeze:
        resolved_freeze = ordinary_freeze
    else:
        resolved_freeze = resolve_matrix_relative_freeze(
            raw_freeze_reference,
            matrix_path=matrix_path,
            repository_root=root,
            supplied_freeze_path=supplied_freeze,
        )
        compatibility_applied = True

    def compatibility_resolver(
        value: object, *, repository_root: Path, label: str
    ) -> Path:
        if label == "analysis freeze manifest" and value == raw_freeze_reference:
            return resolved_freeze
        return original_resolver(
            value,
            repository_root=repository_root,
            label=label,
        )

    audit_identity_compatibility_applied = False
    audit_payload_resolver = original_audit_payloads
    if args.archive_replay is not None:
        direct_replay = read_json(args.replay.resolve())
        archive_replay = read_json(args.archive_replay.resolve())
        try:
            original_audit_payloads(direct_replay)
            original_audit_payloads(archive_replay)
        except ValueError as exc:
            if "duplicate episode audit identity" not in str(exc):
                raise
            direct_payloads = audit_payloads_by_normalized_source(direct_replay)
            archived_payloads = audit_payloads_by_normalized_source(archive_replay)
            if direct_payloads != archived_payloads:
                raise ValueError(
                    "archive-backed replay differs from direct replay under normalized source identities"
                )
            audit_payload_resolver = audit_payloads_by_normalized_source
            audit_identity_compatibility_applied = True

    report_module._resolve_repository_reference = compatibility_resolver
    report_module._audit_payloads = audit_payload_resolver
    try:
        report_module.generate_technical_report(
            repository_root=root,
            analysis_path=analysis_path,
            replay_path=args.replay,
            starting_audit_path=args.starting_audit,
            differential_report_path=args.differential_report,
            calibration_ledger_path=args.calibration_ledger,
            freeze_manifest_path=supplied_freeze,
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
    finally:
        report_module._resolve_repository_reference = original_resolver
        report_module._audit_payloads = original_audit_payloads

    report_path = args.output.resolve()
    reproduction_replacements = [
        (
            "outputs/m8_g0/protocol/freeze_manifest.json",
            supplied_freeze.relative_to(root).as_posix(),
        ),
        (
            "outputs/m8_g0/holdout/matrix.json",
            matrix_path.relative_to(root).as_posix(),
        ),
    ]
    if args.archive_matrix is not None and args.archive_replay is not None:
        reproduction_replacements.extend(
            [
                (
                    "outputs/m8_g0/holdout/matrix.archive.json",
                    args.archive_matrix.resolve().relative_to(root).as_posix(),
                ),
                (
                    "outputs/m8_g0/holdout/replay_validation.archive.json",
                    args.archive_replay.resolve().relative_to(root).as_posix(),
                ),
            ]
        )
    reproduction_path_rewrites = rewrite_fixed_reproduction_paths(
        report_path,
        reproduction_replacements,
    )

    receipt_path = (
        args.compatibility_receipt
        if args.compatibility_receipt is not None
        else args.output.with_suffix(".receipt.json")
    )
    generator_source = Path(report_module.__file__).resolve()
    receipt = {
        "schema_version": 1,
        "milestone": "M8-G0",
        "artifact_kind": "registered_report_compatibility_receipt",
        "compatibility_applied": compatibility_applied or audit_identity_compatibility_applied,
        "compatibility_scope": [
            "Resolve only analysis.freeze_manifest relative to the canonical holdout matrix directory and require it to equal the separately supplied immutable freeze.",
            "When legacy audit identity fields are absent, compare direct and archive replay audits by the normalized raw-event source path plus strategy; require all identities to be unique and all metric/invariant payloads to match.",
            "Replace only the four fixed legacy reproduction-command paths with the exact supplied v3 artifact paths, requiring one unambiguous occurrence each.",
        ],
        "matrix_relative_freeze_compatibility_applied": compatibility_applied,
        "archive_audit_identity_compatibility_applied": audit_identity_compatibility_applied,
        "reproduction_path_rewrites": reproduction_path_rewrites,
        "canonical_analysis_mutated": False,
        "frozen_report_source_mutated": False,
        "analysis": {
            "path": analysis_path.relative_to(root).as_posix(),
            "sha256": sha256_file(analysis_path),
        },
        "freeze_manifest": {
            "path": supplied_freeze.relative_to(root).as_posix(),
            "sha256": sha256_file(supplied_freeze),
            "analysis_reference": str(raw_freeze_reference),
        },
        "registered_report_generator": {
            "path": generator_source.relative_to(root).as_posix(),
            "sha256": sha256_file(generator_source),
        },
        "compatibility_wrapper": {
            "path": Path(__file__).resolve().relative_to(root).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "report": {
            "path": report_path.relative_to(root).as_posix(),
            "sha256": sha256_file(report_path),
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--starting-audit", type=Path, required=True)
    parser.add_argument("--differential-report", type=Path, required=True)
    parser.add_argument("--calibration-ledger", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--cpu-validation", type=Path, required=True)
    parser.add_argument("--native-runtime-audit", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--figure-manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-manifest", type=Path)
    parser.add_argument("--archive-matrix", type=Path)
    parser.add_argument("--archive-replay", type=Path)
    parser.add_argument("--demo-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compatibility-receipt", type=Path)
    return parser.parse_args()


def main() -> None:
    receipt = generate(parse_args())
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
