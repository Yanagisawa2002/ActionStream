"""Validate a relocated ActionStream GPU v2 evidence archive.

Frozen episode rows intentionally retain their original remote absolute paths.
This auditor resolves those paths only when their benchmark-root basename and
relative suffix match the supplied local root. It never rewrites frozen rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


ARTIFACT_FIELDS = {
    "trace_path": "trace_sha256",
    "video_path": "video_sha256",
    "telemetry_jsonl_path": "telemetry_jsonl_sha256",
}
TELEMETRY_REQUIRED_FIELDS = {
    "schema_version",
    "engine_id",
    "event",
    "utc_unix_ns",
    "monotonic_ns",
    "pid",
    "thread",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_recorded_path(value: str, evidence_root: Path) -> tuple[Path, bool]:
    recorded = Path(value)
    if recorded.is_file():
        return recorded.resolve(), False

    normalized = value.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    matching_indices = [
        index for index, part in enumerate(parts) if part == evidence_root.name
    ]
    if len(matching_indices) != 1:
        raise ValueError(
            "Recorded path cannot be safely rebased through the supplied evidence "
            f"root {evidence_root.name!r}: {value}"
        )
    suffix = parts[matching_indices[0] + 1 :]
    candidate = evidence_root.joinpath(*suffix).resolve()
    try:
        candidate.relative_to(evidence_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Rebased path escapes evidence root: {value}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Rebased artifact is absent: {candidate}")
    return candidate, True


def _audit_manifest(evidence_root: Path) -> dict[str, Any]:
    manifest_path = evidence_root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    expected = manifest.get("artifacts")
    if not isinstance(expected, dict):
        raise ValueError(f"Invalid artifact manifest: {manifest_path}")
    actual = {
        path.relative_to(evidence_root).as_posix(): path
        for path in evidence_root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    hash_mismatches = sorted(
        relative
        for relative in set(expected) & set(actual)
        if _sha256(actual[relative]) != expected[relative]
    )
    return {
        "manifest_sha256": _sha256(manifest_path),
        "declared_files": len(expected),
        "actual_files": len(actual),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "hash_mismatches": hash_mismatches,
        "valid": not missing and not unexpected and not hash_mismatches,
    }


def audit_root(evidence_root: Path) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    rows_path = evidence_root / "episodes.jsonl"
    receipt_path = evidence_root / "orchestrator_receipt.json"
    rows = _read_jsonl(rows_path)
    receipt = _read_json(receipt_path)
    errors: list[str] = []
    field_counts = {field: 0 for field in ARTIFACT_FIELDS}
    rebased_counts = {field: 0 for field in ARTIFACT_FIELDS}
    unique_paths = {field: set() for field in ARTIFACT_FIELDS}
    telemetry_events = 0
    telemetry_invalid_events = 0

    for row_number, row in enumerate(rows, start=1):
        for path_field, hash_field in ARTIFACT_FIELDS.items():
            value = row.get(path_field)
            expected_hash = row.get(hash_field)
            if value is None and expected_hash is None:
                continue
            if not isinstance(value, str) or not isinstance(expected_hash, str):
                errors.append(
                    f"row {row_number}: {path_field}/{hash_field} must both be strings"
                )
                continue
            try:
                artifact, rebased = _resolve_recorded_path(value, evidence_root)
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"row {row_number}: {exc}")
                continue
            field_counts[path_field] += 1
            rebased_counts[path_field] += int(rebased)
            unique_paths[path_field].add(artifact)
            if _sha256(artifact) != expected_hash:
                errors.append(f"row {row_number}: hash mismatch for {path_field}")
                continue
            if path_field != "telemetry_jsonl_path":
                continue
            for event_number, event in enumerate(_read_jsonl(artifact), start=1):
                telemetry_events += 1
                if event.get(
                    "schema_version"
                ) != 1 or not TELEMETRY_REQUIRED_FIELDS <= set(event):
                    telemetry_invalid_events += 1
                    errors.append(
                        f"row {row_number}: invalid telemetry event {event_number}"
                    )

    manifest = _audit_manifest(evidence_root)
    if not manifest["valid"]:
        errors.append("artifact manifest validation failed")
    receipt_hash_match = receipt.get("episodes_sha256") == _sha256(rows_path)
    receipt_count_match = receipt.get("record_count") == len(rows)
    if not receipt_hash_match:
        errors.append("orchestrator episodes SHA-256 mismatch")
    if not receipt_count_match:
        errors.append("orchestrator record count mismatch")

    return {
        "root_name": evidence_root.name,
        "record_count": len(rows),
        "orchestrator_receipt_sha256": _sha256(receipt_path),
        "receipt_episodes_hash_match": receipt_hash_match,
        "receipt_record_count_match": receipt_count_match,
        "artifact_references": field_counts,
        "unique_artifacts": {
            field: len(paths) for field, paths in unique_paths.items()
        },
        "rebased_absolute_paths": rebased_counts,
        "telemetry_events": telemetry_events,
        "telemetry_invalid_events": telemetry_invalid_events,
        "manifest": manifest,
        "errors": errors,
        "status": "pass" if not errors else "fail",
    }


def audit_archive(canary_root: Path, holdout_root: Path) -> dict[str, Any]:
    canary = audit_root(canary_root)
    holdout = audit_root(holdout_root)
    return {
        "schema_version": 1,
        "status": "pass"
        if canary["status"] == "pass" and holdout["status"] == "pass"
        else "fail",
        "path_policy": (
            "Original absolute paths are preserved; basename-anchored suffixes "
            "are resolved inside each supplied evidence root."
        ),
        "canary": canary,
        "holdout": holdout,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = audit_archive(args.canary_root, args.holdout_root)
    if args.output is not None:
        _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
