"""Deterministic compressed evidence archives with exact member hashes."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import io
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tarfile
import tempfile
from typing import Any, Iterable

from .m8_protocol import M8_MILESTONE, M8_SCHEMA_VERSION, sha256_file
from .schema import read_json, write_json_atomic


DIRECT_EPISODE_ARTIFACT_FIELDS = (
    "event_log_path",
    "summary_path",
    "fault_trace_file",
    "scenario_file",
)
ARCHIVE_EPISODE_ARTIFACT_FIELDS = (
    "archive_member",
    "summary_archive_member",
    "fault_trace_archive_member",
    "scenario_archive_member",
)
ARCHIVE_MATRIX_BINDING_FIELDS = (
    "evidence_storage",
    "source_matrix",
    "source_matrix_sha256",
    "archive",
    "archive_sha256",
    "archive_manifest",
    "archive_manifest_sha256",
)


def validate_member_name(member_name: str) -> str:
    """Return a canonical POSIX member name or reject an unsafe spelling."""

    if not isinstance(member_name, str) or not member_name:
        raise ValueError("archive member name must be a non-empty string")
    pure = PurePosixPath(member_name)
    windows = PureWindowsPath(member_name)
    if (
        pure.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or "\\" in member_name
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != member_name
    ):
        raise ValueError(f"unsafe archive member name: {member_name!r}")
    return member_name


def _safe_relative_path(value: Any, *, field: str) -> Path:
    text = str(value)
    path = Path(text)
    windows = PureWindowsPath(text)
    posix = PurePosixPath(text.replace("\\", "/"))
    if (
        not text
        or path.is_absolute()
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or "\\" in text
        or text == "."
        or ".." in posix.parts
        or posix.as_posix() != text
    ):
        raise ValueError(f"{field} must be a traversal-free relative path")
    return path


def _relative_binding(target: Path, *, base: Path, field: str) -> str:
    try:
        value = target.resolve().relative_to(base.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{field} must be stored below the derived matrix directory") from exc
    _safe_relative_path(value, field=field)
    return value


def _member_name(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"archive member escapes root: {resolved}") from exc
    name = relative.as_posix()
    return validate_member_name(name)


def build_archive(
    *,
    root: Path | str,
    members: Iterable[Path | str],
    archive_path: Path | str,
    manifest_path: Path | str,
) -> dict[str, Any]:
    source_root = Path(root).resolve()
    archive = Path(archive_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    if archive.exists() or manifest_file.exists():
        raise FileExistsError("refusing to overwrite an evidence archive or manifest")
    selected: dict[str, Path] = {}
    for value in members:
        path = Path(value)
        if not path.is_absolute():
            path = source_root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        name = _member_name(path, source_root)
        if name in selected:
            raise ValueError(f"duplicate archive member: {name}")
        selected[name] = path.resolve()
    if not selected:
        raise ValueError("evidence archive requires at least one member")

    archive.parent.mkdir(parents=True, exist_ok=True)
    tar_fd, tar_name = tempfile.mkstemp(prefix=".m8-evidence-", suffix=".tar", dir=archive.parent)
    os.close(tar_fd)
    temporary_archive = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    records = []
    try:
        with tarfile.open(tar_name, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name in sorted(selected):
                path = selected[name]
                data = path.read_bytes()
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mtime = 0
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                tar.addfile(info, io.BytesIO(data))
                records.append(
                    {
                        "path": name,
                        "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
        with open(tar_name, "rb") as source, temporary_archive.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    compressed.write(block)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_archive, archive)
    finally:
        Path(tar_name).unlink(missing_ok=True)
        temporary_archive.unlink(missing_ok=True)

    payload = {
        "schema_version": M8_SCHEMA_VERSION,
        "milestone": M8_MILESTONE,
        "archive": archive.name,
        "archive_format": "deterministic_tar_gzip",
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "members": records,
    }
    write_json_atomic(manifest_file, payload)
    return payload


def validate_archive(
    archive_path: Path | str,
    manifest_path: Path | str,
) -> dict[str, Any]:
    archive = Path(archive_path)
    manifest = read_json(manifest_path)
    errors: list[str] = []
    if manifest.get("schema_version") != M8_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if manifest.get("milestone") != M8_MILESTONE:
        errors.append("milestone_mismatch")
    if manifest.get("archive_format") != "deterministic_tar_gzip":
        errors.append("archive_format_mismatch")
    if manifest.get("archive") != archive.name:
        errors.append("archive_filename_mismatch")
    actual_archive_sha256 = sha256_file(archive)
    if actual_archive_sha256 != manifest.get("archive_sha256"):
        errors.append("archive_sha256_mismatch")
    if archive.stat().st_size != manifest.get("archive_size_bytes"):
        errors.append("archive_size_mismatch")
    expected: dict[str, dict[str, Any]] = {}
    records = manifest.get("members", [])
    if not isinstance(records, list):
        errors.append("member_records_invalid")
        records = []
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            errors.append("member_record_invalid")
            continue
        try:
            name = validate_member_name(item["path"])
        except ValueError:
            errors.append(f"unsafe_expected_member:{item['path']}")
            continue
        if name in expected:
            errors.append(f"duplicate_expected_member:{name}")
            continue
        expected[name] = item
    actual: dict[str, dict[str, Any]] = {}
    try:
        with tarfile.open(archive, mode="r|gz") as tar:
            for member in tar:
                try:
                    validate_member_name(member.name)
                except ValueError:
                    errors.append(f"unsafe_archive_member:{member.name}")
                    continue
                if not member.isfile():
                    errors.append(f"non_regular_member:{member.name}")
                    continue
                if member.name in actual:
                    errors.append(f"duplicate_member:{member.name}")
                    continue
                stream = tar.extractfile(member)
                if stream is None:
                    errors.append(f"unreadable_member:{member.name}")
                    continue
                data = stream.read()
                actual[member.name] = {
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
    except (OSError, tarfile.TarError) as exc:
        errors.append(f"archive_read_error:{type(exc).__name__}")
    if set(actual) != set(expected):
        errors.append("member_set_mismatch")
    for name in sorted(set(actual) & set(expected)):
        if actual[name] != {
            "size_bytes": expected[name].get("size_bytes"),
            "sha256": expected[name].get("sha256"),
        }:
            errors.append(f"member_hash_or_size_mismatch:{name}")
    return {
        "passed": not errors,
        "errors": errors,
        "member_count": len(actual),
        "archive_sha256": actual_archive_sha256,
    }


def read_archive_member(archive_path: Path | str, member_name: str) -> bytes:
    validate_member_name(member_name)
    pure = PurePosixPath(member_name)
    with tarfile.open(archive_path, mode="r:gz") as tar:
        try:
            member = tar.getmember(pure.as_posix())
        except KeyError as exc:
            raise FileNotFoundError(member_name) from exc
        if not member.isfile():
            raise ValueError(f"archive member is not a regular file: {member_name}")
        stream = tar.extractfile(member)
        if stream is None:
            raise ValueError(f"archive member is unreadable: {member_name}")
        return stream.read()


def iter_archive_members(
    archive_path: Path | str,
    member_names: Iterable[str],
) -> Iterable[tuple[str, bytes]]:
    """Yield selected regular members in physical archive order in one pass."""

    requested = {validate_member_name(name) for name in member_names}
    if not requested:
        return
    found: set[str] = set()
    with tarfile.open(archive_path, mode="r|gz") as tar:
        for member in tar:
            if member.name not in requested:
                continue
            if member.name in found:
                raise ValueError(f"duplicate archive member: {member.name}")
            if not member.isfile():
                raise ValueError(f"archive member is not a regular file: {member.name}")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"archive member is unreadable: {member.name}")
            found.add(member.name)
            yield member.name, stream.read()
    missing = requested - found
    if missing:
        raise FileNotFoundError(f"archive members not found: {sorted(missing)}")


def read_archive_members(
    archive_path: Path | str,
    member_names: Iterable[str],
) -> dict[str, bytes]:
    return dict(iter_archive_members(archive_path, member_names))


def _member_for_direct_artifact(
    artifact_path: Path,
    *,
    member_records: dict[str, dict[str, Any]],
) -> str:
    resolved = artifact_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual_size = resolved.stat().st_size
    actual_sha256 = sha256_file(resolved)
    resolved_posix = resolved.as_posix().casefold()
    matches = [
        name
        for name, record in member_records.items()
        if (
            (resolved_posix == name.casefold() or resolved_posix.endswith(f"/{name.casefold()}"))
            and record.get("size_bytes") == actual_size
            and record.get("sha256") == actual_sha256
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"direct artifact must map to exactly one hash-bound archive member: {resolved}"
        )
    return matches[0]


def build_archive_matrix(
    *,
    source_matrix_path: Path | str,
    archive_path: Path | str,
    archive_manifest_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Derive a non-overwriting, fully archive-backed copy of a direct matrix."""

    source_path = Path(source_matrix_path).resolve()
    archive = Path(archive_path).resolve()
    archive_manifest = Path(archive_manifest_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite derived matrix: {output}")
    if output == source_path:
        raise ValueError("archive-backed matrix must be separate from the source matrix")

    archive_audit = validate_archive(archive, archive_manifest)
    if not archive_audit["passed"]:
        raise ValueError(f"archive validation failed: {archive_audit['errors']}")

    # Completion means the direct matrix can be independently replayed now.
    from .m8_replay import validate_manifest

    source_replay = validate_manifest(source_path)
    if not source_replay["passed"]:
        raise ValueError("source matrix is not a completed replay-clean direct matrix")

    source = read_json(source_path)
    if any(field in source for field in ARCHIVE_MATRIX_BINDING_FIELDS):
        raise ValueError("source matrix is already archive-backed")
    sidecar = read_json(archive_manifest)
    member_records = {
        validate_member_name(str(record["path"])): record
        for record in sidecar.get("members", [])
    }
    episodes = source.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("source matrix contains no episodes")

    derived = deepcopy(source)
    derived_episodes: list[dict[str, Any]] = []
    replacements = dict(zip(DIRECT_EPISODE_ARTIFACT_FIELDS, ARCHIVE_EPISODE_ARTIFACT_FIELDS))
    for entry in episodes:
        if not isinstance(entry, dict):
            raise ValueError("source matrix episode must be an object")
        if any(field in entry for field in ARCHIVE_EPISODE_ARTIFACT_FIELDS):
            raise ValueError("source matrix episode already contains archive members")
        missing = [field for field in DIRECT_EPISODE_ARTIFACT_FIELDS if not entry.get(field)]
        if missing:
            raise ValueError(f"source matrix episode lacks direct artifacts: {missing}")
        converted = deepcopy(entry)
        for direct_field, archive_field in replacements.items():
            relative = _safe_relative_path(entry[direct_field], field=direct_field)
            artifact = (source_path.parent / relative).resolve()
            converted[archive_field] = _member_for_direct_artifact(
                artifact,
                member_records=member_records,
            )
            del converted[direct_field]
        derived_episodes.append(converted)
    derived["episodes"] = derived_episodes
    derived.update(
        {
            "evidence_storage": "deterministic_archive",
            "source_matrix": _relative_binding(
                source_path,
                base=output.parent,
                field="source_matrix",
            ),
            "source_matrix_sha256": sha256_file(source_path),
            "archive": _relative_binding(archive, base=output.parent, field="archive"),
            "archive_sha256": sha256_file(archive),
            "archive_manifest": _relative_binding(
                archive_manifest,
                base=output.parent,
                field="archive_manifest",
            ),
            "archive_manifest_sha256": sha256_file(archive_manifest),
        }
    )
    write_json_atomic(output, derived)
    return derived


def validate_archive_matrix_binding(
    matrix_path: Path | str,
    matrix: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate archive paths, hashes, sidecar, and exact source-matrix derivation."""

    path = Path(matrix_path).resolve()
    payload = matrix if matrix is not None else read_json(path)
    is_archive = any(field in payload for field in ARCHIVE_MATRIX_BINDING_FIELDS)
    if not is_archive:
        if any(
            archive_field in entry
            for entry in payload.get("episodes", [])
            if isinstance(entry, dict)
            for archive_field in ARCHIVE_EPISODE_ARTIFACT_FIELDS
        ):
            raise ValueError("archive episode members require matrix-level archive bindings")
        return None
    missing_bindings = [
        field for field in ARCHIVE_MATRIX_BINDING_FIELDS if field not in payload
    ]
    if missing_bindings:
        raise ValueError(f"archive matrix bindings missing: {missing_bindings}")
    if payload.get("evidence_storage") != "deterministic_archive":
        raise ValueError("archive matrix evidence_storage mismatch")

    bound_paths: dict[str, Path] = {}
    for field in ("source_matrix", "archive", "archive_manifest"):
        relative = _safe_relative_path(payload[field], field=field)
        bound_paths[field] = (path.parent / relative).resolve()
        if not bound_paths[field].is_file():
            raise FileNotFoundError(bound_paths[field])
    if sha256_file(bound_paths["source_matrix"]) != payload.get("source_matrix_sha256"):
        raise ValueError("source matrix SHA-256 mismatch")
    if sha256_file(bound_paths["archive"]) != payload.get("archive_sha256"):
        raise ValueError("archive SHA-256 mismatch")
    if sha256_file(bound_paths["archive_manifest"]) != payload.get(
        "archive_manifest_sha256"
    ):
        raise ValueError("archive manifest SHA-256 mismatch")
    archive_audit = validate_archive(
        bound_paths["archive"],
        bound_paths["archive_manifest"],
    )
    if not archive_audit["passed"]:
        raise ValueError(f"archive validation failed: {archive_audit['errors']}")

    source = read_json(bound_paths["source_matrix"])
    source_episodes = source.get("episodes")
    derived_episodes = payload.get("episodes")
    if not isinstance(source_episodes, list) or not isinstance(derived_episodes, list):
        raise ValueError("source and archive matrices require episode lists")
    if len(source_episodes) != len(derived_episodes):
        raise ValueError("archive matrix episode count differs from source matrix")
    sidecar = read_json(bound_paths["archive_manifest"])
    member_names = {
        validate_member_name(str(record["path"]))
        for record in sidecar.get("members", [])
    }
    replacements = dict(zip(DIRECT_EPISODE_ARTIFACT_FIELDS, ARCHIVE_EPISODE_ARTIFACT_FIELDS))
    for source_entry, derived_entry in zip(source_episodes, derived_episodes, strict=True):
        if not isinstance(source_entry, dict) or not isinstance(derived_entry, dict):
            raise ValueError("matrix episode must be an object")
        if any(field in source_entry for field in ARCHIVE_EPISODE_ARTIFACT_FIELDS):
            raise ValueError("bound source matrix is not direct")
        if any(field in derived_entry for field in DIRECT_EPISODE_ARTIFACT_FIELDS):
            raise ValueError("archive matrix retains loose raw artifact references")
        expected_non_artifacts = {
            key: value
            for key, value in source_entry.items()
            if key not in DIRECT_EPISODE_ARTIFACT_FIELDS
        }
        actual_non_artifacts = {
            key: value
            for key, value in derived_entry.items()
            if key not in ARCHIVE_EPISODE_ARTIFACT_FIELDS
        }
        if actual_non_artifacts != expected_non_artifacts:
            raise ValueError("archive episode metadata differs from source matrix")
        for archive_field in replacements.values():
            value = derived_entry.get(archive_field)
            name = validate_member_name(value)
            if name not in member_names:
                raise ValueError(f"archive episode member absent from sidecar: {name}")
        for direct_field, archive_field in replacements.items():
            direct_name = _safe_relative_path(
                source_entry.get(direct_field),
                field=direct_field,
            ).as_posix()
            archive_name = str(derived_entry[archive_field])
            if not (
                archive_name == direct_name
                or archive_name.endswith(f"/{direct_name}")
                or direct_name.endswith(f"/{archive_name}")
            ):
                raise ValueError(
                    f"archive member does not correspond to source artifact: {direct_field}"
                )

    source_top = {key: value for key, value in source.items() if key != "episodes"}
    derived_top = {
        key: value
        for key, value in payload.items()
        if key != "episodes" and key not in ARCHIVE_MATRIX_BINDING_FIELDS
    }
    if derived_top != source_top:
        raise ValueError("archive matrix metadata differs from source matrix")
    return {
        "archive_path": bound_paths["archive"],
        "archive_manifest_path": bound_paths["archive_manifest"],
        "source_matrix_path": bound_paths["source_matrix"],
        "member_names": frozenset(member_names),
    }
