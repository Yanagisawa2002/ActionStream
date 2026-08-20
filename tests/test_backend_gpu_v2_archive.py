from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from actionstream.backend_gpu_v2_archive import (
    _resolve_recorded_path,
    audit_root,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _fixture(root: Path) -> Path:
    root.mkdir()
    trace = root / "traces" / "episode.json"
    trace.parent.mkdir()
    _write_json(trace, {"trace": [0, 1]})
    telemetry = root / "telemetry" / "events.jsonl"
    telemetry.parent.mkdir()
    event = {
        "schema_version": 1,
        "engine_id": "engine",
        "event": "started",
        "utc_unix_ns": 1,
        "monotonic_ns": 2,
        "pid": 3,
        "thread": "worker",
    }
    _write_json(telemetry, event)
    rows = [
        {
            "trace_path": f"/old/server/results/{root.name}/traces/episode.json",
            "trace_sha256": _sha256(trace),
            "video_path": None,
            "video_sha256": None,
            "telemetry_jsonl_path": (
                f"/old/server/results/{root.name}/telemetry/events.jsonl"
            ),
            "telemetry_jsonl_sha256": _sha256(telemetry),
        }
    ]
    rows_path = root / "episodes.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    _write_json(
        root / "orchestrator_receipt.json",
        {"record_count": 1, "episodes_sha256": _sha256(rows_path)},
    )
    artifacts = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write_json(root / "artifact_manifest.json", {"artifacts": artifacts})
    return root


def test_audit_root_resolves_relocated_absolute_paths(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "actionstream_backend_gpu_v2r1_holdout")

    result = audit_root(root)

    assert result["status"] == "pass"
    assert result["record_count"] == 1
    assert result["rebased_absolute_paths"] == {
        "trace_path": 1,
        "video_path": 0,
        "telemetry_jsonl_path": 1,
    }
    assert result["telemetry_events"] == 1
    assert result["manifest"]["valid"] is True


def test_rebase_rejects_a_path_without_one_root_anchor(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()

    with pytest.raises(ValueError, match="safely rebased"):
        _resolve_recorded_path("/old/server/different/file.json", root)
