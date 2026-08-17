from __future__ import annotations

import json

import pytest

from actionstream.m5_protocol import (
    build_seed_manifests,
    validate_disjoint,
    validate_manifest,
    write_seed_manifests,
)


def test_seed_manifests_are_deterministic_disjoint_and_sized() -> None:
    first = build_seed_manifests()
    second = build_seed_manifests()
    assert first == second
    assert len(first["calibration_seed_manifest.json"]["pairs"]) == 5
    assert len(first["no_shift_seed_manifest.json"]["pairs"]) == 10
    assert len(first["sealed_seed_manifest.json"]["pairs"]) == 30
    validate_disjoint(list(first.values()))


def test_seed_manifest_hash_detects_changes() -> None:
    manifest = build_seed_manifests()["calibration_seed_manifest.json"]
    manifest["pairs"][0]["seed"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_manifest(manifest)


def test_existing_seed_manifest_cannot_be_changed(tmp_path) -> None:
    write_seed_manifests(tmp_path)
    path = tmp_path / "calibration_seed_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        write_seed_manifests(tmp_path)
