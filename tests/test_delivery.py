from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from actionstream import delivery, libero_config


def manifest(data=b"frozen", name="nested/model.bin"):
    return {
        "files": [
            {
                "destination": name,
                "file": name,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "repo": "owner/model",
                "revision": "a" * 40,
            }
        ]
    }


def test_verification_requires_actual_bytes_and_never_repairs(tmp_path):
    pinned = manifest()
    assert delivery.verify(pinned, tmp_path)["verified"] == 0
    file = tmp_path / "nested/model.bin"
    file.parent.mkdir()
    file.write_bytes(b"broken")
    assert delivery.verify(pinned, tmp_path)["status"] == "FAIL"
    assert file.read_bytes() == b"broken"
    file.write_bytes(b"frozen")
    assert delivery.verify(pinned, tmp_path)["status"] == "PASS"


@pytest.mark.parametrize("name", ["../escape", "nested/../../escape", "/absolute"])
def test_paths_cannot_escape_root(tmp_path, name):
    with pytest.raises(ValueError, match="inside"):
        delivery.verify(manifest(name=name), tmp_path)


def test_empty_and_duplicate_manifests_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        delivery.verify({"files": []}, tmp_path)
    pinned = manifest()
    pinned["files"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        delivery.verify(pinned, tmp_path)


def test_external_evidence_dictionary_is_supported(tmp_path):
    (tmp_path / "frozen.pt").write_bytes(b"frozen")
    pinned = {
        "frozen.pt": {"bytes": 6, "sha256": hashlib.sha256(b"frozen").hexdigest()}
    }
    assert delivery.verify(pinned, tmp_path)["verified"] == 1
    with pytest.raises(ValueError, match="owner-provided"):
        delivery.materialize(pinned, tmp_path, tmp_path / "cache")


def test_fetch_checks_all_destinations_before_network_or_writes(tmp_path, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Download must not start before validating the full manifest")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", forbidden)
    pinned = manifest()
    pinned["files"] += manifest(name="../escape")["files"]
    with pytest.raises(ValueError):
        delivery.materialize(pinned, tmp_path / "assets", tmp_path / "cache")
    assert not (tmp_path / "assets").exists()


def test_fetch_verifies_content_then_atomic_copy_and_reuses_offline(
    tmp_path, monkeypatch
):
    downloaded = tmp_path / "download"
    downloaded.write_bytes(b"broken")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return downloaded

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    root = tmp_path / "assets"
    with pytest.raises(ValueError, match="pinned content"):
        delivery.materialize(manifest(), root, tmp_path / "cache")
    assert not root.exists()
    downloaded.write_bytes(b"frozen")
    assert (
        delivery.materialize(manifest(), root, tmp_path / "cache")["status"] == "PASS"
    )
    assert calls[-1]["revision"] == "a" * 40
    before = len(calls)
    assert delivery.materialize(manifest(), root, tmp_path / "cache")["verified"] == 1
    assert len(calls) == before


def test_doctor_reports_runtime_drift(monkeypatch):
    monkeypatch.setattr(delivery.importlib.metadata, "version", lambda name: "0.0.0")
    result = delivery.environment(require_simulator=False)
    assert result["status"] == "FAIL"
    assert any("torch: expected 2.8.0" in error for error in result["errors"])


def test_cli_missing_evidence_returns_failure_receipt(tmp_path):
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest()))
    receipt = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "actionstream.delivery",
            "verify",
            "--manifest",
            str(source),
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(receipt),
        ],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert json.loads(receipt.read_text())["verified"] == 0


def test_explicit_libero_assets_survive_backend_reinitialization(tmp_path, monkeypatch):
    benchmark = tmp_path / "package"
    assets = tmp_path / "verified-assets"
    assets.mkdir()
    monkeypatch.setattr(libero_config, "_canonical_benchmark_root", lambda: benchmark)
    monkeypatch.delenv("ACTIONSTREAM_LIBERO_ASSETS", raising=False)
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    module = SimpleNamespace(_assets_path_cache=None)
    monkeypatch.setattr(libero_config.importlib, "import_module", lambda name: module)
    first = libero_config.ensure_isolated_libero_config(
        tmp_path / "isolated", assets_dir=assets
    )
    second = libero_config.ensure_isolated_libero_config()
    assert first == second
    assert yaml.safe_load(second.read_text())["assets"] == str(assets.resolve())
    assert module._assets_path_cache == str(assets.resolve())


def test_libero_refuses_missing_assets_and_foreign_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        libero_config, "_canonical_benchmark_root", lambda: tmp_path / "package"
    )
    monkeypatch.delenv("ACTIONSTREAM_LIBERO_ASSETS", raising=False)
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    with pytest.raises(RuntimeError, match="missing"):
        libero_config.ensure_isolated_libero_config(
            tmp_path / "config", assets_dir=tmp_path / "absent"
        )
    assets = tmp_path / "assets"
    assets.mkdir()
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.yaml").write_text("user-owned\n")
    with pytest.raises(RuntimeError, match="non-ActionStream"):
        libero_config.ensure_isolated_libero_config(config, assets_dir=assets)
    assert (config / "config.yaml").read_text() == "user-owned\n"
