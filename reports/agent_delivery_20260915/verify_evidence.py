"""Replay delivery receipts; optionally verify the full downloaded bundle."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def replay(bundle=None):
    manifest = read("evidence_manifest.json")
    for name, expected in manifest["files"].items():
        assert sha(HERE / name) == expected, name
    current = read("current-delivery.json")
    assert current["status"] == "PASS" and all(current["checks"].values())
    assert read("environment.json")["status"] == "PASS"
    for name, count in (("model-assets.json", 19), ("libero-assets.json", 585)):
        receipt = read(name)
        assert receipt["status"] == "PASS" and receipt["verified"] == count
    original = json.loads(
        (HERE.parent / "finite_agent_20260915/external_blobs.json").read_text()
    )
    restored = read("restored-verification.json")
    assert restored["status"] == "PASS" and restored["verified"] == len(original) == 21
    assert {row["path"] for row in restored["files"]} == set(original)
    for row in restored["files"]:
        expected = original[row["path"]]
        assert row["passed"]
        assert row["actual_bytes"] == row["expected_bytes"] == expected["bytes"]
        assert row["actual_sha256"] == row["expected_sha256"] == expected["sha256"]
    rebuilt = read("progress.json")
    assert rebuilt["completed"] == rebuilt["matched"] == 20
    for row in rebuilt["episodes"]:
        assert row["byte_identical"]
        assert (
            row["reconstructed"]
            == row["original"]
            == original[f"run/{row['id']}/trajectory.npz"]
        )
    backup = read("backup-verification.json")
    assert backup["status"] == "PASS" and not backup["errors"]
    assert backup["original_blobs_verified"] == 21
    assert backup["archive_sha256"] == manifest["archive"]["sha256"]
    full = "not_requested"
    if bundle is not None:
        files = read("delivery-files.json")
        assert (
            sha(bundle / "delivery-files.json")
            == manifest["files"]["delivery-files.json"]
        )
        for name, expected in files.items():
            path = bundle / name
            assert path.stat().st_size == expected["bytes"], name
            assert sha(path) == expected["sha256"], name
        assert len(files) == backup["verified_files"] == 282
        full = "PASS"
    return dict(
        status="PASS",
        original_blobs=21,
        reconstructed_trajectories=20,
        receipt_replay="PASS",
        full_bundle=full,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    print(json.dumps(replay(parser.parse_args().bundle), indent=2))
