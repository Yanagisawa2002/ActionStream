"""Replay current retained v3 evidence against its original committed sources.

The frozen verifier deliberately hashes source files. Replaying it against new
runtime code would falsely imply that historical evidence validates that code.
No report, expected hash, or frozen verifier is modified here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path("reports/completion_acceptance_v3_20260915")
SOURCE_COMMIT = "8aacc818458356b07d7a583ab17fbc54c7a4fad5"


def main():
    freeze = json.loads((ROOT / REPORT / "freeze.json").read_text())
    with tempfile.TemporaryDirectory(prefix="actionstream-frozen-v3-") as directory:
        target = Path(directory).resolve()
        shutil.copytree(ROOT / REPORT, target / REPORT)
        protocol = "configs/completion_acceptance_v3.json"
        (target / "configs").mkdir()
        shutil.copyfile(ROOT / protocol, target / protocol)
        for name, expected in freeze["source_sha256"].items():
            destination = (target / name).resolve()
            if not destination.is_relative_to(target):
                raise ValueError("Frozen source path escapes replay directory")
            content = subprocess.check_output(
                ["git", "show", f"{SOURCE_COMMIT}:{name}"], cwd=ROOT
            )
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("Original committed source hash mismatch: " + name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        print("Historical v3 source commit:", SOURCE_COMMIT, flush=True)
        return subprocess.run(
            [sys.executable, str(target / REPORT / "verify_evidence.py")],
            check=False,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
