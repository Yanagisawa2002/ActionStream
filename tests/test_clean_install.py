from __future__ import annotations

import os
import shutil
import subprocess


def test_real_lerobot_rollout_cli_discovers_actionstream_plugin() -> None:
    executable = shutil.which("lerobot-rollout")
    assert executable is not None
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        [executable, "--help"],
        capture_output=True,
        check=False,
        env=environment,
        encoding="utf-8",
        errors="replace",
        text=True,
        timeout=60,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    assert completed.returncode == 0, output
    assert "actionstream" in output.lower(), output
