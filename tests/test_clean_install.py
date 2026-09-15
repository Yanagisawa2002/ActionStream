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


def test_installed_finite_agent_cli_is_available_without_model_assets() -> None:
    executable = shutil.which("actionstream-agent")
    assert executable is not None
    completed = subprocess.run(
        [executable, "--help"],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "--request" in output and "--checkpoint" in output, output
