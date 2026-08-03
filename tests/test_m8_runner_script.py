from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "m8_run_isaac.ps1"


def _source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_runner_guards_outputs_and_resamples_gpu_after_build() -> None:
    source = _source()
    output_guard = source.index("Require-PathAbsent -Path $replayPath")
    batch_validation = source.index("$currentStage = 'batch_source_validate_only'")
    first_gpu_gate = source.index("$currentStage = 'gpu_preflight_initial'")
    build = source.index("Invoke-PixiPowerShell -Command $buildCommand")
    second_gpu_gate = source.index("-Phase 'post_build_pre_launch'")
    router_start = source.index("$currentStage = 'router_start'")
    assert (
        output_guard
        < batch_validation
        < first_gpu_gate
        < build
        < second_gpu_gate
        < router_start
    )
    assert "Require-PathAbsent -Path $analysisPath" in source
    assert "Require-PathAbsent -Path $figureDirectory" in source
    assert source.count("Get-GpuPreflightSnapshot `") == 2
    assert "gpu_preflight_post_build" in source


def test_runner_validates_every_batch_from_current_source_before_gpu() -> None:
    source = _source()
    validation_stage = source.index("$currentStage = 'batch_source_validate_only'")
    validation_call = source.index(
        "python -m action_stream_isaac.dynamic_isaac_adapter --matrix-manifest "
    )
    gpu_stage = source.index("$currentStage = 'gpu_preflight_initial'")
    assert validation_stage < validation_call < gpu_stage
    assert "--expected-strategy '$escapedStrategy' --validate-only" in source
    assert "$sourceRoot 'action_stream_benchmark'" in source
    assert "$sourceRoot 'action_stream_isaac'" in source
    assert "$sourceRoot 'action_stream_policy'" in source
    assert '"$batchLogStem.validate_only.log"' in source
    assert "current_source_batch_validation_passed" in source
    assert "current_source_batch_validation_count" in source
    assert "current_source_batch_validation_logs" in source
    assert (
        "$plannedLogFiles = @('router.stdout.log', 'router.stderr.log') + $batchSourceValidationLogs"
        in source
    )


def test_runner_treats_pixi_exit_code_as_authoritative_despite_stderr_warnings() -> None:
    source = _source()
    invocation = source.index("& $PixiExe run --manifest-path $manifestPath")
    preference_relaxation = source.rindex(
        "$ErrorActionPreference = 'Continue'", 0, invocation
    )
    exit_capture = source.index("$exitCode = $LASTEXITCODE", invocation)
    preference_restore = source.index(
        "$ErrorActionPreference = $previousErrorActionPreference", exit_capture
    )
    assert preference_relaxation < invocation < exit_capture < preference_restore
    assert "if ($exitCode -ne 0)" in source[preference_restore:]


def test_runner_bounds_adapter_and_emits_failure_receipts() -> None:
    source = _source()
    assert "[ValidateRange(60, 86400)][int]$BatchTimeoutSeconds = 14400" in source
    assert "$adapterProcess.WaitForExit($BatchTimeoutSeconds * 1000)" in source
    assert "$adapterProcess.WaitForExit(5000)" in source
    assert not re.search(r"\$adapterProcess\.WaitForExit\(\s*\)", source)
    assert "[System.TimeoutException]::new" in source
    assert "failure_receipt.json" in source
    assert "timeout_receipt.json" in source
    assert "Write-NativeFailureReceipt" in source
    assert "available_logs = @(Get-AvailableLogRecords" in source
    assert "[System.IO.FileMode]::CreateNew" in source
    assert "[System.Text.UTF8Encoding]::new($false)" in source
    assert "Stop-OwnedProcessTree -RootProcessId $process.Id" in source
    assert "-not $process.HasExited" in source
    assert "-not $routerProcess.HasExited" in source


def test_runner_archives_logs_and_uses_local_overlay() -> None:
    source = _source()
    assert "install\\local_setup.ps1" in source
    assert "process_logs.tar.gz" in source
    assert "process_logs.manifest.json" in source
    assert "action_stream_benchmark.m8_cli archive" in source
    assert "action_stream_benchmark.m8_cli archive-validate" in source
    assert "process_log_archive_sha256" in source
    assert "process_log_manifest_sha256" in source
    assert "process_and_command_logs" not in source


def test_runner_parses_on_windows_powershell() -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    quoted = str(RUNNER).replace("'", "''")
    command = (
        "$tokens=$null;$errors=$null;"
        f"[System.Management.Automation.Language.Parser]::ParseFile('{quoted}',"
        "[ref]$tokens,[ref]$errors)|Out-Null;"
        "if($errors.Count -gt 0){$errors|ForEach-Object{$_.Message};exit 1}"
    )
    completed = subprocess.run(
        [powershell, "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
