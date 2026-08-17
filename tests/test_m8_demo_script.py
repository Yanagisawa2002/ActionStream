from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "m8_record_demo.ps1"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_demo_consumes_exact_single_matrix_and_validated_freeze() -> None:
    source = _source()
    assert "$DevelopmentMatrixManifest" in source
    assert "$DevelopmentBatchManifest" not in source
    assert "[Parameter(Mandatory = $true)][string]$FreezeManifest" in source
    assert "matrix-create" in source and "--strategy aligned_async" in source
    assert "$matrix.expected_episode_count -ne 1" in source
    assert "$matrix.persistent_batch_count -ne 1" in source
    assert "$matrixEpisode.profile_id -ne 'profile_1_fixed'" in source
    assert "$matrixEpisode.strategy -ne 'aligned_async'" in source
    assert "configs\\m8_demo_seed.json" in source
    assert "selected_candidate_protocol_role" in source
    assert "selected_candidate_protocol_path" in source
    assert "selected_candidate_protocol_file_sha256" in source
    assert "$matrix.protocol_sha256 -ne $developmentCalibration.selected_protocol_sha256" in source
    assert "$selectedCandidateId -eq 'candidate_0'" in source
    assert '"development:$($selectedCandidateId):protocol"' in source
    assert "profile:profile_1_fixed" in source
    assert "development_seeds" in source

    freeze = source.index("freeze-validate")
    initial_gpu = source.index("-Phase 'initial_pre_build'")
    build = source.index("Invoke-PixiPowerShell -Command $buildCommand")
    post_build_gpu = source.index("-Phase 'post_build_pre_launch'")
    adapter = source.index("python -m action_stream_isaac.dynamic_isaac_adapter")
    assert freeze < initial_gpu < build < post_build_gpu < adapter


def test_demo_evidence_is_separate_portable_and_non_headline() -> None:
    source = _source()
    assert "Join-Path $evidenceRoot 'native_demo_logs'" in source
    assert "Join-Path $evidenceRoot 'replay_validation.json'" in source
    assert "Join-Path $evidenceRoot 'demo_receipt.json'" in source
    assert "release/m8_g0/media/" in source
    assert "${videoStem}_native_logs" not in source
    assert "headline_eligible = $false" in source
    assert "benchmark_evidence = $false" in source
    assert "full_matrix_replay = $true" in source
    assert "--manifest '$escapedMatrix'" in source
    assert "[System.Text.UTF8Encoding]::new($false)" in source
    assert "[System.IO.FileMode]::CreateNew" in source
    assert "Get-RepositoryReference" in source
    assert "<repository_root>" in source
    assert "<isaac_workspace>" in source


def test_demo_binds_build_runtime_and_never_kills_unowned_processes() -> None:
    source = _source()
    for expected in (
        "install\\local_setup.ps1",
        "installed_dynamic_adapter_sha256",
        "source_manifest_sha256",
        "executor_sha256",
        "router_sha256",
        "runner_sha256",
        "ffprobe_sha256",
        "ffmpeg_sha256",
        "colcon_merge_install = $true",
    ):
        assert expected in source
    assert source.count("Get-GpuPreflightSnapshot") >= 3  # definition plus two calls
    assert "Stop-OwnedProcessTree -RootProcessId $process.Id" in source
    assert "unrelated_processes_killed = 0" in source
    assert not re.search(r"Get-Process\s*\|.*Stop-Process", source)
    assert "taskkill" not in source.lower()


def test_demo_bounds_adapter_and_emits_failure_or_refusal_receipts() -> None:
    source = _source()
    assert "[ValidateRange(60, 7200)][int]$AdapterTimeoutSeconds = 1800" in source
    assert "$adapterProcess.WaitForExit($AdapterTimeoutSeconds * 1000)" in source
    assert "$adapterProcess.WaitForExit(5000)" in source
    assert not re.search(r"\$adapterProcess\.WaitForExit\(\s*\)", source)
    assert "failure_receipt.json" in source
    assert "timeout_receipt.json" in source
    assert "gpu_refusal_receipt.json" in source
    assert "Write-DemoFailureReceipt" in source
    assert "Write-GpuRefusalReceipt" in source
    assert "refused_without_killing_any_process" in source


def test_demo_verifies_video_content_before_promotion() -> None:
    source = _source()
    verify = source.index("$currentStage = 'video_verification'")
    promote = source.index("$currentStage = 'artifact_promotion'")
    receipt = source.index("$currentStage = 'completion_receipt'")
    assert verify < promote < receipt
    for expected in (
        "ffprobe.exe",
        "ffmpeg.exe",
        "codec_name",
        "nb_read_frames",
        "$videoStream.codec_name -ne 'h264'",
        "$videoStream.width -ne 1280",
        "$videoStream.height -ne 720",
        "[math]::Abs($fps - 20.0)",
        "$frameCount -lt 40",
        "$durationSeconds -lt 2.0",
        "label = 'first'",
        "label = 'middle'",
        "label = 'last'",
        "$distinctFrameHashes -lt 2",
    ):
        assert expected in source
    assert "[System.IO.File]::Move($stagedVideoPath, $videoPath)" in source
    assert "event_log_sha256" in source
    assert "scenario_declared_sha256" in source
    assert "fault_trace_declared_sha256" in source
    assert "task_success = [bool]$summary.metrics.task_success" in source


def test_demo_rolls_back_partial_multi_artifact_promotion() -> None:
    source = _source()
    assert "$videoPromoted = $false" in source
    assert "$replayPromoted = $false" in source

    video_move = source.index("[System.IO.File]::Move($stagedVideoPath, $videoPath)")
    video_flag = source.index("$videoPromoted = $true", video_move)
    replay_move = source.index("[System.IO.File]::Move($stagedReplayPath, $replayPath)")
    replay_flag = source.index("$replayPromoted = $true", replay_move)
    assert video_move < video_flag < replay_move < replay_flag

    catch = source.index("catch {\n    $originalError = $_")
    receipt_preserve = source.index("incomplete_demo_receipt.json", catch)
    replay_rollback = source.index("[System.IO.File]::Move($replayPath, $stagedReplayPath)", catch)
    video_rollback = source.index("[System.IO.File]::Move($videoPath, $stagedVideoPath)", catch)
    assert receipt_preserve < replay_rollback < video_rollback
    for expected in (
        "$replayPromoted -and",
        "$videoPromoted -and",
        "-not (Test-Path -LiteralPath $stagedReplayPath)",
        "-not (Test-Path -LiteralPath $stagedVideoPath)",
        "Could not preserve the partial demo receipt",
        "Could not roll back the promoted replay",
        "Could not roll back the promoted video",
        "throw $originalError",
    ):
        assert expected in source


def test_demo_script_parses_on_windows_powershell() -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    quoted = str(SCRIPT).replace("'", "''")
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
