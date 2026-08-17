<#
.SYNOPSIS
Runs the frozen M7 ROS 2/C++ deterministic test-plant holdout end to end.

.DESCRIPTION
This endpoint is `ros_cpp_test_plant`, not Isaac Sim. The script builds the
pinned ROS 2 Jazzy Docker image if its local tag is absent, creates a new
repository-relative output directory, runs all 216 paired holdout episodes,
independently replays them, analyzes them with 20,000 paired-bootstrap
resamples, and writes three figures as both 300-dpi PNG and vector PDF.

.EXAMPLE
.\scripts\m7_run_ros_holdout.ps1 `
    -OutputSubdirectory outputs/m7_g0/reproductions/ros_holdout_001
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputSubdirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$imageTag = 'actionstream-m7-ros:jazzy'
$repositoryRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot '..')
).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
$repositoryPrefix = $repositoryRoot + [System.IO.Path]::DirectorySeparatorChar

$trimmedOutput = $OutputSubdirectory.Trim()
if ($trimmedOutput -ne $OutputSubdirectory) {
    throw 'OutputSubdirectory must not contain leading or trailing whitespace.'
}
if ([System.IO.Path]::IsPathRooted($trimmedOutput)) {
    throw 'OutputSubdirectory must be repository-relative, not absolute.'
}

$normalizedOutput = $trimmedOutput.Replace('\', '/')
$safeOutputPattern =
    '^outputs/[A-Za-z0-9][A-Za-z0-9._-]*' +
    '(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$'
if ($normalizedOutput -notmatch $safeOutputPattern) {
    throw @"
OutputSubdirectory must name a child of outputs/ using only alphanumeric,
period, underscore, and hyphen characters in each path component.
"@
}
$segments = $normalizedOutput.Split('/')
if ($segments -contains '.' -or $segments -contains '..') {
    throw 'OutputSubdirectory must not contain dot or parent traversal segments.'
}

$windowsRelativeOutput = $normalizedOutput.Replace(
    '/', [System.IO.Path]::DirectorySeparatorChar
)
$outputPath = [System.IO.Path]::GetFullPath(
    (Join-Path $repositoryRoot $windowsRelativeOutput)
)
if (-not $outputPath.StartsWith(
        $repositoryPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
    throw 'Resolved output path escapes the ActionStream repository.'
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Refusing to overwrite existing output path: $outputPath"
}

$requiredRelativePaths = @(
    'docker\m7_ros_jazzy\Dockerfile',
    'configs\m7_development_seeds.json',
    'configs\m7_holdout_seeds.json',
    'ros2_ws\src'
)
foreach ($relativePath in $requiredRelativePaths) {
    $requiredPath = Join-Path $repositoryRoot $relativePath
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required frozen input was not found: $requiredPath"
    }
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker is required but was not found on PATH.'
}

Write-Host 'M7 endpoint: ros_cpp_test_plant (not Isaac Sim)'
Write-Host "Pinned image tag: $imageTag"

& docker image inspect $imageTag *> $null
if ($LASTEXITCODE -ne 0) {
    $dockerContext = Join-Path $repositoryRoot 'docker\m7_ros_jazzy'
    Write-Host "Pinned image is absent; building $imageTag"
    & docker build --tag $imageTag $dockerContext
    if ($LASTEXITCODE -ne 0) {
        throw "Docker image build failed with exit code $LASTEXITCODE."
    }
}
else {
    Write-Host "Using existing pinned image tag: $imageTag"
}

# Reserve a new output directory after the image preflight. A failed run keeps
# its partial evidence and the path cannot be reused accidentally.
$outputParent = Split-Path -Parent $outputPath
if (-not (Test-Path -LiteralPath $outputParent)) {
    New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
}
if (Test-Path -LiteralPath $outputPath) {
    throw "Output path appeared during preflight; refusing reuse: $outputPath"
}
New-Item -ItemType Directory -Path $outputPath | Out-Null

$containerOutput = "/workspace/$normalizedOutput"
$containerTemplate = @'
set -euo pipefail

OUTPUT_ROOT='__OUTPUT_ROOT__'
BUILD_BASE='/tmp/actionstream_m7_build'
INSTALL_BASE='/tmp/actionstream_m7_install'
LOG_BASE='/tmp/actionstream_m7_log'

source /opt/ros/jazzy/setup.bash
cd /workspace/ros2_ws
colcon --log-base "$LOG_BASE" build \
  --base-paths /workspace/ros2_ws/src \
  --build-base "$BUILD_BASE" \
  --install-base "$INSTALL_BASE" \
  --merge-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source "$INSTALL_BASE/setup.bash"
cd /workspace

# Formal frozen matrix: 36 seeds x 2 profiles x 3 strategies = 216 episodes.
# Deliberately no overwrite option is passed.
ros2 run action_stream_benchmark action-stream-benchmark ros-run \
  --split holdout \
  --seed-file /workspace/configs/m7_holdout_seeds.json \
  --comparison-seed-file /workspace/configs/m7_development_seeds.json \
  --profiles profile_a,profile_b \
  --strategies sync_hold,naive_async,aligned_async \
  --output-root "$OUTPUT_ROOT" \
  --request-count 512 \
  --max-steps 180 \
  --request-interval-steps 10 \
  --timeout-seconds 120

ros2 run action_stream_benchmark action-stream-benchmark validate \
  --manifest "$OUTPUT_ROOT/manifest.json" \
  --output "$OUTPUT_ROOT/replay_validation.json"

ros2 run action_stream_benchmark action-stream-benchmark analyze \
  --manifest "$OUTPUT_ROOT/manifest.json" \
  --output "$OUTPUT_ROOT/analysis.json" \
  --bootstrap-resamples 20000

EXAMPLE_EVENT_LOG="$OUTPUT_ROOT/episodes/profile_b/seed_2026080434/aligned_async.events.jsonl"
test -s "$EXAMPLE_EVENT_LOG"
ros2 run action_stream_benchmark action-stream-benchmark figures \
  --analysis "$OUTPUT_ROOT/analysis.json" \
  --example-event-log "$EXAMPLE_EVENT_LOG" \
  --output-dir "$OUTPUT_ROOT/figures"

python3 - "$OUTPUT_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("evidence_class") != "ros_cpp_test_plant":
    raise SystemExit("manifest endpoint is not ros_cpp_test_plant")
if manifest.get("split") != "holdout":
    raise SystemExit("manifest split is not holdout")
if len(manifest.get("episodes", [])) != 216:
    raise SystemExit("manifest does not contain exactly 216 episodes")

figure_names = (
    "m7_success_comparison.png",
    "m7_success_comparison.pdf",
    "m7_efficiency_comparison.png",
    "m7_efficiency_comparison.pdf",
    "m7_async_timeline.png",
    "m7_async_timeline.pdf",
)
missing = [
    name
    for name in figure_names
    if not (root / "figures" / name).is_file()
    or (root / "figures" / name).stat().st_size == 0
]
if missing:
    raise SystemExit(f"missing or empty figure outputs: {missing}")
print("Verified ros_cpp_test_plant holdout: 216 episodes, 6 figures")
PY
'@
$containerCommand = $containerTemplate.Replace(
    '__OUTPUT_ROOT__', $containerOutput
)
$volumeArgument = "${repositoryRoot}:/workspace"
$dockerArguments = @(
    'run',
    '--rm',
    '--volume', $volumeArgument,
    '--workdir', '/workspace',
    $imageTag,
    'bash', '-lc', $containerCommand
)

try {
    & docker @dockerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker holdout exited with code $LASTEXITCODE."
    }
}
catch {
    Write-Warning "Run failed; partial evidence is preserved at $outputPath"
    throw
}

Write-Host 'Completed endpoint: ros_cpp_test_plant (not Isaac Sim)'
Write-Host "Artifacts: $outputPath"
