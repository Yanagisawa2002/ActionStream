<#
.SYNOPSIS
Builds the current checkout and runs M8 persistent native-Isaac batches.

.DESCRIPTION
The audited Windows Isaac environment cannot safely import the standalone ROS
CLI before Kit.  This runner therefore starts its owned Zenoh router and the
installed C++ executor directly, then starts the Kit-first Python adapter.  It
never substitutes the deterministic test plant.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$MatrixSuiteManifest,
    [Parameter(Mandatory = $true)][string]$IsaacWorkspace,
    [string]$PixiExe = '',
    [bool]$Headless = $true,
    [ValidateRange(60, 86400)][int]$BatchTimeoutSeconds = 14400,
    [switch]$AuthorizeNativeGpuRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Require-File {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }
}

function Write-Utf8JsonNoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$InputObject,
        [int]$Depth = 8
    )
    $json = $InputObject | ConvertTo-Json -Depth $Depth
    $encoding = [System.Text.UTF8Encoding]::new($false)
    $bytes = $encoding.GetBytes($json + [Environment]::NewLine)
    $stream = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try {
        $stream.Write($bytes, 0, $bytes.Length)
    }
    finally {
        $stream.Dispose()
    }
}

function Write-Utf8LinesNoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$Lines
    )
    $text = ($Lines -join [Environment]::NewLine) + [Environment]::NewLine
    $encoding = [System.Text.UTF8Encoding]::new($false)
    $bytes = $encoding.GetBytes($text)
    $stream = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try {
        $stream.Write($bytes, 0, $bytes.Length)
    }
    finally {
        $stream.Dispose()
    }
}

function Require-PathAbsent {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to overwrite existing $Description`: $Path"
    }
}

function Get-GpuPreflightSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$NvidiaSmiPath,
        [Parameter(Mandatory = $true)][int]$MemoryThresholdMiB,
        [Parameter(Mandatory = $true)][string]$Phase
    )
    $gpuLines = @(
        & $NvidiaSmiPath --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
    )
    if ($LASTEXITCODE -ne 0 -or $gpuLines.Count -lt 1) {
        throw "nvidia-smi GPU inventory failed during $Phase."
    }
    $gpuInventory = @()
    foreach ($line in $gpuLines) {
        $parts = @($line -split ',' | ForEach-Object { $_.Trim() })
        if ($parts.Count -ne 7) { throw "Unexpected nvidia-smi GPU row during $Phase`: $line" }
        $gpuInventory += [pscustomobject][ordered]@{
            index = [int]$parts[0]
            name = $parts[1]
            uuid = $parts[2]
            driver_version = $parts[3]
            memory_total_mib = [int]$parts[4]
            memory_used_mib = [int]$parts[5]
            utilization_gpu_percent = [int]$parts[6]
        }
    }

    $computeLines = @(
        & $NvidiaSmiPath --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>$null
    )
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi compute-process query failed during $Phase." }
    $computeProcessInventory = @()
    foreach ($line in $computeLines) {
        $parts = @($line -split ',' | ForEach-Object { $_.Trim() })
        if ($parts.Count -eq 4 -and $parts[1] -match '^\d+$') {
            $usedMemory = if ($parts[3] -match '^\d+$') { [int]$parts[3] } else { $null }
            $computeProcessInventory += [pscustomobject][ordered]@{
                gpu_uuid = $parts[0]
                process_id = [int]$parts[1]
                process_name = $parts[2]
                used_memory_mib = $usedMemory
                actionable_compute_allocation = ($null -ne $usedMemory -and $usedMemory -gt 0)
            }
        }
    }
    $actionableComputeProcesses = @(
        $computeProcessInventory |
            Where-Object { $_.actionable_compute_allocation }
    )
    $occupiedGpus = @(
        $gpuInventory |
            Where-Object { $_.memory_used_mib -gt $MemoryThresholdMiB }
    )
    return [pscustomobject][ordered]@{
        phase = $Phase
        captured_utc = [DateTime]::UtcNow.ToString('o')
        gpu_inventory = $gpuInventory
        reported_compute_processes = $computeProcessInventory
        actionable_compute_processes = $actionableComputeProcesses
        occupied_gpus = $occupiedGpus
        passed = ($actionableComputeProcesses.Count -eq 0 -and $occupiedGpus.Count -eq 0)
    }
}

function Write-GpuRefusalReceipt {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Snapshot,
        [Parameter(Mandatory = $true)][int]$MemoryThresholdMiB
    )
    $payload = [ordered]@{
        schema_version = 1
        milestone = 'M8-G0'
        receipt_kind = 'gpu_preflight_refusal'
        created_utc = [DateTime]::UtcNow.ToString('o')
        operator_authorized_native_gpu_run = $true
        preflight_phase = $Snapshot.phase
        threshold_mib = $MemoryThresholdMiB
        gpu_inventory = $Snapshot.gpu_inventory
        reported_compute_processes = $Snapshot.reported_compute_processes
        actionable_compute_processes = $Snapshot.actionable_compute_processes
        occupied_gpus = $Snapshot.occupied_gpus
        action = 'refused_without_killing_any_process'
    }
    Write-Utf8JsonNoBom -Path $Path -InputObject $payload -Depth 8
    return $payload
}

function Get-ContainedRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )
    $baseFull = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\', '/')
    $targetFull = [System.IO.Path]::GetFullPath($TargetPath)
    if ($targetFull.Equals($baseFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        return '.'
    }
    $prefix = $baseFull + [System.IO.Path]::DirectorySeparatorChar
    if (-not $targetFull.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside the declared base directory: $targetFull (base $baseFull)"
    }
    return $targetFull.Substring($prefix.Length).Replace('\', '/')
}

function Get-RepositorySafeRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BaseDirectory,
        [Parameter(Mandatory = $true)][string]$TargetPath,
        [Parameter(Mandatory = $true)][string]$RepositoryRoot
    )
    $rootFull = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\', '/')
    $rootPrefix = $rootFull + [System.IO.Path]::DirectorySeparatorChar
    $baseFull = [System.IO.Path]::GetFullPath($BaseDirectory).TrimEnd('\', '/')
    $targetFull = [System.IO.Path]::GetFullPath($TargetPath)
    foreach ($candidate in @($baseFull, $targetFull)) {
        if (
            -not $candidate.Equals($rootFull, [System.StringComparison]::OrdinalIgnoreCase) -and
            -not $candidate.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)
        ) {
            throw "Relative-path endpoint escapes the repository: $candidate"
        }
    }
    $baseUri = [System.Uri]($baseFull + [System.IO.Path]::DirectorySeparatorChar)
    $targetUri = [System.Uri]$targetFull
    if ($baseUri.Scheme -ne $targetUri.Scheme) {
        throw "Cannot form a relative path across URI schemes: $baseUri and $targetUri"
    }
    return [System.Uri]::UnescapeDataString(
        $baseUri.MakeRelativeUri($targetUri).ToString()
    ).Replace('\', '/')
}

function Get-AvailableLogRecords {
    param([Parameter(Mandatory = $true)][string]$LogDirectory)
    if (-not (Test-Path -LiteralPath $LogDirectory -PathType Container)) {
        return @()
    }
    return @(
        Get-ChildItem -LiteralPath $LogDirectory -File -Filter '*.log' -ErrorAction SilentlyContinue |
            Sort-Object Name |
            ForEach-Object {
                [ordered]@{
                    path = Get-ContainedRelativePath -BasePath $LogDirectory -TargetPath $_.FullName
                    size_bytes = $_.Length
                    sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                }
            }
    )
}

function Write-NativeFailureReceipt {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$LogDirectory,
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)]$ErrorRecord,
        [Parameter(Mandatory = $true)][string]$SuitePath,
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][int]$BatchTimeoutSeconds,
        [string]$CurrentBatch = '',
        [string]$PreflightReceiptPath = '',
        [string]$SourceManifestSha256 = '',
        $InitialGpuPreflight = $null,
        $PostBuildGpuPreflight = $null
    )
    $exception = $ErrorRecord.Exception
    $timedOut = $exception -is [System.TimeoutException]
    $suiteReference = $SuitePath
    try {
        $suiteReference = Get-ContainedRelativePath -BasePath $RepositoryRoot -TargetPath $SuitePath
    }
    catch {
        # A failure receipt must remain writable even if the supplied suite
        # path is itself outside the repository and caused the failure.
    }
    $preflightReference = $null
    $preflightSha256 = $null
    if (
        -not [string]::IsNullOrWhiteSpace($PreflightReceiptPath) -and
        (Test-Path -LiteralPath $PreflightReceiptPath -PathType Leaf)
    ) {
        $preflightReference = Get-ContainedRelativePath -BasePath $LogDirectory -TargetPath $PreflightReceiptPath
        $preflightSha256 = (Get-FileHash -LiteralPath $PreflightReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $payload = [ordered]@{
        schema_version = 1
        milestone = 'M8-G0'
        receipt_kind = if ($timedOut) { 'native_timeout' } else { 'native_failure' }
        status = if ($timedOut) { 'timeout' } else { 'failed' }
        created_utc = [DateTime]::UtcNow.ToString('o')
        stage = $Stage
        current_batch = if ([string]::IsNullOrWhiteSpace($CurrentBatch)) { $null } else { $CurrentBatch }
        batch_timeout_seconds = $BatchTimeoutSeconds
        suite_manifest = $suiteReference
        suite_manifest_sha256 = if (Test-Path -LiteralPath $SuitePath -PathType Leaf) {
            (Get-FileHash -LiteralPath $SuitePath -Algorithm SHA256).Hash.ToLowerInvariant()
        } else { $null }
        source_manifest_sha256 = if ([string]::IsNullOrWhiteSpace($SourceManifestSha256)) { $null } else { $SourceManifestSha256 }
        preflight_receipt = $preflightReference
        preflight_receipt_sha256 = $preflightSha256
        initial_gpu_preflight = $InitialGpuPreflight
        post_build_gpu_preflight = $PostBuildGpuPreflight
        error = [ordered]@{
            type = $exception.GetType().FullName
            message = $exception.Message
            script_stack_trace = $ErrorRecord.ScriptStackTrace
        }
        available_logs = @(Get-AvailableLogRecords -LogDirectory $LogDirectory)
        action = 'stopped_only_owned_process_trees_and_preserved_available_logs'
    }
    Write-Utf8JsonNoBom -Path $Path -InputObject $payload -Depth 10
}

function Stop-OwnedProcessTree {
    param([Parameter(Mandatory = $true)][int]$RootProcessId)
    $snapshot = @(Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId)
    $frontier = [System.Collections.Generic.Queue[int]]::new()
    $descendants = [System.Collections.Generic.List[int]]::new()
    $frontier.Enqueue($RootProcessId)
    while ($frontier.Count -gt 0) {
        $parentId = $frontier.Dequeue()
        foreach ($child in @($snapshot | Where-Object { $_.ParentProcessId -eq $parentId })) {
            $childId = [int]$child.ProcessId
            $descendants.Add($childId)
            $frontier.Enqueue($childId)
        }
    }
    $targets = @($descendants.ToArray())
    [array]::Reverse($targets)
    $targets += $RootProcessId
    foreach ($targetId in $targets) {
        Stop-Process -Id $targetId -ErrorAction SilentlyContinue
    }
}

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$sourceRoot = Join-Path $repositoryRoot 'ros2_ws\src'
$suitePath = [System.IO.Path]::GetFullPath($MatrixSuiteManifest)
$workspacePath = [System.IO.Path]::GetFullPath($IsaacWorkspace)
$manifestPath = Join-Path $workspacePath 'pixi.toml'
Require-File -Path $suitePath -Description 'M8 matrix suite'
Require-File -Path $manifestPath -Description 'official Isaac Pixi manifest'
if (-not $AuthorizeNativeGpuRun) {
    throw 'Native Isaac execution is fail-closed; rerun with -AuthorizeNativeGpuRun after reviewing the GPU preflight contract.'
}
if ([string]::IsNullOrWhiteSpace($PixiExe)) {
    $pixiCommand = Get-Command pixi -ErrorAction SilentlyContinue
    if ($null -eq $pixiCommand) { throw 'Pixi was not found on PATH; pass -PixiExe.' }
    $PixiExe = $pixiCommand.Source
}
Require-File -Path $PixiExe -Description 'Pixi executable'

function Invoke-PixiPowerShell {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [string]$LogPath = ''
    )
    # Pixi writes non-fatal manifest warnings to stderr.  Under the runner's
    # fail-closed ErrorActionPreference those native stderr records would
    # otherwise terminate this function before LASTEXITCODE can be checked.
    # Treat the process exit code as authoritative while still preserving the
    # combined stream in the immutable command log.
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $captured = @(
            & $PixiExe run --manifest-path $manifestPath -- powershell.exe -NoProfile -ExecutionPolicy Bypass -Command $Command 2>&1
        )
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
        if (Test-Path -LiteralPath $LogPath) { throw "Refusing to overwrite run log: $LogPath" }
        $captured | Out-File -LiteralPath $LogPath -Encoding utf8 -NoClobber
    }
    $captured | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0) { throw "Pixi command failed with exit code $exitCode." }
}

function Start-PixiPowerShell {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [string]$StdoutPath = '',
        [string]$StderrPath = ''
    )
    $arguments = @(
        'run', '--manifest-path', $manifestPath, '--', 'powershell.exe',
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $Command
    )
    $parameters = @{
        FilePath = $PixiExe
        ArgumentList = $arguments
        PassThru = $true
        WindowStyle = 'Hidden'
    }
    if (-not [string]::IsNullOrWhiteSpace($StdoutPath)) {
        if (Test-Path -LiteralPath $StdoutPath) { throw "Refusing to overwrite stdout log: $StdoutPath" }
        $parameters.RedirectStandardOutput = $StdoutPath
    }
    if (-not [string]::IsNullOrWhiteSpace($StderrPath)) {
        if (Test-Path -LiteralPath $StderrPath) { throw "Refusing to overwrite stderr log: $StderrPath" }
        $parameters.RedirectStandardError = $StderrPath
    }
    return Start-Process @parameters
}

$suite = Get-Content -Raw -LiteralPath $suitePath | ConvertFrom-Json
if ($suite.milestone -ne 'M8-G0' -or $suite.native_results_status_at_creation -ne 'not_run') {
    throw 'Matrix suite must be a fresh M8-G0 not_run manifest.'
}
$batches = @($suite.batch_manifests)
if ($batches.Count -lt 1) { throw 'Matrix suite has no persistent batch manifests.' }
$suiteDirectory = Split-Path -Parent $suitePath
$logRoot = Join-Path $suiteDirectory 'native_run_logs'
$runId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$logDirectory = Join-Path $logRoot $runId
$preflightReceiptPath = Join-Path $logDirectory 'preflight_receipt.json'
$completionReceiptPath = Join-Path $logDirectory 'completion_receipt.json'
$failureReceiptPath = Join-Path $logDirectory 'failure_receipt.json'
$timeoutReceiptPath = Join-Path $logDirectory 'timeout_receipt.json'
if (Test-Path -LiteralPath $logDirectory) { throw "Refusing to reuse native run log directory: $logDirectory" }
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$replayPath = Join-Path $suiteDirectory 'replay_validation.json'
$analysisPath = if ($suite.split -eq 'frozen_holdout') { Join-Path $suiteDirectory 'analysis.json' } else { $null }
$figureDirectory = if ($suite.split -eq 'frozen_holdout') { Join-Path $suiteDirectory 'figures' } else { $null }
$currentStage = 'output_guard'
$currentBatch = ''
$sourceManifestHash = ''
$initialGpuPreflight = $null
$postBuildGpuPreflight = $null
$batchSourceValidationPassed = $false
$batchSourceValidationLogs = @()

try {
Require-PathAbsent -Path $replayPath -Description 'replay validation output'
if ($null -ne $analysisPath) {
    Require-PathAbsent -Path $analysisPath -Description 'analysis output'
}
if ($null -ne $figureDirectory) {
    Require-PathAbsent -Path $figureDirectory -Description 'figure output directory'
}

$freezeValidated = $false
if ($suite.split -eq 'frozen_holdout') {
    $currentStage = 'freeze_validation'
    $freezePath = [System.IO.Path]::GetFullPath(
        (Join-Path $suiteDirectory ([string]$suite.freeze_manifest))
    )
    Require-File -Path $freezePath -Description 'frozen holdout manifest'
    $escapedRootForFreeze = $repositoryRoot.Replace("'", "''")
    $escapedFreeze = $freezePath.Replace("'", "''")
    $benchmarkSource = (Join-Path $sourceRoot 'action_stream_benchmark').Replace("'", "''")
    $freezeCommand = @"
`$env:PYTHONPATH = '$benchmarkSource'
Set-Location '$escapedRootForFreeze'
python -m action_stream_benchmark.m8_cli freeze-validate --repository-root '$escapedRootForFreeze' --manifest '$escapedFreeze'
"@
    Write-Host 'Validating every live frozen source before native build/execution...'
    Invoke-PixiPowerShell -Command $freezeCommand -LogPath (Join-Path $logDirectory 'freeze_validate.log')
    $freezeValidated = $true
}

# Validate every batch against the current source checkout before touching the
# GPU or building/starting any native process.  This pure-Python path imports
# neither Isaac Sim nor rclpy; it validates protocol, provenance, scenarios,
# fault traces, strategy identity, and non-overwriting output paths.
$currentStage = 'batch_source_validate_only'
$escapedRootForBatchValidation = $repositoryRoot.Replace("'", "''")
$sourcePythonPaths = @(
    (Join-Path $sourceRoot 'action_stream_benchmark'),
    (Join-Path $sourceRoot 'action_stream_isaac'),
    (Join-Path $sourceRoot 'action_stream_policy')
)
$escapedSourcePythonPath = ($sourcePythonPaths -join ';').Replace("'", "''")
foreach ($batchRecord in $batches) {
    $batchRecordPath = [string]$batchRecord.path
    if ([string]::IsNullOrWhiteSpace($batchRecordPath)) {
        throw 'Matrix suite contains a batch record without a path.'
    }
    $batchPath = [System.IO.Path]::GetFullPath((Join-Path $suiteDirectory $batchRecordPath))
    $currentBatch = Get-RepositorySafeRelativePath `
        -BaseDirectory $logDirectory `
        -TargetPath $batchPath `
        -RepositoryRoot $repositoryRoot
    Require-File -Path $batchPath -Description 'persistent M8 batch manifest'
    $batch = Get-Content -Raw -LiteralPath $batchPath | ConvertFrom-Json
    $strategy = [string]$batch.batch_strategy
    if ($strategy -notin @('sync_hold', 'naive_async', 'aligned_async')) {
        throw "Batch has an unknown executor strategy: $batchPath"
    }
    $batchLogStem = (
        [System.IO.Path]::GetFileNameWithoutExtension($batchPath) -replace '[^A-Za-z0-9_.-]', '_'
    )
    $validationLog = Join-Path $logDirectory "$batchLogStem.validate_only.log"
    $batchSourceValidationLogs += [System.IO.Path]::GetFileName($validationLog)
    $escapedBatch = $batchPath.Replace("'", "''")
    $escapedStrategy = $strategy.Replace("'", "''")
    $validationCommand = @"
`$env:PYTHONPATH = '$escapedSourcePythonPath'
Set-Location '$escapedRootForBatchValidation'
python -m action_stream_isaac.dynamic_isaac_adapter --matrix-manifest '$escapedBatch' --expected-strategy '$escapedStrategy' --validate-only
"@
    Write-Host "Validating current-source M8 batch before GPU/build: $currentBatch"
    Invoke-PixiPowerShell -Command $validationCommand -LogPath $validationLog
}
$currentBatch = ''
$batchSourceValidationPassed = $true

$gpuMemoryThresholdMiB = 4096
$currentStage = 'gpu_preflight_initial'
$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($null -eq $nvidiaSmi) { $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue }
if ($null -eq $nvidiaSmi) { throw 'nvidia-smi was not found; refusing native GPU execution.' }
$initialGpuPreflight = Get-GpuPreflightSnapshot `
    -NvidiaSmiPath $nvidiaSmi.Source `
    -MemoryThresholdMiB $gpuMemoryThresholdMiB `
    -Phase 'initial_pre_build'
if (-not $initialGpuPreflight.passed) {
    $gpuRefusalPath = Join-Path $logDirectory 'gpu_refusal_receipt.json'
    $evidenceObject = Write-GpuRefusalReceipt `
        -Path $gpuRefusalPath `
        -Snapshot $initialGpuPreflight `
        -MemoryThresholdMiB $gpuMemoryThresholdMiB
    $evidence = $evidenceObject | ConvertTo-Json -Depth 6
    throw "GPU preflight found competing use; no process was killed and Isaac was not started.`n$evidence"
}

# Fingerprint exactly the source tree supplied to colcon.  The hash is printed
# before any native result and can be retained with the console transcript.
$currentStage = 'source_fingerprint'
$sourceRecords = @(
    Get-ChildItem -LiteralPath $sourceRoot -Recurse -File |
        Where-Object {
            $_.FullName -notmatch '[\\/](?:__pycache__|\.pytest_cache|\.ruff_cache|build|install|log|[^\\/]+\.egg-info)[\\/]' -and
            $_.Extension -notin @('.pyc', '.pyo')
        } |
        Sort-Object FullName |
        ForEach-Object {
            $relative = $_.FullName.Substring($repositoryRoot.TrimEnd('\').Length + 1).Replace('\', '/')
            $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            "$relative|$hash"
        }
)
$sourceManifestText = $sourceRecords -join "`n"
$sourceManifestBytes = [System.Text.Encoding]::UTF8.GetBytes($sourceManifestText)
$sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
    $sourceManifestHash = [System.BitConverter]::ToString(
        $sha256.ComputeHash($sourceManifestBytes)
    ).Replace('-', '').ToLowerInvariant()
}
finally {
    $sha256.Dispose()
}
Write-Host "Current ActionStream ROS source manifest SHA256: $sourceManifestHash"

$currentStage = 'colcon_build'
$escapedWorkspace = $workspacePath.Replace("'", "''")
$escapedSource = $sourceRoot.Replace("'", "''")
$buildCommand = @"
Set-Location '$escapedWorkspace'
colcon build --base-paths '$escapedSource' --packages-select action_stream_msgs action_stream_executor action_stream_policy action_stream_benchmark action_stream_isaac --merge-install --cmake-args -DBUILD_TESTING=ON
"@
Write-Host 'Building the current repository ROS sources into the native Isaac workspace...'
Invoke-PixiPowerShell -Command $buildCommand -LogPath (Join-Path $logDirectory 'colcon_build.log')

$currentStage = 'gpu_preflight_post_build'
$postBuildGpuPreflight = Get-GpuPreflightSnapshot `
    -NvidiaSmiPath $nvidiaSmi.Source `
    -MemoryThresholdMiB $gpuMemoryThresholdMiB `
    -Phase 'post_build_pre_launch'
if (-not $postBuildGpuPreflight.passed) {
    $gpuRefusalPath = Join-Path $logDirectory 'gpu_refusal_receipt.json'
    $evidenceObject = Write-GpuRefusalReceipt `
        -Path $gpuRefusalPath `
        -Snapshot $postBuildGpuPreflight `
        -MemoryThresholdMiB $gpuMemoryThresholdMiB
    $evidence = $evidenceObject | ConvertTo-Json -Depth 6
    throw "Post-build GPU preflight found competing use; no process was killed and Isaac was not started.`n$evidence"
}
$gpuInventory = @($postBuildGpuPreflight.gpu_inventory)
$computeProcessInventory = @($postBuildGpuPreflight.reported_compute_processes)
$computeProcesses = @($postBuildGpuPreflight.actionable_compute_processes)

$currentStage = 'installed_runtime_binding'
$installSetup = Join-Path $workspacePath 'install\local_setup.ps1'
$executor = Join-Path $workspacePath 'install\lib\action_stream_executor\action_stream_executor_node.exe'
$router = Join-Path $workspacePath '.pixi\envs\default\Library\lib\rmw_zenoh_cpp\rmw_zenohd.exe'
Require-File -Path $installSetup -Description 'fresh colcon local overlay setup'
Require-File -Path $executor -Description 'fresh C++ ActionStream executor'
Require-File -Path $router -Description 'Zenoh router executable'
$installedAdapters = @(
    Get-ChildItem -LiteralPath (Join-Path $workspacePath 'install') -Recurse -File -Filter dynamic_isaac_adapter.py
)
if ($installedAdapters.Count -lt 1) {
    throw 'Fresh install does not contain dynamic_isaac_adapter.py.'
}
$sourceAdapter = Join-Path $sourceRoot 'action_stream_isaac\action_stream_isaac\dynamic_isaac_adapter.py'
$sourceAdapterHash = (Get-FileHash -LiteralPath $sourceAdapter -Algorithm SHA256).Hash
$matchingAdapters = @(
    $installedAdapters |
        Where-Object { (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash -eq $sourceAdapterHash }
)
if ($matchingAdapters.Count -lt 1) {
    throw 'Installed dynamic adapter does not match the current checkout after colcon build.'
}
$selectedInstalledAdapter = $matchingAdapters[0].FullName
$suiteRepositoryRelative = Get-ContainedRelativePath -BasePath $repositoryRoot -TargetPath $suitePath
$sourceRepositoryRelative = Get-ContainedRelativePath -BasePath $repositoryRoot -TargetPath $sourceRoot
$plannedLogFiles = @('router.stdout.log', 'router.stderr.log') + $batchSourceValidationLogs
foreach ($batchRecord in $batches) {
    $stem = ([System.IO.Path]::GetFileNameWithoutExtension([string]$batchRecord.path) -replace '[^A-Za-z0-9_.-]', '_')
    $plannedLogFiles += @(
        "$stem.executor.stdout.log",
        "$stem.executor.stderr.log",
        "$stem.adapter.stdout.log",
        "$stem.adapter.stderr.log"
    )
}
$preflightReceipt = [ordered]@{
    schema_version = 1
    milestone = 'M8-G0'
    receipt_kind = 'native_preflight'
    created_utc = [DateTime]::UtcNow.ToString('o')
    operator_authorized_native_gpu_run = $true
    gpu_memory_refusal_threshold_mib = $gpuMemoryThresholdMiB
    gpu_inventory = $gpuInventory
    reported_compute_processes = $computeProcessInventory
    preexisting_compute_processes = $computeProcesses
    initial_gpu_preflight = $initialGpuPreflight
    post_build_gpu_preflight = $postBuildGpuPreflight
    frozen_live_inputs_validated = $freezeValidated
    current_source_batch_validation_passed = $batchSourceValidationPassed
    current_source_batch_validation_count = $batchSourceValidationLogs.Count
    current_source_batch_validation_logs = $batchSourceValidationLogs
    suite_manifest = $suiteRepositoryRelative
    suite_manifest_sha256 = (Get-FileHash -LiteralPath $suitePath -Algorithm SHA256).Hash.ToLowerInvariant()
    source_root = $sourceRepositoryRelative
    source_file_count = $sourceRecords.Count
    source_manifest_sha256 = $sourceManifestHash
    installed_dynamic_adapter = $selectedInstalledAdapter
    installed_dynamic_adapter_sha256 = (Get-FileHash -LiteralPath $selectedInstalledAdapter -Algorithm SHA256).Hash.ToLowerInvariant()
    executor_path = $executor
    executor_sha256 = (Get-FileHash -LiteralPath $executor -Algorithm SHA256).Hash.ToLowerInvariant()
    router_path = $router
    router_sha256 = (Get-FileHash -LiteralPath $router -Algorithm SHA256).Hash.ToLowerInvariant()
    runner_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    batch_timeout_seconds = $BatchTimeoutSeconds
    planned_process_logs = $plannedLogFiles
}
$currentStage = 'preflight_receipt'
Write-Utf8JsonNoBom -Path $preflightReceiptPath -InputObject $preflightReceipt

$headlessValue = if ($Headless) { 'true' } else { 'false' }
$escapedSetup = $installSetup.Replace("'", "''")
$escapedRoot = $repositoryRoot.Replace("'", "''")
$environmentSetup = @"
. '$escapedSetup'
`$env:OMNI_KIT_ACCEPT_EULA = 'YES'
`$env:ROS_DISTRO = 'jazzy'
`$env:RMW_IMPLEMENTATION = 'rmw_zenoh_cpp'
"@
$routerCommand = "Set-Location '$escapedWorkspace'; & '$($router.Replace("'", "''"))'"
$routerStdout = Join-Path $logDirectory 'router.stdout.log'
$routerStderr = Join-Path $logDirectory 'router.stderr.log'
$routerProcess = $null
try {
    $currentStage = 'router_start'
    $routerProcess = Start-PixiPowerShell -Command $routerCommand -StdoutPath $routerStdout -StderrPath $routerStderr
    Start-Sleep -Seconds 2
    if ($routerProcess.HasExited) {
        throw "Owned Zenoh router exited with code $($routerProcess.ExitCode); see $routerStderr."
    }
    foreach ($batchRecord in $batches) {
        $currentStage = 'batch_prepare'
        $batchPath = [System.IO.Path]::GetFullPath((Join-Path $suiteDirectory ([string]$batchRecord.path)))
        $currentBatch = Get-RepositorySafeRelativePath -BaseDirectory $logDirectory -TargetPath $batchPath -RepositoryRoot $repositoryRoot
        Require-File -Path $batchPath -Description 'persistent M8 batch manifest'
        $batch = Get-Content -Raw -LiteralPath $batchPath | ConvertFrom-Json
        if (@($batch.episodes).Count -ne [int]$batch.expected_episode_count) {
            throw "Batch episode count mismatch: $batchPath"
        }
        $strategy = [string]$batch.batch_strategy
        $batchLogStem = ([System.IO.Path]::GetFileNameWithoutExtension($batchPath) -replace '[^A-Za-z0-9_.-]', '_')
        $executorStdout = Join-Path $logDirectory "$batchLogStem.executor.stdout.log"
        $executorStderr = Join-Path $logDirectory "$batchLogStem.executor.stderr.log"
        $adapterStdout = Join-Path $logDirectory "$batchLogStem.adapter.stdout.log"
        $adapterStderr = Join-Path $logDirectory "$batchLogStem.adapter.stderr.log"
        $escapedBatch = $batchPath.Replace("'", "''")
        $executorCommand = @"
$environmentSetup
& '$($executor.Replace("'", "''"))' --ros-args -p use_sim_time:=true -p strategy:=$strategy -p action_dimension:=7 -p 'safe_hold_command:=[0.307015,0.0,0.589907,3.141592653589793,0.0,0.0,1.0]' -p sync_periodic_replan:=true -p observation_topic:=/action_stream/observation -p inference_request_topic:=/action_stream/inference_request -p action_chunk_topic:=/action_stream/action_chunk -p robot_command_topic:=/action_stream/robot_command -p diagnostics_topic:=/action_stream/diagnostics -p runtime_event_topic:=/action_stream/events -p episode_control_topic:=/action_stream/episode_control
"@
        $adapterCommand = @"
$environmentSetup
Set-Location '$escapedRoot'
python -m action_stream_isaac.dynamic_isaac_adapter --matrix-manifest '$escapedBatch' --expected-strategy '$strategy' --headless $headlessValue
"@
        $executorProcess = $null
        $adapterProcess = $null
        try {
            Write-Host "Starting native M8 batch: $($batch.batch_profile_id) / $strategy"
            $currentStage = 'batch_executor_start'
            $executorProcess = Start-PixiPowerShell -Command $executorCommand -StdoutPath $executorStdout -StderrPath $executorStderr
            Start-Sleep -Seconds 2
            if ($executorProcess.HasExited) { throw "C++ executor exited with code $($executorProcess.ExitCode)." }
            $currentStage = 'batch_adapter_start'
            $adapterProcess = Start-PixiPowerShell -Command $adapterCommand -StdoutPath $adapterStdout -StderrPath $adapterStderr
            $currentStage = 'batch_adapter_wait'
            $completedWithinTimeout = $adapterProcess.WaitForExit($BatchTimeoutSeconds * 1000)
            if (-not $completedWithinTimeout) {
                throw [System.TimeoutException]::new(
                    "Dynamic Isaac adapter exceeded the per-batch wall timeout of $BatchTimeoutSeconds seconds: $currentBatch"
                )
            }
            # Give redirected stream pumps a bounded opportunity to flush after
            # the process handle has signaled.  Never reintroduce an unbounded
            # WaitForExit call here.
            if (-not $adapterProcess.WaitForExit(5000)) {
                throw [System.TimeoutException]::new(
                    "Dynamic Isaac adapter exited but its redirected streams did not flush within 5 seconds: $currentBatch"
                )
            }
            if ($adapterProcess.ExitCode -ne 0) { throw "Dynamic Isaac adapter exited with code $($adapterProcess.ExitCode)." }
        }
        finally {
            foreach ($process in @($adapterProcess, $executorProcess)) {
                if ($null -ne $process -and -not $process.HasExited) {
                    Stop-OwnedProcessTree -RootProcessId $process.Id
                }
            }
        }
        $currentStage = 'batch_artifact_validation'
        foreach ($episode in @($batch.episodes)) {
            foreach ($field in @('event_log_path', 'summary_path')) {
                $artifact = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $batchPath) ([string]$episode.$field)))
                Require-File -Path $artifact -Description "complete native $field artifact"
                if ((Get-Item -LiteralPath $artifact).Length -le 0) { throw "Native artifact is empty: $artifact" }
            }
        }
    }
}
finally {
    if ($null -ne $routerProcess -and -not $routerProcess.HasExited) {
        Stop-OwnedProcessTree -RootProcessId $routerProcess.Id
    }
}

$currentBatch = ''
$currentStage = 'replay_validation'
$escapedSuite = $suitePath.Replace("'", "''")
$escapedReplay = $replayPath.Replace("'", "''")
$postCommand = @"
$environmentSetup
Set-Location '$escapedRoot'
python -m action_stream_benchmark.m8_cli validate --manifest '$escapedSuite' --output '$escapedReplay'
"@
Invoke-PixiPowerShell -Command $postCommand -LogPath (Join-Path $logDirectory 'replay_validate.log')

if ($suite.split -eq 'frozen_holdout') {
    $currentStage = 'analysis_and_figures'
    $escapedAnalysis = $analysisPath.Replace("'", "''")
    $escapedFigures = $figureDirectory.Replace("'", "''")
    $analysisCommand = @"
$environmentSetup
Set-Location '$escapedRoot'
python -m action_stream_benchmark.m8_cli analyze --manifest '$escapedSuite' --replay '$escapedReplay' --output '$escapedAnalysis' --bootstrap-resamples 20000
if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }
python -m action_stream_benchmark.m8_cli figures --analysis '$escapedAnalysis' --output-dir '$escapedFigures'
exit `$LASTEXITCODE
"@
    Invoke-PixiPowerShell -Command $analysisCommand -LogPath (Join-Path $logDirectory 'analysis_figures.log')
}

$currentStage = 'process_log_archive'
$processLogArchivePath = Join-Path $logDirectory 'process_logs.tar.gz'
$processLogManifestPath = Join-Path $logDirectory 'process_logs.manifest.json'
$processLogMemberListPath = Join-Path $logDirectory '.process_logs.members.tmp'
Require-PathAbsent -Path $processLogArchivePath -Description 'process-log archive'
Require-PathAbsent -Path $processLogManifestPath -Description 'process-log archive manifest'
Require-PathAbsent -Path $processLogMemberListPath -Description 'temporary process-log member list'
$processLogMembers = @(
    Get-ChildItem -LiteralPath $logDirectory -File -Filter '*.log' |
        Sort-Object Name |
        ForEach-Object { $_.Name }
)
if ($processLogMembers.Count -lt 1) {
    throw 'Native run produced no process/command logs to archive.'
}
Write-Utf8LinesNoBom -Path $processLogMemberListPath -Lines $processLogMembers
$escapedLogDirectory = $logDirectory.Replace("'", "''")
$escapedLogMemberList = $processLogMemberListPath.Replace("'", "''")
$escapedLogArchive = $processLogArchivePath.Replace("'", "''")
$escapedLogManifest = $processLogManifestPath.Replace("'", "''")
$archiveCommand = @"
$environmentSetup
Set-Location '$escapedRoot'
python -m action_stream_benchmark.m8_cli archive --root '$escapedLogDirectory' --member-list '$escapedLogMemberList' --archive '$escapedLogArchive' --manifest '$escapedLogManifest'
if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }
python -m action_stream_benchmark.m8_cli archive-validate --archive '$escapedLogArchive' --manifest '$escapedLogManifest'
exit `$LASTEXITCODE
"@
try {
    Invoke-PixiPowerShell -Command $archiveCommand
}
finally {
    if (Test-Path -LiteralPath $processLogMemberListPath -PathType Leaf) {
        Remove-Item -LiteralPath $processLogMemberListPath -Force
    }
}
Require-File -Path $processLogArchivePath -Description 'validated process-log archive'
Require-File -Path $processLogManifestPath -Description 'validated process-log archive manifest'

$currentStage = 'completion_receipt'
$completionReceipt = [ordered]@{
    schema_version = 1
    milestone = 'M8-G0'
    receipt_kind = 'native_completion'
    completed_utc = [DateTime]::UtcNow.ToString('o')
    status = 'complete'
    preflight_receipt = Get-ContainedRelativePath -BasePath $logDirectory -TargetPath $preflightReceiptPath
    preflight_receipt_sha256 = (Get-FileHash -LiteralPath $preflightReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
    replay_validation = Get-RepositorySafeRelativePath -BaseDirectory $logDirectory -TargetPath $replayPath -RepositoryRoot $repositoryRoot
    replay_validation_sha256 = (Get-FileHash -LiteralPath $replayPath -Algorithm SHA256).Hash.ToLowerInvariant()
    analysis = if ($null -ne $analysisPath) { Get-RepositorySafeRelativePath -BaseDirectory $logDirectory -TargetPath $analysisPath -RepositoryRoot $repositoryRoot } else { $null }
    analysis_sha256 = if ($null -ne $analysisPath) { (Get-FileHash -LiteralPath $analysisPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
    process_log_archive = Get-ContainedRelativePath -BasePath $logDirectory -TargetPath $processLogArchivePath
    process_log_archive_sha256 = (Get-FileHash -LiteralPath $processLogArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    process_log_manifest = Get-ContainedRelativePath -BasePath $logDirectory -TargetPath $processLogManifestPath
    process_log_manifest_sha256 = (Get-FileHash -LiteralPath $processLogManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    process_log_member_count = $processLogMembers.Count
}
Write-Utf8JsonNoBom -Path $completionReceiptPath -InputObject $completionReceipt
$currentStage = 'complete'
Write-Host "Completed native M8 suite; independent replay audit: $replayPath"
Write-Host "Preflight receipt: $preflightReceiptPath"
Write-Host "Completion receipt: $completionReceiptPath"
}
catch {
    $originalError = $_
    $attemptReceiptPath = if ($originalError.Exception -is [System.TimeoutException]) {
        $timeoutReceiptPath
    }
    else {
        $failureReceiptPath
    }
    try {
        Write-NativeFailureReceipt `
            -Path $attemptReceiptPath `
            -LogDirectory $logDirectory `
            -Stage $currentStage `
            -ErrorRecord $originalError `
            -SuitePath $suitePath `
            -RepositoryRoot $repositoryRoot `
            -BatchTimeoutSeconds $BatchTimeoutSeconds `
            -CurrentBatch $currentBatch `
            -PreflightReceiptPath $preflightReceiptPath `
            -SourceManifestSha256 $sourceManifestHash `
            -InitialGpuPreflight $initialGpuPreflight `
            -PostBuildGpuPreflight $postBuildGpuPreflight
        Write-Warning "Native M8 attempt failed at stage '$currentStage'; receipt: $attemptReceiptPath"
    }
    catch {
        Write-Warning "Native M8 attempt failed at stage '$currentStage', and its failure receipt could not be written: $($_.Exception.Message)"
    }
    throw $originalError
}
