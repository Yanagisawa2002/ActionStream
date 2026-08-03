<#
.SYNOPSIS
Records one replay-clean, non-headline M8 Profile-1 aligned development demo.

.DESCRIPTION
The input is the one-episode development matrix produced by matrix-create
--strategy aligned_async. The script validates an existing freeze, rebuilds
the checkout, resamples GPU occupancy, starts only its owned process trees,
replays the complete matrix, and validates H.264 video before release.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$DevelopmentMatrixManifest,
    [Parameter(Mandatory = $true)][string]$FreezeManifest,
    [Parameter(Mandatory = $true)][string]$IsaacWorkspace,
    [Parameter(Mandatory = $true)][string]$VideoOutput,
    [string]$PixiExe = '',
    [ValidateRange(60, 7200)][int]$AdapterTimeoutSeconds = 1800,
    [switch]$AuthorizeNativeGpuRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Require-File {
    param([string]$Path, [string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }
}

function Require-PathAbsent {
    param([string]$Path, [string]$Description)
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to overwrite existing $($Description): $Path"
    }
}

function Write-Utf8JsonNoBom {
    param([string]$Path, $InputObject, [int]$Depth = 12)
    $json = $InputObject | ConvertTo-Json -Depth $Depth
    $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes(
        $json + [Environment]::NewLine
    )
    $stream = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try { $stream.Write($bytes, 0, $bytes.Length) }
    finally { $stream.Dispose() }
}

function Write-Utf8LinesNoBom {
    param([string]$Path, [AllowEmptyCollection()][string[]]$Lines = @())
    $text = if ($Lines.Count -gt 0) {
        ($Lines -join [Environment]::NewLine) + [Environment]::NewLine
    } else { '' }
    $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($text)
    $stream = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    try { $stream.Write($bytes, 0, $bytes.Length) }
    finally { $stream.Dispose() }
}

function Test-FileHasUtf8Bom {
    param([string]$Path)
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    return (
        $bytes.Length -ge 3 -and
        $bytes[0] -eq 0xEF -and
        $bytes[1] -eq 0xBB -and
        $bytes[2] -eq 0xBF
    )
}

function Get-ContainedRelativePath {
    param([string]$BasePath, [string]$TargetPath)
    $baseFull = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\', '/')
    $targetFull = [System.IO.Path]::GetFullPath($TargetPath)
    if ($targetFull.Equals($baseFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        return '.'
    }
    $prefix = $baseFull + [System.IO.Path]::DirectorySeparatorChar
    if (-not $targetFull.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside the declared base: $targetFull"
    }
    return $targetFull.Substring($prefix.Length).Replace('\', '/')
}

function Get-RepositoryReference {
    param([string]$RepositoryRoot, [string]$TargetPath)
    Get-ContainedRelativePath -BasePath $RepositoryRoot -TargetPath $TargetPath
}

function Resolve-RepositoryRelativePath {
    param(
        [string]$Value,
        [string]$BaseDirectory,
        [string]$RepositoryRoot,
        [string]$Description
    )
    if (
        [string]::IsNullOrWhiteSpace($Value) -or
        [System.IO.Path]::IsPathRooted($Value)
    ) {
        throw "$Description must be a non-empty portable relative path."
    }
    $resolved = [System.IO.Path]::GetFullPath((Join-Path $BaseDirectory $Value))
    $null = Get-ContainedRelativePath -BasePath $RepositoryRoot -TargetPath $resolved
    return $resolved
}

function Get-AvailableArtifactRecords {
    param([string]$LogDirectory)
    if (-not (Test-Path -LiteralPath $LogDirectory -PathType Container)) {
        return @()
    }
    @(
        Get-ChildItem -LiteralPath $LogDirectory -Recurse -File -ErrorAction SilentlyContinue |
            Sort-Object FullName |
            ForEach-Object {
                [ordered]@{
                    path = Get-ContainedRelativePath -BasePath $LogDirectory -TargetPath $_.FullName
                    size_bytes = $_.Length
                    sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                }
            }
    )
}

function Convert-ToPortableText {
    param([string]$Text, [string]$RepositoryRoot, [string]$WorkspacePath)
    $portable = [regex]::Replace(
        $Text,
        [regex]::Escape($RepositoryRoot),
        '<repository_root>',
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    $portable = [regex]::Replace(
        $portable,
        [regex]::Escape($WorkspacePath),
        '<isaac_workspace>',
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    )
    $portable.Replace('\', '/')
}

function Stop-OwnedProcessTree {
    param([int]$RootProcessId)
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

function Get-GpuPreflightSnapshot {
    param([string]$NvidiaSmiPath, [int]$MemoryThresholdMiB, [string]$Phase)
    $gpuLines = @(
        & $NvidiaSmiPath --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
    )
    if ($LASTEXITCODE -ne 0 -or $gpuLines.Count -lt 1) {
        throw "nvidia-smi GPU inventory failed during $Phase."
    }
    $gpuInventory = @()
    foreach ($line in $gpuLines) {
        $parts = @($line -split ',' | ForEach-Object { $_.Trim() })
        if ($parts.Count -ne 7) {
            throw "Unexpected nvidia-smi GPU row during $($Phase): $line"
        }
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
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi compute-process query failed during $Phase."
    }
    $computeInventory = @()
    foreach ($line in $computeLines) {
        $parts = @($line -split ',' | ForEach-Object { $_.Trim() })
        if ($parts.Count -eq 4 -and $parts[1] -match '^\d+$') {
            $usedMemory = if ($parts[3] -match '^\d+$') { [int]$parts[3] } else { $null }
            $computeInventory += [pscustomobject][ordered]@{
                gpu_uuid = $parts[0]
                process_id = [int]$parts[1]
                process_name = [System.IO.Path]::GetFileName($parts[2])
                used_memory_mib = $usedMemory
                actionable_compute_allocation = ($null -ne $usedMemory -and $usedMemory -gt 0)
            }
        }
    }
    $actionable = @($computeInventory | Where-Object { $_.actionable_compute_allocation })
    $occupied = @($gpuInventory | Where-Object { $_.memory_used_mib -gt $MemoryThresholdMiB })
    [pscustomobject][ordered]@{
        phase = $Phase
        captured_utc = [DateTime]::UtcNow.ToString('o')
        gpu_inventory = $gpuInventory
        reported_compute_processes = $computeInventory
        actionable_compute_processes = $actionable
        occupied_gpus = $occupied
        passed = ($actionable.Count -eq 0 -and $occupied.Count -eq 0)
    }
}

function Write-GpuRefusalReceipt {
    param(
        [string]$Path,
        $Snapshot,
        [int]$ThresholdMiB,
        [string]$MatrixReference,
        [string]$MatrixSha256,
        [string]$FreezeReference,
        [string]$FreezeSha256
    )
    Write-Utf8JsonNoBom -Path $Path -InputObject ([ordered]@{
        schema_version = 1
        milestone = 'M8-G0'
        receipt_kind = 'native_demo_gpu_refusal'
        status = 'refused'
        created_utc = [DateTime]::UtcNow.ToString('o')
        headline_eligible = $false
        benchmark_evidence = $false
        operator_authorized_native_gpu_run = $true
        matrix_manifest = $MatrixReference
        matrix_manifest_sha256 = $MatrixSha256
        freeze_manifest = $FreezeReference
        freeze_manifest_sha256 = $FreezeSha256
        preflight_phase = $Snapshot.phase
        threshold_mib = $ThresholdMiB
        gpu_inventory = $Snapshot.gpu_inventory
        reported_compute_processes = $Snapshot.reported_compute_processes
        actionable_compute_processes = $Snapshot.actionable_compute_processes
        occupied_gpus = $Snapshot.occupied_gpus
        action = 'refused_without_killing_any_process'
    })
}

function Invoke-PixiPowerShell {
    param([string]$Command, [string]$LogPath, [string]$PixiPath, [string]$PixiManifest)
    Require-PathAbsent -Path $LogPath -Description 'Pixi command log'
    $captured = @(
        & $PixiPath run --manifest-path $PixiManifest -- powershell.exe -NoProfile -ExecutionPolicy Bypass -Command $Command 2>&1 |
            ForEach-Object { [string]$_ }
    )
    $exitCode = $LASTEXITCODE
    Write-Utf8LinesNoBom -Path $LogPath -Lines $captured
    $captured | ForEach-Object { Write-Host $_ }
    if ($exitCode -ne 0) { throw "Pixi command failed with exit code $exitCode." }
}

function Start-PixiPowerShell {
    param(
        [string]$Command,
        [string]$StdoutPath,
        [string]$StderrPath,
        [string]$PixiPath,
        [string]$PixiManifest
    )
    Require-PathAbsent -Path $StdoutPath -Description 'process stdout log'
    Require-PathAbsent -Path $StderrPath -Description 'process stderr log'
    $arguments = @(
        'run', '--manifest-path', $PixiManifest, '--', 'powershell.exe',
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $Command
    )
    $argumentLine = @(
        $arguments | ForEach-Object { ConvertTo-CommandLineArgument -Value $_ }
    ) -join ' '
    Start-Process -FilePath $PixiPath -ArgumentList $argumentLine -PassThru -WindowStyle Hidden -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath
}

function ConvertTo-CommandLineArgument {
    param([string]$Value)
    if ($Value.Length -eq 0) { return '""' }
    if ($Value -notmatch '[\s"]') { return $Value }
    '"' + $Value.Replace('"', '\"') + '"'
}

function Invoke-BoundedNativeTool {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$StdoutPath,
        [string]$StderrPath,
        [ValidateRange(1, 600)][int]$TimeoutSeconds = 120
    )
    Require-PathAbsent -Path $StdoutPath -Description 'native-tool stdout log'
    Require-PathAbsent -Path $StderrPath -Description 'native-tool stderr log'
    $argumentLine = (@(
        $Arguments | ForEach-Object { ConvertTo-CommandLineArgument -Value $_ }
    ) -join ' ')
    $process = Start-Process -FilePath $Executable -ArgumentList $argumentLine -PassThru -WindowStyle Hidden -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath
    try {
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            throw [System.TimeoutException]::new(
                "Native verification tool exceeded $TimeoutSeconds seconds."
            )
        }
        if (-not $process.WaitForExit(5000)) {
            throw [System.TimeoutException]::new(
                'Native verification tool streams did not flush within 5 seconds.'
            )
        }
        if ($process.ExitCode -ne 0) {
            throw "Native verification tool exited with code $($process.ExitCode)."
        }
    }
    finally {
        if (-not $process.HasExited) {
            Stop-OwnedProcessTree -RootProcessId $process.Id
        }
    }
}

function Convert-RationalToDouble {
    param([string]$Value)
    $parts = @($Value -split '/')
    if ($parts.Count -ne 2 -or [double]$parts[1] -eq 0.0) {
        throw "Invalid rational frame rate: $Value"
    }
    [double]$parts[0] / [double]$parts[1]
}

function Get-PositiveDouble {
    param($Primary, $Fallback, [string]$Description)
    foreach ($candidate in @($Primary, $Fallback)) {
        $parsed = 0.0
        if (
            [double]::TryParse(
                [string]$candidate,
                [System.Globalization.NumberStyles]::Float,
                [System.Globalization.CultureInfo]::InvariantCulture,
                [ref]$parsed
            ) -and
            $parsed -gt 0.0 -and
            -not [double]::IsNaN($parsed) -and
            -not [double]::IsInfinity($parsed)
        ) { return $parsed }
    }
    throw "$Description is missing or invalid."
}

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$sourceRoot = Join-Path $repositoryRoot 'ros2_ws\src'
$matrixPath = [System.IO.Path]::GetFullPath($DevelopmentMatrixManifest)
$freezePath = [System.IO.Path]::GetFullPath($FreezeManifest)
$workspacePath = [System.IO.Path]::GetFullPath($IsaacWorkspace)
$videoPath = [System.IO.Path]::GetFullPath($VideoOutput)
$pixiManifest = Join-Path $workspacePath 'pixi.toml'

Require-File -Path $matrixPath -Description 'single-episode M8 development matrix'
Require-File -Path $freezePath -Description 'existing M8 freeze manifest'
Require-File -Path $pixiManifest -Description 'official Isaac Pixi manifest'
if (-not $AuthorizeNativeGpuRun) {
    throw 'Native demo capture is fail-closed; rerun with -AuthorizeNativeGpuRun.'
}
if ([System.IO.Path]::GetExtension($videoPath).ToLowerInvariant() -ne '.mp4') {
    throw 'VideoOutput must end in .mp4.'
}
Require-PathAbsent -Path $videoPath -Description 'release demo video'
if ([string]::IsNullOrWhiteSpace($PixiExe)) {
    $pixiCommand = Get-Command pixi -ErrorAction SilentlyContinue
    if ($null -eq $pixiCommand) { throw 'Pixi was not found on PATH; pass -PixiExe.' }
    $PixiExe = $pixiCommand.Source
}
$PixiExe = [System.IO.Path]::GetFullPath($PixiExe)
Require-File -Path $PixiExe -Description 'Pixi executable'

$matrixReference = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $matrixPath
$freezeReference = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $freezePath
$videoReference = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $videoPath
if (-not $matrixReference.StartsWith('outputs/m8_g0/', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Demo matrix must live under outputs/m8_g0.'
}
if (-not $videoReference.StartsWith('release/m8_g0/media/', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Verified demo MP4 must be under release/m8_g0/media.'
}
$evidenceRoot = Split-Path -Parent $matrixPath
$replayPath = Join-Path $evidenceRoot 'replay_validation.json'
$receiptPath = Join-Path $evidenceRoot 'demo_receipt.json'
Require-PathAbsent -Path $replayPath -Description 'demo replay validation'
Require-PathAbsent -Path $receiptPath -Description 'demo completion receipt'
$runId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$logDirectory = Join-Path (Join-Path $evidenceRoot 'native_demo_logs') $runId
New-Item -ItemType Directory -Path $logDirectory | Out-Null
$stagedVideoPath = Join-Path $logDirectory 'captured_demo.mp4'
$stagedReplayPath = Join-Path $logDirectory 'replay_validation.json'
$failureReceiptPath = Join-Path $logDirectory 'failure_receipt.json'
$timeoutReceiptPath = Join-Path $logDirectory 'timeout_receipt.json'
$refusalReceiptPath = Join-Path $logDirectory 'gpu_refusal_receipt.json'
$currentStage = 'matrix_validation'
$sourceManifestHash = ''
$initialGpuPreflight = $null
$postBuildGpuPreflight = $null

function Write-DemoFailureReceipt {
    param(
        [string]$Path,
        [string]$Stage,
        $ErrorRecord,
        [string]$RepositoryRoot,
        [string]$WorkspacePath,
        [string]$MatrixPath,
        [string]$FreezePath,
        [string]$VideoPath,
        [string]$LogDirectory,
        [int]$TimeoutSeconds,
        [string]$SourceManifestSha256,
        $InitialGpuPreflight,
        $PostBuildGpuPreflight
    )
    $exception = $ErrorRecord.Exception
    $timedOut = $exception -is [System.TimeoutException]
    $payload = [ordered]@{
        schema_version = 1
        milestone = 'M8-G0'
        receipt_kind = if ($timedOut) { 'native_demo_timeout' } else { 'native_demo_failure' }
        status = if ($timedOut) { 'timeout' } else { 'failed' }
        created_utc = [DateTime]::UtcNow.ToString('o')
        headline_eligible = $false
        benchmark_evidence = $false
        stage = $Stage
        adapter_timeout_seconds = $TimeoutSeconds
        matrix_manifest = Get-RepositoryReference -RepositoryRoot $RepositoryRoot -TargetPath $MatrixPath
        matrix_manifest_sha256 = (Get-FileHash -LiteralPath $MatrixPath -Algorithm SHA256).Hash.ToLowerInvariant()
        freeze_manifest = Get-RepositoryReference -RepositoryRoot $RepositoryRoot -TargetPath $FreezePath
        freeze_manifest_sha256 = (Get-FileHash -LiteralPath $FreezePath -Algorithm SHA256).Hash.ToLowerInvariant()
        requested_video_output = Get-RepositoryReference -RepositoryRoot $RepositoryRoot -TargetPath $VideoPath
        run_evidence_directory = Get-RepositoryReference -RepositoryRoot $RepositoryRoot -TargetPath $LogDirectory
        source_manifest_sha256 = if ([string]::IsNullOrWhiteSpace($SourceManifestSha256)) { $null } else { $SourceManifestSha256 }
        initial_gpu_preflight = $InitialGpuPreflight
        post_build_gpu_preflight = $PostBuildGpuPreflight
        error = [ordered]@{
            type = $exception.GetType().FullName
            message = Convert-ToPortableText -Text $exception.Message -RepositoryRoot $RepositoryRoot -WorkspacePath $WorkspacePath
        }
        available_run_artifacts = @(Get-AvailableArtifactRecords -LogDirectory $LogDirectory)
        action = 'stopped_only_owned_process_trees_and_preserved_available_evidence'
    }
    Write-Utf8JsonNoBom -Path $Path -InputObject $payload
}

$videoPromoted = $false
$replayPromoted = $false

try {
    $matrix = Get-Content -Raw -LiteralPath $matrixPath | ConvertFrom-Json
    if (
        [int]$matrix.schema_version -ne 1 -or
        $matrix.milestone -ne 'M8-G0' -or
        $matrix.split -ne 'development' -or
        $matrix.headline_eligible -ne $false -or
        $matrix.native_results_status_at_creation -ne 'not_run'
    ) {
        throw 'Demo input must be a fresh non-headline M8 development matrix.'
    }
    $strategyFilter = @($matrix.development_strategy_filter)
    if ($strategyFilter.Count -ne 1 -or [string]$strategyFilter[0] -ne 'aligned_async') {
        throw 'Demo matrix must be created with exactly --strategy aligned_async.'
    }
    $matrixEpisodes = @($matrix.episodes)
    $batchRecords = @($matrix.batch_manifests)
    if (
        $matrixEpisodes.Count -ne 1 -or
        [int]$matrix.expected_episode_count -ne 1 -or
        $batchRecords.Count -ne 1 -or
        [int]$matrix.persistent_batch_count -ne 1
    ) {
        throw 'Demo matrix must contain exactly one episode and one persistent batch.'
    }
    $profileNames = @($matrix.profiles.PSObject.Properties.Name)
    if ($profileNames.Count -ne 1 -or $profileNames[0] -ne 'profile_1_fixed') {
        throw 'Demo matrix must contain exactly Profile 1 fixed latency.'
    }
    $matrixEpisode = $matrixEpisodes[0]
    if (
        $matrixEpisode.profile_id -ne 'profile_1_fixed' -or
        $matrixEpisode.strategy -ne 'aligned_async'
    ) {
        throw 'Demo episode must be the aligned Profile 1 development method.'
    }
    $batchRecord = $batchRecords[0]
    if (
        $batchRecord.profile_id -ne 'profile_1_fixed' -or
        $batchRecord.strategy -ne 'aligned_async' -or
        [int]$batchRecord.episode_count -ne 1
    ) {
        throw 'Demo batch record must identify one aligned Profile 1 episode.'
    }
    $batchPath = Resolve-RepositoryRelativePath -Value ([string]$batchRecord.path) -BaseDirectory $evidenceRoot -RepositoryRoot $repositoryRoot -Description 'demo batch manifest'
    Require-File -Path $batchPath -Description 'single-episode aligned batch manifest'
    $batch = Get-Content -Raw -LiteralPath $batchPath | ConvertFrom-Json
    $batchEpisodes = @($batch.episodes)
    if (
        [int]$batch.schema_version -ne 1 -or
        $batch.milestone -ne 'M8-G0' -or
        $batch.split -ne 'development' -or
        $batch.headline_eligible -ne $false -or
        $batch.native_results_status_at_creation -ne 'not_run' -or
        $batch.batch_profile_id -ne 'profile_1_fixed' -or
        $batch.batch_strategy -ne 'aligned_async' -or
        [int]$batch.expected_episode_count -ne 1 -or
        $batchEpisodes.Count -ne 1
    ) {
        throw 'Resolved batch is not the fresh single-episode aligned Profile 1 batch.'
    }
    $batchEpisode = $batchEpisodes[0]
    foreach ($field in @(
        'episode_id', 'profile_id', 'strategy', 'fault_trace_file',
        'fault_trace_sha256', 'scenario_file', 'scenario_sha256',
        'event_log_path', 'summary_path'
    )) {
        if ([string]$batchEpisode.$field -ne [string]$matrixEpisode.$field) {
            throw "Matrix and batch episode binding differs for $field."
        }
    }
    if ([int]$batchEpisode.seed -ne [int]$matrixEpisode.seed) {
        throw 'Matrix and batch episode seed binding differs.'
    }

    $demoSeedPath = Join-Path $repositoryRoot 'configs\m8_demo_seed.json'
    Require-File -Path $demoSeedPath -Description 'purpose-built demo seed file'
    $matrixSeedPath = Resolve-RepositoryRelativePath -Value ([string]$matrix.seed_file) -BaseDirectory $evidenceRoot -RepositoryRoot $repositoryRoot -Description 'demo seed file'
    if (-not $matrixSeedPath.Equals($demoSeedPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Demo matrix must bind configs/m8_demo_seed.json.'
    }
    $demoSeedHash = (Get-FileHash -LiteralPath $demoSeedPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($matrix.seed_file_sha256 -ne $demoSeedHash) {
        throw 'Demo matrix seed-file hash does not match configs/m8_demo_seed.json.'
    }
    $demoSeedConfig = Get-Content -Raw -LiteralPath $demoSeedPath | ConvertFrom-Json
    $demoSeeds = @($demoSeedConfig.seeds)
    if (
        $demoSeedConfig.milestone -ne 'M8-G0' -or
        $demoSeedConfig.split -ne 'development_demo' -or
        $demoSeedConfig.headline_eligible -ne $false -or
        $demoSeeds.Count -ne 1 -or
        [int]$demoSeeds[0] -ne [int]$matrixEpisode.seed
    ) {
        throw 'Purpose-built demo seed contract does not match the matrix.'
    }

    $freeze = Get-Content -Raw -LiteralPath $freezePath | ConvertFrom-Json
    if (
        [int]$freeze.schema_version -ne 1 -or
        $freeze.milestone -ne 'M8-G0' -or
        $freeze.frozen_before_first_holdout_result -ne $true -or
        ([string]$freeze.freeze_sha256).Length -ne 64
    ) {
        throw 'Freeze manifest has an invalid M8 identity or declaration.'
    }
    $freezeInputs = @($freeze.inputs)
    $developmentCalibration = $freeze.development_calibration
    $selectedCandidateId = [string]$developmentCalibration.selected_candidate_id
    $selectedCandidateRole = [string]$developmentCalibration.selected_candidate_protocol_role
    $expectedSelectedRole = if ($selectedCandidateId -eq 'candidate_0') {
        'baseline_candidate_protocol'
    } else {
        "development:$($selectedCandidateId):protocol"
    }
    if (
        $selectedCandidateId -notmatch '^candidate_[0-2]$' -or
        $selectedCandidateRole -ne $expectedSelectedRole -or
        ([string]$developmentCalibration.selected_candidate_protocol_file_sha256).Length -ne 64 -or
        ([string]$developmentCalibration.selected_protocol_sha256).Length -ne 64
    ) {
        throw 'Freeze selected-candidate calibration pointer is malformed.'
    }
    $candidateRecords = @(
        $freezeInputs | Where-Object { $_.role -eq $selectedCandidateRole }
    )
    $developmentSeedRecords = @($freezeInputs | Where-Object { $_.role -eq 'development_seeds' })
    $profileRecords = @($freezeInputs | Where-Object { $_.role -eq 'profile:profile_1_fixed' })
    if (
        $candidateRecords.Count -ne 1 -or
        $developmentSeedRecords.Count -ne 1 -or
        $profileRecords.Count -ne 1
    ) {
        throw 'Freeze lacks unique selected-candidate, development-seed, or Profile-1 bindings.'
    }
    $candidatePath = Resolve-RepositoryRelativePath -Value ([string]$matrix.candidate_manifest) -BaseDirectory $evidenceRoot -RepositoryRoot $repositoryRoot -Description 'selected candidate protocol'
    $frozenCandidatePath = Resolve-RepositoryRelativePath -Value ([string]$candidateRecords[0].path) -BaseDirectory $repositoryRoot -RepositoryRoot $repositoryRoot -Description 'selected candidate freeze role'
    $selectedCandidatePath = Resolve-RepositoryRelativePath -Value ([string]$developmentCalibration.selected_candidate_protocol_path) -BaseDirectory $repositoryRoot -RepositoryRoot $repositoryRoot -Description 'selected candidate calibration pointer'
    Require-File -Path $candidatePath -Description 'selected matrix candidate protocol'
    $candidateHash = (Get-FileHash -LiteralPath $candidatePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (
        -not $candidatePath.Equals($frozenCandidatePath, [System.StringComparison]::OrdinalIgnoreCase) -or
        -not $candidatePath.Equals($selectedCandidatePath, [System.StringComparison]::OrdinalIgnoreCase) -or
        $candidateRecords[0].sha256 -ne $candidateHash -or
        $developmentCalibration.selected_candidate_protocol_file_sha256 -ne $candidateHash -or
        $matrix.protocol_sha256 -ne $developmentCalibration.selected_protocol_sha256
    ) {
        throw 'Demo matrix does not bind the immutable selected development candidate.'
    }
    $matrixProfileRecord = $matrix.profiles.profile_1_fixed
    $matrixProfilePath = Resolve-RepositoryRelativePath -Value ([string]$matrixProfileRecord.path) -BaseDirectory $evidenceRoot -RepositoryRoot $repositoryRoot -Description 'matrix Profile-1 config'
    $frozenProfilePath = Resolve-RepositoryRelativePath -Value ([string]$profileRecords[0].path) -BaseDirectory $repositoryRoot -RepositoryRoot $repositoryRoot -Description 'frozen Profile-1 config'
    Require-File -Path $matrixProfilePath -Description 'Profile-1 config'
    $profileFileHash = (Get-FileHash -LiteralPath $matrixProfilePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (
        -not $matrixProfilePath.Equals($frozenProfilePath, [System.StringComparison]::OrdinalIgnoreCase) -or
        $matrixProfileRecord.sha256 -ne $profileFileHash -or
        $profileRecords[0].sha256 -ne $profileFileHash
    ) {
        throw 'Demo Profile-1 config is not the exact frozen profile.'
    }
    $frozenDevelopmentSeedPath = Resolve-RepositoryRelativePath -Value ([string]$developmentSeedRecords[0].path) -BaseDirectory $repositoryRoot -RepositoryRoot $repositoryRoot -Description 'frozen development seeds'
    Require-File -Path $frozenDevelopmentSeedPath -Description 'frozen development seed file'
    $frozenDevelopmentSeedHash = (Get-FileHash -LiteralPath $frozenDevelopmentSeedPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($developmentSeedRecords[0].sha256 -ne $frozenDevelopmentSeedHash) {
        throw 'Frozen development seed file hash is stale.'
    }
    $frozenDevelopmentSeeds = @(
        (Get-Content -Raw -LiteralPath $frozenDevelopmentSeedPath | ConvertFrom-Json).seeds
    )
    if ([int]$matrixEpisode.seed -notin @($frozenDevelopmentSeeds | ForEach-Object { [int]$_ })) {
        throw 'Demo seed is not a member of the frozen development split.'
    }

    $batchDirectory = Split-Path -Parent $batchPath
    $scenarioPath = Resolve-RepositoryRelativePath -Value ([string]$batchEpisode.scenario_file) -BaseDirectory $batchDirectory -RepositoryRoot $repositoryRoot -Description 'demo scenario'
    $tracePath = Resolve-RepositoryRelativePath -Value ([string]$batchEpisode.fault_trace_file) -BaseDirectory $batchDirectory -RepositoryRoot $repositoryRoot -Description 'demo fault trace'
    $eventLogPath = Resolve-RepositoryRelativePath -Value ([string]$batchEpisode.event_log_path) -BaseDirectory $batchDirectory -RepositoryRoot $repositoryRoot -Description 'demo event log'
    $summaryPath = Resolve-RepositoryRelativePath -Value ([string]$batchEpisode.summary_path) -BaseDirectory $batchDirectory -RepositoryRoot $repositoryRoot -Description 'demo summary'
    Require-File -Path $scenarioPath -Description 'materialized demo scenario'
    Require-File -Path $tracePath -Description 'materialized Profile-1 fault trace'
    Require-PathAbsent -Path $eventLogPath -Description 'demo raw event log'
    Require-PathAbsent -Path $summaryPath -Description 'demo episode summary'
    $scenarioFileHashBefore = (Get-FileHash -LiteralPath $scenarioPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $traceFileHashBefore = (Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash.ToLowerInvariant()

    $currentStage = 'freeze_validation'
    $escapedRoot = $repositoryRoot.Replace("'", "''")
    $escapedFreeze = $freezePath.Replace("'", "''")
    $benchmarkSource = (Join-Path $sourceRoot 'action_stream_benchmark').Replace("'", "''")
    $freezeCommand = @(
        "$" + "env:PYTHONPATH = '$benchmarkSource'",
        "Set-Location '$escapedRoot'",
        "python -m action_stream_benchmark.m8_cli freeze-validate --repository-root '$escapedRoot' --manifest '$escapedFreeze'"
    ) -join [Environment]::NewLine
    Invoke-PixiPowerShell -Command $freezeCommand -LogPath (Join-Path $logDirectory 'freeze_validate.log') -PixiPath $PixiExe -PixiManifest $pixiManifest

    $gpuMemoryThresholdMiB = 4096
    $currentStage = 'gpu_preflight_initial'
    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue }
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi was not found; refusing demo capture.' }
    $initialGpuPreflight = Get-GpuPreflightSnapshot -NvidiaSmiPath $nvidiaSmi.Source -MemoryThresholdMiB $gpuMemoryThresholdMiB -Phase 'initial_pre_build'
    if (-not $initialGpuPreflight.passed) {
        Write-GpuRefusalReceipt -Path $refusalReceiptPath -Snapshot $initialGpuPreflight -ThresholdMiB $gpuMemoryThresholdMiB -MatrixReference $matrixReference -MatrixSha256 ((Get-FileHash -LiteralPath $matrixPath -Algorithm SHA256).Hash.ToLowerInvariant()) -FreezeReference $freezeReference -FreezeSha256 ((Get-FileHash -LiteralPath $freezePath -Algorithm SHA256).Hash.ToLowerInvariant())
        throw 'GPU preflight found competing use; no process was killed.'
    }

    $currentStage = 'source_fingerprint'
    $sourceRecords = @(
        Get-ChildItem -LiteralPath $sourceRoot -Recurse -File |
            Where-Object {
                $_.FullName -notmatch '[\\/](?:__pycache__|\.pytest_cache|\.ruff_cache|build|install|log|[^\\/]+\.egg-info)[\\/]' -and
                $_.Extension -notin @('.pyc', '.pyo')
            } |
            Sort-Object { $_.FullName.ToLowerInvariant() } |
            ForEach-Object {
                $relative = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $_.FullName
                $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                "$relative|$hash"
            }
    )
    $sourceManifestBytes = [System.Text.Encoding]::UTF8.GetBytes(
        ($sourceRecords -join [char]10)
    )
    $sourceHasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $sourceManifestHash = [System.BitConverter]::ToString(
            $sourceHasher.ComputeHash($sourceManifestBytes)
        ).Replace('-', '').ToLowerInvariant()
    } finally { $sourceHasher.Dispose() }

    $currentStage = 'colcon_build'
    $escapedWorkspace = $workspacePath.Replace("'", "''")
    $escapedSource = $sourceRoot.Replace("'", "''")
    $buildCommand = @(
        "Set-Location '$escapedWorkspace'",
        "colcon build --base-paths '$escapedSource' --packages-select action_stream_msgs action_stream_executor action_stream_policy action_stream_benchmark action_stream_isaac --merge-install --cmake-args -DBUILD_TESTING=ON"
    ) -join [Environment]::NewLine
    Invoke-PixiPowerShell -Command $buildCommand -LogPath (Join-Path $logDirectory 'colcon_build.log') -PixiPath $PixiExe -PixiManifest $pixiManifest

    $currentStage = 'gpu_preflight_post_build'
    $postBuildGpuPreflight = Get-GpuPreflightSnapshot -NvidiaSmiPath $nvidiaSmi.Source -MemoryThresholdMiB $gpuMemoryThresholdMiB -Phase 'post_build_pre_launch'
    if (-not $postBuildGpuPreflight.passed) {
        Write-GpuRefusalReceipt -Path $refusalReceiptPath -Snapshot $postBuildGpuPreflight -ThresholdMiB $gpuMemoryThresholdMiB -MatrixReference $matrixReference -MatrixSha256 ((Get-FileHash -LiteralPath $matrixPath -Algorithm SHA256).Hash.ToLowerInvariant()) -FreezeReference $freezeReference -FreezeSha256 ((Get-FileHash -LiteralPath $freezePath -Algorithm SHA256).Hash.ToLowerInvariant())
        throw 'Post-build GPU preflight found competing use; no process was killed.'
    }

    $currentStage = 'installed_runtime_binding'
    $installSetup = Join-Path $workspacePath 'install\local_setup.ps1'
    $executor = Join-Path $workspacePath 'install\lib\action_stream_executor\action_stream_executor_node.exe'
    $router = Join-Path $workspacePath '.pixi\envs\default\Library\lib\rmw_zenoh_cpp\rmw_zenohd.exe'
    $ffprobe = Join-Path $workspacePath '.pixi\envs\default\Library\bin\ffprobe.exe'
    $ffmpeg = Join-Path $workspacePath '.pixi\envs\default\Library\bin\ffmpeg.exe'
    Require-File -Path $installSetup -Description 'fresh local overlay setup'
    Require-File -Path $executor -Description 'fresh C++ ActionStream executor'
    Require-File -Path $router -Description 'Zenoh router'
    Require-File -Path $ffprobe -Description 'installed ffprobe'
    Require-File -Path $ffmpeg -Description 'installed ffmpeg'
    $sourceAdapter = Join-Path $sourceRoot 'action_stream_isaac\action_stream_isaac\dynamic_isaac_adapter.py'
    Require-File -Path $sourceAdapter -Description 'dynamic adapter source'
    $sourceAdapterHash = (Get-FileHash -LiteralPath $sourceAdapter -Algorithm SHA256).Hash.ToLowerInvariant()
    $installedAdapters = @(
        Get-ChildItem -LiteralPath (Join-Path $workspacePath 'install') -Recurse -File -Filter dynamic_isaac_adapter.py
    )
    $matchingAdapters = @(
        $installedAdapters | Where-Object {
            (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() -eq $sourceAdapterHash
        }
    )
    if ($matchingAdapters.Count -lt 1) {
        throw 'Installed dynamic adapter does not match the current checkout.'
    }
    $installedAdapter = $matchingAdapters[0].FullName

    $currentStage = 'native_demo_execution'
    $videoDirectory = Split-Path -Parent $videoPath
    if (-not (Test-Path -LiteralPath $videoDirectory -PathType Container)) {
        New-Item -ItemType Directory -Path $videoDirectory | Out-Null
    }
    $escapedSetup = $installSetup.Replace("'", "''")
    $escapedExecutor = $executor.Replace("'", "''")
    $escapedBatch = $batchPath.Replace("'", "''")
    $escapedStagedVideo = $stagedVideoPath.Replace("'", "''")
    $environmentSetup = @(
        ". '$escapedSetup'",
        "$" + "env:OMNI_KIT_ACCEPT_EULA = 'YES'",
        "$" + "env:ROS_DISTRO = 'jazzy'",
        "$" + "env:RMW_IMPLEMENTATION = 'rmw_zenoh_cpp'"
    ) -join [Environment]::NewLine
    $routerCommand = "Set-Location '$escapedWorkspace'; & '$($router.Replace("'", "''"))'"
    $executorCommand = @(
        $environmentSetup,
        "& '$escapedExecutor' --ros-args -p use_sim_time:=true -p strategy:=aligned_async -p action_dimension:=7 -p 'safe_hold_command:=[0.307015,0.0,0.589907,3.141592653589793,0.0,0.0,1.0]' -p sync_periodic_replan:=true -p observation_topic:=/action_stream/observation -p inference_request_topic:=/action_stream/inference_request -p action_chunk_topic:=/action_stream/action_chunk -p robot_command_topic:=/action_stream/robot_command -p diagnostics_topic:=/action_stream/diagnostics -p runtime_event_topic:=/action_stream/events -p episode_control_topic:=/action_stream/episode_control"
    ) -join [Environment]::NewLine
    $adapterCommand = @(
        $environmentSetup,
        "Set-Location '$escapedRoot'",
        "python -m action_stream_isaac.dynamic_isaac_adapter --matrix-manifest '$escapedBatch' --expected-strategy aligned_async --headless false --max-episodes 1 --video-output '$escapedStagedVideo' --video-fps 20 --video-width 1280 --video-height 720 --video-finalize-timeout-seconds 120"
    ) -join [Environment]::NewLine

    $routerProcess = $null
    $executorProcess = $null
    $adapterProcess = $null
    try {
        $routerProcess = Start-PixiPowerShell -Command $routerCommand -StdoutPath (Join-Path $logDirectory 'router.stdout.log') -StderrPath (Join-Path $logDirectory 'router.stderr.log') -PixiPath $PixiExe -PixiManifest $pixiManifest
        Start-Sleep -Seconds 2
        if ($routerProcess.HasExited) {
            throw "Owned Zenoh router exited with code $($routerProcess.ExitCode)."
        }
        $executorProcess = Start-PixiPowerShell -Command $executorCommand -StdoutPath (Join-Path $logDirectory 'executor.stdout.log') -StderrPath (Join-Path $logDirectory 'executor.stderr.log') -PixiPath $PixiExe -PixiManifest $pixiManifest
        Start-Sleep -Seconds 2
        if ($executorProcess.HasExited) {
            throw "C++ executor exited with code $($executorProcess.ExitCode)."
        }
        $adapterProcess = Start-PixiPowerShell -Command $adapterCommand -StdoutPath (Join-Path $logDirectory 'adapter.stdout.log') -StderrPath (Join-Path $logDirectory 'adapter.stderr.log') -PixiPath $PixiExe -PixiManifest $pixiManifest
        if (-not $adapterProcess.WaitForExit($AdapterTimeoutSeconds * 1000)) {
            throw [System.TimeoutException]::new(
                "Dynamic Isaac demo adapter exceeded $AdapterTimeoutSeconds seconds."
            )
        }
        if (-not $adapterProcess.WaitForExit(5000)) {
            throw [System.TimeoutException]::new(
                'Adapter streams did not flush within 5 seconds.'
            )
        }
        if ($adapterProcess.ExitCode -ne 0) {
            throw "Dynamic Isaac demo adapter exited with code $($adapterProcess.ExitCode)."
        }
        if ($executorProcess.HasExited) {
            throw "C++ executor exited before adapter completion with code $($executorProcess.ExitCode)."
        }
        if ($routerProcess.HasExited) {
            throw "Owned router exited before adapter completion with code $($routerProcess.ExitCode)."
        }
    }
    finally {
        foreach ($process in @($adapterProcess, $executorProcess, $routerProcess)) {
            if ($null -ne $process) {
                Stop-OwnedProcessTree -RootProcessId $process.Id
            }
        }
    }

    $currentStage = 'native_artifact_validation'
    Require-File -Path $eventLogPath -Description 'complete native demo event log'
    Require-File -Path $summaryPath -Description 'complete native demo summary'
    Require-File -Path $stagedVideoPath -Description 'staged viewport MP4'
    foreach ($artifact in @($eventLogPath, $summaryPath, $stagedVideoPath)) {
        if ((Get-Item -LiteralPath $artifact).Length -le 0) {
            throw "Native demo artifact is empty: $artifact"
        }
    }
    if (
        (Get-FileHash -LiteralPath $scenarioPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $scenarioFileHashBefore -or
        (Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $traceFileHashBefore
    ) {
        throw 'Scenario or fault trace changed during demo execution.'
    }

    $currentStage = 'full_matrix_replay'
    $escapedMatrix = $matrixPath.Replace("'", "''")
    $escapedStagedReplay = $stagedReplayPath.Replace("'", "''")
    $replayCommand = @(
        $environmentSetup,
        "Set-Location '$escapedRoot'",
        "python -m action_stream_benchmark.m8_cli validate --manifest '$escapedMatrix' --output '$escapedStagedReplay'"
    ) -join [Environment]::NewLine
    Invoke-PixiPowerShell -Command $replayCommand -LogPath (Join-Path $logDirectory 'replay_validate.log') -PixiPath $PixiExe -PixiManifest $pixiManifest
    Require-File -Path $stagedReplayPath -Description 'independent single-matrix replay'
    if (Test-FileHasUtf8Bom -Path $stagedReplayPath) {
        throw 'Independent replay unexpectedly contains a UTF-8 BOM.'
    }
    $replay = Get-Content -Raw -LiteralPath $stagedReplayPath | ConvertFrom-Json
    $replayAudits = @($replay.audits)
    if (
        $replay.passed -ne $true -or
        $replay.split -ne 'development' -or
        [int]$replay.episode_count -ne 1 -or
        [int]$replay.episode_audits_passed -ne 1 -or
        $replay.fairness_passed -ne $true -or
        $replay.provenance_validation_passed -ne $true -or
        $replay.seed_validation_passed -ne $true -or
        $replay.profile_validation_passed -ne $true -or
        $replayAudits.Count -ne 1 -or
        $replayAudits[0].passed -ne $true -or
        $replayAudits[0].strategy -ne 'aligned_async'
    ) {
        throw 'Replay did not validate exactly one aligned Profile-1 episode.'
    }
    $summary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
    if (
        $summary.headline_eligible -ne $false -or
        $summary.native_isaac_physics -ne $true -or
        $summary.split -ne 'development' -or
        $summary.episode_id -ne $matrixEpisode.episode_id -or
        $summary.profile_id -ne 'profile_1_fixed' -or
        $summary.strategy -ne 'aligned_async' -or
        [int]$summary.seed -ne [int]$matrixEpisode.seed
    ) {
        throw 'Native summary identity does not match the demo episode.'
    }
    if ($summary.metrics.task_success -ne $true) {
        throw 'Demo episode failed; preserving development evidence without release promotion.'
    }

    $currentStage = 'video_verification'
    $ffprobeStdout = Join-Path $logDirectory 'ffprobe.json'
    $ffprobeStderr = Join-Path $logDirectory 'ffprobe.stderr.log'
    Invoke-BoundedNativeTool -Executable $ffprobe -Arguments @(
        '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration:format=duration',
        '-of', 'json', $stagedVideoPath
    ) -StdoutPath $ffprobeStdout -StderrPath $ffprobeStderr -TimeoutSeconds 120
    $probe = Get-Content -Raw -LiteralPath $ffprobeStdout | ConvertFrom-Json
    $streams = @($probe.streams)
    if ($streams.Count -ne 1) { throw 'ffprobe did not find exactly one video stream.' }
    $videoStream = $streams[0]
    $fps = Convert-RationalToDouble -Value ([string]$videoStream.avg_frame_rate)
    $frameCount = 0L
    if (-not [long]::TryParse([string]$videoStream.nb_read_frames, [ref]$frameCount)) {
        throw 'ffprobe did not report a numeric decoded frame count.'
    }
    $durationSeconds = Get-PositiveDouble -Primary $videoStream.duration -Fallback $probe.format.duration -Description 'decoded video duration'
    if (
        $videoStream.codec_name -ne 'h264' -or
        [int]$videoStream.width -ne 1280 -or
        [int]$videoStream.height -ne 720 -or
        [math]::Abs($fps - 20.0) -gt 1.0 -or
        $frameCount -lt 40 -or
        $durationSeconds -lt 2.0 -or
        [math]::Abs(($frameCount / $durationSeconds) - $fps) -gt 2.0
    ) {
        throw 'Video must be H.264 1280x720 at about 20 fps with nontrivial duration.'
    }
    $frameDirectory = Join-Path $logDirectory 'decoded_frame_samples'
    New-Item -ItemType Directory -Path $frameDirectory | Out-Null
    $sampleDefinitions = @(
        [pscustomobject]@{ label = 'first'; index = 0L },
        [pscustomobject]@{ label = 'middle'; index = [long][math]::Floor(($frameCount - 1) / 2.0) },
        [pscustomobject]@{ label = 'last'; index = $frameCount - 1 }
    )
    $frameSamples = @()
    foreach ($sample in $sampleDefinitions) {
        $framePath = Join-Path $frameDirectory "$($sample.label).png"
        Invoke-BoundedNativeTool -Executable $ffmpeg -Arguments @(
            '-v', 'error', '-i', $stagedVideoPath,
            '-vf', "select=eq(n\,$($sample.index))",
            '-frames:v', '1', '-y', $framePath
        ) -StdoutPath (Join-Path $logDirectory "ffmpeg_$($sample.label).stdout.log") -StderrPath (Join-Path $logDirectory "ffmpeg_$($sample.label).stderr.log") -TimeoutSeconds 120
        Require-File -Path $framePath -Description "$($sample.label) decoded frame"
        if ((Get-Item -LiteralPath $framePath).Length -le 0) {
            throw "Decoded $($sample.label) frame is empty."
        }
        $frameSamples += [pscustomobject][ordered]@{
            label = $sample.label
            frame_index = [long]$sample.index
            path = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $framePath
            sha256 = (Get-FileHash -LiteralPath $framePath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $distinctFrameHashes = @($frameSamples.sha256 | Sort-Object -Unique).Count
    if ($distinctFrameHashes -lt 2) {
        throw 'First, middle, and last decoded frames show no visual variation.'
    }

    $currentStage = 'artifact_promotion'
    Require-PathAbsent -Path $videoPath -Description 'release demo video'
    Require-PathAbsent -Path $replayPath -Description 'demo replay validation'
    [System.IO.File]::Move($stagedVideoPath, $videoPath)
    $videoPromoted = $true
    [System.IO.File]::Move($stagedReplayPath, $replayPath)
    $replayPromoted = $true
    Require-File -Path $videoPath -Description 'promoted release demo video'
    Require-File -Path $replayPath -Description 'promoted replay'

    $currentStage = 'completion_receipt'
    $receipt = [ordered]@{
        schema_version = 1
        milestone = 'M8-G0'
        receipt_kind = 'native_demo'
        status = 'complete'
        completed_utc = [DateTime]::UtcNow.ToString('o')
        headline_eligible = $false
        benchmark_evidence = $false
        evidence_scope = 'single frozen-development demonstration only'
        operator_authorized_native_gpu_run = $true
        matrix_manifest = $matrixReference
        matrix_manifest_sha256 = (Get-FileHash -LiteralPath $matrixPath -Algorithm SHA256).Hash.ToLowerInvariant()
        batch_manifest = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $batchPath
        batch_manifest_sha256 = (Get-FileHash -LiteralPath $batchPath -Algorithm SHA256).Hash.ToLowerInvariant()
        freeze_manifest = $freezeReference
        freeze_manifest_sha256 = (Get-FileHash -LiteralPath $freezePath -Algorithm SHA256).Hash.ToLowerInvariant()
        freeze_sha256 = [string]$freeze.freeze_sha256
        finalized_freeze_protocol_sha256 = [string]$freeze.protocol_sha256
        freeze_validation_passed = $true
        selected_candidate_id = $selectedCandidateId
        selected_candidate_protocol_role = $selectedCandidateRole
        selected_candidate_manifest = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $candidatePath
        selected_candidate_manifest_file_sha256 = $candidateHash
        selected_candidate_protocol_sha256 = [string]$developmentCalibration.selected_protocol_sha256
        protocol_sha256 = [string]$matrix.protocol_sha256
        episode = [ordered]@{
            episode_id = [string]$matrixEpisode.episode_id
            profile_id = 'profile_1_fixed'
            seed = [int]$matrixEpisode.seed
            strategy = 'aligned_async'
            task_success = [bool]$summary.metrics.task_success
            completion_reason = [string]$summary.metrics.completion_reason
            completion_steps = [int]$summary.metrics.completion_steps
        }
        raw_artifacts = [ordered]@{
            event_log = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $eventLogPath
            event_log_sha256 = (Get-FileHash -LiteralPath $eventLogPath -Algorithm SHA256).Hash.ToLowerInvariant()
            summary = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $summaryPath
            summary_sha256 = (Get-FileHash -LiteralPath $summaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
            scenario = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $scenarioPath
            scenario_file_sha256 = $scenarioFileHashBefore
            scenario_declared_sha256 = [string]$matrixEpisode.scenario_sha256
            fault_trace = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $tracePath
            fault_trace_file_sha256 = $traceFileHashBefore
            fault_trace_declared_sha256 = [string]$matrixEpisode.fault_trace_sha256
        }
        independent_replay = [ordered]@{
            path = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $replayPath
            sha256 = (Get-FileHash -LiteralPath $replayPath -Algorithm SHA256).Hash.ToLowerInvariant()
            full_matrix_replay = $true
            passed = $true
            episode_count = 1
            episode_audits_passed = 1
        }
        video = [ordered]@{
            path = $videoReference
            size_bytes = (Get-Item -LiteralPath $videoPath).Length
            sha256 = (Get-FileHash -LiteralPath $videoPath -Algorithm SHA256).Hash.ToLowerInvariant()
            codec = 'h264'
            width = 1280
            height = 720
            average_fps = $fps
            decoded_frame_count = $frameCount
            duration_seconds = $durationSeconds
            distinct_sample_frame_hash_count = $distinctFrameHashes
            decoded_frame_samples = $frameSamples
        }
        gpu_memory_refusal_threshold_mib = $gpuMemoryThresholdMiB
        initial_gpu_preflight = $initialGpuPreflight
        post_build_gpu_preflight = $postBuildGpuPreflight
        provenance = [ordered]@{
            source_root = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $sourceRoot
            source_file_count = $sourceRecords.Count
            source_manifest_sha256 = $sourceManifestHash
            source_dynamic_adapter = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $sourceAdapter
            source_dynamic_adapter_sha256 = $sourceAdapterHash
            installed_artifact_base = 'isaac_workspace'
            installed_dynamic_adapter = Get-ContainedRelativePath -BasePath $workspacePath -TargetPath $installedAdapter
            installed_dynamic_adapter_sha256 = (Get-FileHash -LiteralPath $installedAdapter -Algorithm SHA256).Hash.ToLowerInvariant()
            local_setup = Get-ContainedRelativePath -BasePath $workspacePath -TargetPath $installSetup
            local_setup_sha256 = (Get-FileHash -LiteralPath $installSetup -Algorithm SHA256).Hash.ToLowerInvariant()
            executor = Get-ContainedRelativePath -BasePath $workspacePath -TargetPath $executor
            executor_sha256 = (Get-FileHash -LiteralPath $executor -Algorithm SHA256).Hash.ToLowerInvariant()
            router = Get-ContainedRelativePath -BasePath $workspacePath -TargetPath $router
            router_sha256 = (Get-FileHash -LiteralPath $router -Algorithm SHA256).Hash.ToLowerInvariant()
            ffprobe = Get-ContainedRelativePath -BasePath $workspacePath -TargetPath $ffprobe
            ffprobe_sha256 = (Get-FileHash -LiteralPath $ffprobe -Algorithm SHA256).Hash.ToLowerInvariant()
            ffmpeg = Get-ContainedRelativePath -BasePath $workspacePath -TargetPath $ffmpeg
            ffmpeg_sha256 = (Get-FileHash -LiteralPath $ffmpeg -Algorithm SHA256).Hash.ToLowerInvariant()
            runner = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $PSCommandPath
            runner_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
            colcon_merge_install = $true
            adapter_timeout_seconds = $AdapterTimeoutSeconds
        }
        run_evidence_directory = Get-RepositoryReference -RepositoryRoot $repositoryRoot -TargetPath $logDirectory
        run_artifacts = @(Get-AvailableArtifactRecords -LogDirectory $logDirectory)
        unrelated_processes_killed = 0
    }
    Write-Utf8JsonNoBom -Path $receiptPath -InputObject $receipt
    $currentStage = 'complete'
    Write-Host "Recorded replay-clean non-headline demo: $videoPath"
    Write-Host "Independent replay: $replayPath"
    Write-Host "Portable receipt: $receiptPath"
}
catch {
    $originalError = $_
    if (
        $currentStage -eq 'completion_receipt' -and
        (Test-Path -LiteralPath $receiptPath -PathType Leaf)
    ) {
        try {
            $incompleteReceiptPath = Join-Path $logDirectory 'incomplete_demo_receipt.json'
            [System.IO.File]::Move($receiptPath, $incompleteReceiptPath)
        }
        catch {
            Write-Warning "Could not preserve the partial demo receipt: $($_.Exception.Message)"
        }
    }
    if (
        $replayPromoted -and
        (Test-Path -LiteralPath $replayPath -PathType Leaf) -and
        -not (Test-Path -LiteralPath $stagedReplayPath)
    ) {
        try {
            [System.IO.File]::Move($replayPath, $stagedReplayPath)
            $replayPromoted = $false
        }
        catch {
            Write-Warning "Could not roll back the promoted replay: $($_.Exception.Message)"
        }
    }
    if (
        $videoPromoted -and
        (Test-Path -LiteralPath $videoPath -PathType Leaf) -and
        -not (Test-Path -LiteralPath $stagedVideoPath)
    ) {
        try {
            [System.IO.File]::Move($videoPath, $stagedVideoPath)
            $videoPromoted = $false
        }
        catch {
            Write-Warning "Could not roll back the promoted video: $($_.Exception.Message)"
        }
    }
    if (-not (Test-Path -LiteralPath $refusalReceiptPath -PathType Leaf)) {
        $attemptReceiptPath = if ($originalError.Exception -is [System.TimeoutException]) {
            $timeoutReceiptPath
        } else { $failureReceiptPath }
        try {
            Write-DemoFailureReceipt -Path $attemptReceiptPath -Stage $currentStage -ErrorRecord $originalError -RepositoryRoot $repositoryRoot -WorkspacePath $workspacePath -MatrixPath $matrixPath -FreezePath $freezePath -VideoPath $videoPath -LogDirectory $logDirectory -TimeoutSeconds $AdapterTimeoutSeconds -SourceManifestSha256 $sourceManifestHash -InitialGpuPreflight $initialGpuPreflight -PostBuildGpuPreflight $postBuildGpuPreflight
            Write-Warning "Native demo failed at stage '$currentStage'; receipt: $attemptReceiptPath"
        }
        catch {
            Write-Warning "Native demo failed at stage '$currentStage', and its failure receipt could not be written: $($_.Exception.Message)"
        }
    } else {
        Write-Warning "Native demo was refused at stage '$currentStage'; receipt: $refusalReceiptPath"
    }
    throw $originalError
}
