[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{2,127}$')]
    [string]$EpisodeId = "m7-isaac-visual-demo-$(Get-Date -Format 'yyyyMMdd-HHmmss')",
    [switch]$Headless,
    [switch]$RecordRosbag,
    [string]$VideoOutput = '',
    [string]$ActionStreamRoot = 'C:\Users\cgliu\OneDrive\Documents\ActionStream',
    [string]$IsaacWorkspace = 'C:\IsaacSim-ros_workspaces\jazzy_ws',
    [string]$PixiExe = 'C:\Users\cgliu\AppData\Local\pixi\bin\pixi.exe',
    [string]$ExecutorParams = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Require-ExistingPath {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Description)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Description was not found: $Path"
    }
}

Require-ExistingPath -Path $ActionStreamRoot -Description 'ActionStream checkout'
Require-ExistingPath -Path $IsaacWorkspace -Description 'Official Isaac Sim ROS workspace'
Require-ExistingPath -Path $PixiExe -Description 'Pixi executable'

if ([string]::IsNullOrWhiteSpace($ExecutorParams)) {
    $ExecutorParams = Join-Path $ActionStreamRoot 'outputs\m7_g0\isaac_smoke\executor.params.yaml'
}
Require-ExistingPath -Path $ExecutorParams -Description 'Validated executor parameter file'
$ExecutorParams = [System.IO.Path]::GetFullPath($ExecutorParams)

if ([string]::IsNullOrWhiteSpace($VideoOutput)) {
    $VideoOutput = Join-Path $ActionStreamRoot "outputs\m7_g0\demo\$EpisodeId.mp4"
}
$VideoOutput = [System.IO.Path]::GetFullPath($VideoOutput)
if ([System.IO.Path]::GetExtension($VideoOutput) -ne '.mp4') {
    throw "VideoOutput must end in .mp4: $VideoOutput"
}
if (Test-Path -LiteralPath $VideoOutput) {
    throw "Refusing to overwrite existing video: $VideoOutput"
}
$frameDirectory = Join-Path (Split-Path -Parent $VideoOutput) "$([System.IO.Path]::GetFileNameWithoutExtension($VideoOutput))_frames"
if (Test-Path -LiteralPath $frameDirectory) {
    throw "Refusing to reuse stale capture frames: $frameDirectory"
}
$videoDirectory = Split-Path -Parent $VideoOutput
if (-not (Test-Path -LiteralPath $videoDirectory)) {
    New-Item -ItemType Directory -Path $videoDirectory -Force | Out-Null
}

$manifest = Join-Path $IsaacWorkspace 'pixi.toml'
$installSetup = Join-Path $IsaacWorkspace 'install\setup.ps1'
$executor = Join-Path $IsaacWorkspace 'install\lib\action_stream_executor\action_stream_executor_node.exe'
$router = Join-Path $IsaacWorkspace '.pixi\envs\default\Library\lib\rmw_zenoh_cpp\rmw_zenohd.exe'
Require-ExistingPath -Path $manifest -Description 'Official Pixi manifest'
Require-ExistingPath -Path $installSetup -Description 'Native ActionStream colcon setup script'
Require-ExistingPath -Path $executor -Description 'Native ActionStream executor'
Require-ExistingPath -Path $router -Description 'Zenoh router executable'

if ($RecordRosbag) {
    throw @"
Rosbag recording was requested, but this audited Windows environment has a known
standalone pre-Kit rclpy/ros2 CLI DLL loader blocker. Do not work around it by
preloading private DLLs. Clear that documented blocker in an officially
supported deployment first, then run the exact bag command in
docs\m7_environment.md before START. No bag was started by this script.
"@
}

$headlessValue = if ($Headless) { 'true' } else { 'false' }
$envSetup = @"
. '$installSetup'
`$env:OMNI_KIT_ACCEPT_EULA = 'YES'
`$env:ROS_DISTRO = 'jazzy'
`$env:RMW_IMPLEMENTATION = 'rmw_zenoh_cpp'
"@

# Pixi supplies the official Python 3.12/Isaac/ROS environment. The adapter
# itself loads SimulationApp and isaacsim.ros2.bridge before importing rclpy;
# this script never preloads DLLs or substitutes the deterministic test plant.
$routerCommand = @"
Set-Location '$IsaacWorkspace'
& '$router'
"@
$executorCommand = @"
$envSetup
& '$executor' --ros-args --params-file '$ExecutorParams'
"@
$adapterCommand = @"
$envSetup
Set-Location '$ActionStreamRoot'
python -m action_stream_isaac.isaac_adapter --headless $headlessValue --auto-start-episode '$EpisodeId' --exit-after-auto-episode true --command-timeout-seconds 10 --video-output '$VideoOutput'
"@

function Start-PixiPowerShell {
    param([Parameter(Mandatory = $true)][string]$Command)
    return Start-Process -FilePath $PixiExe -ArgumentList @(
        'run', '--manifest-path', $manifest, '--', 'powershell.exe', '-NoProfile',
        '-ExecutionPolicy', 'Bypass', '-Command', $Command
    ) -PassThru -WindowStyle Hidden
}

function Stop-OwnedProcessTree {
    param([Parameter(Mandatory = $true)][int]$RootProcessId)

    $snapshot = @(Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId)
    $frontier = [System.Collections.Generic.Queue[int]]::new()
    $frontier.Enqueue($RootProcessId)
    $descendants = [System.Collections.Generic.List[int]]::new()
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

Write-Host "Starting bounded M7 Isaac safe-hold recording: $EpisodeId"
Write-Host "MP4 output: $VideoOutput"
Write-Host 'This safe-hold demonstration is not a policy-driven reach/lift benchmark.'

$routerProcess = $null
$executorProcess = $null
$adapterProcess = $null
try {
    $routerProcess = Start-PixiPowerShell -Command $routerCommand
    Start-Sleep -Seconds 2
    $executorProcess = Start-PixiPowerShell -Command $executorCommand
    Start-Sleep -Seconds 2
    $adapterProcess = Start-PixiPowerShell -Command $adapterCommand
    $adapterProcess.WaitForExit()
    if ($adapterProcess.ExitCode -ne 0) {
        throw "Isaac adapter exited with code $($adapterProcess.ExitCode). Inspect its window/log before retrying."
    }
    if (-not (Test-Path -LiteralPath $VideoOutput -PathType Leaf)) {
        throw "Isaac adapter exited successfully but did not create the requested MP4: $VideoOutput"
    }
    $video = Get-Item -LiteralPath $VideoOutput
    if ($video.Length -le 0) {
        throw "Isaac adapter created an empty MP4: $VideoOutput"
    }
    $videoHash = (Get-FileHash -LiteralPath $VideoOutput -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "Completed bounded safe-hold recording: $EpisodeId"
    Write-Host "MP4 bytes: $($video.Length); SHA256: $videoHash"
}
finally {
    foreach ($process in @($adapterProcess, $executorProcess, $routerProcess)) {
        if ($null -ne $process) {
            Stop-OwnedProcessTree -RootProcessId $process.Id
        }
    }
}
