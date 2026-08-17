# M8-G0 environment and reproduction boundary

This document separates locally verified native-Isaac capability from results
that have actually completed.  The deterministic ROS test plant remains useful
for regression tests, but it is not an eligible M8 headline endpoint.

## Audited starting state

M8-G0 began on `master` at
`1c51d3e0489a57224c4b4d2f4f6ddafae5302713`, three commits ahead of and zero
behind `origin/master`, with a clean worktree.  The accepted M3 through M7 tree
objects were recorded before edits in
`outputs/m8_g0/audit/initial_repository_environment_audit.json`.

The host audit recorded:

| Component | Verified version |
|---|---|
| Host | Windows 11 Home build 26200, x86-64 |
| GPU | NVIDIA RTX 4090, 24,564 MiB |
| Driver | 591.86 |
| CUDA toolkit | 13.0.88 |
| Host Python / PyTorch | 3.13.5 / 2.6.0+cu124 |
| WSL | Ubuntu 24.04.4, kernel 6.6.87.2 |
| WSL compiler | GCC/G++ 13.3.0, CMake 3.28.3 |
| Docker | client/server 29.3.1 |
| Visual Studio | Community 2022 17.14.17 selected |
| Pixi | 0.75.0 |
| Native Python / Isaac Sim | 3.12.13 / 6.0.1.0 |
| ROS / middleware | Jazzy / rmw-zenoh-cpp 0.2.9 |

The formal native environment is the official
`https://github.com/isaac-sim/IsaacSim-ros_workspaces.git` repository's
`jazzy_ws` at commit `dd3eeede7912755996a18f4884285d9f50843f79`. The canonical
Git blob for `jazzy_ws/pixi.toml` uses LF, is 5,467 bytes, and has SHA-256
`b4e7a34c264e88f19ba6bfb3c7a72ee46b0843f3e6eb7e75619dc0ebb87b313d`.
The audited all-CRLF form is 5,618 bytes with SHA-256
`9649bf57644781a1fe0203ed6b80828ccb42ea07555475080e5d11a9b0c3e1ae`.
Cross-platform checks must accept only those two byte forms after proving that
CRLF-to-LF normalization equals the canonical blob; this is a newline
portability rule, not permission to change manifest content.
The same rule applies to `jazzy_ws/pixi.lock`: its canonical LF Git blob is
1,491,808 bytes with SHA-256
`ba8e59eef962cbf49a1ff06ff947ed5eaa4547d018389b048771a1e7d8bb890d`;
the audited Windows CRLF working copy is 1,533,045 bytes with SHA-256
`2c2f9097b129847735b5abb805045a22076a731e2138c8192caac95d09a2866e`.
The installed supported manipulator path is
`isaacsim.robot.experimental.manipulators.examples.franka.Franka`.  M8 uses
its damped-least-squares end-effector controller and the official downward
orientation; it does not substitute the deterministic ROS plant.

## GPU safety preflight

The first preflight observed approximately 15.8 GiB of VRAM in use by three
unrelated Unity Editor processes. A formal runner invocation on 2026-08-03
refused at 15,565 MiB against its 4,096 MiB threshold; the superseded-suite
marker preserves that attempt without retaining a redundant per-run log tree.
After the current candidate-0 suite was regenerated, a later formal invocation
first passed strict current-source validation for all 20 planned episodes, then
refused at the same gate with 16,022 MiB in use. A compact redacted receipt is
retained at `outputs/m8_g0/audit/native_preflight_refusal.json`; the raw runner
receipts and transcript are hash-recorded but intentionally not committed.
No unrelated process was terminated and Isaac was not started in either
attempt. Every native run repeats:

```powershell
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

On Windows/WDDM, `nvidia-smi` also reports ordinary graphics processes with
`used_memory=N/A`. The runner retains those rows for audit but regards only a
positive numeric process allocation as actionable; the independent total-used
VRAM threshold still fails closed. An inability to obtain a safe native-Isaac
slot is an environment limitation, not a zero-success policy result. It cannot
be replaced by ROS test-plant evidence.

## Build and non-GPU validation

The pinned ROS Docker image is
`actionstream-m7-ros:jazzy@sha256:64505ceaa908d1158844f0470fee5d8cda6ad0898ff1f94ad9e5193a3c34c952`.
It is valid for the C++ state-machine build, differential probe, ROS interface
generation, and CPU tests only.

```powershell
docker run --rm --volume "${PWD}:/workspace/repo:ro" `
  --workdir /workspace/build_ws actionstream-m7-ros:jazzy bash -lc '
    set -eo pipefail
    source /opt/ros/jazzy/setup.bash
    colcon build --base-paths /workspace/repo/ros2_ws/src \
      --packages-select action_stream_msgs action_stream_executor \
        action_stream_policy action_stream_benchmark action_stream_isaac \
      --merge-install --cmake-args -DBUILD_TESTING=ON
    source install/setup.bash
    colcon test --base-paths /workspace/repo/ros2_ws/src --merge-install \
      --python-testing pytest \
      --packages-select action_stream_msgs action_stream_executor \
        action_stream_policy action_stream_benchmark action_stream_isaac \
      --event-handlers console_direct+
    colcon test-result --verbose
  '
```

The final validation record reports the exact test count produced by this
invocation, including the C++ state-machine executable. The read-only mount
ensures Docker cannot alter the checkout.

Run the repository tests with the pinned local Python 3.12 environment and all
source-layout packages visible:

```powershell
$repoSrc = (Resolve-Path 'src').Path
$benchmarkSrc = (Resolve-Path 'ros2_ws/src/action_stream_benchmark').Path
$isaacSrc = (Resolve-Path 'ros2_ws/src/action_stream_isaac').Path
$policySrc = (Resolve-Path 'ros2_ws/src/action_stream_policy').Path
$env:PYTHONPATH = "$repoSrc;$benchmarkSrc;$isaacSrc;$policySrc"
& '.external/m6_venv312/Scripts/python.exe' -m pytest -q
```

## Baseline, freeze, and native execution

### Linux/cloud native runner

`scripts/m8_run_isaac.sh` is the only runner accepted for a portable formal
native receipt. It expects Linux, the exact official repository/commit above,
a tracked and clean `jazzy_ws/pixi.toml` plus `jazzy_ws/pixi.lock`, Bash, host
Python 3.9 or newer, `setsid`, `ps`, GNU `find`/`sort`, and a working
`nvidia-smi`. It also requires the official Linux Pixi 0.75.0 executable:
77,311,024 bytes, SHA-256
`4383aed18b2d5569cf34a19638daf954aa4415cc87ad3a9da9f34059cc4a004c`,
with literal version output `pixi 0.75.0`. Every Pixi launch uses
`pixi run --frozen --manifest-path ...`; no lock update or solve is permitted.

The cloud runner preserves the same matrix, protocol, seed, profile, scenario,
and fault-trace bytes. It performs current-source `--validate-only` checks
before touching the GPU, samples only the explicitly selected GPU before and
after the build, requires the same GPU UUID both times, and binds every native
process to that UUID. It fails if that GPU has any compute-process row or more
than 4,096 MiB in use. It builds the current checkout, requires the Linux
overlay's `setup.bash` and `local_setup.bash` to be nonempty regular non-symlink
files, and locates the executor, Zenoh router, and installed adapter. It then
copies the exact runner, runner-support module, installed adapter, built C++
executor, `pixi.toml`, and `pixi.lock` bytes into the receipt directory. Its
dependency-free receipt helpers use host Python 3.9+ when present, or only the
Python executable inside the pinned Pixi `default` environment on a minimal
host with no system Python.
`external_environment.json` binds those copies to the official URL, commit,
workspace, Pixi binary, Python 3.12.13, Isaac Sim packages 6.0.1.0, ROS Jazzy,
`rclpy` 7.1.9, `rosgraph-msgs` 2.0.3, and `rmw_zenoh_cpp` 0.2.9. The preflight
also cross-links the selected GPU index, UUID, name, driver, and total memory
to both GPU snapshots and its top-level identity. These receipt-local copies
remain independently verifiable after the rental install directory disappears.

Each router, executor, and Kit-first adapter starts in a new `setsid` process
group. Cleanup signals only the PID/PGID pairs created by that invocation; the
script contains no process-name kill, GPU reset, or host shutdown operation.
It refuses existing canonical replay/analysis/figure outputs and writes a
timestamped failure, timeout, GPU-refusal, preflight, or completion receipt
without overwriting a previous attempt.

From either a fresh Linux checkout or a byte-for-byte verified minimal archive
extraction whose `.git/ARCHIVE_CHECKOUT` marker records the local source commit
and archive SHA-256, and whose M8 inputs match the intended local checkout:

```bash
cd /workspace/ActionStream

isaac_ws=/workspace/IsaacSim-ros_workspaces/jazzy_ws
pixi_exe="$(command -v pixi)"

nvidia-smi \
  --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
test -f "$isaac_ws/pixi.toml"
test -f "$isaac_ws/pixi.lock"

bash scripts/m8_run_isaac.sh \
  --matrix-suite-manifest \
    outputs/m8_g0/baseline_gate/candidate_0/matrix.json \
  --isaac-workspace "$isaac_ws" \
  --pixi-exe "$pixi_exe" \
  --gpu-index 0 \
  --headless true \
  --batch-timeout-seconds 14400 \
  --authorize-native-gpu-run
```

The authorization flag approves native execution only. It does not authorize
the runner to terminate unrelated processes, stop/delete the rental instance,
advance the calibration ledger, create development/holdout matrices, or cross
the 18/20 baseline gate. A successful baseline must still be copied back with
its complete `native_run_logs/<successful-run>/` directory and raw episode
artifacts, replayed locally, and passed to `baseline-record` below. The local
structural and helper tests for this path do not themselves constitute a
native-Isaac result.

### Audited Windows runner (nonformal)

The PowerShell runner remains for legacy Windows audits, but the portable
formal validator rejects Windows completion receipts. It cannot substitute for
the Linux runner's exact Pixi, manifest/lock, runtime, and selected-GPU proof.

Make the source-layout packages visible once per PowerShell session:

```powershell
$env:PYTHONPATH = @(
  (Resolve-Path 'ros2_ws/src/action_stream_benchmark').Path
  (Resolve-Path 'ros2_ws/src/action_stream_isaac').Path
  (Resolve-Path 'ros2_ws/src/action_stream_policy').Path
) -join ';'
$py = (Resolve-Path '.external/m6_venv312/Scripts/python.exe').Path
$pixi = "$env:LOCALAPPDATA\pixi\bin\pixi.exe"
$isaacWs = 'C:\IsaacSim-ros_workspaces\jazzy_ws'
```

Create the preregistered 20-seed, Profile-0 synchronous baseline matrix. This
materializes scenarios and traces only; `native_results_status_at_creation`
remains an immutable creation fact, while later completion is proved by replay
and signed-by-hash runner receipts.

```powershell
& $py -m action_stream_benchmark.m8_cli matrix-create `
  --repository-root . `
  --candidate-protocol configs/m8_g0.json `
  --seeds configs/m8_baseline_gate_seeds.json `
  --profile ros2_ws/src/action_stream_benchmark/config/m8_profile_0_sanity.json `
  --split baseline_gate `
  --output-root outputs/m8_g0/baseline_gate/candidate_0/raw `
  --manifest outputs/m8_g0/baseline_gate/candidate_0/matrix.json
```

```bash
bash scripts/m8_run_isaac.sh \
  --matrix-suite-manifest outputs/m8_g0/baseline_gate/candidate_0/matrix.json \
  --isaac-workspace /root/autodl-tmp/IsaacSim-ros_workspaces/jazzy_ws \
  --pixi-exe /root/autodl-tmp/bin/pixi \
  --gpu-index 0 \
  --headless true \
  --batch-timeout-seconds 14400 \
  --authorize-native-gpu-run
```

The runner refuses a busy GPU, rebuilds the current checkout into the native
workspace, starts only its own Zenoh/executor/Kit-first process tree, records
stdout/stderr and source hashes, and independently writes
`outputs/m8_g0/baseline_gate/candidate_0/replay_validation.json`. Record the
successful native run before creating any development matrix; this command
replays the raw evidence, requires at least 18/20 successes, validates the
completion/preflight/source/runner/installed-adapter/built-executor/process-log
bindings, and
requires an exact nonzero current-source validate-only result for every
persistent batch with every validation log bound into the process-log archive.
It then atomically advances only a pristine open ledger:

```powershell
& $py -m action_stream_benchmark.m8_cli baseline-record `
  --repository-root . `
  --calibration-ledger configs/m8_calibration_ledger.json `
  --baseline-seeds configs/m8_baseline_gate_seeds.json `
  --profile ros2_ws/src/action_stream_benchmark/config/m8_profile_0_sanity.json `
  --matrix outputs/m8_g0/baseline_gate/candidate_0/matrix.json `
  --replay outputs/m8_g0/baseline_gate/candidate_0/replay_validation.json `
  --completion-receipt `
    outputs/m8_g0/baseline_gate/candidate_0/native_run_logs/<successful-run>/completion_receipt.json
```

Profile 1/2 development and holdout are prohibited unless all 20 baseline
episodes replay and at least 18 succeed.

Only after that gate passes, create candidate 0 on the disjoint 12-seed
development split. This is non-headline evidence. Never reuse an output path;
if a bounded calibration is necessary, retain the immutable candidate and copy
its protocol to a new file for `candidate_1` or `candidate_2`. If a latency
profile changes, retain the old profile and write a new profile file under a
candidate configuration directory outside `ros2_ws/src` (for example,
`configs/m8_candidates/candidate_1/`); every candidate protocol binds its exact
profile paths and byte SHA-256 values. Do not edit controller source after the
baseline: the native preflight source fingerprint deliberately makes that a
hard failure. Controller calibration is through the bounded protocol values. A
maximum of two changes therefore means at most three candidates: candidate 0,
candidate 1, and candidate 2.

```powershell
& $py -m action_stream_benchmark.m8_cli matrix-create `
  --repository-root . `
  --candidate-protocol configs/m8_g0.json `
  --seeds configs/m8_development_seeds.json `
  --profile ros2_ws/src/action_stream_benchmark/config/m8_profile_1_fixed.json `
  --profile ros2_ws/src/action_stream_benchmark/config/m8_profile_2_faults.json `
  --split development `
  --output-root outputs/m8_g0/development/candidate_0/raw `
  --manifest outputs/m8_g0/development/candidate_0/matrix.json
```

```bash
bash scripts/m8_run_isaac.sh \
  --matrix-suite-manifest outputs/m8_g0/development/candidate_0/matrix.json \
  --isaac-workspace /root/autodl-tmp/IsaacSim-ros_workspaces/jazzy_ws \
  --pixi-exe /root/autodl-tmp/bin/pixi \
  --gpu-index 0 \
  --headless true \
  --batch-timeout-seconds 14400 \
  --authorize-native-gpu-run
```

```powershell
& $py -m action_stream_benchmark.m8_cli candidate-record `
  --repository-root . `
  --calibration-ledger configs/m8_calibration_ledger.json `
  --candidate-id candidate_0 `
  --protocol configs/m8_g0.json `
  --matrix outputs/m8_g0/development/candidate_0/matrix.json `
  --replay outputs/m8_g0/development/candidate_0/replay_validation.json `
  --development-seeds configs/m8_development_seeds.json
```

Each candidate must independently pass raw replay. Development summaries may
guide at most the two recorded calibrations, but they are never copied into
the frozen holdout analysis. Record `candidate_1` or `candidate_2` only after
creating and natively running a new immutable protocol/matrix directory. The
same command requires `--reason '<development-only reason>'` for those two
candidates. It derives the exact adjacent contract diff itself; no operator
enters `before` or `after` values. A third change, an out-of-order ID, a reused
artifact path, a failed replay, or an unauthorized field fails without
changing the ledger. Moving a profile, changing only its byte hash, or changing
only `candidate_status` is metadata, not a calibration: each recorded change
must also alter at least one allowlisted runtime task, controller, or numeric
profile value.

For example, after independently creating and running `candidate_1`:

```powershell
& $py -m action_stream_benchmark.m8_cli candidate-record `
  --repository-root . `
  --calibration-ledger configs/m8_calibration_ledger.json `
  --candidate-id candidate_1 `
  --protocol configs/m8_candidates/candidate_1/m8_g0.json `
  --matrix outputs/m8_g0/development/candidate_1/matrix.json `
  --replay outputs/m8_g0/development/candidate_1/replay_validation.json `
  --development-seeds configs/m8_development_seeds.json `
  --reason 'bounded stability adjustment justified by candidate_0 replay'
```

For every candidate, the staged ledger records the
repository-relative protocol, matrix, and replay paths; their byte/canonical
hashes; the composite protocol-plus-profile contract hash; the seed count; and
the non-headline/replay status. Every change is an ordered
`candidate_N -> candidate_N+1` transition with an exact JSON-pointer
`before`/`after` diff, both contract hashes, a reason, and the preceding
candidate's matrix/replay hashes as development evidence.

The closed-ledger shape is intentionally explicit (hash values abbreviated
here only):

```json
{
  "bounded_calibration_changes": [{
    "change_id": "calibration_1",
    "before_candidate_id": "candidate_0",
    "after_candidate_id": "candidate_1",
    "reason": "development-only reason",
    "before_contract_sha256": "<64 hex>",
    "after_contract_sha256": "<64 hex>",
    "before": {"/protocol/controller/maximum_translation_per_step_m": 0.01},
    "after": {"/protocol/controller/maximum_translation_per_step_m": 0.008},
    "development_evidence": {
      "candidate_id": "candidate_0",
      "matrix_sha256": "<64 hex>",
      "replay_sha256": "<64 hex>"
    }
  }],
  "development": {
    "status": "complete",
    "seed_count_per_candidate": 12,
    "headline_eligible": false,
    "selected_candidate_id": "candidate_1",
    "candidates": ["<full candidate_0 record>", "<full candidate_1 record>"]
  }
}
```

Each full candidate record contains `protocol_path`,
`protocol_file_sha256`, canonical `protocol_sha256`, composite
`contract_sha256`, `matrix_path`/`matrix_sha256`,
`replay_path`/`replay_sha256`, `seed_count`, `replay_validated: true`, and
`headline_eligible: false`.

Only controller stability parameters, physical object/target ranges,
destination-switch timing, and the bounded Profile 1/2 latency/fault fields
are calibratable. Success tolerances, stable-placement requirements, collision
thresholds, runtime semantics, paired-reset fairness, gates, statistics, and
Profile 0 are not. The validator derives the diff from the immutable candidate
files; a ledger-authored diff cannot authorize an extra change.

Once the selected candidate is fixed, create `configs/m8_g0_frozen.json` from
that selected candidate by changing only `protocol_status` to
`ready_for_holdout_freeze` and `holdout_freeze_status` to `ready_to_freeze`.
Close and select it with the supported command (replace `candidate_0` with the
actual selection). The command replays every recorded development candidate,
verifies the exact diff chain, checks that the finalized behavioral contract
equals the selection, then atomically authorizes freeze:

```powershell
& $py -m action_stream_benchmark.m8_cli ledger-close `
  --repository-root . `
  --calibration-ledger configs/m8_calibration_ledger.json `
  --selected-candidate candidate_0 `
  --protocol configs/m8_g0_frozen.json `
  --development-seeds configs/m8_development_seeds.json
```

The freeze command reruns each candidate replay, verifies the exact ordered change
chain, binds every historical candidate input, and exposes the immutable
selected candidate through
`development_calibration.selected_candidate_protocol_role`. That role is
`baseline_candidate_protocol` for candidate 0 and
`development:candidate_N:protocol` for candidate 1 or 2.

After the baseline and any bounded development calibration complete, the
closed ledger and finalized protocol remain tracked working-tree artifacts for
the single final M8 commit; they are not committed in an intermediate step.
Create the immutable holdout freeze with every evidence and executable
dependency bound:

```powershell
$frozenProtocol = Get-Content configs/m8_g0_frozen.json -Raw | ConvertFrom-Json
$profile0 = $frozenProtocol.profiles.profile_0_sanity.profile_file
$profile1 = $frozenProtocol.profiles.profile_1_fixed.profile_file
$profile2 = $frozenProtocol.profiles.profile_2_faults.profile_file

& $py -m action_stream_benchmark.m8_cli freeze `
  --repository-root . `
  --protocol configs/m8_g0_frozen.json `
  --baseline-seeds configs/m8_baseline_gate_seeds.json `
  --development-seeds configs/m8_development_seeds.json `
  --holdout-seeds configs/m8_holdout_seeds.json `
  --calibration-ledger configs/m8_calibration_ledger.json `
  --baseline-matrix outputs/m8_g0/baseline_gate/candidate_0/matrix.json `
  --baseline-replay outputs/m8_g0/baseline_gate/candidate_0/replay_validation.json `
  --baseline-completion-receipt `
    outputs/m8_g0/baseline_gate/candidate_0/native_run_logs/<successful-run>/completion_receipt.json `
  --profile $profile0 `
  --profile $profile1 `
  --profile $profile2 `
  --output outputs/m8_g0/protocol/freeze_manifest.json

& $py -m action_stream_benchmark.m8_cli freeze-validate `
  --repository-root . `
  --manifest outputs/m8_g0/protocol/freeze_manifest.json
```

`<successful-run>` is the timestamped directory recorded in the completed
baseline evidence, never the GPU-refusal directory. The validator reruns raw
baseline replay, recomputes exact raw-reset integrity hashes, applies the
preregistered paired physical-reset tolerances, and recomputes the exact success count,
validates the native completion/preflight receipts, and checks that only the
two protocol status fields changed between the selected candidate and final
contract. Any allowed differences from the baseline candidate must be the exact
one- or two-change development chain recorded above.

Create the 60-seed frozen suite, then execute it with the formal Linux runner:

```powershell
$freeze = Get-Content outputs/m8_g0/protocol/freeze_manifest.json -Raw | ConvertFrom-Json
$profile0 = ($freeze.inputs | Where-Object role -eq 'profile:profile_0_sanity').path
$profile1 = ($freeze.inputs | Where-Object role -eq 'profile:profile_1_fixed').path
$profile2 = ($freeze.inputs | Where-Object role -eq 'profile:profile_2_faults').path

& $py -m action_stream_benchmark.m8_cli matrix-create `
  --repository-root . `
  --freeze-manifest outputs/m8_g0/protocol/freeze_manifest.json `
  --seeds configs/m8_holdout_seeds.json `
  --profile $profile0 `
  --profile $profile1 `
  --profile $profile2 `
  --split frozen_holdout `
  --output-root outputs/m8_g0/holdout/raw `
  --manifest outputs/m8_g0/holdout/matrix.json

```

```bash
bash scripts/m8_run_isaac.sh \
  --matrix-suite-manifest outputs/m8_g0/holdout/matrix.json \
  --isaac-workspace /root/autodl-tmp/IsaacSim-ros_workspaces/jazzy_ws \
  --pixi-exe /root/autodl-tmp/bin/pixi \
  --gpu-index 0 \
  --headless true \
  --batch-timeout-seconds 14400 \
  --authorize-native-gpu-run
```

The runner must start `rmw_zenohd`, the compiled C++ executor, and one headless
Isaac `SimulationApp`, retain that app across a bounded batch, use reset rather
than a cold process per episode, and clean up only processes it owns.  Profile
0 runs first.  Profiles 1 and 2 are prohibited unless the native synchronous
baseline reaches at least 18/20 successes with deterministic reset and no
systematic controller failure.

## Replay, analysis, figures, and archive

For a frozen holdout, the formal Linux `m8_run_isaac.sh` runner creates the canonical
`replay_validation.json`, `analysis.json`, and `figures/` outputs next to the
matrix, and binds the replay and analysis hashes into its completion receipt.
Do not rerun postprocessing into those paths. The following optional
independent rechecks use distinct outputs and operate only on completed raw
native logs; they do not infer missing episodes as failures or successes:

```powershell
& $py -m action_stream_benchmark.m8_cli `
  validate --manifest outputs/m8_g0/holdout/matrix.json `
  --output outputs/m8_g0/holdout/replay_validation.recheck.json

& $py -m action_stream_benchmark.m8_cli `
  analyze --manifest outputs/m8_g0/holdout/matrix.json `
  --replay outputs/m8_g0/holdout/replay_validation.recheck.json `
  --output outputs/m8_g0/holdout/analysis.recheck.json

& $py -m action_stream_benchmark.m8_cli `
  figures --analysis outputs/m8_g0/holdout/analysis.recheck.json `
  --output-dir outputs/m8_g0/holdout/figures_recheck

$holdoutCompletionReceipt = `
  'outputs/m8_g0/holdout/native_run_logs/<successful-holdout-run>/completion_receipt.json'

& $py -m action_stream_benchmark.m8_cli `
  report --repository-root . `
  --analysis outputs/m8_g0/holdout/analysis.json `
  --replay outputs/m8_g0/holdout/replay_validation.json `
  --starting-audit outputs/m8_g0/audit/initial_repository_environment_audit.json `
  --differential-report outputs/m8_g0/differential/differential_replay_report.json `
  --calibration-ledger configs/m8_calibration_ledger.json `
  --freeze-manifest outputs/m8_g0/protocol/freeze_manifest.json `
  --cpu-validation outputs/m8_g0/audit/cpu_validation.json `
  --native-runtime-audit outputs/m8_g0/audit/native_runtime_contract_audit.json `
  --completion-receipt $holdoutCompletionReceipt `
  --figure-manifest outputs/m8_g0/holdout/figures/figure_manifest.json `
  --output outputs/m8_g0/report/m8_g0_report.md
```

`<successful-holdout-run>` is the timestamped directory named by the completed
holdout evidence, never a refusal, timeout, or failure directory. The report
uses the runner-created canonical artifacts, not the recheck copies, and fails
closed if their hashes or semantics disagree. The final report must state
which commands actually completed. A command listed here is not itself
evidence that the corresponding run exists.

After a completed replay-clean holdout, create and verify one deterministic
archive containing the complete raw tree. Keep representative raw episodes
outside the archive for direct inspection; do not remove any other raw file
until an archive-backed matrix has itself replayed successfully.

```powershell
$archiveArgs = @(
  '-m', 'action_stream_benchmark.m8_cli', 'archive',
  '--root', '.',
  '--archive', 'outputs/m8_g0/holdout/complete_raw.tar.gz',
  '--manifest', 'outputs/m8_g0/holdout/complete_raw.manifest.json'
)
Get-ChildItem outputs/m8_g0/holdout/raw -Recurse -File |
  Sort-Object FullName |
  ForEach-Object {
    $archiveArgs += @('--member', $_.FullName)
  }
& $py @archiveArgs

& $py -m action_stream_benchmark.m8_cli archive-validate `
  --archive outputs/m8_g0/holdout/complete_raw.tar.gz `
  --manifest outputs/m8_g0/holdout/complete_raw.manifest.json

& $py -m action_stream_benchmark.m8_cli archive-matrix `
  --source-matrix outputs/m8_g0/holdout/matrix.json `
  --archive outputs/m8_g0/holdout/complete_raw.tar.gz `
  --archive-manifest outputs/m8_g0/holdout/complete_raw.manifest.json `
  --output outputs/m8_g0/holdout/matrix.archive.json

& $py -m action_stream_benchmark.m8_cli validate `
  --manifest outputs/m8_g0/holdout/matrix.archive.json `
  --output outputs/m8_g0/holdout/replay_validation.archive.json
```

The archive manifest records every member size and SHA-256. Archive creation
alone does not authorize deletion: the direct matrix remains canonical until
the archive-backed copy is generated and passes the independent validator.
Before pruning any nonrepresentative loose raw file, compare the direct and
archive-backed replay audits and require identical recomputed metrics and
invariant-violation counts for every episode. Keep representative event,
summary, scenario, and fault-trace files loose for direct inspection.

## Optional non-headline Profile-1 demo

Only after the immutable freeze exists, create one fresh development matrix
for the selected candidate, the exact frozen Profile-1 configuration, the
purpose-built demo seed, and `aligned_async` only. The demo remains
illustrative non-headline evidence and must never be added to holdout metrics.

```powershell
$freeze = Get-Content outputs/m8_g0/protocol/freeze_manifest.json -Raw |
  ConvertFrom-Json
$selectedRole = $freeze.development_calibration.selected_candidate_protocol_role
$selectedCandidate = ($freeze.inputs | Where-Object role -eq $selectedRole).path
$profile1 = ($freeze.inputs | Where-Object role -eq 'profile:profile_1_fixed').path

& $py -m action_stream_benchmark.m8_cli matrix-create `
  --repository-root . `
  --candidate-protocol $selectedCandidate `
  --seeds configs/m8_demo_seed.json `
  --profile $profile1 `
  --split development `
  --strategy aligned_async `
  --output-root outputs/m8_g0/demo/raw `
  --manifest outputs/m8_g0/demo/matrix.json

.\scripts\m8_record_demo.ps1 `
  -DevelopmentMatrixManifest outputs/m8_g0/demo/matrix.json `
  -FreezeManifest outputs/m8_g0/protocol/freeze_manifest.json `
  -IsaacWorkspace $isaacWs `
  -PixiExe $pixi `
  -VideoOutput release/m8_g0/media/actionstream_m8_profile1_aligned.mp4 `
  -AuthorizeNativeGpuRun
```

The recorder refuses reused outputs, a busy GPU, any candidate/profile/freeze
drift, or any matrix other than one selected-candidate Profile-1 aligned
episode. On success it writes an independent replay and portable non-headline
receipt beside the demo matrix and promotes the verified H.264 video only after
content checks pass. If a completed demo is referenced by the final report,
append `--demo-receipt outputs/m8_g0/demo/demo_receipt.json` to the report
command; otherwise the report must state that the illustrative demo is
unavailable.
