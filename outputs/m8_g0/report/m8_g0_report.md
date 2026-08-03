# M8-G0 technical report: native evaluation unavailable

## Decision

M8-G0 is **NO-GO** under the preregistered fallback because the native Isaac
simulator baseline could not start in a safe GPU slot. This is an
`unavailable_not_run` result, not 0% success. No native M8 episode completed,
no deterministic ROS test-plant result was substituted, and no success,
latency, recovery, or obsolete-action metric is reported as zero.

| Boundary | Result |
|---|---|
| Engineering implementation | Complete and CPU/ROS validated |
| Current-source 20-seed baseline manifest validation | Passed |
| Native Isaac episode execution | Unavailable / not run |
| Profile 0 baseline gate | Unavailable |
| Development calibration | Prohibited after unavailable baseline |
| Frozen holdout | Not frozen or run |
| Headline classification | NO-GO |

## Starting state and preserved evidence

Work began on clean `master` at
`1c51d3e0489a57224c4b4d2f4f6ddafae5302713`, three commits ahead of and zero
behind `origin/master`. The starting audit records Windows 11 build 26200, an
RTX 4090 with 24,564 MiB, driver 591.86, CUDA 13.0.88, Docker 29.3.1, Pixi
0.75.0, native Python 3.12.13, Isaac Sim 6.0.1.0, ROS 2 Jazzy, and
`rmw-zenoh-cpp` 0.2.9.

M3-M7 raw evidence was not regenerated or reinterpreted. M7 remains the honest
static-task NO-GO: `naive_async` and `aligned_async` both reached 36/36 in its
Profile-A test-plant matrix, so its reliability gate failed despite semantic
correctness and efficiency improvements. Two source-only prior-test
reproducibility repairs are separately recorded; neither changes a historical
executor or raw result.

## M4-to-C++ differential replay

The canonical accepted M4 Python runtime remained byte-identical. The final
native C++ probe replayed 74 events and found exact common-domain behavioral
equivalence, zero common mismatches, all declared extended differences, and
zero unresolved unintended discrepancies.

Phase 0 found and repaired one pre-benchmark migration defect: a generation
advance could retain executable old-generation queue entries. A later
pre-native concurrency audit found a separate ROS transport race: a zero-delay
response could cross its request on another topic and be rejected before
registration. The C++ state machine now holds one response per request in a
bounded pre-registration buffer, validates it after matching provenance
registers, and protects persistent episodes from same-ID cross-episode
eviction. Five added C++ regression tests, including a 100-iteration concurrent
request/response test, pass on Windows and Linux.

## Native task and controller design

The implemented benchmark uses Isaac Sim 6.0.1 native physics, the supported
experimental Franka articulation/controller API, deterministic scene reset,
and a physically checked destination-switch pick-and-place task. The task
requires approach, grasp, lift, carry, release in the post-switch destination,
and 20 stable placement steps.

One deterministic observation-conditioned controller produces 30 finite
absolute Cartesian/gripper commands from current end-effector pose, object
pose, grasp state, task phase, active destination, and disturbance generation.
Tests prove that destination, object position, and phase changes alter the
output while commands remain finite and workspace-bounded. All three methods
share that controller, observations, request cadence, scene, thresholds, and
fault traces.

The preregistered protocol uses 20 baseline seeds, at most 12 development
seeds per candidate, a maximum of two behavioral calibration changes, and 60
disjoint holdout seeds. Profile 0 has no artificial faults; Profile 1 uses a
fixed 850 ms delay; Profile 2 uses 850 ms base delay, 200 ms jitter, 3% drops,
10% additional 900 ms delay, 2% duplicates, and 3% 250 ms pauses.

## Native preflight outcome

The fresh candidate-0 matrix contains exactly 20 Profile-0 `sync_hold`
episodes. The official Isaac Pixi environment validated every manifest,
scenario, profile, controller field, and fault trace against current source.
Before build or Isaac startup, the runner then observed 16,022 MiB of GPU
memory in use, above its fixed 4,096 MiB safety threshold. Three unrelated
Unity Editor processes were present and the visible Vulture project had
unsaved changes. The runner refused without terminating any process.

Consequently:

- native task success and the Profile-0 >=90% gate are unavailable;
- raw event logs and episode summaries do not exist;
- `baseline-record` was not allowed to advance the pristine calibration ledger;
- no candidate calibration, immutable freeze, or holdout evaluation occurred;
- sync/naive/aligned denominators, paired outcomes, confidence intervals,
  recovery latency, completion time, obsolete-action exposure, and all other
  requested metrics are unavailable;
- analytical figures and a meaningful native demo video were not generated,
  because doing so without native results would be decorative or misleading.

## Validation completed

- Pinned repository Python: 314 collected, 312 passed, 2 environment skips.
- Native Windows ROS Jazzy: 5 packages built; 198 reported results, 0 errors,
  0 failures, 0 skips (197 leaf cases plus one CTest wrapper).
- Read-only Linux ROS Jazzy Docker: the same 5 packages and 198 clean results.
- C++ state machine: 35 GTests, including generation, reset, response-order,
  duplicate, queue-atomicity, and concurrency coverage.
- Differential replay: 74 events, exact common-domain equivalence, replay gate
  unblocked.
- Scoped Ruff, Python compilation, both PowerShell AST parses, 72 scoped JSON
  files (configs, compact M8 evidence, ROS benchmark configs, and the M8 dynamic-policy
  config), and `git diff --check`: passed.
- Repository-wide Ruff retains one preexisting out-of-scope F841 at
  `src/actionstream/m5_validation.py:501`.

The two local Python skips are the opt-in LIBERO scene integration check and a
test-local compiler discovery skip. The separately built native Windows and
Linux C++ probes and GTests passed.

## Gate evaluation

The primary Profile-1 gate cannot be evaluated because its prerequisite native
Profile-0 ceiling was not established. There is no paired success difference,
confidence interval, recovery comparison, or strong-GO efficiency comparison.
Semantic and CPU/replay tooling correctness alone is not task-reliability
evidence. The only defensible classification is **NO-GO: native evaluation
unavailable/not run**.

## Evidence index

| Artifact | SHA-256 |
|---|---|
| [Starting audit](../audit/initial_repository_environment_audit.json) | `8e2c9c27c849a8fd7c6bc40ec648dba3882aa0162e1604cf67eebede122a43b2` |
| [CPU/ROS validation](../audit/cpu_validation.json) | `596df5be4801c62646b13d5137ffdda75f0d81e2df6abb57f6e5f27b0ff16edd` |
| [Native contract audit](../audit/native_runtime_contract_audit.json) | `12434cf7672726e933a8ba31a026665b1422236dcf859235a3f844776ee3f260` |
| [Differential report](../differential/differential_replay_report.json) | `26cee2a75011dc24ec3b1457022d628ab1f27d9306b5ff0074ebc1df960f8bb6` |
| [Candidate-0 matrix](../baseline_gate/candidate_0/matrix.json) | `dfe61569c74fd8faf942871c5f9675f4b0c8f2063e48b44f70cc58ddcc328c4e` |
| [Redacted native preflight receipt](../audit/native_preflight_refusal.json) | `5cc2f762746b1d6bc2093a124ba984ee96ebecfbcb0b6b075d830436430f3455` |
| Raw GPU refusal receipt (hash only; not retained) | `55bdb866f887292a6691af730baf40b1e20d21f9a1f344d5985e64b57cc7d561` |
| [Machine-readable unavailable result](m8_g0_unavailable.json) | `58ed028d34ebf82188d3b5d2b3c378a722a76b374c5a602744d59e262068708a` |

## Reproduction and resume

The [environment and reproduction guide](../../../docs/m8_environment.md)
records the full validated command sequence and the clean-GPU resume boundary.

From the repository root, first save and close unrelated GPU applications and
verify memory use is below the unchanged safety threshold:

```powershell
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
$pixi = "$env:LOCALAPPDATA\pixi\bin\pixi.exe"
$isaacWs = 'C:\IsaacSim-ros_workspaces\jazzy_ws'
```

Run the structurally and preflight-validated one-seed controller-smoke manifest
before the formal matrix:

```powershell
.\scripts\m8_run_isaac.ps1 `
  -MatrixSuiteManifest outputs/m8_g0/development/controller_smoke_0/matrix.json `
  -IsaacWorkspace $isaacWs -PixiExe $pixi -Headless $true `
  -AuthorizeNativeGpuRun
```

If that episode is physically stable, run the unchanged 20-seed baseline:

```powershell
.\scripts\m8_run_isaac.ps1 `
  -MatrixSuiteManifest outputs/m8_g0/baseline_gate/candidate_0/matrix.json `
  -IsaacWorkspace $isaacWs -PixiExe $pixi -Headless $true `
  -AuthorizeNativeGpuRun
```

Only a replay-clean completion with at least 18/20 successes permits
`baseline-record`, bounded development calibration, freeze, and the 60-seed
holdout. Exact follow-on commands are in `docs/m8_environment.md`. No source,
runner, protocol, matrix, profile, scenario, or fault-trace byte may change
between a successful baseline run and `baseline-record`.

## Honest resume bullet

- Implemented and CPU/ROS-validated M8-G0 dynamic-target recovery with exact
  M4/C++ common-domain equivalence and a tested cross-topic registration
  barrier, but native Isaac evaluation was unavailable because unrelated Unity
  sessions held 16,022 MiB of VRAM; no M8 task episode, reliability metric,
  analytical figure, or demo was claimed, and the milestone remains NO-GO
  pending a clean-GPU rerun (not real-robot validation).
