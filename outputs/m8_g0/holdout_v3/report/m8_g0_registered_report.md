# ActionStream M8-G0 dynamic-recovery report

## Decision and evidence boundary

Classification: **GO**

Evidence class: `ros_cpp_isaac_sim`. Native frozen-holdout episodes: 420.
Replay: 420/420 episodes passed; paired fairness=true; frozen provenance=true.
Development results and the illustrative video are excluded from every headline result.

## Starting commit and environment

The clean starting checkout was `master` at `1c51d3e0489a57224c4b4d2f4f6ddafae5302713` tracking `origin/master` (3 ahead, 0 behind).
Starting audit host only (not the native holdout runtime): Windows Microsoft Windows 11 Home build 26200; Python 3.13.5; PyTorch 2.6.0+cu124; CUDA toolkit 13.0.88.
Starting audit GPU only (not the native holdout runtime): `NVIDIA GeForce RTX 4090` with driver 591.86.
Native holdout runtime: GPU `NVIDIA GeForce RTX 5090` UUID `GPU-81d81329-c2d4-2f0f-d141-2871873dac1e` with 32607 MiB and driver 580.76.05; IsaacSim-ros_workspaces commit `dd3eeede7912755996a18f4884285d9f50843f79`; Pixi 0.75.0; Python 3.12.13; Isaac Sim 6.0.1.0; ROS jazzy with rmw_zenoh_cpp 0.2.9.

## Reused components and frozen ActionStream semantics

- m4 python executor: `src/actionstream/runtime.py:ActionQueue`.
- m4 benchmark orchestration: `src/actionstream/benchmark.py`.
- m7 cpp executor: `ros2_ws/src/action_stream_executor/src/executor_state_machine.cpp`.
- m7 cpp executor contract: `ros2_ws/src/action_stream_executor/include/action_stream_executor/executor_state_machine.hpp`.
- m7 messages: `ros2_ws/src/action_stream_msgs/msg`.
- m7 benchmark orchestration: `ros2_ws/src/action_stream_benchmark/action_stream_benchmark/cli.py`.
- m7 fault injector: `ros2_ws/src/action_stream_benchmark/action_stream_benchmark/fault_injector_node.py`.
- m7 event recorder: `ros2_ws/src/action_stream_benchmark/action_stream_benchmark/event_recorder_node.py`.
- m7 replay validator: `ros2_ws/src/action_stream_benchmark/action_stream_benchmark/replay.py`.
- m7 isaac adapter: `ros2_ws/src/action_stream_isaac/action_stream_isaac/isaac_adapter.py`.
- m7 policy: `ros2_ws/src/action_stream_policy/action_stream_policy/scripted_policy.py`.
- m7 report: `outputs/m7_g0/report/m7_report.md`.

The frozen runtime is 20.0 Hz with a 30-step horizon, 10-step request interval, and `absolute Cartesian xyz, axis-angle xyz, gripper` actions. Generation IDs remain invalidation epochs; aligned replacement remains atomic.

## Phase 0: M4-to-M7 differential replay

Common-domain behavioral equivalence: **true** over 42 common events. Unresolved unintended discrepancies: 0.
Migration discrepancy found and repaired before benchmark execution:
- `generation_advance_retained_executable_old_generation_queue`: Propagate generation on observations, advance it before command selection, atomically invalidate non-naive old-generation queue entries, and retain a defensive execution-time purge. Status: `resolved`.

## Native task, controller, and observation dependence

Task `dynamic_target_pick_place_v1` uses the physical `destination_switch` disturbance. Success requires release inside the final destination with 0.04 m planar tolerance and 20 stable steps.
The deterministic `deterministic_observation_conditioned_cartesian_waypoint` controller consumes the latest end-effector, object, grasp, phase, active-destination, and disturbance state; bounded policy tests are part of the hash-bound CPU/native validation suite. No neural policy was trained.

## CPU/native validation and simulator baseline gate

Repository Python tests: 312 passed, 0 failed, 2 skipped.
Native Windows ROS: 197 leaf tests, 0 failures, 0 errors.
Native Isaac Profile-0 baseline: 20/20 success (100.0%); replay validated=true; required ceiling=90.0%.

## Frozen calibration, profiles, and seeds

Selected development candidate: `candidate_0` after 0 bounded calibration change(s). Development remains non-headline with 12 seeds.
Frozen holdout seed count: 60; holdout seed SHA-256 `951a340a51ff15c037cd9c4af2f8446e4a2cdcbf2110256b991790ecabb72591`; freeze SHA-256 `4d63b6b6f2af1458c063ed52eb73e1a2bcb82f412bb0a9b19d352659bbca923f`.
- `profile_0_sanity`: `ros2_ws/src/action_stream_benchmark/config/m8_profile_0_sanity.json` (SHA-256 `6ed5d115855037a5805976a653eb36ff5a329ed43a7102a0d21c2a7a0ec3db60`).
- `profile_1_fixed`: `ros2_ws/src/action_stream_benchmark/config/m8_profile_1_fixed.json` (SHA-256 `996010206f7ca48321c4df2ac7ebfbf95933b66b0beb869c45109f198e05cda4`).
- `profile_2_faults`: `ros2_ws/src/action_stream_benchmark/config/m8_profile_2_faults.json` (SHA-256 `0ecb35143cdd9f0e4b263be4ce67cb0739b735d80e6bb334e1e710e345612dc2`).

## Preregistered gates

| Gate condition | Passed |
|---|---:|
| aligned forbidden execution count is zero | yes |
| aligned minus naive success at least 15pp | yes |
| frozen protocol validated without post holdout change | yes |
| native isaac evidence only | yes |
| paired 95 ci excludes zero | yes |
| paired fairness passes | yes |
| profile 0 ceiling at least 90pct | yes |
| replay passes every episode | yes |
| primary gate overall | yes |
| strong gate eligible | yes |
| strong gate overall | no |

## Exact frozen-holdout results

### `profile_0_sanity` (60 paired seeds)

| Method | Success | Rate | Median steps | Mean steps | Median wall (s) | Mean wall (s) | Median penalized recovery | Mean obsolete commands | Mean hold (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sync_hold` | 60/60 | 100.0% | 231.0 | 225.3 | 18.636 | 18.282 | 2.0 | 10.40 | 0.050 |

### `profile_1_fixed` (60 paired seeds)

| Method | Success | Rate | Median steps | Mean steps | Median wall (s) | Mean wall (s) | Median penalized recovery | Mean obsolete commands | Mean hold (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sync_hold` | 1/60 | 1.7% | 202.0 | 196.2 | 16.147 | 15.749 | 13.0 | 2.40 | 5.813 |
| `naive_async` | 0/60 | 0.0% | 82.0 | 82.3 | 6.717 | 6.793 | 221.0 | 0.00 | 0.579 |
| `aligned_async` | 49/60 | 81.7% | 227.0 | 216.1 | 18.435 | 17.527 | 13.0 | 5.32 | 1.176 |

Aligned minus naive: 81.7 percentage points; paired 95% CI [71.7, 90.0] percentage points over 60 exact paired outcomes.

### `profile_2_faults` (60 paired seeds)

| Method | Success | Rate | Median steps | Mean steps | Median wall (s) | Mean wall (s) | Median penalized recovery | Mean obsolete commands | Mean hold (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sync_hold` | 15/60 | 25.0% | 212.0 | 188.3 | 17.063 | 15.253 | 14.0 | 2.37 | 6.515 |
| `naive_async` | 0/60 | 0.0% | 82.0 | 83.2 | 6.811 | 6.893 | 221.0 | 0.00 | 0.636 |
| `aligned_async` | 52/60 | 86.7% | 228.0 | 226.1 | 19.027 | 18.731 | 12.0 | 6.60 | 1.445 |

Aligned minus naive: 86.7 percentage points; paired 95% CI [76.7, 95.0] percentage points over 60 exact paired outcomes.

## Recovery, obsolete-action, and unavailable metric semantics

| Method | Mean inference latency (ms) | P95 inference latency (ms) | Mean action age (steps) | P95 action age (steps) |
|---|---:|---:|---:|---:|
| `sync_hold` | 881.774 | 898.453 | 16.961 | 21.000 |
| `naive_async` | 905.740 | 907.706 | 40.174 | 60.550 |
| `aligned_async` | 893.844 | 902.840 | 17.366 | 22.000 |

A distribution with no observations is reported as `unavailable`, never as zero. The analysis artifact retains every required per-method descriptive metric, including collision/timeout rates, failure-reason counts, simulation time, obsolete motion, pre-switch command exposure, expired/duplicate/stale handling, request/response counts, queue rebuilds, and deadline misses.

## Semantic, replay, and fairness checks

- Aligned forbidden executions: 0.
- Seed validation: true; profile validation: true.
- Replay provenance: true; paired fairness: true.
- Bootstrap: 20000 paired nonparametric resamples at 95% confidence.

## Figures, archive, and illustrative demo

Three analytical figure families are hash-bound by `outputs/m8_g0/holdout_v3/figures/figure_manifest.json`.
The complete raw archive and archive-backed replay are present and exactly reproduce the direct replay metrics.
Illustrative demo: unavailable/not recorded; no video claim is made.

## Exact reproduction commands

```bash
python -m action_stream_benchmark.m8_cli freeze-validate --repository-root . \
  --manifest outputs/m8_g0/protocol_v3/freeze_manifest.json
bash scripts/m8_run_isaac.sh \
  --matrix-suite-manifest outputs/m8_g0/holdout_v3/matrix.json \
  --isaac-workspace "$isaac_ws" --pixi-exe "$pixi" \
  --gpu-index 0 --headless true --authorize-native-gpu-run
python -m action_stream_benchmark.m8_cli validate \
  --manifest outputs/m8_g0/holdout_v3/matrix.archive.json \
  --output outputs/m8_g0/holdout_v3/replay_validation.archive.json
```

The native runner builds the current checkout, validates live frozen inputs before execution, and creates replay, analysis, and the three analytical figures. Archive and demo commands are separate because neither is headline evidence.

## Limitations

This is native Isaac Sim evidence with a deterministic scripted controller, not real-robot validation or production readiness. It covers one Franka destination-switch task and two asynchronous profiles. In Profile 2, a dropped synchronous response remains in flight until terminal drain and may produce an episode-long hold. Video is illustrative and never benchmark evidence. The M7-G0 static-plant saturation result remains a separate honest NO-GO and is not reinterpreted.

## Bound artifacts

- Analysis: `outputs/m8_g0/holdout_v3/analysis.json` (SHA-256 `ed5d28baa3f92a2f030337bd66523aeb98f1be3e44ee07623dcb5099afb14bab`).
- Replay: `outputs/m8_g0/holdout_v3/replay_validation.json` (SHA-256 `2c4bb98c77f145d998ffa9c259bb7bc9f2e368382ab3edf605b5c57776116695`).
- Starting Audit: `outputs/m8_g0/audit/initial_repository_environment_audit.json` (SHA-256 `8e2c9c27c849a8fd7c6bc40ec648dba3882aa0162e1604cf67eebede122a43b2`).
- Phase-0 Differential: `outputs/m8_g0/differential/differential_replay_report.json` (SHA-256 `26cee2a75011dc24ec3b1457022d628ab1f27d9306b5ff0074ebc1df960f8bb6`).
- Calibration Ledger: `configs/m8_calibration_ledger_v3.json` (SHA-256 `6e8687e51b0273f55b1073214024955a698a0760dc85e528447e43c37bc432c0`).
- Freeze Manifest: `outputs/m8_g0/protocol_v3/freeze_manifest.json` (SHA-256 `ae84b08dd083d12befe59058897f3abe52949173a08ee4a6b477fba6e62e3a86`).
- Cpu/Native Validation: `outputs/m8_g0/audit/cpu_validation.json` (SHA-256 `596df5be4801c62646b13d5137ffdda75f0d81e2df6abb57f6e5f27b0ff16edd`).
- Native Runtime Audit: `outputs/m8_g0/audit/native_runtime_contract_audit_v3.json` (SHA-256 `3485b4f91ec35126d8894a42ed4284a342420d8d4685811897c48b378b518745`).
- Holdout Completion Receipt: `outputs/m8_g0/holdout_v3/native_run_logs/20260816T054257825323061Z/completion_receipt.json` (SHA-256 `ff793fd3cd497a562ec4eca5e9162658effe609481ae6d61f33bd823212ec408`).
- Figure Manifest: `outputs/m8_g0/holdout_v3/figures/figure_manifest.json` (SHA-256 `25100580c2b8d339e33a7200ed584c5fee1485bcc51be51bb26b3ea0f29d9ddc`).
- Holdout Matrix: `outputs/m8_g0/holdout_v3/matrix.json` (SHA-256 `384be6047398c58b2b871b313b2184dae7917d715f7e43e5e688748610c22562`).
- Holdout Preflight Receipt: `outputs/m8_g0/holdout_v3/native_run_logs/20260816T054257825323061Z/preflight_receipt.json` (SHA-256 `f9fa0db4ab3008ba9078d778874a300b0a39e207c09a4fc5b7ac12e9db2f4844`).
- Holdout Installed-Adapter Evidence: `outputs/m8_g0/holdout_v3/native_run_logs/20260816T054257825323061Z/installed_dynamic_adapter.py` (SHA-256 `a584eadb952cd0041b27521a5c72f0bf25be1721c7e0e2ee88ad3433a11757d9`).
- Holdout Installed-Executor Evidence: `outputs/m8_g0/holdout_v3/native_run_logs/20260816T054257825323061Z/action_stream_executor_node` (SHA-256 `dec2423adf253928b5c7f431efd6577524f240856fb078a2620ba8c1f032bad9`).
- Holdout Runner Evidence: `outputs/m8_g0/holdout_v3/native_run_logs/20260816T054257825323061Z/m8_run_isaac.runner.sh` (SHA-256 `7e4cac9bf0b62360ac5b685a94f0d917b667467971f452f48db908daeec6bca7`).
- Holdout Runner-Support Evidence: `outputs/m8_g0/holdout_v3/native_run_logs/20260816T054257825323061Z/m8_linux_runner_support.py` (SHA-256 `304da243f5c9efdeb7896a52c90d96f9d308ae3e9355bdff4e78fc5b7c38ab39`).
- Holdout External-Environment Evidence: `outputs/m8_g0/holdout_v3/native_run_logs/20260816T054257825323061Z/external_environment.json` (SHA-256 `696ba2ce3e03822f0565cec8289bc7542f5d47f37a2c4d04a648b701adc33ab3`).
- Holdout Pixi.Toml Evidence: `outputs/m8_g0/holdout_v3/native_run_logs/20260816T054257825323061Z/isaac_workspace.pixi.toml` (SHA-256 `b4e7a34c264e88f19ba6bfb3c7a72ee46b0843f3e6eb7e75619dc0ebb87b313d`).
- Holdout Pixi.Lock Evidence: `outputs/m8_g0/holdout_v3/native_run_logs/20260816T054257825323061Z/isaac_workspace.pixi.lock` (SHA-256 `ba8e59eef962cbf49a1ff06ff947ed5eaa4547d018389b048771a1e7d8bb890d`).
- Complete Raw Archive: `outputs/m8_g0/holdout_v3/complete_raw.tar.gz` (SHA-256 `1375d95ea9656906ba93a060088f24683b8fa2e97a9e1c95ef4a536d77ddef19`).
- Archive Manifest: `outputs/m8_g0/holdout_v3/complete_raw.manifest.json` (SHA-256 `e5413fe0686d6bd1a46057276d4afd3ad83604afb51ace4c831c98c4ee2bbed4`).
- Archive-Backed Matrix: `outputs/m8_g0/holdout_v3/matrix.archive.json` (SHA-256 `5f4d9e236855cd0ec390082d04bfbd40e4368093dac95507ec4693440c88e1a8`).
- Archive-Backed Replay: `outputs/m8_g0/holdout_v3/replay_validation.archive.json` (SHA-256 `374730043d4606294d5d0e37308b9a82c72a503ab4616a462a69fb0767d1d649`).

## Release-time repository status

The final M8 commit hash, exact changed-file list, and clean-worktree confirmation are release-time metadata and are unavailable while this report is being built for inclusion in that commit. They must be supplied by the final handoff; no value is synthesized here.

## Resume-ready result

- Built and replay-validated a native Isaac Sim Franka destination-switch benchmark across 420 frozen holdout episodes; Profile-1 aligned versus naive success differed by 81.7 percentage points (paired 95% CI [71.7, 90.0]), yielding **GO**; this is simulation evidence, not real-robot validation.
