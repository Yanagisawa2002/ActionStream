# ActionStream M7-G0 technical report

**Decision: NO-GO.** The mixed ROS 2/Isaac implementation is operational and
replay-auditable, but the predeclared reliability result did not materialize.
On frozen Profile B, naive and aligned asynchronous execution both succeeded
36/36, producing a 0.0 percentage-point paired difference with 95% bootstrap
CI [0.0, 0.0]. The sole policy-driven Isaac development episode also failed to
lift the object by step 180. No threshold was changed and no additional Isaac
attempt was run after that measured task failure.

## Scope and evidence boundary

M7 is a runtime and systems-integration milestone; it trains no model. The
headline 216-episode holdout uses the deterministic `ros_cpp_test_plant`
endpoint so ROS topics, the compiled executor, realistic fault delivery, and
event replay can be evaluated reproducibly. It is not Isaac physics. A separate
single development episode uses the canonical `ros_cpp_isaac_sim` evidence
class and exercises the full official Franka path, but it is not a paired
benchmark or holdout. Historical M4 results remain historical and are not
pooled with M7.

The work began on clean `master` at
`e786a25c5eeadffaed9bf35d6785fe4468ba7462`. Accepted M3/M4/M5-G0/M6-G0 tree
objects were recorded before modification and preserved. The initial host was
Windows 11 Home build 26200 with an RTX 4090/24,564 MiB, driver 591.86, CUDA
13.0.88, WSL Ubuntu 24.04.4, Python 3.12.3, GCC/G++ 13.3, and no installed ROS
or Isaac runtime.

The supported environment installed for M7 is ROS 2 Jazzy plus Isaac Sim
6.0.1.0/Python 3.12.13 through NVIDIA's official `IsaacSim-ros_workspaces`
`jazzy_ws` at commit `dd3eeede7912755996a18f4884285d9f50843f79`, Pixi
0.75.0, and `rmw_zenoh_cpp`. The installed compatibility checker passed the
host, RTX 4090, VRAM, and 591.86 driver. The ROS Docker image is
`actionstream-m7-ros:jazzy` at image ID
`sha256:64505ceaa908d1158844f0470fee5d8cda6ad0898ff1f94ad9e5193a3c34c952`,
built from the digest-pinned official Jazzy base.

## Runtime architecture

Five packages divide responsibility:

| Package | Responsibility |
|---|---|
| `action_stream_msgs` | Eight typed observation, request, chunk, command, lifecycle, diagnostic, and event messages |
| `action_stream_policy` | Deterministic scripted reach/descend/grasp/lift chunks and post-Kit node factory |
| `action_stream_executor` | Thread-safe C++17 strategy state machine, atomic queue updates, command selection, reason-coded diagnostics |
| `action_stream_benchmark` | Frozen paired traces, ROS test plant, fault injector, recorder, replay, paired analysis, figures |
| `action_stream_isaac` | Official Isaac Franka scene, 60 Hz physics/20 Hz control bridge, `/clock`, request driver, task lifecycle |

The C++ node uses explicit callback groups with a ROS
`MultiThreadedExecutor`. Observation/request callbacks never block for policy
inference. Python nodes handle policy work, controlled response delivery,
simulation glue, recording, and analysis. The full graph and timing diagrams
are in `docs/m7_architecture.md`.

## Frozen action-time and generation semantics

Observation step `O` is the last completed control step. A chunk created from
that observation explicitly labels targets `O+1` through `O+30`; target steps
are never inferred from arrival order. The 20 Hz final action is an absolute
seven-vector `[x, y, z, axis-angle x, axis-angle y, axis-angle z, gripper]`
with `+1` open and `-1` closed.

Generation is an explicit invalidation epoch, not a request counter. Routine
requests every ten steps stay in one generation. Cross-generation and
cross-episode responses are rejected; within a generation, freshness is the
lexicographic maximum `(source_observation_step, request_id)`. A target is
expired if it precedes queue insertion or already executed. First duplicate
target wins within a chunk, and aligned installation atomically replaces the
future executable queue. A finite, nonzero canonical 7D command is required
for safe hold.

`sync_hold` permits no hidden concurrent request and waits using that explicit
safe command. `naive_async` reasonably appends response-arrival chunks and
executes them sequentially with minimal bounds checks. `aligned_async` removes
expired prefixes, rejects stale provenance, retains only meaningful future
targets, and rebuilds atomically.

## Frozen task, faults, and split

The deterministic reach/lift task succeeds only when a grasped object stays at
least 0.12 m above its initial height for 20 consecutive steps; timeout is 180
steps. Profile A has fixed 950 ms response latency and no artificial faults.
Final Profile B has 900 ms base latency, +/-250 ms jitter, 5% drops, 12%
additional 1000 ms delay, 2% duplicates, and 5% 300 ms communication pauses.

Eight development seeds were excluded from headline metrics. The first Profile
B candidate was saturated at 8/8 for every method. The one permitted bounded
adjustment produced final development results `sync_hold` 7/8,
`naive_async` 8/8, and `aligned_async` 8/8; it was frozen despite showing no
aligned reliability advantage. The holdout was then locked at 36 disjoint
seeds before its first episode. All three methods consumed the same persisted
trace for each of 72 seed/profile blocks; trace mismatches were 0/72.

## Frozen holdout results

| Profile | Method | Success | Median steps | Median wall | Mean hold | Mean deadline-miss rate |
|---|---|---:|---:|---:|---:|---:|
| A | `sync_hold` | 36/36 | 47 | 4.350 s | 1.908 s | 0.00% |
| A | `naive_async` | 36/36 | 88 | 5.956 s | 0.000 s | 0.00% |
| A | `aligned_async` | 36/36 | 57 | 4.456 s | 0.000 s | 0.00% |
| B | `sync_hold` | 34/36 | 47 | 4.375 s | 2.800 s | 6.56% |
| B | `naive_async` | 36/36 | 88 | 5.708 s | 0.125 s | 4.49% |
| B | `aligned_async` | 36/36 | 57 | 4.262 s | 0.318 s | 11.95% |

For both profiles, aligned minus naive success is 0.0 percentage points with
paired-bootstrap 95% CI [0.0, 0.0] over 36 exact paired outcomes. Profile A is
fully saturated and does not reproduce the historical qualitative separation.
On Profile B, aligned reduces mean hold 88.6% versus sync, median steps 35.2%
versus naive, and median wall time 25.3% versus naive. However, aligned holds
more than naive (0.318 versus 0.125 s) and has a higher mean deadline-miss rate.
Efficiency improved; the predeclared reliability improvement did not.

Every required episode metric is present and descriptively summarized in
`outputs/m7_g0/holdout/analysis.json`, including exact raw paired outcomes,
requests/responses, fault counts, action ages, inference latencies, queue
mutations, completion reasons, and source-generation accounting.

## Semantic and replay audit

Independent replay read the 216 JSONL logs rather than trusting summaries:

- 216/216 audits passed and independently recomputed metrics matched 216/216;
- `aligned_async` and `sync_hold` had zero invariant violations;
- aligned removed 3,937 expired actions and executed no expired, duplicate,
  rejected-generation, or previous-episode action;
- the intentionally naive baseline executed 3,857 expired source actions, all
  retained as expected baseline behavior rather than hidden;
- every non-hold executed action was accounted for by source generation.

The holdout contained Profile B drops, duplicates, and out-of-order deliveries,
but no explicit generation transition. Cross-generation overwrite, pending
reset, post-termination response, duplicate target, completely expired chunk,
and concurrent queue-update cases are therefore acceptance-test evidence, not
claimed holdout observations.

## Native Isaac result

The supported native path starts `SimulationApp`, enables and updates
`isaacsim.ros2.bridge`, and only then imports `rclpy` and custom messages. It
uses the shipped experimental Franka class and no private DLL preload or test
plant fallback. One corrected bounded run completed:

| Evidence | Result |
|---|---|
| Endpoint | official Isaac Sim 6.0.1.0 Franka, `ros_cpp_isaac_sim` |
| Condition | Profile A, seed 2026080300, `aligned_async`, development only |
| Task | 0/1 success; `step_limit` at 180 steps |
| Timing | 9.0 s simulation; 12.334 s wall; 0.75 s hold |
| Policy traffic | 18 requests, 18 responses, 16 accepted chunks, 165 actions |
| Replay | passed 1,070 events; metrics matched; zero semantic violations |

The first launch attempt is retained: an inline Windows ROS parameter list
failed to parse and triggered the explicit bootstrap watchdog. The one
corrected attempt used the persisted parameter YAML and exited normally. No
retry or task tuning followed the measured manipulation failure. Exact commands
and hashes are in `outputs/m7_g0/isaac_smoke/launch_audit.json`.

## Gate and classification

| Condition | Outcome |
|---|---|
| Profile B aligned-minus-naive success >=20 points | **Fail:** 0.0 points |
| Paired 95% CI excludes zero | **Fail:** [0.0, 0.0] |
| Aligned hold reduction versus sync >=50% | Pass: 88.6% |
| No forbidden aligned execution | Pass: zero |
| Acceptance tests and replays pass | Pass |
| Strong: aligned within 5 points of sync | Pass |
| Strong: aligned median wall reduction versus sync >=20% | **Fail:** 2.6% |

The primary gate fails. Strong GO is ineligible. M7 is classified **NO-GO**
because the primary reliability claim failed, the sole Isaac task episode was
unsuccessful, and no paired Isaac holdout was justified after those failures.
The thresholds were not weakened.

## Validation

- Complete repository suite in the pinned WSL environment: 131 passed, one
  optional local LIBERO scene smoke skipped by its explicit environment gate.
- Clean five-package ROS Docker build and colcon test: 38 tests, zero errors,
  failures, or skips, including 23 C++ tests from 14 suites.
- Root M7 tests: 19 passed; current Isaac-adapter focused tests: 10 passed.
- Python `compileall`, Black checks, scoped flake8 checks, PowerShell AST checks,
  JSON consistency, deterministic figure export, and vector-PDF checks passed.
- The bounded recording command produced 180 H.264 frames at 1280x720/20 fps
  for 9.0 seconds; media, visual-frame, intermediate cleanup, and owned-process
  cleanup checks passed.
- An ad-hoc distro-wide flake8 invocation activated quote/import/docstring
  plugins that conflict with the Black convention and is not claimed as the
  repository lint contract; hard-error and scoped configured checks are used.

## Exact reproduction

Run the complete ROS test-plant holdout, replay, analysis, and six figure
exports from a fresh path:

```powershell
.\scripts\m7_run_ros_holdout.ps1 `
  -OutputSubdirectory outputs/m7_g0/reproductions/ros_holdout_001
```

Regenerate the committed audit and figures from existing raw results inside the
pinned container:

```bash
export PYTHONPATH=/workspace/ros2_ws/src/action_stream_benchmark:/workspace/ros2_ws/src/action_stream_policy:/workspace/src
python3 -m action_stream_benchmark.cli validate --manifest outputs/m7_g0/holdout/manifest.json --output outputs/m7_g0/holdout/replay_validation.json
python3 -m action_stream_benchmark.cli analyze --manifest outputs/m7_g0/holdout/manifest.json --output outputs/m7_g0/holdout/analysis.json --bootstrap-resamples 20000
python3 -m action_stream_benchmark.cli figures --analysis outputs/m7_g0/holdout/analysis.json --example-event-log outputs/m7_g0/holdout/episodes/profile_b/seed_2026080434/aligned_async.events.jsonl --output-dir outputs/m7_g0/figures
```

The exact official native build, Zenoh router, executor parameter-file, in-Kit
policy episode, and rosbag commands are in `docs/m7_environment.md`. The bounded
`scripts/m7_record_demo.ps1` command was media-validated as a 180-frame,
1280x720, 20 fps H.264 safe-hold recording; the machine-readable check is
`outputs/m7_g0/audit/demo_capture_validation.json`. It is not benchmark evidence.

## Limitations and next decision

The paired matrix is a deterministic ROS plant, not Isaac physics. The only
policy-driven Isaac task failed, standalone pre-Kit Windows `rclpy`/`ros2` CLI
loading remains blocked, and no rosbag, policy-driven demo video, reset
validation, three-strategy Isaac comparison, or Isaac holdout exists. The
validated video is safe-hold-only and is not committed as benchmark evidence.
The policy is scripted, the task set contains one reach/lift task, the plant is
not a real robot, and 20 Hz pacing is not a hard-real-time guarantee. The
justified next work would diagnose the failed lift on development-only inputs
and require a successful, replay-clean Isaac episode before authorizing any
paired Isaac benchmark; it is not authorized by this NO-GO closeout.

## Resume-ready bullet

Built a mixed C++17/Python ROS 2 ActionStream runtime with generation-aware
target-step queues, deterministic asynchronous fault injection, and independent
replay; validated 216 paired ROS test-plant episodes plus one end-to-end native
Isaac Sim Franka policy episode, while preserving zero aligned semantic
violations and reporting the failed reliability gate and Isaac lift honestly as
M7-G0 NO-GO.
