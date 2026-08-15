# M8-G0 technical report: native baseline passed

## Decision

M8-G0 now has valid native Isaac evidence. The preregistered Profile-0
`sync_hold` baseline passed 20/20 task episodes and 20/20 independent replay
audits. This opens development; it does not by itself establish superiority of
the asynchronous method.

The requested one-seed Profile-1 native pair is also complete and replay-valid.
It shows a directional semantic and visual advantage for aligned execution,
but both methods failed the full task. No frozen holdout was run. The current
classification is therefore **IN PROGRESS**, not GO and no longer
`unavailable_not_run`.

| Boundary | Result |
|---|---|
| Native environment and current-source build | Passed |
| Profile-0 baseline | 20/20 task success |
| Baseline independent replay | 20/20 passed |
| Baseline ledger | `baseline_passed_development_open` |
| Profile-1 policy-driven pair | 2/2 replay-valid; 0/1 task success per method |
| Frozen holdout | Not frozen or run |
| Headline classification | Not available |

## Native task and policy

The benchmark uses Isaac Sim 6.0.1 native physics and the supported
experimental Franka articulation/controller API. A cube must be approached,
grasped, lifted, carried to the destination that is active after a seeded
switch, released within tolerance, and remain stable for 20 steps.

The shared policy is deterministic and observation-conditioned. Every request
uses the live end-effector and object poses, grasp state, task phase, active
destination, and generation to produce a bounded 30-step absolute Cartesian,
axis-angle, and gripper chunk. This is a genuine closed-loop policy-driven run,
but it is not a learned VLA and is not real-robot evidence.

## Environment and provenance

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, driver 595.71.05.
- Isaac Sim: 6.0.1.0.
- Official workspace commit:
  `dd3eeede7912755996a18f4884285d9f50843f79`.
- Pixi: 0.75.0 with the frozen workspace manifest and lock.
- ROS 2 Jazzy with `rmw_zenoh_cpp` 0.2.9.
- ActionStream baseline source commit: `e32983d`.
- ActionStream paired-run source commit: `ee2c840`.

The completion receipts bind the GPU snapshots, source manifest, runner,
runner-support helper, installed dynamic adapter, built C++ executor, Pixi
files, process-log manifest/archive, matrix, and replay result. The local
`baseline-record` command independently revalidated those bindings before
advancing the calibration ledger.

## Profile-0 baseline

All 20 preregistered seeds completed with
`correct_destination_stable_placement`. Completion steps ranged from 201 to
251, with median 231. Independent replay passed all 20 event logs with no
invariant or metric mismatch; seed, profile, provenance, and fairness checks
also passed.

This is a positive simulator result: the observation-conditioned policy and
native task are viable without artificial latency. It is not evidence that
aligned async is better than another runtime because Profile 0 contains only
the baseline `sync_hold` strategy.

## Profile-1 same-reset paired result

The development pair uses seed `2026081100`, a destination switch at step 110,
and fixed 850 ms response latency at 20 Hz. Both methods used exactly the same
reset and fault trace; all pairwise reset deltas were zero except a
`3.39e-08` quaternion-norm floating-point residual, well inside the frozen
`1e-06` bound.

| Metric | Naive async | Aligned async |
|---|---:|---:|
| Task success | 0/1 | 0/1 |
| Completion reason | `failed_approach` | `joint_or_workspace_limit` |
| Completion step | 81 | 149 |
| Grasp success | No | Yes |
| Recovery observed | No; ended before switch | Yes |
| Recovery latency | Not observed | 13 steps / 0.65 s |
| Final-destination progress | Not observed | 15 steps after switch |
| Expired actions executed | 71 | 0 |
| Expired actions removed | 0 | 154 |
| Mean action age | 38.676 steps | 17.230 steps |
| P95 action age | 59 steps | 22 steps |
| Maximum applied target delta | 0.2341 m | 0.1291 m |
| Hold-control steps | 10 | 23 |

Aligned execution reduced mean action age by 55.5%, reduced maximum applied
target discontinuity by 44.9%, and eliminated the 71 expired executions seen
in naive async. Visually, naive approached but never grasped the cube and
terminated 29 steps before the switch. Aligned grasped and lifted the cube,
observed the active marker move from the red to the green destination,
recovered in 13 steps, and moved toward the new target before the workspace
guard stopped it.

This is not a positive task-success comparison. Both methods scored 0/1, the
aligned run did not place the cube, and one development pair cannot support a
confidence interval or reliability conclusion.

## Native issues found and resolved

The successful run followed several fail-closed attempts that were retained
outside the canonical evidence path:

1. Isaac Sim 6 required explicit contact-report API enablement.
2. The in-process single-threaded ROS executor needed a bounded callback drain
   between control steps to avoid response backlog.
3. Cross-topic recorder arrival order could not reconstruct the executor's
   queue-insertion boundary. `queue_updated` now records that boundary and both
   native recorders decode it.
4. A shared Colcon workspace could retain a CMake source directory from an
   earlier Git worktree. The Linux runner now clears the CMake cache before
   every current-source build.

These are runtime/replay integrity fixes, not task or threshold calibration.
The successful baseline was rerun from a clean Git worktree after all fixes.

## Validation

- Native Profile-0 task success: 20/20.
- Native Profile-0 replay: 20/20.
- Native Profile-1 paired replay: 2/2.
- Paired-reset fairness and provenance: passed.
- C++ executor on the remote ROS workspace: 35/35 GTests passed.
- Relevant local Python replay, recorder, adapter, and Linux-runner tests:
  passed.
- Three local MP4s (two raw captures and one labelled composite): H.264,
  20 fps, full FFmpeg 8.1.1 decode passed.

The broader local suite passes when the unrelated optional PyAV-dependent M4
capture test is excluded; PyAV is not installed in the pinned local test venv
and was not installed solely for this M8 run.

## Gate evaluation

The >=90% native baseline prerequisite passed. The one-seed development pair
does not evaluate the preregistered Profile-1 success-difference confidence
interval, and neither method succeeded. Candidate selection, freeze, and the
60-seed holdout remain unexecuted. M8 therefore has positive native viability
and semantic evidence, but no headline GO result.

## Evidence index

- [Current machine-readable status](m8_g0_current.json)
- [Calibration ledger](../../../configs/m8_calibration_ledger.json)
- [Baseline replay](../baseline_gate/candidate_0/replay_validation.json)
- [Baseline completion receipt](../baseline_gate/candidate_0/native_run_logs/20260815T213757268580084Z/completion_receipt.json)
- [Paired findings](../development/policy_pair_0/PAIRED_FINDINGS.md)
- [Paired replay](../development/policy_pair_0/replay_validation.json)
- [Paired video validation](../development/policy_pair_0/video_validation.json)
- [Paired final frame](../development/policy_pair_0/paired_final.png)
- [Environment and reproduction guide](../../../docs/m8_environment.md)

The earlier [unavailable result](m8_g0_unavailable.json) and RTX-4090 refusal
remain historical records of the prior machine state. They are not the current
M8 result.

## Honest resume bullet

- Built and replay-validated a native Isaac Sim 6.0.1 Franka
  observation-conditioned action-stream benchmark: 20/20 zero-fault baseline
  successes; in one same-reset 850 ms development pair, aligned execution
  grasped and recovered the destination switch with zero expired executions
  while naive failed before the switch after 71 expired executions, although
  both failed final placement and no holdout or real-robot claim is made.
