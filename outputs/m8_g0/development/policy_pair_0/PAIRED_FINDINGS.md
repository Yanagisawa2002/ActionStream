# Native Isaac policy-driven paired run

Status: completed, replay-valid development diagnostic. This is one paired
seed and is not headline or statistical evidence.

## Frozen setup

- Isaac Sim 6.0.1.0 with the native Franka articulation and physics.
- Deterministic observation-conditioned Cartesian waypoint policy; no replayed
  trajectory and no learned VLA checkpoint.
- Profile `profile_1_fixed`: fixed 850 ms injected response latency, 20 Hz
  control, 30-action chunks.
- Development seed `2026081100`, destination switch at step 110.
- Exact same reset and fault trace across `naive_async` and `aligned_async`.
- Independent replay passed 2/2 episodes; paired-reset fairness and provenance
  checks passed.

## Outcome

| Metric | Naive async | Aligned async |
|---|---:|---:|
| Task success | 0/1 | 0/1 |
| Termination | `failed_approach` | `joint_or_workspace_limit` |
| Completion step | 81 | 149 |
| Grasp achieved | No | Yes |
| Destination-switch recovery observed | No; episode ended before step 110 | Yes; 13 steps |
| First final-destination progress | Not observed | 15 steps after switch |
| Expired actions executed | 71 | 0 |
| Expired actions removed before installation | 0 | 154 |
| Mean executed action age | 38.676 steps | 17.230 steps |
| Maximum applied target delta | 0.2341 m | 0.1291 m |
| Hold-control steps | 10 | 23 |

Aligned execution therefore showed a real qualitative and semantic advantage
in this seed: it grasped and lifted the cube, survived the destination switch,
made progress toward the new green target, executed no expired action, and
reduced mean action age by 55.5% and maximum applied target discontinuity by
44.9%. Naive execution never grasped the cube and failed 29 steps before the
switch.

This is not a positive task-success result. Aligned execution later hit the
joint/workspace guard before placement, so both methods scored 0/1. With only
one development pair there is no confidence interval or reliability claim.

## Visual review

The locally retained raw captures are:

- `videos/batch_profile_1_fixed_naive_async.mp4` (4.05 s, 1280x720, H.264,
  20 fps)
- `videos/batch_profile_1_fixed_aligned_async.mp4` (7.45 s, 1280x720, H.264,
  20 fps)
- `videos/m8_profile1_seed2026081100_naive_vs_aligned.mp4` (labelled paired
  composite, 7.45 s, 1280x360, H.264, 20 fps)

All three decode end-to-end with FFmpeg 8.1.1. The final paired frame is
tracked as [paired_final.png](paired_final.png); MP4 files remain local and are
ignored by Git.

## Evidence

- [Replay validation](replay_validation.json)
- [Video validation](video_validation.json)
- [Naive summary](raw/summaries/m8-development-profile_1_fixed-2026081100-naive_async.json)
- [Aligned summary](raw/summaries/m8-development-profile_1_fixed-2026081100-aligned_async.json)
- [Completion receipt](native_run_logs/20260815T215433107018071Z/completion_receipt.json)
