# Native Isaac successful policy-driven pair

Status: completed and replay-valid development example. This pair provides a
concrete visual task-success contrast; the 12-seed statistics live in the
separate full candidate artifact.

## Frozen setup

- Isaac Sim 6.0.1 native Franka articulation and physics on an RTX 5090.
- Deterministic observation-conditioned Cartesian waypoint policy; no replayed
  trajectory and no learned VLA checkpoint.
- Profile `profile_1_fixed`: fixed 850 ms injected response latency at 20 Hz
  with 30-action chunks.
- Development seed `2026081101`, destination switch at step 110.
- Exact same reset, scenario, and fault trace for `naive_async` and
  `aligned_async`.
- Independent replay passed 2/2 episodes; paired-reset fairness and provenance
  checks passed.

## Outcome

| Metric | Naive async | Aligned async |
|---|---:|---:|
| Task success | 0/1 | 1/1 |
| Termination | `failed_approach` | `correct_destination_stable_placement` |
| Completion step | 81 | 227 |
| Grasp achieved | No | Yes |
| Switch recovery | Not reached | 13 steps / 0.65 s |
| First final-destination progress | Not reached | 14 steps after switch |
| Expired actions executed | 70 | 0 |
| Expired actions removed | 0 | 247 |
| Mean executed action age | 39.357 steps | 17.176 steps |
| P95 action age | 59.55 steps | 22 steps |
| Maximum applied target delta | 0.2283 m | 0.1313 m |
| Hold-control steps | 11 | 23 |

Naive execution never grasped the cube and failed 29 steps before the switch.
Aligned execution grasped and lifted it, recovered the new destination in 13
steps, placed the cube on the active red target, released it, and maintained
the stable-placement criterion. The green marker visible in the final frame is
the obsolete target after the seeded switch.

## Visual review

The locally retained captures are:

- `videos/batch_profile_1_fixed_naive_async.mp4` (4.05 s, 1280x720, H.264,
  20 fps)
- `videos/batch_profile_1_fixed_aligned_async.mp4` (11.35 s, 1280x720, H.264,
  20 fps)
- `videos/m8_profile1_seed2026081101_naive_fail_vs_aligned_success.mp4`
  (labelled side-by-side composite, 11.35 s, 1280x360, H.264, 20 fps)

All three MP4s decode end-to-end with FFmpeg 8.1.1. Git retains the compact
[paired final frame](paired_success_final.png), hashes, raw logs, summaries,
replay audit, and native receipts while the MP4 payloads remain local and are
ignored by Git.

## Scope boundary

This example is one development seed, not an independent statistical claim.
The matching 12-seed Profile-1 result is 0/12 naive versus 10/12 aligned with a
paired-bootstrap 95% CI of [+58.3, +100.0] percentage points. Neither the
single video nor that development matrix substitutes for the frozen M8
holdout, a learned-policy comparison, or a real-robot run.

Evidence: [replay validation](replay_validation.json),
[video validation](video_validation.json),
[naive summary](raw/summaries/m8-development-profile_1_fixed-2026081101-naive_async.json),
[aligned summary](raw/summaries/m8-development-profile_1_fixed-2026081101-aligned_async.json),
and [native completion receipt](native_run_logs/20260815T225519395475168Z/completion_receipt.json).
