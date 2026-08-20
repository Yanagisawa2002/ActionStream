# ActionStream LeRobot GPU backend v1 results

## Verdict

The formal X-VLA GPU benchmark is complete and auditable, but it is a mixed
systems result rather than a general algorithmic win. The ActionStream aligned
backend matched official latest-only at fixed 250 ms and jitter, improved burst
success by one paired episode, and regressed significantly at fixed 950 ms.
The guarded backend is a formal **NO-GO** under fixed 950 ms and burst faults.

The native learned-policy Isaac work remains development evidence. X-VLA safely
completed the Object0 and Spatial2 proxy tasks. Goal2 reached both the official
LIBERO predicate and the native predicate, but exceeded the unchanged 40 N
collision gate (96.46 N), so it is a safety **NO-GO**. A native paired runtime
holdout was not run because the current native runner has neither disjoint reset
support nor the formal asynchronous runtime backends.

## Frozen formal benchmark

- Model: X-VLA, one frozen checkpoint/revision per paired condition.
- Task families: `libero_object`, `libero_spatial`, and `libero_goal`.
- Formal episodes: 375 (15 paired episodes for every runtime/network operating
  point); canary plus holdout evidence totals 435 episodes and 435 traces.
- Runtimes: sync-hold, official LeRobot latest-only, official LeRobot weighted
  average Async, ActionStream aligned, and ActionStream guarded.
- Network profiles: zero delay, fixed 250 ms, fixed 950 ms, frozen jitter, and
  frozen burst/outage.
- Pairing invariants include task, reset, seed, checkpoint, delay-trace hash,
  control frequency, and chunk size.

## Headline quantitative results

Success counts below are pooled over the three task families (`n=15` paired
episodes per cell).

| Profile | Latest-only | Aligned | Guarded | Official Async | Sync-hold |
|---|---:|---:|---:|---:|---:|
| Fixed 0 ms | 11/15 | 12/15 | 12/15 | 10/15 | 14/15 |
| Fixed 250 ms | 13/15 | 13/15 | 13/15 | 9/15 | 14/15 |
| Fixed 950 ms | 14/15 | 10/15 | 8/15 | 11/15 | 14/15 |
| Jitter | 14/15 | 14/15 | 12/15 | 8/15 | 14/15 |
| Burst/outage | 12/15 | 13/15 | 8/15 | 9/15 | 14/15 |

Against official latest-only, aligned burst success changed by +0.0667
(paired bootstrap 95% CI `[0.0000, 0.2000]`), while its completion steps changed
by +5.67 (`[-21.13, 27.00]`). At fixed 950 ms, aligned success changed by
-0.2667 (`[-0.5333, -0.0667]`) and completion steps by +41.53
(`[3.40, 84.33]`). Guarded was still worse: -0.4000 success
(`[-0.6667, -0.1333]`) and +66.53 steps (`[23.87, 111.40]`).

Sync-hold achieved 14/15 at every network profile and 130.13 mean environment
steps because the simulator waits for inference and transport. It is a useful
behavioral upper bound, not a real-time deployment result.

## GPU and runtime evidence

- RTX 5090 formal execution; 12,258.95 MiB peak *allocated CUDA memory* during
  scored episodes. This is not total `nvidia-smi` process residency, which was
  not sampled formally.
- The formal table reports GPU action throughput, GPU-synchronized model
  inference p50/p95, end-to-end delivery p50/p95, queue age, discards,
  depletion, recovery, and fallback where each runtime exposes the metric.
- 75,043 final 7-D actions, 435 episode traces, 135 videos, and 10 receipts were
  verified against the generated manifests.
- The disconnect/recovery check is a backend-only canary and is excluded from
  fair paired effects against official baselines.

SmolVLA RTC completed a 9/9 plumbing canary and the upstream RTC capability was
verified, but SmolVLA failed the frozen zero-delay sync gate. No formal SmolVLA
or RTC holdout result is claimed.

## Native learned-policy Isaac development evidence

| Task | Official predicate | Native predicate | Collision gate | Development status |
|---|---:|---:|---:|---|
| Object0 | pass at step 150 | pass | 0 N max | PASS |
| Spatial2 | pass at step 120 | pass | 0 N max | PASS |
| Goal2 | pass at step 90 | pass | 96.46 N max | SAFETY NO-GO |

All three runs loaded the learned X-VLA checkpoint, used CUDA inference, drove
native Franka motion, dynamically synchronized robot/object state, evaluated
the official render bridge, and recorded video. They all use development reset
index 0 and are explicitly ineligible for headline or holdout claims.

Isaac Lab-Arena was not executed. Isaac Sim 6.0.1 was available, but the
official Isaac Lab/Arena stack and task registry were absent. The repository's
adapter is structural integration only and is not labeled as an Arena result.

## Evidence

- [`formal_xvla/report.md`](formal_xvla/report.md): full main table and paired effects.
- [`formal_xvla/latency_success_operating_points.png`](formal_xvla/latency_success_operating_points.png): sampled operating points, not a continuous curve.
- [`formal_xvla/failure_taxonomy.csv`](formal_xvla/failure_taxonomy.csv): failure classes.
- [`native_dev_summary.json`](native_dev_summary.json): native development gates.
- [`qualitative/`](qualitative/): inspected paired contact sheets and native task contact sheets.
- [`local_transfer_receipt.json`](local_transfer_receipt.json): local/remote archive hashes and evidence boundary.
- [`remote_shutdown_receipt.json`](remote_shutdown_receipt.json): authorized shutdown invocation and bounded offline check.

The 166 MiB raw replay/video archive and the selected video bundle are retained
locally and remotely with matching SHA-256 hashes and are intentionally excluded
from ordinary Git history.
