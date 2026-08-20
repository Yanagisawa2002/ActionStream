# ActionStream v1.1.0

v1.1.0 turns the research prototype into a reviewable LeRobot inference
backend and publishes the first formal learned-policy GPU systems matrix.

## Included

- `ActionStreamInferenceEngine` lifecycle, asynchronous worker, thread-safe
  aligned queue, stale-response rejection, bounded depletion hold, timeout,
  disconnect/recovery, latest-only fallback, and telemetry.
- A LeRobot third-party inference-engine plugin and the upstream extension
  boundary tracked in huggingface/lerobot PR #4466.
- A frozen X-VLA benchmark across LIBERO Goal, Object, and Spatial with sync,
  official latest-only, official weighted-average Async, aligned, and guarded.
- Five frozen network profiles: 0 ms, 250 ms, 950 ms, jitter, and burst/outage.
- GPU throughput, allocated CUDA memory, inference and delivery latency,
  queue age, discard, depletion, recovery, and fallback evidence.
- A 74-second paired demo with positive and negative result cells.

## Formal result

The matrix contains 375 holdout condition-episodes. Canary plus holdout
evidence totals 435 episodes and traces, 135 videos, and 75,043 verified final
7-D actions. Peak allocated CUDA memory was 12,258.95 MiB on one RTX 5090.

Latest-only remains the primary reference. Aligned matched its success at
250 ms and jitter and gained one success under burst/outage, but regressed at
950 ms from 14/15 to 10/15. Guarded regressed to 8/15 at both 950 ms and burst
and is a formal NO-GO.

The aligned-versus-official-Async result is explicitly secondary. Under frozen
600 ± 400 ms jitter, aligned improved success from 8/15 to 14/15: +40.0
percentage points with a paired 95% bootstrap CI of `[13.3, 66.7]`, while mean
steps fell from 207.20 to 144.87. This post-holdout analysis reuses unchanged
predeclared cells and does not replace the primary comparator.

## Native and RTC boundaries

Native learned-policy Isaac runs are development evidence only. Object0 and
Spatial2 passed; Goal2 reached its task predicate but violated the frozen 40 N
collision gate at 96.46 N. No native paired holdout or real-robot safety claim
is made.

SmolVLA v2 and v3 used separately frozen zero-delay sync canaries across Goal,
Object, and Spatial. v2 failed Goal at 1/2; after preserving that NO-GO, v3
failed Object at 1/2. The all-suites 2/2 gate therefore remained closed:
formal holdout records are 0, official RTC records are 0, and no RTC paired
effect or confidence interval exists. See the
[gate report](../../reports/actionstream_backend_gpu_v1/smolvla_rtc_gate_v2_v3/report.md).

## Evidence integrity

- Full raw archive SHA-256:
  `b721f425341b74e5ab081ce59953992b3f13cc90f13064b89e3d592659fb39bc`.
- SmolVLA v2/v3 gate archive SHA-256:
  `8176c702c3af827937997cadd8c945d4c6a4f1b6f5464ee82f48cf9b684af340`.
- Release asset hashes and video encoding metadata are in
  [`asset_manifest.json`](asset_manifest.json).
- The full report and paired confidence intervals are in
  [`../../reports/actionstream_backend_gpu_v1/formal_xvla/report.md`](../../reports/actionstream_backend_gpu_v1/formal_xvla/report.md).
- Raw traces, full video sets, model weights, and simulator assets are excluded
  from ordinary Git.

## License

First-party ActionStream source is Apache-2.0. Third-party assets and code
retain their upstream licenses; see `THIRD_PARTY_NOTICES.md`.
