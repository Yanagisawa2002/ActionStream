# Asynchronous timing and bounded physical recovery

Implementation is under validation. Historical synchronous success rates do not
validate this controller. The new protocol is
[`async_agent_acceptance_v1.json`](../configs/async_agent_acceptance_v1.json).

The installed `actionstream-agent` accepts `--execution-mode async` or
`--execution-mode recovery` with the same verified assets and language contract
as the [finite Agent](finite-agent.md). The default remains `sync`.

## Execution boundary

- The production `ActionStreamInferenceEngine` owns VLA inference and processor
  reset on its worker. The control thread owns the simulator and physical writes.
- Controls are paced at 50 ms. Startup warmup has a separate timing record. Slow
  iterations are retained and the loop never issues a burst to catch up.
- VLA warmup runs on its actual inference worker, with a thirty-second startup
  cap. The warmup queue is invalidated before refreshing the unmoved initial
  observation. Its extra engine epoch is recorded separately from request
  revisions. RGB warmup runs on the control owner.
- Every dispatched policy action carries its source observation, epoch and
  request ordinal. Foreign revisions and sources older than one second are
  rejected. Queue starvation permits at most twenty pose holds before an
  explicit unavailable outcome. Latest-only stale fallback is disabled.
- RGB confirmation requires fresh observations, consecutive controls and an
  unchanged request/revision. Results arriving after the 150 ms freshness limit
  cannot confirm success. A stale observation or retry clears RGB history.
- Confirmation invalidates pending actions before forty physical hold/open
  controls. No policy actions are pulled during this continuation.
- Recovery allows one retry after 300 policy controls plus 60 settling controls
  if fresh RGB still says incomplete. It clears the old queue, increments the
  revision, opens the gripper for twenty controls, and resets policy state on
  the worker before resuming from the current physical scene. It never resets
  the object layout or reads private truth to choose an action.

Direct CUDA cancellation invalidates results but cannot preempt an executing
kernel. The engine reports advisory deadlines. Measured simulation timing is
separate from a real-robot control guarantee.

## Acceptance

First complete current-runtime delivery with isolated offline caches. Freeze
source, protocol and exclusions before the new layouts. The normal set has ten
new seeds; the recovery set has ten other new seeds with the gripper physically
clamped open during the first 300 commands. The controller receives only the
resulting RGB/proprioception and does not receive the fault schedule.

The private scorer reconstructs released/stable truth, checks confirmation,
source identities, physical actions, full trajectory coverage and RGB clip
hashes. It reports every timing miss. Recovery requires actual first-attempt
physical failure and later stable completion in the same scene. Cases without
a proven failure do not count as recovered, and failures remain in the
predeclared denominator. No seed replacement, fitting, or performance retry is
allowed after acceptance begins.

The two-second confirmation-delay gate applies to both simulated time and
elapsed wall time. Initial development exposed a thread-specific CUDA cold
start (5.19 seconds versus 75 milliseconds for the next inference); the failed
development run is retained separately and no new acceptance seeds were used.

```bash
python scripts/engineering/evaluate_async_agent.py freeze \
  --store "$STORE" --output "$STORE/async-acceptance-v1" --source-commit "$COMMIT"
python scripts/engineering/evaluate_async_agent.py development \
  --store "$STORE" --output "$STORE/async-acceptance-v1" --source-commit "$COMMIT"
python scripts/engineering/evaluate_async_agent.py normal \
  --store "$STORE" --output "$STORE/async-acceptance-v1" --source-commit "$COMMIT" \
  --delivery-receipt "$STORE/current-delivery.json"
python scripts/engineering/evaluate_async_agent.py recovery \
  --store "$STORE" --output "$STORE/async-acceptance-v1" --source-commit "$COMMIT" \
  --delivery-receipt "$STORE/current-delivery.json"
```

Use the same isolated EGL/offline environment as the delivery runbook. Each
phase is single use. Keep the entire output directory, including every NPZ,
and verify its file hashes after copying to a second storage location.
