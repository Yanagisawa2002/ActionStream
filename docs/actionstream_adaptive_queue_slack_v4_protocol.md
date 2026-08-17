# ActionStream Adaptive Queue-Slack v4 development protocol

Frozen at `2026-08-17T14:45:00Z`, before any v4 learned-policy canary row was
run. Selector:
`configs/actionstream_adaptive_queue_slack_v4_selector.json`, SHA-256
`7e437cbaeb3d274e098d8a068fafa482697148933eb5431c28c48cd51632a021`.

## Evidence boundary

Adaptive v2 remains a formal `NO-GO`. Adaptive Phase-Stable v3 remains a
`PARTIAL_POSITIVE` canary with a separate `NO-GO` mechanism probe. Their
selectors, states 30/31, traces, videos, and verdicts are immutable and are not
relabeled as v4 evidence. V4 is a new development candidate; no v4 formal
holdout is registered.

The v4 selector and the three canary manifests must not change after the first
canary row is read. A change to scheduler constants, tasks, state, seed,
network trace, success rule, or episode limit requires a new selector ID and a
new disjoint development split.

## Method registered before the canary

V4 inherits v3's age-aligned execution, risk gate, queue-preserving rejection,
minimum mode residency, and gripper phase lock. It adds a task-label-blind
request scheduler using only online quantities available before the next
action is selected:

1. Observe request-to-delivery service time in control steps and update a
   clipped EWMA (`alpha=0.5`, maximum 28 steps). The delivered control-step age
   is a lower bound, so head-of-line blocking is observable.
2. Prefetch when queue depth is at or below
   `min(chunk_size-1, ceil(service_EWMA)+5)`. Request spacing is
   `clip(chunk_size-ceil(service_EWMA)-5, 3, 10)` control steps.
3. If a request is still unresolved when executable depth reaches five, repeat
   the last finite absolute command without consuming those five queued
   commands. A replenishing accepted result releases the reserve.
4. Reject hard-stale or unsafe arrivals without clearing the reserve. This
   addresses the v3 failure mechanism in which every hard outage arrived only
   after queue depth had reached zero.

The runtime may not inspect a delay-profile key, task/suite identity, success,
future observations, or canary results. Each episode must record prefetch
requests, observed/estimated service steps, reserve activations/releases/hold
steps, outstanding requests, queue-preserving rejections, and guard discards.

## Frozen disjoint canary

All three task families use state 32, which is disjoint from v3 states 30/31:

| Family | Suite/task | State | Environment seed | Low/high trace seeds |
|---|---|---:|---:|---:|
| Object | `libero_object` task 5 | 32 | 2026081751 | 2026081761 / 2026081771 |
| Spatial | `libero_spatial` task 7 | 32 | 2026081752 | 2026081762 / 2026081772 |
| Goal | `libero_goal` task 2 | 32 | 2026081753 | 2026081763 / 2026081773 |

Each suite compares sync capability, official LeRobot `latest_only`, static
ActionStream aligned, and v4 Adaptive on the same reset and trace. The three
new operating profiles are:

- low jitter: seeded uniform 25--125 ms;
- high jitter: seeded uniform 600--1100 ms;
- outage: 425 ms base with 1850 ms delays at request ordinals 2 and 6.

The outage trace SHA-256 is
`f48ae269042496be4569dc229d8f2c8461dfd3c9a754f447819ed101a3a51b1d`.
The Object/Spatial/Goal low-jitter hashes are respectively
`8e6ba535fa128eb33bf75f622cc05451418c647968ec4276bd8041b1bff18c6c`,
`bc93e85f1d5881dabcd27ec3941e668e06e9fe2a26407ff497366cf67a0e611f`,
and `56ad6a01e22b81ddb81a6a8ec48981cff7350801af8f37e8317dcd4d2c024031`.
The high-jitter hashes are
`962823c931a928e998aa9b9cc57d710ba2465e1bd02a9fd3939cb7aff9a16e7e`,
`e686f6ab786e9e2fa6af877fd659dbee80bf0f96cffb342be8d411d0e796a911`,
and `7f7aab944cbf3b4eb41978f7496bdcbdbe3e837c5e011bc3a5c09cc5e5eb167c`.

Manifest SHA-256 values are Object
`88bd26354614872cfc29b1becbd71ba23a0cd11c85f50e7bdce6254034bf2451`,
Spatial
`12a81f489d603f42c831cda2113bddd5350cf075eaf8960e9df7c3b844c0b057`,
and Goal
`cabe2d0bd2790598e07bc2aa1f3081e1e92007e01857b2d196782f4c8b709cdb`.

## Predeclared verdict gate

The exact candidate is `GO` only if all conditions hold:

- sync succeeds on all three low-jitter task/state pairs, proving policy
  capability on the new resets;
- Adaptive succeeds on all nine task/profile pairs, with no success regression
  against a successful official latest-only pair;
- every Adaptive row records at least one prefetch request;
- every Adaptive outage row observes both frozen outage ordinals, activates the
  reserve, preserves a nonempty queue through at least one rejected result,
  and records zero pending actions discarded by the guard;
- all 36 rows complete without runtime exceptions, all traces are finite and
  hash-valid, and every requested video decodes.

Completion steps versus static runtimes are diagnostic because there is only
one reset per family/profile. Passing permits registration of a new disjoint
multi-state holdout; it is not itself a generalization or 95% CI claim.

## Baseline and hardware boundary

`latest_only` is pinned upstream LeRobot and aligned is the existing
ActionStream target-step queue. X-VLA reports no RTC support, so RTC is not
faked inside Adaptive. This is learned-policy LIBERO simulation, not Isaac
Lab-Arena and not a real-robot result.
