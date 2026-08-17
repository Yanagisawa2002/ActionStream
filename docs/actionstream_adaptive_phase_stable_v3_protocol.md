# ActionStream Adaptive Phase-Stable v3 development protocol

Frozen at `2026-08-17T13:35:00Z`. Selector file:
`configs/actionstream_adaptive_phase_stable_v3_selector.json`, SHA-256
`c824762ed9b05d37369181812795313c870b3bf8f5714d4c4ccd98e221021c98`.

## Evidence boundary

Adaptive v2 and its 270-episode formal holdout are immutable negative
evidence. V2's formal verdict remains `NO-GO`: it was useful at fixed 950 ms,
but regressed under jitter and burst/outage and did not beat static aligned.
No v2 threshold, row, task, state, or trace is relabeled as v3 evidence.

V3 is a new development candidate. Its canary may be inspected, but the
selector file and canary manifests must not change after the first canary row
is read. A failure makes this exact candidate `NO-GO`; any revision requires a
new selector ID, file hash, and disjoint development split. No v3 formal
holdout is registered yet.

## Structural changes from v2

The age and frame-local risk thresholds are inherited unchanged. V3 changes
execution semantics in three preregistered ways:

1. Ages 21--28 steps use target-step alignment instead of returning to
   latest-only. The first command actually dispatched by official latest-only,
   rather than raw chunk index zero, is used for its risk check.
2. A stale or unsafe incoming chunk no longer clears a nonempty validated
   active queue. The runtime rejects the arrival, drains that queue, and holds
   the last finite action only after the queue is empty.
3. Backend changes have a ten-control-step minimum residence. Once the
   executed X-VLA/LIBERO gripper command reaches the closed phase (`>= +0.5`),
   a nonempty active queue is locked to its backend. Three consecutive open
   commands (`<= -0.5`) release the lock. This is action-state hysteresis; it
   does not read task identity, success, or a network-profile label.

| Delivered result age | Preferred execution |
|---:|---|
| 0--5 steps | Official LeRobot latest-only |
| 6--20 steps | ActionStream age-aligned |
| 21--28 steps | ActionStream age-aligned |
| More than 28 steps or unsafe | Reject arrival; preserve active queue, else safe hold |

Each episode records proposed and effective mode counts, effective switches,
phase-lock steps/rejections, residency rejections, queue-preserving rejections,
empty-queue holds, and any commands discarded by a guard.

## Frozen canary

The canary uses tasks, state, seeds, and traces not used by the v2 canary or
formal holdout:

- LIBERO Object task 5, state 30, seed 2026081730;
- LIBERO Spatial task 7, state 30, seed 2026081731;
- LIBERO Goal task 2, state 30, seed 2026081732;
- zero delay, fixed 950 ms, and a deterministic 350 ms trace with 1700 ms
  outages at inference-request ordinals 5 and 13;
- sync, official LeRobot latest-only, static ActionStream aligned, and Adaptive
  Phase-Stable v3 on the same reset and trace.

Request ordinals 5 and 13 were registered before running the new tasks. They
probe the typical post-approach/grasp and late manipulation/release windows;
the trace must record the actual gripper phase reached at each arrival. They
are not allowed to move after viewing a video.

The exact v3 canary is a `GO` only if all of the following hold:

- X-VLA sync at zero delay succeeds on all three task families, establishing
  that the selected learned-policy tasks are executable;
- Adaptive succeeds on all nine task/profile pairs and therefore has no
  one-pair success regression against a successful static runtime;
- every action/trace is finite and replayable, every requested video decodes,
  selector/protocol hashes match, and no pending action is discarded by the
  v3 rejection guard;
- phase/outage rows contain the registered delay events and complete without a
  runtime exception. Phase-lock activity is reported, not required to be
  fabricated when a task never reaches closed gripper.

Completion steps and regret to the best static runtime are diagnostic canary
metrics, not headline claims. Passing this gate permits registration of a new,
disjoint multi-state formal holdout; it does not itself establish superiority.

## Baseline and real-robot boundary

`latest-only` is the pinned upstream LeRobot implementation. `aligned` is the
existing ActionStream target-step queue. X-VLA reports no RTC support, so RTC
remains an external official baseline and is not faked inside v3. The canary is
learned-policy LIBERO simulation, not Isaac Lab-Arena and not a real-robot A/B.
A real-robot claim still requires a named robot/driver, calibrated limits,
operator and E-stop receipt, paired physical resets, real network traces, and
synchronized videos.
