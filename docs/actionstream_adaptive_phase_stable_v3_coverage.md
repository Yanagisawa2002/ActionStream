# Adaptive Phase-Stable v3 post-canary mechanism coverage

Registered at `2026-08-17T14:00:00Z`, after the frozen state-30 canary had
completed. This is a diagnostic coverage probe, not a canary rerun, formal
holdout, performance comparison, or retrofit of the original verdict.

## Why a separate probe exists

The state-30 canary completed 36/36 rows and all nine Adaptive rows succeeded.
All hashes and videos validated. However, success termination censored request
ordinal 13 in the Goal Adaptive and Spatial aligned rows, and no learned-policy
row exercised a queue-preserving phase-lock rejection. The original manifests,
selector hash, rows, and `PARTIAL` coverage finding remain unchanged.

The probe retains selector
`actionstream_adaptive_phase_stable_v3` at SHA-256
`c824762ed9b05d37369181812795313c870b3bf8f5714d4c4ccd98e221021c98`.
It changes only the development reset and deterministic network traces.

## Frozen state-31 matrix

The probe uses Object task 5, Spatial task 7, and Goal task 2 at previously
unused state 31. Each task runs Adaptive v3 under exactly three profiles:

- zero delay, to show the new reset is executable;
- 350 ms base delay with 1700 ms outages at request ordinals 2 and 5, so both
  registered outages occur before normal task completion;
- 350 ms base delay with one 0 ms fresh-return pulse during a task-specific
  closed-gripper window observed in the completed state-30 trace: ordinal 10
  for Object and ordinal 8 for Spatial and Goal.

Using the completed trace to choose the pulse is disclosed post-canary branch
coverage, not blind performance evaluation. State 31 is fixed before any probe
row is read. The pulse ordinal may not move if the branch does not fire.

## Coverage gates

The probe passes only if:

1. each zero-delay row succeeds;
2. each early-outage row records 1700 ms at both ordinals 2 and 5, completes
   without a runtime exception, and discards zero pending commands by the
   rejection guard;
3. each fresh-return row actually records its 0 ms override and reports at
   least one `adaptive_phase_lock_rejections` and one
   `adaptive_preserved_queue_rejections`, with zero guard discards;
4. all nine traces match their recorded hashes and all nine MP4s decode with
   non-static content.

Task success and completion steps in the two targeted profiles are diagnostic.
They cannot support an effect-size or superiority claim because this probe has
one runtime and branch-targeted traces.
