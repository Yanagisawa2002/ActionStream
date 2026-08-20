# ActionStream LeRobot GPU benchmark protocol v1

Frozen at 2026-08-20T10:06:00Z after the development-only state-36 smoke and before any state-37 canary or state-40--44 holdout result was opened.

## Candidate and provenance

- Base checkout: `6751b25adea2748b198935cdef6e1b3995b3a5c5`.
- Runtime entry point SHA-256: `src/actionstream/current_baselines.py` = `0e929c7ee82acfed38d06ab67a36aadbdeb22b7b1b0f34eca0f35ccd4722d0fd`.
- Inference engine SHA-256: `src/actionstream/lerobot_inference.py` = `d04549c847f7101ab78df52567a273cf1a9b707d0c219c6033237bd36cdb19d2`.
- Upstream LeRobot: `6adf51511b7625090eade8d82d9f61a1846ebe56` (0.6.2).
- X-VLA: `lerobot/xvla-libero@12e8783e996944f5c97e490d37d4c145484ed70a`.
- SmolVLA: `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`; RTC is only compared within this policy.

Any runtime-source change after this freeze invalidates v1 and requires a new protocol namespace. Analysis/report-only code may be added without changing the candidate.

## Data boundary

| Role | Initial-state indices | Use |
|---|---:|---|
| Development | 36 | implementation and learned-policy smoke; never scored |
| Frozen canary | 37 | gate only; never pooled into holdout |
| Non-scored GPU warmup | 38 | one inference call before steady-state cells |
| Frozen holdout | 40, 41, 42, 43, 44 | formal paired estimates |
| Reserve | 45--49 | untouched by v1 |

Canary, warmup, and holdout base seeds are disjoint. Every seeded jitter/burst trace has a task- and split-specific unused seed. The same generated trace is reused across runtimes within a paired task cell. Holdout traces are not used by canary.

## Three task families

1. LIBERO-Object task 5: pick up the tomato sauce and place it in the basket.
2. LIBERO-Spatial task 7: pick up the black bowl on the stove and place it on the plate.
3. LIBERO-Goal task 2: put the wine bottle on top of the cabinet.

These are different official LIBERO suites, not different resets of one task.

## Runtime cells

X-VLA is paired across `sync_hold`, upstream `lerobot_weighted_average` (official Async aggregation), upstream `lerobot_latest_only`, `actionstream_backend_aligned`, and `actionstream_backend_guarded`. The last two instantiate the formal `ActionStreamInferenceEngine`; guarded alone enables latest-only fallback after bounded hold/depletion.

SmolVLA is paired separately across `sync_hold`, upstream `lerobot_latest_only`, and upstream `lerobot_rtc`. Cross-model effect pooling is forbidden.

Holdout operating points are zero delay, fixed 250 ms, fixed 950 ms, seeded 200--1000 ms jitter, and a seeded 150--350 ms base trace with 1.6--2.2 s bursts. The canary also includes a backend-only disconnect profile at request ordinals 0 and 7. That profile tests recovery telemetry and is excluded from effects against official baselines because the local official adapter has no equivalent recoverable transport fault hook.

## Predeclared gates

- X-VLA canary must write completed records for all declared non-disconnect cells, achieve sync success on all three zero-delay tasks, and show nonfatal disconnect/recovery in both backend modes.
- SmolVLA object canary must complete sync/latest-only/RTC cells, declare RTC support through the pinned upstream capability, and achieve zero-delay sync success before its holdout is started.
- A gate failure stops that model's holdout. It may be debugged only in a new development namespace; v1 is not edited or rerun with changed thresholds.
- After a holdout starts, no result-driven runtime, task, state, trace, threshold, or episode-length change is permitted.

## Required report

Report success rate, paired effect with 95% CI, environment steps, environment throughput, allocated peak VRAM, p50/p95 model and delivery latency, queue depth/age where the runtime exposes valid provenance, dropped/stale actions, bounded hold, depletion, disconnect/recovery, and fallback. Unsupported telemetry is `N/A` with an implementation reason, not zero. Include per-episode trace/hash provenance, failure taxonomy, latency-success operating-point figure, and representative paired videos.

Development smoke, canary, holdout, SmolVLA RTC, and native Isaac results remain separate evidence classes. LIBERO success does not prove Isaac or real-robot safety.
