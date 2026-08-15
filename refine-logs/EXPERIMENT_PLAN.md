# Experiment Plan

**Problem**: Current VLA policies produce action chunks whose prediction time can overlap execution, but queue aggregation, inference delay, jitter, and policy-specific RTC support change the closed-loop outcome.
**Method Thesis**: ActionStream should be evaluated as a transparent target-step baseline against the current official LeRobot asynchronous and RTC runtimes, not presented as a novel replacement for them.
**Date**: 2026-08-15

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1: Runtime behavior is policy- and delay-dependent | The frozen M4 X-VLA result and synthetic M6 audit cannot establish current cross-model value | Paired closed-loop LIBERO results for at least two frozen policies, multiple fixed delays, one seeded jitter profile, and current official LeRobot baselines | B1, B2, B3 |
| C2: Target-step freshness changes live recovery in native physics | Synthetic queues and deterministic ROS plants do not prove behavior in Isaac physics | Same-seed, same-reset native Isaac episodes driven from live observations under at least naive and aligned execution, with replay-valid traces and video | B4 |

Anti-claims to rule out:

- Any apparent gain is only a timeout artifact, unequal reset, different action representation, or different policy checkpoint.
- The X-VLA stale-prefix rule is equivalent to current LeRobot Async or RTC.
- A ROS test plant, scripted trace replay, or structural Isaac check is a native policy-driven Isaac result.

## Paper Storyline

- Main result must report the current official baseline that wins per compatible policy and profile, even when ActionStream loses.
- Appendix can contain full delay curves, queue diagnostics, and the optional third policy.
- Training new policies, modifying policy weights, and calling incompatible RTC paths are intentionally cut.

## Experiment Blocks

### B1: Compatibility and zero-delay parity

- Claim tested: C1.
- Why this block exists: prevents API drift or processor differences from masquerading as runtime effects.
- Dataset / split / task: LIBERO Object tasks 0-2; one fixed episode per policy for smoke, followed by three paired episodes if parity passes.
- Compared systems: synchronous reference, official LeRobot Async `weighted_average`, official `latest_only`, ActionStream aligned; RTC only when `policy.supports_rtc()` passes.
- Metrics: success, executed steps, wall time, action-shape/finite checks, reset identity, zero-delay trace agreement.
- Setup details: LeRobot commit `6adf51511b7625090eade8d82d9f61a1846ebe56` (source 0.6.2); frozen model revisions in the run manifest; policy-native control mode and processors.
- Success criterion: every compatible runtime launches, emits valid environment-ready actions, and preserves the synchronous task success on its smoke episode.
- Failure interpretation: stop that policy/runtime cell and report incompatibility; never coerce RTC into an unsupported policy.
- Table / figure target: compatibility matrix.
- Priority: MUST-RUN.

### B2: X-VLA current Async comparison

- Claim tested: C1.
- Why this block exists: directly updates M4 from naive replacement to official current Async aggregation in the original closed-loop setting.
- Dataset / split / task: LIBERO Object tasks 0-2, paired initial states and seeds disjoint from tuning.
- Compared systems: `sync_hold`, official `weighted_average`, official `latest_only`, ActionStream aligned. X-VLA RTC is excluded unless the pinned policy explicitly reports support.
- Metrics: paired success, steps, wall time, hold/underrun fraction, stale actions executed, temporal error, adjacent-action and replacement-boundary discontinuity.
- Setup details: X-VLA revision `12e8783e996944f5c97e490d37d4c145484ed70a`, 20 Hz, chunk 30, request interval 10; fixed delays 0/250/500/950 ms plus seeded delivery jitter centered at 500 ms with +/-250 ms support.
- Success criterion: complete paired raw traces and confidence intervals; no directional result is required.
- Failure interpretation: a stronger official runtime closes the resume novelty claim and becomes the recommended baseline.
- Table / figure target: main delay-profile table and success/steps curves.
- Priority: MUST-RUN.

### B3: RTC-capable policy comparison

- Claim tested: C1 and the frontier-necessity anti-claim.
- Why this block exists: RTC changes policy generation, not only queue replacement, and is the strongest current compatible baseline.
- Dataset / split / task: the same LIBERO Object task IDs, paired within policy.
- Compared systems: synchronous chunking, official Async `weighted_average`, official Async `latest_only`, RTC, and a target-step aligned queue where the final action contract permits it.
- Metrics: the B2 metrics plus inter-chunk velocity/acceleration peaks and measured inference-delay estimate.
- Setup details: primary RTC policy `lerobot/pi05_libero_finetuned` revision `8e174154ef5f6c60a8da12ae99c303d8963138c1`; optional third policy `lerobot/smolvla_libero` revision `31d453f7edd78c839a8bbc39744a292686daf0de`; policy-native chunk length, processors, and control mode.
- Success criterion: at least Pi0.5 completes the smoke and the fixed/jitter decision matrix with RTC enabled through the official implementation.
- Failure interpretation: report the exact compatibility or memory blocker; do not substitute an imitation of RTC.
- Table / figure target: cross-policy method table and boundary-smoothness plot.
- Priority: MUST-RUN for Pi0.5; NICE-TO-HAVE for SmolVLA.

### B4: Native Isaac live-policy paired evidence

- Claim tested: C2.
- Why this block exists: closes the current M8 native-physics evidence gap.
- Dataset / split / task: M8 `dynamic_target_pick_place_v1`; preregistered baseline gate first, then development seed `2026081100` under the fixed-delay profile.
- Compared systems: `naive_async` and `aligned_async`; include `sync_hold` when the generated suite requires the complete pair set.
- Metrics: task success, obsolete-destination execution, recovery latency, holds, collision/timeout reason, reset fairness, replay validation, wall/simulation time, and paired video.
- Setup details: official Isaac Sim 6.0.1 Franka, ROS 2 Jazzy, live observation-conditioned Cartesian waypoint policy, frozen 20 Hz/30-action/10-step request contract.
- Success criterion: native runner completion receipt, identical paired reset within tolerance, replay-valid raw traces, and decodable video for at least one same-seed strategy pair.
- Failure interpretation: retain the exact native blocker; ROS test-plant results remain ineligible substitutes.
- Table / figure target: qualitative paired timeline and one-row native evidence table.
- Priority: MUST-RUN.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Freeze current source/model identities and validate adapters | CPU tests plus one X-VLA zero-delay smoke | All action contracts and provenance checks pass | <1 GPU-hour | Current API drift |
| M1 | Select official Async baseline | X-VLA task-0, one pair at 0/500/950 ms and jitter | At least one official Async mode and aligned complete identically seeded episodes | 1-2 GPU-hours | Queue timestamp convention |
| M2 | Establish RTC compatibility | Pi0.5 task-0 zero-delay and 500-ms smoke | Official `supports_rtc()` and environment success both pass | 1-2 GPU-hours plus model download | Memory or processor mismatch |
| M3 | Run compact decision matrix | Tasks 0-2, three paired episodes, fixed delays and jitter | Complete raw paired cells before expanding | 8-16 GPU-hours | Timeouts dominate cost |
| M4 | Close native Isaac gap | 20-episode sync baseline gate, then one paired development seed | Native receipt, fairness, replay, and video pass | 2-6 GPU-hours plus install | Isaac/ROS installation |
| M5 | Optional expansion | Ten episodes/task and SmolVLA | Run only if M3/M4 produce interpretable evidence | 20+ GPU-hours | Weak return on compute |

## Compute and Data Budget

- Initial must-run budget: one RTX 5090, approximately 12-24 GPU-hours after dependencies are cached.
- Model downloads: X-VLA approximately 3.28 GiB; Pi0.5 6.96 GiB; SmolVLA approximately 0.84 GiB optional. Cache state is recorded by each remote run rather than assumed by the protocol.
- Data preparation: no training data download; only model/config assets and standard LIBERO/Isaac assets.
- Human evaluation: paired video inspection after metrics and replay validation.
- Biggest bottleneck: Isaac Sim 6.0.1/ROS workspace installation, not model inference.

## Risks and Mitigations

- Current LeRobot Async is designed around a robot client/server: call pinned upstream timing and aggregation methods directly in the LIBERO adapter and record the exact source references.
- RTC is policy-specific: require the official capability check and use no fallback implementation.
- Cross-policy action spaces differ: compare runtimes only within the same policy and native control mode.
- Jitter can break pairing: pre-generate immutable per-request delay traces and reuse their hashes across methods.
- Existing M8 worktree changes are unfinished: preserve them, validate them separately, and commit by scope.

## Final Checklist

- [ ] Main tables use paired raw episodes.
- [ ] Current LeRobot source and model revisions are frozen.
- [ ] Novelty is not claimed from a known baseline.
- [ ] RTC is evaluated only on officially supported policies.
- [ ] Fixed-delay and jitter profiles share immutable traces.
- [ ] Native Isaac evidence is distinguished from ROS/test-plant evidence.
- [ ] Nice-to-have runs do not delay the compact decision matrix.
