# Learned-policy closure report — 2026-08-17

## Executive result

The result is a **partial learned-policy closure**, not a completed native
Isaac benchmark:

- Real X-VLA inference dynamically drives native Isaac robot and object state
  on three genuinely different manipulation task families. Two of the three
  native sync canaries reached their pre-existing development success
  condition; the spatial placement canary also passed the pinned LIBERO
  `_check_success` predicate after exact state writeback.
- The official LeRobot/LIBERO evidence already compares upstream Async
  aggregation, `latest_only`, ActionStream aligned execution, and RTC where the
  policy supports RTC. These are real learned-policy episodes, but they are a
  separate evidence class from the native Isaac sync canaries.
- A 180-episode X-VLA expansion gives the clean positive result: under seeded
  500 +/- 250 ms jitter, both `latest_only` and ActionStream succeeded 30/30,
  while ActionStream completed 13.7 steps sooner on average (paired bootstrap
  95% CI -15.6 to -11.9).
- The repository presentation assets are ready, but public release is not:
  the GitHub repository is private and the project has no first-party license.
  No license was invented and `master` was not merged.
- Isaac Lab-Arena and real-robot paired A/B are **not implemented**.

## Evidence classes

| Evidence class | Simulator/runtime | What it establishes | What it does not establish |
|---|---|---|---|
| Native learned sync canary | X-VLA -> official-render/state bridge -> Isaac Sim 6.0.1 | GPU policy inference, dynamically synchronized robot/object observations, physical Franka motion, video, and two development task successes | A frozen multi-seed holdout or native async-runtime ranking |
| Official paired runtime matrix | Pinned LeRobot 0.6.2 and LIBERO | Fair learned-policy comparison of upstream Async, `latest_only`, RTC where supported, and ActionStream under fixed delay and seeded jitter | Native Isaac performance or real-robot performance |
| M8 frozen native holdout | Deterministic live Cartesian policy in Isaac | Strong runtime/provenance evidence over 420 episodes | Learned-policy quality |

The evidence classes are intentionally not pooled into a single success rate.

## Native X-VLA sync canaries

The policy checkpoint is `lerobot/xvla-libero` revision
`12e8783e996944f5c97e490d37d4c145484ed70a`. Policy inference used an RTX 5090
and peaked at 12,250.825 MiB allocated CUDA memory. Every policy request wrote
the measured Isaac Panda end-effector/gripper and named dynamic object state
into the pinned official LIBERO environment, solved the render robot pose by
IK, rendered official agent and wrist observations, and returned a fresh X-VLA
chunk. Exact object and gripper writeback checks passed.

| Suite / task family | Instruction | Native execution | Result |
|---|---|---:|---|
| `libero_object` task 0 / object into container | Pick up the alphabet soup and place it in the basket | 300 steps, 10 requests | **Development pass**: object moved into the basket proxy; all integration checks passed; no official final predicate was recorded in this older canary |
| `libero_spatial` task 2 / object on object | Pick the black bowl from table center and place it on the plate | 120 steps, 5 requests | **Pass**: native proxy true and pinned LIBERO `_check_success` true at control step 120; all integration checks passed |
| `libero_goal` task 5 / planar contact push | Push the plate to the front of the stove | 300 steps, 10 requests | **Fail**: native and official predicates false; a joint/workspace-limit event disqualified the smoke |

This satisfies the lower bound of two different native task families only as a
development canary. It is not a frozen holdout: one task uses a development
proxy condition and each native task has one reset.

The third task is an informative bridge/physics failure, not evidence that the
checkpoint lacks the task skill. The same pinned X-VLA checkpoint succeeds in
the official LIBERO environment on spatial task 2 in 101 steps and goal task 5
in 148 steps. For goal task 5, the native proxy was upgraded to the canonical
ten oriented plate collision boxes and compiled body mass
0.0115220815776 kg. It still finished with the object near
`[0.755771, 0.002808, 0.002513]`, outside the goal, after abnormal robot-limit
contact. Further seed or geometry tuning was stopped.

Native evidence source commits are `5594d5b` (object task), `2b5e657`
(spatial task), and `7cb6bae` (final failed plate canary). The targeted bridge,
proxy, task, and learned-control contract suites passed 31/31 tests on the
remote checkout. The local Windows Python lacks `pytest`, so no local test
result is substituted for that remote receipt.

## Official LeRobot Async/RTC comparison

The adapter calls the pinned upstream `PolicyServer` timing and `RobotClient`
aggregation paths directly. RTC is invoked only when
`policy.supports_rtc()` is true and uses the upstream RTC queue merge. The
local code injects transport faults and runs the LIBERO control loop; it does
not relabel a local queue as LeRobot Async or RTC. LeRobot was pinned at commit
`6adf51511b7625090eade8d82d9f61a1846ebe56` (v0.6.2).

### X-VLA task/seed expansion

The expansion contains 180 episodes: three LIBERO Object tasks, ten paired
initial states/seeds per task, two runtimes, and zero, fixed 950 ms, and seeded
500 +/- 250 ms jitter profiles.

| Network profile | LeRobot `latest_only` | ActionStream aligned | Paired completion-step effect, aligned - latest |
|---|---:|---:|---:|
| Zero delay | 30/30 | 30/30 | **+3.6** steps, 95% CI [2.4, 4.9] — regression |
| Fixed 950 ms | 30/30 | 28/30 | **-24.7** steps pooled — faster but loses two task-1 trials |
| 500 +/- 250 ms jitter | 30/30 | 30/30 | **-13.7** steps, 95% CI [-15.6, -11.9] |

The jitter gain reproduces by task: -18.0, -10.9, and -12.2 steps, with every
task-level interval excluding zero. The fixed-950 result is a tradeoff, and the
zero-delay result is a clear reason to use a latency-aware selector rather than
replace `latest_only` unconditionally.

### Current Async/RTC matrix

The compact matrix contains 105 learned-policy episodes over five delay
profiles and three paired states per cell:

- X-VLA: upstream LeRobot `weighted_average` completed 10/15 episodes,
  `latest_only` 15/15, and ActionStream 15/15. At 950 ms, ActionStream used
  36.3 fewer steps than `latest_only` (95% CI -37 to -36), but held the last
  action more often. X-VLA reports RTC as not applicable because this
  checkpoint does not support the RTC policy signature.
- SmolVLA: upstream RTC completed 3/3 at zero delay, 1/3 at 250 ms, and 0/3 at
  500 ms, 950 ms, and jitter under this harness's post-inference transport
  delay. This is policy- and fault-placement-specific, not a general RTC
  ranking.
- Pi0.5 is unavailable because its official processor requires access to the
  gated PaliGemma tokenizer. It is recorded as unavailable, not as zero.

Detailed tables and raw traces remain in
[`outputs/current_lerobot_async_rtc`](../outputs/current_lerobot_async_rtc/) and
[`outputs/xvla_task_seed_expansion_20260816`](../outputs/xvla_task_seed_expansion_20260816/).

## Failure classification

| Failure | Scope | Classification |
|---|---|---|
| Native goal task 5 does not push plate into region | One native planar-contact canary | Bridge/native contact and robot-limit failure after canonical collision reconstruction; preserved as failed evidence |
| ActionStream loses 2/30 fixed-950 X-VLA trials | Official three-task expansion | High-delay task heterogeneity and hold tradeoff |
| ActionStream is 3.6 steps slower at zero delay | Official three-task expansion | Low-delay regression; aligned should not be unconditional |
| SmolVLA RTC fails all 500/950/jitter trials | Compact official matrix | RTC is not robust to this post-inference delay placement; not a universal RTC claim |
| Native async/RTC paired matrix absent | Native Isaac | Not run; official LIBERO baselines must not be presented as native Isaac results |

## Public-readiness audit

Audit snapshot on 2026-08-17:

- GitHub repository `Yanagisawa2002/ActionStream` is **PRIVATE**; default branch
  is `master`; the working branch was 47 commits ahead and 0 behind before this
  report commit.
- Tracked content and all Git history returned zero matches for common GitHub,
  Hugging Face, OpenAI, and AWS token prefixes; private-key headers; and the
  temporary SeetaCloud endpoint. This is a targeted pattern audit, not a
  substitute for an organization-grade secret scanner.
- `THIRD_PARTY_NOTICES.md` records LeRobot, LIBERO, and X-VLA provenance and
  states that weights/datasets are not redistributed.
- The repository has **no first-party `LICENSE`**. Public release and merge to
  `master` remain blocked until the owner chooses a license compatible with
  the intended source and asset distribution.
- No files are managed by Git LFS. Five currently tracked files exceed 10 MiB;
  the largest is a 44,012,541-byte JSONL trace. Loose Git objects occupy about
  322.94 MiB. This is below GitHub's single-file hard limit but is poor public
  clone hygiene and should be reduced in a release branch.

The release-safe action taken here is therefore to prepare the README, hero,
report, and local demo on the review branch while keeping the repository
private and leaving `master` untouched.

## Presentation artifacts

- Tracked README hero:
  [`docs/assets/actionstream_hero.png`](assets/actionstream_hero.png)
- Local 75.45-second paired demo:
  `artifacts/release/actionstream_learned_policy_demo_20260817.mp4`
  - H.264, 1280x640, 20 fps, 1,509 frames
  - SHA-256
    `5f5ec006178ab4253abf363f0fcb9f13fade4414edd324347237c73a7e71b0d3`
  - decoded end to end with FFmpeg and visually checked through a contact sheet
- Local native evidence archive:
  `artifacts/m8/learned_isaac_closure_20260817/actionstream_learned_isaac_closure_20260817.tgz`
  - SHA-256
    `ef7182a9c3c093b6a4bb4606f4b2e8cc7ff15b3176181c08d170dc2549ed3972`

The MP4 and raw archive remain outside normal Git and are covered by targeted
ignore rules. They should be attached to a future GitHub release or external
portfolio only after the license decision.

## Isaac Lab-Arena status

**Not done.** The current repository has no Isaac Lab-Arena dependency,
adapter, task registry, run receipt, or Arena result. The current bridge is a
custom native Isaac scene plus pinned official LIBERO render/state evaluation.
Calling it an Arena benchmark would be inaccurate.

The next industry-relevant extension is to port the frozen task/provenance
contract to Arena's task registry and GPU-parallel runner, then repeat paired
runtime evaluation on disjoint tasks, resets, and network traces. This should
follow the license/release cleanup and should not replace the still-missing
two-task real-robot paired A/B.

## Completion ledger

| Requested item | Status |
|---|---|
| Dynamic robot/object state bridge | Completed for the native development canaries |
| X-VLA sync success on 2–3 different task families | Partial pass: 2/3 development successes, one with official predicate |
| LeRobot Async, `latest_only`, RTC, ActionStream comparison | Completed in official LeRobot/LIBERO evidence; not repeated natively |
| README hero and 60–90 second demo | Completed and validated |
| Credential/provenance audit | Targeted audit completed; no credential match |
| Public release / merge | Blocked by missing first-party license and release-history cleanup |
| Minimal real-robot paired A/B | Not done |
| Isaac Lab-Arena integration | Not done |
