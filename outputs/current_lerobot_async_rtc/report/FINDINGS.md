# Findings: current LeRobot Async/RTC paired evaluation

## Scope and evidence

- 105 completed policy-driven LIBERO Object task-0 episodes.
- 35 within-policy runtime/delay cells, each with three paired initial
  states/seeds and a 280-step horizon.
- 105 raw traces, 19,512 finite environment-ready 7D actions, and 35 hashed,
  locally decoded first-pair videos.
- Fixed delivery delays: 0, 250, 500, and 950 ms. Jitter uses one immutable,
  seeded 500 +/- 250 ms trace reused across runtimes.
- Policy/runtime comparisons are paired within X-VLA or within SmolVLA. No
  cross-policy quality ranking is made.

## Finding 1: X-VLA has a high-delay crossover, not a universal win

**Observation.** X-VLA `latest_only` and ActionStream aligned both completed
15/15 episodes; upstream `weighted_average` completed 10/15. At zero delay,
aligned required a paired mean 10.3 more steps than `latest_only` (95% paired
bootstrap interval 6 to 17). At 950 ms, aligned required 36.3 fewer steps
(interval -37 to -36): aligned episodes finished in 139-151 steps versus
175-187 for `latest_only`. Under jitter, the paired mean was 9.7 fewer steps
(interval -19 to 0).

**Interpretation.** Target-step replacement is useful for this X-VLA task when
delivery age approaches the chunk/replan timescale, but current upstream
`latest_only` is the stronger low-delay baseline. The old claim cannot be
stated as a blanket runtime improvement.

**Implication.** The defensible result is a latency-regime crossover. The
strongest current official baseline must remain in every comparison.

**Cost.** At 950 ms, aligned held the last action for 17.5% of dispatches on
average, versus 3.3% for `latest_only`. The faster completion came with a
material hold tradeoff, not a free improvement.

## Finding 2: RTC is policy- and fault-model-dependent

**Observation.** SmolVLA RTC completed 3/3 zero-delay episodes in 143-152
steps, while aligned completed 2/3. With added delivery delay, RTC fell to 1/3
at 250 ms and 0/3 at 500 ms, 950 ms, and jitter. Across all 15 SmolVLA pairs,
success totals were 9/15 for `weighted_average`, 10/15 for `latest_only`,
10/15 for aligned, and 4/15 for RTC.

**Interpretation.** Current RTC is the best zero-delay SmolVLA cell here, but
it is not robust to the injected post-inference transport delay used by this
harness. This does not establish that RTC is generally weak; it establishes
that its advantage is not invariant to policy and delay placement.

**Implication.** “RTC-capable” is not equivalent to “RTC dominates.” Reports
must preserve the exact policy, delay injection point, and upstream queue/policy
path.

## Finding 3: the behavior difference is visible, not only numerical

**Observation.** In the captured same-seed X-VLA 950 ms pair, both runtimes
eventually place the target in the basket, but aligned finishes 36 control
steps earlier than `weighted_average`; across the three paired `latest_only`
episodes, aligned finishes a mean 36.3 steps earlier. In the captured SmolVLA
500 ms pair, aligned
finishes at step 137 while RTC continues to the 280-step horizon without task
success.

**Interpretation.** The quantitative step/success differences correspond to
different closed-loop trajectories in the simulator rather than report-only
queue counters.

**Implication.** The two local 1280x640 paired videos are suitable as concrete
content evidence, with the caveat that each is one representative seed rather
than a statistical sample.

## Negative and blocked results

- Pi0.5 weights load, but its official processor requires the gated
  `google/paligemma-3b-pt-224` tokenizer. No Hugging Face authorization was
  available, so Pi0.5 RTC results are unavailable rather than zero.
- The native Isaac runner did not launch. Three official workspace dependency
  attempts ended in prefix.dev connection resets; the last failed fetching
  `libcups-2.3.3` after three retries. The preserved failure receipt is before
  source validation/build/GPU launch, so no native metric or video exists.

## Limitations and next decision

- This is a compact decision matrix: one LIBERO task and three paired states.
  Bootstrap intervals are descriptive and coarse.
- Per-episode delivery p95 includes the first cold inference and changes with
  the number of inference calls. The main table therefore uses p50; raw p95 is
  retained in JSON and CSV.
- The next high-value run is task expansion on the selected X-VLA
  `latest_only`/aligned cells, followed by the unchanged policy-driven Isaac
  pair after prefix.dev or an approved mirror is stable.
