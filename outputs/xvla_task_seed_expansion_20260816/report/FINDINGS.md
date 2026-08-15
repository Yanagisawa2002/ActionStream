# Findings: X-VLA task and seed expansion

## Scope and evidence

- 180 completed policy-driven LIBERO Object episodes: three tasks, ten paired
  initial states/seeds, two runtimes, and three delay profiles.
- Each task contributes 60 episodes. All cells use the exact X-VLA revision
  `12e8783e996944f5c97e490d37d4c145484ed70a` and LeRobot commit
  `6adf51511b7625090eade8d82d9f61a1846ebe56`.
- The report verified 180 raw traces, 25,003 finite environment-ready 7D
  actions, and 18 hashed first-seed videos.
- FFmpeg 8.1.1 independently decoded every raw video and all six labelled
  paired composites without an error. Videos remain local and gitignored.

## Finding 1: the jitter gain replicates across all three tasks

Under seeded 500 +/- 250 ms delivery jitter, both runtimes succeeded in all
30/30 episodes. ActionStream aligned completed 13.7 steps sooner on average
than LeRobot `latest_only` (paired 95% bootstrap interval -15.6 to -11.9).
The task-specific effects were -18.0, -10.9, and -12.2 steps for tasks 0, 1,
and 2, respectively, and every interval excluded zero.

This is the clean positive result from the expansion: it is paired, retains
perfect success, and reproduces across three tasks rather than one task only.

## Finding 2: fixed 950 ms is a heterogeneous tradeoff

At fixed 950 ms, pooled completion was 24.7 steps earlier for aligned, but its
success rate was 28/30 versus 30/30 for `latest_only`. Tasks 0 and 2 retained
10/10 success and improved by 43.0 and 31.1 steps. Task 1 dropped to 8/10
success; its step difference was 0.1 with a wide interval (-31.5 to 46.0).
The two failed task-1 trials were seeds 2026081601 and 2026081602, both ending
at the 280-step horizon.

Aligned also held the last action more often at fixed 950 ms. The task-level
hold-fraction increases were 0.154, 0.152, and 0.131. The high-delay result is
therefore not a universal win and should not be reported without the task-1
failures and hold tradeoff.

## Finding 3: zero-delay remains a regression

Both runtimes succeeded in all 30 zero-delay episodes, but aligned required
3.6 more steps on average. The task-level regressions were 2.6, 5.9, and 2.3
steps, and all three paired intervals excluded zero. The method should be
described as useful in selected delayed regimes, not as an unconditional
runtime replacement.

## Content-visible evidence

Six local 1280x640 videos place the exact same-seed LeRobot `latest_only` run
on the left and ActionStream aligned run on the right for fixed 950 ms and
jitter on every task. For the captured seed 2026081600:

| task | delay | latest-only steps | aligned steps | outcome |
|---:|---|---:|---:|---|
| 0 | fixed 950 ms | 168 | 144 | both succeed |
| 0 | jitter | 151 | 135 | both succeed |
| 1 | fixed 950 ms | 153 | 122 | both succeed |
| 1 | jitter | 134 | 126 | both succeed |
| 2 | fixed 950 ms | 163 | 118 | both succeed |
| 2 | jitter | 134 | 118 | both succeed |

These videos demonstrate different closed-loop trajectories for one paired
seed per cell; the ten-seed tables, including the two task-1 failures, remain
the statistical evidence.

## Evidence boundary

- This is a within-policy runtime comparison, not a cross-policy ranking.
- Bootstrap intervals are descriptive and use the episode as the statistical
  unit.
- The native policy-driven Isaac paired run is a separate evidence class and
  requires its own completed Linux runner receipt, replay validation, and
  captured viewport videos.
