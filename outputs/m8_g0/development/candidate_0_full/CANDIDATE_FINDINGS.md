# M8 candidate 0: full native development findings

Status: **strong positive development result**, replay-valid and recorded in the
open calibration ledger. This is not frozen holdout or headline evidence.

## Result

The candidate contains 12 same-reset paired seeds under each of the two
preregistered asynchronous profiles, for 48 native Isaac Sim episodes. Each
profile compares the same deterministic observation-conditioned policy under
`naive_async` and `aligned_async` queue semantics.

| Profile | Naive success | Aligned success | Aligned - naive | Paired bootstrap 95% CI | Exact McNemar |
|---|---:|---:|---:|---:|---:|
| Profile 1: fixed 850 ms | 0/12 | 10/12 | +83.3 pp | [+58.3, +100.0] pp | p=0.001953 |
| Profile 2: 850 ms + jitter/faults | 0/12 | 8/12 | +66.7 pp | [+41.7, +91.7] pp | p=0.007812 |

The confidence intervals use the preregistered 20,000-resample paired
nonparametric bootstrap at the paired-seed level. The exact two-sided McNemar
tests use only discordant task-success pairs. These statistics describe
candidate development and must not be presented as confirmatory holdout tests.

| Profile / metric | Naive async | Aligned async |
|---|---:|---:|
| Profile 1 grasp / recovery | 0/12 / 0/12 | 12/12 / 12/12 |
| Profile 1 expired executed | 846 | 0 |
| Profile 1 mean action age | 40.433 steps | 17.427 steps |
| Profile 1 mean max target delta | 0.2294 m | 0.1349 m |
| Profile 2 grasp / recovery | 0/12 / 0/12 | 11/12 / 11/12 |
| Profile 2 expired executed | 842 | 0 |
| Profile 2 mean action age | 41.767 steps | 18.747 steps |
| Profile 2 mean max target delta | 0.2238 m | 0.1697 m |

Across the 24 profile/seed blocks, a descriptive-only pool is 0/24 versus
18/24 task success, 1,688 versus 0 expired actions executed, a 56.0% reduction
in mean action age, and a 32.8% reduction in mean maximum applied-target
discontinuity. No pooled confidence interval is reported because both profiles
reuse the same 12 seeds.

Profile 1 aligned failures were seeds `2026081100` and `2026081108`, both at
the joint/workspace guard. Profile 2 aligned failures were `2026081100`,
`2026081107`, and `2026081110` at that guard, plus `2026081105` where the
switch precondition was missed. Naive failed every full task.

## Evidence integrity

- Independent replay passed 48/48 event logs.
- All 24 paired-reset blocks passed numeric and canonical fairness checks.
- Seed, profile, fault-trace, current-source, and native-provenance checks
  passed.
- Matrix SHA-256:
  `deb16f17fa7d8a874f318cd14ae5a768664bbd4201325f84c4f94594c732e4b2`.
- Replay SHA-256:
  `77c91b14a704eb9de1c845ec8f5be64d9dd3a31f556e5e059590fcb6c0d3b14c`.
- The baseline and both candidate profile runs bind the same 101-file source
  manifest:
  `1648d72d2c0a539a4f5f64a4ff870c5428d491b0242c68feec3905034749f945`.
- The source-matched Profile-0 baseline passed 20/20 before candidate
  registration.
- `candidate_0` is immutably recorded in
  [the v2 calibration ledger](../../../../configs/m8_calibration_ledger_v2.json),
  which remains `development_open`; no candidate selection or holdout freeze
  has been performed.

## Scope boundary

This is a genuine policy-driven native Isaac Sim comparison, not a trace
replay: the shared Cartesian policy consumes live end-effector/object poses,
grasp state, task phase, active destination, and generation. It is still a
deterministic controller rather than a learned VLA, and it is not real-robot
evidence.

Evidence: [machine-readable analysis](candidate_analysis.json),
[matrix](matrix.json), and [independent replay](replay_validation.json).
