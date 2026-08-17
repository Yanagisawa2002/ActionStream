# ActionStream Adaptive Budgeted-Release v5 development protocol

Frozen at `2026-08-17T15:31:00Z`, before any v5 learned-policy episode was
run. Selector:
`configs/actionstream_adaptive_budgeted_release_v5_selector.json`, SHA-256
`51b3dbd120a0a39183a21f68e10d801151d255946c9aa7b8821f022ccf425a1b`.

## Evidence boundary

Queue-Slack v4 remains an immutable `NO-GO` canary. Its state 32, network
traces, videos, parameters, and verdict are not development input for v5 beyond
the already recorded diagnosis: the Object/high-jitter row timed out after 25
reserve activations and 83 reserve-hold steps. V5 inherits every v4 age, risk,
phase, service-EWMA, prefetch, and five-command-reserve constant unchanged. It
changes only the release policy described below.

No v5 formal holdout is registered. Any change to the selector, constants,
tasks, state, environment seed, network trace, success rule, or 300-step limit
after the first canary row is read requires a new selector ID and a new split.

## Method registered before learned-policy execution

V5 replaces v4's indefinite five-command freeze with bounded spending:

1. A reserve activation starts when queue depth is in `[1, 5]` and a submitted
   inference request remains unresolved.
2. At activation and after each spend, set a release deadline to
   `current_step + clip(predicted_service_steps - oldest_request_age, 1, 4)`.
3. Before that deadline, release one already validated queued command when the
   measured robot EEF moves less than 2 mm and rotates less than 0.02 rad over
   three executed control steps. This is a runtime motion signal, not task-goal
   progress.
4. At the deadline, release one queued command even if EEF motion continues.
5. Spend at most four commands per activation and never consume the final
   protected command. Otherwise repeat the last finite absolute command until
   a result arrives.
6. An accepted replenishing result or the absence of an unresolved request
   closes the activation and resets its four-command budget.

The runtime may not inspect a profile key, task/suite identity, object/goal
pose, success, future observation, or canary result. It records per-step EEF
progress, release deadline, spend count and reason, plus aggregate deadline,
progress, budget-exhaustion, protected-floor, prefetch, preservation, and guard
counters.

## Development-only smoke and mechanism probe

The structural smoke/probe manifest is
`configs/actionstream_adaptive_budgeted_release_v5_dev_probe.json`, SHA-256
`6821e50670c2e4b9b530a9a3f2fb4b85b67708aa02387ddd65e0cde5735df52a`.
It consumes state 34 for low-jitter smoke and state 35 for a separate outage
probe. Its trace hashes are:

- smoke low jitter: `54ca819f192c3214d5d55d5d38292161fe66f4f9fe2cff49ffebf38533d1e2dc`;
- probe outage at ordinals 1/4:
  `9a031e73235eb776b15fefe659c7c88a44a6f6af1e31f02165a5754788b8e1d4`.

These rows may identify implementation defects but may not tune selector
constants. They do not consume state 33 or any formal-canary trace.

## Frozen disjoint canary

Repository and result-receipt searches before freezing found no prior use of
states 33, 34, or 35. The formal canary uses only state 33:

| Family | Suite/task | State | Environment seed | Low/high trace seeds |
|---|---|---:|---:|---:|
| Object | `libero_object` task 5 | 33 | 2026081801 | 2026081811 / 2026081821 |
| Spatial | `libero_spatial` task 7 | 33 | 2026081802 | 2026081812 / 2026081822 |
| Goal | `libero_goal` task 2 | 33 | 2026081803 | 2026081813 / 2026081823 |

Each family compares sync capability, official LeRobot `latest_only`, static
ActionStream aligned, and v5 Adaptive on the same reset and trace. Profiles:

- low jitter: seeded uniform 25--125 ms;
- high jitter: seeded uniform 600--1100 ms;
- outage: 425 ms base with 1850 ms delays at request ordinals 3 and 8.

Trace SHA-256 values:

| Family | Low jitter | High jitter | Outage 3/8 |
|---|---|---|---|
| Object | `30ea38b30d377e5d339da7dc93fb23babd3aab55bcf5f55f0aa18d1122ad8df8` | `a20e94e5e344c450b601f4f7d127d252bd1bdb0e92c04484e05c650c515cd77f` | `8379a0321797b4a2871bd0816cac6150665746fa6418625ec41517cb75ee9a57` |
| Spatial | `7a4c95de1e852676c61741c80e49b71e3e08a0349c1ced83aa7fc7a5d7618d7e` | `ba5d19176f7c0e2bde1e6ece67d394d928f528bd97786e6f71a189c6a2b74c29` | `8379a0321797b4a2871bd0816cac6150665746fa6418625ec41517cb75ee9a57` |
| Goal | `b86f71c28a8680ae4ab1120c444f7b33b59e6a10d47f4e4156f3c21f33c797a6` | `55cc6e5b15c85491e9f0b4248683ea48075b4aa7368ef607ae4066dad9ffe451` | `8379a0321797b4a2871bd0816cac6150665746fa6418625ec41517cb75ee9a57` |

Manifest SHA-256 values are Object
`a43923efc4f475dfd9e7474d08d4af97b8ef3066e25bbcb94735aa3d90140e5b`,
Spatial
`a4317434d5ac10593d7dfb7861a3489bc7b7f433a2c8fce05030aa916959b00e`,
and Goal
`8592aa9a92d6c01e9126574d858f8e41b81ffb9052012964f4643972ac010862`.

## Predeclared verdict gate

The exact candidate is `GO` only if all conditions hold:

- sync succeeds on all three low-jitter task/state pairs;
- Adaptive succeeds on all nine task/profile pairs, including Object/high
  jitter, with no success regression against a successful latest-only pair;
- every Adaptive row records prefetch and measured EEF progress observations;
- every Adaptive high-jitter and outage row activates reserve spending and
  releases at least one validated command by a registered deadline or progress
  trigger;
- every Adaptive outage row observes both frozen 1850 ms ordinals, preserves a
  nonempty queue through at least one rejected result, and records zero guard
  discards;
- action traces prove at most four spends per activation and at least one
  protected queued command whenever reserve is active with an unresolved
  request;
- all 36 rows complete without runtime exceptions, all traces are finite and
  hash-valid, and every requested video decodes.

Completion steps are diagnostic because there is one reset per
family/profile. Passing permits registration of a new multi-state holdout; it
is not itself an IID effect size, 95% CI, or generalization claim.

## Baseline and hardware boundary

`latest_only` is pinned upstream LeRobot and aligned is the existing
ActionStream target-step queue. X-VLA reports no RTC support, so RTC is not
reimplemented or relabeled. This is learned-policy LIBERO simulation, not
native Isaac, Isaac Lab-Arena, or real-robot evidence.
