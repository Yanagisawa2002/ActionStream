# ActionStream-Adaptive v1 frozen protocol

Status: **frozen before new canary or holdout results**

The v1 method is a deterministic, inspectable runtime selector. It converts the
existing crossover result into one deployable decision layer instead of adding
another static runtime comparison.

## Problem anchor

The development matrix shows mutually incompatible operating points for a
single static rule. ActionStream aligned is slower than latest-only at zero
delay, materially faster under 500 +/- 250 ms jitter, and faster but less
successful at fixed 950 ms. The method therefore targets regime selection, not
a universal aligned-runtime win.

## Frozen decision

For every delivered policy chunk, v1 measures its observation age in control
steps and evaluates both the first full-chunk command and the execution-aligned
command against the last dispatched safe action.

| Delivered result age | Preferred execution |
|---:|---|
| 0--5 steps | Full chunk |
| 6--20 steps | Drop the age-matched stale prefix |
| 21--28 steps | Full chunk so the queue does not collapse below ten actions |
| More than 28 steps | Hold the last safe action and request a fresh chunk |

The hard risk gate precedes the preference table. Non-finite actions, commands
outside the frozen absolute workspace, or excessive translation/rotation
disagreement are not executed. If the preferred candidate fails but the other
candidate passes, v1 uses the safe alternate and records the override. If both
fail, it holds the last safe command. An unsafe first chunk aborts instead of
inventing a command.

This first version deliberately does not switch into RTC. RTC-capable policies
change the inference signature and upstream queue merge together; RTC remains
an official external baseline until a later selector freezes that additional
state transition.

## Data boundary

Thresholds use only the existing `libero_object` tasks 0--2 development
matrix. The exact inputs and hashes are frozen in
`configs/actionstream_adaptive_v1_selector.json`.

New development canaries use task/state pairs that were absent from that
matrix:

| Suite | Canary task | State |
|---|---:|---:|
| `libero_object` | 3, BBQ sauce into basket | 13 |
| `libero_spatial` | 3, bowl from cookie box onto plate | 13 |
| `libero_goal` | 1, bowl onto stove | 13 |

Formal simulator holdout is frozen separately from the canaries:

| Suite | Holdout task | States |
|---|---:|---|
| `libero_object` | 4, ketchup into basket | 20--24 |
| `libero_spatial` | 6, bowl next to cookie box onto plate | 20--24 |
| `libero_goal` | 0, open middle drawer | 20--24 |

The holdout uses zero delay, fixed 950 ms, three independently seeded moderate
jitter traces, and one independently seeded burst/outage trace. Within a suite,
each runtime sees the same task, initial state, policy seed, checkpoint, and
network trace.

## Baselines and acceptance

The formal paired comparison is:

- pinned official LeRobot `latest_only`;
- static ActionStream aligned;
- ActionStream-Adaptive v1.

Success and safety dominate speed lexicographically. The frozen acceptance gate
is:

1. zero-delay success is no worse than latest-only on any of the three task
   families;
2. moderate-jitter success is no worse than the better static runtime and the
   pooled completion-step gain versus latest-only is at least 8%;
3. fixed-950 and burst/outage profiles introduce no additional failures versus
   latest-only;
4. every executed action passes the hard risk gate, every selector decision is
   replayable from the trace, and no profile key is supplied to the selector;
5. formal thresholds remain unchanged after the first holdout result is read.

Failure of a gate is retained as a negative method result. It is not repaired by
changing the frozen thresholds or swapping the holdout task.

## Minimal real-robot boundary

The real-robot A/B is a separate gate: two manipulation tasks, twenty paired
resets per task, matched initial-state fixtures, replayable network traces,
workspace/joint/force limits, an operator emergency stop, and synchronized
video. It compares the best frozen static runtime against Adaptive. Simulator
success does not authorize robot motion and is never reported as real-robot
evidence.
