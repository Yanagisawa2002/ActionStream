# ActionStream-Adaptive v2 frozen protocol

Frozen at `2026-08-17T11:20:00Z`. Selector file:
`configs/actionstream_adaptive_v2_selector.json`, SHA-256
`8679606e12b85accc26bd1ad1a1d710bade6682c738cebc031a4196d4a367572`.

## Why v2 exists

The v1 development canary is retained as negative evidence. On LIBERO Spatial
task 3/state 13, X-VLA sync succeeded in 81 steps, but v1 aborted before the
first action. Its global object-suite bound rejected target `z=1.183 m` even
though the request observation reported robot EEF `z=1.176 m`. Formal holdout
had not been opened. V2 changes only the coordinate contract of the risk gate;
it does not change the frozen age thresholds or select a different canary or
holdout task after observing an outcome.

## Runtime decision

The selector may read delivered-result age, queue depth, the incoming action
chunk, the request-time robot EEF pose, and the last executed action. It may not
read a delay-profile key, task success, future observation, or any holdout
result.

| Delivered result age | Preferred execution |
|---:|---|
| 0--5 steps | Official LeRobot latest-only branch |
| 6--20 steps | ActionStream age-aligned branch |
| 21--28 steps | Official LeRobot latest-only branch |
| More than 28 steps | Hold the last safe action and request a fresh chunk |

The first chunk is checked against the request-time robot EEF position and
orientation. Later chunks are checked against the last command actually sent.
This makes the 0.25 m translation and 0.35 rad SO(3) gates frame-local across
LIBERO task suites. Missing first-chunk EEF state still aborts; no command is
invented. Every decision records its reference source and reference pose.

## Development canary and sealed holdout

The canary remains one reset from each genuinely different suite:

- Object task 3/state 13;
- Spatial task 3/state 13;
- Goal task 1/state 13;
- each under zero delay, seeded 500+/-250 ms jitter, and fixed 950 ms.

Only after all three canary suites execute with replayable provenance may the
v2 holdout manifests be opened. The holdout remains disjoint and unchanged
from v1: Object task 4, Spatial task 6, Goal task 0; states 20--24; independent
seeds; zero, three jitter traces, 950 ms, and burst/outage. Runtimes are pinned
LeRobot latest-only, static ActionStream aligned, and Adaptive v2. Thresholds
must not change after the first formal row is read.

## Claims and real-robot boundary

V2 may support a learned-policy, frame-local, latency/risk-aware runtime claim.
It is not a learned selector, an RTC implementation, or a safety
certification. A real-robot result requires a separate paired manifest, named
robot/driver, operator and E-stop receipt, calibrated limits, paired resets,
real network traces, video, and an immutable episode ledger. Simulation cannot
be relabeled as the minimum real-robot A/B.
