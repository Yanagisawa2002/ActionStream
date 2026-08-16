# M8 frozen-holdout findings

## Decision

Official registered classification: **GO** (primary GO=true, strong GO=false).

## Raw main table

| Network profile | Method | Task success (Wilson 95% CI) | Mean of episode-mean latency (ms) | Median episode p50-p95 (ms) | Grasp | Recovery | Expired executed |
|---|---|---:|---:|---:|---:|---:|---:|
| P0 sanity | Sync hold | 60/60 (100.0%) [94.0%, 100.0%] | 15.7 | 15.9-16.4 | 60/60 | 60/60 | 0 |
| P1 fixed 850 ms | Sync hold | 1/60 (1.7%) [0.3%, 8.9%] | 881.8 | 877.9-898.5 | 60/60 | 60/60 | 0 |
| P1 fixed 850 ms | Naive async | 0/60 (0.0%) [0.0%, 6.0%] | 905.7 | 895.5-907.7 | 0/60 | 0/60 | 4244 |
| P1 fixed 850 ms | ActionStream aligned | 49/60 (81.7%) [70.1%, 89.4%] | 893.8 | 894.5-902.8 | 60/60 | 60/60 | 0 |
| P2 850 ms + jitter/faults | Sync hold | 15/60 (25.0%) [15.8%, 37.2%] | 1008.9 | 925.7-1379.9 | 45/60 | 44/60 | 0 |
| P2 850 ms + jitter/faults | Naive async | 0/60 (0.0%) [0.0%, 6.0%] | 986.7 | 918.8-1454.5 | 0/60 | 0/60 | 4231 |
| P2 850 ms + jitter/faults | ActionStream aligned | 52/60 (86.7%) [75.8%, 93.1%] | 975.9 | 912.8-1654.9 | 60/60 | 60/60 | 0 |

## Key findings

1. **Observation:** P1 aligned success is 49/60 (81.7%) versus 0/60 naive, a 81.7 pp paired gain with 95% CI [71.7, 90.0] pp. **Interpretation:** execution-aligned queue semantics survive the fixed remote-inference delay. **Implication:** the development result replicates within this scripted-policy holdout over 60 unseen seeds. **Next:** repeat with a learned policy and current async/RTC baselines.
2. **Observation:** P2 aligned success is 52/60 (86.7%) versus 0/60 naive, a 86.7 pp gain with 95% CI [76.7, 95.0] pp. **Interpretation:** the gain remains under jitter, drops, duplicate responses, extra delay, and pauses. **Implication:** this is a runtime result robust within the registered P2 fault profile, not only a fixed-delay artifact. **Next:** sweep preregistered latency levels rather than infer a continuous curve from three operating points.
3. **Observation:** aligned executed 0 expired actions versus 8475 for naive, with zero aligned collisions and timeouts. **Interpretation:** the result is consistent with the intended stale-action safety semantics. **Implication:** the success gain is not purchased by forbidden execution. **Next:** add a risk-aware selector using queue age, disagreement, and safety margin.
4. **Observation:** STRONG GO is false because aligned is not at least 15% faster than sync in wall clock and the obsolete-step reduction condition is not met. **Interpretation:** naive often fails before reaching the post-switch phase, making that secondary comparison structurally weak. **Implication:** claim GO, not STRONG GO. **Next:** keep the registered classification and introduce a better failure-aware secondary metric only in a new preregistered benchmark.

## Failure classification

| Network profile | Method | Failure category | Completion reason | Count | % of trials |
|---|---|---|---|---:|---:|
| P1 fixed 850 ms | Sync hold | safety/workspace limit | `joint_or_workspace_limit` | 4 | 6.7% |
| P1 fixed 850 ms | Sync hold | grasp instability | `unstable_grasp` | 55 | 91.7% |
| P1 fixed 850 ms | Naive async | pre-grasp control | `failed_approach` | 60 | 100.0% |
| P1 fixed 850 ms | ActionStream aligned | safety/workspace limit | `joint_or_workspace_limit` | 11 | 18.3% |
| P2 850 ms + jitter/faults | Sync hold | switch recovery timeout | `destination_switch_recovery_timeout` | 3 | 5.0% |
| P2 850 ms + jitter/faults | Sync hold | pre-grasp control | `failed_approach` | 11 | 18.3% |
| P2 850 ms + jitter/faults | Sync hold | grasp acquisition miss | `grasp_miss` | 2 | 3.3% |
| P2 850 ms + jitter/faults | Sync hold | safety/workspace limit | `joint_or_workspace_limit` | 3 | 5.0% |
| P2 850 ms + jitter/faults | Sync hold | switch timing/precondition | `switch_precondition_missed` | 2 | 3.3% |
| P2 850 ms + jitter/faults | Sync hold | grasp instability | `unstable_grasp` | 24 | 40.0% |
| P2 850 ms + jitter/faults | Naive async | pre-grasp control | `failed_approach` | 55 | 91.7% |
| P2 850 ms + jitter/faults | Naive async | switch timing/precondition | `switch_precondition_missed` | 5 | 8.3% |
| P2 850 ms + jitter/faults | ActionStream aligned | safety/workspace limit | `joint_or_workspace_limit` | 6 | 10.0% |
| P2 850 ms + jitter/faults | ActionStream aligned | grasp instability | `unstable_grasp` | 2 | 3.3% |

## Interpretation boundary

This is native Isaac Sim evidence for one deterministic observation-conditioned policy and one task family over 60 unseen scenario/initial-state seeds and 180 unseen network traces. It is not a learned VLA result, a task-family holdout, or real-robot evidence. The latency figure is an operating-point plot, not a causal continuous latency sweep.
The latency mean averages per-episode mean inference latency. The p50-p95 range reports the median episode-level p50 and p95, not a confidence interval.
