# Current LeRobot Async/RTC paired results

Validated 180 completed episodes and 180 raw traces containing 25003 finite 7D actions. The statistical unit is the episode; task, initial state, episode index, and seed are paired within each model.

These results are paired within a policy only. They do not rank policy quality across X-VLA and SmolVLA.

## Raw condition table

| model | delay | runtime | n | success | steps mean±sd | hold mean±sd | delivery p50 mean±sd | discontinuity mean±sd |
|---|---|---|---:|---:|---:|---:|---:|---:|
| xvla | fixed_0000 | lerobot_latest_only | 30 | 1.000 | 130.200±10.839 | 0.000±0.000 | 0.147±0.013 | 0.143±0.090 |
| xvla | fixed_0000 | actionstream_aligned | 30 | 1.000 | 133.800±10.990 | 0.000±0.000 | 0.145±0.015 | 0.211±0.133 |
| xvla | fixed_0950 | lerobot_latest_only | 30 | 1.000 | 162.233±16.521 | 0.039±0.004 | 1.056±0.003 | 0.170±0.110 |
| xvla | fixed_0950 | actionstream_aligned | 30 | 0.933 | 137.567±40.175 | 0.185±0.027 | 1.100±0.020 | 0.159±0.100 |
| xvla | jitter_0500_pm0250 | lerobot_latest_only | 30 | 1.000 | 141.667±15.144 | 0.021±0.009 | 0.640±0.020 | 0.125±0.083 |
| xvla | jitter_0500_pm0250 | actionstream_aligned | 30 | 1.000 | 127.967±11.550 | 0.000±0.000 | 0.718±0.025 | 0.154±0.088 |

## Task-stratified raw condition table

| model | task | delay | runtime | n | success | steps mean±sd | hold mean±sd |
|---|---:|---|---|---:|---:|---:|---:|
| xvla | 0 | fixed_0000 | lerobot_latest_only | 10 | 1.000 | 143.800±5.432 | 0.000±0.000 |
| xvla | 0 | fixed_0000 | actionstream_aligned | 10 | 1.000 | 146.400±5.147 | 0.000±0.000 |
| xvla | 0 | fixed_0950 | lerobot_latest_only | 10 | 1.000 | 182.800±6.356 | 0.036±0.004 |
| xvla | 0 | fixed_0950 | actionstream_aligned | 10 | 1.000 | 139.800±4.872 | 0.190±0.021 |
| xvla | 0 | jitter_0500_pm0250 | lerobot_latest_only | 10 | 1.000 | 159.700±9.684 | 0.024±0.012 |
| xvla | 0 | jitter_0500_pm0250 | actionstream_aligned | 10 | 1.000 | 141.700±6.111 | 0.000±0.000 |
| xvla | 1 | fixed_0000 | lerobot_latest_only | 10 | 1.000 | 126.400±3.777 | 0.000±0.000 |
| xvla | 1 | fixed_0000 | actionstream_aligned | 10 | 1.000 | 132.300±5.458 | 0.000±0.000 |
| xvla | 1 | fixed_0950 | lerobot_latest_only | 10 | 1.000 | 156.600±3.204 | 0.039±0.002 |
| xvla | 1 | fixed_0950 | actionstream_aligned | 10 | 0.800 | 156.700±65.134 | 0.191±0.032 |
| xvla | 1 | jitter_0500_pm0250 | lerobot_latest_only | 10 | 1.000 | 136.600±5.739 | 0.015±0.002 |
| xvla | 1 | jitter_0500_pm0250 | actionstream_aligned | 10 | 1.000 | 125.700±4.191 | 0.000±0.000 |
| xvla | 2 | fixed_0000 | lerobot_latest_only | 10 | 1.000 | 120.400±2.547 | 0.000±0.000 |
| xvla | 2 | fixed_0000 | actionstream_aligned | 10 | 1.000 | 122.700±4.138 | 0.000±0.000 |
| xvla | 2 | fixed_0950 | lerobot_latest_only | 10 | 1.000 | 147.300±8.706 | 0.043±0.003 |
| xvla | 2 | fixed_0950 | actionstream_aligned | 10 | 1.000 | 116.200±3.882 | 0.174±0.028 |
| xvla | 2 | jitter_0500_pm0250 | lerobot_latest_only | 10 | 1.000 | 128.700±5.964 | 0.022±0.007 |
| xvla | 2 | jitter_0500_pm0250 | actionstream_aligned | 10 | 1.000 | 116.500±3.689 | 0.000±0.000 |

## Paired runtime effects

Differences are estimate minus reference. Intervals are descriptive; raw outcomes and task-stratified effects remain the primary evidence.

| model | delay | estimate - reference | n | success Δ [95% CI] | steps Δ [95% CI] | discontinuity Δ [95% CI] |
|---|---|---|---:|---:|---:|---:|
| xvla | fixed_0000 | actionstream_aligned - lerobot_latest_only | 30 | 0.000 [0.000, 0.000] | 3.600 [2.400, 4.900] | 0.068 [0.027, 0.111] |
| xvla | fixed_0950 | actionstream_aligned - lerobot_latest_only | 30 | -0.067 [-0.167, 0.000] | -24.667 [-36.433, -8.233] | -0.011 [-0.041, 0.020] |
| xvla | jitter_0500_pm0250 | actionstream_aligned - lerobot_latest_only | 30 | 0.000 [0.000, 0.000] | -13.700 [-15.600, -11.933] | 0.029 [0.005, 0.053] |

## Task-stratified paired runtime effects

| model | task | delay | estimate - reference | n | success Δ [95% CI] | steps Δ [95% CI] | hold Δ [95% CI] |
|---|---:|---|---|---:|---:|---:|---:|
| xvla | 0 | fixed_0000 | actionstream_aligned - lerobot_latest_only | 10 | 0.000 [0.000, 0.000] | 2.600 [1.000, 4.000] | 0.000 [0.000, 0.000] |
| xvla | 0 | fixed_0950 | actionstream_aligned - lerobot_latest_only | 10 | 0.000 [0.000, 0.000] | -43.000 [-47.200, -37.900] | 0.154 [0.143, 0.166] |
| xvla | 0 | jitter_0500_pm0250 | actionstream_aligned - lerobot_latest_only | 10 | 0.000 [0.000, 0.000] | -18.000 [-21.700, -14.500] | -0.024 [-0.031, -0.018] |
| xvla | 1 | fixed_0000 | actionstream_aligned - lerobot_latest_only | 10 | 0.000 [0.000, 0.000] | 5.900 [3.900, 8.500] | 0.000 [0.000, 0.000] |
| xvla | 1 | fixed_0950 | actionstream_aligned - lerobot_latest_only | 10 | -0.200 [-0.500, 0.000] | 0.100 [-31.500, 46.000] | 0.152 [0.134, 0.172] |
| xvla | 1 | jitter_0500_pm0250 | actionstream_aligned - lerobot_latest_only | 10 | 0.000 [0.000, 0.000] | -10.900 [-12.200, -9.700] | -0.015 [-0.017, -0.014] |
| xvla | 2 | fixed_0000 | actionstream_aligned - lerobot_latest_only | 10 | 0.000 [0.000, 0.000] | 2.300 [0.300, 3.900] | 0.000 [0.000, 0.000] |
| xvla | 2 | fixed_0950 | actionstream_aligned - lerobot_latest_only | 10 | 0.000 [0.000, 0.000] | -31.100 [-35.100, -27.500] | 0.131 [0.115, 0.148] |
| xvla | 2 | jitter_0500_pm0250 | actionstream_aligned - lerobot_latest_only | 10 | 0.000 [0.000, 0.000] | -12.200 [-14.600, -9.800] | -0.022 [-0.027, -0.019] |

## Evidence boundary

- Smoke episodes are excluded from this report.
- A 280-step horizon failure is a task outcome; a shorter intentional smoke truncation is not.
- Cross-model comparisons are not paired and are not used for method claims.
- Per-episode delivery p95 includes the first cold inference and depends on the number of inference calls, so the compact table uses p50; p95 remains in JSON/CSV.
- Native Isaac evidence is reported separately and requires a completed native runner receipt plus a live observation-conditioned policy.
