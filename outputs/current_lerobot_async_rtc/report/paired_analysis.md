# Current LeRobot Async/RTC paired results

Validated 105 completed episodes and 105 raw traces containing 19512 finite 7D actions. The statistical unit is the episode; task, initial state, episode index, and seed are paired within each model.

These results are paired within a policy only. They do not rank policy quality across X-VLA and SmolVLA.

## Raw condition table

| model | delay | runtime | n | success | steps mean±sd | hold mean±sd | delivery p50 mean±sd | discontinuity mean±sd |
|---|---|---|---:|---:|---:|---:|---:|---:|
| smolvla | fixed_0000 | lerobot_weighted_average | 3 | 1.000 | 149.667±22.855 | 0.000±0.000 | 0.270±0.004 | 0.107±0.038 |
| smolvla | fixed_0000 | lerobot_latest_only | 3 | 1.000 | 165.000±48.508 | 0.000±0.000 | 0.247±0.010 | 0.111±0.041 |
| smolvla | fixed_0000 | actionstream_aligned | 3 | 0.667 | 182.333±84.595 | 0.000±0.000 | 0.243±0.008 | 0.124±0.062 |
| smolvla | fixed_0000 | lerobot_rtc | 3 | 1.000 | 148.333±4.726 | 0.000±0.000 | 0.319±0.016 | 0.096±0.013 |
| smolvla | fixed_0250 | lerobot_weighted_average | 3 | 0.333 | 233.000±81.406 | 0.000±0.000 | 0.514±0.038 | 0.144±0.054 |
| smolvla | fixed_0250 | lerobot_latest_only | 3 | 0.333 | 232.333±82.561 | 0.000±0.000 | 0.525±0.028 | 0.161±0.059 |
| smolvla | fixed_0250 | actionstream_aligned | 3 | 0.667 | 185.667±81.709 | 0.000±0.000 | 0.510±0.016 | 0.129±0.060 |
| smolvla | fixed_0250 | lerobot_rtc | 3 | 0.333 | 228.667±88.912 | 0.000±0.000 | 0.577±0.004 | 0.151±0.061 |
| smolvla | fixed_0500 | lerobot_weighted_average | 3 | 0.333 | 228.667±88.912 | 0.000±0.000 | 0.758±0.015 | 0.109±0.078 |
| smolvla | fixed_0500 | lerobot_latest_only | 3 | 0.667 | 182.000±85.159 | 0.000±0.000 | 0.743±0.042 | 0.124±0.063 |
| smolvla | fixed_0500 | actionstream_aligned | 3 | 0.667 | 182.333±84.654 | 0.000±0.000 | 0.782±0.026 | 0.161±0.122 |
| smolvla | fixed_0500 | lerobot_rtc | 3 | 0.000 | 280.000±0.000 | 0.000±0.000 | 0.843±0.039 | 0.179±0.024 |
| smolvla | fixed_0950 | lerobot_weighted_average | 3 | 0.667 | 177.667±88.659 | 0.001±0.002 | 1.179±0.017 | 0.115±0.060 |
| smolvla | fixed_0950 | lerobot_latest_only | 3 | 0.667 | 174.000±91.848 | 0.000±0.000 | 1.177±0.018 | 0.131±0.085 |
| smolvla | fixed_0950 | actionstream_aligned | 3 | 0.667 | 178.000±88.357 | 0.000±0.000 | 1.203±0.054 | 0.145±0.111 |
| smolvla | fixed_0950 | lerobot_rtc | 3 | 0.000 | 280.000±0.000 | 0.005±0.005 | 1.261±0.012 | 0.217±0.184 |
| smolvla | jitter_0500_pm0250 | lerobot_weighted_average | 3 | 0.667 | 176.667±89.540 | 0.000±0.000 | 0.794±0.032 | 0.106±0.045 |
| smolvla | jitter_0500_pm0250 | lerobot_latest_only | 3 | 0.667 | 177.000±89.314 | 0.000±0.000 | 0.750±0.013 | 0.120±0.065 |
| smolvla | jitter_0500_pm0250 | actionstream_aligned | 3 | 0.667 | 178.333±88.059 | 0.000±0.000 | 0.816±0.040 | 0.137±0.094 |
| smolvla | jitter_0500_pm0250 | lerobot_rtc | 3 | 0.000 | 280.000±0.000 | 0.000±0.000 | 0.836±0.020 | 0.206±0.015 |
| xvla | fixed_0000 | lerobot_weighted_average | 3 | 0.667 | 193.000±75.425 | 0.000±0.000 | 0.144±0.009 | 0.130±0.082 |
| xvla | fixed_0000 | lerobot_latest_only | 3 | 1.000 | 144.333±4.041 | 0.000±0.000 | 0.143±0.023 | 0.149±0.107 |
| xvla | fixed_0000 | actionstream_aligned | 3 | 1.000 | 154.667±9.074 | 0.000±0.000 | 0.125±0.003 | 0.169±0.113 |
| xvla | fixed_0250 | lerobot_weighted_average | 3 | 0.667 | 186.000±81.707 | 0.000±0.000 | 0.378±0.011 | 0.158±0.113 |
| xvla | fixed_0250 | lerobot_latest_only | 3 | 1.000 | 147.667±5.774 | 0.000±0.000 | 0.361±0.007 | 0.138±0.086 |
| xvla | fixed_0250 | actionstream_aligned | 3 | 1.000 | 146.333±8.083 | 0.000±0.000 | 0.397±0.009 | 0.117±0.076 |
| xvla | fixed_0500 | lerobot_weighted_average | 3 | 0.667 | 193.333±75.381 | 0.000±0.000 | 0.618±0.016 | 0.149±0.098 |
| xvla | fixed_0500 | lerobot_latest_only | 3 | 1.000 | 139.000±5.568 | 0.000±0.000 | 0.612±0.006 | 0.177±0.128 |
| xvla | fixed_0500 | actionstream_aligned | 3 | 1.000 | 143.000±1.732 | 0.000±0.000 | 0.640±0.024 | 0.151±0.105 |
| xvla | fixed_0950 | lerobot_weighted_average | 3 | 0.667 | 221.333±52.444 | 0.030±0.007 | 1.056±0.003 | 0.207±0.184 |
| xvla | fixed_0950 | lerobot_latest_only | 3 | 1.000 | 180.667±6.028 | 0.033±0.001 | 1.056±0.003 | 0.156±0.085 |
| xvla | fixed_0950 | actionstream_aligned | 3 | 1.000 | 144.333±6.110 | 0.175±0.044 | 1.089±0.032 | 0.108±0.088 |
| xvla | jitter_0500_pm0250 | lerobot_weighted_average | 3 | 0.667 | 199.667±69.659 | 0.016±0.005 | 0.645±0.030 | 0.130±0.130 |
| xvla | jitter_0500_pm0250 | lerobot_latest_only | 3 | 1.000 | 150.333±9.452 | 0.015±0.003 | 0.637±0.011 | 0.084±0.066 |
| xvla | jitter_0500_pm0250 | actionstream_aligned | 3 | 1.000 | 140.667±3.215 | 0.000±0.000 | 0.700±0.021 | 0.155±0.108 |

## Paired runtime effects

Differences are estimate minus reference. With three pairs, intervals are descriptive and coarse; raw outcomes remain the primary evidence.

| model | delay | estimate - reference | n | success Δ [95% CI] | steps Δ [95% CI] | discontinuity Δ [95% CI] |
|---|---|---|---:|---:|---:|---:|
| smolvla | fixed_0000 | actionstream_aligned - lerobot_weighted_average | 3 | -0.333 [-1.000, 0.000] | 32.667 [-3.000, 104.000] | 0.018 [0.003, 0.045] |
| smolvla | fixed_0000 | actionstream_aligned - lerobot_latest_only | 3 | -0.333 [-1.000, 0.000] | 17.333 [-4.000, 59.000] | 0.014 [0.001, 0.038] |
| smolvla | fixed_0000 | lerobot_rtc - actionstream_aligned | 3 | 0.333 [0.000, 1.000] | -34.000 [-137.000, 20.000] | -0.028 [-0.115, 0.018] |
| smolvla | fixed_0000 | lerobot_rtc - lerobot_weighted_average | 3 | 0.000 [0.000, 0.000] | -1.333 [-33.000, 17.000] | -0.011 [-0.070, 0.021] |
| smolvla | fixed_0000 | lerobot_rtc - lerobot_latest_only | 3 | 0.000 [0.000, 0.000] | -16.667 [-78.000, 16.000] | -0.015 [-0.076, 0.019] |
| smolvla | fixed_0250 | actionstream_aligned - lerobot_weighted_average | 3 | 0.333 [0.000, 1.000] | -47.333 [-140.000, 0.000] | -0.015 [-0.049, 0.004] |
| smolvla | fixed_0250 | actionstream_aligned - lerobot_latest_only | 3 | 0.333 [0.000, 1.000] | -46.667 [-140.000, 0.000] | -0.032 [-0.103, 0.007] |
| smolvla | fixed_0250 | lerobot_rtc - actionstream_aligned | 3 | -0.333 [-1.000, 0.000] | 43.000 [-14.000, 143.000] | 0.022 [-0.037, 0.113] |
| smolvla | fixed_0250 | lerobot_rtc - lerobot_weighted_average | 3 | 0.000 [-1.000, 1.000] | -4.333 [-154.000, 141.000] | 0.008 [-0.058, 0.117] |
| smolvla | fixed_0250 | lerobot_rtc - lerobot_latest_only | 3 | 0.000 [-1.000, 1.000] | -3.667 [-154.000, 143.000] | -0.009 [-0.112, 0.115] |
| smolvla | fixed_0500 | actionstream_aligned - lerobot_weighted_average | 3 | 0.333 [0.000, 1.000] | -46.333 [-150.000, 11.000] | 0.052 [0.014, 0.104] |
| smolvla | fixed_0500 | actionstream_aligned - lerobot_latest_only | 3 | 0.000 [0.000, 0.000] | 0.333 [-10.000, 11.000] | 0.037 [-0.005, 0.105] |
| smolvla | fixed_0500 | lerobot_rtc - actionstream_aligned | 3 | -0.667 [-1.000, 0.000] | 97.667 [0.000, 150.000] | 0.018 [-0.112, 0.108] |
| smolvla | fixed_0500 | lerobot_rtc - lerobot_weighted_average | 3 | -0.333 [-1.000, 0.000] | 51.333 [0.000, 154.000] | 0.070 [-0.008, 0.147] |
| smolvla | fixed_0500 | lerobot_rtc - lerobot_latest_only | 3 | -0.667 [-1.000, 0.000] | 98.000 [0.000, 154.000] | 0.055 [-0.007, 0.103] |
| smolvla | fixed_0950 | actionstream_aligned - lerobot_weighted_average | 3 | 0.000 [0.000, 0.000] | 0.333 [0.000, 1.000] | 0.030 [0.000, 0.089] |
| smolvla | fixed_0950 | actionstream_aligned - lerobot_latest_only | 3 | 0.000 [0.000, 0.000] | 4.000 [0.000, 11.000] | 0.014 [-0.002, 0.044] |
| smolvla | fixed_0950 | lerobot_rtc - actionstream_aligned | 3 | -0.667 [-1.000, 0.000] | 102.000 [0.000, 155.000] | 0.072 [-0.099, 0.338] |
| smolvla | fixed_0950 | lerobot_rtc - lerobot_weighted_average | 3 | -0.667 [-1.000, 0.000] | 102.333 [0.000, 156.000] | 0.102 [-0.022, 0.339] |
| smolvla | fixed_0950 | lerobot_rtc - lerobot_latest_only | 3 | -0.667 [-1.000, 0.000] | 106.000 [0.000, 162.000] | 0.086 [-0.055, 0.336] |
| smolvla | jitter_0500_pm0250 | actionstream_aligned - lerobot_weighted_average | 3 | 0.000 [0.000, 0.000] | 1.667 [0.000, 4.000] | 0.031 [0.003, 0.087] |
| smolvla | jitter_0500_pm0250 | actionstream_aligned - lerobot_latest_only | 3 | 0.000 [0.000, 0.000] | 1.333 [-1.000, 5.000] | 0.017 [0.001, 0.050] |
| smolvla | jitter_0500_pm0250 | lerobot_rtc - actionstream_aligned | 3 | -0.667 [-1.000, 0.000] | 101.667 [0.000, 154.000] | 0.069 [-0.026, 0.129] |
| smolvla | jitter_0500_pm0250 | lerobot_rtc - lerobot_weighted_average | 3 | -0.667 [-1.000, 0.000] | 103.333 [0.000, 158.000] | 0.100 [0.061, 0.132] |
| smolvla | jitter_0500_pm0250 | lerobot_rtc - lerobot_latest_only | 3 | -0.667 [-1.000, 0.000] | 103.000 [0.000, 159.000] | 0.087 [0.024, 0.130] |
| xvla | fixed_0000 | actionstream_aligned - lerobot_weighted_average | 3 | 0.333 [0.000, 1.000] | -38.333 [-132.000, 12.000] | 0.040 [0.000, 0.078] |
| xvla | fixed_0000 | actionstream_aligned - lerobot_latest_only | 3 | 0.000 [0.000, 0.000] | 10.333 [6.000, 17.000] | 0.020 [-0.012, 0.074] |
| xvla | fixed_0250 | actionstream_aligned - lerobot_weighted_average | 3 | 0.333 [0.000, 1.000] | -39.667 [-135.000, 9.000] | -0.042 [-0.070, 0.002] |
| xvla | fixed_0250 | actionstream_aligned - lerobot_latest_only | 3 | 0.000 [0.000, 0.000] | -1.333 [-6.000, 4.000] | -0.021 [-0.049, -0.003] |
| xvla | fixed_0500 | actionstream_aligned - lerobot_weighted_average | 3 | 0.333 [0.000, 1.000] | -50.333 [-139.000, 1.000] | 0.002 [-0.013, 0.019] |
| xvla | fixed_0500 | actionstream_aligned - lerobot_latest_only | 3 | 0.000 [0.000, 0.000] | 4.000 [0.000, 11.000] | -0.025 [-0.088, 0.013] |
| xvla | fixed_0950 | actionstream_aligned - lerobot_weighted_average | 3 | 0.333 [0.000, 1.000] | -77.000 [-141.000, -36.000] | -0.099 [-0.336, 0.048] |
| xvla | fixed_0950 | actionstream_aligned - lerobot_latest_only | 3 | 0.000 [0.000, 0.000] | -36.333 [-37.000, -36.000] | -0.048 [-0.131, 0.004] |
| xvla | jitter_0500_pm0250 | actionstream_aligned - lerobot_weighted_average | 3 | 0.333 [0.000, 1.000] | -59.000 [-143.000, -14.000] | 0.025 [-0.020, 0.092] |
| xvla | jitter_0500_pm0250 | actionstream_aligned - lerobot_latest_only | 3 | 0.000 [0.000, 0.000] | -9.667 [-19.000, 0.000] | 0.071 [0.016, 0.104] |

## Evidence boundary

- Smoke episodes are excluded from this report.
- A 280-step horizon failure is a task outcome; a shorter intentional smoke truncation is not.
- Cross-model comparisons are not paired and are not used for method claims.
- Per-episode delivery p95 includes the first cold inference and depends on the number of inference calls, so the compact table uses p50; p95 remains in JSON/CSV.
- Native Isaac evidence is reported separately and requires a completed native runner receipt plus a live observation-conditioned policy.
