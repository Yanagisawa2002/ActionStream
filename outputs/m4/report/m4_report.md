# ActionStream M4 Final Report

## 1. M3 validated results

### Evidence validation

| evidence | episodes | traces | actions | commit |
|---|---:|---:|---:|---|
| M3 frozen matrix | 180 | 180 | 29073 | `9f0be1055d4216ca5a60702dd75f6f5a4e1c8fcc` |
| sync_hold 0/200 ms | 60 | 60 | 8829 | `e33c7d127fddfa069ca8dbcb2ce9049ed58aa132` |
| calibration | 24 | 24 | 7582 | `e33c7d127fddfa069ca8dbcb2ce9049ed58aa132` |
| selected pressure | 90 | 90 | 27357 | `e33c7d127fddfa069ca8dbcb2ce9049ed58aa132` |

All 180 episodes succeeded. Success-rate differences are ceilinged and do not establish equal robustness.

| mode | delay | n | success mean/median | steps mean/median | wall s mean/median | delivery s mean/median | discontinuity mean/median |
|---|---:|---:|---:|---:|---:|---:|---:|
| async_aligned | 0 | 30 | 1.000/1.000 | 136.600/134.500 | 9.235/9.070 | 0.131/0.131 | 0.158/0.143 |
| async_aligned | 200 | 30 | 1.000/1.000 | 142.733/130.500 | 10.075/8.897 | 0.457/0.458 | 0.183/0.151 |
| async_naive | 0 | 30 | 1.000/1.000 | 189.700/185.500 | 11.947/11.611 | 0.128/0.128 | 0.210/0.183 |
| async_naive | 200 | 30 | 1.000/1.000 | 246.267/238.000 | 15.217/14.932 | 0.452/0.456 | 0.275/0.269 |
| blocking evaluator baseline | 0 | 30 | 1.000/1.000 | 126.900/126.000 | 8.831/8.652 | 0.151/0.157 | 0.115/0.132 |
| blocking evaluator baseline | 200 | 30 | 1.000/1.000 | 126.900/126.000 | 10.435/10.140 | 0.490/0.505 | 0.115/0.132 |

## 2. Paired statistical analysis

Fixed seed `20260730`, 10,000 paired episode resamples. Differences are estimate minus reference. The table below renders the overall M3 effects; task-level effects remain preserved in the JSON evidence.

| comparison | metric | n | reference mean/median | estimate mean/median | mean difference [95% CI] | relative difference [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| async_aligned_minus_async_naive_delay0 | episode success | 30 | 1.000/1.000 | 1.000/1.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| async_aligned_minus_async_naive_delay0 | environment steps | 30 | 189.700/185.500 | 136.600/134.500 | -53.100 [-56.367, -50.067] | -27.887% [-28.808, -26.968]% |
| async_aligned_minus_async_naive_delay0 | wall-clock seconds | 30 | 11.947/11.611 | 9.235/9.070 | -2.712 [-2.921, -2.511] | -22.640% [-23.954, -21.278]% |
| async_aligned_minus_async_naive_delay0 | observation-to-delivery seconds | 30 | 0.128/0.128 | 0.131/0.131 | 0.003 [0.001, 0.005] | 2.232% [0.574, 3.907]% |
| async_aligned_minus_async_naive_delay0 | mean adjacent-action 7D L2 | 30 | 0.210/0.183 | 0.158/0.143 | -0.052 [-0.089, -0.015] | -6.016% [-27.029, 18.248]% |
| async_aligned_minus_blocking_sync_delay0 | episode success | 30 | 1.000/1.000 | 1.000/1.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| async_aligned_minus_blocking_sync_delay0 | environment steps | 30 | 126.900/126.000 | 136.600/134.500 | 9.700 [7.933, 11.667] | 7.713% [6.309, 9.239]% |
| async_aligned_minus_blocking_sync_delay0 | wall-clock seconds | 30 | 8.831/8.652 | 9.235/9.070 | 0.404 [0.238, 0.581] | 4.633% [2.774, 6.604]% |
| async_aligned_minus_blocking_sync_delay0 | observation-to-delivery seconds | 30 | 0.151/0.157 | 0.131/0.131 | -0.020 [-0.029, -0.011] | -10.845% [-16.596, -4.688]% |
| async_aligned_minus_blocking_sync_delay0 | mean adjacent-action 7D L2 | 30 | 0.115/0.132 | 0.158/0.143 | 0.044 [0.012, 0.075] | 81.366% [39.035, 126.801]% |
| async_aligned_minus_async_naive_delay200 | episode success | 30 | 1.000/1.000 | 1.000/1.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| async_aligned_minus_async_naive_delay200 | environment steps | 30 | 246.267/238.000 | 142.733/130.500 | -103.533 [-118.833, -79.999] | -41.766% [-47.204, -32.335]% |
| async_aligned_minus_async_naive_delay200 | wall-clock seconds | 30 | 15.217/14.932 | 10.075/8.897 | -5.142 [-6.232, -3.679] | -33.887% [-40.608, -24.534]% |
| async_aligned_minus_async_naive_delay200 | observation-to-delivery seconds | 30 | 0.452/0.456 | 0.457/0.458 | 0.005 [-0.000, 0.013] | 1.389% [0.003, 3.568]% |
| async_aligned_minus_async_naive_delay200 | mean adjacent-action 7D L2 | 30 | 0.275/0.269 | 0.183/0.151 | -0.092 [-0.129, -0.054] | -19.393% [-40.136, 7.212]% |
| async_aligned_minus_blocking_sync_delay200 | episode success | 30 | 1.000/1.000 | 1.000/1.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| async_aligned_minus_blocking_sync_delay200 | environment steps | 30 | 126.900/126.000 | 142.733/130.500 | 15.833 [3.100, 37.967] | 12.804% [2.513, 30.749]% |
| async_aligned_minus_blocking_sync_delay200 | wall-clock seconds | 30 | 10.435/10.140 | 10.075/8.897 | -0.359 [-1.401, 1.060] | -3.644% [-13.115, 9.684]% |
| async_aligned_minus_blocking_sync_delay200 | observation-to-delivery seconds | 30 | 0.490/0.505 | 0.457/0.458 | -0.033 [-0.048, -0.016] | -5.848% [-9.039, -2.253]% |
| async_aligned_minus_blocking_sync_delay200 | mean adjacent-action 7D L2 | 30 | 0.115/0.132 | 0.183/0.151 | 0.068 [0.035, 0.101] | 96.452% [52.047, 144.528]% |
| sync_delay200_minus_delay0 | episode success | 30 | 1.000/1.000 | 1.000/1.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| sync_delay200_minus_delay0 | environment steps | 30 | 126.900/126.000 | 126.900/126.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| sync_delay200_minus_delay0 | wall-clock seconds | 30 | 8.831/8.652 | 10.435/10.140 | 1.604 [1.426, 1.807] | 18.133% [16.275, 20.235]% |
| sync_delay200_minus_delay0 | observation-to-delivery seconds | 30 | 0.151/0.157 | 0.490/0.505 | 0.339 [0.318, 0.359] | 234.978% [208.925, 262.689]% |
| sync_delay200_minus_delay0 | mean adjacent-action 7D L2 | 30 | 0.115/0.132 | 0.115/0.132 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| async_naive_delay200_minus_delay0 | episode success | 30 | 1.000/1.000 | 1.000/1.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| async_naive_delay200_minus_delay0 | environment steps | 30 | 189.700/185.500 | 246.267/238.000 | 56.567 [53.367, 59.633] | 29.810% [28.546, 31.009]% |
| async_naive_delay200_minus_delay0 | wall-clock seconds | 30 | 11.947/11.611 | 15.217/14.932 | 3.270 [3.089, 3.440] | 27.412% [26.114, 28.585]% |
| async_naive_delay200_minus_delay0 | observation-to-delivery seconds | 30 | 0.128/0.128 | 0.452/0.456 | 0.323 [0.316, 0.328] | 252.735% [246.752, 257.905]% |
| async_naive_delay200_minus_delay0 | mean adjacent-action 7D L2 | 30 | 0.210/0.183 | 0.275/0.269 | 0.065 [0.024, 0.105] | 72.292% [25.459, 130.346]% |
| async_aligned_delay200_minus_delay0 | episode success | 30 | 1.000/1.000 | 1.000/1.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| async_aligned_delay200_minus_delay0 | environment steps | 30 | 136.600/134.500 | 142.733/130.500 | 6.133 [-6.400, 28.167] | 4.705% [-4.607, 21.211]% |
| async_aligned_delay200_minus_delay0 | wall-clock seconds | 30 | 9.235/9.070 | 10.075/8.897 | 0.840 [-0.174, 2.246] | 8.584% [-1.679, 23.467]% |
| async_aligned_delay200_minus_delay0 | observation-to-delivery seconds | 30 | 0.131/0.131 | 0.457/0.458 | 0.326 [0.323, 0.329] | 249.662% [244.558, 254.667]% |
| async_aligned_delay200_minus_delay0 | mean adjacent-action 7D L2 | 30 | 0.158/0.143 | 0.183/0.151 | 0.024 [-0.012, 0.060] | 39.455% [7.531, 77.535]% |

## 3. Blocking sync versus sync_hold semantics

Historical `sync` blocks simulator time during inference. `sync_hold` waits for the first valid chunk without stepping, then advances the real-time simulator with the last finite 7D command whenever a consumed chunk leaves inference pending.

### sync_hold at ordinary latency

| mode | delay | scope | n | success mean/median | steps mean/median | wall s mean/median | delivery s mean/median | inference s mean/median | control s mean/median | queue depth mean/median | request headroom mean/median | holds mean/median | stale mean/median | age mean/median |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sync_hold | 0 | all | 30 | 1.000/1.000 | 138.300/135.000 | 8.897/8.541 | 0.162/0.162 | 0.148/0.150 | 0.051/0.051 | 14.833/15.000 | 0.000/0.000 | 12.300/13.000 | 0.000/0.000 | 3.185/3.155 |
| sync_hold | 0 | task 0 | 10 | 1.000/1.000 | 154.600/154.000 | 9.877/9.707 | 0.173/0.172 | 0.160/0.157 | 0.051/0.051 | 14.900/15.000 | 0.000/0.000 | 14.500/14.000 | 0.000/0.000 | 3.389/3.349 |
| sync_hold | 0 | task 1 | 10 | 1.000/1.000 | 137.100/135.000 | 8.670/8.541 | 0.166/0.176 | 0.153/0.164 | 0.050/0.050 | 14.600/15.000 | 0.000/0.000 | 13.600/14.500 | 0.000/0.000 | 3.318/3.527 |
| sync_hold | 0 | task 2 | 10 | 1.000/1.000 | 123.200/122.500 | 8.145/8.167 | 0.146/0.147 | 0.132/0.133 | 0.051/0.051 | 15.000/15.000 | 0.000/0.000 | 8.800/8.500 | 0.000/0.000 | 2.849/2.850 |
| sync_hold | 200 | all | 30 | 1.000/1.000 | 156.000/155.500 | 9.927/9.811 | 0.415/0.411 | 0.202/0.195 | 0.051/0.051 | 12.800/13.000 | 0.000/0.000 | 29.533/31.000 | 0.000/0.000 | 8.171/8.005 |
| sync_hold | 200 | task 0 | 10 | 1.000/1.000 | 174.200/173.500 | 11.007/10.781 | 0.402/0.402 | 0.189/0.190 | 0.051/0.051 | 12.950/13.000 | 0.000/0.000 | 33.300/33.500 | 0.000/0.000 | 7.889/7.856 |
| sync_hold | 200 | task 1 | 10 | 1.000/1.000 | 155.000/155.500 | 9.683/9.750 | 0.417/0.416 | 0.204/0.200 | 0.050/0.050 | 12.350/12.250 | 0.000/0.000 | 30.700/31.500 | 0.000/0.000 | 8.320/8.294 |
| sync_hold | 200 | task 2 | 10 | 1.000/1.000 | 138.800/139.000 | 9.091/9.093 | 0.427/0.420 | 0.214/0.207 | 0.051/0.051 | 13.100/13.000 | 0.000/0.000 | 24.600/24.000 | 0.000/0.000 | 8.304/8.110 |

## 4. Queue-pressure calibration

- Derived queue headroom: 20 steps (1.055 s at the measured median control step; 1.241 s at p95).
- M3 control-step duration median/p95: 0.053/0.062 s.
- M3 model-inference latency median/p95: 0.188/0.262 s.
- M3 observation-to-delivery latency median/p95: 0.369/0.473 s.
- Candidate derivation used the 0 ms observation-to-delivery median 0.128785 s in `target_age_steps * median_control_step - median_0ms_delivery`; the pooled 0/200 ms median above was not used in that formula.
- Candidate delays and target ages: 400 ms -> 10.000 steps, 650 ms -> 15.000 steps, 950 ms -> 20.000 steps, 1150 ms -> 24.000 steps.
- Tested delays: [400, 650, 950, 1150] ms.
- Selected pressure delay: **950 ms**, recomputed as the smallest qualifying tested delay.
- 400 ms: median age 10.443 steps, holds 0, fully stale 0, qualifies False.
- 650 ms: median age 15.489 steps, holds 0, fully stale 0, qualifies False.
- 950 ms: median age 21.509 steps, holds 42, fully stale 0, qualifies True.
- 1150 ms: median age 25.418 steps, holds 574, fully stale 0, qualifies True.

## 5. Full selected-pressure results

| mode | delay | scope | n | success mean/median | steps mean/median | wall s mean/median | delivery s mean/median | inference s mean/median | control s mean/median | queue depth mean/median | request headroom mean/median | holds mean/median | stale mean/median | age mean/median |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sync_hold | 950 | all | 30 | 1.000/1.000 | 211.333/213.000 | 13.673/13.439 | 1.232/1.232 | 0.269/0.269 | 0.051/0.051 | 6.133/6.250 | 0.000/0.000 | 84.867/87.500 | 0.000/0.000 | 24.128/24.042 |
| sync_hold | 950 | task 0 | 10 | 1.000/1.000 | 236.600/236.500 | 15.206/14.983 | 1.235/1.243 | 0.273/0.280 | 0.051/0.052 | 6.050/6.000 | 0.000/0.000 | 95.500/95.500 | 0.000/0.000 | 23.996/24.051 |
| sync_hold | 950 | task 1 | 10 | 1.000/1.000 | 211.000/213.000 | 13.433/13.410 | 1.238/1.245 | 0.276/0.280 | 0.050/0.050 | 5.250/5.000 | 0.000/0.000 | 87.200/88.500 | 0.000/0.000 | 24.552/24.663 |
| sync_hold | 950 | task 2 | 10 | 1.000/1.000 | 186.400/186.000 | 12.381/12.342 | 1.222/1.225 | 0.259/0.263 | 0.051/0.051 | 7.100/7.000 | 0.000/0.000 | 71.900/72.000 | 0.000/0.000 | 23.837/23.820 |
| async_naive | 950 | all | 30 | 0.633/1.000 | 550.233/439.500 | 30.296/26.176 | 1.070/1.072 | 0.105/0.106 | 0.051/0.051 | 25.000/25.000 | 21.333/21.000 | 1.500/1.000 | 0.000/0.000 | 21.102/21.064 |
| async_naive | 950 | task 0 | 10 | 0.500/0.500 | 624.000/690.500 | 33.905/37.125 | 1.075/1.075 | 0.110/0.111 | 0.051/0.051 | 25.000/25.000 | 21.500/21.500 | 1.500/1.000 | 0.000/0.000 | 21.134/21.185 |
| async_naive | 950 | task 1 | 10 | 0.800/1.000 | 474.100/390.500 | 26.755/23.052 | 1.070/1.071 | 0.105/0.105 | 0.051/0.051 | 25.000/25.000 | 21.100/21.000 | 1.300/1.000 | 0.000/0.000 | 21.160/21.142 |
| async_naive | 950 | task 2 | 10 | 0.600/1.000 | 552.600/474.000 | 30.226/27.277 | 1.066/1.065 | 0.100/0.099 | 0.051/0.051 | 25.000/25.000 | 21.400/21.000 | 1.700/1.000 | 0.000/0.000 | 21.012/20.802 |
| async_aligned | 950 | all | 30 | 0.967/1.000 | 150.333/128.000 | 10.370/9.174 | 1.065/1.065 | 0.099/0.099 | 0.051/0.051 | 5.367/5.000 | 0.000/0.000 | 16.367/12.500 | 0.000/0.000 | 20.959/20.954 |
| async_aligned | 950 | task 0 | 10 | 1.000/1.000 | 142.700/144.500 | 10.210/10.142 | 1.066/1.065 | 0.100/0.099 | 0.051/0.051 | 5.250/5.000 | 0.000/0.000 | 13.200/12.500 | 0.000/0.000 | 20.858/20.860 |
| async_aligned | 950 | task 1 | 10 | 0.900/1.000 | 193.500/128.000 | 12.269/9.174 | 1.065/1.065 | 0.099/0.100 | 0.050/0.050 | 4.900/5.000 | 0.000/0.000 | 27.600/14.500 | 0.000/0.000 | 21.234/21.236 |
| async_aligned | 950 | task 2 | 10 | 1.000/1.000 | 114.800/116.000 | 8.629/8.684 | 1.064/1.064 | 0.099/0.098 | 0.051/0.051 | 5.950/6.000 | 0.000/0.000 | 8.300/9.000 | 0.000/0.000 | 20.786/20.759 |

### M4 paired effects

Fixed seed `20260730`, 10,000 paired episode resamples. Differences are estimate minus reference.

| comparison | scope | metric | n | reference mean | estimate mean | mean difference [95% CI] | relative difference [95% CI] |
|---|---|---|---:|---:|---:|---:|---:|
| sync_hold_delay200_minus_delay0 | all | episode success | 30 | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| sync_hold_delay200_minus_delay0 | all | environment steps | 30 | 138.300 | 156.000 | 17.700 [16.533, 18.900] | 12.818% [12.055, 13.619]% |
| sync_hold_delay200_minus_delay0 | all | wall-clock seconds | 30 | 8.897 | 9.927 | 1.030 [0.951, 1.109] | 11.622% [10.726, 12.540]% |
| sync_hold_delay200_minus_delay0 | all | episode p50 observation-to-delivery seconds | 30 | 0.162 | 0.415 | 0.254 [0.241, 0.266] | 164.072% [146.740, 181.802]% |
| sync_hold_delay200_minus_delay0 | all | episode p50 model-inference seconds | 30 | 0.148 | 0.202 | 0.054 [0.042, 0.067] | 41.585% [30.053, 53.736]% |
| sync_hold_delay200_minus_delay0 | all | episode p50 control-step seconds | 30 | 0.051 | 0.051 | -0.000 [-0.000, 0.000] | -0.011% [-0.243, 0.217]% |
| sync_hold_delay200_minus_delay0 | all | episode p50 queue depth (steps) | 30 | 14.833 | 12.800 | -2.033 [-2.167, -1.900] | -13.716% [-14.595, -12.827]% |
| sync_hold_delay200_minus_delay0 | all | episode p50 request headroom (steps) | 30 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | all | queue-hold steps | 30 | 12.300 | 29.533 | 17.233 [16.333, 18.100] | 149.279% [133.996, 164.861]% |
| sync_hold_delay200_minus_delay0 | all | fully stale chunks | 30 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | all | episode p50 delivery age (steps) | 30 | 3.185 | 8.171 | 4.985 [4.749, 5.226] | 164.023% [146.825, 181.627]% |
| sync_hold_delay200_minus_delay0 | all | mean adjacent-action 7D L2 | 30 | 0.111 | 0.096 | -0.015 [-0.021, -0.010] | -12.153% [-17.626, -7.312]% |
| sync_hold_delay200_minus_delay0 | all | mean incoming stale fraction | 30 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | all | accepted chunks | 30 | 4.500 | 4.500 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| sync_hold_delay200_minus_delay0 | all | rejected chunks | 30 | 0.100 | 0.100 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | all | chunks rejected after episode end | 30 | 0.100 | 0.100 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | all | mean replacement position L2 | 30 | 0.013 | 0.013 | -0.000 [-0.000, 0.000] | -1.526% [-3.415, 0.328]% |
| sync_hold_delay200_minus_delay0 | all | mean replacement rotation geodesic angle (rad) | 30 | 0.017 | 0.017 | 0.000 [-0.000, 0.001] | 1.216% [-0.662, 3.354]% |
| sync_hold_delay200_minus_delay0 | all | replacement gripper switches | 30 | 0.033 | 0.067 | 0.033 [0.000, 0.100] | n/a |
| sync_hold_delay200_minus_delay0 | task 0 | episode success | 10 | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| sync_hold_delay200_minus_delay0 | task 0 | environment steps | 10 | 154.600 | 174.200 | 19.600 [18.500, 21.200] | 12.704% [11.844, 13.876]% |
| sync_hold_delay200_minus_delay0 | task 0 | wall-clock seconds | 10 | 9.877 | 11.007 | 1.130 [1.044, 1.221] | 11.492% [10.525, 12.558]% |
| sync_hold_delay200_minus_delay0 | task 0 | episode p50 observation-to-delivery seconds | 10 | 0.173 | 0.402 | 0.229 [0.219, 0.239] | 134.248% [121.495, 146.888]% |
| sync_hold_delay200_minus_delay0 | task 0 | episode p50 model-inference seconds | 10 | 0.160 | 0.189 | 0.029 [0.019, 0.039] | 19.223% [12.238, 26.065]% |
| sync_hold_delay200_minus_delay0 | task 0 | episode p50 control-step seconds | 10 | 0.051 | 0.051 | -0.000 [-0.000, 0.000] | -0.145% [-0.652, 0.289]% |
| sync_hold_delay200_minus_delay0 | task 0 | episode p50 queue depth (steps) | 10 | 14.900 | 12.950 | -1.950 [-2.100, -1.800] | -13.076% [-14.083, -12.034]% |
| sync_hold_delay200_minus_delay0 | task 0 | episode p50 request headroom (steps) | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 0 | queue-hold steps | 10 | 14.500 | 33.300 | 18.800 [18.100, 19.400] | 131.414% [119.721, 140.597]% |
| sync_hold_delay200_minus_delay0 | task 0 | fully stale chunks | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 0 | episode p50 delivery age (steps) | 10 | 3.389 | 7.889 | 4.500 [4.312, 4.686] | 134.503% [122.282, 146.584]% |
| sync_hold_delay200_minus_delay0 | task 0 | mean adjacent-action 7D L2 | 10 | 0.093 | 0.083 | -0.009 [-0.013, -0.004] | -6.395% [-11.228, 2.837]% |
| sync_hold_delay200_minus_delay0 | task 0 | mean incoming stale fraction | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 0 | accepted chunks | 10 | 5.000 | 5.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| sync_hold_delay200_minus_delay0 | task 0 | rejected chunks | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 0 | chunks rejected after episode end | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 0 | mean replacement position L2 | 10 | 0.013 | 0.013 | -0.000 [-0.001, -0.000] | -2.675% [-5.244, 0.017]% |
| sync_hold_delay200_minus_delay0 | task 0 | mean replacement rotation geodesic angle (rad) | 10 | 0.016 | 0.017 | 0.000 [-0.000, 0.001] | 1.345% [-0.637, 2.935]% |
| sync_hold_delay200_minus_delay0 | task 0 | replacement gripper switches | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 1 | episode success | 10 | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| sync_hold_delay200_minus_delay0 | task 1 | environment steps | 10 | 137.100 | 155.000 | 17.900 [15.700, 20.300] | 13.099% [11.466, 14.968]% |
| sync_hold_delay200_minus_delay0 | task 1 | wall-clock seconds | 10 | 8.670 | 9.683 | 1.014 [0.861, 1.181] | 11.760% [9.877, 13.840]% |
| sync_hold_delay200_minus_delay0 | task 1 | episode p50 observation-to-delivery seconds | 10 | 0.166 | 0.417 | 0.251 [0.231, 0.272] | 160.284% [129.504, 193.650]% |
| sync_hold_delay200_minus_delay0 | task 1 | episode p50 model-inference seconds | 10 | 0.153 | 0.204 | 0.051 [0.031, 0.072] | 39.386% [20.490, 60.083]% |
| sync_hold_delay200_minus_delay0 | task 1 | episode p50 control-step seconds | 10 | 0.050 | 0.050 | 0.000 [-0.000, 0.000] | 0.127% [-0.134, 0.339]% |
| sync_hold_delay200_minus_delay0 | task 1 | episode p50 queue depth (steps) | 10 | 14.600 | 12.350 | -2.250 [-2.450, -2.050] | -15.381% [-16.714, -14.238]% |
| sync_hold_delay200_minus_delay0 | task 1 | episode p50 request headroom (steps) | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 1 | queue-hold steps | 10 | 13.600 | 30.700 | 17.100 [15.400, 19.000] | 132.089% [108.173, 159.583]% |
| sync_hold_delay200_minus_delay0 | task 1 | fully stale chunks | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 1 | episode p50 delivery age (steps) | 10 | 3.318 | 8.320 | 5.001 [4.595, 5.418] | 159.953% [129.314, 193.200]% |
| sync_hold_delay200_minus_delay0 | task 1 | mean adjacent-action 7D L2 | 10 | 0.097 | 0.078 | -0.019 [-0.035, -0.008] | -17.663% [-29.983, -10.791]% |
| sync_hold_delay200_minus_delay0 | task 1 | mean incoming stale fraction | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 1 | accepted chunks | 10 | 4.500 | 4.500 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| sync_hold_delay200_minus_delay0 | task 1 | rejected chunks | 10 | 0.300 | 0.300 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 1 | chunks rejected after episode end | 10 | 0.300 | 0.300 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 1 | mean replacement position L2 | 10 | 0.014 | 0.013 | -0.000 [-0.001, 0.000] | -1.818% [-6.098, 2.594]% |
| sync_hold_delay200_minus_delay0 | task 1 | mean replacement rotation geodesic angle (rad) | 10 | 0.017 | 0.017 | 0.000 [-0.000, 0.001] | 2.379% [-2.359, 8.140]% |
| sync_hold_delay200_minus_delay0 | task 1 | replacement gripper switches | 10 | 0.100 | 0.200 | 0.100 [0.000, 0.300] | n/a |
| sync_hold_delay200_minus_delay0 | task 2 | episode success | 10 | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| sync_hold_delay200_minus_delay0 | task 2 | environment steps | 10 | 123.200 | 138.800 | 15.600 [14.200, 17.000] | 12.652% [11.534, 13.728]% |
| sync_hold_delay200_minus_delay0 | task 2 | wall-clock seconds | 10 | 8.145 | 9.091 | 0.946 [0.816, 1.074] | 11.614% [10.026, 13.165]% |
| sync_hold_delay200_minus_delay0 | task 2 | episode p50 observation-to-delivery seconds | 10 | 0.146 | 0.427 | 0.280 [0.261, 0.300] | 197.685% [170.671, 223.476]% |
| sync_hold_delay200_minus_delay0 | task 2 | episode p50 model-inference seconds | 10 | 0.132 | 0.214 | 0.082 [0.062, 0.102] | 66.145% [47.214, 84.974]% |
| sync_hold_delay200_minus_delay0 | task 2 | episode p50 control-step seconds | 10 | 0.051 | 0.051 | -0.000 [-0.000, 0.000] | -0.016% [-0.440, 0.415]% |
| sync_hold_delay200_minus_delay0 | task 2 | episode p50 queue depth (steps) | 10 | 15.000 | 13.100 | -1.900 [-2.100, -1.650] | -12.692% [-14.140, -10.968]% |
| sync_hold_delay200_minus_delay0 | task 2 | episode p50 request headroom (steps) | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 2 | queue-hold steps | 10 | 8.800 | 24.600 | 15.800 [14.500, 17.100] | 184.334% [158.039, 208.000]% |
| sync_hold_delay200_minus_delay0 | task 2 | fully stale chunks | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 2 | episode p50 delivery age (steps) | 10 | 2.849 | 8.304 | 5.455 [5.097, 5.814] | 197.613% [171.269, 222.851]% |
| sync_hold_delay200_minus_delay0 | task 2 | mean adjacent-action 7D L2 | 10 | 0.145 | 0.127 | -0.017 [-0.023, -0.012] | -12.400% [-14.437, -10.829]% |
| sync_hold_delay200_minus_delay0 | task 2 | mean incoming stale fraction | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 2 | accepted chunks | 10 | 4.000 | 4.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| sync_hold_delay200_minus_delay0 | task 2 | rejected chunks | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 2 | chunks rejected after episode end | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| sync_hold_delay200_minus_delay0 | task 2 | mean replacement position L2 | 10 | 0.013 | 0.013 | -0.000 [-0.000, 0.000] | -0.085% [-2.208, 1.761]% |
| sync_hold_delay200_minus_delay0 | task 2 | mean replacement rotation geodesic angle (rad) | 10 | 0.018 | 0.018 | -0.000 [-0.000, 0.000] | -0.077% [-2.160, 2.117]% |
| sync_hold_delay200_minus_delay0 | task 2 | replacement gripper switches | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_aligned_minus_async_naive | all | episode success | 30 | 0.633 | 0.967 | 0.333 [0.133, 0.533] | n/a |
| pressure_async_aligned_minus_async_naive | all | environment steps | 30 | 550.233 | 150.333 | -399.900 [-482.336, -315.066] | -69.378% [-76.219, -58.315]% |
| pressure_async_aligned_minus_async_naive | all | wall-clock seconds | 30 | 30.296 | 10.370 | -19.926 [-23.672, -16.034] | -62.984% [-69.259, -53.538]% |
| pressure_async_aligned_minus_async_naive | all | episode p50 observation-to-delivery seconds | 30 | 1.070 | 1.065 | -0.005 [-0.007, -0.004] | -0.504% [-0.662, -0.342]% |
| pressure_async_aligned_minus_async_naive | all | episode p50 model-inference seconds | 30 | 0.105 | 0.099 | -0.006 [-0.008, -0.004] | -5.307% [-7.040, -3.549]% |
| pressure_async_aligned_minus_async_naive | all | episode p50 control-step seconds | 30 | 0.051 | 0.051 | 0.000 [-0.000, 0.000] | 0.179% [-0.400, 0.823]% |
| pressure_async_aligned_minus_async_naive | all | episode p50 queue depth (steps) | 30 | 25.000 | 5.367 | -19.633 [-19.817, -19.450] | -78.533% [-79.267, -77.800]% |
| pressure_async_aligned_minus_async_naive | all | episode p50 request headroom (steps) | 30 | 21.333 | 0.000 | -21.333 [-21.500, -21.167] | -100.000% [-100.000, -100.000]% |
| pressure_async_aligned_minus_async_naive | all | queue-hold steps | 30 | 1.500 | 16.367 | 14.867 [9.533, 24.400] | 1304.889% [746.000, 2294.972]% |
| pressure_async_aligned_minus_async_naive | all | fully stale chunks | 30 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_aligned_minus_async_naive | all | episode p50 delivery age (steps) | 30 | 21.102 | 20.959 | -0.142 [-0.272, -0.022] | -0.655% [-1.261, -0.088]% |
| pressure_async_aligned_minus_async_naive | all | mean adjacent-action 7D L2 | 30 | 0.292 | 0.182 | -0.110 [-0.157, -0.065] | -36.150% [-49.812, -21.571]% |
| pressure_async_aligned_minus_async_naive | all | mean incoming stale fraction | 30 | 0.000 | 0.644 | 0.644 [0.638, 0.651] | n/a |
| pressure_async_aligned_minus_async_naive | all | accepted chunks | 30 | 53.167 | 13.367 | -39.800 [-48.000, -31.400] | -71.726% [-78.515, -60.601]% |
| pressure_async_aligned_minus_async_naive | all | rejected chunks | 30 | 2.233 | 2.067 | -0.167 [-0.334, 0.000] | -4.444% [-11.111, 2.778]% |
| pressure_async_aligned_minus_async_naive | all | chunks rejected after episode end | 30 | 2.233 | 2.067 | -0.167 [-0.334, 0.000] | -4.444% [-11.111, 2.778]% |
| pressure_async_aligned_minus_async_naive | all | mean replacement position L2 | 30 | 0.058 | 0.025 | -0.033 [-0.040, -0.022] | -54.780% [-64.704, -38.167]% |
| pressure_async_aligned_minus_async_naive | all | mean replacement rotation geodesic angle (rad) | 30 | 0.082 | 0.047 | -0.035 [-0.043, -0.026] | -38.496% [-46.792, -29.566]% |
| pressure_async_aligned_minus_async_naive | all | replacement gripper switches | 30 | 11.633 | 1.867 | -9.767 [-14.401, -4.767] | -72.084% [-93.565, -36.806]% |
| pressure_async_aligned_minus_async_naive | task 0 | episode success | 10 | 0.500 | 1.000 | 0.500 [0.200, 0.800] | n/a |
| pressure_async_aligned_minus_async_naive | task 0 | environment steps | 10 | 624.000 | 142.700 | -481.300 [-595.403, -364.900] | -74.861% [-79.674, -69.874]% |
| pressure_async_aligned_minus_async_naive | task 0 | wall-clock seconds | 10 | 33.905 | 10.210 | -23.695 [-28.716, -18.608] | -67.967% [-73.282, -62.425]% |
| pressure_async_aligned_minus_async_naive | task 0 | episode p50 observation-to-delivery seconds | 10 | 1.075 | 1.066 | -0.009 [-0.012, -0.007] | -0.867% [-1.075, -0.647]% |
| pressure_async_aligned_minus_async_naive | task 0 | episode p50 model-inference seconds | 10 | 0.110 | 0.100 | -0.011 [-0.013, -0.008] | -9.616% [-11.676, -7.466]% |
| pressure_async_aligned_minus_async_naive | task 0 | episode p50 control-step seconds | 10 | 0.051 | 0.051 | 0.000 [-0.000, 0.001] | 0.438% [-0.399, 1.284]% |
| pressure_async_aligned_minus_async_naive | task 0 | episode p50 queue depth (steps) | 10 | 25.000 | 5.250 | -19.750 [-19.950, -19.550] | -79.000% [-79.800, -78.200]% |
| pressure_async_aligned_minus_async_naive | task 0 | episode p50 request headroom (steps) | 10 | 21.500 | 0.000 | -21.500 [-21.800, -21.200] | -100.000% [-100.000, -100.000]% |
| pressure_async_aligned_minus_async_naive | task 0 | queue-hold steps | 10 | 1.500 | 13.200 | 11.700 [10.700, 13.000] | 926.667% [693.333, 1170.000]% |
| pressure_async_aligned_minus_async_naive | task 0 | fully stale chunks | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_aligned_minus_async_naive | task 0 | episode p50 delivery age (steps) | 10 | 21.134 | 20.858 | -0.276 [-0.465, -0.092] | -1.281% [-2.161, -0.424]% |
| pressure_async_aligned_minus_async_naive | task 0 | mean adjacent-action 7D L2 | 10 | 0.362 | 0.167 | -0.195 [-0.291, -0.112] | -54.422% [-72.901, -35.393]% |
| pressure_async_aligned_minus_async_naive | task 0 | mean incoming stale fraction | 10 | 0.000 | 0.648 | 0.648 [0.645, 0.652] | n/a |
| pressure_async_aligned_minus_async_naive | task 0 | accepted chunks | 10 | 60.400 | 12.600 | -47.800 [-59.300, -36.100] | -76.809% [-81.608, -71.879]% |
| pressure_async_aligned_minus_async_naive | task 0 | rejected chunks | 10 | 2.300 | 2.100 | -0.200 [-0.600, 0.200] | -5.000% [-20.000, 11.667]% |
| pressure_async_aligned_minus_async_naive | task 0 | chunks rejected after episode end | 10 | 2.300 | 2.100 | -0.200 [-0.600, 0.200] | -5.000% [-20.000, 11.667]% |
| pressure_async_aligned_minus_async_naive | task 0 | mean replacement position L2 | 10 | 0.059 | 0.020 | -0.039 [-0.047, -0.031] | -64.685% [-71.271, -56.936]% |
| pressure_async_aligned_minus_async_naive | task 0 | mean replacement rotation geodesic angle (rad) | 10 | 0.063 | 0.047 | -0.016 [-0.026, -0.006] | -22.361% [-35.395, -7.917]% |
| pressure_async_aligned_minus_async_naive | task 0 | replacement gripper switches | 10 | 13.500 | 0.600 | -12.900 [-18.900, -6.700] | -83.253% [-98.307, -59.990]% |
| pressure_async_aligned_minus_async_naive | task 1 | episode success | 10 | 0.800 | 0.900 | 0.100 [-0.200, 0.400] | n/a |
| pressure_async_aligned_minus_async_naive | task 1 | environment steps | 10 | 474.100 | 193.500 | -280.600 [-424.200, -121.200] | -57.091% [-73.947, -28.545]% |
| pressure_async_aligned_minus_async_naive | task 1 | wall-clock seconds | 10 | 26.755 | 12.269 | -14.486 [-21.212, -6.987] | -52.350% [-67.205, -28.325]% |
| pressure_async_aligned_minus_async_naive | task 1 | episode p50 observation-to-delivery seconds | 10 | 1.070 | 1.065 | -0.005 [-0.007, -0.003] | -0.469% [-0.693, -0.243]% |
| pressure_async_aligned_minus_async_naive | task 1 | episode p50 model-inference seconds | 10 | 0.105 | 0.099 | -0.005 [-0.008, -0.003] | -4.910% [-7.042, -2.788]% |
| pressure_async_aligned_minus_async_naive | task 1 | episode p50 control-step seconds | 10 | 0.051 | 0.050 | -0.000 [-0.001, -0.000] | -0.815% [-1.316, -0.265]% |
| pressure_async_aligned_minus_async_naive | task 1 | episode p50 queue depth (steps) | 10 | 25.000 | 4.900 | -20.100 [-20.300, -20.000] | -80.400% [-81.200, -80.000]% |
| pressure_async_aligned_minus_async_naive | task 1 | episode p50 request headroom (steps) | 10 | 21.100 | 0.000 | -21.100 [-21.300, -21.000] | -100.000% [-100.000, -100.000]% |
| pressure_async_aligned_minus_async_naive | task 1 | queue-hold steps | 10 | 1.300 | 27.600 | 26.300 [12.800, 52.500] | 2435.000% [950.000, 5160.000]% |
| pressure_async_aligned_minus_async_naive | task 1 | fully stale chunks | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_aligned_minus_async_naive | task 1 | episode p50 delivery age (steps) | 10 | 21.160 | 21.234 | 0.074 [-0.032, 0.180] | 0.355% [-0.149, 0.858]% |
| pressure_async_aligned_minus_async_naive | task 1 | mean adjacent-action 7D L2 | 10 | 0.224 | 0.161 | -0.063 [-0.124, -0.001] | -24.811% [-51.351, 4.221]% |
| pressure_async_aligned_minus_async_naive | task 1 | mean incoming stale fraction | 10 | 0.000 | 0.657 | 0.657 [0.648, 0.672] | n/a |
| pressure_async_aligned_minus_async_naive | task 1 | accepted chunks | 10 | 45.600 | 17.700 | -27.900 [-42.100, -12.200] | -59.426% [-76.298, -30.380]% |
| pressure_async_aligned_minus_async_naive | task 1 | rejected chunks | 10 | 2.300 | 2.100 | -0.200 [-0.600, 0.200] | -5.000% [-20.000, 10.000]% |
| pressure_async_aligned_minus_async_naive | task 1 | chunks rejected after episode end | 10 | 2.300 | 2.100 | -0.200 [-0.600, 0.200] | -5.000% [-20.000, 10.000]% |
| pressure_async_aligned_minus_async_naive | task 1 | mean replacement position L2 | 10 | 0.061 | 0.036 | -0.026 [-0.043, 0.003] | -40.389% [-65.974, 5.406]% |
| pressure_async_aligned_minus_async_naive | task 1 | mean replacement rotation geodesic angle (rad) | 10 | 0.092 | 0.052 | -0.040 [-0.053, -0.024] | -42.885% [-55.441, -26.883]% |
| pressure_async_aligned_minus_async_naive | task 1 | replacement gripper switches | 10 | 7.900 | 4.900 | -3.000 [-11.800, 7.100] | -34.667% [-91.000, 57.333]% |
| pressure_async_aligned_minus_async_naive | task 2 | episode success | 10 | 0.600 | 1.000 | 0.400 [0.100, 0.700] | n/a |
| pressure_async_aligned_minus_async_naive | task 2 | environment steps | 10 | 552.600 | 114.800 | -437.800 [-562.600, -312.800] | -76.180% [-81.020, -71.180]% |
| pressure_async_aligned_minus_async_naive | task 2 | wall-clock seconds | 10 | 30.226 | 8.629 | -21.597 [-27.189, -15.986] | -68.636% [-73.949, -63.179]% |
| pressure_async_aligned_minus_async_naive | task 2 | episode p50 observation-to-delivery seconds | 10 | 1.066 | 1.064 | -0.002 [-0.004, 0.000] | -0.176% [-0.359, 0.007]% |
| pressure_async_aligned_minus_async_naive | task 2 | episode p50 model-inference seconds | 10 | 0.100 | 0.099 | -0.001 [-0.004, 0.001] | -1.395% [-3.485, 0.707]% |
| pressure_async_aligned_minus_async_naive | task 2 | episode p50 control-step seconds | 10 | 0.051 | 0.051 | 0.000 [-0.000, 0.001] | 0.914% [-0.376, 2.304]% |
| pressure_async_aligned_minus_async_naive | task 2 | episode p50 queue depth (steps) | 10 | 25.000 | 5.950 | -19.050 [-19.150, -19.000] | -76.200% [-76.600, -76.000]% |
| pressure_async_aligned_minus_async_naive | task 2 | episode p50 request headroom (steps) | 10 | 21.400 | 0.000 | -21.400 [-21.700, -21.100] | -100.000% [-100.000, -100.000]% |
| pressure_async_aligned_minus_async_naive | task 2 | queue-hold steps | 10 | 1.700 | 8.300 | 6.600 [5.400, 7.800] | 553.000% [376.000, 735.000]% |
| pressure_async_aligned_minus_async_naive | task 2 | fully stale chunks | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_aligned_minus_async_naive | task 2 | episode p50 delivery age (steps) | 10 | 21.012 | 20.786 | -0.225 [-0.492, 0.028] | -1.039% [-2.287, 0.144]% |
| pressure_async_aligned_minus_async_naive | task 2 | mean adjacent-action 7D L2 | 10 | 0.289 | 0.217 | -0.072 [-0.123, -0.025] | -29.216% [-49.861, -10.368]% |
| pressure_async_aligned_minus_async_naive | task 2 | mean incoming stale fraction | 10 | 0.000 | 0.627 | 0.627 [0.622, 0.631] | n/a |
| pressure_async_aligned_minus_async_naive | task 2 | accepted chunks | 10 | 53.500 | 9.800 | -43.700 [-56.100, -31.300] | -78.942% [-83.310, -74.494]% |
| pressure_async_aligned_minus_async_naive | task 2 | rejected chunks | 10 | 2.100 | 2.000 | -0.100 [-0.300, 0.000] | -3.333% [-10.000, 0.000]% |
| pressure_async_aligned_minus_async_naive | task 2 | chunks rejected after episode end | 10 | 2.100 | 2.000 | -0.100 [-0.300, 0.000] | -3.333% [-10.000, 0.000]% |
| pressure_async_aligned_minus_async_naive | task 2 | mean replacement position L2 | 10 | 0.054 | 0.021 | -0.033 [-0.040, -0.025] | -59.266% [-64.666, -53.157]% |
| pressure_async_aligned_minus_async_naive | task 2 | mean replacement rotation geodesic angle (rad) | 10 | 0.091 | 0.043 | -0.048 [-0.061, -0.034] | -50.241% [-60.609, -39.326]% |
| pressure_async_aligned_minus_async_naive | task 2 | replacement gripper switches | 10 | 13.500 | 0.100 | -13.400 [-20.900, -6.298] | -98.333% [-100.000, -95.000]% |
| pressure_async_aligned_minus_sync_hold | all | episode success | 30 | 1.000 | 0.967 | -0.033 [-0.100, 0.000] | -3.333% [-10.000, 0.000]% |
| pressure_async_aligned_minus_sync_hold | all | environment steps | 30 | 211.333 | 150.333 | -61.000 [-87.200, -12.533] | -27.917% [-40.142, -4.213]% |
| pressure_async_aligned_minus_sync_hold | all | wall-clock seconds | 30 | 13.673 | 10.370 | -3.304 [-4.589, -0.957] | -23.371% [-32.701, -5.531]% |
| pressure_async_aligned_minus_sync_hold | all | episode p50 observation-to-delivery seconds | 30 | 1.232 | 1.065 | -0.167 [-0.176, -0.158] | -13.508% [-14.134, -12.871]% |
| pressure_async_aligned_minus_sync_hold | all | episode p50 model-inference seconds | 30 | 0.269 | 0.099 | -0.170 [-0.179, -0.161] | -62.805% [-63.985, -61.564]% |
| pressure_async_aligned_minus_sync_hold | all | episode p50 control-step seconds | 30 | 0.051 | 0.051 | -0.000 [-0.000, -0.000] | -0.467% [-0.770, -0.157]% |
| pressure_async_aligned_minus_sync_hold | all | episode p50 queue depth (steps) | 30 | 6.133 | 5.367 | -0.767 [-1.050, -0.483] | -10.516% [-15.334, -5.240]% |
| pressure_async_aligned_minus_sync_hold | all | episode p50 request headroom (steps) | 30 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_aligned_minus_sync_hold | all | queue-hold steps | 30 | 84.867 | 16.367 | -68.500 [-75.833, -57.099] | -80.188% [-86.856, -67.779]% |
| pressure_async_aligned_minus_sync_hold | all | fully stale chunks | 30 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_aligned_minus_sync_hold | all | episode p50 delivery age (steps) | 30 | 24.128 | 20.959 | -3.169 [-3.344, -2.993] | -13.099% [-13.726, -12.465]% |
| pressure_async_aligned_minus_sync_hold | all | mean adjacent-action 7D L2 | 30 | 0.073 | 0.182 | 0.109 [0.077, 0.143] | 144.931% [109.179, 185.850]% |
| pressure_async_aligned_minus_sync_hold | all | mean incoming stale fraction | 30 | 0.000 | 0.644 | 0.644 [0.638, 0.651] | n/a |
| pressure_async_aligned_minus_sync_hold | all | accepted chunks | 30 | 4.500 | 13.367 | 8.867 [6.367, 13.567] | 203.667% [142.667, 320.500]% |
| pressure_async_aligned_minus_sync_hold | all | rejected chunks | 30 | 0.100 | 2.067 | 1.967 [1.833, 2.100] | n/a |
| pressure_async_aligned_minus_sync_hold | all | chunks rejected after episode end | 30 | 0.100 | 2.067 | 1.967 [1.833, 2.100] | n/a |
| pressure_async_aligned_minus_sync_hold | all | mean replacement position L2 | 30 | 0.013 | 0.025 | 0.012 [0.007, 0.022] | 90.812% [57.362, 147.929]% |
| pressure_async_aligned_minus_sync_hold | all | mean replacement rotation geodesic angle (rad) | 30 | 0.017 | 0.047 | 0.030 [0.025, 0.036] | 184.555% [150.476, 222.771]% |
| pressure_async_aligned_minus_sync_hold | all | replacement gripper switches | 30 | 0.067 | 1.867 | 1.800 [0.167, 4.900] | n/a |
| pressure_async_aligned_minus_sync_hold | task 0 | episode success | 10 | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| pressure_async_aligned_minus_sync_hold | task 0 | environment steps | 10 | 236.600 | 142.700 | -93.900 [-98.400, -89.600] | -39.668% [-41.368, -38.084]% |
| pressure_async_aligned_minus_sync_hold | task 0 | wall-clock seconds | 10 | 15.206 | 10.210 | -4.996 [-5.253, -4.756] | -32.883% [-34.566, -31.308]% |
| pressure_async_aligned_minus_sync_hold | task 0 | episode p50 observation-to-delivery seconds | 10 | 1.235 | 1.066 | -0.169 [-0.189, -0.151] | -13.670% [-15.003, -12.387]% |
| pressure_async_aligned_minus_sync_hold | task 0 | episode p50 model-inference seconds | 10 | 0.273 | 0.100 | -0.173 [-0.192, -0.155] | -63.022% [-65.455, -60.719]% |
| pressure_async_aligned_minus_sync_hold | task 0 | episode p50 control-step seconds | 10 | 0.051 | 0.051 | -0.000 [-0.001, -0.000] | -0.733% [-1.207, -0.257]% |
| pressure_async_aligned_minus_sync_hold | task 0 | episode p50 queue depth (steps) | 10 | 6.050 | 5.250 | -0.800 [-1.050, -0.550] | -12.949% [-16.795, -8.910]% |
| pressure_async_aligned_minus_sync_hold | task 0 | episode p50 request headroom (steps) | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_aligned_minus_sync_hold | task 0 | queue-hold steps | 10 | 95.500 | 13.200 | -82.300 [-83.400, -81.100] | -86.182% [-87.289, -84.939]% |
| pressure_async_aligned_minus_sync_hold | task 0 | fully stale chunks | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_aligned_minus_sync_hold | task 0 | episode p50 delivery age (steps) | 10 | 23.996 | 20.858 | -3.138 [-3.493, -2.798] | -13.032% [-14.296, -11.780]% |
| pressure_async_aligned_minus_sync_hold | task 0 | mean adjacent-action 7D L2 | 10 | 0.061 | 0.167 | 0.107 [0.049, 0.170] | 145.164% [90.812, 197.958]% |
| pressure_async_aligned_minus_sync_hold | task 0 | mean incoming stale fraction | 10 | 0.000 | 0.648 | 0.648 [0.645, 0.652] | n/a |
| pressure_async_aligned_minus_sync_hold | task 0 | accepted chunks | 10 | 5.000 | 12.600 | 7.600 [7.200, 8.000] | 152.000% [144.000, 160.000]% |
| pressure_async_aligned_minus_sync_hold | task 0 | rejected chunks | 10 | 0.000 | 2.100 | 2.100 [2.000, 2.300] | n/a |
| pressure_async_aligned_minus_sync_hold | task 0 | chunks rejected after episode end | 10 | 0.000 | 2.100 | 2.100 [2.000, 2.300] | n/a |
| pressure_async_aligned_minus_sync_hold | task 0 | mean replacement position L2 | 10 | 0.013 | 0.020 | 0.007 [0.005, 0.010] | 58.205% [40.071, 78.391]% |
| pressure_async_aligned_minus_sync_hold | task 0 | mean replacement rotation geodesic angle (rad) | 10 | 0.017 | 0.047 | 0.031 [0.024, 0.038] | 201.405% [147.809, 260.622]% |
| pressure_async_aligned_minus_sync_hold | task 0 | replacement gripper switches | 10 | 0.000 | 0.600 | 0.600 [0.300, 0.900] | n/a |
| pressure_async_aligned_minus_sync_hold | task 1 | episode success | 10 | 1.000 | 0.900 | -0.100 [-0.300, 0.000] | -10.000% [-30.000, 0.000]% |
| pressure_async_aligned_minus_sync_hold | task 1 | environment steps | 10 | 211.000 | 193.500 | -17.500 [-91.800, 122.800] | -5.671% [-41.617, 64.536]% |
| pressure_async_aligned_minus_sync_hold | task 1 | wall-clock seconds | 10 | 13.433 | 12.269 | -1.164 [-4.754, 5.578] | -6.938% [-34.095, 45.586]% |
| pressure_async_aligned_minus_sync_hold | task 1 | episode p50 observation-to-delivery seconds | 10 | 1.238 | 1.065 | -0.173 [-0.188, -0.156] | -13.935% [-15.014, -12.744]% |
| pressure_async_aligned_minus_sync_hold | task 1 | episode p50 model-inference seconds | 10 | 0.276 | 0.099 | -0.176 [-0.192, -0.160] | -63.638% [-65.748, -61.237]% |
| pressure_async_aligned_minus_sync_hold | task 1 | episode p50 control-step seconds | 10 | 0.050 | 0.050 | -0.000 [-0.000, -0.000] | -0.522% [-0.976, -0.082]% |
| pressure_async_aligned_minus_sync_hold | task 1 | episode p50 queue depth (steps) | 10 | 5.250 | 4.900 | -0.350 [-1.100, 0.350] | -2.457% [-15.952, 11.234]% |
| pressure_async_aligned_minus_sync_hold | task 1 | episode p50 request headroom (steps) | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_aligned_minus_sync_hold | task 1 | queue-hold steps | 10 | 87.200 | 27.600 | -59.600 [-78.700, -29.100] | -65.921% [-84.236, -30.915]% |
| pressure_async_aligned_minus_sync_hold | task 1 | fully stale chunks | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_aligned_minus_sync_hold | task 1 | episode p50 delivery age (steps) | 10 | 24.552 | 21.234 | -3.318 [-3.605, -3.028] | -13.484% [-14.513, -12.436]% |
| pressure_async_aligned_minus_sync_hold | task 1 | mean adjacent-action 7D L2 | 10 | 0.063 | 0.161 | 0.098 [0.056, 0.147] | 165.535% [93.176, 262.619]% |
| pressure_async_aligned_minus_sync_hold | task 1 | mean incoming stale fraction | 10 | 0.000 | 0.657 | 0.657 [0.648, 0.672] | n/a |
| pressure_async_aligned_minus_sync_hold | task 1 | accepted chunks | 10 | 4.500 | 17.700 | 13.200 [6.200, 26.800] | 314.000% [134.000, 659.500]% |
| pressure_async_aligned_minus_sync_hold | task 1 | rejected chunks | 10 | 0.300 | 2.100 | 1.800 [1.400, 2.200] | n/a |
| pressure_async_aligned_minus_sync_hold | task 1 | chunks rejected after episode end | 10 | 0.300 | 2.100 | 1.800 [1.400, 2.200] | n/a |
| pressure_async_aligned_minus_sync_hold | task 1 | mean replacement position L2 | 10 | 0.013 | 0.036 | 0.022 [0.008, 0.049] | 150.156% [61.157, 304.487]% |
| pressure_async_aligned_minus_sync_hold | task 1 | mean replacement rotation geodesic angle (rad) | 10 | 0.017 | 0.052 | 0.035 [0.025, 0.049] | 202.643% [149.428, 271.373]% |
| pressure_async_aligned_minus_sync_hold | task 1 | replacement gripper switches | 10 | 0.200 | 4.900 | 4.700 [-0.100, 13.800] | n/a |
| pressure_async_aligned_minus_sync_hold | task 2 | episode success | 10 | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000% [0.000, 0.000]% |
| pressure_async_aligned_minus_sync_hold | task 2 | environment steps | 10 | 186.400 | 114.800 | -71.600 [-73.300, -69.800] | -38.413% [-39.301, -37.550]% |
| pressure_async_aligned_minus_sync_hold | task 2 | wall-clock seconds | 10 | 12.381 | 8.629 | -3.751 [-3.860, -3.650] | -30.293% [-31.004, -29.647]% |
| pressure_async_aligned_minus_sync_hold | task 2 | episode p50 observation-to-delivery seconds | 10 | 1.222 | 1.064 | -0.158 [-0.167, -0.149] | -12.920% [-13.527, -12.280]% |
| pressure_async_aligned_minus_sync_hold | task 2 | episode p50 model-inference seconds | 10 | 0.259 | 0.099 | -0.160 [-0.170, -0.151] | -61.756% [-63.127, -60.242]% |
| pressure_async_aligned_minus_sync_hold | task 2 | episode p50 control-step seconds | 10 | 0.051 | 0.051 | -0.000 [-0.000, 0.000] | -0.147% [-0.756, 0.440]% |
| pressure_async_aligned_minus_sync_hold | task 2 | episode p50 queue depth (steps) | 10 | 7.100 | 5.950 | -1.150 [-1.300, -1.000] | -16.143% [-18.000, -14.286]% |
| pressure_async_aligned_minus_sync_hold | task 2 | episode p50 request headroom (steps) | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_aligned_minus_sync_hold | task 2 | queue-hold steps | 10 | 71.900 | 8.300 | -63.600 [-64.900, -62.300] | -88.462% [-90.103, -86.992]% |
| pressure_async_aligned_minus_sync_hold | task 2 | fully stale chunks | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_aligned_minus_sync_hold | task 2 | episode p50 delivery age (steps) | 10 | 23.837 | 20.786 | -3.051 [-3.289, -2.815] | -12.782% [-13.673, -11.882]% |
| pressure_async_aligned_minus_sync_hold | task 2 | mean adjacent-action 7D L2 | 10 | 0.096 | 0.217 | 0.121 [0.063, 0.185] | 124.093% [74.459, 173.584]% |
| pressure_async_aligned_minus_sync_hold | task 2 | mean incoming stale fraction | 10 | 0.000 | 0.627 | 0.627 [0.622, 0.631] | n/a |
| pressure_async_aligned_minus_sync_hold | task 2 | accepted chunks | 10 | 4.000 | 9.800 | 5.800 [5.500, 6.000] | 145.000% [137.500, 150.000]% |
| pressure_async_aligned_minus_sync_hold | task 2 | rejected chunks | 10 | 0.000 | 2.000 | 2.000 [2.000, 2.000] | n/a |
| pressure_async_aligned_minus_sync_hold | task 2 | chunks rejected after episode end | 10 | 0.000 | 2.000 | 2.000 [2.000, 2.000] | n/a |
| pressure_async_aligned_minus_sync_hold | task 2 | mean replacement position L2 | 10 | 0.013 | 0.021 | 0.008 [0.006, 0.009] | 64.075% [47.965, 81.062]% |
| pressure_async_aligned_minus_sync_hold | task 2 | mean replacement rotation geodesic angle (rad) | 10 | 0.018 | 0.043 | 0.024 [0.017, 0.032] | 149.617% [93.476, 213.524]% |
| pressure_async_aligned_minus_sync_hold | task 2 | replacement gripper switches | 10 | 0.000 | 0.100 | 0.100 [0.000, 0.300] | n/a |
| pressure_async_naive_minus_sync_hold | all | episode success | 30 | 1.000 | 0.633 | -0.367 [-0.533, -0.200] | -36.667% [-53.333, -20.000]% |
| pressure_async_naive_minus_sync_hold | all | environment steps | 30 | 211.333 | 550.233 | 338.900 [271.398, 409.067] | 161.535% [129.132, 195.924]% |
| pressure_async_naive_minus_sync_hold | all | wall-clock seconds | 30 | 13.673 | 30.296 | 16.622 [13.557, 19.776] | 122.736% [99.883, 146.825]% |
| pressure_async_naive_minus_sync_hold | all | episode p50 observation-to-delivery seconds | 30 | 1.232 | 1.070 | -0.161 [-0.170, -0.152] | -13.070% [-13.686, -12.439]% |
| pressure_async_naive_minus_sync_hold | all | episode p50 model-inference seconds | 30 | 0.269 | 0.105 | -0.164 [-0.173, -0.155] | -60.655% [-61.981, -59.231]% |
| pressure_async_naive_minus_sync_hold | all | episode p50 control-step seconds | 30 | 0.051 | 0.051 | -0.000 [-0.001, -0.000] | -0.623% [-1.198, -0.099]% |
| pressure_async_naive_minus_sync_hold | all | episode p50 queue depth (steps) | 30 | 6.133 | 25.000 | 18.867 [18.517, 19.233] | 321.272% [292.949, 352.327]% |
| pressure_async_naive_minus_sync_hold | all | episode p50 request headroom (steps) | 30 | 0.000 | 21.333 | 21.333 [21.167, 21.500] | n/a |
| pressure_async_naive_minus_sync_hold | all | queue-hold steps | 30 | 84.867 | 1.500 | -83.367 [-87.533, -79.067] | -98.191% [-98.555, -97.724]% |
| pressure_async_naive_minus_sync_hold | all | fully stale chunks | 30 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_naive_minus_sync_hold | all | episode p50 delivery age (steps) | 30 | 24.128 | 21.102 | -3.026 [-3.251, -2.783] | -12.497% [-13.356, -11.562]% |
| pressure_async_naive_minus_sync_hold | all | mean adjacent-action 7D L2 | 30 | 0.073 | 0.292 | 0.218 [0.182, 0.258] | 443.898% [311.425, 610.603]% |
| pressure_async_naive_minus_sync_hold | all | mean incoming stale fraction | 30 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_naive_minus_sync_hold | all | accepted chunks | 30 | 4.500 | 53.167 | 48.667 [41.867, 55.733] | 1091.000% [936.329, 1256.333]% |
| pressure_async_naive_minus_sync_hold | all | rejected chunks | 30 | 0.100 | 2.233 | 2.133 [1.933, 2.333] | n/a |
| pressure_async_naive_minus_sync_hold | all | chunks rejected after episode end | 30 | 0.100 | 2.233 | 2.133 [1.933, 2.333] | n/a |
| pressure_async_naive_minus_sync_hold | all | mean replacement position L2 | 30 | 0.013 | 0.058 | 0.045 [0.041, 0.049] | 355.790% [315.901, 395.983]% |
| pressure_async_naive_minus_sync_hold | all | mean replacement rotation geodesic angle (rad) | 30 | 0.017 | 0.082 | 0.065 [0.058, 0.072] | 392.344% [341.177, 443.963]% |
| pressure_async_naive_minus_sync_hold | all | replacement gripper switches | 30 | 0.067 | 11.633 | 11.567 [7.866, 15.467] | n/a |
| pressure_async_naive_minus_sync_hold | task 0 | episode success | 10 | 1.000 | 0.500 | -0.500 [-0.800, -0.200] | -50.000% [-80.000, -20.000]% |
| pressure_async_naive_minus_sync_hold | task 0 | environment steps | 10 | 236.600 | 624.000 | 387.400 [272.397, 501.500] | 163.338% [115.391, 210.588]% |
| pressure_async_naive_minus_sync_hold | task 0 | wall-clock seconds | 10 | 15.206 | 33.905 | 18.699 [13.640, 23.762] | 124.168% [89.920, 158.381]% |
| pressure_async_naive_minus_sync_hold | task 0 | episode p50 observation-to-delivery seconds | 10 | 1.235 | 1.075 | -0.160 [-0.180, -0.141] | -12.913% [-14.283, -11.584]% |
| pressure_async_naive_minus_sync_hold | task 0 | episode p50 model-inference seconds | 10 | 0.273 | 0.110 | -0.162 [-0.182, -0.144] | -59.021% [-61.865, -56.187]% |
| pressure_async_naive_minus_sync_hold | task 0 | episode p50 control-step seconds | 10 | 0.051 | 0.051 | -0.001 [-0.001, -0.000] | -1.150% [-2.046, -0.236]% |
| pressure_async_naive_minus_sync_hold | task 0 | episode p50 queue depth (steps) | 10 | 6.050 | 25.000 | 18.950 [18.700, 19.200] | 315.385% [297.436, 335.256]% |
| pressure_async_naive_minus_sync_hold | task 0 | episode p50 request headroom (steps) | 10 | 0.000 | 21.500 | 21.500 [21.200, 21.800] | n/a |
| pressure_async_naive_minus_sync_hold | task 0 | queue-hold steps | 10 | 95.500 | 1.500 | -94.000 [-94.400, -93.600] | -98.435% [-98.843, -98.023]% |
| pressure_async_naive_minus_sync_hold | task 0 | fully stale chunks | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_naive_minus_sync_hold | task 0 | episode p50 delivery age (steps) | 10 | 23.996 | 21.134 | -2.862 [-3.314, -2.372] | -11.870% [-13.629, -9.973]% |
| pressure_async_naive_minus_sync_hold | task 0 | mean adjacent-action 7D L2 | 10 | 0.061 | 0.362 | 0.301 [0.228, 0.383] | 712.729% [411.916, 1102.942]% |
| pressure_async_naive_minus_sync_hold | task 0 | mean incoming stale fraction | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_naive_minus_sync_hold | task 0 | accepted chunks | 10 | 5.000 | 60.400 | 55.400 [43.800, 67.000] | 1108.000% [876.000, 1340.000]% |
| pressure_async_naive_minus_sync_hold | task 0 | rejected chunks | 10 | 0.000 | 2.300 | 2.300 [2.000, 2.600] | n/a |
| pressure_async_naive_minus_sync_hold | task 0 | chunks rejected after episode end | 10 | 0.000 | 2.300 | 2.300 [2.000, 2.600] | n/a |
| pressure_async_naive_minus_sync_hold | task 0 | mean replacement position L2 | 10 | 0.013 | 0.059 | 0.046 [0.040, 0.053] | 369.835% [313.748, 424.744]% |
| pressure_async_naive_minus_sync_hold | task 0 | mean replacement rotation geodesic angle (rad) | 10 | 0.017 | 0.063 | 0.047 [0.037, 0.057] | 313.752% [221.170, 416.093]% |
| pressure_async_naive_minus_sync_hold | task 0 | replacement gripper switches | 10 | 0.000 | 13.500 | 13.500 [7.200, 19.600] | n/a |
| pressure_async_naive_minus_sync_hold | task 1 | episode success | 10 | 1.000 | 0.800 | -0.200 [-0.500, 0.000] | -20.000% [-50.000, 0.000]% |
| pressure_async_naive_minus_sync_hold | task 1 | environment steps | 10 | 211.000 | 474.100 | 263.100 [173.097, 376.802] | 125.278% [82.862, 177.984]% |
| pressure_async_naive_minus_sync_hold | task 1 | wall-clock seconds | 10 | 13.433 | 26.755 | 13.322 [9.090, 18.606] | 99.990% [67.982, 139.579]% |
| pressure_async_naive_minus_sync_hold | task 1 | episode p50 observation-to-delivery seconds | 10 | 1.238 | 1.070 | -0.168 [-0.183, -0.152] | -13.531% [-14.577, -12.423]% |
| pressure_async_naive_minus_sync_hold | task 1 | episode p50 model-inference seconds | 10 | 0.276 | 0.105 | -0.171 [-0.187, -0.156] | -61.743% [-63.978, -59.492]% |
| pressure_async_naive_minus_sync_hold | task 1 | episode p50 control-step seconds | 10 | 0.050 | 0.051 | 0.000 [-0.000, 0.000] | 0.299% [-0.196, 0.822]% |
| pressure_async_naive_minus_sync_hold | task 1 | episode p50 queue depth (steps) | 10 | 5.250 | 25.000 | 19.750 [19.050, 20.400] | 396.050% [336.521, 457.585]% |
| pressure_async_naive_minus_sync_hold | task 1 | episode p50 request headroom (steps) | 10 | 0.000 | 21.100 | 21.100 [21.000, 21.300] | n/a |
| pressure_async_naive_minus_sync_hold | task 1 | queue-hold steps | 10 | 87.200 | 1.300 | -85.900 [-92.500, -79.100] | -98.485% [-98.808, -98.111]% |
| pressure_async_naive_minus_sync_hold | task 1 | fully stale chunks | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_naive_minus_sync_hold | task 1 | episode p50 delivery age (steps) | 10 | 24.552 | 21.160 | -3.392 [-3.666, -3.083] | -13.786% [-14.775, -12.640]% |
| pressure_async_naive_minus_sync_hold | task 1 | mean adjacent-action 7D L2 | 10 | 0.063 | 0.224 | 0.161 [0.129, 0.200] | 326.298% [221.659, 433.129]% |
| pressure_async_naive_minus_sync_hold | task 1 | mean incoming stale fraction | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_naive_minus_sync_hold | task 1 | accepted chunks | 10 | 4.500 | 45.600 | 41.100 [32.100, 52.600] | 927.500% [713.987, 1194.013]% |
| pressure_async_naive_minus_sync_hold | task 1 | rejected chunks | 10 | 0.300 | 2.300 | 2.000 [1.500, 2.500] | n/a |
| pressure_async_naive_minus_sync_hold | task 1 | chunks rejected after episode end | 10 | 0.300 | 2.300 | 2.000 [1.500, 2.500] | n/a |
| pressure_async_naive_minus_sync_hold | task 1 | mean replacement position L2 | 10 | 0.013 | 0.061 | 0.048 [0.042, 0.053] | 367.385% [309.460, 426.809]% |
| pressure_async_naive_minus_sync_hold | task 1 | mean replacement rotation geodesic angle (rad) | 10 | 0.017 | 0.092 | 0.075 [0.068, 0.083] | 453.044% [388.039, 520.802]% |
| pressure_async_naive_minus_sync_hold | task 1 | replacement gripper switches | 10 | 0.200 | 7.900 | 7.700 [2.900, 13.700] | n/a |
| pressure_async_naive_minus_sync_hold | task 2 | episode success | 10 | 1.000 | 0.600 | -0.400 [-0.700, -0.100] | -40.000% [-70.000, -10.000]% |
| pressure_async_naive_minus_sync_hold | task 2 | environment steps | 10 | 186.400 | 552.600 | 366.200 [240.100, 491.600] | 195.989% [128.975, 263.529]% |
| pressure_async_naive_minus_sync_hold | task 2 | wall-clock seconds | 10 | 12.381 | 30.226 | 17.846 [12.220, 23.463] | 144.051% [98.487, 189.758]% |
| pressure_async_naive_minus_sync_hold | task 2 | episode p50 observation-to-delivery seconds | 10 | 1.222 | 1.066 | -0.156 [-0.165, -0.148] | -12.766% [-13.391, -12.166]% |
| pressure_async_naive_minus_sync_hold | task 2 | episode p50 model-inference seconds | 10 | 0.259 | 0.100 | -0.159 [-0.168, -0.150] | -61.199% [-62.615, -59.940]% |
| pressure_async_naive_minus_sync_hold | task 2 | episode p50 control-step seconds | 10 | 0.051 | 0.051 | -0.001 [-0.001, 0.000] | -1.019% [-2.103, 0.010]% |
| pressure_async_naive_minus_sync_hold | task 2 | episode p50 queue depth (steps) | 10 | 7.100 | 25.000 | 17.900 [17.750, 18.000] | 252.381% [245.238, 257.143]% |
| pressure_async_naive_minus_sync_hold | task 2 | episode p50 request headroom (steps) | 10 | 0.000 | 21.400 | 21.400 [21.100, 21.700] | n/a |
| pressure_async_naive_minus_sync_hold | task 2 | queue-hold steps | 10 | 71.900 | 1.700 | -70.200 [-71.100, -69.200] | -97.653% [-98.595, -96.458]% |
| pressure_async_naive_minus_sync_hold | task 2 | fully stale chunks | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_naive_minus_sync_hold | task 2 | episode p50 delivery age (steps) | 10 | 23.837 | 21.012 | -2.826 [-3.131, -2.492] | -11.836% [-13.065, -10.491]% |
| pressure_async_naive_minus_sync_hold | task 2 | mean adjacent-action 7D L2 | 10 | 0.096 | 0.289 | 0.193 [0.151, 0.238] | 292.667% [177.788, 455.006]% |
| pressure_async_naive_minus_sync_hold | task 2 | mean incoming stale fraction | 10 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | n/a |
| pressure_async_naive_minus_sync_hold | task 2 | accepted chunks | 10 | 4.000 | 53.500 | 49.500 [37.000, 62.100] | 1237.500% [925.000, 1552.500]% |
| pressure_async_naive_minus_sync_hold | task 2 | rejected chunks | 10 | 0.000 | 2.100 | 2.100 [2.000, 2.300] | n/a |
| pressure_async_naive_minus_sync_hold | task 2 | chunks rejected after episode end | 10 | 0.000 | 2.100 | 2.100 [2.000, 2.300] | n/a |
| pressure_async_naive_minus_sync_hold | task 2 | mean replacement position L2 | 10 | 0.013 | 0.054 | 0.041 [0.032, 0.049] | 330.148% [246.337, 415.638]% |
| pressure_async_naive_minus_sync_hold | task 2 | mean replacement rotation geodesic angle (rad) | 10 | 0.018 | 0.091 | 0.072 [0.061, 0.083] | 410.236% [337.688, 493.211]% |
| pressure_async_naive_minus_sync_hold | task 2 | replacement gripper switches | 10 | 0.000 | 13.500 | 13.500 [6.400, 20.900] | n/a |

## 6. Supported, unsupported, and still-untested claims

- Failures occurred; use the paired success effects above rather than assuming equal robustness.
- Success over async_naive: **supported**; aligned minus naive = 0.333 [0.133, 0.533].
- Success heterogeneity (aligned minus naive): task 0 0.500 [0.200, 0.800], task 1 0.100 [-0.200, 0.400], task 2 0.400 [0.100, 0.700].
- Success over sync_hold: **not supported**; aligned minus sync_hold = -0.033 [-0.100, 0.000].
- Underrun reduction across both comparators: **not supported as stated; mixed by comparator**. Aligned minus async_naive holds = 14.867 [9.533, 24.400]; aligned minus sync_hold holds = -68.500 [-75.833, -57.099]. Neither async mode produced a fully stale chunk.
- Step/wall-time efficiency over async_naive: **supported**.
- Fully stale-chunk rejection remains untested at every M4 delay.
- Decision rule: Supported requires a favorable paired mean difference whose 95% paired bootstrap confidence interval excludes zero. Directional-only means the mean is favorable but its interval includes zero.
- Queue-pressure and ordinary-latency evidence are reported separately.

### Limitations

- The 100% success rates in M3 and the ordinary-latency sync_hold runs are ceilings, not evidence that those modes are equally robust. The selected pressure matrix is not ceilinged.
- Historical sync is a blocking evaluator baseline; sync_hold advances real-time simulator steps with repeated finite commands during inference.
- Model inference latency, inference queue wait, and injected delivery delay are distinct; observation-to-delivery includes all three.
- Ordinary 0/200 ms results and the selected queue-pressure results answer different operating-regime questions.
- M3 has no explicit replacement boundary indices; its component jump metrics remain unavailable. M4 component metrics use directly logged events.
- Calibration uses task 3 and three fixed states, while final pressure effects use 30 paired episodes over tasks 0-2.
- Pressure rows record the selected delay but not the selection-file hash; ordering provenance relies on the frozen selection flag and runner contract.
- Bootstrap intervals use episodes, not actions or replacement events, as independent units.
- Primary episode step/time effects include 11 naive and one aligned 800-step timeout endpoints. In a post-hoc sensitivity restricted to the 18 pairs where both modes succeeded, aligned still used 274.8 fewer steps and 14.39 fewer seconds on average; this conditioned subset is diagnostic, not the primary estimate.

### Suggested next experiment

Repeat the frozen 950 ms protocol on additional paired tasks and states to test generalization and diagnose the aligned task-1 state-16 failure. A separately preregistered higher-delay or smaller-headroom study would be needed to exercise fully stale-chunk rejection.
