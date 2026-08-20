# ActionStream LeRobot GPU benchmark v1

Validated 435 episodes, 435 traces, 135 videos, and 75043 final 7D actions.

## Gate status

- xvla_sync_zero_delay_three_suite_gate: `True`
- xvla_sync_success_suites: `['libero_goal', 'libero_object', 'libero_spatial']`
- xvla_disconnect_recovery_gate: `True`
- xvla_disconnect_rows: `6`
- smolvla_sync_zero_delay_gate: `False`
- smolvla_rtc_completion_gate: `True`
- smolvla_upstream_rtc_capability_gate: `True`

## Holdout coverage

The report contains 375 holdout condition-episodes across 1 model(s) and 5 runtime label(s).
Peak allocated CUDA memory was 12259.0 MiB during scored episodes and 12250.8 MiB during non-scored cold warmup.

See `main_table.csv`, `paired_effects.csv`, `failure_taxonomy.csv`, and `latency_success_operating_points.png` for the auditable results. The separately labeled `secondary_paired_effects.csv` contrasts aligned with official weighted-average Async.

## Holdout main table

| Model | Profile | Runtime | Success | Mean steps | GPU actions/s | Peak CUDA MiB | Model infer p50/p95 ms | Delivery p50/p95 ms | Queue age p50/p95 steps | Discard | Depletion | Fallback |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xvla | burst_0250_to2000_holdout | actionstream_backend_aligned | 13/15 (86.7%) | 186.07 | 169.86 | 12259.0 | 176.6/2025.7 | 465.1/3273.6 | 14.00/25.10 | 2393.00 | 731.00 | 0.00 |
| xvla | fixed_0000 | actionstream_backend_aligned | 12/15 (80.0%) | 184.93 | 351.38 | 12259.0 | 85.4/126.4 | 94.3/139.7 | 3.00/5.00 | 2735.00 | 0.00 | 0.00 |
| xvla | fixed_0250 | actionstream_backend_aligned | 13/15 (86.7%) | 149.00 | 181.77 | 12259.0 | 165.0/1268.5 | 431.6/1538.0 | 13.00/18.00 | 2142.00 | 0.00 | 0.00 |
| xvla | fixed_0950 | actionstream_backend_aligned | 10/15 (66.7%) | 206.80 | 165.40 | 12259.0 | 181.4/3023.9 | 1147.7/3986.9 | 25.00/29.20 | 2942.00 | 1628.00 | 0.00 |
| xvla | jitter_0600_pm0400_holdout | actionstream_backend_aligned | 14/15 (93.3%) | 144.87 | 172.52 | 12259.0 | 173.9/2777.4 | 844.6/3608.2 | 21.00/29.00 | 2026.00 | 344.00 | 0.00 |
| xvla | burst_0250_to2000_holdout | actionstream_backend_guarded | 8/15 (53.3%) | 232.60 | 163.60 | 12259.0 | 183.4/303.7 | 469.4/2180.4 | 15.00/56.05 | 2133.00 | 454.00 | 31.00 |
| xvla | fixed_0000 | actionstream_backend_guarded | 12/15 (80.0%) | 184.07 | 349.50 | 12259.0 | 85.8/127.2 | 94.3/139.7 | 3.00/5.00 | 2715.00 | 0.00 | 0.00 |
| xvla | fixed_0250 | actionstream_backend_guarded | 13/15 (86.7%) | 148.67 | 177.61 | 12259.0 | 168.9/1264.2 | 434.0/1529.2 | 13.00/18.00 | 2143.00 | 0.00 | 0.00 |
| xvla | fixed_0950 | actionstream_backend_guarded | 8/15 (53.3%) | 231.80 | 170.41 | 12259.0 | 176.0/2778.3 | 1143.5/3742.5 | 25.00/29.65 | 3316.00 | 1876.00 | 0.00 |
| xvla | jitter_0600_pm0400_holdout | actionstream_backend_guarded | 12/15 (80.0%) | 168.33 | 172.33 | 12259.0 | 174.1/2520.8 | 856.8/3161.4 | 21.00/29.00 | 2413.00 | 415.00 | 0.00 |
| xvla | burst_0250_to2000_holdout | lerobot_latest_only | 12/15 (80.0%) | 180.40 | 326.32 | 12259.0 | 91.9/158.1 | 392.1/1867.6 | N/A/N/A | 5833.00 | N/A | N/A |
| xvla | fixed_0000 | lerobot_latest_only | 11/15 (73.3%) | 171.47 | 245.20 | 12259.0 | 122.4/180.8 | 166.5/294.2 | N/A/N/A | 1614.00 | N/A | N/A |
| xvla | fixed_0250 | lerobot_latest_only | 13/15 (86.7%) | 147.87 | 306.18 | 12259.0 | 98.0/170.8 | 394.8/478.0 | N/A/N/A | 3408.00 | N/A | N/A |
| xvla | fixed_0950 | lerobot_latest_only | 14/15 (93.3%) | 165.27 | 346.52 | 12259.0 | 86.6/132.9 | 1072.9/1130.5 | N/A/N/A | 13857.00 | N/A | N/A |
| xvla | jitter_0600_pm0400_holdout | lerobot_latest_only | 14/15 (93.3%) | 144.33 | 341.27 | 12259.0 | 87.9/147.9 | 749.4/1113.2 | N/A/N/A | 9242.00 | N/A | N/A |
| xvla | burst_0250_to2000_holdout | lerobot_weighted_average | 9/15 (60.0%) | 208.80 | 312.07 | 12259.0 | 96.1/158.5 | 394.9/1929.7 | N/A/N/A | 6563.00 | N/A | N/A |
| xvla | fixed_0000 | lerobot_weighted_average | 10/15 (66.7%) | 189.00 | 250.54 | 12259.0 | 119.7/162.7 | 164.4/292.0 | N/A/N/A | 1762.00 | N/A | N/A |
| xvla | fixed_0250 | lerobot_weighted_average | 9/15 (60.0%) | 194.60 | 298.42 | 12259.0 | 100.5/156.9 | 382.6/451.4 | N/A/N/A | 4576.00 | N/A | N/A |
| xvla | fixed_0950 | lerobot_weighted_average | 11/15 (73.3%) | 187.00 | 345.56 | 12259.0 | 86.8/124.4 | 1071.9/1118.3 | N/A/N/A | 16705.00 | N/A | N/A |
| xvla | jitter_0600_pm0400_holdout | lerobot_weighted_average | 8/15 (53.3%) | 207.20 | 328.86 | 12259.0 | 91.2/152.2 | 760.6/1095.6 | N/A/N/A | 15623.00 | N/A | N/A |
| xvla | burst_0250_to2000_holdout | sync_hold | 14/15 (93.3%) | 130.13 | 329.12 | 3529.0 | 91.2/102.2 | 315.1/416.5 | N/A/N/A | 0.00 | N/A | N/A |
| xvla | fixed_0000 | sync_hold | 14/15 (93.3%) | 130.13 | 332.09 | 3520.8 | 90.3/104.2 | 99.5/112.6 | N/A/N/A | 0.00 | N/A | N/A |
| xvla | fixed_0250 | sync_hold | 14/15 (93.3%) | 130.13 | 326.00 | 3529.0 | 92.0/103.3 | 349.6/361.2 | N/A/N/A | 0.00 | N/A | N/A |
| xvla | fixed_0950 | sync_hold | 14/15 (93.3%) | 130.13 | 325.19 | 3529.0 | 92.3/103.0 | 1050.2/1060.8 | N/A/N/A | 0.00 | N/A | N/A |
| xvla | jitter_0600_pm0400_holdout | sync_hold | 14/15 (93.3%) | 130.13 | 326.22 | 3529.0 | 92.0/101.7 | 852.1/941.7 | N/A/N/A | 0.00 | N/A | N/A |

## Paired effects versus official latest-only

Positive success differences favor the estimate runtime; negative step differences are faster.

| Model | Profile | Runtime | Metric | Paired n | Mean difference | 95% bootstrap CI |
|---|---|---|---|---:|---:|---:|
| xvla | burst_0250_to2000_holdout | actionstream_backend_aligned | success | 15 | 0.0667 | [0.0000, 0.2000] |
| xvla | burst_0250_to2000_holdout | actionstream_backend_aligned | environment_steps | 15 | 5.67 | [-21.13, 27.00] |
| xvla | burst_0250_to2000_holdout | actionstream_backend_guarded | success | 15 | -0.2667 | [-0.5333, 0.0000] |
| xvla | burst_0250_to2000_holdout | actionstream_backend_guarded | environment_steps | 15 | 52.20 | [12.13, 90.27] |
| xvla | burst_0250_to2000_holdout | lerobot_weighted_average | success | 15 | -0.2000 | [-0.4667, 0.0667] |
| xvla | burst_0250_to2000_holdout | lerobot_weighted_average | environment_steps | 15 | 28.40 | [-11.13, 70.93] |
| xvla | burst_0250_to2000_holdout | sync_hold | success | 15 | 0.1333 | [0.0000, 0.3333] |
| xvla | burst_0250_to2000_holdout | sync_hold | environment_steps | 15 | -50.27 | [-82.27, -25.27] |
| xvla | fixed_0000 | actionstream_backend_aligned | success | 15 | 0.0667 | [-0.1333, 0.2667] |
| xvla | fixed_0000 | actionstream_backend_aligned | environment_steps | 15 | 13.47 | [-31.27, 52.67] |
| xvla | fixed_0000 | actionstream_backend_guarded | success | 15 | 0.0667 | [-0.1333, 0.2667] |
| xvla | fixed_0000 | actionstream_backend_guarded | environment_steps | 15 | 12.60 | [-32.07, 51.67] |
| xvla | fixed_0000 | lerobot_weighted_average | success | 15 | -0.0667 | [-0.2000, 0.0000] |
| xvla | fixed_0000 | lerobot_weighted_average | environment_steps | 15 | 17.53 | [1.07, 43.53] |
| xvla | fixed_0000 | sync_hold | success | 15 | 0.2000 | [0.0000, 0.4000] |
| xvla | fixed_0000 | sync_hold | environment_steps | 15 | -41.33 | [-83.53, -3.60] |
| xvla | fixed_0250 | actionstream_backend_aligned | success | 15 | 0.0000 | [-0.2000, 0.2000] |
| xvla | fixed_0250 | actionstream_backend_aligned | environment_steps | 15 | 1.13 | [-34.40, 31.67] |
| xvla | fixed_0250 | actionstream_backend_guarded | success | 15 | 0.0000 | [-0.2000, 0.2000] |
| xvla | fixed_0250 | actionstream_backend_guarded | environment_steps | 15 | 0.80 | [-34.87, 31.33] |
| xvla | fixed_0250 | lerobot_weighted_average | success | 15 | -0.2667 | [-0.4667, -0.0667] |
| xvla | fixed_0250 | lerobot_weighted_average | environment_steps | 15 | 46.73 | [11.00, 86.20] |
| xvla | fixed_0250 | sync_hold | success | 15 | 0.0667 | [-0.1333, 0.2667] |
| xvla | fixed_0250 | sync_hold | environment_steps | 15 | -17.73 | [-58.67, 17.07] |
| xvla | fixed_0950 | actionstream_backend_aligned | success | 15 | -0.2667 | [-0.5333, -0.0667] |
| xvla | fixed_0950 | actionstream_backend_aligned | environment_steps | 15 | 41.53 | [3.40, 84.33] |
| xvla | fixed_0950 | actionstream_backend_guarded | success | 15 | -0.4000 | [-0.6667, -0.1333] |
| xvla | fixed_0950 | actionstream_backend_guarded | environment_steps | 15 | 66.53 | [23.87, 111.40] |
| xvla | fixed_0950 | lerobot_weighted_average | success | 15 | -0.2000 | [-0.4000, 0.0000] |
| xvla | fixed_0950 | lerobot_weighted_average | environment_steps | 15 | 21.73 | [-0.20, 48.73] |
| xvla | fixed_0950 | sync_hold | success | 15 | 0.0000 | [0.0000, 0.0000] |
| xvla | fixed_0950 | sync_hold | environment_steps | 15 | -35.13 | [-41.67, -27.87] |
| xvla | jitter_0600_pm0400_holdout | actionstream_backend_aligned | success | 15 | 0.0000 | [0.0000, 0.0000] |
| xvla | jitter_0600_pm0400_holdout | actionstream_backend_aligned | environment_steps | 15 | 0.53 | [-6.00, 8.33] |
| xvla | jitter_0600_pm0400_holdout | actionstream_backend_guarded | success | 15 | -0.1333 | [-0.3333, 0.0000] |
| xvla | jitter_0600_pm0400_holdout | actionstream_backend_guarded | environment_steps | 15 | 24.00 | [-3.87, 63.40] |
| xvla | jitter_0600_pm0400_holdout | lerobot_weighted_average | success | 15 | -0.4000 | [-0.6667, -0.1333] |
| xvla | jitter_0600_pm0400_holdout | lerobot_weighted_average | environment_steps | 15 | 62.87 | [22.00, 104.60] |
| xvla | jitter_0600_pm0400_holdout | sync_hold | success | 15 | 0.0000 | [0.0000, 0.0000] |
| xvla | jitter_0600_pm0400_holdout | sync_hold | environment_steps | 15 | -14.20 | [-20.53, -8.07] |

## Secondary paired contrast versus official weighted-average Async

This contrast uses the same frozen X-VLA holdout cells and pairing invariants. The official Async baseline was included in the frozen protocol, but latest-only remains the primary reference. This secondary table was added after the holdout and must not be presented as a comparator switch or universal superiority claim.

| Profile | Async success | Aligned success | Success difference [95% CI] | Async mean steps | Aligned mean steps | Step difference [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| burst_0250_to2000_holdout | 9/15 | 13/15 | 0.2667 [0.0000, 0.5333] | 208.80 | 186.07 | -22.73 [-66.73, 19.53] |
| fixed_0000 | 10/15 | 12/15 | 0.1333 [-0.1333, 0.4000] | 189.00 | 184.93 | -4.07 [-50.40, 38.67] |
| fixed_0250 | 9/15 | 13/15 | 0.2667 [0.0000, 0.5333] | 194.60 | 149.00 | -45.60 [-92.60, -1.20] |
| fixed_0950 | 11/15 | 10/15 | -0.0667 [-0.4000, 0.2667] | 187.00 | 206.80 | 19.80 [-32.87, 75.53] |
| jitter_0600_pm0400_holdout | 8/15 | 14/15 | 0.4000 [0.1333, 0.6667] | 207.20 | 144.87 | -62.33 [-105.20, -21.53] |

Queue-age cells unsupported by upstream queues are blank rather than zero. Disconnect/recovery is a backend-only canary and is not treated as a fair official-baseline effect. This benchmark is LIBERO simulation evidence, not real-robot safety evidence.
