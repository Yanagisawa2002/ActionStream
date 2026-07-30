# ActionStream M3 Results

All six conditions reached 30/30 success. The binary success metric is therefore ceilinged; paired step and wall-time reductions are the informative outcomes.

## Raw aggregate table

| mode | delay | success | per task | steps mean +/- sd | wall mean | infer p50 | delivery p50 | holds | stale prefix | missed ticks |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| async_aligned | 0 ms | 30/30 | 0:10/10, 1:10/10, 2:10/10 | 136.1 +/- 12.9 | 10.48 s | 0.110 s | 0.124 s | 0.0 | 2.66 | 6.5 |
| async_aligned | 200 ms | 30/30 | 0:10/10, 1:10/10, 2:10/10 | 131.9 +/- 11.4 | 11.20 s | 0.158 s | 0.373 s | 0.0 | 6.76 | 8.6 |
| async_naive | 0 ms | 30/30 | 0:10/10, 1:10/10, 2:10/10 | 188.3 +/- 19.6 | 13.49 s | 0.110 s | 0.126 s | 0.0 | 0.00 | 12.3 |
| async_naive | 200 ms | 30/30 | 0:10/10, 1:10/10, 2:10/10 | 245.8 +/- 26.6 | 17.88 s | 0.219 s | 0.433 s | 0.0 | 0.00 | 19.9 |
| sync | 0 ms | 30/30 | 0:10/10, 1:10/10, 2:10/10 | 126.9 +/- 11.9 | 8.90 s | 0.117 s | 0.154 s | 0.0 | 0.00 | 14.1 |
| sync | 200 ms | 30/30 | 0:10/10, 1:10/10, 2:10/10 | 126.9 +/- 11.7 | 10.37 s | 0.200 s | 0.471 s | 0.0 | 0.00 | 38.1 |

## Paired comparisons

| delay | comparison | success | steps baseline -> candidate | paired reduction [95% CI] | relative | wins/ties/losses |
|---:|---|---:|---:|---:|---:|---:|
| 0 ms | async_naive -> async_aligned | 30 -> 30 | 188.3 -> 136.1 | 52.2 [49.3, 55.2] | 27.7% | 30/0/0 |
| 0 ms | sync -> async_aligned | 30 -> 30 | 126.9 -> 136.1 | -9.2 [-11.0, -7.6] | -7.3% | 0/0/30 |
| 200 ms | async_naive -> async_aligned | 30 -> 30 | 245.8 -> 131.9 | 113.9 [108.1, 119.7] | 46.3% | 30/0/0 |
| 200 ms | sync -> async_aligned | 30 -> 30 | 126.9 -> 131.9 | -5.0 [-6.7, -3.2] | -3.9% | 3/2/25 |
