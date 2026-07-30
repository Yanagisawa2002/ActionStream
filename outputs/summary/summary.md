# ActionStream M3 Results

All six conditions reached 30/30 success. The binary success metric is therefore ceilinged; paired step and wall-time reductions are the informative outcomes.

## Raw aggregate table

| mode | delay | success | per task | steps mean +/- sd | wall mean | infer p50 | delivery p50 | holds | stale prefix | missed ticks |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| async_aligned | 0 ms | 30/30 | 0:10/10, 1:10/10, 2:10/10 | 136.6 +/- 13.0 | 9.23 s | 0.115 s | 0.131 s | 0.0 | 2.57 | 11.8 |
| async_aligned | 200 ms | 30/30 | 0:10/10, 1:10/10, 2:10/10 | 142.7 +/- 54.9 | 10.08 s | 0.244 s | 0.457 s | 0.0 | 8.11 | 10.0 |
| async_naive | 0 ms | 30/30 | 0:10/10, 1:10/10, 2:10/10 | 189.7 +/- 19.8 | 11.95 s | 0.113 s | 0.128 s | 0.0 | 0.00 | 11.9 |
| async_naive | 200 ms | 30/30 | 0:10/10, 1:10/10, 2:10/10 | 246.3 +/- 26.6 | 15.22 s | 0.239 s | 0.452 s | 0.0 | 0.00 | 17.8 |
| sync | 0 ms | 30/30 | 0:10/10, 1:10/10, 2:10/10 | 126.9 +/- 11.9 | 8.83 s | 0.112 s | 0.151 s | 0.0 | 0.00 | 12.7 |
| sync | 200 ms | 30/30 | 0:10/10, 1:10/10, 2:10/10 | 126.9 +/- 11.9 | 10.43 s | 0.210 s | 0.490 s | 0.0 | 0.00 | 40.0 |

## Paired comparisons

| delay | comparison | success | steps baseline -> candidate | paired reduction [95% CI] | relative | wins/ties/losses |
|---:|---|---:|---:|---:|---:|---:|
| 0 ms | async_naive -> async_aligned | 30 -> 30 | 189.7 -> 136.6 | 53.1 [50.0, 56.3] | 28.0% | 30/0/0 |
| 0 ms | sync -> async_aligned | 30 -> 30 | 126.9 -> 136.6 | -9.7 [-11.7, -7.9] | -7.6% | 0/0/30 |
| 200 ms | async_naive -> async_aligned | 30 -> 30 | 246.3 -> 142.7 | 103.5 [79.9, 118.7] | 42.0% | 29/0/1 |
| 200 ms | sync -> async_aligned | 30 -> 30 | 126.9 -> 142.7 | -15.8 [-38.0, -3.1] | -12.5% | 4/1/25 |
