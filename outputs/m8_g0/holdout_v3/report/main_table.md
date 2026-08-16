| Network profile | Method | Task success (Wilson 95% CI) | Mean of episode-mean latency (ms) | Median episode p50-p95 (ms) | Grasp | Recovery | Expired executed |
|---|---|---:|---:|---:|---:|---:|---:|
| P0 sanity | Sync hold | 60/60 (100.0%) [94.0%, 100.0%] | 15.7 | 15.9-16.4 | 60/60 | 60/60 | 0 |
| P1 fixed 850 ms | Sync hold | 1/60 (1.7%) [0.3%, 8.9%] | 881.8 | 877.9-898.5 | 60/60 | 60/60 | 0 |
| P1 fixed 850 ms | Naive async | 0/60 (0.0%) [0.0%, 6.0%] | 905.7 | 895.5-907.7 | 0/60 | 0/60 | 4244 |
| P1 fixed 850 ms | ActionStream aligned | 49/60 (81.7%) [70.1%, 89.4%] | 893.8 | 894.5-902.8 | 60/60 | 60/60 | 0 |
| P2 850 ms + jitter/faults | Sync hold | 15/60 (25.0%) [15.8%, 37.2%] | 1008.9 | 925.7-1379.9 | 45/60 | 44/60 | 0 |
| P2 850 ms + jitter/faults | Naive async | 0/60 (0.0%) [0.0%, 6.0%] | 986.7 | 918.8-1454.5 | 0/60 | 0/60 | 4231 |
| P2 850 ms + jitter/faults | ActionStream aligned | 52/60 (86.7%) [75.8%, 93.1%] | 975.9 | 912.8-1654.9 | 60/60 | 60/60 | 0 |
