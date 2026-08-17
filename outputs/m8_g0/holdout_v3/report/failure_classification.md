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
