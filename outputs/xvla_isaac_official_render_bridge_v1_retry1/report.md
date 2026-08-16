# X-VLA official-render / Isaac-state bridge canary

Status: **passed frozen first-chunk canary**.

This is a reset-scoped offline paired diagnostic, not episode or task-success evidence.

| Metric | Mean +/- std | Frozen threshold | Result |
| --- | ---: | ---: | --- |
| Full chunk RMSE | 0.020933 +/- 0.000003 | <= 0.250000 | PASS |
| First action XYZ L2 (m) | 0.003393 +/- 0.000040 | <= 0.050000 | PASS |
| First action 7D L2 | 0.017097 +/- 0.000150 | <= 0.750000 | PASS |
| Candidate close fraction, worst seed | 1.000000 | >= 0.900000 | PASS |

The official renderer uses matching reset objects and an EEF-IK-retargeted canonical Panda; the policy state remains the measured Isaac mapping. Dynamic object synchronization and closed-loop task capability are not tested here.
