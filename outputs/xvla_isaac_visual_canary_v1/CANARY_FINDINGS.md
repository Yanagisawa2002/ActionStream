# X-VLA native-Isaac visual canary

Status: **failed frozen canary**. The official-table candidate did not qualify for a sync capability gate, so no sync or async holdout was launched.

This is an offline paired first-action-chunk diagnostic, not task-success evidence.

## Raw paired results

| Inference seed | Chunk RMSE | First XYZ L2 (m) | First 7D L2 | Candidate close fraction | Official close fraction | Candidate first Z (m) | Official first Z (m) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026081701 | 0.761554 | 0.079331 | 2.001840 | 0.00 | 1.00 | 0.162708 | 0.241577 |
| 2026081702 | 0.761557 | 0.079329 | 2.001841 | 0.00 | 1.00 | 0.162776 | 0.241644 |
| 2026081703 | 0.761551 | 0.079250 | 2.001833 | 0.00 | 1.00 | 0.162823 | 0.241611 |

## Frozen acceptance

| Metric | Mean or worst seed | Threshold | Result |
| --- | ---: | ---: | --- |
| Full chunk RMSE | 0.761554 | <= 0.250000 | FAIL |
| First action XYZ L2 (m) | 0.079303 | <= 0.050000 | FAIL |
| First action 7D L2 | 2.001838 | <= 0.750000 | FAIL |
| Minimum close fraction across seeds | 0.000000 | >= 0.900000 | FAIL |

## Comparison with the plain native render

| Metric | Plain native | Official-table V2 | Relative improvement (smaller is better) |
| --- | ---: | ---: | ---: |
| Full chunk RMSE | 0.760888 | 0.761554 | -0.09% |
| First action XYZ L2 (m) | 0.115453 | 0.079303 | +31.31% |
| First action 7D L2 | 2.003562 | 2.001838 | +0.09% |

## Key findings

1. **Observation:** first-action XYZ displacement improved from 0.115453 m to 0.079303 m (31.3% smaller), but remained above the frozen 0.05 m threshold.
   **Interpretation:** adding the official diffuse tabletop recovers part of the vertical motion cue.
   **Implication:** appearance parity matters, but tabletop texture alone is insufficient.
   **Next step:** use an official-render/Isaac-state bridge or camera-specific robot render layer before any new runtime benchmark.

2. **Observation:** full-chunk RMSE was 0.761554 +/- 0.000003, essentially unchanged from the plain native render, and the candidate opened the gripper for every one of 30 actions in every seed while official closed it.
   **Interpretation:** the remaining visual domain gap dominates the chunk after the partially recovered first Z command; visible differences include table extent, exposed blue ground, lighting, and the Isaac hand/body silhouette.
   **Implication:** the previous all-zero native episodes cannot be cleanly interpreted as pure async-scheduling evidence while this visual-domain blocker remains.
   **Next step:** validate a hybrid renderer or exact official camera/robot appearance contract with this same frozen canary before spending task-level episodes.

3. **Observation:** across-seed standard deviations are below 5e-5 for the primary distances.
   **Interpretation:** this is a stable blocker, not a noisy unlucky seed.
   **Implication:** adding more seeds to the same invalid visual contract has low information value.
   **Next step:** change the rendering contract, then preregister a new candidate version; do not relax the V2 thresholds.

## Figures

- `visual_input_comparison.png`: for the same task, reset seed, and instruction, the top row is the official LIBERO reference observation and the lower rows are the plain and official-table V2 native-Isaac reset renders; the two native rows share the same native state contract.
- `first_chunk_action_comparison.png`: mean 30-step absolute Z and gripper commands over three paired inference seeds, with official robot state held fixed for each official-vs-native image pair. It is a first-chunk diagnostic, not task-success evidence.

Exact actions and paired statistics are preserved in `xvla_isaac_visual_canary_v2_inference/summary.json`; `canary_evaluation.json` records the frozen gate decision.
