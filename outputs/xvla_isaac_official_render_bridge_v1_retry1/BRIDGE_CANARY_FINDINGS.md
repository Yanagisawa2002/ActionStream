# X-VLA official-render / Isaac-state bridge findings

Status: **passed the unchanged frozen V2 first-chunk canary**. This opens a disjoint sync capability gate; it is not task-success or async evidence.

## Raw paired results

| Seed | Chunk RMSE | First XYZ L2 (m) | First 7D L2 | Bridge close fraction | Official close fraction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2026081701 | 0.020934 | 0.003381 | 0.017047 | 1.00 | 1.00 |
| 2026081702 | 0.020929 | 0.003438 | 0.017265 | 1.00 | 1.00 |
| 2026081703 | 0.020934 | 0.003361 | 0.016979 | 1.00 | 1.00 |

## Frozen gate and V2 comparison

| Metric | V2 native image / official state | Bridge official render / Isaac state, mean +/- std | 95% CI | Frozen threshold | Improvement vs V2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full chunk RMSE | 0.761554 | 0.020933 +/- 0.000003 | [0.020925, 0.020940] | <= 0.250000 | 97.25% |
| First action XYZ L2 (m) | 0.079303 | 0.003393 +/- 0.000040 | [0.003295, 0.003492] | <= 0.050000 | 95.72% |
| First action 7D L2 | 2.001838 | 0.017097 +/- 0.000150 | [0.016725, 0.017468] | <= 0.750000 | 99.15% |
| Close-command fraction, worst seed | 0.00 | 1.00 | official worst seed 1.00 | >= 0.90 | sign restored |

The 95% intervals use a t interval with n=3 paired inference seeds; they describe this frozen canary and are not a population-level generalization claim.

## Key findings

1. **Observation:** all four frozen checks passed. Chunk RMSE is 0.020933, first XYZ L2 is 0.003393 m, first 7D L2 is 0.017097, and every bridge chunk closes the gripper.  **Interpretation:** within this reset-scoped diagnostic, combining the official visual domain with the mapped Isaac policy state restores the first-chunk action manifold under the frozen tolerances.  **Implication:** together with the prior factorial V2 result, where changing state alone had a much smaller effect, this is consistent with renderer and robot appearance dominating the earlier reset-time error. The bridge comparison itself changes both image and policy state relative to the official reference, so it is not a new single-factor causal estimate and does not establish episode-level causality.  **Next step:** preregister a disjoint sync capability gate.

2. **Observation:** EEF IK converged in 3 iterations with 5.332e-08 m position error and 2.291e-08 rad orientation error.  **Interpretation:** the bridge is state-conditioned rather than a copied static reference frame.  **Implication:** reset-time camera evidence is structurally aligned.  **Next step:** synchronize object poses and robot state at every control request before calling the bridge episode-ready.

3. **Observation:** the initial formal attempt stopped during checkpoint loading; the unchanged offline-cache retry completed all six paired inferences.  **Interpretation:** this was an infrastructure retry, not result-driven rerunning.  **Implication:** the frozen comparison remains valid.  **Next step:** carry `HF_HOME` and offline cache provenance into the sync-gate launcher.

4. **Observation:** the mean bridge Z command remains close initially but separates from the official curve later in the 30-step chunk.  **Interpretation:** passing the preregistered error gates does not mean the two chunks are identical.  **Implication:** reset-time alignment is a prerequisite result, not evidence of stable closed-loop execution.  **Next step:** synchronize robot and object visual state on every policy request, then run a fresh disjoint sync capability gate.

## Figure scope

- `bridge_visual_comparison.png`: for the same task, scene/object reset, instruction, and official renderer, the top row is the official LIBERO reset observation and the bottom row retargets the canonical Panda end-effector (EEF) pose and gripper from the measured Isaac reset. The robot state intentionally differs; this is a reset diagnostic, not task-success evidence.
- `bridge_canary_gate_comparison.png`: for the same task/reset/instruction and three paired seeds, the V2 native-image/official-state and bridge official-render/Isaac-state errors are each divided by the same metric-specific frozen threshold. Values below 1 show gate margin, not directly comparable physical units; the right panel reports worst-seed close-command fraction derived from the raw summaries. This cross-candidate comparison changes both image and policy state and is not a one-factor ablation.
- `bridge_first_chunk_action_comparison.png`: mean 30-step Z and gripper commands over the same three paired inference seeds and official reference, with the same task, reset, and instruction. V2 uses native V2 images with official policy state; the bridge uses state-conditioned official-render images with mapped Isaac policy state. This is first-chunk diagnostic behavior, not episode success or a one-factor causal comparison.
