# Learned X-VLA native Isaac development status

Status date: 2026-08-16

Branch: `codex/actionstream/async-rtc-isaac`

Gate: **BLOCKED before paired holdout**

## Outcome

The real `lerobot/xvla-libero` checkpoint now runs end to end against a native
Isaac Sim 6.0.1 Franka scene. The integration smoke passed, but the learned
policy did not demonstrate task capability on the canonical LIBERO Object
task 0 instruction, “pick up the alphabet soup and place it in the basket.”

This is a partial systems result. It is not a task-success result, a paired
runtime comparison, or a cross-task benchmark. No scripted-policy result is
substituted for the failed learned-policy gate.

## What is genuinely working

- The frozen X-VLA checkpoint loads through the official LeRobot processors and
  performs GPU inference. Peak allocated CUDA memory was 12,250.56 MiB.
- Native Isaac receives finite, changing absolute 7D policy commands. The
  300-step run contained 300 unique mapped commands and moved the measured EEF
  by as much as 0.2586 m.
- The scene uses seven canonical LIBERO visual assets, robot-base-anchored world
  coordinates, the audited task-0 body poses and agent camera, a dynamic target,
  basket collision proxies, and a physical Franka controller.
- Robot-state parity is explicit: the audited LIBERO EEF matrix and quaternion
  fields remain separate, and one-reset joint offsets map Isaac state onto the
  LIBERO observation manifold.
- Policy axis-angle commands are preserved after proving that the canonical
  LIBERO action reconstructs the audited EEF matrix within 0.58 degrees. The
  earlier fixed Isaac-downward wrist discarded about 90 degrees of yaw.
- The initial wrist uses the canonical LIBERO orientation. The second policy
  image is a real EEF-relative camera whose audited reference transform is
  rigidly transported by measured EEF translation and orientation.
- The complete 300-step smoke passed all eight integration checks: checkpoint
  loaded, GPU inference, finite actions, action variation, commands applied,
  Franka motion, video recording, and no disqualifying collision.
- The post-run learned/Isaac/policy contract suite passed 86/86 tests (exit 0).

## Bounded capability result

| Metric | Result |
|---|---:|
| Control steps / inference requests | 300 / 30 |
| Native task success | 0/1 |
| Closest measured EEF-to-target distance | 0.200093 m at step 174 |
| Maximum alphabet-soup target displacement | 0.000000 m |
| Collision events | 0 |
| Joint/workspace-limit events | 0 |
| Raw X-VLA gripper rows | 898 positive, 2 negative out of 900 |
| Video | H.264, 360x360, 20 fps, 301 frames, 15.05 s |

The video was copied locally, decoded, and inspected as a contact sheet plus
selected agent and wrist frames. The arm moves toward the wrong/offset object
region, closes without a valid target contact, never moves the alphabet-soup
object, and times out. Registered development failure:

`wrong-object or offset approach -> premature empty close -> grasp miss -> task timeout`

For comparison, the existing official LIBERO task-0 trace from the same X-VLA
checkpoint succeeds in 145 steps. Its first action is approximately
`[-0.1472, -0.0042, 0.2416, -2.1859, -2.2037, 0.0628, -1]`; the final native
Isaac candidate begins near
`[-0.1454, 0.0024, 0.1266, -2.2027, -2.1853, 0.0793, +1]`.
The remaining position and gripper divergence is therefore a real
observation/domain capability gap, not a queue-runtime result.

## Frozen image/state counterfactual

A result-frozen four-cell diagnostic now isolates that gap at the first X-VLA
action chunk. Across three paired inference seeds, replacing only the official
LIBERO images with the native Isaac agent/wrist images produced 0.760888
full-chunk RMSE and 0.115453 m first-action XYZ displacement. Replacing only the
official state with the native mapped state produced 0.005239 RMSE and 0.013787
m displacement. The image intervention is 145.25x larger by chunk RMSE.

More concretely, official images produce an initial Z near 0.242 m and 30/30
negative gripper rows under either state. Native images produce Z near 0.126 m
and 30/30 positive gripper rows under either state. This reproduces the failed
rollout's low approach and opposite gripper command without involving any async
runtime.
The official/official first action remains within 0.000399 L2 of the prior
successful official task-0 trace, which validates the diagnostic path.

This is a **positive attribution result, not a task-success result**. It shows
that visual/camera domain mismatch is the dominant immediate source of the
first-chunk error. The mapped joint-position mismatch (native-vs-official L2
1.988671) remains real and should be corrected, but its controlled initial
effect is much smaller; later-episode effects are not ruled out. See the
[counterfactual report](../outputs/xvla_isaac_input_counterfactual_v1/report.md)
and [content-level figure](../outputs/xvla_isaac_input_counterfactual_v1/input_counterfactual.png).

## Frozen visual-canary result (2026-08-17)

One bounded development candidate replaced the plain brown Isaac surface with
the canonical LIBERO living-room table OBJ and diffuse texture. The first
capture was rejected before policy comparison because aligning the OBJ's raised
maximum-Z edge hid the dominant tabletop below the native ground. V2 instead
mapped the audited 0.267306 m tabletop plane to the task surface and removed
only the missing upstream `map_Bump` dependency; both choices are recorded in
the frozen candidate contract.

The valid V2 frames visibly restore a wood surface, but retain different table
extent, exposed blue ground, lighting, and Panda/hand appearance. On three
paired inference seeds, holding official robot state fixed, V2 obtained:

| Metric | Plain native | V2 | Frozen V2 threshold | Result |
|---|---:|---:|---:|---|
| Full-chunk RMSE | 0.760888 | 0.761554 +/- 0.000003 | <= 0.25 | fail |
| First-action XYZ L2 | 0.115453 m | 0.079303 +/- 0.000046 m | <= 0.05 m | fail |
| First-action 7D L2 | 2.003562 | 2.001838 +/- 0.000005 | <= 0.75 | fail |
| Candidate close fraction | 0.00 | 0.00 | >= 0.90 each seed | fail |

The 31.3% smaller first-action XYZ error is a reproducible partial positive:
table appearance recovers some of the vertical cue. It does **not** restore the
chunk: full-chunk error is unchanged and all 30 candidate gripper commands have
the opposite sign from official in every seed. Per the preregistered stop rule,
no sync capability gate or async holdout was launched. See the
[canary findings](../outputs/xvla_isaac_visual_canary_v1/CANARY_FINDINGS.md),
[input comparison](../outputs/xvla_isaac_visual_canary_v1/visual_input_comparison.png),
and [action comparison](../outputs/xvla_isaac_visual_canary_v1/first_chunk_action_comparison.png).

## Frozen official-render / Isaac-state bridge canary (2026-08-17)

The first bridge candidate replayed the Isaac joint vector directly into the
official LIBERO Panda. Its qpos/qvel writes were exact, but the rendered EEF
missed the requested Isaac EEF by 0.032654 m and the arm/wrist appearance was
visibly inconsistent. It was rejected structurally before policy inference.

The accepted candidate instead uses deterministic damped-least-squares IK to
retarget the official Panda to the measured Isaac reset EEF, then writes the
mapped gripper state exactly. It converged in 3 iterations with
5.332e-08 m position error and 2.291e-08 rad orientation error. No MuJoCo
physics step or copied static reference frame is used.

The runner then reused the exact three V2 inference seeds and unchanged frozen
thresholds. The official reference and bridge condition share task,
instruction, checkpoint, action space, and seed. The reference uses the
official reset state; the bridge uses the mapped Isaac policy state and an
official-render observation retargeted to that state. This bridge comparison
therefore changes both image and policy state relative to the reference; it is
a frozen capability gate, not a new one-factor ablation.

| Metric | Official-table V2 | Bridge mean +/- std | Frozen threshold | Result |
|---|---:|---:|---:|---|
| Full-chunk RMSE | 0.761554 | 0.020933 +/- 0.000003 | <= 0.25 | pass |
| First-action XYZ L2 | 0.079303 m | 0.003393 +/- 0.000040 m | <= 0.05 m | pass |
| First-action 7D L2 | 2.001838 | 0.017097 +/- 0.000150 | <= 0.75 | pass |
| Candidate close fraction, worst seed | 0.00 | 1.00 | >= 0.90 | pass |

Relative to V2, the three error metrics fell by 97.25%, 95.72%, and 99.15%,
and the gripper sign was restored on all three seeds. This is a strong positive
reset-time first-chunk result. Together with the prior factorial V2 result,
where changing state alone had a much smaller effect, it is consistent with
renderer and robot appearance dominating the earlier reset-time error. The
bridge result alone is not a single-factor causal estimate. It is still not
episode success, task success, or async-runtime evidence. The bridge Z command
also separates from official later in the 30-step chunk, so passing the frozen
tolerances does not imply identical closed-loop behavior.

The first formal launch stopped before any policy inference because the cloned
Hugging Face cache was absent. A retry changed only the runtime cache location
and offline flags; source, candidate, config, seeds, and thresholds remained
unchanged. It completed all six paired inferences.

## Evidence and provenance

Local small evidence root (kept outside normal Git staging):

- `outputs/learned_isaac_libero_taskcap_rigidcamera_300_v1/summary.json`
- `outputs/learned_isaac_libero_taskcap_rigidcamera_300_v1/events.jsonl`
- `outputs/learned_isaac_libero_taskcap_rigidcamera_300_v1/policy_actions.jsonl`
- `outputs/learned_isaac_libero_taskcap_rigidcamera_300_v1/learned_isaac_smoke.mp4`
- `outputs/learned_isaac_libero_taskcap_rigidcamera_300_v1/video_contact_sheet.jpg`

Frozen counterfactual evidence:

- `outputs/xvla_isaac_input_counterfactual_v1/summary.json`
- `outputs/xvla_isaac_input_counterfactual_v1/report.md`
- `outputs/xvla_isaac_input_counterfactual_v1/input_counterfactual.png`

Frozen visual-canary evidence:

- `outputs/xvla_isaac_visual_canary_v1/CANARY_FINDINGS.md`
- `outputs/xvla_isaac_visual_canary_v1/canary_evaluation.json`
- `outputs/xvla_isaac_visual_canary_v1/visual_input_comparison.png`
- `outputs/xvla_isaac_visual_canary_v1/first_chunk_action_comparison.png`
- `outputs/xvla_isaac_visual_canary_v1/xvla_isaac_visual_canary_v2_inference/summary.json`

Frozen official-render bridge evidence:

- `outputs/xvla_isaac_official_render_bridge_v1_retry1/BRIDGE_CANARY_FINDINGS.md`
- `outputs/xvla_isaac_official_render_bridge_v1_retry1/summary.json`
- `outputs/xvla_isaac_official_render_bridge_v1_retry1/formal_run_receipt.json`
- `outputs/xvla_isaac_official_render_bridge_v1_retry1/bridge_visual_comparison.png`
- `outputs/xvla_isaac_official_render_bridge_v1_retry1/bridge_canary_gate_comparison.png`
- `outputs/xvla_isaac_official_render_bridge_v1_retry1/bridge_first_chunk_action_comparison.png`

Remote complete evidence root:
`/root/autodl-tmp/results/learned_isaac_libero_taskcap_rigidcamera_300_v1`

| Artifact | SHA-256 |
|---|---|
| Local/remote small evidence archive | `c2d134f2e74036ce7d2f33f2b93cdf7ad8696c37f0a4a9c5468a25491bea4cd0` |
| MP4 | `5ac904d2fc47904814fd2b01100139c001d724e35d70fd69abc8242607d48d33` |
| Deployed source archive v20 | `5485e562d0635199de000310c5b7e3d488f14774e85c65b8a031bb45561e1092` |
| Task-scene contract | `d73b4af3b7b39aed010cb6dc50435b4f90ccbd939e5047d7cc967390575b21e7` |
| Adapter contract | `f66a453461416ca14af5005eb0253a354d56156aaf19b80388d5bcb95a6c830d` |
| Runtime source manifest | `513fd63b98cac1fa88be404a4da18f431b8b5a186b9309070c9453033fd3a403` |
| Official-render bridge frozen config | `eab9cfb3f0424b3f383d93d5d2085b14a41954dc0d96244661b5b833cdeed72d` |
| Official-render bridge formal summary | `e883de7764f7a1a0ac1340ce94624802310d3b56bdbe1ccd9a61e259db60ce77` |
| Official-render bridge raw result archive | `301bd78f346fa5b097269611a20b5ce7adf8c595cc763d428639d0ec7877ab5f` |

## Why the paired runtime matrix is still not opened

The accepted bridge now passes the reset-time prerequisite that V2 failed, but
it is not yet episode-ready. After motion, each policy request must synchronize
the dynamic robot and object poses into the official renderer. Reusing the
reset frames would be a structural placeholder and would not test a learned
closed-loop policy.

The next valid gate is therefore:

1. synchronize robot and task-object visual state into the official renderer
   before every policy request;
2. freeze a small disjoint X-VLA sync capability protocol before inspecting its
   results; and
3. require nonzero task success before spending the multi-task async holdout.

An Isaac-trained or Isaac-domain-adapted checkpoint remains a valid alternative.
Only after learned sync task capability is nonzero should the candidate be
frozen for `sync`, `latest-only`, official LeRobot async, official RTC, and
ActionStream aligned execution.
