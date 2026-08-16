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
rollout's low approach and premature close without involving any async runtime.
The official/official first action remains within 0.000399 L2 of the prior
successful official task-0 trace, which validates the diagnostic path.

This is a **positive attribution result, not a task-success result**. It shows
that visual/camera domain mismatch is the dominant immediate source of the
first-chunk error. The mapped joint-position mismatch (native-vs-official L2
1.988671) remains real and should be corrected, but its controlled initial
effect is much smaller; later-episode effects are not ruled out. See the
[counterfactual report](../outputs/xvla_isaac_input_counterfactual_v1/report.md)
and [content-level figure](../outputs/xvla_isaac_input_counterfactual_v1/input_counterfactual.png).

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

## Why the paired runtime matrix was not opened

The intended comparison requires the same task-capable learned policy under
`sync`, `latest-only`, official LeRobot async, official RTC, and ActionStream
aligned execution. With zero target contact under the native sync capability
gate, the likely matrix is an uninformative all-zero task-success table. Opening
the formal holdout would also spend the disjoint tasks, seeds, initial states,
and network traces without testing the runtime hypothesis.

The next valid gate is one of:

1. an Isaac-trained or Isaac-domain-adapted X-VLA/SmolVLA/Pi policy;
2. an exact canonical LIBERO visual, camera, collision, and control-domain
   recreation with demonstrated native task success; or
3. a short, predeclared adaptation stage followed by a fresh disjoint capability
   gate.

Only after learned sync/native task capability is nonzero should the candidate
be frozen and the multi-task paired runtime protocol be opened.
