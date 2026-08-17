# M5-G0 task and entity audit

The audit inspected exactly two M4 tasks. Both already achieved 10/10 with
`async_aligned` at the frozen 950 ms pressure point.

The verified starting commit was `0a2412625b858af8d507be6650612b13a28b91a5`
on a clean `master` worktree matching `origin/master`. A fresh, isolated
task-0 M4 smoke at 950 ms succeeded in 143 environment steps; its episode
record SHA-256 is
`a72b4dd20d5e24cdd436f96e5333344ba8191a7eb7490b8c5451d8b4b5492bc2`.

| order | task | instruction | stationary task-critical entity | decision |
|---:|---:|---|---|---|
| 1 | 0 | pick up the alphabet soup and place it in the basket | `basket_1` | selected |
| 2 | 2 | pick up the salad dressing and place it in the basket | `basket_1` | one permitted backup |

## Selected task

- Task ID: `0`
- Instruction: `pick up the alphabet soup and place it in the basket`
- Manipulated object: `alphabet_soup_1`
- Moved entity: `basket_1`
- MuJoCo body: `basket_1_main`
- Free joint: `basket_1_joint0`
- Goal: `(In alphabet_soup_1 basket_1_contain_region)`

The basket is normally stationary and the instruction does not ask the robot
to manipulate it. Its containment site is attached to its physical body, so a
free-joint translation moves both the rendered/physical receptacle and the
unchanged LIBERO success region. No success predicate is edited.

Pose is read from the resolved LIBERO object/body and its MuJoCo free joint.
Mutation changes only planar x, preserves z and quaternion, zeros the six
free-joint velocities, and calls `sim.forward()`. A fresh policy observation
is regenerated normally from the changed simulator state; pixels are never
edited.

## Physical feasibility

A disposable reset/teleport/forward/restore probe covered both candidates,
20 initial states per task, both x signs, and 30/50/70 mm magnitudes
(120 configurations). Maximum achieved-displacement error was below
`7e-18 m`; there were zero non-floor collision cases. Expected floor support
penetration remained below `1.5e-5 m`.

The fixed direction is negative x, toward the robot. Across tested initial
states, the moved containment center remained 0.577-0.646 m from the robot
base for the candidate magnitudes and stayed inside the declared workspace.

## Frozen timing

The shift is tied to the first replenishment request whose active queue is
non-empty. In the M4 trace this request is captured at step 10 with queue depth
20. At 20 Hz, `floor(0.950 * 20) = 19` delay steps, so:

```text
shift_step = 10 + max(1, floor(19 / 2)) = 19
```

The corresponding old result normally merges at step 31 or 32. At the shift
boundary the end effector remains over 0.40 m away from the basket in y, and
the first gripper-close proxy is not until steps 49-56. The shift is therefore
observation-first, physical-change-second, old-result-third, and pre-contact.

## Limitations

The geometry audit establishes a safe physical perturbation, not behavioral
recovery. The frozen policy must still pass the five-pair calibration after a
fresh observation. The detector is simulator-ground-truth oracle sensing, not
a deployable perception stack. Floor support is an allowed contact; any other
moved-entity contact makes the perturbation explicitly invalid.
