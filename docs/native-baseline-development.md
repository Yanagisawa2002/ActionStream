# Native synchronous development acceptance

This entry point exercises the fixed X-VLA checkpoint and real LIBERO simulator
directly. It does not pass through the asynchronous runtime, train a model, tune
the scheduler, or invoke a language planner. Unit-test results are engineering
checks only. Actual model, render, and rollout evidence comes from new native
receipts and images, never a test fixture or historical result.

`configs/native_baseline_development.json` fixes checkpoint revision, LeRobot
commit, required package versions, checkpoint/tokenizer hashes, asset revision,
and four reset identities before execution. Object task 5 means “pick up the
tomato sauce and place it in the basket.” The smoke uses initial state 45; the
three development episodes use states 0, 1, and 2. These are development cases,
not a held-out benchmark or an estimate of generalization.

## Environment prerequisites

Use a new Python 3.12 Linux environment in an owned store. The dispatch-specific
offline provisioner and all source/package/asset manifests are retained in
`artifacts/embodied_native_baseline_20260913` of the task workspace. The native
runner requires an `environment_receipt.json` containing the fixed LeRobot
commit and every installed LeRobot Python file's SHA-256, plus a
`libero_assets_receipt.json` with every simulator asset's SHA-256. It verifies
those installed sources and assets before loading a model.

The installed ActionStream package must contain the reviewed runner. Do not use
an ambient `PYTHONPATH` or silently borrow another environment. Keep all caches,
the model, tokenizer, and LIBERO assets in the owned store. Link the new
environment's `libero/libero/assets` directory to the verified asset directory;
this prevents hf-libero from downloading an unpinned default asset snapshot.
The supervisor supplies offline Hugging Face settings, a dedicated LIBERO
configuration directory, and EGL settings before Python imports.

## Fixed execution commands

Run from the reviewed source checkout under WSL. `STORE` is the dedicated Linux
path and `EVIDENCE` is this dispatch's new evidence directory. No training data
download is needed. The commands deliberately preserve prior output directories.

```bash
STORE=/mnt/d/CodexValidation/ActionStreamVLA-20260913
EVIDENCE=/mnt/c/Users/cgliu/.codex/worktrees/f3c0/ActionStream/artifacts/embodied_native_baseline_20260913
python3 scripts/engineering/run_native_baseline.py \
  --store "$STORE" --config configs/native_baseline_development.json \
  --phase smoke --output "$EVIDENCE/smoke"
python3 scripts/engineering/run_native_baseline.py \
  --store "$STORE" --config configs/native_baseline_development.json \
  --phase development --output "$EVIDENCE/development" \
  --smoke-receipt "$EVIDENCE/smoke/run_receipt.json"
```

Inspect the smoke receipt and images before starting development. The smoke must
complete native input/action/render contracts and an episode; completing the
manipulation task is reported separately. No more than one development batch is
allowed by the supervisor. A failed phase is retained. Any limited installation
repair must use a new output namespace and keep the original failure receipt.

The supervisor first refuses a busy GPU or insufficient host disk reserves. An
exclusive owned ledger in `STORE/runs/budget.json` charges the entire lifetime
of every native child, including imports, verification, rendering, model load,
inference, stepping, and cleanup, against a cumulative 1,200-second limit. Smoke
is limited to 600 seconds. This is stricter than counting GPU inference alone.
Timeouts terminate only that invocation's new process group. An unfinished
ledger is an error requiring inspection, not an invitation to reset its budget.

## Observation, action, and result contracts

Each real reset supplies agent-view and wrist images, end-effector position and
quaternion, gripper position, and the native task instruction. Raw images are
360 x 360 RGB, as configured by the fixed native `LiberoEnv`. The checkpoint's
256 x 256 feature metadata is not the simulator render size: native
`XVLAPolicy._prepare_images` uses the checkpoint's `resize_imgs_with_padding`
setting to produce 224 x 224 model images. The runner preserves that native
transform. The XVLA-specific native LIBERO processor constructs the checkpoint's
20D `max_state_dim` layout: end-effector position, 6D rotation, and the native
zero fields. The checkpoint still lists 8D state feature metadata; that is not
the final XVLA processor output shape. Robot/gripper observations are passed
intact to this official processor, which determines the consumed state fields.
The saved tokenizer constructs the language input. Checkpoint chunk length is 30; no config-class
default substitutes for downloaded metadata. Both official postprocessing
stages run per timestep: raw `[1,30,20]` becomes finite `[1,30,7]` absolute
position, axis-angle rotation, and gripper commands. The runner does not slice
the raw output or clip absolute pose coordinates to a generic action Box.

One chunk executes synchronously for at most 30 controls before taking the
current observation for the next inference. The real controller is configured
at 20 Hz. Each episode permits 300 explicit control calls; the native reset's
10 settling actions are recorded separately and excluded from those 300.
Nominal simulated control frequency does not imply 20 Hz wall-clock inference.
Slow synchronous inference blocks the next control; the runner never hides it
with catch-up steps or queue-supply metrics.

`native_trace.jsonl` records reset identity and asset hashes, every attempted
inference and its completed action contract, and every completed control step.
Model time is CUDA-synchronized around `predict_action_chunk`. Synchronous
inference time includes preprocessing, prediction, official postprocessing, and
the final CPU action copy/validation. Step time surrounds `LIBERO.step` through
the thin backend. Control time includes any inference, the native step, the
initial step-return journal, and observation validation. It excludes the final
step journal write, PNG writing, and pacing
sleep. Episode wall time includes reset, settling, all controls, logging, and
pacing. These measurements are neither GPU kernel profiles nor speedup claims.

Only the native boolean `info['is_success']` establishes task completion.
Gymnasium can wrap `numpy.bool_` in an object array; the runner validates the
scalar boolean type and rejects numeric or string truthiness. A `step_returned`
event preserves the issued action and native completion container before
validating the returned observation and flag. A
300-step timeout with no native success is a failed completed episode. A
pre-rollout error is NOT RUN; an interrupted partial rollout has unavailable
success, not zero success. Separate readiness, model loading, rollout execution,
and task completion fields prevent one from being mistaken for another. Saved
frames preserve raw policy-observation orientation; no visualization transform
changes policy input or the native completion decision.
