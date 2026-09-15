# Visual completion training pilot

This experiment trains a finite RGB-only verifier for **tomato sauce placed in
the basket**. It is an offline candidate for the ActionStream completion module.
Training does not enable runtime acceptance or establish a general VLA result.

## Why new labels are required

The [official LIBERO exporter](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py)
sets `rewards[-1] = 1` and all earlier rewards to zero. These are terminal markers,
not per-frame completion truth. It also records rendered observations after the
action while storing selected original simulator states. The pilot therefore
re-renders each sampled state and calls `env.check_success()` at that exact state.
Neither original rewards nor original RGB/state alignment are used as labels.

Simulator state is used only in offline dataset construction. The learned model
receives two RGB images, in fixed external-camera and wrist-camera slots. It does
not receive state vectors, frame indices, rewards, or terminal flags.

## Frozen first-run configuration

- Source: `yifengzhu-hf/LIBERO-datasets` at
  `f13aa24a3da8c43c7225569f28c562979fa0e35a`.
- File: `libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5`.
- Official file SHA-256:
  `43c52bdaa78b4c6aff3afb54d932d6a0e065f5c30c8e30a6c17f78749d487ea9`.
- Split: 60% training, 20% validation, 20% untouched test episodes; seed 20260915.
  The manifest records every episode assignment and rendered shard hash.
- Sampling: every third simulator state plus the final recorded state.
- Model: shared ImageNet-pretrained ResNet18 encoder; fixed-order dual-view MLP;
  all parameters finetuned. This is a compact supervised baseline, not Qwen LoRA.
- Training: 30 epochs, batch 64, AdamW, learning rate 0.0001, weight decay 0.01,
  BF16 autocast. Positive class weight comes from the training split only.
- Reporting: validation loss and counts of false completion, true completion,
  false incompletion and abstention. Fixed thresholds are 0.9 and 0.1.
- `best.pt` is selected by validation loss. `last.pt` includes optimizer and Torch
  RNG states. Each epoch writes metrics; the test split is not evaluated here.

## Deployment

Use Python 3.12 and a CUDA-compatible Torch installation. The first 5090 run
reuses Torch 2.8.0+cu128 and torchvision 0.23.0+cu128 in an isolated venv with
system site packages. Its additional packages include `hf-libero==0.1.4`,
`numpy==2.2.6`, and `h5py==3.14.0`. Actual versions are recorded in run `config.json`.
Headless rendering needs the EGL dispatcher (`libegl1` on Ubuntu) and driver
libraries. Source `/etc/network_turbo` when provided by the server.

Materialize `lerobot/libero-assets` (dataset revision
`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`) into the isolated environment's
`libero/libero/assets` directory before preparing data. Use an explicit successful
`snapshot_download` with that revision: hf-libero's automatic download can fail
after creating partial directories and later mistake them for complete assets.
For this task, restrict `allow_patterns` to `scenes/*`, `textures/*`,
`stable_scanned_objects/basket/*`, and the `bbq_sauce`, `butter`,
`chocolate_pudding`, `milk`, `orange_juice`, and `tomato_sauce` directories under
`stable_hope_objects`. The original demonstrations' `chiliocosm/assets` paths are
rebased to this directory; robot paths are resolved by the upstream utility.
The 5090 deployment uses explicit downloads with Xet disabled after the automatic
download failed; a mirror can help, but reduce concurrency if it rate-limits.
The demonstration file was obtained through `hf-mirror.com` and checked against
its official full SHA-256.

Copy this script and `src/actionstream/libero_config.py` into an isolated source
directory, and obtain the pinned HDF5 file. Verify its full hash before use.

```bash
export PYTHONPATH="$BASE/source/src"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PYTHON="$BASE/venv/bin/python"
SCRIPT="$BASE/source/scripts/engineering/train_visual_completion.py"
"$PYTHON" "$SCRIPT" prepare --hdf5 "$BASE/data/tomato-xet.hdf5" --output "$BASE/data/rendered-v1"
"$PYTHON" "$SCRIPT" train --data "$BASE/data/rendered-v1" --output "$BASE/runs/smoke-v1" --smoke --epochs 1
"$PYTHON" "$SCRIPT" train --data "$BASE/data/rendered-v1" --output "$BASE/runs/completion-v1" --epochs 30
```

`launch_visual_completion.sh` sequences these phases under a process lock and
requires the real-data GPU backward/save/reload smoke to pass before training.
It never shuts down the server. Launch it with a process timeout for a bounded run.

## Acceptance boundary

The demonstration splits measure this task's demonstration distribution. They do
not cover arbitrary failed rollouts, camera interventions, wrong-object completion,
or general instructions. Before runtime use, evaluate the selected frozen model
on untouched episodes and fresh failure/intervention cases. Preserve independent
scoring and require useful completion recall as well as low false completion;
an all-unknown predictor is not a successful repair.
