# Three-task integration candidate

This branch adds development tooling for Object 5, Spatial 7 and Goal 2. It has
**not** passed native three-task acceptance. Follow
[the experiment plan](../refine-logs/EXPERIMENT_PLAN.md) and
[tracker](../refine-logs/EXPERIMENT_TRACKER.md). The old single-task CLI stays available.

## Runtime path

`original text → pinned Qwen proposal → full-utterance host permit → registered task`

`task text embedding + temporal RGB → shared visual classifier → existing confirmation/retry controller`

Only the validated task selects the native suite/task ID. Public observations
cross to the isolated VLA process; simulator truth stays in the private recorder
and post-run scorer. Scene labels never trigger online stopping or retry.

Finite accepted examples:

- `Please put the tomato sauce in the basket.`
- `Pick up the black bowl on the stove and place it on the plate.`
- `Put the wine bottle on top of the cabinet.`

The host permits put/place/move and pick-up-then-place forms with the exact
registered entity phrases and relation, optional polite prefix/suffix and terminal
punctuation. Other paraphrases may be conservatively rejected. Omitting the
Spatial source qualifier is rejected because it leaves the target ambiguous.

## Development commands

Use the locked Linux CUDA environment and previously verified model/LIBERO assets.
`$STORE` below is the server's asset-store directory. Set actual paths; do not
copy historical server addresses into scripts. Both CLI help commands run locally:

```bash
python -m actionstream.llm_vla.multitask_cli --help
python scripts/engineering/multitask_completion.py --help
```

Prepare `consumed.json` as `{"episodes":[{"task_key":null,"layout_sha256":"..."}]}`
from all retained historical exclusions/layouts and all new development runs.
Null task keys conservatively exclude a legacy layout from every task. Do not
start with an empty inventory on this project. Keep entries task-scoped for new
runs and hash/archive the inventory before collection. The CLI retains a copy.

Example collection (a development seed must be declared in the run inventory):

```bash
python -m actionstream.llm_vla.multitask_cli \
  --mode collect --split train --seed "$DEV_SEED" \
  --request 'Put the wine bottle on top of the cabinet.' \
  --assets "$STORE/model-assets" --asset-manifest configs/completion_runtime_assets.json \
  --libero-assets "$STORE/libero-assets" --libero-asset-manifest configs/libero_runtime_assets.json \
  --language-config configs/finite_agent_language.json \
  --consumed-manifest consumed.json --output outputs/multitask/goal-train-01
```

Collection always returns unknown to the stopping controller, then replays
segmentation for dataset labels after execution. It must reach the bounded
horizon to enter training. Use `--force-open` to collect unsuccessful trajectories;
`--recovery` is available only during learned-model evaluation. Preserve all failed
runs and mark their layouts consumed even when no dataset entry is produced.

```bash
python scripts/engineering/multitask_completion.py manifest \
  --episodes outputs/multitask/*-train-* outputs/multitask/*-validation-* \
  --output outputs/multitask/development.json
python scripts/engineering/multitask_completion.py train \
  --manifest outputs/multitask/development.json --output outputs/multitask/conditioned
python scripts/engineering/multitask_completion.py train \
  --manifest outputs/multitask/development.json --ablation no_task \
  --output outputs/multitask/no-task
```

Evaluate the conditioned checkpoint with the same runtime arguments, replacing
`--mode collect --split train` by `--mode evaluate --split validation`, and adding
`--checkpoint .../candidate.pt --checkpoint-sha256 ...`. For fault/recovery
development, also pass `--force-open --recovery`. The no-task ablation checkpoint
is intentionally rejected by the production candidate predictor.

The scorer verifies the authorization, task/condition identities, retained RGB
clips, action provenance, independent truth calculation and timing. A later fresh
simulator replay must independently validate the recorded physical facts.

```bash
python scripts/engineering/audit_multitask.py \
  --episode outputs/multitask/goal-validation-01 \
  --libero-assets "$STORE/libero-assets" --output outputs/multitask/goal-audit-01
```

This tool checks the retained file manifest, replays the RGB/action journal, then
restores every physical state in a new simulator with neither learned model loaded.
Its implementation has not yet run on the target GPU environment.

## Limits that block acceptance

- No trained three-task checkpoint or native pilot yet; upstream API compatibility
  and task contact semantics are unit-tested with doubles, not physically validated.
- The same-scene counterfactual dataset/labeler and final frozen acceptance runner
  remain pending. Their absence prevents a task-conditioning or three-task GO claim.
- A task ID or semantic vector is not evidence that the model uses task semantics.
  Separate scenes can allow background shortcuts. E2 is mandatory.
- There is no heldout execution switch. Add one only after the stage-one GO and
  a reviewed, frozen task-exposure protocol. Metadata candidates are not certified
  unseen tasks, and X-VLA declares LIBERO training.
