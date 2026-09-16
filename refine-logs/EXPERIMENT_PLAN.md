# ActionStream: three-task integration, then task generalization

Status: **development implementation; no three-task native acceptance yet**.
Protocol: `configs/multitask_development_v1.json`. Historical timing NO-GO remains
in `reports/async_agent_20260915`; the later single-task budget result does not
validate this new model or these new tasks.

## Two claims and the evidence each needs

1. **Multi-task integration:** one authorized language interface and one shared,
   task-conditioned visual completion model can drive Object 5, Spatial 7 and
   Goal 2 through paced execution, visually confirmed stopping and bounded retry.
   All three tasks must independently pass new native normal and fault trials.
2. **Local task transfer:** after that implementation, model and protocol are
   frozen, the completion/agent stack transfers to tasks never used for our
   training, tuning or model selection. This does not establish an upstream VLA
   zero-shot capability. Failed heldout trials remain failed; tuning consumes them.

## Tasks and boundaries

| Family | Development integration task | Why it needs separate acceptance |
|---|---|---|
| Object 5 | Tomato sauce into basket | Preserve containment, release, support and stability |
| Spatial 7 | Black bowl **on the stove** onto plate | Two bowls; source qualifier cannot be dropped |
| Goal 2 | Wine bottle onto cabinet top | Goal is a site; actual contact with cabinet is also required |

These exact policies ran in historical H1-R2. That is backend experience, not
evidence for the new parser, completion model or recovery chain.

Provisional heldout candidates: Object 1 (cream cheese into basket), Spatial 5
(bowl on ramekin onto plate), Goal 9 (wine bottle onto rack). Only pinned catalog
and BDDL **metadata** were inspected in this task. Their trajectories, images,
labels and results are not opened. Complete the historical-use audit before
declaring these tasks reserved; replace any previously consumed candidate before
the reservation is frozen. The runtime and training loader currently reject all
tasks outside the three-task development registry.

## Model and information boundary

- Qwen at revision `89644892e4d85e24eaac8bacfd4f463576704203` proposes a task and
  verbatim entity quotes. A host grammar covers the entire request, including
  relation and source qualifier, before simulator construction.
- The same frozen Qwen encodes the host canonical instruction as a masked mean
  of its last hidden layer, L2 normalized. This runs once before control. It sees
  text only. Cache the condition and its hash; release Qwen before VLA execution.
- Train a shared ResNet18 temporal visual encoder and a text/vision fusion head.
  Inputs: three times at offsets 10/5/0, two 192×192 RGB views, and the public text
  embedding. No one-hot task IDs, simulator state, reward, `done`, success flags,
  segmentation, contact or ground-truth labels are model inputs.
- The existing 11-frame visual history and 11-positive confirmation window,
  observation age checks, queue invalidation and at-most-one retry are reused.
- Privileged target/contact/goal/speed data is recorded separately. Segmentation
  for label observability is replayed **after** collection, outside the control
  loop. The visibility threshold is a proxy, not proof that completion is visible.
- The BDDL hashes bind the expected native tasks. Offline completion requires
  the correct goal predicate, released target, actual destination-body contact,
  low linear/angular velocity and 11 consecutive stable states.

## Five experiment blocks

### E0 — integration and native sanity (must run first)

Local adversarial tests: wrong task, wrong quote, missing bowl qualifier, wrong
relation, added goal, forged permit, foreign condition, split leakage, recovery
revision and post-stop dispatch. Native pilot: one development rollout per task;
inspect authorized instruction, exact native task, output contract, condition
shape, independently restored physical labels, and startup/step timing.

Stop if any routing/label discrepancy or online privileged-input path exists.
Do not start a training batch until all three native pilots work.

### E1 — guarded data and training (must run)

Initial pilot allocation per task: 8 training and 4 validation trajectories,
including successful, unsuccessful and open-gripper interventions. Split whole
episodes using `(task_key, initial_layout_sha256)`; exclude historical and current
consumed layouts, including earlier acceptance. Never split frames of one rollout
across folds. If positive/negative coverage is missing, explicitly expand only
development data and record the consumption before fitting.

Freeze Qwen and VLA. Train shared vision/fusion parameters for 16 epochs with
AdamW, learning rate 2e-5, weight decay 1e-4, seed 20260916, batch size 8. Sample
RGB clips every 5 controls. Train blackout examples as unknown. Select only on
the mean of per-task validation cross entropy; record per-task false-complete
counts and recall. Offline selection is not a stopping-safety pass.

### E2 — does the visual model use the goal? (must run before freezing)

Compare conditioned model with a separately trained no-task ablation, on identical
development splits. Add matched **same-scene** counterfactual goals with offline
predicate labels: correct object/wrong destination, distractor object at the
destination, target held above support, near-goal but unreleased, occluded target,
and blacked-out cameras. Never label an arbitrary task swap false just because its
task ID differs; evaluate the substituted goal or mark it unobservable/unknown.

The counterfactual corpus/labeler is **still to be implemented and audited after
native scene sanity**. It must not use the reserved tasks. Allocate different
development counterfactual goals if a proposed probe overlaps a reserved goal.
Require zero confident false completions on audited invalid/ambiguous probes and
report paired task-change behavior. If text has no measurable effect, restrict
the claim to multi-scene support and do not call task conditioning validated.

### E3 — fresh three-task acceptance (must run after development freeze)

Freeze source SHA, environment, assets, condition encoder, trained checkpoint,
language grammar/prompt, thresholds, protocol, seed list and all consumed layout
hashes. Reserve **10 normal + 10 physical-fault trials per task** only then.
New acceptance runner/freeze receipt remains pending E0–E2; the development CLI
deliberately offers no acceptance switch.

Use the existing per-task gates: ≥8/10 safe normal completions; 10 eligible
physical failures and ≥8 recoveries; zero premature stops, missed completion
events or integrity errors; confirmation delay ≤2 s in simulated and wall time;
normal dispatch p95 ≤55 ms, maximum ≤100 ms, work-budget misses ≤1% **per episode**.
Apply open-gripper clamp for controls [0,300), allow one bounded retry in the same
physical scene, and observe 40 post-stop controls. Preserve every denominator,
including ineligible faults, errors, horizons and no-completion cases.

Independently restore states in a fresh simulator and verify strict label identity,
RGB clip identity and task binding. Report each task separately; a pooled average
cannot hide a failing task. No seed replacement, repeat-until-pass or relaxed gate.

### E4 — heldout-task transfer (only after E3 GO)

Lock the final task partition and exposure ledger before accessing heldout samples.
Permit metadata-only semantic registration and text encoding without fitting;
record that host metadata integration separately from learned-model transfer.
No new model/head/threshold/selection/grammar tuning from heldout observations.
Evaluate the frozen system, report failures, uncertainty, per-task metrics and
upstream exposure. If subsequently adapted, reclassify these tasks as development
and reserve new tasks before the next transfer claim.

## Provenance and claims

The [pinned X-VLA model card](https://huggingface.co/lerobot/xvla-libero/blob/12e8783e996944f5c97e490d37d4c145484ed70a/README.md)
declares LIBERO training and names `HuggingFaceVLA/libero`. Exact per-task overlap
is not established. Qwen's task-specific exposure is also unknown. Record both;
absence of local training is not proof of absence from pretraining.

This is simulation. The project does not have a physical robot. Statistical
50 ms behavior under measured workload is not a hard-real-time guarantee.

## Resources and milestones

One Linux CUDA GPU matching the locked environment is sufficient for the planned
sequence. The previous server is shut down. E0 supplies actual seconds/episode,
text-encoding memory and training throughput before a spending estimate. Planning
range for E0–E3 is a few GPU hours, not a measured budget; retained RGB/state
trajectories can require several GB. Stop after each failed milestone.

M0 local implementation/tests → M1 native pilots/label replay → M2 data/model and
counterfactual gate → M3 freeze/fresh three-task acceptance → M4 reserved-task
transfer. Current completed work is M0; M1 requires a live GPU endpoint.
