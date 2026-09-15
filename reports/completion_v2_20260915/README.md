# Completion v2: corrected labels, hard negatives, fresh acceptance

**Result: NO-GO.** The corrected candidate sharply reduces false completion, but still declares four placements complete before the required stability interval and misses 416 of 812 stable positive clips. It is not approved to control stopping in the live agent.

## What changed

The private simulator label now requires containment, no contact by either finger pad, basket support, linear speed <= 0.03 m/s and angular speed <= 0.3 rad/s at **11 consecutive 20 Hz samples spanning 0.5 seconds**. The RGB model receives none of these simulator fields.

The new model consumes three times and two views, with a spatially pooled ResNet18 and incomplete / complete / unknown outputs. It starts from ImageNet weights. It does not continue training the v1 checkpoint. See the [implementation and protocol](../../VISUAL_COMPLETION_V2.md).

Only the original 30 training and 10 validation trajectories were used for fitting and selection. The old 10 test trajectories / 485 frames were excluded. Each development episode gets a uniform 60-control open-gripper settling extension; paired hard negatives retain the arm/camera context while changing the target state.

| Development split | Clips | Incomplete | Stable complete | Hidden / unknown | Containment-positive controls corrected to incomplete |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train: 30 episodes | 2,257 | 1,500 | 307 | 450 | 621 |
| Validation: 10 episodes | 747 | 492 | 105 | 150 | 201 |

The **822 label corrections count individual simulator controls**, while the clip columns count sampled temporal examples. Physical negatives cover actual held/unsettled states, raised target, wrong object, target moved outside, and a target leaving only in the final frame. Camera blackouts teach abstention. The training split includes 600 paired physical interventions and 450 camera-blackout clips.

## Frozen training and genuinely fresh trajectories

- Real RTX 5090 forward/backward/save/reload smoke passed before full training.
- Full training: **40 epochs / 3,800 optimizer steps**, one training seed. The recorded training interval was 103.39 seconds, excluding data preparation and fresh VLA collection.
- Validation-only selection chose **epoch 25**. Its validation result already had 2 false completions / 492 negatives and 76 / 105 positive recall (72.4%). Candidate selection prioritizes false-complete count, then missed-complete count, then cross entropy.
- Frozen v2 checkpoint SHA-256: `80dc22b4e646bb8647a5e211b407ac198a1c64ea5db9db71a309c69ccfcd54c7`.
- Thresholds fixed before holdout: P(complete) >= 0.95; P(incomplete) >= 0.90; otherwise unknown.
- After freezing, collected **20 fresh procedural X-VLA trajectories**, seeds 2026091700 through 2026091719. Packaged fixed initial states were disabled. The collector rejected layouts duplicating any original demo or another fresh episode.
- Each new trajectory contains 300 X-VLA controls plus the same 60-control open-gripper tail. Collection does not consult the candidate and records native termination without auto-resetting the simulator.
- Evaluation contains **1,420 natural trajectory clips + 380 paired physical failures + 285 camera blackouts = 2,085 temporal clips**. Each clip has three times and two views; these are not 2,085 independent trials.
- One trajectory, `fresh_05`, never supplies a stable positive. It remains in the results and receives no stable-context interventions. Nineteen trajectories supply five contexts each, giving 95 examples per intervention category.
- No candidate or threshold changes followed inspection of these results. These 20 trajectories are now consumed as a test set.

## Paired results against corrected truth

Both models see the same heldout observations. The v1 baseline retains its epoch-4 checkpoint and 0.1 / 0.9 thresholds; its 128-pixel images are rendered directly from the same original-resolution views using its original preprocessing.

| Metric | Frozen v1 baseline | Frozen v2 candidate |
| --- | ---: | ---: |
| False complete, all physical negatives | 417 / 988 (42.21%) | **4 / 988 (0.40%)** |
| False complete, natural trajectories only | 109 / 608 (17.93%) | **4 / 608 (0.66%)** |
| Confirmed stable positives | 812 / 812 (100%) | **396 / 812 (48.77%)** |
| Stable positives missed | 0 | **416: 316 incomplete + 100 unknown** |
| Total unknown outputs | 155 / 2,085 | 390 / 2,085 |
| Hidden inputs receiving a confident answer | 177 / 285 | **0 / 285** |

False-complete count falls 99.04% in this paired set, at a substantial recall cost. The predeclared gates require **zero false completions and at least 80% complete recall**, so both of those gates fail for v2. Abstention on blackout inputs passes. Code/CI success does not change this model acceptance result.

| Negative category | Examples | v1 false complete | v2 false complete |
| --- | ---: | ---: | ---: |
| Actually held | 270 | 3 | 0 |
| Other incomplete natural states | 270 | 41 | 0 |
| Inside but not yet continuously stable | 68 | 65 | **4** |
| Target above basket | 95 | 23 | 0 |
| Wrong object in target position | 95 | 95 | 0 |
| Target moved outside | 95 | 95 | 0 |
| Target leaves in final temporal frame | 95 | 95 | 0 |

## Remaining failure and interpretation

All four false completions have a currently released, supported, slow target. Earlier controls in their half-second windows exceed the motion limit, so they have not yet accumulated a valid stability interval.

| Fresh episode | Control | P(complete) | First subsequent strict completion | Premature by |
| --- | ---: | ---: | ---: | ---: |
| fresh_01 | 140 | 0.988264 | 145 | 0.25 s |
| fresh_03 | 140 | 0.999938 | 142 | 0.10 s |
| fresh_10 | 145 | 0.982496 | 146 | 0.05 s |
| fresh_18 | 145 | 0.999999 | 148 | 0.15 s |

Four of 20 trajectories contain at least one false-complete shadow output. This is not a live stop rate: the candidate did not control these trajectories. The natural holdout provides correlated clips, and many positives occur after an already completed placement or during the uniform settling tail.

Observed: the targeted spatial failures are rejected in this holdout, while near-completion timing and stable-positive recall remain poor. Interpretation: three sparse visual samples do not reliably establish the required continuous stability interval, and the candidate is conservative on fresh VLA states. This does not establish which architectural or data change will fix those errors.

The next separate experiment should add development-only examples spanning release and settling transitions, diversify correctly stable VLA positives, and evaluate a denser visual history plus explicit temporal confirmation. Thresholds must be chosen on development data. Any resulting candidate needs **another new holdout**, followed by candidate-controlled end-to-end trials if it passes. Simulator truth must stay out of the RGB decision path.

## Validation and retained evidence

- 77 targeted local regression tests passed: the prior 66 plus 11 strict-label, temporal, abstention and seed-contract cases.
- Independent standard-library replay checks **15,558 recorded simulator controls**, all 2,085 predictions, split exclusion, threshold decisions, validation selection, source/manifest hashes and the freeze-before-generation receipts. Replay passes; model acceptance remains NO-GO.
- [Full CI passed on `29eef20`](https://github.com/Yanagisawa2002/ActionStream/actions/runs/34938998212): 134 core/runtime tests, 45 downstream contract tests against pinned upstream, and 103 upstream tests. These suites overlap; do not sum them as unique tests.
- A sibling CI run on `2de6eba` exposed a real-IPC timing flake: the 150 ms test deadline also expired on a healthy recovery reply. Follow-up `29eef20` uses a bounded 1-second test deadline and retains exact one-timeout / two-process assertions. Production timeouts and model artifacts are unchanged. The locked CI environment validates the transport regression; local minimal environments lack its optional LeRobot dependencies.

Run the independent evidence audit from the repository:

```bash
python reports/completion_v2_20260915/verify_evidence.py
```

Files:

- [Machine result](evaluation/summary.json), [raw predictions](evaluation/predictions.jsonl), [independent replay and error details](evidence_replay.json).
- [Training freeze](training/frozen.json), [40-epoch learning curve](training/metrics.jsonl), [smoke receipt](training-smoke/frozen.json).
- [Development manifest](development/manifest.json), [fresh trajectory manifest](fresh_holdout/manifest.json).
- `development/private_truth.tar.gz`: all 40 development per-control truth journals and sampled labels.
- `fresh_holdout/private_truth.tar.gz`: all 20 fresh per-control truth journals, labels and action journals. Each member's original byte hash is verified against its manifest; archives avoid an unreadable bulk text diff.
- [Exact launch commands](launch.sh); the training, collector and evaluator bytes correspond to implementation `2de6eba` and are identified by hash in the manifests.

Full RGB shards, simulator state arrays and checkpoints remain under `/root/autodl-tmp/actionstream-agent-20260915/completion-v2` on the existing GPU server. They are not committed to GitHub. The compact local replay checks their recorded provenance; it does not re-render images or re-run the remote weights.

This is one task, one training seed, simulated fresh VLA trajectories and controlled interventions. Complete camera blackouts do not establish natural-occluder robustness, and displaced states do not establish natural drop-dynamics coverage. It is an offline shadow evaluation, not a new live closed-loop acceptance. [V1 evidence](../completion_evaluation_20260915/README.md) remains frozen and unchanged.
