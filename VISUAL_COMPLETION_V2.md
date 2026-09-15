# Visual completion v2: released and stable

This is a separate candidate. V1's frozen evidence and its NO-GO result remain available in [the previous evaluation](reports/completion_evaluation_20260915/README.md).

## Completion semantics

Containment alone is insufficient. At every 20 Hz simulator control the private evaluator records:

1. Target satisfies the original containment predicate.
2. Neither finger pad contacts the target (stronger than only rejecting a bilateral grasp).
3. The target contacts the basket.
4. Target linear speed is at most 0.03 m/s and angular speed is at most 0.3 rad/s.
5. All four conditions hold at eleven consecutive samples spanning 0.5 seconds.

Missing controls, release/regrasp, loss of support, excessive motion, and leaving the basket invalidate the history immediately. These simulator fields only create labels and evaluate results; they never enter the learned model.

## Candidate

Shared ImageNet-initialized ResNet18 features from three times and two cameras. Images use the same vertical flip in both views, at 192 pixels. A 2 by 2 spatial pooling grid retains more object-location information than global average pooling. The head consumes all six feature maps and predicts incomplete / complete / unknown.

Temporal offsets are 10, 5 and 0 controls. The candidate needs a full half-second RGB history; the old single-frame checker interface cannot directly substitute it.

Frozen thresholds: complete probability at least 0.95, incomplete probability at least 0.90, otherwise unknown. This is a three-class distribution; the latter threshold is not a threshold on completion probability.

## Development data

Only the original 30 training and 10 validation episodes are used. The previous 10 test episodes / 485 frames are excluded from fitting and candidate selection.

Each development trajectory receives the same 60-control open-gripper settling extension. This produces supported stable positives rather than relabeling an image as stable without physical evidence. Actual pre-release and unsettled states remain negative.

Five stable contexts per episode produce paired interventions with the arm and cameras unchanged:

- Target above the basket.
- Target and BBQ sauce positions exchanged.
- Target moved outside.
- Target leaves only in the final frame of a previously positive temporal clip.
- One or both camera streams fully hidden, labeled unknown.

All physical negative interventions must fail containment in the rendered state. Complete-camera occlusion is a controlled sensor-failure proxy; it does not establish natural-occluder coverage.

## Selection and fresh acceptance

The pre-execution [protocol](configs/completion_v2.json) fixes 40 training epochs, seed, architecture, loss, thresholds and selection. Cross entropy uses inverse class-frequency weights. Selection minimizes validation false-complete count, then missed-complete count, then cross entropy. No holdout predictions are available during selection.

After the checkpoint is hashed and frozen, collect 20 new procedural LIBERO rollouts at the declared unused seeds. Disable the packaged fixed initial states and reject duplicate old/fresh object layouts. Each trajectory receives 300 controls from the pinned X-VLA policy and a uniform 60-control open-gripper settling tail.

The candidate is absent during collection. Native containment termination is recorded rather than resetting the offline simulator, so released-and-stable ground truth can be observed after containment. These are fixed-horizon data-collection rollouts, not candidate-controlled live closed-loop trials.

The fresh set also gets the predeclared paired interventions. V1 and v2 are scored on identical heldout observations against the corrected truth. Acceptance requires zero false-complete outputs, at least 80% complete recall, observable positives and negatives, and abstention on hidden-camera inputs. An all-unknown model fails. No retraining or threshold changes follow holdout inspection in this run.

## Execution

The existing RTX 5090 and cached models are reused. The training environment handles data development and training; the separate LeRobot runtime generates fresh VLA data. Real-data optimizer/backward/checkpoint-reload smoke precedes training.

Entry point: scripts/engineering/completion_v2.py with phases prepare, train, fresh, evaluate and explicit --base / --protocol arguments. Training smoke uses train --smoke. Evaluation directories are single-use to preserve the first results.

Full datasets, simulator states and weights remain under the remote completion-v2 directory. Compact protocols, manifests, hashes, learning curves, per-example predictions and result summaries are retained in the repository after completion.

## Status

Implementation and label-regression tests are ready. Data generation and the sequential training/evaluation run are in progress. This document does not grant runtime acceptance.
