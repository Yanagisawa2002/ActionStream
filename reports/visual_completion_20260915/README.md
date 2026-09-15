# RTX 5090 completion pilot: completed

The real-data CUDA backward/save/reload smoke passed, followed by **30 epochs and
720 optimizer steps**. This trains a compact dual-view ResNet18 completion model
for tomato sauce placed in a basket. It does not retrain the VLA action policy.

**Subsequent frozen evaluation:** [heldout, stress and closed-loop results](../completion_evaluation_20260915/README.md)
are now available. The candidate remains **NO-GO**: 2/435 heldout negative frames
were called complete and 16/20 controlled physical failure cases were missed.
The three real closed loops produced two confirmations and one unknown stop.
Containment-only training labels do not require release or stability.

Source implementation: `3d7fe381548ea564990e4834567b9d1c279a24dc`.
Its [GitHub CI passed](https://github.com/Yanagisawa2002/ActionStream/actions/runs/34930602674).
There were 16 targeted local tests; source hashes and runtime package versions
are in `receipt.json`.

| Split | Episodes | Frames | Complete | Incomplete |
|---|---:|---:|---:|---:|
| Training | 30 | 1,519 | 130 | 1,389 |
| Validation | 10 | 496 | 47 | 449 |
| Heldout, evaluated separately after training | 10 | 485 | 50 | 435 |

The frozen selection rule chose **epoch 4** by minimum validation loss (0.03006).
At the fixed completion threshold of 0.9, it incorrectly confirms 0/449 incomplete
frames and correctly confirms 43/47 complete frames. At the incomplete threshold
of 0.1 it incorrectly rejects one complete frame; 9/496 predictions fall between
the thresholds and abstain. Accuracy at the separate 0.5 decision threshold is
98.79%. These are frame counts from ten validation episodes, not independent
trials or a held-out test claim.

The final epoch is also retained in `metrics.jsonl`; the checkpoint selection was
not changed after inspecting results. The GPU training loop and validation took
21.32 seconds for this small pilot, excluding environment setup, downloads,
rendering, and smoke. Recorded peak allocated training memory was 686.43 MiB.

## Retained artifacts

- `data_manifest.json`: official source hash, exact rendered shard hashes and
  episode assignments. Labels are simulator success at the rendered state.
- `smoke_pass.json`: successful real GPU optimizer step and equal checkpoint reload.
- `metrics.jsonl`: all 30 epochs, including abstention and false-completion counts.
- `receipt.json`: implementation fingerprint, environment, completion status and
  selected checkpoint SHA-256.

Remote checkpoint:
`/root/autodl-tmp/actionstream-agent-20260915/runs/completion-v1/best.pt`

SHA-256:
`35c122539730783bc4c61a65d7515a075385a5ac81a0dd8fa2a34c445a55e7c5`

The sibling `last.pt` preserves epoch 30 and optimizer state. The successful
pipeline log is `logs/pipeline-4.log` under the same remote task directory. Earlier
logs retain the EGL, interrupted asset download, and legacy XML path failures.
The fixes installed the EGL dispatcher, fetched only the task's pinned assets,
and rebased obsolete demonstration asset paths.

## Status

Training is complete. The subsequent frozen evaluation consumed the test split,
ran controlled failure/camera interventions and three real simulator closed loops.
The candidate remains NO-GO; see the linked evaluation report above. Further
natural-occlusion, spontaneous-drop and post-stop stability coverage is still open.
