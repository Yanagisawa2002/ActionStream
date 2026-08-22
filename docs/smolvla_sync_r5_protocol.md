# SmolVLA sync R5: frozen one-hypothesis protocol

Status: **frozen locally; not run because the GPU server is off.**

## One hypothesis

SmolVLA v4 predicted 50 actions and executed all 50 before observing again. It
passed 4/6 frozen zero-delay resets: Object 2/2, Spatial 1/2, Goal 1/2. R5 asks
only whether a 10-step receding execution horizon repairs that sync gate. It
does not change the 50-step predicted chunk, checkpoint, task, observation,
action space, LeRobot revision, or success predicate.

This is a controller-contract hypothesis, not an RTC result. Shorter horizons
will request inference more often; a pass therefore establishes stability, not
compute efficiency.

## Sequential stop rules

1. Re-run only the two known v4 failure identities with the 10-step horizon:
   Spatial task 0/state 29 and Goal task 5/state 30, preserving their original
   policy seeds. If either fails, stop and record `NO_GO_REPAIR`.
2. Only after 2/2 repair, run Object task 3, Spatial task 0, and Goal task 5 on
   states 48 and 49 with new seeds. These six exact reset identities had zero
   matches in 669 prior episode rows and zero prior protocol reservations.
3. The unseen gate is exactly 6/6. Any failure is `NO_GO_CANARY`; do not tune
   the horizon, substitute tasks, retry a consumed identity, or run RTC.
4. A 6/6 pass only permits a separately frozen official RTC experiment. It
   must not be presented as RTC performance.

Every task family runs in a new process with two actual-worker warmup calls,
`nvidia-smi` residency, request/s, video capture, and immutable trace hashes.
The orchestrator refuses to start the canary without a passing repair receipt.

The freeze receipt is
[`configs/actionstream_backend_gpu_smolvla_sync_r5_freeze_receipt.json`](../configs/actionstream_backend_gpu_smolvla_sync_r5_freeze_receipt.json).
