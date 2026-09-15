# Agent delivery, asynchronous timing, and physical recovery

This work starts from finite-Agent PR #5, commit
`42e0c74fb933edb8fd955fa8b113e9a5dafaf3ea`. Historical source snapshots,
checkpoints, protocols and accepted/failed results remain frozen.

## Ordered gates

1. Reproducible delivery: align the locked Python environment with measured
   inference dependencies; remove the ambient BART cache requirement; verify
   the 21 external evidence blobs; run the installed single-request CLI with
   isolated caches and both supported and blocked language.
2. Asynchronous timing: connect the actual ActionStreamInferenceEngine to
   the finite Agent; freeze timing, freshness, cancellation and execution
   gates before new native evaluation. Retain every deadline miss.
3. Physical failure recovery: provide a bounded observable recovery state
   machine, then freeze and evaluate actual simulated failure trajectories.
   Software interruption is not evidence of physical recovery.

## Reproducible delivery measured on 2026-09-15

The clean Python 3.12 environment passed the runtime lock check on RTX 5090.
All 19 model/tokenizer files and 585 LIBERO files matched their pinned manifests.
The installed supported CLI exited 0 with independent physical completion,
0.30 simulation seconds of confirmation delay, and stable post-stop continuation.
The unsupported book request exited 2 with `backend_created=false` and no
trajectory. Both runs left their separate offline caches empty.

All twenty consumed historical seeds reproduced their original trajectory
bytes, closing the original external-file gate at **21/21**. New run logs are
retained separately from the historical records. The 623,243,676-byte delivery
archive has SHA-256
`f3912c2129d18527dbf9c66ce4255154e0da30589395f1cdee2b6343ada21be1`.

Asynchronous timing and physical recovery implementation has passed automated
checks; native acceptance has not yet run. It will use the separate frozen
protocol and new seeds after delivery backup verification.
