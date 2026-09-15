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

Current state: implementation in progress. No new model/simulator acceptance
has run. No server is to be shut down or released by this work.
