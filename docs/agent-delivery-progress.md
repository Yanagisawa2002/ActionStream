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

The full archive was downloaded to independent storage: all 282 file hashes and
all 21 original external identities passed. Hash-pinned receipts and the optional
full-bundle verifier are in [the delivery report](../reports/agent_delivery_20260915/README.md).

## Frozen native acceptance

Runtime source `4570e7caa9fc2e2fcce6094ee40917833f31393c` was frozen before
twenty new seeds. The normal group safely completed 10/10 tasks, but exceeded
the 50 ms work budget on 102/2,187 controls (4.66%). Every episode exceeded the
1% miss-fraction limit, so **asynchronous timing is NO-GO**.

The recovery group proved first-attempt physical failure in 10/10 scenes and
recovered in 9/10, passing its at-least-8/10 threshold. The remaining case
exhausted its bounded two attempts without claiming completion. Neither group
had premature stops, missed completion events or journal integrity errors.
This recovery result is limited to the declared simulated gripper fault.

Full results, independent state restoration and raw evidence are described in
[the acceptance report](../reports/async_agent_20260915/README.md). Overall
acceptance remains **NO-GO** because the timing requirement is not met.
