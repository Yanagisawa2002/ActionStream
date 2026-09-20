# ActionStream evidence index

This page keeps the detailed experimental status and evidence boundaries that are intentionally kept out of the README's first-pass review surface.

For the maintained runtime, start with the [repository README](../README.md), the [RPC hardening contract](rpc-runtime-hardening.md), and the [remote inference case study](case-studies/actionstream-remote-inference-case.md).

## Current evidence boundary

| Surface | Current status | What it establishes |
|---|---|---|
| Core lifecycle / stale rejection | validated CPU contracts + upstream compatibility CI | reset/stop generations, queue ownership and failure paths are explicit and tested |
| H1-R2 delayed-delivery pipeline | **GO** at frozen 950 ms simulated delivery | the serialized delivery wait was a real request-supply / queue-depletion bottleneck |
| H2 five-step compute budget | **NO-GO** | 61.6% fewer calls did not clear the frozen success gate |
| RGB completion + continuous confirmation | **GO only within one predeclared LIBERO Object task protocol** | 19/19 physically completed fresh trajectories stopped correctly; one VLA noncompletion was not falsely stopped |
| TCP loopback fault contract | **VALIDATED** | deterministic real-socket framing, deadlines, reconnects and lifecycle fault contracts in CPU CI |
| Remote transport replication v1 | **VALID_NEGATIVE / stopped after Run 1** | the original 5 s deadline failed against the observed connection-first cold path |
| V1 postmortem | **COMPLETED** | direct/RPC/SSH timing and retained traces localized first-use work and retry/backlog amplification |
| Remote transport replication v2 | **COMPLETED · PARTIAL SUPPORT · 7/8 hard gates · full gate set NO-GO** | 51 valid two-host runs, 47 task successes; original recovery gate remains FAIL |
| Persistent executor mechanism | **SUPPORTED within tested two-host X-VLA/LIBERO replication** | cold compute remained about 7 s; successful reconnect-first compute was 90/113 ms p50/p95 |
| External validity | **NOT ESTABLISHED** | historical cohort; outcome-informed v2 is not an independent v1 replication |
| Physical robot safety | **NOT EVALUATED** | simulation does not establish hardware safety or E-stop behavior |

The completed [v2 report](../reports/rpc_remote_transport_replication_v2/report.md)
and separate [final status](../reports/rpc_remote_transport_replication_v2/final_status.json)
record the results. The frozen v1/v2 preregistration configs keep their original
bytes and historical status fields; those fields are not the completed-result ledger.
V1 stopped at its valid negative first run. V2 completed all 51 declared runs
without rerunning valid outcomes or changing gates.

V2 recorded **100 injected disconnects, 99 actual reconnects and 97 demonstrated
later recoveries**. The original all-event recovery criterion is **97/100, FAIL**.
A separate post-hoc analysis finds 97/97 uncensored observable opportunities;
three terminal-edge events explain the missing observations but do not change
NO-GO. Under 950 ± 250 ms application delivery, **87.58–88.30% post-first-action
polling depletion** persisted despite high task success. All four v2 task failures
were Spatial; historical Object failures remain in the linked reports.

External-validity v1 in
[`configs/rpc_external_validity_v1.json`](../configs/rpc_external_validity_v1.json)
remains an **aborted, never-executed preregistration (NO_RUN)** after identity
collisions and incomplete historical coverage. UNKNOWN remains UNKNOWN.
See [remote-RPC evidence boundaries](remote-rpc-validation.md) and the
[technical case study](case-studies/actionstream-remote-inference-case.md).
Historical language/visual failures remain preserved rather than rewritten; see
[`docs/embodied-development-status.md`](embodied-development-status.md).

## Frozen GPU evidence

The current H1/H2 reports use X-VLA on an RTX 5090, three different LIBERO task
families, fresh runtime processes, disjoint reset identities and a fixed 950 ms
delivery delay.

### H1-R2 — delivery pipeline mechanism GO

H1-R2 changed one variable: whether the 950 ms response-delivery wait blocked the
single inference worker.

| Metric | Serialized aligned | Pipelined aligned |
|---|---:|---:|
| Success | 10/15 | 13/15 |
| Mean steps, failures = 300 | 201.33 | 137.93 |
| Median request rate | 0.902/s | 10.110/s |
| Queue depletion | 46.6% | 0.0% |
| Inference calls | 159 | 1,261 |
| Steady process VRAM max | 4,844 MiB | 4,844 MiB |

Request supply increased 11.43x and all 1,407 depletion pulls disappeared. Mean
paired completion steps improved by 63.40 with 95% CI `[-118.20, -5.47]`.
Success increased by three episodes, but its paired CI crossed zero and Object
regressed from 5/5 to 3/5. The supported claim is the pipeline bottleneck, not
universal policy superiority.

### H2 — fixed compute budget NO-GO

H2 kept the pipelined runtime and changed the minimum request interval from one
to five controller steps.

| Metric | Unbounded pipeline | Five-step budget |
|---|---:|---:|
| Success | 14/15 | 13/15 |
| Mean steps, failures = 300 | 125.67 | 139.80 |
| Inference calls | 1,109 | 426 |
| Median request rate | 9.901/s | 3.462/s |
| Queue depletion | 0.0% | 0.0% |

The budget saved 61.6% of calls while keeping the queue supplied, but lost one
paired success and failed the preregistered success gate. It remains a measured
compute/success frontier, not a promoted scheduler default.

## Completion / confirmation case study

The September 15 frozen acceptance evaluates a separate RGB completion verifier
plus eleven-control continuous confirmation on one finite task: LIBERO Object 5,
tomato sauce into the basket.

On 20 previously unused procedural trajectories, 19 tasks physically completed
and all 19 stopped correctly. The remaining VLA trajectory never physically
completed and never received a success declaration. The run recorded zero
premature first stops, zero missed completed episodes, median confirmation delay
0.55 s, maximum delay 1.50 s, and 19/19 stable two-second post-stop continuations.

This GO belongs to the **adapted verifier + continuous confirmation** combination.
It is not permission to stop on one raw model output, and it does not establish
other tasks, objects, natural occlusions, real robots or rare-error rates.

## Evidence and release discipline

Tracked evidence contains aggregate tables, paired effects, compact figures,
representative media, frozen protocols and content hashes. Raw traces, full
episode videos, model weights, simulator assets and machine-local archives remain
outside ordinary Git when licensing or size makes that necessary.

Negative results stay visible. H2 remains NO-GO; earlier language/visual failures
remain historical records even after the later single-task completion acceptance.
New GPU experiments must use untouched identities and freeze protocol changes
before outcomes are collected.

## Limitations

- H1/H2 cover one X-VLA checkpoint in LIBERO simulation.
- H1's 950 ms condition was simulated in-process delivery, not actual network
  jitter; v2 measured SSH-forwarded two-host TCP with application-level faults.
- V2 is completed with partial support and full hard-gate NO-GO. The historical
  cohort and coupled architecture/budget revision do not establish external validity.
- Roughly 88% post-first-action polling depletion at 950 ms remains a strong
  negative queue result; high task success does not establish healthy supply.
- TCP client deadlines do not terminate already-running remote CUDA kernels.
- The completion acceptance covers one finite simulated Object task and one
  trained checkpoint.
- No physical robot, hardware safety controller or E-stop claim exists.
- Short deterministic transport tests are not a production soak, CUDA-OOM test or
  GPU-driver recovery result.

