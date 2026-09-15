# Finite Agent P0: language-to-VLA integration acceptance

**GO for the frozen, synchronous single-task simulation integration.** Twenty new supported requests were accepted and their twenty fresh procedural trajectories completed with confirmed stopping. Twenty unsupported language-only requests were blocked. The current result covers the complete finite application boundary; it preserves the earlier model, thresholds and confirmation policy.

## What was connected

`actionstream-agent` now performs real Qwen parsing → immutable whole-request authorization → official native X-VLA action conversion → three-time/two-camera RGB verification → eleven-consecutive-control confirmation → discard pending actions → hold/open and score forty additional physical controls.

The CLI and acceptance runner call the same `parse_request` and `execute_parsed` application functions. The runtime controller imports no simulator truth or evaluator labels. Native containment is recorded only in the private port; the runtime continues through release and stability. Unknown evidence resets confirmation while execution remains within the frozen 300-policy + 60-settling-control budget. Historical v1/v2 entrypoints and evidence remain unchanged.

The batch acceptance parsed all forty requests once, unloaded Qwen, and then executed the twenty authorized task requests. The single-request CLI performs the same stages for one request. This separates GPU residency without changing original-text authorization.

## Freeze and independent data

- Implementation/configuration committed and pushed before execution: `4b653fe180e3f45fb48d42b52965c5dbad33a3ee`.
- Checkpoint SHA-256: `28482b40e470d932dfe44312d2dbcb7fa732146a2b2f9c7ebf7683b47cdfa7f9`.
- [Freeze receipt](freeze.json) SHA-256: `6c75702af5f2439cd8b50ef7747d9bfa6f8b5bcd0beae3cae48eb321c86d7cdf`.
- Unchanged input history 10/5/0, two vertically flipped 192-pixel camera views, complete threshold 0.95, incomplete threshold 0.90, eleven positive controls spanning 0.5 simulated seconds.
- New procedural seeds `2026092000`–`2026092019`. Initial object layouts were checked against 102 consumed layouts: fifty original demonstrations, twenty v2 holdouts, twelve development controls and twenty v3 holdouts; no duplicate was admitted.
- Twenty newly authored supported English requests, plus twenty unsupported language-only challenges. Host grammar, aliases, Qwen weights/prompt and completion combination were fixed before inference. The protocol's labels and seeds were absent from model prompts.
- No training, candidate selection, instruction/seed replacement, threshold changes or model/policy performance retries occurred.

The [protocol](protocol.json) declares every request, seed, gate and exclusion rule. The [smoke](smoke.json) uses only an already-consumed development RGB clip before new data. Source snapshots and prior-manifest hashes bind the actual run; runtime assets retain their pinned hashes and upstream terms.

## Primary results

| Metric | Predeclared gate | Observed |
| --- | --- | ---: |
| Supported requests accepted | At least 18/20 | **20/20** |
| Unsupported requests admitted | Zero | **0/20** |
| Model/schema errors | Zero | **0/40** |
| Requested tasks safely completed | At least 16/20 | **20/20** |
| Premature first stops | Zero | **0/20** |
| Missed whole completed events | Zero | **0/20** |
| Confirmation delay | Maximum <= 2 seconds | **Median 0.425 s; maximum 0.85 s** |
| Post-stop stability | All forty controls at every stop stable | **20/20 for 2 seconds** |
| Missing/invalid execution evidence | Zero | **0** |

There were 3,144 recorded controls before confirmation and 800 post-stop controls, totaling **3,944 recorded physical control calls**, plus twenty recorded reset states. Native reset warmup is outside these controller-call counts. Mean confirmation delay was 0.4725 seconds, sample standard deviation 0.1697 seconds. All twenty required task requests remain in the denominator.

The previous [completion-only acceptance](../completion_acceptance_v3_20260915/README.md) measured 19/20 task completion and median/max confirmation delays 0.55/1.50 seconds on different layouts. This run demonstrates integration with real language authorization on another finite set. Different trajectories and stopping horizons prevent interpreting the numeric difference as a policy or model improvement; both weights were unchanged.

Full per-request outcomes and latency values are in [analysis.json](analysis.json). Qwen model-call latency median/max was 0.299/0.998 seconds. Agent episode wall time averaged 8.754 seconds, with maximum 31.099 seconds; this includes port construction on the first episode and evidence serialization. These figures do not establish a per-control wall-clock deadline.

## The accepted system still depends on both checks

### Language authorization

On the identical forty Qwen responses:

| Language decision layer | Supported accepted | Unsupported accepted |
| --- | ---: | ---: |
| Raw Qwen decision | 20/20 | **13/20** |
| Original-text references + whole-utterance authorization | **20/20** | **0/20** |

The thirteen raw false accepts include multiple objects, unsupported target/destination, unresolved references, extra actions, physical constraints and conditional requests. Host checks blocked each one. For example, “wash … then place” and “keep … upright” retain unresolved obligations; unsupported or invented entity quotes cannot authorize execution. A zero final false-admission rate does **not** establish reliable unconstrained Qwen semantics. The original-text/finite-grammar boundary remains part of the accepted system.

These are newly authored finite-contract challenges, not a natural-language generalization benchmark. The twenty unsupported cases were declared language-only before inference. Their false acceptance would fail the language gate; they were never used as a robot execution cohort.

### Visual confirmation

Only predictions before the actual first stop enter this diagnostic table; warmup and post-stop observations have no model prediction.

| Strict physical label | Incomplete | Complete | Unknown | Total |
| --- | ---: | ---: | ---: | ---: |
| Incomplete | 2,689 | **48** | 18 | 2,755 |
| Complete | 5 | **186** | 18 | 209 |

The raw model still gives 48 transient false-complete predictions. Continuous confirmation suppresses them: zero confirmed premature stops. Raw complete recognition is 186/209 (88.995%) before stopping; these truncated and correlated observations are a diagnostic, not a replacement event metric. The model alone must not trigger a stop.

## Independent evidence verification

The stdlib [verifier](verify_evidence.py) imports no production controller, authorization, model or scorer. It verifies archive/source hashes and freeze ordering, independently re-derives text authorization from raw Qwen output and frozen grammar, binds authorization receipts to original requests, recomputes strict completion from physical facts, derives class decisions from probabilities, replays continuous confirmation, matches physical actions against policy chunks/hold actions, recomputes all primary metrics and checks exact request/execution coverage.

Independent local replay **PASS / GO**: forty original language calls, twenty execution journals and 2,964 RGB predictions. A separate simulator audit restored **3,964 states**, including every post-stop state, and checked all **2,964** prediction clip hashes against saved RGB arrays. Twenty-seven contact/containment cache differences appeared after restoration, with **zero changes to strict completion labels**. All retained differences are available in the archive.

The verification-only audit was restarted twice while diagnosing runtime cost: first with limited BLAS/OMP/llvmpipe threads, then after fixing repeated decompression of the complete RGB array for every clip. The final audit loads RGB/states once per trajectory and performs the same checks. Initial partial records, source snapshots and [restart receipts](audit_cache_restart.json) are retained. Neither adjustment reran model inference or policy execution or changed the primary outcomes.

```bash
python reports/finite_agent_20260915/verify_evidence.py
python reports/finite_agent_20260915/analyze_results.py
```

The [lossless record archive](raw_records.tar.gz) is bound by [evidence_manifest.json](evidence_manifest.json), SHA-256 `51cb72c810e84a341b80cbee15dc39fa1761e8888b08063cd7602b13dc9bec61`. [External blob hashes](external_blobs.json) cover full RGB/state arrays and the frozen checkpoint retained on the existing server; the checkpoint also has the prior local backup. Readable protocol, verdict, analysis, execution receipt and independent replay remain outside the archive.

## Software validation and scope

- 98 focused local tests passed, including 31 new finite-Agent tests, plus existing authorization, completion, truth and confirmation tests. These scopes overlap with earlier reports and are not additive across runs.
- Ruff and strict public-release audit passed. The implementation's [complete CI passed](https://github.com/Yanagisawa2002/ActionStream/actions/runs/34956382718).
- CI additionally checks discovery of the installed `actionstream-agent --help` command and independently replays this acceptance evidence, alongside existing runtime, plugin and upstream LeRobot compatibility suites.

This closes P0 integration for one synchronous simulated tomato-to-basket task. The asynchronous inference engine, real-time deadlines, natural occlusions, recovery from failed placement, other tasks/languages and physical robots remain separate work. The twenty new trajectories are now consumed acceptance data. Future model, thresholds, confirmation or integration-policy changes require a new untouched acceptance set.
