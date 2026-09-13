# DISPATCH_004: finite text-backed authorization and independent vision diagnostic

The starting v1 trial remains A=FAIL (12/12 supported, 2/12 negative false
accepts), B/C=NOT_RUN. It is preserved verbatim for the new paired comparison.

V2 removes generated canonical instructions and redundant skill slots. An
accept generates only `decision`, `target_quote` and `destination_quote`; a
reject/unknown generates `decision` and a bounded reason code. The host checks
exact unique quote spans in the immutable original utterance, matches the
predeclared entity alias registry and applies conservative negation/meta guards.
Only then does it construct `GroundedTask` and assemble the existing native
control spec from the unchanged skill registry.

Each executable v2 task binds request ID, original text/SHA256, raw model
output/SHA256, contract SHA256 and both mention spans. `execute_authorized`
recomputes this binding before touching the native port. A v1 accept cannot be
automatically promoted into a v2 authorization. These are trusted application
boundaries, not cryptographic authentication of arbitrary Python callers.

The public contract in `configs/llm_vla_dispatch004.json` declares all aliases,
normalization, lexical guards and reason codes. Exact snippets must retain
original case/spelling and occur once; alias normalization only casefolds,
collapses whitespace and removes one leading article. Context-free pronouns,
generic sauce and unnamed containers are not entities. Repeated ambiguous
quotes fail closed. Explicit negation/meta tokens are conservatively outside
this finite interface, which can reduce recall outside the declared scope.

Mention validation does **not** establish correct direction, argument roles,
absence of extra goals or semantic entailment. A forged role-reversed model
claim with real entity mentions demonstrates that residual limitation in a CPU
test. Real model quality therefore remains a separate gate; format/evidence
success is never claimed as complete language understanding.

Before the first new inference, freeze the new 48-case challenge (24 supported,
24 negative), family design, public inputs, scorer-only answers, v2 contract,
schema/prompt and source hashes. Run one v1 and one v2 call per shuffled new case,
then v2 on the already-seen old 24. No answer/prompt/alias/code tuning after
seeing outcomes. `repair_scoring.py` independently routes v1 raw output through
the old validator and v2 through the new one. Report both raw and admitted
decisions, schema validity, failures, unknowns, recall and false accepts by family.

The same old 20 real paired snapshots and checker prompt form a separate
`DIAGNOSTIC_ONLY` shadow run, without robot execution or a language prerequisite.
`grounded_native` still checks **both** intact language and visual gates before
loading any model or environment. It reuses the existing native runner/port and
bounded controller rather than duplicating a rollout implementation.

All actual model/native children share the new DISPATCH_004 1,200-second ledger.
The supervisor's `--dispatch 004` mode owns a separate store, reuses the old
Qwen files read-only, and permits at most one comparison, shadow and native
phase. New downloads are capped at 256 MiB and incremental storage at 1 GiB;
no new model/dependency is planned. The fixed Qwen parameters, native checkpoint,
processors, action semantics, reset states and 20 Hz control remain unchanged.

The actual v1/v2/vision results, conditional native status, CPU XML, provenance,
latency/usage and final local commit are delivered in the task artifact
`artifacts/embodied_llm_vla_grounding_20260913/REPAIR_REPORT.md`. This protocol
authorizes only this finite trial. Any further design/model choice requires the
coordinator's next instruction; remaining GPU budget does not authorize retries.
