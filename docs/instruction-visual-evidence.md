# DISPATCH_005: complete instruction coverage and RGB evidence

This finite offline protocol starts at merged commit
`3d6aae49ff7a55897c4dde1dfebdf225fd672b32` (tree
`e5a1c3c001dd63d1cc0f5068cd50ace898b92f93`). It preserves the original
lifecycle/native code and all earlier failed experiments. No robot reset,
native episode, control action, recovery, training or remote inference is
authorized in this round. The coordinator handles publication separately.

## Frozen language method

`coverage.py` requires an entire utterance to match one explicit finite
production, then requires v2's model decision and exact entity-quote binding.
The original text is immutable and hashed. Every lexical span has an offset,
exact quote and disposition; accepted derivations additionally bind moved
object, stationary destination, quantity one, affirmative polarity and the
ordered same-object movement substeps to one atomic goal. Unmatched content
retains an unresolved whole-utterance obligation and prevents authorization.
This conservative unresolved representation does not pretend to extract every
semantic role from unsupported language.

The grammar supports direct movement, destination-fronted clauses, passive
goal requests and lift/carry/place sequences with locally bound pronouns.
Commas and final punctuation have explicit surface productions. Semicolons
can join supported same-object substeps; `and` remains part of the grammar.
It does not delete arbitrary conjunctions, physical modifiers or extra goals.
Negation is explicitly rejected. Unsupported quantities, conditions, roles,
entities and residual clauses remain unknown unless v2 already rejects them.
An explicit source such as `from the counter` is conservatively unverified;
`from where it is` introduces no extra scene-location assertion. `with care`
is an unimplemented physical constraint. Finite support costs recall.

The v2 model makes a semantic proposal, not a coverage certificate. One real
v2 call per original is interpreted by both v2 and v3 on the **same raw output**.
This isolates the host change; it is not two independent model experiments.
The new 48 authored cases contain 24 positives and 24 negatives covering
additional actions/objects, quantities, negation, order, role reversal,
conditions, ambiguity, constraints, quotation and natural paraphrases.
Old round-4 48 and round-3 24 cases are consumed regressions. Total: 120 real
text calls. Public inputs contain only opaque request ID and original text;
labels and families remain scorer-only. Code and cases were developed before
the one frozen model run; this is not a real-world held-out benchmark.

`CoveragePermit` binds original request, exact model output, v2 contract,
finite grammar and complete adjudication proof. `execute_covered` recomputes
it before calling any backend factory. Caller flags, altered text and forged
proofs cannot bypass this new application entrypoint. The old v1/v2 research
interfaces remain available for reproducibility; this is not a Python sandbox
against a caller directly importing an older API. Independently, dispatch005's
supervisor accepts only offline phases and cannot launch the native runner.

Gate: new >=20/24 accepted, old48 >=20/24, old24 >=10/12, with zero negative
admissions in every set. Report supported recall, false accepts, abstentions,
schema failures, reasons, host timing, shared model latency and peak memory.

## Frozen visual method and data boundary

The old 20 dual-camera snapshots are all consumed diagnostics. The five from
smoke_state45 form the calibration group; the 15 from development_state0/1/2
form episode-disjoint diagnostic evaluation (9 authoritative negatives, 3
positives and 3 unlabeled resets). No neighbouring frames cross this split.
Only smoke_state45's earliest reset RGB image supplies static target/opening
appearance references, jitter positives and distractor negatives. No future
frame supplies runtime context. Scorer annotations were made from inspected
RGB sheets before model output: approximate visible boxes, identity,
visibility and visible containment. They are one analyst's development
annotations, not independent blind human ground truth. Native success labels
are used only by the final outcome scorer.

The first 24 vision calls are causal diagnostics: one predefined observation
per episode, each with real dual view, uniform-gray blank views, swapped
cameras, earlier-reset wrist mismatch, a fixed [90,90,270,270] occlusion, and
center [60,60,300,300] crops. Prompt bytes and observation ID remain identical
within each six-call group. The crop changes resolution; the others preserve
shape. RGB hashes, actual processor tensor hashes, per-image tensor slices,
grid, dtype and shape are recorded. Same-grid changed pixels are expected for
same-size interventions; grid changes alone are not evidence of perception.
Blank/mismatch outcomes measure sensitivity, not labelled accuracy.

Then every old observation receives three calls: original, horizontal flip,
and deterministic target-box masking, 60 calls. Each receives the current two
views plus two explicitly static appearance reference crops, never filenames,
source frame numbers, source episode labels, future images or simulator poses.
The model proposes target and basket **opening** boxes in 0..1000 coordinates,
their identity, visibility and containment for both current views. A frozen
token-trie decoder constrains only JSON structure, number range and enum
vocabulary; it permits wrong boxes and any decision. No code fences from older
results are stripped or rescored. Maximum new visual generation is 320 tokens.

Independent checks require:

- Target and opening box validity, and IoU >=0.5 after inverse horizontal flip.
- HSV 8x4x4 histogram similarity to the two calibration RGB crops. Threshold
  is the midpoint of worst 2-pixel-jitter positive and best distractor score;
  nonseparable references are disabled with threshold 1.01. This checks color
  appearance, not semantic identity by itself.
- Target and basket identity must agree as match across original and flip;
  both regions must be visible or partially visible. Partial visibility still
  needs all independent checks; absent/unknown cannot establish containment.
- Mask each proposed target box with gray, padded by 10% (minimum 2 pixels).
  Invalid proposals cause full-view blanking. The response must change target
  visibility to absent/unknown and containment to unknown. There is no gold box
  in this intervention. Actual current-image tensors and RGB must change while
  prompt bytes remain the same. This is a necessary consistency test, not
  independent accuracy ground truth.
- Verified inside: >=80% target-box area inside opening box, center inside,
  and matching inside proposals in both views. Verified outside: <=5% overlap,
  center outside and matching outside proposals in a view with no verified
  inside contradiction. Otherwise unknown. Bounding-box overlap is a 2D
  surrogate; it does not prove physical 3D containment.

`EvidenceHistory` checks strictly increasing observed timestamps per opaque
stream and flags a verified complete-to-incomplete contradiction. Missing
reset timestamps produce unknown. Sparse captures do not establish continuous
tracking or post-release persistence. No frame index or success state enters
this history check; two current verified views can support the finite candidate
without inventing later frames.

Report localization IoU/center error, identity plus localization, visibility,
containment, all verification checks, causal output/input changes, false
complete, true complete recall and non-unknown coverage. Evaluation gate keeps
>=8 negatives, >=3 positives, zero false complete, >=80% labelled valid coverage
and >=3 true complete. All-unknown explicitly fails. A small-sample pass would
still not be production reliability or a fresh generalization result.

## Reproduction and resources

Use existing Ubuntu-24.04 on the local RTX4090, the frozen LeRobot environment
and Qwen3-VL-2B revision `89644892e4d85e24eaac8bacfd4f463576704203`.
No weights or dependencies are changed. Model weights remain at
`$BASE/dispatch003-llm-vla/assets/qwen3-vl-8964489`. X-VLA is not loaded.
Create a fresh owned `$STORE=$BASE/dispatch005-instruction-visual` with scope
`DISPATCH_005 instruction and visual evidence`; retain its independent ledger.
Full public inputs, private annotations, frozen checksums and raw receipts are
local evidence, not published data. The task's `prepare_protocol.py` is the
one-time input/annotation preparation record. Do not rerun it over a freeze.

With `$PACKAGE` pointing to the frozen source snapshot and `$EVIDENCE` to the
new round's evidence directory, execute once each:

```sh
python scripts/engineering/run_llm_vla.py --dispatch 005 --phase evidence \
  --store "$STORE" --evidence "$EVIDENCE" --package "$PACKAGE"
python scripts/engineering/run_llm_vla.py --dispatch 005 --phase coverage \
  --store "$STORE" --evidence "$EVIDENCE" --package "$PACKAGE"
PYTHONPATH="$PACKAGE" python -m actionstream.llm_vla.research_scoring \
  --evidence "$EVIDENCE"
```

The supervisor verifies model/frozen file identities, refuses existing compute
load, uses owned offline caches, and charges all child wall time including
imports, model loading, errors and cleanup against 1800 seconds. Text <=120,
vision scheduled 84 (authorization cap 100). No retries or native phase.
CPU contracts are separate from model and visual quality gates:

```sh
python -m pytest tests/test_instruction_visual_evidence.py tests/test_llm_vla.py \
  tests/test_grounded_language.py -p no:cacheprovider
```

Results are appended only after the frozen trial. Failure does not authorize
retuning the same rules/answers and reporting a first-pass success.
