# Three-task integration tracker

Updated: 2026-09-16. No native experiment was run in this change.

| Block | State | Evidence / next gate |
|---|---|---|
| E0 local implementation | Implemented and locally tested | Task registry, whole-request authorization, shared controller, task-conditioned RGB architecture, task-specific recorder/scorer, split guards; 27 new tests and 242 tests in the repository CI selection passed locally |
| E0 native pilots | BLOCKED: GPU offline | Need current Linux GPU SSH endpoint; no prior single-task receipt substitutes |
| E1 collection/training | Tools implemented; NOT RUN | CLI collection, guarded manifest, shared model training and no-task ablation; no trained candidate yet |
| E2 same-scene counterfactual corpus | Pending native scene audit | Required to distinguish task dependence from scene memorization; training loader does not fabricate wrong-task labels |
| E3 three-task acceptance | NOT RUN | Requires E0–E2, frozen checkpoint/protocol and a fresh 60-episode run; acceptance runner still pending |
| E4 task generalization | NOT STARTED | Candidate metadata reserved provisionally; runtime/train allowlists prohibit loading these tasks |

## Evidence boundaries

- Existing single-task normal/recovery results and the earlier NO-GO are retained.
- Unit tests use synthetic observations/ports; they prove routing and invariants,
  not visual stopping quality, GPU timing, physical recovery or transfer.
- No completion rate, timing percentile, recovery rate or generalization result
  exists yet for this three-task candidate. These metrics are **unavailable**.
- The Qwen text-encoding API and native task adapters require E0 on the locked
  Linux environment; local architecture tests do not validate those external APIs.

## Local verification

- Ruff lint/format and CLI help checks pass.
- Original async NO-GO and later single-task budget evidence replay pass without
  changing either historical outcome; consumed-data diagnosis also replays.
- Broader `pytest tests` initially ran 273 tests: 272 passed, one unchanged legacy
  `test_runtime.py::test_sync_hold_waits_for_first_chunk_then_advances_with_exact_hold`
  failed with zero division in `benchmark.py:548`. The isolated test also fails.
  This local Python 3.12 clock reports `GetTickCount64()` with 15.625 ms resolution;
  the 200 Hz fake benchmark produces repeated timestamps. Neither file was changed
  by this work. This full-suite result must not be represented as a clean pass.
- Fresh physical replay tooling is implemented (`audit_multitask.py`) but NOT RUN
  against LIBERO here. Final freeze/acceptance and counterfactual data remain open.
