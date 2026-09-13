# Finite language and observation integration — DISPATCH_003

The first frozen real language gate failed. Qwen accepted all 12 supported
paraphrases, but also accepted both ambiguous negative requests. Ten other
negative outputs violated the strict TaskSpec capability/null constraints and
were blocked. Shadow vision and all three conditional native episodes are
**NOT_RUN**. The completion checker and interruption path have CPU contract
coverage only. This implementation is not ready to control a robot from open
language requests.

## Scope and fixed execution

Only `pick_place / tomato_sauce / basket` is supported. The unchanged canonical
instruction feeds the existing `LeRobotBackend.infer_action_chunk`; its X-VLA
model, native observation processor, per-timestep action processor, absolute
7D controls, reset and 20 Hz control semantics remain unchanged. No second
skill, training, model search, runtime oracle substitution or prompt retry was
performed.

The local model is
`Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`, loaded
through the existing transformers 5.5.4 official class in BF16 with SDPA and
greedy generation, at most 192 new tokens. The unchanged fixed-revision
AutoProcessor handles chat formatting. See [official model documentation](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).
Only text inference actually ran in this round; loading the VLM class does not
validate its vision branch.

`configs/llm_vla_dispatch003.json` is byte-frozen before inference. Its hash is
`e4f56d3a78a1e4aeefb8bed4cd56e4d861d04cdc796b5ec1499a74b429869cd8`.
The explanatory JSON schema mirrors the pre-inference Python contracts; it
does not introduce case-specific repair rules. Invalid JSON, duplicate fields,
wrong identities or capability mismatches are explicit failures, never coerced
to an executable task. Request IDs must echo the caller-provided opaque ID.

## Boundaries

- `contracts.py` defines typed TaskSpec, CheckDecision, immutable public RGB /
  proprio observations and the bounded execution state. Observations whitelist
  only two cameras and existing end-effector/gripper/joint sensor fields.
- `qwen.py` accepts only prompt strings, opaque IDs, utterances or two PIL RGB
  images. The text parser sees no scene/task IDs or references. The checker
  sees goal semantics and simultaneous images, without file paths or labels.
- `gate_runner.py` reads only frozen public inputs, preserves every prompt,
  raw generation, model error and schema failure. `scoring.py` separately reads
  reference answers, reparses raw outputs and refuses missing/duplicate IDs.
- `integration.py` sees only `PublicPort` observations/actions/boundaries and
  checker decisions. It cannot use native success to declare completion or
  recover. An unknown decision stops. A native terminal boundary forces a stop
  and is logged separately from the checker claim.
- `native_adapter.py` keeps native success and physical-state identity hashes
  in a private journal. Only the independent post-episode evaluator reads that
  journal. Its simulator access refreshes camera observations without physics
  advancement; this native refresh is implemented but not exercised here.
- The prospective controller checks every 30 physical controls and at native
  termination, caps each episode at 300 dispatched controls, and permits one
  software interruption after control 67. It drops the remaining old chunk,
  increments revision, refreshes observation and resumes only on `incomplete`.
  Errors reserve the attempted control call and cannot create a free retry.

This is application-level data separation, not an operating-system sandbox or
an adversarial proof that arbitrary Python cannot introspect private objects.

## Reproduction and evidence

The worktree artifact directory `artifacts/embodied_llm_vla_20260913` holds the
full report, original API/asset manifests, frozen protocol, shuffled public
inputs, separate scorer references, raw language receipt, independent score,
NOT_RUN statuses, CPU XML and audited hashes. The model-facing shadow set was
frozen to all 20 pairs from the previous four completed episodes: 12 negative,
4 positive and 4 unlabeled reset snapshots. The main camera is always rotated
180 degrees; the wrist camera is always unchanged. No image was selected or
transformed according to its label.

The original environment is reused read-only. A new owned source overlay under
`D:/CodexValidation/ActionStreamVLA-20260913/dispatch003-llm-vla/python/gate_v1`
provided the exact Python modules for the real gate, with no package installs.
The source receipt and supervisor record all module hashes. Native-only modules
were completed after that snapshot; they were never imported by the real gate.
After the gate, a supervisor SIGTERM cleanup handler was added without changing
the ledger or rerunning inference. The exact supervisor used for the language
run is also retained in the artifact directory.

The GPU supervisor reuses only the old GPU-inventory/cache helpers and owns a
separate 1,200-second budget ledger. It refuses repeated phases, unfinished
children, busy GPU, changed frozen data/assets, and later phases after a failed
gate. Only its own process group may be terminated on timeout. Imports, loading,
inference, errors and cleanup all count. This trial used one child for
132.89996183098992 seconds. The remaining budget is not permission to bypass
the failed gate.

CPU verification (using the repository's normal test environment):

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m pytest tests/test_llm_vla.py -p no:cacheprovider --junitxml=cpu.xml
python -m ruff check src/actionstream/llm_vla scripts/engineering/run_llm_vla.py tests/test_llm_vla.py
```

One authorized phase uses the WSL supervisor, with explicit owned `--store`,
`--evidence`, `--package` and `--phase language` arguments. Independent scoring
uses `python -m actionstream.llm_vla.scoring --phase language --receipt ...
--answers ... --config ... --output ...`. Exact absolute commands are retained
in the report and supervisor receipt. Existing receipts cannot be overwritten
or rerun by the supervisor. A future trial requires a separately authorized,
newly frozen protocol and storage/compute accounting.

The two false accepts demonstrate unsupported resolution of pronouns and an
unspecified destination, despite the prompt's ambiguity rule. The next useful
work is a separately frozen ambiguity/abstention validation, rather than adding
skills to this failed parser. Structured generation may reduce format failures
but does not itself prevent schema-valid semantic false accepts. No claim of
generalization, recovery improvement, checker reliability or algorithm gain is
supported by this round.
