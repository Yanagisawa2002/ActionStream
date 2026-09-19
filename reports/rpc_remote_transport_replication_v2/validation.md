# Result-consolidation validation

Date: 2026-09-19. Source implementation/protocol:
`e33eb98c94013da14109ab2ac710414fea63ebaf`.
This change only consolidates documentation, tracked summaries and a local
read-only post-hoc replay. No GPU process, simulator episode, remote connection,
diagnostic, deployment, remote lifecycle action or additional experiment ran.
The 51 formal outcomes and 7 PASS / 1 FAIL verdict remain frozen.

## Local checks

| Check | Result |
|---|---|
| Full repository CPU pytest suite | **223 passed**, 0 failed/errors/skipped; JUnit duration 33.547 s; process exit 0 |
| Ruff check, existing CI scope plus post-hoc replay | PASS |
| CI-configured Ruff format scope plus replay | PASS, 69 files |
| compileall, source/tests/plugin plus replay | PASS |
| Post-hoc replay to a separate ignored output directory | PASS, JSON and 100-event CSV byte-identical to tracked LF outputs |
| Original metrics projection and coverage/paired contents | PASS against sealed aggregate; only line endings normalized in copied CSV/JSON |
| Strict public release audit, without license bypass | PASS; tracked new summaries included |
| Local Markdown links and Git whitespace | PASS; 51 local links and staged/unstaged whitespace checks |
| Protected tracked files against implementation commit | PASS: configs, runtime, tests, integrations, lockfile, pyproject, CI and v1 matrix unchanged |

Python 3.12.11 and PyTorch 2.11.0+cpu came from the existing environment. Both
package import paths were set to this worktree (`src` and
`integrations/lerobot/src`), the interpreter directory was prefixed to PATH for
CLI subprocess tests, and CUDA_VISIBLE_DEVICES was empty. No dependency was
installed or changed. This is not a fresh Linux locked-install or upstream
LeRobot-suite receipt. The full local suite includes the real local CLI discovery
subprocess and socket/lifecycle tests; it does not run X-VLA weights or LIBERO.

Representative commands, with that environment active:

```text
python -m pytest -q -o faulthandler_timeout=45 --junitxml=artifacts/rpc_v2_result_consolidation_20260919/tests.xml
python -m ruff check src tests scripts/release scripts/engineering integrations/lerobot reports/rpc_remote_transport_replication_v2/recompute_posthoc.py
python -m ruff format --check <paths from .github/workflows/ci.yml> reports/rpc_remote_transport_replication_v2/recompute_posthoc.py
python -m compileall -q src tests integrations/lerobot/src reports/rpc_remote_transport_replication_v2/recompute_posthoc.py
python reports/rpc_remote_transport_replication_v2/recompute_posthoc.py --output artifacts/rpc_v2_result_consolidation_20260919/replay
python scripts/release/audit_public_release.py --output artifacts/rpc_v2_result_consolidation_20260919/release_audit.json
git diff --check
git diff --cached --check
```

The strict release audit checks repository packaging, credentials and licensing;
its PASS does not change the transport gate or establish production readiness.

The format command uses the explicit CI path list, not a broader repository
reformat. Only the new replay script was formatted. Its default run verifies
source hashes before computing results; do not invoke Python with `-O`, which
disables its integrity assertions. Replay outputs must be outside the sealed
experiment root. Tracked summaries use LF line endings to keep committed hashes
and cross-platform replay stable.

## Byte preservation

The completed top-level experiment manifest remains
`9c8e376f83ca4dce162b452defd2d9c42ea6bc8e44478133951846c8068fbe2f`.
The whole-root manifest remains
`3c8cc214071455ac0ad2a10101239ec85af5e8e41a1e8aca02751c90d87f7864`;
all **2,889 entries** match. Every one of the 51 run manifests and both-host raw
manifests also passed independent hashing during replay (2,641 run-manifest
entries; these overlap the whole-root inventory and are not additional files).

| Preserved item | SHA256 / verified entries |
|---|---|
| V2 config | `661ee2fbcd565ec2a04deda57d8021cc088f05d2a3cb915bc59e379486fbb4f1` |
| V2 Run1 manifest | `9d4ad51b781679eb0481c3d498074ab1bd3548c5b86c7e9a260a9b6cbc98d4b4`; 140 entries |
| V1 config | `b2fc8f8a0f7f66eb60cddbd79645647b7313435deaafc84831dd396616749019` |
| Aborted external-validity config | `7d663780396e28c0bc4b867a363e087a40a09cd8bc84e1fab9d24122bc614887` |
| V1 formal manifest | `9b0e223c00b92f907a7d0c0803c94f41a07e01a46fe052d0ac28bc4f9609329d`; 123 entries |
| V1 diagnostic manifest | `f515e1f6324c34d7c0cd4933df3c437971bfe1c34e8ac399fff504aa9171a0be`; 77 entries |
| uv.lock | `803833a9ec37bb9fd7de0fea5b3cd10eb1db060495d42525eb5f614e7e7c9e57` |

V1 evidence was read from its original
`actionstream-rpc-reset-audit-20260918` worktree. Neither its files nor the v2
raw bundle were copied over, edited or committed. The config status fields remain
the preregistered values; completed status lives in [final_status.json](final_status.json).
No timeout, gate, result classification, dependency or runtime code changed.

[validation.json](validation.json) contains exact changed files, source checks,
curated hashes and machine-readable outcomes. Raw local validation logs remain
ignored under `artifacts/rpc_v2_result_consolidation_20260919/`. A checkout without
the original ignored experiment bundles cannot replay the raw measurements;
tracked aggregates and hashes are a review surface, not a substitute for that data.
