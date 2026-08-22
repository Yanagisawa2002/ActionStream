# ActionStream backend for `lerobot-rollout`

This small package registers `--inference.type=actionstream` with LeRobot's
rollout inference factory. It depends on the generic third-party inference-engine
registry proposed in
`upstream/lerobot/0001-feat-rollout-allow-third-party-inference-engines.patch`.

The backend executes action-chunk inference on one worker, aligns returned chunks
to observation age, rejects pre-reset and timed-out responses, bounds queue-empty
holds, and exposes queue/fallback/transport telemetry. It is validated with mock
transport failures and simulator rollouts; it is not a real-robot safety claim.

From a clean ActionStream clone, one command installs the root package, the
pinned LeRobot revision, and this plugin:

```bash
uv sync --locked --all-packages
```

`uv run lerobot-rollout --help` then exercises LeRobot's actual CLI subprocess
and must list `actionstream` as an inference choice.

The default `transport_mode=direct` keeps local CUDA policy inference in the
worker thread. Its timeout is advisory because Python cannot safely interrupt a
blocked CUDA call; telemetry therefore records `deadline_enforced=false`.
Deployments with a restartable remote/client factory can select
`transport_mode=process` and a pinned `module:callable` factory. That mode owns
the client in a child process, enforces startup and request deadlines, and
terminates the child on timeout, reset, or stop before accepting more work.

Set `telemetry_jsonl_path` to append schema-v1 lifecycle, request, queue,
discard, depletion, recovery, and fallback events. Frames, observations,
actions, credentials, and model payloads are deliberately excluded. The field
contract is documented in `docs/telemetry_jsonl_v1.md` and validated by
`schemas/actionstream.telemetry.v1.schema.json`.
