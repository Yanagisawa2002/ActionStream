# ActionStream backend for `lerobot-rollout`

This small package registers `--inference.type=actionstream` with LeRobot's
rollout inference factory. It depends on the generic third-party inference-engine
registry proposed in
`upstream/lerobot/0001-feat-rollout-allow-third-party-inference-engines.patch`.

The backend executes action-chunk inference on one worker, aligns returned chunks
to observation age, rejects pre-reset and timed-out responses, bounds queue-empty
holds, and exposes queue/fallback/transport telemetry. It is validated with mock
transport failures and simulator rollouts; it is not a real-robot safety claim.
