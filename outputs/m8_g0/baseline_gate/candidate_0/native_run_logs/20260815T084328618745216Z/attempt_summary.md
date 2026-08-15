# M8-G0 native Isaac attempt — blocked before launch

Date: 2026-08-15

Host GPU: NVIDIA RTX 5090, 32,607 MiB

Requested evidence: same-reset, live observation-conditioned policy pair in
native Isaac Sim physics.

## Frozen inputs

- Isaac/ROS workspace:
  `IsaacSim-ros_workspaces@dd3eeede7912755996a18f4884285d9f50843f79`
- Pixi: `0.75.0`
- Pixi executable SHA-256:
  `4383aed18b2d5569cf34a19638daf954aa4415cc87ad3a9da9f34059cc4a004c`
- Suite manifest SHA-256:
  `dfe61569c74fd8faf942871c5f9675f4b0c8f2063e48b44f70cc58ddcc328c4e`
- Batch timeout: 14,400 seconds
- Native GPU authorization flag was supplied to the runner.

## Attempts

1. The frozen runner stopped at `batch_source_validate_only` while Pixi fetched
   the official Jazzy workspace dependencies through the accelerated proxy.
   The preserved failure receipt and validation log record this attempt.
2. A second accelerated Pixi attempt failed with another prefix.dev connection
   reset while fetching a different package.
3. After sourcing `/etc/network_turbo`, a direct prefix.dev retry with proxy
   variables removed downloaded substantially more of the environment, then
   failed after three retries fetching
   `libcups-2.3.3-h7a8fb5f_6.conda` with `connection reset`.

The third repeated network failure is the stop gate. No further environment
retry was made.

## Preserved evidence

- `failure_receipt.json` SHA-256:
  `94c45543e44b06784d70fd9bf47da1fbff01264252fac755a3565abe9ad057a7`
- `batch_profile_0_sanity_sync_hold.validate_only.log` SHA-256:
  `5612bb66cc0a1e742469252c251f2ab6dfb558df487b60eb09493cd93c08d13f`

## Evidence boundary

The failure happened before source validation completed, before the Isaac/ROS
build, and before any GPU/native episode launched. Therefore this attempt
produced no native success metric, fairness receipt, replay-valid trace, or
paired video. Existing ROS test-plant and LIBERO policy runs are not substitutes
for the requested policy-driven Isaac evidence.

The next safe action is to reuse the same pinned inputs after prefix.dev access
or an approved mirror is stable, then rerun the unchanged native runner. The
remote server was not shut down by this task.
