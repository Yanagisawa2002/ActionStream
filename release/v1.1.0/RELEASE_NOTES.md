# ActionStream v1.1 private review candidate

Status: **engineering candidate; not published or merged**.

## Included

- LeRobot-compatible `ActionStreamInferenceEngine` and third-party plugin.
- Cancellable delivery scheduler and restartable process transport.
- Stale/reset rejection, bounded depletion hold, fallback, and schema-v1 JSONL telemetry.
- Clean-install/real-CLI, lifecycle, hung-call, reset, stop, recovery, and plugin tests.
- H1-R2 and H2 curated reports with paired statistics and explicit claim boundaries.
- A 61.9-second paired demo and content-addressed source/output manifest.

## Result status

- **H1-R2 GO:** pipelining increased request supply 11.43x and reduced aggregate
  depletion from 46.6% to 0.0% across 30 fixed-950-ms X-VLA/LIBERO episodes.
- **H2 NO-GO:** a five-step budget saved 61.6% of inference calls but reduced
  success from 14/15 to 13/15 and failed the frozen success gate.

The release claims a delivery-pipeline bottleneck and a measured
compute/success frontier. It does not claim universal runtime superiority,
official RTC evidence, native-Isaac holdout success, production remote-service
soak, or real-robot safety.

## External status

- GitHub repository visibility remains `PRIVATE`.
- The candidate is not merged into `master` and no GitHub Release is published.
- LeRobot PR #4466 remains a separately reviewed upstream extension boundary.
