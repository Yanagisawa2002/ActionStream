# Changelog

All notable changes to ActionStream are documented here.

## 1.0.0 - 2026-07-31

- Published the validated M0-M3 benchmark and frozen evidence.
- Added the M4 paired analysis, real-time `sync_hold` baseline, calibrated
  queue-pressure protocol, and full 950 ms pressure evaluation.
- Added reproducible release figures and a captioned demonstration video
  generated from frozen evidence.
- Added release integrity checks that keep the validated runtime and M3/M4
  evidence unchanged from commit `d84cb64`.
- Replaced the host-specific setup path in the maintained documentation with
  repository-relative instructions.

The original hypothesis is only partially supported. At 950 ms,
`async_aligned` improved paired success and observed episode efficiency over
`async_naive`, but it produced more queue-hold steps than `async_naive`.
No fully stale chunk was observed.
