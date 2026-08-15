# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | Source/API provenance | LeRobot `6adf515` | source audit | commit, version, signatures | MUST | DONE | Remote checkout clean; source version 0.6.2 |
| R002 | M0 | GPU preflight | RTX 5090 | remote host | VRAM/use/processes | MUST | DONE | 2 MiB / 32607 MiB, 0%, no compute process |
| R003 | M0 | Adapter unit tests | official Async + ActionStream | synthetic | exact queue/timestamp behavior | MUST | DONE | Direct-upstream adapter reused without altering frozen M6; 31 targeted runtime/conformance tests pass |
| R004 | M0 | X-VLA parity smoke | sync reference | task 0 / fixed pair | success, action validity | MUST | TODO | Frozen revision `12e8783` |
| R005 | M1 | X-VLA baseline selection | weighted/latest/aligned | task 0 / 0,500,950,jitter | success, steps, holds, temporal error | MUST | TODO | One pair per profile first |
| R006 | M2 | Pi0.5 compatibility | sync + RTC | task 0 / 0,500 ms | success, delay estimate, VRAM | MUST | RUNNING | Revision `8e17415` cached (7.0 GB); remote policy/env smoke next |
| R007 | M3 | Compact X-VLA matrix | selected runtimes | tasks 0-2 / 3 pairs | paired primary metrics | MUST | TODO | Expand only after R005 |
| R008 | M3 | Compact Pi0.5 matrix | selected Async + RTC | tasks 0-2 / 3 pairs | paired primary metrics | MUST | TODO | Expand only after R006 |
| R009 | M3 | Optional third model | SmolVLA | task 0 then tasks 0-2 | paired primary metrics | NICE | TODO | Revision `31d453f`; stop on checkpoint/env mismatch |
| R010 | M4 | Native Isaac install/preflight | Isaac 6.0.1 + ROS Jazzy | remote RTX 5090 | compatibility/build/GPU receipt | MUST | TODO | All downloads after `source /etc/network_turbo` |
| R011 | M4 | Native baseline gate | `sync_hold` | 20 preregistered seeds | success and replay | MUST | TODO | Must pass before M8 development |
| R012 | M4 | Native paired evidence | naive/aligned (+sync if required) | seed 2026081100 / profile 1 | success, recovery, fairness, video | MUST | TODO | Live observation-conditioned policy only |
| R013 | M5 | Aggregate/report | all completed cells | paired analysis | CIs, plots, limitations | MUST | TODO | Preserve negative results |
