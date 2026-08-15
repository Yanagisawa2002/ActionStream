# M4 paired A/B content capture

This is a fresh same-condition visualization rerun, not footage from the
original frozen M4 episodes. It supplements, but does not replace, the frozen
M4 aggregate statistics.

## Fixed condition

- Task: LIBERO Object task 0, "Pick the alphabet soup and place it in the basket"
- Episode index: 2
- Initial state index: 4
- Base / episode seed: 142 / 144
- Injected delivery delay: 950 ms
- Replan interval: 10 control steps
- Controller: 20 Hz, maximum 800 steps
- Model revision: `12e8783e996944f5c97e490d37d4c145484ed70a`
- Frozen M4 runner source: `d84cb64e9e48b681e083828b24df5a38709c78c8`

## Fresh paired result

| Mode | Result | Environment steps | Video time | Episode wall time | Historical reference |
| --- | --- | ---: | ---: | ---: | --- |
| `async_naive` | Fail / timeout | 800 | 40.05 s | 45.931 s | Exact match: fail at 800 |
| `async_aligned` | Success | 139 | 7.00 s | 13.910 s | Success matches; terminal step was 139 instead of 150 |

For this pair, aligned execution used 661 fewer control steps (82.6%) and
39.0% lower mean action discontinuity (0.1667 versus 0.2732 L2). These are
single-pair rerun observations, not aggregate estimates.

## Visual reading

- Both sides start from the same scene and state.
- The aligned runner carries the alphabet-soup can over the basket around step
  130 and reaches terminal success at step 139.
- The naive runner remains active until step 800, misses the placement, and by
  the end has displaced several objects and tipped the basket.
- In the paired video, the shorter successful side holds its final frame while
  the naive side continues to timeout.

## Artifacts

- `m4_task0_seed144_state4_paired.mp4`: primary 1280x720, 20 FPS, 40.05 s A/B video
- `m4_task0_seed144_state4_contact_sheet.png`: six sampled paired frames
- `m4_task0_seed144_state4_t6_5s.png`: paired frame at 6.5 s
- `before_naive/` and `after_aligned/`: individual videos, metrics, traces, and receipts
- `m4_task0_seed144_state4_paired.receipt.json`: composition receipt
- `run_final.log`: remote execution log
- `m4_task0_seed144_state4_20260815T071335Z.tar.gz`: downloaded source bundle

The downloaded bundle SHA-256 is
`d291bbc3a1f90839738bc8f0d8c625350d26a50b6caf92ab251f0f13d1eb47e8`.
All metric, trace, and video hashes were rechecked locally against their
receipts.

## Runtime identity

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB; driver 595.71.05
- Python 3.12.3
- PyTorch 2.8.0+cu128 / CUDA runtime 12.8
- LeRobot 0.6.0, Transformers 5.5.4
- MuJoCo 3.8.1, robosuite 1.4.0, PyAV 15.1.0

The original M4 environment used PyTorch 2.11.0+cu130 on an RTX 4090. The
success/failure pattern reproduced here, while the aligned terminal step moved
from 150 to 139; the runtime difference is therefore retained as a provenance
boundary.
