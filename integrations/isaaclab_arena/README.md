# Isaac Lab-Arena adapter

This package targets the alpha `release/0.2.1` Arena API at commit
`8b4a3a47fc53de23e8205089d71109a2e2348acd`. Install it together with the
ActionStream checkout inside Arena's official Docker source environment.

The frozen protocol is `configs/isaaclab_arena_frozen_v1.json`. It uses three
different DROID manipulation families with eight GPU-vectorized environments,
three real camera views, disjoint development/holdout reset seeds, and disjoint
network replays. X-VLA and SmolVLA revisions are exact pins.

`ActionStreamArenaPolicy` currently executes only `actionstream_aligned` and
`actionstream_guarded`. It deliberately rejects the sync, latest-only, and RTC
cells until vector-safe wrappers call the pinned upstream LeRobot
implementations. A local semantic reimplementation must not be reported as an
official baseline.

The observation bridge rejects missing or duplicated camera roles. The action
bridge converts each queued absolute target into the current Arena DROID
relative-IK frame on the control thread. A depleted queue yields a relative
zero-motion hold that preserves gripper state. These are simulation runtime
guards, not real-robot safety evidence.

Arena import, vector stepping, learned GPU inference, videos, success metrics,
and per-episode provenance remain separate evidence gates. An import-only or
zero-action smoke cannot be reported as a learned-policy result.
