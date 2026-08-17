# M5-G0 audit evidence

This directory separates development/runtime smokes from the formal
calibration under `../calibration/`. None of these smoke rows is pooled into a
calibration decision, statistical test, or acceptance result.

The iterative smoke directories are retained rather than deleting unfavorable
development evidence. They use the same task-0 seed/state pair while source
integrity and crash-safe persistence were being completed:

| directory | implementation source hash | aligned result | gated result | role |
|---|---|---:|---:|---|
| `runtime_smoke/` | `7741a49c00c8b81a495187a67096ba8a36246b776016706c224e6f0b7bfdb042` | success, 144 steps | success, 225 steps | early direct JSONL smoke before transactional bundles |
| `runtime_smoke_final/` | `02c9ba879ffe8df334bb6361de60e9efbbfd0ee3480995157f6ccc789f191799` | success, 143 steps | failure, 800 steps | intermediate transactional smoke |
| `runtime_smoke_frozen/` | `b5f60604489ed20ec83e2fe0b3fcaabc38480f7ae0e406aca147d7f6e1846e2c` | success, 143 steps | success, 219 steps | intermediate source-freeze smoke |
| `runtime_smoke_release/` | `d85a931f096f41f8c25c27ea925a902c6159e5d9512254229b441282b2a2e517` | success, 143 steps | success, 448 steps | intermediate release-path smoke |
| `runtime_smoke_formal_ready/` | `fc920da5221cbe7d7039fc87512b6757fd476ba33a92535f2fb63d42a3d6fd35` | success, 143 steps | success, 223 steps | final runtime smoke matching the formal calibration producer hash |

The separate `m4_baseline_smoke/episodes.jsonl` is the fresh compatibility
check referenced by `task_entity_audit.json`. The static task/entity and
geometry audit is in `task_entity_audit.json` and `task_entity_audit.md`.
