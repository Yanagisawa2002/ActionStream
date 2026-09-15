# Reproducing the finite Agent

Use Linux x86-64, Python 3.12, uv 0.11.32, a CUDA-capable NVIDIA driver supporting
CUDA 12.8, and EGL. On Ubuntu 22.04 the EGL dispatcher is `libegl1=1.4.0-1`;
the host also needs its matching NVIDIA EGL implementation. Python packages come
from `uv.lock`; do not use system-site-packages or an ambient PYTHONPATH.

## Install and materialize

Start in this source checkout with a new task-owned store. The first two fetches
need network access. Every file is pinned by upstream commit, byte count and
SHA-256; a download mirror does not change those identities.

```bash
uv sync --locked --all-packages --python python3.12
export STORE=/absolute/path/to/new-actionstream-store
mkdir -p "$STORE"
uv run --no-sync actionstream-delivery doctor --cuda --output "$STORE/environment.json"
uv run --no-sync actionstream-delivery fetch \
  --manifest configs/completion_runtime_assets.json \
  --root "$STORE/assets" --cache "$STORE/download-cache" \
  --output "$STORE/model-assets.json"
uv run --no-sync actionstream-delivery fetch \
  --manifest configs/libero_runtime_assets.json \
  --root "$STORE/libero-assets" --cache "$STORE/download-cache" \
  --output "$STORE/libero-assets.json"
```

Supply the separately retained `frozen.pt` completion model. Its only accepted
identity is 57,369,399 bytes and SHA-256
`28482b40e470d932dfe44312d2dbcb7fa732146a2b2f9c7ebf7683b47cdfa7f9`.
This is a frozen trained model; runtime installation does not retrain it.
The source repository's manifests alone cannot recreate missing model bytes.

## Run with empty offline caches

Use a new empty cache and output for each diagnostic. All required model and
simulator inputs must come from the verified directories above.

```bash
export HF_HOME="$STORE/empty-offline-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
uv run --no-sync actionstream-agent \
  --request 'Please put the tomato sauce in the basket.' --seed 2026092000 \
  --assets "$STORE/assets" --checkpoint "$STORE/frozen.pt" \
  --libero-assets "$STORE/libero-assets" \
  --libero-asset-manifest configs/libero_runtime_assets.json \
  --language-config configs/finite_agent_language.json \
  --asset-manifest configs/completion_runtime_assets.json \
  --output "$STORE/supported-diagnostic"
```

Repeat with a different empty cache/output and request `Put the book in the
basket.` Expected: `LANGUAGE_BLOCKED`, `backend_created=false`, exit 2, no
`trajectory.npz`. The supported diagnostic must exit 0 with an independent score
and full trajectory. The example seed is intentionally an already consumed
reproduction case; it provides no new heldout generalization evidence.

Keep all output files, the source commit, `uv.lock`, environment and asset
receipts together. Hash the full state/RGB `trajectory.npz` before transfer and
verify after downloading to a second storage location. Compact JSON logs alone
are insufficient for state/RGB replay.

## Historical evidence completeness

```bash
uv run --no-sync actionstream-delivery verify \
  --manifest reports/finite_agent_20260915/external_blobs.json \
  --root /path/to/original-finite-agent-bundle --output historical-blobs.json
```

This requires all 21 original files and verifies their original hashes. The
20 historical trajectories were not migrated or backed up, as confirmed by
the owner on 2026-09-15. Only the frozen model was recovered. Therefore the
historical full-blob gate remains **FAIL (1/21)**; the independently replayable
compact logs retain their original status. A newly generated trajectory cannot
replace a historical file or close that gate.

Current implementation/validation progress is tracked in
[agent-delivery-progress.md](agent-delivery-progress.md). Asynchronous timing and
physical recovery must have separate protocols and new measured evidence.
