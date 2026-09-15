# Reproducible delivery: PASS

Measured on 2026-09-15 with runtime source
`452a3351a217edcd87320adccd175de8f92831d6`, isolated Python 3.12 / CUDA 12.8,
and RTX 5090. See [the runbook](../../docs/agent-delivery.md).

| Check | Result |
| --- | --- |
| Runtime lock | PASS |
| Model/tokenizer identities | 19/19 |
| LIBERO asset identities | 585/585 |
| Supported installed CLI, empty offline cache | Exit 0; physical completion; 0.30 simulation-second confirmation delay; stable post-stop |
| Unsupported installed CLI, separate empty offline cache | Exit 2; no backend; no trajectory |
| Original external evidence identities | 21/21 |
| Independent downloaded copy | 282/282 file hashes |

There was no historical trajectory backup. Consumed-seed reruns with retained
parser calls reproduced **all twenty original trajectory files byte-for-byte**.
Only exact original size/SHA-256 matches were placed in `restored-blobs`.
New logs, timestamps and reconstruction provenance are retained separately.
This closes historical artifact completeness and provides no new heldout success
estimate or asynchronous/recovery evidence.

The full 623,243,676-byte archive `current-delivery-v1.tar.gz` has SHA-256
`f3912c2129d18527dbf9c66ce4255154e0da30589395f1cdee2b6343ada21be1`.
It is retained in private Draft release ID `389087456`, asset ID `565649962`,
and an independently downloaded Windows copy. Download with authorized GitHub
access:

```bash
gh release download agent-delivery-20260915 \
  --repo Yanagisawa2002/ActionStream --pattern current-delivery-v1.tar.gz
```

Verify the archive identity, extract it to a new directory, then run:

```bash
python reports/agent_delivery_20260915/verify_evidence.py \
  --bundle /absolute/path/to/extracted/current-delivery-v1
```

Without `--bundle`, the script replays the retained hash-pinned receipts and
original manifest comparisons only. Full trajectories and model weights are
external artifacts; the repository does not contain their bytes.
