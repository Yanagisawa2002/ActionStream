# Public release boundary

The `1.1.0rc1` snapshot is being prepared as a small, reviewable source and
curated-evidence repository. Raw simulator events, per-episode traces, videos,
archives, model weights and datasets are not part of the public Git tree.

The pre-cleanup evidence remains content-addressed by source commit
`763cdeeed0168e38b7d95f0fd77bc4d5dbe9499d` and outputs tree
`aa2a0bc4c5f8464e5e00fa09339e3d4e8d33777f` on the preserved private branch
`codex/actionstream/async-rtc-isaac`. Public reports retain the aggregate tables,
figures, frozen manifests, receipts and representative content visualizations.

Technical release CI may run with `--allow-missing-license`; that flag does not
declare the repository publicly releasable. Public release remains blocked until
the owner selects and adds a first-party `LICENSE`. Apache-2.0 is the recommended
candidate because the upstream LeRobot integration targets an Apache-2.0 project,
but this document does not grant that license.

The remaining owner-controlled actions are:

1. approve and add the first-party license;
2. review the credential and third-party asset audit;
3. squash the prepared snapshot onto `master` (do not merge the 71-commit
   development history verbatim);
4. change GitHub visibility from private to public;
5. optionally submit the separately reviewable LeRobot registry patch upstream.

Run the strict gate with:

```bash
python scripts/release/audit_public_release.py
```

Until a `LICENSE` exists, that command must fail the legal gate even when every
technical check passes.
