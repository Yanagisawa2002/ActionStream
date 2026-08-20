# Public release boundary

The `1.1.0` snapshot is a small, reviewable source and
curated-evidence repository. Raw simulator events, per-episode traces, videos,
archives, model weights and datasets are not part of the public Git tree.

The pre-cleanup evidence remains content-addressed by source commit
`763cdeeed0168e38b7d95f0fd77bc4d5dbe9499d` and outputs tree
`aa2a0bc4c5f8464e5e00fa09339e3d4e8d33777f` on the preserved private branch
`codex/actionstream/async-rtc-isaac`. Public reports retain the aggregate tables,
figures, frozen manifests, receipts and representative content visualizations.

The repository is licensed under Apache-2.0. Third-party policy weights,
datasets, simulator assets, and upstream code retain their original terms and
are not redistributed as first-party ActionStream source.

The release sequence is:

1. run the credential, license, dependency, size, and third-party asset audit;
2. preserve and merge the full development history onto `master`;
3. change GitHub visibility from private to public;
4. publish the content-addressed v1.1.0 release assets;
5. maintain the separately reviewable LeRobot registry patch upstream.

Run the strict gate with:

```bash
python scripts/release/audit_public_release.py
```

The command must pass without `--allow-missing-license` before visibility or
release publication changes.
