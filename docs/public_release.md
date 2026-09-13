# v1.1 review and publication boundary

`codex/actionstream/release-v1-1-reviewable` is a private review candidate. It
does not change repository visibility and does not authorize publication.

The branch is assembled from `master` as a compact review surface containing:

- the LeRobot inference backend, cancellable process transport, and plugin;
- backend, failure-path, real CLI subprocess, and stress-fixture tests;
- locked dependencies, CI, license, and third-party notices;
- the H1-R2 mechanism-GO and H2 fixed-budget-NO-GO aggregate reports;
- one 61.9-second demo, poster, visual-review receipt, and content hashes.

It does not add raw episode archives, full trace sets, source MP4 collections,
policy weights, simulator assets, old selector experiments, Isaac/Arena code,
or ROS workspaces. Existing v1.0 files already present on `master` are not part
of the v1.1 diff.

The preserved evidence lineage is:

- H1-R2 report commit: `75371d9c148771658cdb3e13c9edf8fe4cd3dcb8`;
- H2 formal verdict commit: `2abf461b4e1e08eae989bbd590057a03420028a7`;
- backend formatting/CI fix: `4bda989c1cb4896e058a43203b5e525197f40324`.

Run the strict local gate with:

```bash
uv sync --locked --all-packages
uv run python scripts/release/audit_public_release.py
```

Making the repository public, merging to `master`, creating a GitHub Release,
or publishing model/simulator assets are separate external actions requiring
explicit approval.
