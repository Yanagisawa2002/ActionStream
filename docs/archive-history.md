# Historical experiment archive

The default branch intentionally keeps the maintained runtime, compact frozen
evidence, release media, and the report trees required by CI/review. Generated
runtime outputs and superseded development-report trees are not carried forward
indefinitely.

The complete pre-slimming tree is pinned at commit
[`ed05167541fc9d83c93c1f9f1b422bcb05fee54f`](https://github.com/Yanagisawa2002/ActionStream/tree/ed05167541fc9d83c93c1f9f1b422bcb05fee54f).

Removed from the default branch in the repository-slimming change:

- `outputs/` — generated benchmark/runtime outputs (450 tracked files);
- `reports/blocker_diagnosis_20260915/`;
- `reports/completion_evaluation_20260915/`;
- `reports/completion_v2_20260915/`;
- `reports/embodied_development_20260913/`;
- `reports/visual_completion_20260915/`.

These removals do not rewrite experiment outcomes. Historical documents link back
to the pinned commit when they reference one of these removed trees.

Still retained on the default branch because they are part of the current public
review/validation surface:

- `reports/actionstream_transport_h1_r2/`;
- `reports/actionstream_transport_h2_budget_holdout/`;
- `reports/rpc_remote_transport_replication_v2/`;
- `reports/completion_development_20260915/`;
- `reports/completion_acceptance_v3_20260915/`.

The two completion trees remain because CI replays their evidence checks. H1/H2
and remote-RPC v2 remain because they support the headline systems claims in the
README and evidence index.
