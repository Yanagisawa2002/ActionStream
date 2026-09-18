# Remote external-validity identity preflight — NO_RUN

## Verdict

**NO_RUN_INSUFFICIENT_FRESH_IDENTITIES**

The preregistered remote-host X-VLA/LIBERO external-validity experiment was not
started. Identity preflight found no task family with the minimum five clean,
previously unused reset identities required by the protocol.

This is a pre-execution validity result, not a model/runtime outcome.

## Identity audit

All 45 audited task/reset states exist.

| Task family | CLEAN | CONSUMED | UNKNOWN |
|---|---:|---:|---:|
| Object 5 | 0 | 15 | 0 |
| Spatial 7 | 0 | 13 | 2 (35, 39) |
| Goal 2 | 0 | 13 | 2 (35, 39) |

Object states 35–49 are all consumed. Therefore resolving the remaining access
gaps for Spatial/Goal cannot recover the protocol requirement of at least five
fresh identities **per family**.

The evidence audit validated **2,235 deduplicated evidence records**. Of those,
**709 records were warmup initialization records**. Warmups that actually
executed a reset count as consumed identities; they are not reclassified as
fresh merely because they were non-scored.

## Integrity / stop conditions

- audit verification: PASS
- selected identities: none
- new seed mapping: empty
- protocol modification during audit: none
- protocol commit during audit: none
- experiment launch: none
- audit-time repository HEAD: `f6fb04ea940fc809564b06552599c1872f6b4e03`
- `audit.json` SHA-256:
  `07a6af8b07afdce26b5b3d251f13685cbd1bf292c799e4918c50843dd625ba8a`

The frozen protocol remains unchanged. This report was added only after the audit
completed, to record the preflight decision.

## Interpretation

This NO_RUN is the intended behavior of the identity gate. Reusing consumed
resets, excluding executed warmups after seeing the shortage, or silently
changing the identity requirement would invalidate the external-validity claim.

The real TCP transport and CPU fault matrix remain independently valid software
and lifecycle evidence. What is blocked is the planned **fresh-identity,
multi-task remote-GPU validation** under this specific frozen protocol.

A future external-validity experiment requires a genuinely new identity source
(e.g. a new simulator task/reset pool or a separately frozen benchmark), not a
post-hoc remapping of already consumed evidence.
