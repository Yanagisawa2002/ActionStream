"""Paired statistical analysis and preregistered M8 gate evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path
import random
from statistics import median, pstdev
from typing import Any, Iterable

from .m8_protocol import (
    M8_MILESTONE,
    M8_SCHEMA_VERSION,
    NATIVE_ISAAC_EVIDENCE_CLASS,
    PROFILE_STRATEGIES,
    STRONG_OBSOLETE_STEP_REDUCTION_THRESHOLD,
    sha256_file,
)
from .m8_replay import (
    HOLDOUT_FAIRNESS_FIELDS,
    REQUIRED_METRICS,
    _load_episode_artifacts,
    _repository_root_for,
    validate_episode_rows,
)
from .schema import read_json, write_json_atomic


BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260808


def _portable_reference(path: Path | str, *, anchor: Path) -> str:
    resolved = Path(path).resolve()
    repository_root = _repository_root_for(anchor)
    if repository_root is not None:
        try:
            return resolved.relative_to(repository_root).as_posix()
        except ValueError:
            pass
    return Path(os.path.relpath(resolved, anchor.resolve().parent)).as_posix()


def paired_bootstrap_ci(
    differences: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    values = tuple(float(value) for value in differences)
    if not values:
        raise ValueError("paired bootstrap requires at least one pair")
    if not 0.0 < confidence < 1.0 or resamples < 1_000:
        raise ValueError("invalid paired-bootstrap configuration")
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        estimates.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    return (
        estimates[int(alpha * (resamples - 1))],
        estimates[int((1.0 - alpha) * (resamples - 1))],
    )


def _load_summaries(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(manifest_path)
    if manifest.get("milestone") != M8_MILESTONE:
        raise ValueError("analysis requires an M8-G0 manifest")
    if manifest.get("split") != "frozen_holdout" or manifest.get("headline_eligible") is not True:
        raise ValueError("headline analysis requires the frozen headline-eligible holdout")
    summaries = []
    for entry in manifest.get("episodes", []):
        rows, summary, source = _load_episode_artifacts(manifest_path, manifest, entry)
        if summary.get("milestone") != M8_MILESTONE:
            raise ValueError("episode summary milestone mismatch")
        if summary.get("evidence_class") != NATIVE_ISAAC_EVIDENCE_CLASS:
            raise ValueError("headline analysis accepts native Isaac evidence only")
        missing = [metric for metric in REQUIRED_METRICS if metric not in summary.get("metrics", {})]
        if missing:
            raise ValueError(f"episode summary missing M8 metrics: {missing}")
        missing_fairness = [field for field in HOLDOUT_FAIRNESS_FIELDS if field not in summary]
        if missing_fairness:
            raise ValueError(f"episode summary missing fairness fields: {missing_fairness}")
        # Analysis is intentionally a second raw-evidence consumer.  A summary
        # is never allowed to reach a statistic until its complete event log
        # has independently reconstructed every metric and passed semantic
        # replay.  The copied summary below uses the reconstructed values so a
        # later refactor cannot accidentally return to trusting cached metrics.
        audit = validate_episode_rows(
            rows,
            summary,
            source=source,
            required_fairness_fields=HOLDOUT_FAIRNESS_FIELDS,
        )
        if not audit["passed"]:
            raise ValueError(
                "raw episode replay failed before analysis: "
                f"{source}: mismatches={sorted(audit['metric_mismatches'])}, "
                f"violations={audit['invariant_violation_counts']}"
            )
        verified = dict(summary)
        verified["metrics"] = audit["recomputed_metrics"]
        verified["raw_event_source"] = source
        summaries.append(verified)
    if not summaries:
        raise ValueError("manifest contains no episode summaries")
    return manifest, summaries


def _numeric_summary(values: list[float | int | None]) -> dict[str, Any]:
    observed = [float(value) for value in values if value is not None]
    result: dict[str, Any] = {
        "count": len(values),
        "observed_count": len(observed),
        "null_count": len(values) - len(observed),
    }
    if observed:
        result.update(
            {
                "sum": sum(observed),
                "mean": sum(observed) / len(observed),
                "std": pstdev(observed),
                "median": median(observed),
                "min": min(observed),
                "max": max(observed),
            }
        )
    else:
        result.update({name: None for name in ("sum", "mean", "std", "median", "min", "max")})
    return result


def _metric_summary(values: list[Any]) -> dict[str, Any]:
    nonnull = [value for value in values if value is not None]
    if not nonnull:
        return {
            "kind": "unavailable",
            "count": len(values),
            "observed_count": 0,
            "null_count": len(values),
        }
    if nonnull and all(isinstance(value, bool) for value in nonnull):
        true_count = sum(bool(value) for value in nonnull)
        return {
            "kind": "boolean",
            "count": len(values),
            "observed_count": len(nonnull),
            "null_count": len(values) - len(nonnull),
            "true_count": true_count,
            "false_count": len(nonnull) - true_count,
            "rate": true_count / len(nonnull),
        }
    if nonnull and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in nonnull):
        return {"kind": "numeric", **_numeric_summary(values)}
    if nonnull and all(isinstance(value, str) for value in nonnull):
        counts = Counter(nonnull)
        return {
            "kind": "categorical",
            "count": len(values),
            "observed_count": len(nonnull),
            "null_count": len(values) - len(nonnull),
            "counts": dict(sorted(counts.items())),
        }
    raise ValueError(f"unsupported mixed metric values: {[type(value).__name__ for value in values]}")


def _strategy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row["metrics"] for row in rows]
    successes = sum(bool(metric["task_success"]) for metric in metrics)
    return {
        "successes": successes,
        "trials": len(rows),
        "success_rate": successes / len(rows),
        "correct_destination_successes": sum(
            bool(metric["correct_destination_placement_success"]) for metric in metrics
        ),
        "obsolete_destination_placements": sum(
            int(metric["obsolete_destination_placement_count"]) for metric in metrics
        ),
        "median_completion_steps": median(float(metric["completion_steps"]) for metric in metrics),
        "mean_completion_steps": sum(float(metric["completion_steps"]) for metric in metrics) / len(metrics),
        "median_wall_clock_seconds": median(float(metric["wall_clock_seconds"]) for metric in metrics),
        "mean_wall_clock_seconds": sum(float(metric["wall_clock_seconds"]) for metric in metrics) / len(metrics),
        "median_penalized_recovery_latency_steps": median(
            float(metric["penalized_recovery_latency_steps"]) for metric in metrics
        ),
        "mean_obsolete_destination_command_steps": sum(
            float(metric["obsolete_destination_command_steps"]) for metric in metrics
        ) / len(metrics),
        "mean_pre_switch_observation_command_steps": sum(
            float(metric["pre_switch_observation_command_steps"]) for metric in metrics
        ) / len(metrics),
        "mean_hold_control_seconds": sum(float(metric["hold_control_seconds"]) for metric in metrics) / len(metrics),
        "mean_response_wait_wall_seconds": sum(
            float(metric["response_wait_wall_seconds"]) for metric in metrics
        ) / len(metrics),
        "descriptive_metrics": {
            metric: _metric_summary([row["metrics"][metric] for row in rows])
            for metric in REQUIRED_METRICS
        },
    }


def _reduction(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        # No baseline exposure means there is no demonstrated reduction.  In
        # particular, 0 -> 0 must not manufacture a 100% secondary gain.
        return 0.0
    return 1.0 - candidate / baseline


def _replay_identifies_manifest(
    identity: object,
    *,
    manifest_path: Path,
    replay_path: Path,
) -> bool:
    if not isinstance(identity, str) or not identity:
        return False
    reference = Path(identity)
    expected = manifest_path.resolve()
    if reference.is_absolute():
        # Read-only compatibility for replay artifacts created before portable
        # identities were introduced.
        return reference.resolve() == expected
    candidates = [manifest_path.resolve().parent / reference, replay_path.resolve().parent / reference]
    repository_root = _repository_root_for(manifest_path)
    if repository_root is not None:
        candidates.append(repository_root / reference)
    return any(candidate.resolve() == expected for candidate in candidates)


def analyze_manifest(
    manifest_path: Path | str,
    *,
    replay_path: Path | str,
    output_path: Path | str | None = None,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest, rows = _load_summaries(path)
    replay = read_json(replay_path)
    if replay.get("milestone") != M8_MILESTONE:
        raise ValueError("replay audit is not M8-G0")
    if replay.get("split") != "frozen_holdout":
        raise ValueError("replay audit is not bound to the frozen holdout")
    if replay.get("manifest_sha256") != sha256_file(path):
        raise ValueError("replay audit does not bind the exact matrix manifest bytes")
    if not _replay_identifies_manifest(
        replay.get("manifest"),
        manifest_path=path,
        replay_path=Path(replay_path),
    ):
        raise ValueError("replay audit path does not identify the analyzed manifest")
    if not replay.get("passed"):
        raise ValueError("replay audit failed; refusing to compute headline statistics")
    if not replay.get("fairness_passed"):
        raise ValueError("paired fairness failed; refusing to compute headline statistics")
    if not replay.get("freeze_validation_passed"):
        raise ValueError("frozen-protocol validation failed; refusing headline statistics")
    if int(replay.get("episode_count", -1)) != len(rows):
        raise ValueError("replay audit episode count does not match analysis evidence")
    if not replay.get("seed_validation_passed") or int(replay.get("seed_count", 0)) < 40:
        raise ValueError("frozen holdout seed binding failed or contains fewer than 40 seeds")
    if not replay.get("profile_validation_passed"):
        raise ValueError("frozen holdout profile files/hashes are not exactly freeze-bound")

    by_profile_seed: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        profile_id = str(row["profile_id"])
        seed = int(row["seed"])
        strategy = str(row["strategy"])
        if strategy in by_profile_seed[profile_id][seed]:
            raise ValueError(f"duplicate profile/seed/strategy episode: {profile_id}/{seed}/{strategy}")
        by_profile_seed[profile_id][seed][strategy] = row

    if set(by_profile_seed) != set(PROFILE_STRATEGIES):
        raise ValueError(f"manifest must contain exactly profiles {sorted(PROFILE_STRATEGIES)}")
    profiles: dict[str, Any] = {}
    for profile_id in PROFILE_STRATEGIES:
        paired = by_profile_seed[profile_id]
        expected = set(PROFILE_STRATEGIES[profile_id])
        incomplete = {
            seed: {"expected": sorted(expected), "actual": sorted(methods)}
            for seed, methods in paired.items()
            if set(methods) != expected
        }
        if incomplete:
            raise ValueError(f"incomplete paired blocks for {profile_id}: {incomplete}")
        seeds = sorted(paired)
        strategies = {
            strategy: _strategy_summary([paired[seed][strategy] for seed in seeds])
            for strategy in PROFILE_STRATEGIES[profile_id]
        }
        profile_result: dict[str, Any] = {
            "paired_trial_count": len(seeds),
            "seeds": seeds,
            "strategies": strategies,
        }
        if {"naive_async", "aligned_async"}.issubset(strategies):
            differences = [
                float(paired[seed]["aligned_async"]["metrics"]["task_success"])
                - float(paired[seed]["naive_async"]["metrics"]["task_success"])
                for seed in seeds
            ]
            profile_seed = BOOTSTRAP_SEED + int.from_bytes(
                hashlib.sha256(profile_id.encode("utf-8")).digest()[:2], "big"
            )
            ci_low, ci_high = paired_bootstrap_ci(
                differences, resamples=bootstrap_resamples, seed=profile_seed
            )
            aligned = strategies["aligned_async"]
            naive = strategies["naive_async"]
            sync = strategies["sync_hold"]
            difference = sum(differences) / len(differences)
            profile_result.update(
                {
                    "aligned_minus_naive_success_rate": difference,
                    "aligned_minus_naive_percentage_points": 100.0 * difference,
                    "paired_bootstrap_95_ci_rate": [ci_low, ci_high],
                    "paired_bootstrap_95_ci_percentage_points": [
                        100.0 * ci_low,
                        100.0 * ci_high,
                    ],
                    "raw_paired_outcomes": [
                        {
                            "seed": seed,
                            **{
                                strategy: bool(paired[seed][strategy]["metrics"]["task_success"])
                                for strategy in PROFILE_STRATEGIES[profile_id]
                            },
                        }
                        for seed in seeds
                    ],
                    "aligned_recovery_latency_reduction_vs_naive": _reduction(
                        naive["median_penalized_recovery_latency_steps"],
                        aligned["median_penalized_recovery_latency_steps"],
                    ),
                    "aligned_obsolete_command_reduction_vs_naive": _reduction(
                        naive["mean_obsolete_destination_command_steps"],
                        aligned["mean_obsolete_destination_command_steps"],
                    ),
                    "aligned_median_wall_reduction_vs_sync": _reduction(
                        sync["median_wall_clock_seconds"],
                        aligned["median_wall_clock_seconds"],
                    ),
                }
            )
        profiles[profile_id] = profile_result

    sanity = profiles["profile_0_sanity"]["strategies"]["sync_hold"]
    primary = profiles["profile_1_fixed"]
    aligned_primary = primary["strategies"]["aligned_async"]
    naive_primary = primary["strategies"]["naive_async"]
    sync_primary = primary["strategies"]["sync_hold"]
    aligned_forbidden_count = sum(
        int(row["metrics"]["expired_actions_executed"])
        for row in rows
        if row["strategy"] == "aligned_async"
    ) + sum(
        sum(
            count
            for reason, count in audit.get("invariant_violation_counts", {}).items()
            if reason in {
                "previous_episode_execution",
                "rejected_response_execution",
                "stale_generation_execution",
                "expired_action_execution",
            }
        )
        for audit in replay.get("audits", [])
        if audit.get("strategy") == "aligned_async"
    )
    conditions = {
        "aligned_minus_naive_success_at_least_15pp": (
            primary["aligned_minus_naive_success_rate"] >= 0.15
        ),
        "paired_95_ci_excludes_zero": primary["paired_bootstrap_95_ci_rate"][0] > 0.0,
        "aligned_forbidden_execution_count_is_zero": aligned_forbidden_count == 0,
        "replay_passes_every_episode": bool(replay.get("passed")),
        "profile_0_ceiling_at_least_90pct": sanity["success_rate"] >= 0.90,
        "paired_fairness_passes": bool(replay.get("fairness_passed")),
        "frozen_protocol_validated_without_post_holdout_change": bool(
            replay.get("freeze_validation_passed")
        ),
        "native_isaac_evidence_only": all(
            row.get("evidence_class") == NATIVE_ISAAC_EVIDENCE_CLASS for row in rows
        ),
    }
    primary_passed = all(conditions.values())
    strong_conditions = {
        "aligned_within_5pp_of_sync": aligned_primary["success_rate"] >= sync_primary["success_rate"] - 0.05,
        "aligned_wall_at_least_15pct_lower_than_sync": primary["aligned_median_wall_reduction_vs_sync"] >= 0.15,
        "aligned_recovery_at_least_20pct_lower_than_naive": primary["aligned_recovery_latency_reduction_vs_naive"] >= 0.20,
        "aligned_materially_fewer_obsolete_steps": (
            primary["aligned_obsolete_command_reduction_vs_naive"]
            >= STRONG_OBSOLETE_STEP_REDUCTION_THRESHOLD
        ),
    }
    strong_passed = primary_passed and all(strong_conditions.values())
    material_secondary = any(
        (
            primary["aligned_recovery_latency_reduction_vs_naive"] >= 0.20,
            primary["aligned_obsolete_command_reduction_vs_naive"] >= 0.20,
            aligned_primary["median_wall_clock_seconds"]
            <= 0.85 * naive_primary["median_wall_clock_seconds"],
        )
    )
    semantic_ok = (
        conditions["aligned_forbidden_execution_count_is_zero"]
        and conditions["replay_passes_every_episode"]
        and conditions["paired_fairness_passes"]
    )
    if strong_passed:
        classification = "STRONG GO"
    elif primary_passed:
        classification = "GO"
    elif semantic_ok and conditions["profile_0_ceiling_at_least_90pct"] and material_secondary:
        classification = "PARTIAL GO"
    else:
        classification = "NO-GO"

    result = {
        "schema_version": M8_SCHEMA_VERSION,
        "milestone": M8_MILESTONE,
        "manifest": str(replay["manifest"]),
        "manifest_sha256": sha256_file(path),
        "replay": _portable_reference(replay_path, anchor=path),
        "replay_sha256": sha256_file(replay_path),
        "freeze_manifest": manifest.get("freeze_manifest"),
        "freeze_sha256": manifest.get("freeze_sha256"),
        "holdout_seed_file_sha256": manifest.get("seed_file_sha256"),
        "evidence_class": NATIVE_ISAAC_EVIDENCE_CLASS,
        "episode_count": len(rows),
        "profiles": profiles,
        "bootstrap": {
            "method": "paired_nonparametric_bootstrap",
            "resamples": bootstrap_resamples,
            "confidence": 0.95,
            "base_seed": BOOTSTRAP_SEED,
        },
        "primary_go_gate": {
            "profile": "profile_1_fixed",
            "conditions": conditions,
            "aligned_forbidden_execution_count": aligned_forbidden_count,
            "passed": primary_passed,
        },
        "strong_go_gate": {
            "eligible": primary_passed,
            "conditions": strong_conditions,
            "passed": strong_passed,
        },
        "partial_go_material_secondary_improvement": material_secondary,
        "classification": classification,
    }
    if output_path is not None:
        write_json_atomic(output_path, result)
    return result
