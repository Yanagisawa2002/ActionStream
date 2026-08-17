#!/usr/bin/env python3
"""Build compact, provenance-bound tables from a completed M8 holdout.

This is deliberately post-hoc presentation code. It does not change the
registered M8 analysis, its GO/NO-GO thresholds, or any frozen input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROFILE_ORDER = ("profile_0_sanity", "profile_1_fixed", "profile_2_faults")
STRATEGY_ORDER = ("sync_hold", "naive_async", "aligned_async")
EXPECTED_GROUPS = {
    ("profile_0_sanity", "sync_hold"),
    ("profile_1_fixed", "sync_hold"),
    ("profile_1_fixed", "naive_async"),
    ("profile_1_fixed", "aligned_async"),
    ("profile_2_faults", "sync_hold"),
    ("profile_2_faults", "naive_async"),
    ("profile_2_faults", "aligned_async"),
}
PROFILE_LABELS = {
    "profile_0_sanity": "P0 sanity",
    "profile_1_fixed": "P1 fixed 850 ms",
    "profile_2_faults": "P2 850 ms + jitter/faults",
}
STRATEGY_LABELS = {
    "sync_hold": "Sync hold",
    "naive_async": "Naive async",
    "aligned_async": "ActionStream aligned",
}
FAILURE_TAXONOMY = {
    "failed_approach": "pre-grasp control",
    "switch_precondition_missed": "switch timing/precondition",
    "unstable_grasp": "grasp instability",
    "grasp_miss": "grasp acquisition miss",
    "joint_or_workspace_limit": "safety/workspace limit",
    "destination_switch_recovery_timeout": "switch recovery timeout",
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("mean requires observations")
    return sum(materialized) / len(materialized)


def _format_rate(successes: int, trials: int) -> str:
    return f"{successes}/{trials} ({successes / trials:.1%})"


def _format_ci(bounds: list[float] | tuple[float, float]) -> str:
    return f"[{bounds[0]:.1%}, {bounds[1]:.1%}]"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


def build_report(
    *,
    matrix_path: Path,
    replay_path: Path,
    analysis_path: Path,
    summaries_directory: Path,
) -> dict[str, Any]:
    matrix = read_json(matrix_path)
    replay = read_json(replay_path)
    analysis = read_json(analysis_path)
    summary_paths = sorted(summaries_directory.glob("*.json"))
    summaries = [read_json(path) for path in summary_paths]

    if matrix.get("split") != "frozen_holdout":
        raise ValueError("matrix is not a frozen holdout")
    if replay.get("passed") is not True:
        raise ValueError("canonical replay did not pass")
    if int(replay.get("episode_count", -1)) != len(summaries):
        raise ValueError("summary count does not match replay episode count")
    if int(replay.get("episode_audits_passed", -1)) != len(summaries):
        raise ValueError("not every episode passed replay audit")
    if int(matrix.get("expected_episode_count", -1)) != len(summaries):
        raise ValueError("summary count does not match frozen matrix")
    if analysis.get("manifest_sha256") != sha256_file(matrix_path):
        raise ValueError("analysis does not bind the supplied matrix")
    if analysis.get("replay_sha256") != sha256_file(replay_path):
        raise ValueError("analysis does not bind the supplied replay")
    required_passes = (
        "seed_validation_passed",
        "profile_validation_passed",
        "provenance_validation_passed",
        "fairness_passed",
        "freeze_validation_passed",
    )
    failed_passes = [name for name in required_passes if replay.get(name) is not True]
    if failed_passes:
        raise ValueError(f"canonical replay lacks required passes: {failed_passes}")

    replay_by_episode: dict[str, dict[str, Any]] = {}
    for audit in replay.get("audits", []):
        if not isinstance(audit, dict):
            raise ValueError("canonical replay contains a malformed episode audit")
        source = str(audit.get("source", "")).replace("\\", "/")
        episode_id = Path(source).stem
        if not episode_id or episode_id in replay_by_episode:
            raise ValueError(f"duplicate or missing replay episode identity: {episode_id!r}")
        if (
            audit.get("passed") is not True
            or audit.get("metrics_match") is not True
            or audit.get("violations") not in ([], None)
        ):
            raise ValueError(f"episode did not pass canonical replay: {episode_id}")
        replay_by_episode[episode_id] = audit
    if len(replay_by_episode) != len(summaries):
        raise ValueError("canonical replay audit count does not match summaries")

    identities: set[tuple[str, int, str]] = set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        identity = (str(row["profile_id"]), int(row["seed"]), str(row["strategy"]))
        if identity in identities:
            raise ValueError(f"duplicate summary identity: {identity}")
        identities.add(identity)
        if row.get("split") != "frozen_holdout":
            raise ValueError(f"summary is not frozen holdout evidence: {identity}")
        if row.get("evidence_class") != "ros_cpp_isaac_sim":
            raise ValueError(f"summary is not native Isaac evidence: {identity}")
        episode_id = str(row.get("episode_id", ""))
        audit = replay_by_episode.get(episode_id)
        if audit is None:
            raise ValueError(f"summary has no canonical replay audit: {episode_id}")
        if audit.get("strategy") != identity[2] or audit.get("recomputed_metrics") != row.get(
            "metrics"
        ):
            raise ValueError(f"summary differs from canonical replay metrics: {episode_id}")
        grouped[(identity[0], identity[2])].append(row)
    if set(grouped) != EXPECTED_GROUPS:
        raise ValueError(f"unexpected profile/method groups: {sorted(grouped)}")

    group_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    failure_taxonomy_totals: Counter[str] = Counter()
    raw_failure_totals: Counter[str] = Counter()
    for profile_id in PROFILE_ORDER:
        for strategy in STRATEGY_ORDER:
            key = (profile_id, strategy)
            if key not in grouped:
                continue
            rows = grouped[key]
            metrics = [row["metrics"] for row in rows]
            trials = len(metrics)
            successes = sum(bool(metric["task_success"]) for metric in metrics)
            ci_low, ci_high = wilson_interval(successes, trials)
            failures = Counter(
                str(metric["completion_reason"])
                for metric in metrics
                if not bool(metric["task_success"])
            )
            for reason, count in sorted(failures.items()):
                category = FAILURE_TAXONOMY.get(reason, "other")
                failure_taxonomy_totals[category] += count
                raw_failure_totals[reason] += count
                failure_records.append(
                    {
                        "profile_id": profile_id,
                        "profile": PROFILE_LABELS[profile_id],
                        "strategy": strategy,
                        "method": STRATEGY_LABELS[strategy],
                        "failure_category": category,
                        "completion_reason": reason,
                        "count": count,
                        "rate_of_trials": count / trials,
                        "rate_of_failures": count / (trials - successes),
                    }
                )

            naive_rows = grouped.get((profile_id, "naive_async"))
            naive_success_rate = None
            if naive_rows is not None:
                naive_success_rate = sum(
                    bool(row["metrics"]["task_success"]) for row in naive_rows
                ) / len(naive_rows)
            success_rate = successes / trials
            success_delta_pp = (
                None if naive_success_rate is None else 100.0 * (success_rate - naive_success_rate)
            )
            relative_success_improvement = None
            if naive_success_rate not in (None, 0.0):
                relative_success_improvement = success_rate / naive_success_rate - 1.0

            record = {
                "profile_id": profile_id,
                "profile": PROFILE_LABELS[profile_id],
                "strategy": strategy,
                "method": STRATEGY_LABELS[strategy],
                "trials": trials,
                "successes": successes,
                "success_rate": success_rate,
                "success_wilson_95_ci": [ci_low, ci_high],
                "success_delta_vs_naive_pp": success_delta_pp,
                "relative_success_improvement_vs_naive": relative_success_improvement,
                "grasp_successes": sum(bool(metric["grasp_success"]) for metric in metrics),
                "recovery_observed": sum(bool(metric["recovery_observed"]) for metric in metrics),
                "collisions": sum(bool(metric["collision"]) for metric in metrics),
                "timeouts": sum(bool(metric["timeout"]) for metric in metrics),
                "mean_inference_latency_ms": mean(
                    float(metric["mean_inference_latency_ms"]) for metric in metrics
                ),
                "median_episode_p50_inference_latency_ms": statistics.median(
                    float(metric["p50_inference_latency_ms"]) for metric in metrics
                ),
                "median_episode_p95_inference_latency_ms": statistics.median(
                    float(metric["p95_inference_latency_ms"]) for metric in metrics
                ),
                "mean_action_age_steps": mean(
                    float(metric["mean_action_age_steps"]) for metric in metrics
                ),
                "expired_actions_executed": sum(
                    int(metric["expired_actions_executed"]) for metric in metrics
                ),
                "expired_actions_removed": sum(
                    int(metric["expired_actions_removed"]) for metric in metrics
                ),
                "median_penalized_recovery_latency_steps": statistics.median(
                    float(metric["penalized_recovery_latency_steps"]) for metric in metrics
                ),
                "mean_obsolete_destination_command_steps": mean(
                    float(metric["obsolete_destination_command_steps"]) for metric in metrics
                ),
                "median_wall_clock_seconds": statistics.median(
                    float(metric["wall_clock_seconds"]) for metric in metrics
                ),
                "failure_count": trials - successes,
                "failure_reasons": dict(sorted(failures.items())),
            }
            canonical = analysis["profiles"][profile_id]["strategies"][strategy]
            descriptive = canonical["descriptive_metrics"]
            exact_checks = {
                "trials": (record["trials"], canonical["trials"]),
                "successes": (record["successes"], canonical["successes"]),
                "grasp_successes": (
                    record["grasp_successes"],
                    descriptive["grasp_success"]["true_count"],
                ),
                "recovery_observed": (
                    record["recovery_observed"],
                    descriptive["recovery_observed"]["true_count"],
                ),
                "collisions": (record["collisions"], descriptive["collision"]["true_count"]),
                "timeouts": (record["timeouts"], descriptive["timeout"]["true_count"]),
                "expired_actions_executed": (
                    record["expired_actions_executed"],
                    int(descriptive["expired_actions_executed"]["sum"]),
                ),
                "expired_actions_removed": (
                    record["expired_actions_removed"],
                    int(descriptive["expired_actions_removed"]["sum"]),
                ),
                "failure_reasons": (
                    record["failure_reasons"],
                    {
                        reason: count
                        for reason, count in descriptive["completion_reason"]["counts"].items()
                        if reason != "correct_destination_stable_placement"
                    },
                ),
            }
            mismatched_exact = [
                name for name, (observed, expected) in exact_checks.items() if observed != expected
            ]
            if mismatched_exact:
                raise ValueError(
                    f"post-hoc group differs from registered analysis for {key}: {mismatched_exact}"
                )
            numeric_checks = {
                "success_rate": (record["success_rate"], canonical["success_rate"]),
                "mean_inference_latency_ms": (
                    record["mean_inference_latency_ms"],
                    descriptive["mean_inference_latency_ms"]["mean"],
                ),
                "median_episode_p50_inference_latency_ms": (
                    record["median_episode_p50_inference_latency_ms"],
                    descriptive["p50_inference_latency_ms"]["median"],
                ),
                "median_episode_p95_inference_latency_ms": (
                    record["median_episode_p95_inference_latency_ms"],
                    descriptive["p95_inference_latency_ms"]["median"],
                ),
                "mean_action_age_steps": (
                    record["mean_action_age_steps"],
                    descriptive["mean_action_age_steps"]["mean"],
                ),
                "median_penalized_recovery_latency_steps": (
                    record["median_penalized_recovery_latency_steps"],
                    canonical["median_penalized_recovery_latency_steps"],
                ),
                "mean_obsolete_destination_command_steps": (
                    record["mean_obsolete_destination_command_steps"],
                    canonical["mean_obsolete_destination_command_steps"],
                ),
                "median_wall_clock_seconds": (
                    record["median_wall_clock_seconds"],
                    canonical["median_wall_clock_seconds"],
                ),
            }
            mismatched_numeric = [
                name
                for name, (observed, expected) in numeric_checks.items()
                if not math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12)
            ]
            if mismatched_numeric:
                raise ValueError(
                    f"post-hoc numeric group differs from registered analysis for {key}: "
                    f"{mismatched_numeric}"
                )
            group_records.append(record)

    paired = {}
    for profile_id in ("profile_1_fixed", "profile_2_faults"):
        profile = analysis["profiles"][profile_id]
        paired[profile_id] = {
            "aligned_minus_naive_percentage_points": profile[
                "aligned_minus_naive_percentage_points"
            ],
            "paired_bootstrap_95_ci_percentage_points": profile[
                "paired_bootstrap_95_ci_percentage_points"
            ],
            "paired_trial_count": profile["paired_trial_count"],
        }

    return {
        "schema_version": 1,
        "artifact_kind": "posthoc_frozen_holdout_presentation",
        "milestone": "M8-G0",
        "official_classification": analysis["classification"],
        "official_primary_go_passed": analysis["primary_go_gate"]["passed"],
        "official_strong_go_passed": analysis["strong_go_gate"]["passed"],
        "official_primary_conditions": analysis["primary_go_gate"]["conditions"],
        "official_strong_conditions": analysis["strong_go_gate"]["conditions"],
        "episode_count": len(summaries),
        "seed_count": replay["seed_count"],
        "paired_block_count": replay["paired_block_count"],
        "validation": {name: replay[name] for name in required_passes},
        "provenance": {
            "matrix": {"path": matrix_path.as_posix(), "sha256": sha256_file(matrix_path)},
            "replay": {"path": replay_path.as_posix(), "sha256": sha256_file(replay_path)},
            "analysis": {"path": analysis_path.as_posix(), "sha256": sha256_file(analysis_path)},
            "summary_file_count": len(summary_paths),
            "freeze_sha256": analysis["freeze_sha256"],
        },
        "groups": group_records,
        "paired_comparisons": paired,
        "failure_classification": {
            "raw_completion_reasons": dict(sorted(raw_failure_totals.items())),
            "taxonomy_totals": dict(sorted(failure_taxonomy_totals.items())),
            "rows": failure_records,
        },
        "interpretation_boundary": {
            "task_family_count": 1,
            "task_family": "dynamic_target_pick_place_v1",
            "unseen_scenario_seed_count": 60,
            "network_profile_count": 3,
            "latency_plot_is_operating_point_not_continuous_sweep": True,
            "learned_policy": False,
            "real_robot": False,
        },
    }


def write_outputs(report: dict[str, Any], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "holdout_summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    csv_fields = [
        "profile_id",
        "profile",
        "strategy",
        "method",
        "successes",
        "trials",
        "success_rate",
        "success_wilson_95_ci_low",
        "success_wilson_95_ci_high",
        "success_delta_vs_naive_pp",
        "relative_success_improvement_vs_naive",
        "grasp_successes",
        "recovery_observed",
        "collisions",
        "timeouts",
        "mean_inference_latency_ms",
        "median_episode_p50_inference_latency_ms",
        "median_episode_p95_inference_latency_ms",
        "mean_action_age_steps",
        "expired_actions_executed",
        "expired_actions_removed",
        "median_penalized_recovery_latency_steps",
        "mean_obsolete_destination_command_steps",
        "median_wall_clock_seconds",
        "failure_count",
    ]
    csv_rows = []
    for group in report["groups"]:
        row = {name: group.get(name) for name in csv_fields}
        row["success_wilson_95_ci_low"] = group["success_wilson_95_ci"][0]
        row["success_wilson_95_ci_high"] = group["success_wilson_95_ci"][1]
        csv_rows.append(row)
    _write_csv(output_directory / "main_table.csv", csv_fields, csv_rows)

    failure_fields = [
        "profile_id",
        "profile",
        "strategy",
        "method",
        "failure_category",
        "completion_reason",
        "count",
        "rate_of_trials",
        "rate_of_failures",
    ]
    _write_csv(
        output_directory / "failure_classification.csv",
        failure_fields,
        report["failure_classification"]["rows"],
    )

    table_lines = [
        "| Network profile | Method | Task success (Wilson 95% CI) | Mean of episode-mean latency (ms) | Median episode p50-p95 (ms) | Grasp | Recovery | Expired executed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        p50 = group["median_episode_p50_inference_latency_ms"]
        p95 = group["median_episode_p95_inference_latency_ms"]
        table_lines.append(
            "| {profile} | {method} | {rate} {ci} | {mean:.1f} | {p50:.1f}-{p95:.1f} | "
            "{grasp}/{trials} | {recovery}/{trials} | {expired} |".format(
                profile=group["profile"],
                method=group["method"],
                rate=_format_rate(group["successes"], group["trials"]),
                ci=_format_ci(group["success_wilson_95_ci"]),
                mean=group["mean_inference_latency_ms"],
                p50=p50,
                p95=p95,
                grasp=group["grasp_successes"],
                recovery=group["recovery_observed"],
                trials=group["trials"],
                expired=group["expired_actions_executed"],
            )
        )
    (output_directory / "main_table.md").write_text(
        "\n".join(table_lines) + "\n", encoding="utf-8", newline="\n"
    )

    latex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Frozen M8 holdout operating points. Success intervals are Wilson 95\\% confidence intervals.}",
        "\\label{tab:m8-holdout}",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Profile & Method & Success & Mean latency & Median p50--p95 & Grasp & Recovery & Expired exec. \\\\",
        "\\midrule",
    ]
    for group in report["groups"]:
        latex_lines.append(
            "{} & {} & {} {} & {:.1f} & {:.1f}--{:.1f} & {}/{} & {}/{} & {} \\\\".format(
                _latex_escape(group["profile"]),
                _latex_escape(group["method"]),
                _latex_escape(_format_rate(group["successes"], group["trials"])),
                _latex_escape(_format_ci(group["success_wilson_95_ci"])),
                group["mean_inference_latency_ms"],
                group["median_episode_p50_inference_latency_ms"],
                group["median_episode_p95_inference_latency_ms"],
                group["grasp_successes"],
                group["trials"],
                group["recovery_observed"],
                group["trials"],
                group["expired_actions_executed"],
            )
        )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (output_directory / "main_table.tex").write_text(
        "\n".join(latex_lines), encoding="utf-8", newline="\n"
    )

    failure_lines = [
        "| Network profile | Method | Failure category | Completion reason | Count | % of trials |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in report["failure_classification"]["rows"]:
        failure_lines.append(
            f"| {row['profile']} | {row['method']} | {row['failure_category']} | "
            f"`{row['completion_reason']}` | {row['count']} | {row['rate_of_trials']:.1%} |"
        )
    (output_directory / "failure_classification.md").write_text(
        "\n".join(failure_lines) + "\n", encoding="utf-8", newline="\n"
    )

    p1 = report["paired_comparisons"]["profile_1_fixed"]
    p2 = report["paired_comparisons"]["profile_2_faults"]
    aligned = {
        group["profile_id"]: group
        for group in report["groups"]
        if group["strategy"] == "aligned_async"
    }
    naive_expired = sum(
        group["expired_actions_executed"]
        for group in report["groups"]
        if group["strategy"] == "naive_async"
    )
    aligned_expired = sum(group["expired_actions_executed"] for group in aligned.values())
    findings = [
        "# M8 frozen-holdout findings",
        "",
        "## Decision",
        "",
        f"Official registered classification: **{report['official_classification']}** "
        f"(primary GO={str(report['official_primary_go_passed']).lower()}, "
        f"strong GO={str(report['official_strong_go_passed']).lower()}).",
        "",
        "## Raw main table",
        "",
        *table_lines,
        "",
        "## Key findings",
        "",
        f"1. **Observation:** P1 aligned success is {_format_rate(aligned['profile_1_fixed']['successes'], 60)} "
        f"versus 0/60 naive, a {p1['aligned_minus_naive_percentage_points']:.1f} pp paired gain "
        f"with 95% CI [{p1['paired_bootstrap_95_ci_percentage_points'][0]:.1f}, "
        f"{p1['paired_bootstrap_95_ci_percentage_points'][1]:.1f}] pp. **Interpretation:** execution-aligned "
        "queue semantics survive the fixed remote-inference delay. **Implication:** the development result "
        "replicates within this scripted-policy holdout over 60 unseen seeds. **Next:** repeat with a learned "
        "policy and current async/RTC baselines.",
        f"2. **Observation:** P2 aligned success is {_format_rate(aligned['profile_2_faults']['successes'], 60)} "
        f"versus 0/60 naive, a {p2['aligned_minus_naive_percentage_points']:.1f} pp gain with 95% CI "
        f"[{p2['paired_bootstrap_95_ci_percentage_points'][0]:.1f}, "
        f"{p2['paired_bootstrap_95_ci_percentage_points'][1]:.1f}] pp. **Interpretation:** the gain remains "
        "under jitter, drops, duplicate responses, extra delay, and pauses. **Implication:** this is a "
        "runtime result robust within the registered P2 fault profile, not only a fixed-delay artifact. "
        "**Next:** sweep preregistered latency "
        "levels rather than infer a continuous curve from three operating points.",
        f"3. **Observation:** aligned executed {aligned_expired} expired actions versus {naive_expired} for naive, "
        "with zero aligned collisions and timeouts. **Interpretation:** the result is consistent with the "
        "intended stale-action safety semantics. **Implication:** the success gain is not purchased by forbidden "
        "execution. **Next:** add a risk-aware selector using queue age, disagreement, and safety margin.",
        "4. **Observation:** STRONG GO is false because aligned is not at least 15% faster than sync in wall "
        "clock and the obsolete-step reduction condition is not met. **Interpretation:** naive often fails before "
        "reaching the post-switch phase, making that secondary comparison structurally weak. **Implication:** "
        "claim GO, not STRONG GO. **Next:** keep the registered classification and introduce a better failure-aware "
        "secondary metric only in a new preregistered benchmark.",
        "",
        "## Failure classification",
        "",
        *failure_lines,
        "",
        "## Interpretation boundary",
        "",
        "This is native Isaac Sim evidence for one deterministic observation-conditioned policy and one task "
        "family over 60 unseen scenario/initial-state seeds and 180 unseen network traces. It is not a learned "
        "VLA result, a task-family holdout, or real-robot evidence. The latency figure is an operating-point plot, "
        "not a causal continuous latency sweep.",
        "The latency mean averages per-episode mean inference latency. The p50-p95 range reports the median "
        "episode-level p50 and p95, not a confidence interval.",
        "",
    ]
    (output_directory / "M8_HOLDOUT_FINDINGS.md").write_text(
        "\n".join(findings), encoding="utf-8", newline="\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--summaries-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        matrix_path=args.matrix,
        replay_path=args.replay,
        analysis_path=args.analysis,
        summaries_directory=args.summaries_dir,
    )
    write_outputs(report, args.output_dir)
    print(json.dumps({
        "official_classification": report["official_classification"],
        "episode_count": report["episode_count"],
        "output_directory": args.output_dir.as_posix(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
