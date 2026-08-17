"""Generate the X-VLA native-Isaac visual-canary report and figures."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "xvla_isaac_visual_canary_v1"
CURRENT_DIR = OUTPUT / "xvla_isaac_visual_canary_v2_inference"
CAPTURE_DIR = OUTPUT / "learned_isaac_libero_visual_canary_v2_capture"
PREVIOUS_SUMMARY = ROOT / "outputs" / "xvla_isaac_input_counterfactual_v1" / "summary.json"
CURRENT_SUMMARY = CURRENT_DIR / "summary.json"
INFERENCE_CONFIG = ROOT / "configs" / "xvla_isaac_visual_canary_v2_inference.json"

COLORS = {
    "official": "#0072B2",
    "plain": "#E69F00",
    "candidate": "#CC79A7",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def records_by_condition(summary: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    records = [record for record in summary["records"] if record["condition"] == condition]
    return sorted(records, key=lambda record: int(record["inference_seed"]))


def mean_actions(summary: dict[str, Any], condition: str) -> np.ndarray:
    records = records_by_condition(summary, condition)
    return np.asarray([record["actions"] for record in records], dtype=np.float64).mean(axis=0)


def aggregate_metric(summary: dict[str, Any], metric: str) -> dict[str, Any]:
    return summary["analysis"]["aggregate"]["image_effect_holding_official_state"][metric]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "stix",
        }
    )


def save_qualitative_grid() -> None:
    paths = [
        CURRENT_DIR / "official_image.png",
        CURRENT_DIR / "official_image2.png",
        OUTPUT / "visual_inputs" / "plain_native_agentview.png",
        OUTPUT / "visual_inputs" / "plain_native_wrist.png",
        CAPTURE_DIR / "policy_frames" / "frame_000000.png",
        CAPTURE_DIR / "policy_wrist_frames" / "frame_000000.png",
    ]
    images = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    row_labels = ["Official LIBERO", "Plain native Isaac", "Official-table candidate V2"]
    column_labels = ["Agent view", "Wrist view"]
    fig, axes = plt.subplots(3, 2, figsize=(5.8, 8.4))
    for row in range(3):
        for column in range(2):
            axis = axes[row, column]
            axis.imshow(images[row * 2 + column])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_xlabel(column_labels[column], labelpad=5)
                axis.xaxis.set_label_position("top")
            if column == 0:
                axis.set_ylabel(row_labels[row], labelpad=8)
    fig.subplots_adjust(wspace=0.025, hspace=0.06)
    fig.savefig(OUTPUT / "visual_input_comparison.png")
    fig.savefig(OUTPUT / "visual_input_comparison.pdf")
    plt.close(fig)


def save_action_plot(previous: dict[str, Any], current: dict[str, Any]) -> None:
    official = mean_actions(current, "official_images__official_state")
    candidate = mean_actions(current, "native_images__official_state")
    plain = mean_actions(previous, "native_images__official_state")
    steps = np.arange(len(official))
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True)
    series = [
        (official, "Official LIBERO", COLORS["official"], "-"),
        (plain, "Plain native Isaac", COLORS["plain"], "--"),
        (candidate, "Official-table candidate V2", COLORS["candidate"], "-."),
    ]
    for actions, label, color, linestyle in series:
        axes[0].plot(steps, actions[:, 2], label=label, color=color, linestyle=linestyle, linewidth=2)
        axes[1].plot(steps, actions[:, 6], label=label, color=color, linestyle=linestyle, linewidth=2)
    axes[0].set_ylabel("Absolute Z command (m)")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].legend(frameon=False, ncol=1, loc="best")
    axes[1].set_xlabel("Action-chunk step")
    axes[1].set_ylabel("Gripper command")
    axes[1].set_yticks([-1.0, 0.0, 1.0], labels=["close (-1)", "0", "open (+1)"])
    axes[1].set_ylim(-1.2, 1.2)
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.subplots_adjust(hspace=0.12)
    fig.savefig(OUTPUT / "first_chunk_action_comparison.png")
    fig.savefig(OUTPUT / "first_chunk_action_comparison.pdf")
    plt.close(fig)


def build_analysis(previous: dict[str, Any], current: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    current_effects = current["analysis"]["aggregate"]["image_effect_holding_official_state"]
    previous_effects = previous["analysis"]["aggregate"]["image_effect_holding_official_state"]
    candidate_records = records_by_condition(current, "native_images__official_state")
    official_records = records_by_condition(current, "official_images__official_state")
    per_seed_effects = {
        int(row["inference_seed"]): row["effects"]["image_effect_holding_official_state"]
        for row in current["analysis"]["per_seed"]
    }
    raw_rows: list[dict[str, Any]] = []
    for candidate, official in zip(candidate_records, official_records, strict=True):
        seed = int(candidate["inference_seed"])
        metrics = per_seed_effects[seed]
        raw_rows.append(
            {
                "inference_seed": seed,
                "full_chunk_rmse": metrics["full_chunk_rmse"],
                "first_action_xyz_l2_m": metrics["first_action_xyz_l2"],
                "first_action_7d_l2": metrics["first_action_7d_l2"],
                "candidate_gripper_negative_fraction": candidate["descriptors"][
                    "gripper_negative_fraction"
                ],
                "official_gripper_negative_fraction": official["descriptors"][
                    "gripper_negative_fraction"
                ],
                "candidate_first_action_z_m": candidate["actions"][0][2],
                "official_first_action_z_m": official["actions"][0][2],
            }
        )

    thresholds = config["acceptance"]
    candidate_gripper_min = min(
        row["candidate_gripper_negative_fraction"] for row in raw_rows
    )
    checks = {
        "full_chunk_rmse": {
            "value": current_effects["full_chunk_rmse"]["mean"],
            "operator": "<=",
            "threshold": thresholds["aggregate_mean_full_chunk_rmse_max"],
        },
        "first_action_xyz_l2_m": {
            "value": current_effects["first_action_xyz_l2"]["mean"],
            "operator": "<=",
            "threshold": thresholds["aggregate_mean_first_action_xyz_l2_max_m"],
        },
        "first_action_7d_l2": {
            "value": current_effects["first_action_7d_l2"]["mean"],
            "operator": "<=",
            "threshold": thresholds["aggregate_mean_first_action_7d_l2_max"],
        },
        "gripper_negative_fraction_each_seed": {
            "value": candidate_gripper_min,
            "operator": ">=",
            "threshold": thresholds["candidate_gripper_negative_fraction_min_each_seed"],
        },
    }
    for key, check in checks.items():
        if check["operator"] == "<=":
            check["pass"] = check["value"] <= check["threshold"]
        else:
            check["pass"] = check["value"] >= check["threshold"]
        check["margin"] = (
            check["threshold"] - check["value"]
            if check["operator"] == "<="
            else check["value"] - check["threshold"]
        )

    relative: dict[str, Any] = {}
    for metric in ("full_chunk_rmse", "first_action_xyz_l2", "first_action_7d_l2"):
        old = previous_effects[metric]["mean"]
        new = current_effects[metric]["mean"]
        relative[metric] = {
            "plain_native_mean": old,
            "candidate_v2_mean": new,
            "smaller_is_better_relative_improvement_percent": 100.0 * (old - new) / old,
        }

    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "failed_frozen_canary",
        "all_acceptance_checks_pass": all(check["pass"] for check in checks.values()),
        "acceptance_checks": checks,
        "raw_seed_table": raw_rows,
        "aggregate_image_effect": current_effects,
        "relative_to_plain_native": relative,
        "stop_action": "No sync capability gate or async holdout was launched.",
        "claim_boundary": config["analysis_contract"],
        "provenance": {
            "inference_config": str(INFERENCE_CONFIG.relative_to(ROOT)),
            "inference_config_sha256": sha256_file(INFERENCE_CONFIG),
            "current_summary": str(CURRENT_SUMMARY.relative_to(ROOT)),
            "current_summary_sha256": sha256_file(CURRENT_SUMMARY),
            "previous_summary": str(PREVIOUS_SUMMARY.relative_to(ROOT)),
            "previous_summary_sha256": sha256_file(PREVIOUS_SUMMARY),
        },
    }


def write_raw_csv(analysis: dict[str, Any]) -> None:
    rows = analysis["raw_seed_table"]
    with (OUTPUT / "raw_seed_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(analysis: dict[str, Any]) -> None:
    rows = analysis["raw_seed_table"]
    checks = analysis["acceptance_checks"]
    relative = analysis["relative_to_plain_native"]
    aggregate = analysis["aggregate_image_effect"]
    lines = [
        "# X-VLA native-Isaac visual canary",
        "",
        "Status: **failed frozen canary**. The official-table candidate did not qualify for a sync capability gate, so no sync or async holdout was launched.",
        "",
        "This is an offline paired first-action-chunk diagnostic, not task-success evidence.",
        "",
        "## Raw paired results",
        "",
        "| Inference seed | Chunk RMSE | First XYZ L2 (m) | First 7D L2 | Candidate close fraction | Official close fraction | Candidate first Z (m) | Official first Z (m) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['inference_seed']} | {row['full_chunk_rmse']:.6f} | "
            f"{row['first_action_xyz_l2_m']:.6f} | {row['first_action_7d_l2']:.6f} | "
            f"{row['candidate_gripper_negative_fraction']:.2f} | "
            f"{row['official_gripper_negative_fraction']:.2f} | "
            f"{row['candidate_first_action_z_m']:.6f} | "
            f"{row['official_first_action_z_m']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen acceptance",
            "",
            "| Metric | Mean or worst seed | Threshold | Result |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    labels = {
        "full_chunk_rmse": "Full chunk RMSE",
        "first_action_xyz_l2_m": "First action XYZ L2 (m)",
        "first_action_7d_l2": "First action 7D L2",
        "gripper_negative_fraction_each_seed": "Minimum close fraction across seeds",
    }
    for key, label in labels.items():
        check = checks[key]
        lines.append(
            f"| {label} | {check['value']:.6f} | {check['operator']} {check['threshold']:.6f} | "
            f"{'PASS' if check['pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Comparison with the plain native render",
            "",
            "| Metric | Plain native | Official-table V2 | Relative improvement (smaller is better) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key, label in (
        ("full_chunk_rmse", "Full chunk RMSE"),
        ("first_action_xyz_l2", "First action XYZ L2 (m)"),
        ("first_action_7d_l2", "First action 7D L2"),
    ):
        item = relative[key]
        lines.append(
            f"| {label} | {item['plain_native_mean']:.6f} | {item['candidate_v2_mean']:.6f} | "
            f"{item['smaller_is_better_relative_improvement_percent']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Key findings",
            "",
            f"1. **Observation:** first-action XYZ displacement improved from {relative['first_action_xyz_l2']['plain_native_mean']:.6f} m to {relative['first_action_xyz_l2']['candidate_v2_mean']:.6f} m ({relative['first_action_xyz_l2']['smaller_is_better_relative_improvement_percent']:.1f}% smaller), but remained above the frozen 0.05 m threshold.",
            "   **Interpretation:** adding the official diffuse tabletop recovers part of the vertical motion cue.",
            "   **Implication:** appearance parity matters, but tabletop texture alone is insufficient.",
            "   **Next step:** use an official-render/Isaac-state bridge or camera-specific robot render layer before any new runtime benchmark.",
            "",
            f"2. **Observation:** full-chunk RMSE was {aggregate['full_chunk_rmse']['mean']:.6f} +/- {aggregate['full_chunk_rmse']['std']:.6f}, essentially unchanged from the plain native render, and the candidate opened the gripper for every one of 30 actions in every seed while official closed it.",
            "   **Interpretation:** the remaining visual domain gap dominates the chunk after the partially recovered first Z command; visible differences include table extent, exposed blue ground, lighting, and the Isaac hand/body silhouette.",
            "   **Implication:** the previous all-zero native episodes cannot be cleanly interpreted as pure async-scheduling evidence while this visual-domain blocker remains.",
            "   **Next step:** validate a hybrid renderer or exact official camera/robot appearance contract with this same frozen canary before spending task-level episodes.",
            "",
            "3. **Observation:** across-seed standard deviations are below 5e-5 for the primary distances.",
            "   **Interpretation:** this is a stable blocker, not a noisy unlucky seed.",
            "   **Implication:** adding more seeds to the same invalid visual contract has low information value.",
            "   **Next step:** change the rendering contract, then preregister a new candidate version; do not relax the V2 thresholds.",
            "",
            "## Figures",
            "",
            "- `visual_input_comparison.png`: for the same task, reset seed, and instruction, the top row is the official LIBERO reference observation and the lower rows are the plain and official-table V2 native-Isaac reset renders; the two native rows share the same native state contract.",
            "- `first_chunk_action_comparison.png`: mean 30-step absolute Z and gripper commands over three paired inference seeds, with official robot state held fixed for each official-vs-native image pair. It is a first-chunk diagnostic, not task-success evidence.",
            "",
            "Exact actions and paired statistics are preserved in `xvla_isaac_visual_canary_v2_inference/summary.json`; `canary_evaluation.json` records the frozen gate decision.",
            "",
        ]
    )
    (OUTPUT / "CANARY_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def write_latex(analysis: dict[str, Any]) -> None:
    checks = analysis["acceptance_checks"]
    table = [
        r"\begin{tabular}{lrrc}",
        r"\toprule",
        r"Metric & Observed & Frozen threshold & Result \\",
        r"\midrule",
    ]
    for key, label in (
        ("full_chunk_rmse", "Chunk RMSE"),
        ("first_action_xyz_l2_m", "First XYZ L2 (m)"),
        ("first_action_7d_l2", "First 7D L2"),
        ("gripper_negative_fraction_each_seed", "Minimum close fraction"),
    ):
        check = checks[key]
        table.append(
            f"{label} & {check['value']:.4f} & ${check['operator']}$ {check['threshold']:.4f} & "
            f"{'PASS' if check['pass'] else 'FAIL'} " + r"\\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}", ""])
    (OUTPUT / "TABLE_canary.tex").write_text("\n".join(table), encoding="utf-8")
    includes = r"""% Qualitative visual-input comparison
\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\textwidth]{outputs/xvla_isaac_visual_canary_v1/visual_input_comparison.pdf}
  \caption{Policy inputs for the same task, reset seed, and instruction. The top row is the official LIBERO reference observation; the lower rows are plain and official-table V2 native-Isaac reset renders under the same native state contract. V2 restores a wood surface but retains table-extent, lighting, and robot-appearance mismatch.}
  \label{fig:xvla-isaac-visual-inputs}
\end{figure}

% First action-chunk comparison
\begin{figure}[t]
  \centering
  \includegraphics[width=0.75\textwidth]{outputs/xvla_isaac_visual_canary_v1/first_chunk_action_comparison.pdf}
  \caption{Mean X-VLA first action chunk across three paired inference seeds, with official robot state held fixed within each official-vs-native image pair. The table candidate partially recovers the Z command but retains the opposite gripper sign throughout the chunk. This is not task-success evidence.}
  \label{fig:xvla-isaac-first-chunk}
\end{figure}
"""
    (OUTPUT / "latex_includes.tex").write_text(includes, encoding="utf-8")


def main() -> None:
    previous = load_json(PREVIOUS_SUMMARY)
    current = load_json(CURRENT_SUMMARY)
    config = load_json(INFERENCE_CONFIG)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    analysis = build_analysis(previous, current, config)
    (OUTPUT / "canary_evaluation.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_raw_csv(analysis)
    write_report(analysis)
    write_latex(analysis)
    save_qualitative_grid()
    save_action_plot(previous, current)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "all_acceptance_checks_pass": analysis["all_acceptance_checks_pass"],
                "output": str(OUTPUT),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
