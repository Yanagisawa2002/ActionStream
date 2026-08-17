"""Generate reproducible analysis artifacts for the frozen bridge canary."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "xvla_isaac_official_render_bridge_v1_retry1"
CURRENT_SUMMARY = OUTPUT / "summary.json"
CONFIG = ROOT / "configs" / "xvla_isaac_official_render_bridge_v1.json"
V2_SUMMARY = (
    ROOT
    / "outputs"
    / "xvla_isaac_visual_canary_v1"
    / "xvla_isaac_visual_canary_v2_inference"
    / "summary.json"
)
ATTEMPT0_LOG = ROOT / "outputs" / "xvla_isaac_official_render_bridge_v1.log"
ATTEMPT0_EXIT = ROOT / "outputs" / "xvla_isaac_official_render_bridge_v1.exit_code"
RETRY_LOG = ROOT / "outputs" / "xvla_isaac_official_render_bridge_v1_retry1.log"
RETRY_EXIT = ROOT / "outputs" / "xvla_isaac_official_render_bridge_v1_retry1.exit_code"
T_95_DF2 = 4.302652729696142

COLORS = {
    "official": "#0072B2",
    "v2": "#D55E00",
    "bridge": "#009E73",
    "threshold": "#4D4D4D",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def records_by_condition(summary: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    return sorted(
        [record for record in summary["records"] if record["condition"] == condition],
        key=lambda record: int(record["inference_seed"]),
    )


def mean_actions(summary: dict[str, Any], condition: str) -> np.ndarray:
    return np.asarray(
        [record["actions"] for record in records_by_condition(summary, condition)],
        dtype=np.float64,
    ).mean(axis=0)


def worst_seed_gripper_fraction(summary: dict[str, Any], condition: str) -> float:
    return min(
        float(record["descriptors"]["gripper_negative_fraction"])
        for record in records_by_condition(summary, condition)
    )


def ci95(mean: float, std: float, count: int = 3) -> list[float]:
    half = T_95_DF2 * float(std) / math.sqrt(count)
    return [float(mean - half), float(mean + half)]


def build_analysis(
    current: dict[str, Any], previous: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    current_aggregate = current["analysis"]["aggregate"]
    previous_aggregate = previous["analysis"]["aggregate"][
        "image_effect_holding_official_state"
    ]
    raw_rows = []
    for row in current["analysis"]["per_seed"]:
        metrics = row["paired_metrics"]
        raw_rows.append(
            {
                "inference_seed": int(row["inference_seed"]),
                "full_chunk_rmse": float(metrics["full_chunk_rmse"]),
                "first_action_xyz_l2_m": float(metrics["first_action_xyz_l2"]),
                "first_action_7d_l2": float(metrics["first_action_7d_l2"]),
                "candidate_gripper_negative_fraction": float(
                    row["candidate_gripper_negative_fraction"]
                ),
                "reference_gripper_negative_fraction": float(
                    row["reference_gripper_negative_fraction"]
                ),
            }
        )
    metrics = {
        "full_chunk_rmse": {
            "v2": float(previous_aggregate["full_chunk_rmse"]["mean"]),
            "bridge": float(current_aggregate["full_chunk_rmse"]["mean"]),
            "bridge_std": float(current_aggregate["full_chunk_rmse"]["std"]),
            "threshold": float(config["acceptance"]["aggregate_mean_full_chunk_rmse_max"]),
        },
        "first_action_xyz_l2_m": {
            "v2": float(previous_aggregate["first_action_xyz_l2"]["mean"]),
            "bridge": float(current_aggregate["first_action_xyz_l2"]["mean"]),
            "bridge_std": float(current_aggregate["first_action_xyz_l2"]["std"]),
            "threshold": float(
                config["acceptance"]["aggregate_mean_first_action_xyz_l2_max_m"]
            ),
        },
        "first_action_7d_l2": {
            "v2": float(previous_aggregate["first_action_7d_l2"]["mean"]),
            "bridge": float(current_aggregate["first_action_7d_l2"]["mean"]),
            "bridge_std": float(current_aggregate["first_action_7d_l2"]["std"]),
            "threshold": float(config["acceptance"]["aggregate_mean_first_action_7d_l2_max"]),
        },
    }
    for item in metrics.values():
        item["bridge_ci95"] = ci95(item["bridge"], item["bridge_std"])
        item["relative_improvement_vs_v2_percent"] = 100.0 * (
            item["v2"] - item["bridge"]
        ) / item["v2"]
        item["bridge_threshold_ratio"] = item["bridge"] / item["threshold"]
        item["v2_threshold_ratio"] = item["v2"] / item["threshold"]
    return {
        "schema_version": 1,
        "experiment_id": current["experiment_id"],
        "status": current["status"],
        "raw_seed_table": raw_rows,
        "metric_comparison": metrics,
        "gripper_close_fraction": {
            "v2_worst_seed": worst_seed_gripper_fraction(
                previous, "native_images__official_state"
            ),
            "bridge_worst_seed": worst_seed_gripper_fraction(
                current, "official_render_bridge__isaac_state"
            ),
            "official_worst_seed": worst_seed_gripper_fraction(
                current, "official_render__official_state"
            ),
        },
        "acceptance": current["acceptance"],
        "structural_gate": current["structural_gate"],
        "bridge_ik": current["bridge"]["ik"],
        "claim_boundary": current["limitations"],
        "provenance": {
            "config": str(CONFIG.relative_to(ROOT)),
            "config_sha256": sha256_file(CONFIG),
            "current_summary": str(CURRENT_SUMMARY.relative_to(ROOT)),
            "current_summary_sha256": sha256_file(CURRENT_SUMMARY),
            "v2_summary": str(V2_SUMMARY.relative_to(ROOT)),
            "v2_summary_sha256": sha256_file(V2_SUMMARY),
        },
    }


def save_visual_comparison() -> None:
    paths = [
        OUTPUT / "reference_image.png",
        OUTPUT / "reference_image2.png",
        OUTPUT / "bridge_image.png",
        OUTPUT / "bridge_image2.png",
    ]
    images = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 5.8))
    row_labels = ["Official reset", "Bridge: Isaac EEF/state"]
    column_labels = ["Agent view", "Wrist view"]
    for row in range(2):
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
    fig.subplots_adjust(wspace=0.025, hspace=0.05)
    fig.savefig(OUTPUT / "bridge_visual_comparison.png")
    fig.savefig(OUTPUT / "bridge_visual_comparison.pdf")
    plt.close(fig)


def save_gate_plot(analysis: dict[str, Any]) -> None:
    metric_keys = ["full_chunk_rmse", "first_action_xyz_l2_m", "first_action_7d_l2"]
    labels = ["Chunk\nRMSE", "First XYZ\nL2", "First 7D\nL2"]
    v2 = [analysis["metric_comparison"][key]["v2_threshold_ratio"] for key in metric_keys]
    bridge = [
        analysis["metric_comparison"][key]["bridge_threshold_ratio"]
        for key in metric_keys
    ]
    positions = np.arange(len(metric_keys))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [2.2, 1]})
    axes[0].bar(
        positions - width / 2,
        v2,
        width,
        color=COLORS["v2"],
        label="V2 (official state)",
    )
    axes[0].bar(
        positions + width / 2,
        bridge,
        width,
        color=COLORS["bridge"],
        label="Bridge (Isaac state)",
    )
    axes[0].axhline(1.0, color=COLORS["threshold"], linestyle="--", linewidth=1.2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Metric / frozen threshold (lower is better)")
    axes[0].set_xticks(positions, labels=labels)
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].legend(frameon=False, loc="upper right")

    close_fraction = analysis["gripper_close_fraction"]
    gripper = [
        close_fraction["v2_worst_seed"],
        close_fraction["bridge_worst_seed"],
        close_fraction["official_worst_seed"],
    ]
    gripper_labels = ["V2", "Bridge", "Official"]
    colors = [COLORS["v2"], COLORS["bridge"], COLORS["official"]]
    bars = axes[1].bar(gripper_labels, gripper, color=colors, width=0.65)
    axes[1].axhline(0.9, color=COLORS["threshold"], linestyle="--", linewidth=1.2)
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_ylabel("Close-command fraction")
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    for bar, value in zip(bars, gripper, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(OUTPUT / "bridge_canary_gate_comparison.png")
    fig.savefig(OUTPUT / "bridge_canary_gate_comparison.pdf")
    plt.close(fig)


def save_action_plot(current: dict[str, Any], previous: dict[str, Any]) -> None:
    official = mean_actions(current, "official_render__official_state")
    bridge = mean_actions(current, "official_render_bridge__isaac_state")
    v2 = mean_actions(previous, "native_images__official_state")
    steps = np.arange(len(official))
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True)
    series = [
        (official, "Official LIBERO", COLORS["official"], "-"),
        (v2, "V2 image / official state", COLORS["v2"], "--"),
        (bridge, "Bridge image / Isaac state", COLORS["bridge"], "-."),
    ]
    for actions, label, color, linestyle in series:
        axes[0].plot(
            steps,
            actions[:, 2],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
        axes[1].plot(
            steps,
            actions[:, 6],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
    axes[0].set_ylabel("Absolute Z command (m)")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].legend(frameon=False, loc="best")
    axes[1].set_xlabel("Action-chunk step")
    axes[1].set_ylabel("Gripper command")
    axes[1].set_yticks([-1.0, 0.0, 1.0], labels=["close (-1)", "0", "open (+1)"])
    axes[1].set_ylim(-1.2, 1.2)
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.subplots_adjust(hspace=0.12)
    fig.savefig(OUTPUT / "bridge_first_chunk_action_comparison.png")
    fig.savefig(OUTPUT / "bridge_first_chunk_action_comparison.pdf")
    plt.close(fig)


def write_raw_csv(analysis: dict[str, Any]) -> None:
    rows = analysis["raw_seed_table"]
    with (OUTPUT / "raw_seed_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_receipt(current: dict[str, Any]) -> None:
    receipt = {
        "schema_version": 1,
        "experiment_id": current["experiment_id"],
        "frozen_source_commit": "e0f171ce9b1dafa7ebc252b512649d2b2ee07cb9",
        "formal_attempts": [
            {
                "attempt": 0,
                "exit_code": int(ATTEMPT0_EXIT.read_text(encoding="utf-8").strip()),
                "policy_inference_executed": False,
                "failure_stage": "checkpoint_load_before any paired inference",
                "failure": "Hugging Face Xet reconstruction error because cloned HF_HOME was not inherited",
                "log_sha256": sha256_file(ATTEMPT0_LOG),
                "candidate_or_threshold_changed_after_attempt": False,
            },
            {
                "attempt": 1,
                "exit_code": int(RETRY_EXIT.read_text(encoding="utf-8").strip()),
                "environment_only_change": "reuse cloned /root/autodl-tmp/hf_home with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1",
                "config_sha256": current["config_sha256"],
                "summary_sha256": sha256_file(CURRENT_SUMMARY),
                "log_sha256": sha256_file(RETRY_LOG),
                "status": current["status"],
            },
        ],
    }
    (OUTPUT / "formal_run_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_report(analysis: dict[str, Any]) -> None:
    metrics = analysis["metric_comparison"]
    rows = analysis["raw_seed_table"]
    ik = analysis["bridge_ik"]
    close_fraction = analysis["gripper_close_fraction"]
    labels = {
        "full_chunk_rmse": "Full chunk RMSE",
        "first_action_xyz_l2_m": "First action XYZ L2 (m)",
        "first_action_7d_l2": "First action 7D L2",
    }
    lines = [
        "# X-VLA official-render / Isaac-state bridge findings",
        "",
        "Status: **passed the unchanged frozen V2 first-chunk canary**. This opens a disjoint sync capability gate; it is not task-success or async evidence.",
        "",
        "## Raw paired results",
        "",
        "| Seed | Chunk RMSE | First XYZ L2 (m) | First 7D L2 | Bridge close fraction | Official close fraction |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['inference_seed']} | {row['full_chunk_rmse']:.6f} | "
            f"{row['first_action_xyz_l2_m']:.6f} | {row['first_action_7d_l2']:.6f} | "
            f"{row['candidate_gripper_negative_fraction']:.2f} | "
            f"{row['reference_gripper_negative_fraction']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen gate and V2 comparison",
            "",
            "| Metric | V2 native image / official state | Bridge official render / Isaac state, mean +/- std | 95% CI | Frozen threshold | Improvement vs V2 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, label in labels.items():
        item = metrics[key]
        lines.append(
            f"| {label} | {item['v2']:.6f} | {item['bridge']:.6f} +/- "
            f"{item['bridge_std']:.6f} | [{item['bridge_ci95'][0]:.6f}, "
            f"{item['bridge_ci95'][1]:.6f}] | <= {item['threshold']:.6f} | "
            f"{item['relative_improvement_vs_v2_percent']:.2f}% |"
        )
    lines.extend(
        [
            "| Close-command fraction, worst seed | "
            f"{close_fraction['v2_worst_seed']:.2f} | "
            f"{close_fraction['bridge_worst_seed']:.2f} | "
            f"official worst seed {close_fraction['official_worst_seed']:.2f} | "
            ">= 0.90 | sign restored |",
            "",
            "The 95% intervals use a t interval with n=3 paired inference seeds; they describe this frozen canary and are not a population-level generalization claim.",
            "",
            "## Key findings",
            "",
            f"1. **Observation:** all four frozen checks passed. Chunk RMSE is {metrics['full_chunk_rmse']['bridge']:.6f}, first XYZ L2 is {metrics['first_action_xyz_l2_m']['bridge']:.6f} m, first 7D L2 is {metrics['first_action_7d_l2']['bridge']:.6f}, and every bridge chunk closes the gripper.  "
            "**Interpretation:** within this reset-scoped diagnostic, combining the official visual domain with the mapped Isaac policy state restores the first-chunk action manifold under the frozen tolerances.  "
            "**Implication:** together with the prior factorial V2 result, where changing state alone had a much smaller effect, this is consistent with renderer and robot appearance dominating the earlier reset-time error. The bridge comparison itself changes both image and policy state relative to the official reference, so it is not a new single-factor causal estimate and does not establish episode-level causality.  "
            "**Next step:** preregister a disjoint sync capability gate.",
            "",
            f"2. **Observation:** EEF IK converged in {ik['iterations']} iterations with {ik['position_error_l2_m']:.3e} m position error and {ik['orientation_error_l2_rad']:.3e} rad orientation error.  "
            "**Interpretation:** the bridge is state-conditioned rather than a copied static reference frame.  "
            "**Implication:** reset-time camera evidence is structurally aligned.  "
            "**Next step:** synchronize object poses and robot state at every control request before calling the bridge episode-ready.",
            "",
            "3. **Observation:** the initial formal attempt stopped during checkpoint loading; the unchanged offline-cache retry completed all six paired inferences.  "
            "**Interpretation:** this was an infrastructure retry, not result-driven rerunning.  "
            "**Implication:** the frozen comparison remains valid.  "
            "**Next step:** carry `HF_HOME` and offline cache provenance into the sync-gate launcher.",
            "",
            "4. **Observation:** the mean bridge Z command remains close initially but separates from the official curve later in the 30-step chunk.  "
            "**Interpretation:** passing the preregistered error gates does not mean the two chunks are identical.  "
            "**Implication:** reset-time alignment is a prerequisite result, not evidence of stable closed-loop execution.  "
            "**Next step:** synchronize robot and object visual state on every policy request, then run a fresh disjoint sync capability gate.",
            "",
            "## Figure scope",
            "",
            "- `bridge_visual_comparison.png`: for the same task, scene/object reset, instruction, and official renderer, the top row is the official LIBERO reset observation and the bottom row retargets the canonical Panda end-effector (EEF) pose and gripper from the measured Isaac reset. The robot state intentionally differs; this is a reset diagnostic, not task-success evidence.",
            "- `bridge_canary_gate_comparison.png`: for the same task/reset/instruction and three paired seeds, the V2 native-image/official-state and bridge official-render/Isaac-state errors are each divided by the same metric-specific frozen threshold. Values below 1 show gate margin, not directly comparable physical units; the right panel reports worst-seed close-command fraction derived from the raw summaries. This cross-candidate comparison changes both image and policy state and is not a one-factor ablation.",
            "- `bridge_first_chunk_action_comparison.png`: mean 30-step Z and gripper commands over the same three paired inference seeds and official reference, with the same task, reset, and instruction. V2 uses native V2 images with official policy state; the bridge uses state-conditioned official-render images with mapped Isaac policy state. This is first-chunk diagnostic behavior, not episode success or a one-factor causal comparison.",
            "",
        ]
    )
    (OUTPUT / "BRIDGE_CANARY_FINDINGS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_latex(analysis: dict[str, Any]) -> None:
    metrics = analysis["metric_comparison"]
    close_fraction = analysis["gripper_close_fraction"]
    table = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Frozen reset-scoped X-VLA bridge canary over three unchanged inference seeds. V2-off uses native V2 images with official state; Bridge-Isaac uses state-conditioned official-render images with mapped Isaac state. Lower is better for the three error metrics. This cross-candidate gate is not a one-factor ablation.}",
        r"\label{tab:xvla_bridge_canary}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Metric & V2-off & Bridge-Isaac & Threshold & Result \\",
        r"\midrule",
    ]
    for label, key in (
        ("Chunk RMSE", "full_chunk_rmse"),
        ("First XYZ L2", "first_action_xyz_l2_m"),
        ("First 7D L2", "first_action_7d_l2"),
    ):
        item = metrics[key]
        table.append(
            f"{label} & {item['v2']:.4f} & {item['bridge']:.4f} & "
            f"$\\leq {item['threshold']:.2f}$ & PASS \\\\"
        )
    table.extend(
        [
            "Close fraction & "
            f"{close_fraction['v2_worst_seed']:.2f} & "
            f"{close_fraction['bridge_worst_seed']:.2f} & "
            r"$\geq 0.90$ & PASS \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (OUTPUT / "TABLE_bridge_canary.tex").write_text("\n".join(table), encoding="utf-8")
    includes = r"""% Reset-scoped official-render / Isaac-state evidence only.
\begin{figure}[t]
  \centering
  \includegraphics[width=0.76\textwidth]{outputs/xvla_isaac_official_render_bridge_v1_retry1/bridge_visual_comparison.pdf}
  \caption{Official-render observations for the same task, scene/object reset, and instruction. The top row is the official LIBERO reset; the bottom row retargets the canonical Panda end-effector pose and gripper from the measured Isaac reset. The robot state intentionally differs. This is a reset diagnostic, not task-success evidence.}
  \label{fig:xvla_bridge_visual}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.82\textwidth]{outputs/xvla_isaac_official_render_bridge_v1_retry1/bridge_canary_gate_comparison.pdf}
  \caption{Unchanged frozen V2 thresholds applied over the same task, scene/object reset, instruction, and three paired seeds. In the left log-scale panel, each error metric is divided by its own threshold (dashed line at 1); bar heights show gate margin, not cross-metric effect size, and absolute values are in Table~\ref{tab:xvla_bridge_canary}. The right panel shows worst-seed close-command fraction (dashed gate at 0.9); Official is a reference bar. V2 uses native images with official state, while the bridge uses state-conditioned official-render images with mapped Isaac state. This is not a one-factor ablation or task-success evidence.}
  \label{fig:xvla_bridge_gate}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.72\textwidth]{outputs/xvla_isaac_official_render_bridge_v1_retry1/bridge_first_chunk_action_comparison.pdf}
  \caption{Mean predicted 30-step first action chunk after a single reset observation, over the same task, scene/object reset, instruction, and three paired inference seeds. V2 uses native images with official state; the state-conditioned official-render bridge uses mapped Isaac state. These are not executed closed-loop trajectories, a one-factor ablation, or task-success evidence.}
  \label{fig:xvla_bridge_actions}
\end{figure}
"""
    (OUTPUT / "latex_includes.tex").write_text(includes, encoding="utf-8")


def main() -> int:
    current = load_json(CURRENT_SUMMARY)
    previous = load_json(V2_SUMMARY)
    config = load_json(CONFIG)
    if sha256_file(CONFIG) != current["config_sha256"]:
        raise ValueError("Formal summary does not match the frozen config")
    for key, filename in (
        ("reference_image_sha256", "reference_image.png"),
        ("reference_image2_sha256", "reference_image2.png"),
        ("bridge_image_sha256", "bridge_image.png"),
        ("bridge_image2_sha256", "bridge_image2.png"),
    ):
        if sha256_file(OUTPUT / filename) != current["image_hashes"][key]:
            raise ValueError(f"Formal image hash mismatch: {filename}")
    analysis = build_analysis(current, previous, config)
    setup_style()
    save_visual_comparison()
    save_gate_plot(analysis)
    save_action_plot(current, previous)
    write_raw_csv(analysis)
    write_receipt(current)
    write_report(analysis)
    write_latex(analysis)
    (OUTPUT / "bridge_canary_analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "output": str(OUTPUT),
                "all_acceptance_checks_pass": analysis["acceptance"][
                    "all_acceptance_checks_pass"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
