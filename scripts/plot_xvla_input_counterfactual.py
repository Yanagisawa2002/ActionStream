"""Render the frozen X-VLA Isaac image/state counterfactual summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CONDITIONS = [
    "official_images__official_state",
    "native_images__official_state",
    "official_images__native_state",
    "native_images__native_state",
]
SHORT_CONDITIONS = ["O img / O state", "N img / O state", "O img / N state", "N img / N state"]
OFFICIAL_COLOR = "#2A6FBB"
NATIVE_COLOR = "#D65F2E"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--official-image", type=Path, required=True)
    parser.add_argument("--native-image", type=Path, required=True)
    parser.add_argument("--official-image2", type=Path, required=True)
    parser.add_argument("--native-image2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _condition_means(records: list[dict[str, object]]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(
            [record["actions"] for record in records if record["condition"] == key],
            dtype=np.float64,
        ).mean(axis=0)
        for key in CONDITIONS
    }


def main() -> int:
    args = _parser().parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    effects = summary["analysis"]["aggregate"]
    means = _condition_means(summary["records"])

    figure = plt.figure(figsize=(15.5, 9.0), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1.05, 1.05, 1.0, 1.0),
        left=0.035,
        right=0.985,
        top=0.88,
        bottom=0.17,
        wspace=0.23,
        hspace=0.22,
    )
    image_specs = [
        (grid[0, 0], args.official_image, "Official LIBERO · agent view"),
        (grid[0, 1], args.native_image, "Native Isaac · agent view"),
        (grid[1, 0], args.official_image2, "Official LIBERO · wrist view"),
        (grid[1, 1], args.native_image2, "Native Isaac · wrist view"),
    ]
    for spec, path, title in image_specs:
        axis = figure.add_subplot(spec)
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#D5D9E0")

    comparison_labels = ["Images only", "State only"]
    colors = [NATIVE_COLOR, OFFICIAL_COLOR]
    chunk_values = [
        effects["image_effect_holding_official_state"]["full_chunk_rmse"]["mean"],
        effects["state_effect_holding_official_images"]["full_chunk_rmse"]["mean"],
    ]
    xyz_values = [
        effects["image_effect_holding_official_state"]["first_action_xyz_l2"]["mean"],
        effects["state_effect_holding_official_images"]["first_action_xyz_l2"]["mean"],
    ]

    axis = figure.add_subplot(grid[0, 2])
    bars = axis.bar(comparison_labels, chunk_values, color=colors, width=0.62)
    axis.set_title("Full action-chunk shift", fontsize=11, fontweight="bold")
    axis.set_ylabel("RMSE across 30 × 7 actions")
    axis.set_ylim(0.0, max(chunk_values) * 1.22)
    axis.bar_label(bars, labels=[f"{value:.4f}" for value in chunk_values], padding=4)
    axis.grid(axis="y", alpha=0.22)

    axis = figure.add_subplot(grid[0, 3])
    bars = axis.bar(comparison_labels, xyz_values, color=colors, width=0.62)
    axis.set_title("First commanded position shift", fontsize=11, fontweight="bold")
    axis.set_ylabel("XYZ L2 distance (m)")
    axis.set_ylim(0.0, max(xyz_values) * 1.22)
    axis.bar_label(bars, labels=[f"{value:.4f}" for value in xyz_values], padding=4)
    axis.grid(axis="y", alpha=0.22)

    x = np.arange(len(CONDITIONS))
    condition_colors = [OFFICIAL_COLOR, NATIVE_COLOR, OFFICIAL_COLOR, NATIVE_COLOR]
    axis = figure.add_subplot(grid[1, 2])
    gripper = [float(means[key][0, 6]) for key in CONDITIONS]
    bars = axis.bar(x, gripper, color=condition_colors, width=0.68)
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_title("First raw gripper command", fontsize=11, fontweight="bold")
    axis.set_ylabel("X-VLA action sign")
    axis.set_ylim(-1.25, 1.25)
    axis.set_xticks(x, SHORT_CONDITIONS, rotation=24, ha="right", fontsize=8.5)
    axis.bar_label(bars, labels=[f"{value:+.0f}" for value in gripper], padding=3)
    axis.grid(axis="y", alpha=0.22)

    axis = figure.add_subplot(grid[1, 3])
    first_z = [float(means[key][0, 2]) for key in CONDITIONS]
    bars = axis.bar(x, first_z, color=condition_colors, width=0.68)
    axis.set_title("First commanded Z", fontsize=11, fontweight="bold")
    axis.set_ylabel("Absolute LIBERO-space Z (m)")
    axis.set_ylim(0.0, max(first_z) * 1.25)
    axis.set_xticks(x, SHORT_CONDITIONS, rotation=24, ha="right", fontsize=8.5)
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in first_z], padding=3)
    axis.grid(axis="y", alpha=0.22)

    image_ratio = chunk_values[0] / chunk_values[1]
    figure.suptitle(
        "X-VLA first-chunk counterfactual: rendered images dominate the native-Isaac shift",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        f"3 paired inference seeds · same checkpoint, instruction, processor and reset RNG · "
        f"image-only shift is {image_ratio:.1f}× state-only by chunk RMSE · "
        "O = official, N = native · offline diagnostic, no action executed",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
