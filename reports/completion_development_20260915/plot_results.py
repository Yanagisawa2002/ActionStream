"""Generate the development-only control figure from retained raw evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    replay = read("independent_replay.json")
    temporal = read("temporal/summary.json")
    live = read("live-development-validation/summary.json")
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    ax = axes[0]
    for i, factor in enumerate(("arm", "gripper", "camera")):
        r = replay["nuisance"][factor]
        delta = r["deltas"]
        ax.scatter(
            np.full(len(delta), i) + np.linspace(-0.08, 0.08, len(delta)),
            delta,
            s=35,
            color="#235789",
            alpha=0.8,
        )
        ax.text(
            i,
            0.015,
            f"{r['complete_to_other']}/{r['eligible_pairs']} flips",
            ha="center",
            fontsize=9,
        )
    ax.axhline(0, color="#888888", linewidth=0.7)
    ax.set(
        xticks=[0, 1, 2],
        xticklabels=["Arm", "Gripper", "Camera"],
        ylim=(-0.135, 0.045),
        ylabel="Change in completion probability",
        title="A. Fixed object, changed context",
    )
    ax.text(
        0.03,
        0.04,
        "4 training scenes; valid, visible pairs only",
        transform=ax.transAxes,
        fontsize=8,
    )

    ax = axes[1]
    modes = ["sparse_4hz", "dense_20hz", "dense_20hz_confirm_0.5s"]
    comp = replay["temporal"]["train"]
    x = np.arange(3)
    for offset, field, label, color in (
        (-0.18, "premature_stops", "Premature stop", "#b33c35"),
        (0.18, "missed_completed_events", "Missed completed event", "#d29c38"),
    ):
        bars = ax.bar(
            x + offset, [comp[m][field] for m in modes], 0.35, label=label, color=color
        )
        ax.bar_label(bars, padding=3)
    ax.set(
        xticks=x,
        xticklabels=["4 Hz", "20 Hz", "20 Hz +\n0.5 s confirm"],
        ylim=(0, 8.5),
        ylabel="Episodes out of 8",
        title="B. Same releases, frozen v2",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[2]
    old = [
        r["modes"]["dense_20hz_confirm_0.5s"]["confirmation_delay_s"]
        for r in temporal["results"]
        if r["split"] == "validation"
    ]
    new = [r["confirmation_delay_s"] for r in live["episodes"]]
    x = np.arange(4)
    ax.bar(x - 0.18, old, 0.35, color="#8998a5", label="Frozen v2 + confirmation")
    bars = ax.bar(
        x + 0.18, new, 0.35, color="#247a58", label="Adapted + confirmation (live)"
    )
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.axhline(2, linestyle="--", color="#b33c35", linewidth=1)
    ax.text(3.45, 2.04, "2 s budget", ha="right", fontsize=8, color="#b33c35")
    ax.set(
        xticks=x,
        xticklabels=["1808", "1809", "1810", "1811"],
        xlabel="Development seed suffix",
        ylabel="Safe confirmation delay (seconds)",
        ylim=(0, 2.85),
        title="C. Same 4 validation scenes",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Completion controls: development evidence, independent acceptance pending",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(HERE / "development_controls.png", dpi=180)
    fig.savefig(HERE / "development_controls.svg")
    svg = HERE / "development_controls.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
