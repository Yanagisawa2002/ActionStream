"""Draw the fixed heldout outcome comparison from retained records."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads((HERE / path).read_text())


def main():
    verdict = read("verdict.json")
    paired = read("paired-evaluation/summary.json")["results"]
    live = read("live/summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    rows = [
        ("V2 + confirmation\npaired replay", verdict["paired_baseline"], "#8c9aa5"),
        ("Adapted + confirmation\npaired replay", verdict["paired_adapted"], "#386b9e"),
        ("Adapted + confirmation\nactual live", verdict["live"], "#278162"),
    ]
    keys = [
        "premature_stops",
        "missed_completed_episodes",
        "post_stop_failures",
        "late_confirmations",
    ]
    x = np.arange(4)
    for i, (label, row, color) in enumerate(rows):
        bars = axes[0].bar(
            x + (i - 1) * 0.25, [row[k] for k in keys], 0.24, color=color, label=label
        )
        axes[0].bar_label(bars, padding=3, fontsize=9)
    ceiling = max(r[k] for _, r, _ in rows for k in keys)
    axes[0].set(
        xticks=x,
        xticklabels=[
            "Premature\nstop",
            "Missed\ncompletion",
            "Unstable\nafter stop",
            "Delay\n> 2 seconds",
        ],
        ylabel="Episodes out of 20",
        ylim=(0, max(4, ceiling + 4)),
        title="A. Predeclared closed-loop failures",
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1]
    for offset, label, data, color in [
        (-0.15, "V2 paired", paired["baseline"]["episodes"], "#8c9aa5"),
        (0, "Adapted paired", paired["adapted"]["episodes"], "#386b9e"),
        (0.15, "Adapted live", live["episodes"], "#278162"),
    ]:
        points = [
            (i + offset, e["confirmation_delay_s"])
            for i, e in enumerate(data)
            if e["confirmation_delay_s"] is not None
        ]
        if points:
            ax.scatter(*zip(*points), s=30, color=color, label=label, alpha=0.85)
    ax.axhline(2, color="#b33431", linestyle="--", linewidth=1)
    ax.set(
        xticks=np.arange(0, 20, 2),
        xticklabels=[str(i) for i in range(0, 20, 2)],
        xlabel="Heldout seed index (2026091900 + index)",
        ylabel="Safe first-stop delay (seconds)",
        title="B. Confirmation delays on fresh seeds",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.02,
        0.02,
        "Safe stops only; VLA noncompletion has no delay",
        transform=ax.transAxes,
        fontsize=8,
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Frozen independent acceptance: {verdict['status']}\n20 new trajectories; single simulated tomato-to-basket task",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(HERE / "heldout_outcomes.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
