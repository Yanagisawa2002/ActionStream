"""Generate paired M3 aligned-minus-naive latency effects."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from _plot_style import ACCENT, GRID, ROOT, add_panel_label, configure_style, save_figure


def load_analysis() -> dict:
    path = ROOT / "outputs" / "m4" / "paired_m3" / "paired_analysis.json"
    analysis = json.loads(path.read_text(encoding="utf-8"))
    assert analysis["m3_validation"]["passed"] is True
    assert analysis["episode_count"] == 180
    assert analysis["bootstrap"]["seed"] == 20260730
    assert analysis["bootstrap"]["resamples"] == 10_000
    return analysis


def metric_effect(comparison: dict, metric: str) -> tuple[float, float, float]:
    value = comparison["overall"][metric]
    estimate = value["paired_mean_difference"]
    low, high = value["paired_mean_difference_95pct_bootstrap_ci"]
    return estimate, low, high


def main() -> None:
    configure_style()
    analysis = load_analysis()
    comparisons = {
        item["comparison_id"]: item for item in analysis["paired_comparisons"]
    }
    selected = [
        comparisons["async_aligned_minus_async_naive_delay0"],
        comparisons["async_aligned_minus_async_naive_delay200"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.7))
    specs = (
        ("environment_steps", "Aligned - naive steps", "steps"),
        ("wall_clock_episode_seconds", "Aligned - naive wall time (s)", "s"),
    )
    y_positions = np.array([0.72, 0.28])

    for index, (ax, (metric, x_label, unit)) in enumerate(
        zip(axes, specs, strict=True)
    ):
        effects = [metric_effect(item, metric) for item in selected]
        estimates = np.array([item[0] for item in effects])
        lows = np.array([item[1] for item in effects])
        highs = np.array([item[2] for item in effects])
        xerr = np.vstack((estimates - lows, highs - estimates))

        ax.axvline(0, color="#6B7280", linewidth=0.9, linestyle="--")
        ax.errorbar(
            estimates,
            y_positions,
            xerr=xerr,
            fmt="o",
            color=ACCENT,
            ecolor=ACCENT,
            elinewidth=2.0,
            capsize=4,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
        ax.set_yticks(y_positions, ["0 ms", "200 ms"])
        ax.set_ylim(0.05, 0.95)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Injected delivery delay")
        ax.xaxis.grid(True, color=GRID, linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
        add_panel_label(ax, chr(ord("a") + index))

        span = max(highs) - min(lows)
        margin = max(span * 0.18, 0.5)
        ax.set_xlim(min(lows) - margin, max(0.0, max(highs)) + margin)
        for y, estimate, low, high in zip(
            y_positions, estimates, lows, highs, strict=True
        ):
            ax.annotate(
                f"{estimate:+.3f} {unit}\n[{low:.3f}, {high:.3f}]",
                xy=(estimate, y),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
            )

    fig.text(
        0.5,
        -0.02,
        "Mean paired difference with 95% episode-bootstrap CI; all four M3 conditions were 30/30 successful.",
        ha="center",
        fontsize=8.2,
    )
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "m3_latency_effects")
    plt.close(fig)


if __name__ == "__main__":
    main()
