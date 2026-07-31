"""Generate the M4 pressure headline figure from the validated report."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt

from _plot_style import (
    GRID,
    MODE_COLORS,
    MODE_LABELS,
    MODE_ORDER,
    ROOT,
    add_panel_label,
    configure_style,
    save_figure,
)


def load_report() -> dict:
    path = ROOT / "outputs" / "m4" / "report" / "m4_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "validated"
    assert report["validation"]["passed"] is True
    assert report["bootstrap"]["seed"] == 20260730
    assert report["bootstrap"]["resamples"] == 10_000
    return report


def effect_annotation(comparison: dict, metric: str, scale: float, unit: str) -> str:
    value = comparison["overall"][metric]
    estimate = value["paired_mean_difference"] * scale
    low, high = [
        item * scale
        for item in value["paired_mean_difference_95pct_bootstrap_ci"]
    ]
    return (
        f"Aligned - naive: {estimate:+.1f}{unit}\n"
        f"95% paired CI [{low:.1f}, {high:.1f}]{unit}"
    )


def main() -> None:
    configure_style()
    report = load_report()
    pressure_delay = report["calibration"]["selected_pressure_delay_ms"]
    assert pressure_delay == 950

    summaries = {
        item["runtime_mode"]: item
        for item in report["pressure_condition_summaries"]
    }
    assert tuple(mode for mode in MODE_ORDER if mode in summaries) == MODE_ORDER
    assert all(summaries[mode]["injected_delay_ms"] == pressure_delay for mode in MODE_ORDER)

    comparison = next(
        item
        for item in report["m4_paired_comparisons"]
        if item["comparison_id"] == "pressure_async_aligned_minus_async_naive"
    )

    panels = (
        ("success", "Success rate (%)", 100.0, " pp"),
        ("environment_steps", "Environment steps", 1.0, " steps"),
        ("wall_clock_episode_seconds", "Wall-clock time (s)", 1.0, " s"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.45))

    for panel_index, (ax, (metric, y_label, scale, unit)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        values = [
            summaries[mode]["overall"][metric]["mean"] * scale
            for mode in MODE_ORDER
        ]
        bars = ax.bar(
            range(len(MODE_ORDER)),
            values,
            width=0.68,
            color=[MODE_COLORS[mode] for mode in MODE_ORDER],
            edgecolor="white",
            linewidth=0.7,
        )
        ax.set_xticks(
            range(len(MODE_ORDER)),
            [MODE_LABELS[mode] for mode in MODE_ORDER],
        )
        ax.set_ylabel(y_label)
        ax.yaxis.grid(True, color=GRID, linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
        ax.margins(y=0.20)
        add_panel_label(ax, chr(ord("a") + panel_index))

        if metric == "success":
            counts = [
                round(
                    summaries[mode]["overall"][metric]["mean"]
                    * summaries[mode]["overall"][metric]["episode_count"]
                )
                for mode in MODE_ORDER
            ]
            labels = [
                f"{value:.1f}%\n({count}/30)"
                for value, count in zip(values, counts, strict=True)
            ]
            ax.set_ylim(0, 112)
        elif metric == "environment_steps":
            labels = [f"{value:.1f}" for value in values]
            ax.set_ylim(0, max(values) * 1.27)
        else:
            labels = [f"{value:.2f}" for value in values]
            ax.set_ylim(0, max(values) * 1.27)

        ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
        ax.text(
            0.5,
            0.055,
            effect_annotation(comparison, metric, scale, unit),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.4,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": GRID,
                "alpha": 0.93,
            },
        )

    fig.text(
        0.5,
        -0.015,
        "Frozen 950 ms queue-pressure protocol; 30 paired episodes per mode.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(w_pad=1.3)
    save_figure(fig, "m4_pressure_headline")
    plt.close(fig)


if __name__ == "__main__":
    main()
