"""Generate the queue-pressure calibration selection figure."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt

from _plot_style import ACCENT, GRID, ROOT, configure_style, save_figure


def main() -> None:
    configure_style()
    path = ROOT / "outputs" / "m4" / "calibration" / "selection.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    assert selection["status"] == "selected"
    assert selection["selection_rule_frozen_before_full_pressure_run"] is True

    decisions = sorted(
        selection["delay_decisions"],
        key=lambda item: item["injected_delay_ms"],
    )
    delays = [item["injected_delay_ms"] for item in decisions]
    medians = [
        item["pooled_effective_delivery_age_steps"]["median"]
        for item in decisions
    ]
    p95_values = [
        item["pooled_effective_delivery_age_steps"]["p95"]
        for item in decisions
    ]
    holds = [item["queue_hold_steps_total"] for item in decisions]
    headroom = selection["queue_headroom_steps"]
    selected_delay = selection["selected_pressure_delay_ms"]
    assert selected_delay == 950

    fig, ax = plt.subplots(figsize=(6.5, 3.65))
    ax.plot(
        delays,
        medians,
        color=ACCENT,
        marker="o",
        linewidth=2.0,
        markersize=6,
        label="Median delivery age",
    )
    ax.plot(
        delays,
        p95_values,
        color="#56B4E9",
        marker="s",
        linewidth=1.4,
        markersize=4.5,
        linestyle="--",
        label="P95 delivery age",
    )
    ax.axhline(
        headroom,
        color="#6B7280",
        linestyle="--",
        linewidth=1.1,
        label=f"Queue headroom ({headroom} steps)",
    )
    ax.axvline(selected_delay, color="#E69F00", linewidth=1.0, alpha=0.9)

    for delay, median, hold_count in zip(delays, medians, holds, strict=True):
        label = f"{median:.1f} steps\n{hold_count} holds"
        offset = (0, 10) if delay != selected_delay else (0, -31)
        ax.annotate(
            label,
            xy=(delay, median),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color="#202124",
        )

    selected_index = delays.index(selected_delay)
    ax.scatter(
        [selected_delay],
        [medians[selected_index]],
        marker="*",
        s=150,
        color="#E69F00",
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="First qualifying pressure point",
    )
    ax.set_xlabel("Injected delivery delay (ms)")
    ax.set_ylabel("Effective delivery age (control steps)")
    ax.set_xticks(delays)
    ax.set_ylim(0, max(p95_values) + 5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.text(
        0.5,
        -0.02,
        "Calibration used six episodes per delay; no fully stale chunk occurred.",
        ha="center",
        fontsize=8.2,
    )
    fig.tight_layout()
    save_figure(fig, "queue_pressure_calibration")
    plt.close(fig)


if __name__ == "__main__":
    main()
