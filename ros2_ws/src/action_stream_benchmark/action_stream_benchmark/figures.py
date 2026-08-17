"""Reproducible, artifact-backed benchmark figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import read_json, read_jsonl


COLORS = {
    "sync_hold": "#6B7280",
    "naive_async": "#D97706",
    "aligned_async": "#2563EB",
}
STRATEGIES = tuple(COLORS)
LABELS = {
    "sync_hold": "Sync hold",
    "naive_async": "Naive async",
    "aligned_async": "Aligned async",
}
FIGURE_DPI = 300
BASE_FONT_SIZE = 10
_TIMELINE_EVENTS = {
    "inference_request": (0, "Request", "#7C3AED", "^", 34),
    "chunk_arrived": (1, "Response", "#059669", "o", 30),
    "chunk_rejected": (2, "Reject", "#DC2626", "X", 38),
    "queue_updated": (3, "Queue update", "#2563EB", "s", 28),
    "command_executed": (4, "Execute", "#111827", ".", 18),
}


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on ROS image
        raise RuntimeError("figure generation requires matplotlib") from exc
    plt.rcParams.update(
        {
            "font.size": BASE_FONT_SIZE,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "axes.labelsize": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE,
            "xtick.labelsize": BASE_FONT_SIZE - 1,
            "ytick.labelsize": BASE_FONT_SIZE - 1,
            "legend.fontsize": BASE_FONT_SIZE - 1,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _save(fig, path: Path) -> tuple[Path, Path]:
    """Save one deterministic 300-dpi PNG and one vector PDF."""

    stem = path.with_suffix("")
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        pad_inches=0.05,
        metadata={"Software": "ActionStream M7"},
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.05,
        metadata={
            "Creator": "ActionStream M7",
            "Producer": "ActionStream M7",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    return png_path, pdf_path


def plot_success(analysis: dict[str, Any], output_path: Path) -> None:
    plt = _matplotlib()
    profiles = sorted(analysis["profiles"])
    strategies = STRATEGIES
    width = 0.24
    fig, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for offset, strategy in enumerate(strategies):
        x = [index + (offset - 1) * width for index in range(len(profiles))]
        values = [
            100.0 * analysis["profiles"][profile]["strategies"][strategy]["success_rate"]
            for profile in profiles
        ]
        bars = axis.bar(x, values, width, label=LABELS[strategy], color=COLORS[strategy])
        for bar, profile in zip(bars, profiles, strict=True):
            summary = analysis["profiles"][profile]["strategies"][strategy]
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{summary['successes']}/{summary['trials']}",
                ha="center",
                va="bottom",
                fontsize=BASE_FONT_SIZE - 1,
            )
    axis.set_xticks(
        range(len(profiles)),
        [profile.replace("_", " ").title() for profile in profiles],
    )
    axis.set_ylabel("Task success (%)")
    axis.set_xlabel("Frozen fault profile")
    axis.set_ylim(0, 112)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7)
    axis.legend(frameon=False, ncol=3, loc="upper center", borderaxespad=0.2)
    _save(fig, output_path)
    plt.close(fig)


def plot_efficiency(analysis: dict[str, Any], output_path: Path) -> None:
    plt = _matplotlib()
    profile_id = (
        "profile_b" if "profile_b" in analysis["profiles"] else sorted(analysis["profiles"])[0]
    )
    profile = analysis["profiles"][profile_id]
    strategies = STRATEGIES
    wall = [profile["strategies"][item]["median_wall_clock_seconds"] for item in strategies]
    hold = [profile["strategies"][item]["mean_total_hold_seconds"] for item in strategies]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), constrained_layout=True)
    for panel, axis, values, ylabel in (
        ("a", axes[0], wall, "Median wall-clock completion (s)"),
        ("b", axes[1], hold, "Mean inference hold time (s)"),
    ):
        x = list(range(len(strategies)))
        bars = axis.bar(x, values, color=[COLORS[item] for item in strategies])
        axis.set_xticks(x, [LABELS[item] for item in strategies], rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7)
        axis.text(
            0.0,
            1.01,
            f"({panel}) {profile_id.replace('_', ' ').title()}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.2f}" for value in values],
            padding=3,
            fontsize=BASE_FONT_SIZE - 1,
        )
        upper = max(values, default=0.0)
        axis.set_ylim(0.0, max(0.1, upper * 1.18))
    _save(fig, output_path)
    plt.close(fig)


def plot_timeline(event_log_path: Path | str, output_path: Path) -> None:
    plt = _matplotlib()
    rows = read_jsonl(event_log_path)
    relevant_rows = [row for row in rows if str(row.get("event_type")) in _TIMELINE_EVENTS]
    wall_times = [int(row.get("wall_time_ns", 0)) for row in relevant_rows]
    origin_ns = min(wall_times, default=0)
    fig, axis = plt.subplots(figsize=(10.0, 3.8), constrained_layout=True)
    used_labels: set[str] = set()
    present_event_types: set[str] = set()
    for row in relevant_rows:
        event_type = str(row.get("event_type"))
        present_event_types.add(event_type)
        level, label, color, marker, size = _TIMELINE_EVENTS[event_type]
        if event_type == "command_executed" and bool(row.get("hold")):
            color = "#9CA3AF"
            label = "Hold command"
            marker = "|"
            size = 30
        axis.scatter(
            (int(row.get("wall_time_ns", 0)) - origin_ns) / 1e9,
            level,
            s=size,
            color=color,
            marker=marker,
            label=label if label not in used_labels else None,
            alpha=0.9,
            linewidths=0.6,
        )
        used_labels.add(label)
    event_labels = []
    for event_type, (_, label, _, _, _) in _TIMELINE_EVENTS.items():
        event_labels.append(label if event_type in present_event_types else f"{label} (none)")
    axis.set_yticks(range(len(_TIMELINE_EVENTS)), event_labels)
    axis.set_xlabel("Elapsed wall-clock time (s)")
    axis.set_ylabel("Runtime event")
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.7)
    if used_labels:
        axis.legend(
            frameon=False,
            ncol=min(6, len(used_labels)),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            borderaxespad=0.0,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No request, response, rejection, queue, or execution " "events recorded",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    _save(fig, output_path)
    plt.close(fig)


def generate_figures(
    *,
    analysis_path: Path | str,
    example_event_log: Path | str,
    output_dir: Path | str,
) -> list[str]:
    output = Path(output_dir)
    analysis = read_json(analysis_path)
    stems = [
        output / "m7_success_comparison.png",
        output / "m7_efficiency_comparison.png",
        output / "m7_async_timeline.png",
    ]
    plot_success(analysis, stems[0])
    plot_efficiency(analysis, stems[1])
    plot_timeline(example_event_log, stems[2])
    return [str(stem.with_suffix(suffix)) for stem in stems for suffix in (".png", ".pdf")]
