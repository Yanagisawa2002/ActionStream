"""Publication-quality analytical figures derived only from validated M8 analysis."""

from __future__ import annotations

from html import escape
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from .m8_protocol import (
    M8_MILESTONE,
    M8_SCHEMA_VERSION,
    NATIVE_ISAAC_EVIDENCE_CLASS,
    sha256_file,
)
from .schema import read_json, read_jsonl, write_json_atomic


METHODS = ("sync_hold", "naive_async", "aligned_async")
# Okabe-Ito-derived, print-safe method colors with deliberately different luminance.
COLORS = ("#6B7280", "#D55E00", "#0072B2")


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on deployment image
        raise RuntimeError("M8 figure generation requires matplotlib") from exc
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _save_figure(fig, stem: Path) -> list[dict[str, Any]]:
    records = []
    for extension, dpi in (("pdf", None), ("png", 300)):
        destination = stem.with_suffix(f".{extension}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.{extension}")
        try:
            fig.savefig(temporary, format=extension, dpi=dpi)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        records.append(
            {
                "format": extension,
                "path": destination.name,
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                **({"dpi": 300} if extension == "png" else {"vector": True}),
            }
        )
    return records


def _bar_figure(
    values: Mapping[str, float],
    *,
    ylabel: str,
    axis_maximum: float | None,
    value_text: Callable[[str, float], str],
):
    plt = _matplotlib()
    fig, axis = plt.subplots(figsize=(5.0, 3.2))
    heights = [float(values[method]) for method in METHODS]
    bars = axis.bar(
        range(len(METHODS)),
        heights,
        color=COLORS,
        edgecolor="#262626",
        linewidth=0.6,
        width=0.64,
    )
    axis.set_xticks(range(len(METHODS)), labels=[method.replace("_", "\n") for method in METHODS])
    axis.set_ylabel(ylabel)
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="#D1D5DB", linewidth=0.55)
    if axis_maximum is not None:
        axis.set_ylim(0.0, axis_maximum)
    else:
        maximum = max(heights, default=0.0)
        axis.set_ylim(0.0, max(1.0, maximum * 1.22))
    offset = max(axis.get_ylim()[1] * 0.025, 0.02)
    for method, value, bar in zip(METHODS, heights, bars, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            value_text(method, value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    return fig


def generate_analysis_figures(
    analysis_path: Path | str,
    output_directory: Path | str,
) -> dict[str, Any]:
    analysis = read_json(analysis_path)
    if analysis.get("milestone") != M8_MILESTONE:
        raise ValueError("figures require an M8-G0 analysis artifact")
    if analysis.get("evidence_class") != NATIVE_ISAAC_EVIDENCE_CLASS:
        raise ValueError("M8 headline figures require validated native-Isaac evidence")
    if int(analysis.get("episode_count", 0)) <= 0:
        raise ValueError("native M8 analysis contains no episodes")
    profile = analysis.get("profiles", {}).get("profile_1_fixed", {})
    strategies = profile.get("strategies", {})
    if set(strategies) != set(METHODS):
        raise ValueError("Profile 1 analysis must contain all three methods")
    if any(int(strategies[method].get("trials", 0)) <= 0 for method in METHODS):
        raise ValueError("Profile 1 figure denominators must be positive")

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    figure_specs = (
        {
            "id": "profile_1_success",
            "caption": (
                "Correct-destination success under the frozen fixed-latency Profile 1. "
                "Labels report raw successes and exact paired denominators."
            ),
            "figure": lambda: _bar_figure(
                {method: float(strategies[method]["success_rate"]) for method in METHODS},
                ylabel="Correct-destination success rate",
                axis_maximum=1.0,
                value_text=lambda method, value: (
                    f"{int(strategies[method]['successes'])}/{int(strategies[method]['trials'])}\n({value:.0%})"
                ),
            ),
        },
        {
            "id": "profile_1_recovery_latency",
            "caption": (
                "Median penalized control-step latency from the destination switch to the first "
                "executed post-switch-generation command under Profile 1; lower is better."
            ),
            "figure": lambda: _bar_figure(
                {
                    method: float(strategies[method]["median_penalized_recovery_latency_steps"])
                    for method in METHODS
                },
                ylabel="Penalized recovery latency (steps)",
                axis_maximum=None,
                value_text=lambda _method, value: f"{value:.1f}",
            ),
        },
        {
            "id": "profile_1_obsolete_commands",
            "caption": (
                "Mean post-switch control steps whose Cartesian commands are geometrically closer "
                "to the obsolete destination than the active destination under Profile 1; lower is better."
            ),
            "figure": lambda: _bar_figure(
                {
                    method: float(strategies[method]["mean_obsolete_destination_command_steps"])
                    for method in METHODS
                },
                ylabel="Obsolete-destination command steps",
                axis_maximum=None,
                value_text=lambda _method, value: f"{value:.1f}",
            ),
        },
    )
    generated = []
    latex = []
    plt = _matplotlib()
    for index, spec in enumerate(figure_specs, start=1):
        fig = spec["figure"]()
        try:
            files = _save_figure(fig, destination / spec["id"])
        finally:
            plt.close(fig)
        generated.append({"figure_id": spec["id"], "caption": spec["caption"], "files": files})
        latex.extend(
            [
                f"% Figure {index}: {spec['id']}",
                "\\begin{figure}[t]",
                "  \\centering",
                f"  \\includegraphics[width=0.48\\textwidth]{{{spec['id']}.pdf}}",
                f"  \\caption{{{spec['caption']}}}",
                f"  \\label{{fig:m8-{spec['id'].replace('_', '-')}}}",
                "\\end{figure}",
                "",
            ]
        )
    latex_path = destination / "latex_includes.tex"
    latex_path.write_text("\n".join(latex), encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": M8_SCHEMA_VERSION,
        "milestone": M8_MILESTONE,
        "source_analysis": Path(
            os.path.relpath(Path(analysis_path).resolve(), destination.resolve())
        ).as_posix(),
        "source_analysis_sha256": sha256_file(analysis_path),
        "evidence_class": NATIVE_ISAAC_EVIDENCE_CLASS,
        "classification": analysis.get("classification"),
        "style": "publication",
        "png_dpi": 300,
        "figures": generated,
        "latex_includes": {
            "path": latex_path.name,
            "sha256": sha256_file(latex_path),
        },
    }
    write_json_atomic(destination / "figure_manifest.json", manifest)
    return manifest


def generate_episode_timeline(event_log_path: Path | str, output_path: Path | str) -> dict[str, Any]:
    """Render the six required causal roles on recorded event order.

    Source-observation steps are deliberately not used as coordinates: a delayed
    old response retains its old source step, so plotting that value would move
    the response backward in time and conceal the out-of-order arrival.
    """

    rows = read_jsonl(event_log_path)
    indices = [row.get("event_index") for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("timeline requires contiguous recorded event_index chronology")
    switches = [row for row in rows if row.get("event_type") == "destination_switched"]
    if len(switches) != 1:
        raise ValueError("timeline requires exactly one destination switch")
    switch = switches[0]
    switch_index = int(switch["event_index"])
    generation_after = int(switch.get("generation_after", switch.get("generation_id", 0)))

    old_requests = [
        row
        for row in rows
        if row.get("event_type") == "inference_request"
        and int(row["event_index"]) < switch_index
        and int(row.get("generation_id", -1)) < generation_after
    ]
    new_requests = [
        row
        for row in rows
        if row.get("event_type") == "inference_request"
        and int(row["event_index"]) > switch_index
        and int(row.get("generation_id", -1)) == generation_after
    ]
    if not old_requests or not new_requests:
        raise ValueError("timeline lacks old/new requests around the destination switch")
    old_request = old_requests[-1]
    new_request = new_requests[0]
    old_id = int(old_request.get("request_id", -1))
    new_id = int(new_request.get("request_id", -1))

    queue_replacements = [
        row
        for row in rows
        if row.get("event_type") == "queue_updated"
        and row.get("reason") == "aligned_atomic_rebuild"
        and int(row.get("request_id", -1)) == new_id
        and int(row["event_index"]) > int(new_request["event_index"])
    ]
    out_of_order_responses = [
        row
        for row in rows
        if row.get("event_type") == "chunk_arrived"
        and row.get("out_of_order") is True
        and int(row.get("request_id", -1)) == old_id
    ]
    obsolete_rejections = [
        row
        for row in rows
        if row.get("event_type") == "chunk_rejected"
        and row.get("reason") == "stale_generation"
        and int(row.get("request_id", -1)) == old_id
    ]
    if not queue_replacements or not out_of_order_responses or not obsolete_rejections:
        raise ValueError(
            "timeline lacks aligned queue replacement, out-of-order old response, or stale rejection"
        )
    queue_replacement = queue_replacements[0]
    out_of_order_response = out_of_order_responses[0]
    obsolete_rejection = obsolete_rejections[0]
    selected = (
        ("old_request", "old request", old_request),
        ("destination_switch", "destination switch", switch),
        ("new_request", "new request", new_request),
        ("queue_replacement", "atomic queue replacement", queue_replacement),
        ("out_of_order_response", "out-of-order old response", out_of_order_response),
        ("obsolete_action_rejection", "obsolete action rejection", obsolete_rejection),
    )
    selected_indices = [int(row["event_index"]) for _role, _label, row in selected]
    if selected_indices != sorted(selected_indices):
        raise ValueError("timeline causal roles do not occur in the required recorded order")
    low, high = min(selected_indices), max(selected_indices)
    span = max(1, high - low)
    plot_left, plot_width = 100, 1040
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="500" viewBox="0 0 1200 500">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<line x1="100" y1="430" x2="1140" y2="430" stroke="#111827" stroke-width="2"/>',
    ]
    causal_roles: dict[str, Any] = {}
    for lane, (role, label, row) in enumerate(selected):
        event_index = int(row["event_index"])
        x = plot_left + plot_width * (event_index - low) / span
        y = 385 - 47 * lane
        color = "#D55E00" if role in {"destination_switch", "obsolete_action_rejection"} else "#0072B2"
        request_suffix = f" r{int(row['request_id'])}" if row.get("request_id") else ""
        rendered_label = label + request_suffix
        elements.extend(
            [
                f'<line x1="{x:.2f}" y1="430" x2="{x:.2f}" y2="{y:.2f}" stroke="#d1d5db"/>',
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"/>',
                f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" font-family="serif" font-size="11" transform="rotate(-35 {x + 7:.2f} {y - 7:.2f})">{escape(rendered_label)}</text>',
            ]
        )
        causal_roles[role] = {
            "event_index": event_index,
            "event_type": row["event_type"],
            "request_id": row.get("request_id"),
            "generation_id": row.get("generation_id"),
            "reason": row.get("reason"),
        }
    elements.extend(
        [
            f'<text x="100" y="460" font-family="serif" font-size="12">event {low}</text>',
            f'<text x="1140" y="460" text-anchor="end" font-family="serif" font-size="12">event {high}</text>',
            '<text x="620" y="485" text-anchor="middle" font-family="serif" font-size="12">recorded event order</text>',
            '</svg>',
        ]
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text("\n".join(elements) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema_version": M8_SCHEMA_VERSION,
        "milestone": M8_MILESTONE,
        "source_event_log": Path(
            os.path.relpath(Path(event_log_path).resolve(), destination.parent.resolve())
        ).as_posix(),
        "source_event_log_sha256": sha256_file(event_log_path),
        "path": destination.name,
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "coordinate": "recorded_event_index",
        "event_count": len(selected),
        "causal_roles": causal_roles,
    }
