#!/usr/bin/env python3
"""Render a publication-style M8 latency-success operating-point figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STYLE = {
    "sync_hold": {"label": "Sync hold", "color": "#0072B2", "marker": "o", "linestyle": "none"},
    "naive_async": {"label": "Naive async", "color": "#D55E00", "marker": "X", "linestyle": "none"},
    "aligned_async": {
        "label": "ActionStream aligned",
        "color": "#009E73",
        "marker": "s",
        "linestyle": "none",
    },
}
PROFILE_SHORT = {
    "profile_0_sanity": "P0",
    "profile_1_fixed": "P1",
    "profile_2_faults": "P2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def render(summary_path: Path, output_directory: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    if summary.get("artifact_kind") != "posthoc_frozen_holdout_presentation":
        raise ValueError("input is not an M8 post-hoc frozen-holdout summary")
    if int(summary.get("episode_count", 0)) != 420:
        raise ValueError("latency-success figure requires the complete 420-episode holdout")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.usetex": False,
            "mathtext.fontset": "stix",
        }
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    data_rows = []
    for group in summary["groups"]:
        p50 = float(group["median_episode_p50_inference_latency_ms"])
        p95 = float(group["median_episode_p95_inference_latency_ms"])
        ci_low, ci_high = (float(value) for value in group["success_wilson_95_ci"])
        data_rows.append(
            {
                "profile_id": group["profile_id"],
                "profile_short": PROFILE_SHORT[group["profile_id"]],
                "strategy": group["strategy"],
                "x_latency_p50_ms": p50,
                "x_latency_p95_ms": p95,
                "success_rate": float(group["success_rate"]),
                "success_wilson_95_ci": [ci_low, ci_high],
                "successes": int(group["successes"]),
                "trials": int(group["trials"]),
            }
        )

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    annotation_offsets = {
        ("sync_hold", "profile_0_sanity"): (5, 7),
        ("sync_hold", "profile_1_fixed"): (-44, 6),
        ("sync_hold", "profile_2_faults"): (8, 9),
        ("naive_async", "profile_1_fixed"): (-34, 15),
        ("naive_async", "profile_2_faults"): (12, 15),
        ("aligned_async", "profile_1_fixed"): (-23, -22),
        ("aligned_async", "profile_2_faults"): (8, 9),
    }
    for strategy in ("sync_hold", "naive_async", "aligned_async"):
        rows = sorted(
            (row for row in data_rows if row["strategy"] == strategy),
            key=lambda row: row["x_latency_p50_ms"],
        )
        style = STYLE[strategy]
        x = [row["x_latency_p50_ms"] for row in rows]
        y = [row["success_rate"] for row in rows]
        xerr = [[0.0 for _ in rows], [max(0.0, row["x_latency_p95_ms"] - row["x_latency_p50_ms"]) for row in rows]]
        yerr = [
            [
                max(0.0, row["success_rate"] - row["success_wilson_95_ci"][0])
                for row in rows
            ],
            [
                max(0.0, row["success_wilson_95_ci"][1] - row["success_rate"])
                for row in rows
            ],
        ]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.4,
            markersize=6,
            capsize=2.5,
            elinewidth=1.0,
            zorder=3,
        )
        for row in rows:
            dx, dy = annotation_offsets[(strategy, row["profile_id"])]
            ax.annotate(
                row["profile_short"],
                (row["x_latency_p50_ms"], row["success_rate"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                color=style["color"],
                arrowprops=(
                    {
                        "arrowstyle": "-",
                        "color": style["color"],
                        "linewidth": 0.55,
                        "shrinkA": 1,
                        "shrinkB": 2,
                    }
                    if strategy == "naive_async"
                    or (strategy == "sync_hold" and row["profile_id"] == "profile_1_fixed")
                    else None
                ),
            )

    ax.set_xscale("log")
    ax.set_xlim(10, 1900)
    ax.set_ylim(-0.075, 1.075)
    ax.set_xlabel("Median episode inference-latency p50 to p95 (ms, log scale)")
    ax.set_ylabel("Task success rate")
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.02))
    fig.tight_layout()

    stem = output_directory / "latency_success_operating_points"
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(
        pdf_path,
        format="pdf",
        metadata={"Creator": "ActionStream", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)

    data_path = stem.with_suffix(".data.json")
    data_payload = {
        "schema_version": 1,
        "artifact_kind": "m8_latency_success_operating_points",
        "source_summary": summary_path.as_posix(),
        "source_summary_sha256": sha256_file(summary_path),
        "x_definition": "median per-episode p50 inference latency; right error reaches median per-episode p95",
        "y_definition": "task success rate with Wilson 95% confidence interval",
        "causal_boundary": "P0/P1/P2 differ in both latency and fault regime; this is not a continuous causal sweep",
        "rows": data_rows,
    }
    data_path.write_text(
        json.dumps(data_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    caption = (
        "Frozen M8 holdout latency-success operating points. Unconnected markers avoid implying a "
        "causal latency sweep. Horizontal bars span the median per-episode p50-to-p95 measured "
        "inference latency; vertical bars are Wilson 95% confidence "
        "intervals over 60 unseen seeds. P0 is sanity, P1 is fixed delay, and P2 adds jitter and "
        "response faults. P0/P1/P2 change both latency and fault regime."
    )
    (output_directory / "latency_success_operating_points.caption.md").write_text(
        caption + "\n", encoding="utf-8", newline="\n"
    )
    (output_directory / "latex_includes.tex").write_text(
        "\n".join(
            [
                "% M8 latency-success operating points",
                "\\begin{figure*}[t]",
                "  \\centering",
                "  \\includegraphics[width=0.95\\textwidth]{latency_success_operating_points.pdf}",
                f"  \\caption{{{caption}}}",
                "  \\label{fig:m8-latency-success}",
                "\\end{figure*}",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    manifest = {
        "schema_version": 1,
        "artifact_kind": "posthoc_publication_figure",
        "source_summary": summary_path.as_posix(),
        "source_summary_sha256": sha256_file(summary_path),
        "style": "publication",
        "png_dpi": 300,
        "files": [
            {"path": pdf_path.name, "sha256": sha256_file(pdf_path)},
            {"path": png_path.name, "sha256": sha256_file(png_path)},
            {"path": data_path.name, "sha256": sha256_file(data_path)},
        ],
        "caption": caption,
    }
    (output_directory / "latency_success_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = render(args.summary, args.output_dir)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
