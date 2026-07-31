"""Shared publication style for ActionStream release figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "release" / "v1.0.0" / "figures"

MODE_ORDER = ("sync_hold", "async_naive", "async_aligned")
MODE_LABELS = {
    "sync_hold": "Sync hold",
    "async_naive": "Naive async",
    "async_aligned": "Aligned async",
}

# Okabe-Ito-derived colors plus a neutral gray.
MODE_COLORS = {
    "sync_hold": "#6B7280",
    "async_naive": "#E69F00",
    "async_aligned": "#0072B2",
}
ACCENT = "#009E73"
TEXT = "#202124"
GRID = "#D1D5DB"


def configure_style() -> None:
    """Apply one consistent, print-friendly style."""

    matplotlib.rcParams.update(
        {
            "font.size": 9.5,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Add a compact panel label without introducing an in-figure title."""

    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    """Save vector PDF and 300 dpi PNG variants."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    print(f"Saved {pdf_path.relative_to(ROOT)}")
    print(f"Saved {png_path.relative_to(ROOT)}")
    return pdf_path, png_path
