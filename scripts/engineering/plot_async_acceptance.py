"""Render acceptance metrics and retained recovery RGB from an extracted bundle."""

import argparse
import json
import pathlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument(
    "--skip-frames",
    action="store_true",
    help="Render metrics from compact records without NPZ files",
)
args = parser.parse_args()
ROOT, OUT = args.run, args.output
OUT.mkdir(parents=True, exist_ok=True)


def read(p):
    return json.loads(p.read_text())


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(11, 5.1), layout="constrained")
r = read(ROOT / "normal/result.json")
s = [e["score"] for e in r["episodes"]]
x = np.arange(10)
fractions = [100 * e["timing"]["deadline_misses"] / e["timing"]["controls"] for e in s]
axes[0].bar(x, fractions, color="#b34c3c")
axes[0].axhline(1, color="#20334d", ls="--", label="Frozen maximum: 1% per episode")
axes[0].set(
    xlabel="Normal episode (0–9)",
    ylabel="Work over 50 ms (%)",
    title="All normal cases miss the timing gate",
    xticks=x,
)
axes[0].legend(fontsize=8)
rec = read(ROOT / "recovery/result.json")
sc = [e["score"] for e in rec["episodes"]]
for i, e in enumerate(sc):
    end = e["timing"]["controls"] / 20
    axes[1].broken_barh([(0, 15)], (i - 0.35, 0.7), facecolors="#b7c2cf")
    axes[1].broken_barh(
        [(15, 3), (18, 1)], (i - 0.35, 0.7), facecolors=["#eac980", "#ad9cc6"]
    )
    axes[1].broken_barh(
        [(19, max(0, end - 19))],
        (i - 0.35, 0.7),
        facecolors="#2d8b70" if e["recovery"]["recovered_from_failure"] else "#b34c3c",
    )
axes[1].set(
    xlabel="Simulation time (s)",
    ylabel="Recovery episode (0–9)",
    title="Physical fault → settle → retry",
    yticks=x,
)
axes[1].invert_yaxis()
axes[1].legend(
    handles=[
        Patch(color=color, label=label)
        for color, label in [
            ("#b7c2cf", "Gripper forced open"),
            ("#eac980", "Settling"),
            ("#ad9cc6", "Recovery opening"),
            ("#2d8b70", "Recovered"),
            ("#b34c3c", "Unrecovered"),
        ]
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    fontsize=8,
)
fig.suptitle("ActionStream frozen acceptance | RTX 5090 | 2026-09-15", fontsize=13)
fig.savefig(OUT / "acceptance.png", dpi=180)
plt.close(fig)
if args.skip_frames:
    raise SystemExit(0)
failures = [i for i, e in enumerate(sc) if not e["recovery"]["recovered_from_failure"]]
selected = [0] + failures[:1]
fig, axes = plt.subplots(
    len(selected) * 2,
    4,
    figsize=(11, len(selected) * 4),
    layout="constrained",
    squeeze=False,
)
for row, i in enumerate(selected):
    d = ROOT / "recovery" / f"episode-{i:02d}"
    score = read(d / "independent_score.json")
    with np.load(d / "trajectory.npz", allow_pickle=False) as data:
        rgb = data["rgb"]
    controls = [0, 300, 380, len(rgb) - 1]
    for camera in range(2):
        for col, c in enumerate(controls):
            ax = axes[row * 2 + camera, col]
            ax.imshow(rgb[c, camera])
            ax.set_axis_off()
            if camera == 0:
                ax.set_title(f"Episode {i:02d} | control {c}")
        axes[row * 2 + camera, 0].set_ylabel(
            "Agent view" if camera == 0 else "Wrist view"
        )
fig.suptitle(
    "Retained RGB: first recovery case and first unrecovered case\nControls 0 / 300 (fault ends) / 380 (retry begins) / final",
    fontsize=12,
)
fig.savefig(OUT / "recovery-frames.png", dpi=160)
plt.close(fig)
print("plots saved")
