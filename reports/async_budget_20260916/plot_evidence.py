"""Plot retained acceptance journals; never invokes the controller or models."""

import argparse
import json
from pathlib import Path
from record_io import open_archive

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(archive, output):
    with open_archive(archive) as tar:

        def read(name):
            return json.load(tar.extractfile("budget-acceptance-v2/" + name))

        normal = read("normal/result.json")
        recovery = read("recovery/result.json")
        config = read("protocol.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    fig.suptitle("Frozen 50 ms work-budget acceptance | 2026-09-16", fontsize=15)
    ax = axes[0]
    scores = [e["score"] for e in normal["episodes"]]
    rates = [
        100 * s["timing"]["deadline_misses"] / s["timing"]["controls"] for s in scores
    ]
    ax.bar(range(10), rates, color="#127b80", width=0.7)
    for i, s in enumerate(scores):
        t = s["timing"]
        ax.text(
            i,
            rates[i] + 0.035,
            f"{t['deadline_misses']}/{t['controls']}",
            ha="center",
            fontsize=8,
        )
    gate = config["gates"]["normal_deadline_miss_fraction_max"] * 100
    ax.axhline(gate, color="#ac3434", linestyle="--", label="Every-case gate: 1%")
    ax.set(
        ylim=(0, 1.22),
        xticks=range(10),
        xlabel="Normal case (seed suffix 2300–2309)",
        ylabel="Controls exceeding 50 ms (%)",
        title=f"Normal timing: {normal['status']}",
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax = axes[1]
    for i, e in enumerate(recovery["episodes"]):
        s = e["score"]
        with open_archive(archive) as tar:
            base = f"budget-acceptance-v2/recovery/episode-{i:02d}/"
            out = json.load(tar.extractfile(base + "outcome.json"))
            events = [
                json.loads(line) for line in tar.extractfile(base + "runtime.jsonl")
            ]
        retry = next(
            row["control"] for row in events if row["event"] == "recovery_started"
        )
        end = out["first_claim_control"] or out["control_steps"]
        color = "#127b80" if s["recovery"]["recovered_from_failure"] else "#ac3434"
        ax.plot([0, end / 20], [i, i], color="#b8c3cb", lw=3)
        ax.plot([0, 300 / 20], [i, i], color="#db9957", lw=5)
        ax.scatter(retry / 20, i, marker="|", s=130, color="#344454")
        ax.scatter(
            end / 20, i, color=color, marker="o" if s["safely_completed"] else "x"
        )
    ax.axvline(300 / 20, color="#b67732", lw=0.8, linestyle=":")
    ax.set(
        yticks=range(10),
        ylabel="Recovery case (seed suffix 2400–2409)",
        xlabel="Simulated time (seconds)",
        title=f"Physical recovery: {recovery['status']}",
    )
    ax.set_ylim(10.1, -0.5)
    ax.text(
        0.02,
        0.02,
        "Orange: forced-open fault | Tick: retry | Dot: safe stop",
        transform=ax.transAxes,
        fontsize=8,
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path(__file__).with_name("raw_records.tar.gz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("acceptance.png")
    )
    args = parser.parse_args()
    plot(args.archive, args.output)
