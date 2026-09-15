"""Summarize nested timing spans without adding overlapping CPU/GPU durations."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return dict(
        n=len(values),
        mean_ms=statistics.mean(values) * 1000,
        p95_ms=values[int(0.95 * (len(values) - 1))] * 1000,
        max_ms=max(values) * 1000,
    )


def summarize(root):
    totals = defaultdict(list)
    misses = defaultdict(list)
    outcomes = []
    for path in sorted(root.glob("*/profile.json")):
        data = json.loads(path.read_text())
        events = [
            json.loads(s)
            for s in (path.parent / "runtime.jsonl").read_text().splitlines()
        ]
        dispatches = [e for e in events if e["event"] == "dispatch"]
        score = json.loads((path.parent / "independent_score.json").read_text())
        outcomes.append(dict(seed=int(path.parent.name), **score))
        for dispatch in dispatches:
            a = dispatch["started_monotonic"]
            b = a + dispatch["work_wall_s"]
            is_miss = dispatch["work_wall_s"] > 0.05
            totals["control_work"].append(b - a)
            if is_miss:
                misses["control_work"].append(b - a)
            for span in data["cpu"]:
                if a <= span["start"] and span["end"] <= b:
                    value = span["end"] - span["start"]
                    totals[span["label"]].append(value)
                    if is_miss:
                        misses[span["label"]].append(value)
            for span in data["gpu"]:
                if a <= span["start"] < b:
                    totals["rgb_cuda_stream"].append(span["stream_elapsed_s"])
                    if is_miss:
                        misses["rgb_cuda_stream"].append(span["stream_elapsed_s"])
    return dict(
        episodes=outcomes,
        spans={k: stats(v) for k, v in totals.items()},
        missed_control_spans={k: stats(v) for k, v in misses.items()},
        interpretation="Nested spans overlap; GPU stream elapsed includes contention and is not exclusive GPU time.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    (args.root / "profile-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
