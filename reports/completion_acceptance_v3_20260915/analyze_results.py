from collections import Counter
import json
from pathlib import Path
import statistics
import tarfile

root = Path(__file__).resolve().parent


def read(path):
    file = root / path
    if file.exists():
        return json.loads(file.read_text())
    with tarfile.open(file.parent / "raw_records.tar.gz") as archive:
        return json.loads(archive.extractfile(file.name).read())


live = read("live/summary.json")["episodes"]
paired = read("paired-evaluation/summary.json")["results"]
before = {e["seed"]: e for e in paired["baseline"]["episodes"]}
both = [
    (before[e["seed"]]["confirmation_delay_s"], e["confirmation_delay_s"])
    for e in live
    if e["confirmation_delay_s"] is not None
    and before[e["seed"]]["confirmation_delay_s"] is not None
]
misses = []
for e in paired["baseline"]["episodes"]:
    if not e["missed_completed_event"]:
        continue
    facts = read("collection/" + e["episode"] + "_truth.json")
    rows = read("paired-evaluation/" + e["episode"] + "_baseline_predictions.json")
    positive = [r for r in rows if facts[r["control"]]["strict_complete"]]
    longest, run = 0, 0
    for r in rows:
        run = run + 1 if r["decision"] == "complete" else 0
        longest = max(longest, run)
    misses.append(
        dict(
            seed=e["seed"],
            positive_decisions=dict(Counter(r["decision"] for r in positive)),
            longest_raw_positive_run=longest,
        )
    )
failures = []
for e in live:
    if e["first_truth_control"] is not None:
        continue
    fs = read("live/" + e["episode"] + "_truth.json")
    failures.append(
        dict(
            seed=e["seed"],
            final_facts=fs[-1],
            controls_inside=sum(f["inside"] for f in fs),
            controls_finger_contact=sum(f["finger_contact"] for f in fs),
            controls_basket_contact=sum(f["basket_contact"] for f in fs),
        )
    )
report = dict(
    matched_safe_stop_pairs=len(both),
    mean_delay_reduction_s=statistics.mean(a - b for a, b in both),
    median_delay_reduction_s=statistics.median(a - b for a, b in both),
    improved=sum(b < a for a, b in both),
    unchanged=sum(a == b for a, b in both),
    worsened=sum(b > a for a, b in both),
    baseline_event_misses=misses,
    policy_failures=failures,
    live_delay_min_s=min(
        e["confirmation_delay_s"] for e in live if e["confirmation_delay_s"] is not None
    ),
    live_delay_mean_s=statistics.mean(
        e["confirmation_delay_s"] for e in live if e["confirmation_delay_s"] is not None
    ),
    live_delay_sample_std_s=statistics.stdev(
        e["confirmation_delay_s"] for e in live if e["confirmation_delay_s"] is not None
    ),
)
(root / "diagnosis.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
)
print(json.dumps(report, indent=2))
