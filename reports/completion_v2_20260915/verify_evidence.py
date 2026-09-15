"""Replay compact evidence without Torch, the simulator, or model code.

Run from any directory. Checks the retained records; does not re-render images
or re-run the model weights retained on the GPU server.
"""

import hashlib
import json
import math
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(rows, field="decision"):
    cells = {
        (label, decision): 0
        for label in (0, 1, 2)
        for decision in ("complete", "incomplete", "unknown")
    }
    for row in rows:
        cells[row["label"], row[field]] += 1
    totals = {
        label: sum(n for (lab, _), n in cells.items() if lab == label)
        for label in (0, 1, 2)
    }
    return dict(
        samples=len(rows),
        positive=totals[1],
        negative=totals[0],
        unobservable=totals[2],
        false_complete=cells[0, "complete"],
        true_complete=cells[1, "complete"],
        missed_complete=cells[1, "incomplete"] + cells[1, "unknown"],
        unknown=sum(cells[label, "unknown"] for label in (0, 1, 2)),
        hidden_confident=totals[2] - cells[2, "unknown"],
    )


def physical_ok(f, truth):
    return (
        f["inside"]
        and not f["finger_contact"]
        and f["basket_contact"]
        and math.isfinite(f["linear_speed"])
        and math.isfinite(f["angular_speed"])
        and 0 <= f["linear_speed"] <= truth["linear_speed_max"]
        and 0 <= f["angular_speed"] <= truth["angular_speed_max"]
    )


def main():
    protocol = read(ROOT / "configs/completion_v2.json")
    frozen = read(HERE / "training/frozen.json")
    summary = read(HERE / "evaluation/summary.json")
    development = read(HERE / "development/manifest.json")
    holdout = read(HERE / "fresh_holdout/manifest.json")
    execution = read(HERE / "execution_receipt.json")
    for name, recorded in execution["training_source_sha256"].items():
        assert digest(ROOT / name) == recorded
    for item in (frozen, summary, development, holdout):
        assert item["protocol_sha256"] == digest(ROOT / "configs/completion_v2.json")
    code_hash = digest(ROOT / "scripts/engineering/completion_v2.py")
    assert frozen["trainer_sha256"] == summary["evaluator_sha256"] == code_hash
    assert development["builder_sha256"] == holdout["builder_sha256"] == code_hash
    assert frozen["development_manifest_sha256"] == digest(
        HERE / "development/manifest.json"
    )
    assert holdout["freeze_sha256"] == digest(HERE / "training/frozen.json")
    assert summary["holdout_manifest_sha256"] == digest(
        HERE / "fresh_holdout/manifest.json"
    )
    assert summary["predictions_sha256"] == digest(
        HERE / "evaluation/predictions.jsonl"
    )
    assert summary["checkpoint_sha256"] == frozen["checkpoint_sha256"]
    assert holdout["generated_unix"] > frozen["frozen_unix"]
    assert frozen["status"] == "FROZEN" and not frozen["test_seen"]
    assert (
        not holdout["model_predictions_used"] and not summary["test_used_for_selection"]
    )
    train = set(development["splits"]["train"])
    validation = set(development["splits"]["validation"])
    old_test = set(development["excluded_old_test"])
    assert len(train) == 30 and len(validation) == len(old_test) == 10
    assert not (train & validation or train & old_test or validation & old_test)
    entries = holdout["entries"]
    assert len(entries) == 20
    assert [e["metadata"]["seed"] for e in entries] == protocol["fresh_holdout"][
        "seeds"
    ]
    assert len({e["metadata"]["initial_layout_sha256"] for e in entries}) == 20
    learning = [
        json.loads(line)
        for line in (HERE / "training/metrics.jsonl").read_text().splitlines()
    ]
    assert len(learning) == 40
    selected = min(
        learning,
        key=lambda e: (
            e["validation"]["false_complete"],
            e["validation"]["missed_complete"],
            e["validation_cross_entropy"],
        ),
    )
    assert (
        selected == frozen["validation"]
        and selected["epoch"] == frozen["chosen_epoch"] == 25
    )

    episodes = {}
    controls = 0
    sample_lookup = {}
    for folder, manifest in (("development", development), ("fresh_holdout", holdout)):
        with tarfile.open(HERE / folder / "private_truth.tar.gz") as archive:
            for entry in manifest["entries"]:
                episode = entry["episode"]
                raw = archive.extractfile(episode + "_truth.json").read()
                assert hashlib.sha256(raw).hexdigest() == entry["truth_sha256"]
                trace = json.loads(raw)
                fs = trace["per_control"]
                controls += len(fs)
                width = protocol["truth"]["window_steps"] + 1
                for i, f in enumerate(fs):
                    expected = i + 1 >= width and all(
                        physical_ok(x, protocol["truth"])
                        for x in fs[i - width + 1 : i + 1]
                    )
                    assert bool(expected) == f["strict_complete"], (episode, i)
                assert entry["label_changes_from_containment"] == sum(
                    f["inside"] and not f["strict_complete"] for f in fs
                )
                samples = trace["samples"]
                assert len(samples) == entry["samples"]
                assert {
                    str(k): sum(r["label"] == k for r in samples) for k in (0, 1, 2)
                } == entry["counts"]
                for row in samples:
                    if row["category"] in {"stable", "held", "unsettled", "incomplete"}:
                        assert row["label"] == int(
                            fs[row["control_step"]]["strict_complete"]
                        )
                    else:
                        assert row["label"] == (
                            2 if row["category"].endswith("hidden") else 0
                        )
                if folder == "fresh_holdout":
                    assert len(fs) == 361
                    assert trace["metadata"]["init_states"] is False
                    raw_actions = archive.extractfile(episode + "_actions.json").read()
                    assert (
                        hashlib.sha256(raw_actions).hexdigest()
                        == entry["metadata"]["action_sha256"]
                    )
                    assert len(json.loads(raw_actions)) == 300
                    episodes[episode] = fs
                    for row in samples:
                        key = (episode, row["control_step"], row["category"])
                        assert key not in sample_lookup
                        sample_lookup[key] = row

    rows = [
        json.loads(line)
        for line in (HERE / "evaluation/predictions.jsonl").read_text().splitlines()
    ]
    seen = set()
    for row in rows:
        key = (row["episode"], row["control_step"], row["category"])
        assert key not in seen
        seen.add(key)
        for k, v in sample_lookup[key].items():
            assert row[k] == v
        probs = row["probabilities"]
        assert len(probs) == 3 and all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
        assert abs(sum(probs) - 1) < 1e-5
        expected = (
            "complete"
            if probs[1] >= protocol["thresholds"]["complete"]
            else "incomplete"
            if probs[0] >= protocol["thresholds"]["incomplete"]
            else "unknown"
        )
        assert row["decision"] == expected
        old = row["baseline_probability"]
        assert row["baseline_decision"] == (
            "complete" if old >= 0.9 else "incomplete" if old <= 0.1 else "unknown"
        )
    assert seen == sample_lookup.keys()
    assert metrics(rows) == summary["candidate"]
    assert metrics(rows, "baseline_decision") == summary["baseline"]
    for category, result in summary["categories"].items():
        subset = [r for r in rows if r["category"] == category]
        assert metrics(subset) == result["candidate"]
        assert metrics(subset, "baseline_decision") == result["baseline"]
    for episode, result in summary["episodes"].items():
        assert metrics([r for r in rows if r["episode"] == episode]) == result
    natural = [
        r
        for r in rows
        if r["category"] in {"stable", "held", "unsettled", "incomplete"}
    ]
    assert metrics(natural) == summary["natural_trajectory"]
    m = metrics(rows)
    recall = m["true_complete"] / m["positive"]
    assert recall == summary["complete_recall"]
    acceptance = (
        m["false_complete"] == 0 and recall >= 0.8 and m["hidden_confident"] == 0
    )
    assert not acceptance and summary["status"] == "NO_GO"

    errors = []
    for row in rows:
        if row["label"] != 0 or row["decision"] != "complete":
            continue
        step = row["control_step"]
        fs = episodes[row["episode"]]
        next_stable = next(i for i in range(step, len(fs)) if fs[i]["strict_complete"])
        errors.append(
            dict(
                episode=row["episode"],
                control_step=step,
                completion_probability=row["probabilities"][1],
                currently_released_supported_slow=bool(
                    physical_ok(fs[step], protocol["truth"])
                ),
                window_max_linear_speed=max(
                    f["linear_speed"] for f in fs[step - 10 : step + 1]
                ),
                window_max_angular_speed=max(
                    f["angular_speed"] for f in fs[step - 10 : step + 1]
                ),
                next_strict_complete_step=next_stable,
                premature_seconds=(next_stable - step) / 20,
            )
        )
    output = dict(
        status="EVIDENCE_REPLAY_PASS",
        acceptance="NO_GO",
        episodes=20,
        checked_controls=controls,
        checked_predictions=len(rows),
        candidate=m,
        baseline=metrics(rows, "baseline_decision"),
        natural_baseline=metrics(natural, "baseline_decision"),
        missed_as_incomplete=sum(
            r["label"] == 1 and r["decision"] == "incomplete" for r in rows
        ),
        missed_as_unknown=sum(
            r["label"] == 1 and r["decision"] == "unknown" for r in rows
        ),
        false_complete_diagnostics=errors,
        limitation="Checks retained labels, predictions and hashes; image shards, state arrays and weights remain on server.",
    )
    (HERE / "evidence_replay.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
