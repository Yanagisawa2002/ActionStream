"""Independently replay compact development evidence using the standard library.

Does not import the model, confirmation implementation, or simulator. Recomputes
labels from private physical facts, decisions from probabilities, event metrics,
selection, split separation, and hashes. State restoration is a separate audit.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def raw(path):
    if path.exists():
        return path.read_bytes()
    with tarfile.open(path.parent / "raw_records.tar.gz") as archive:
        return archive.extractfile(path.name).read()


def read(path):
    return json.loads(raw(path))


def sha(path):
    return hashlib.sha256(raw(path)).hexdigest()


def physical(f):
    return (
        f["inside"]
        and not f["finger_contact"]
        and f["basket_contact"]
        and math.isfinite(f["linear_speed"])
        and math.isfinite(f["angular_speed"])
        and 0 <= f["linear_speed"] <= 0.03
        and 0 <= f["angular_speed"] <= 0.3
    )


def truth_check(facts, prior=()):
    whole = list(prior) + facts
    values = [physical(f) for f in whole]
    for i, f in enumerate(facts, len(prior)):
        assert f["strict_complete"] == (i >= 10 and all(values[i - 10 : i + 1])), i


def classify(r):
    p = r["probabilities"]
    assert len(p) == 3 and all(math.isfinite(x) and 0 <= x <= 1 for x in p)
    assert abs(sum(p) - 1) < 1e-5
    decision = (
        "complete" if p[1] >= 0.95 else "incomplete" if p[0] >= 0.9 else "unknown"
    )
    assert r["decision"] == decision
    return decision


def claims(rows, mode):
    assert all(b["control"] > a["control"] for a, b in zip(rows, rows[1:]))
    result, positives = [], []
    for r in rows:
        i, d = r["control"], classify(r)
        if mode == "sparse_4hz" and i % 5:
            continue
        if d == "complete":
            positives = positives + [i] if positives and i == positives[-1] + 1 else [i]
        else:
            positives = []
        confirmed = len(positives) >= 11 if "confirm" in mode else d == "complete"
        if "confirmed" in r:
            assert r["confirmed"] == confirmed
        if confirmed:
            result.append(i)
    return result


def events(facts, indices, post):
    truth = [f["strict_complete"] for f in facts]
    first_true = next((i for i, f in enumerate(truth) if f), None)
    first = next(iter(indices), None)
    correct = next((i for i in indices if truth[i]), None)
    early = first is not None and not truth[first]
    if first is None:
        assert post is None
    else:
        assert len(post) == 40
    return dict(
        first_truth_control=first_true,
        first_claim_control=first,
        first_correct_control=correct,
        premature_stop=early,
        missed_completed_event=first_true is not None and correct is None,
        no_stop=first is None,
        confirmation_delay_s=(first - first_true) / 20
        if first is not None and first_true is not None and not early
        else None,
        post_stop_stable=all(f["strict_complete"] for f in post) if post else None,
    )


def aggregate(rows):
    delays = [
        r["confirmation_delay_s"] for r in rows if r["confirmation_delay_s"] is not None
    ]
    return dict(
        episodes=len(rows),
        premature_stops=sum(r["premature_stop"] for r in rows),
        missed_completed_events=sum(r["missed_completed_event"] for r in rows),
        post_stop_failures=sum(r["post_stop_stable"] is False for r in rows),
        max_confirmation_delay_s=max(delays) if delays else None,
    )


def key(log):
    s = log["validation"]
    return (
        not s["passed"],
        s["premature_stops"],
        s["missed_events"],
        s["post_stop_failures"],
        s["late_confirmations"],
        s["auxiliary"]["physical_false_complete"],
        s["auxiliary"]["blackout_confident"],
        s["max_delay_s"] if s["max_delay_s"] is not None else 999,
    )


def main():
    p = read(ROOT / "configs/completion_development_controls.json")
    a = read(ROOT / "configs/completion_adaptation.json")
    assert p["truth"] == dict(
        window_steps=10, linear_speed_max=0.03, angular_speed_max=0.3
    )
    assert p["temporal"]["confirmation_controls"] == 10
    collection = read(HERE / "collection/manifest.json")
    nuisance = read(HERE / "nuisance/summary.json")
    temporal = read(HERE / "temporal/summary.json")
    added = read(HERE / "adaptation-data/manifest.json")
    frozen = read(HERE / "adaptation-training/frozen.json")
    live = read(HERE / "live-development-validation/summary.json")
    execution = read(HERE / "execution_receipt.json")
    for path, expected in execution["source_sha256"].items():
        assert sha(ROOT / path) == expected
    assert execution["checkpoint_sha256"] == frozen["checkpoint_sha256"]
    assert read(HERE / "adaptation-training/smoke_receipt.json") == dict(
        status="PASS", backward_optimizer_steps=1, exact_reload=True
    )
    if (HERE / "archived_records.json").exists():
        for name, expected in read(HERE / "archived_records.json").items():
            assert sha(HERE / name) == expected
    for folder, obj in (
        ("collection", collection),
        ("nuisance", nuisance),
        ("temporal", temporal),
        ("adaptation-data", added),
        ("adaptation-training", frozen),
        ("live-development-validation", live),
    ):
        source = (
            "completion_controls.py"
            if folder in ("collection", "nuisance", "temporal")
            else "completion_adapt.py"
        )
        config = (
            "completion_development_controls.json"
            if source == "completion_controls.py"
            else "completion_adaptation.json"
        )
        assert (
            obj["source_sha256"]
            == sha(HERE / folder / "source_snapshot.py")
            == sha(ROOT / "scripts/engineering" / source)
        )
        assert (
            obj["protocol_sha256"]
            == sha(HERE / folder / "protocol.json")
            == sha(ROOT / "configs" / config)
        )
    assert (
        not collection["model_predictions_used"]
        and not frozen["independent_test_seen"]
        and not live["independent_acceptance"]
    )
    assert (
        frozen["collection_sha256"]
        == added["collection_sha256"]
        == sha(HERE / "collection/manifest.json")
    )
    assert frozen["additional_data_sha256"] == sha(
        HERE / "adaptation-data/manifest.json"
    )
    assert added["nuisance_summary_sha256"] == sha(HERE / "nuisance/summary.json")
    assert live["frozen_sha256"] == sha(HERE / "adaptation-training/frozen.json")
    assert live["checkpoint_sha256"] == frozen["checkpoint_sha256"]
    assert (
        a["warm_start_sha256"]
        == p["frozen_v2_sha256"]
        == temporal["checkpoint_sha256"]
        == nuisance["checkpoint_sha256"]
    )
    entries = collection["entries"]
    assert len(entries) == len({e["initial_layout_sha256"] for e in entries}) == 12
    train = {e["episode"] for e in entries if e["split"] == "train"}
    validation = {e["episode"] for e in entries if e["split"] == "validation"}
    assert len(train) == 8 and len(validation) == 4 and not train & validation
    for split in ("train", "validation"):
        assert [e["seed"] for e in entries if e["split"] == split] == p[
            split + "_seeds"
        ]
    old = read(HERE.parent / "completion_v2_20260915/fresh_holdout/manifest.json")
    assert not {e["initial_layout_sha256"] for e in entries} & {
        e["metadata"]["initial_layout_sha256"] for e in old["entries"]
    }
    truth = {}
    for e in entries:
        name = e["episode"]
        path = HERE / "collection" / (name + "_truth.json")
        assert sha(path) == e["truth_sha256"]
        facts = read(path)
        assert len(facts) == 361 and e["controls"] == 360
        truth_check(facts)
        assert (
            sum(f["strict_complete"] for f in facts)
            == e["strict_positive_controls"]
            > 0
        )
        actions = HERE / "collection" / (name + "_actions.json")
        assert sha(actions) == e["actions_sha256"] and len(read(actions)) == 300
        truth[name] = facts

    state_audit = read(HERE / "state-replay-audit/summary.json")
    assert state_audit["source_sha256"] == sha(
        HERE / "state-replay-audit/source_snapshot.py"
    )
    restored = {}
    state_controls, contact_changes = 0, 0
    for record in state_audit["records"]:
        name, stage = record["episode"], record["stage"]
        replayed = read(HERE / "state-replay-audit" / (stage + "_" + name + ".json"))
        original = (
            truth[name]
            if stage == "collection"
            else read(HERE / stage / (name + "_private_truth.json"))["controls"]
        )
        truth_check(replayed)
        assert len(replayed) == len(original) == record["controls"]
        assert all(
            a["strict_complete"] == b["strict_complete"]
            for a, b in zip(original, replayed, strict=True)
        )
        expected_changes = []
        for i, (before, after) in enumerate(zip(original, replayed, strict=True)):
            fields = [
                k
                for k in (
                    "inside",
                    "finger_contact",
                    "basket_contact",
                    "strict_complete",
                )
                if before[k] != after[k]
            ]
            if fields:
                expected_changes.append(
                    dict(control=i, fields=fields, recorded=before, restored=after)
                )
        assert expected_changes == record["changed_controls"]
        assert record["max_speed_delta"] == 0
        if stage == "collection":
            restored[name] = replayed
            entry = next(e for e in entries if e["episode"] == name)
            assert entry["shard_sha256"] == record["image_state_sha256"]
        else:
            entry = next(e for e in live["episodes"] if e["episode"] == name)
            assert entry["image_state_sha256"] == record["image_state_sha256"]
        state_controls += len(replayed)
        contact_changes += len(expected_changes)
    assert state_controls == 4952 and contact_changes == 17

    variants = read(HERE / "nuisance/rows.json")
    assert len(variants) == nuisance["variants"] == 72
    eligible_rows = []
    for r in variants:
        classify(r)
        assert r["physically_valid"] == (
            r["objects_unchanged"] and physical(r["facts"])
        )
        assert r["visibility_proxy"] == (max(r["target_pixels"]) >= 64)
        if r["physically_valid"] and r["visibility_proxy"]:
            eligible_rows.append(r)
    assert len(eligible_rows) == nuisance["valid_visible"] == 52
    factor_results = {}
    for factor in ("arm", "gripper", "camera"):
        pairs = [
            r for r in nuisance["pairs"] if r["factor"] == factor and r["eligible"]
        ]
        for pair in pairs:
            same = [r for r in variants if r["episode"] == pair["episode"]]
            ref = next(
                r
                for r in same
                if r["arm"] == "reference"
                and r["gripper"] == "open"
                and r["camera_dx"] == 0
            )
            changed = next(
                r
                for r in same
                if all(r[k] == pair[k] for k in ("arm", "gripper", "camera_dx"))
            )
            assert ref in eligible_rows and changed in eligible_rows
            assert (
                pair["delta"] == changed["probabilities"][1] - ref["probabilities"][1]
            )
            assert (
                pair["reference_decision"] == ref["decision"]
                and pair["intervention_decision"] == changed["decision"]
            )
        factor_results[factor] = dict(
            eligible_pairs=len(pairs),
            complete_to_other=sum(
                r["reference_decision"] == "complete"
                and r["intervention_decision"] != "complete"
                for r in pairs
            ),
            deltas=[r["delta"] for r in pairs],
        )

    comparisons = {s: {} for s in ("train", "validation")}
    for result in temporal["results"]:
        name = result["episode"]
        rows = read(HERE / "temporal" / (name + "_predictions.json"))
        assert [r["control"] for r in rows] == list(range(10, 361))
        for mode, recorded in result["modes"].items():
            indices = claims(rows, mode)
            post = (
                read(HERE / "temporal" / (name + "_" + mode + "_post_stop.json"))
                if indices
                else None
            )
            if post:
                truth_check(
                    post, prior=restored[name][indices[0] - 10 : indices[0] + 1]
                )
            assert events(truth[name], indices, post) == recorded
    for split in comparisons:
        for mode in p["temporal"]["modes"]:
            comparisons[split][mode] = aggregate(
                [r["modes"][mode] for r in temporal["results"] if r["split"] == split]
            )
    assert comparisons["train"] == temporal["train_comparison"]

    additions = read(HERE / "adaptation-data/rows.json")
    assert len(additions) == added["samples"] == 620
    assert {str(k): v for k, v in Counter(r["label"] for r in additions).items()} == {
        k: v for k, v in added["counts"].items() if v
    }
    assert {r["episode"] for r in additions} == train
    for r in additions:
        if r["source"] == "new_vla":
            assert r["control"] in range(10, 361, 5)
            assert (
                r["physical_complete"]
                == truth[r["episode"]][r["control"]]["strict_complete"]
            )
            assert r["label"] == (
                int(r["physical_complete"]) if max(r["target_pixels"]) >= 64 else 2
            )
        else:
            assert r["label"] == 1 and variants[r["nuisance_clip"]] in eligible_rows
    assert frozen["train_samples"] == 2877 and frozen["train_counts"] == {
        "0": 1724,
        "1": 703,
        "2": 450,
    }
    logs = [
        json.loads(line)
        for line in raw(HERE / "adaptation-training/metrics.jsonl").splitlines()
    ]
    assert [r["epoch"] for r in logs] == list(range(1, 21))
    assert [r["steps"] for r in logs] == list(range(120, 2401, 120))
    selected = min(logs, key=key)
    assert (
        selected == frozen["selected_validation"]
        and selected["epoch"] == frozen["chosen_epoch"]
    )
    journals = read(HERE / "adaptation-training/selected_predictions.json")
    assert set(journals) == validation
    clip_diagnostic = Counter()
    for e in selected["events"]:
        name = e["episode"]
        rows = journals[name]
        indices = claims(rows, "dense_20hz_confirm_0.5s")
        post = (
            read(HERE / "adaptation-training" / f"{name}_stop_{indices[0]}.json")
            if indices
            else None
        )
        if post:
            truth_check(post, prior=restored[name][indices[0] - 10 : indices[0] + 1])
        assert all(e[k] == v for k, v in events(truth[name], indices, post).items())
        for r in rows:
            clip_diagnostic[
                (truth[name][r["control"]]["strict_complete"], r["decision"])
            ] += 1
    auxiliary = read(HERE / "adaptation-training/selected_auxiliary.json")
    for r in auxiliary:
        classify(r)
        assert r["label"] == (2 if r["category"].endswith("hidden") else 0)
    assert len(auxiliary) == 492
    assert sum(r["label"] == 0 for r in auxiliary) == 342
    assert sum(r["label"] == 2 for r in auxiliary) == 150
    assert sum(r["label"] == 0 and r["decision"] == "complete" for r in auxiliary) == 0
    assert sum(r["label"] == 2 and r["decision"] != "unknown" for r in auxiliary) == 0
    live_controls = 0
    for e in live["episodes"]:
        name = e["episode"]
        rows = read(
            HERE / "live-development-validation" / (name + "_observations.json")
        )
        assert (
            sha(HERE / "live-development-validation" / (name + "_observations.json"))
            == e["observation_sha256"]
        )
        private = read(
            HERE / "live-development-validation" / (name + "_private_truth.json")
        )
        facts, post = private["controls"], private["post_stop"]
        truth_check(facts)
        truth_check(post, prior=facts)
        indices = claims(rows, "dense_20hz_confirm_0.5s")
        assert len(indices) == 1 and indices[0] == len(facts) - 1
        assert (
            len(read(HERE / "live-development-validation" / (name + "_actions.json")))
            == len(facts) - 1
        )
        assert all(e[k] == v for k, v in events(facts, indices, post).items())
        live_controls += len(facts) + len(post)
    result = dict(
        status="PASS",
        collection_controls=sum(map(len, truth.values())),
        baseline_prediction_records=12 * 351,
        live_controls_and_post=live_controls,
        simulator_restored_states=state_controls,
        restored_contact_or_containment_changes=contact_changes,
        restored_strict_label_changes=0,
        nuisance=factor_results,
        temporal=comparisons,
        chosen_epoch=selected["epoch"],
        total_optimizer_steps=logs[-1]["steps"],
        passing_epochs=sum(r["validation"]["passed"] for r in logs),
        new_vla_positive_before_tail=sum(
            r["source"] == "new_vla" and r["label"] == 1 and r["control"] <= 300
            for r in additions
        ),
        new_vla_positive_tail=sum(
            r["source"] == "new_vla" and r["label"] == 1 and r["control"] > 300
            for r in additions
        ),
        selected_clip_diagnostic={
            f"truth_{int(k[0])}_{k[1]}": v for k, v in clip_diagnostic.items()
        },
        live=aggregate(live["episodes"]),
        independent_acceptance=False,
        limitation="record replay; does not re-render or rerun model inference",
    )
    (HERE / "independent_replay.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
