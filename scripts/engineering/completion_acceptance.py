"""Frozen independent acceptance; no training or performance-based retries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from actionstream.completion_labels import StableTruth, simulator_facts
from actionstream.llm_vla.confirmation import ContinuousConfirmation, episode_metrics
from actionstream.llm_vla.temporal_completion import (
    camera_rgb,
    classify,
    TemporalPredictor,
)
from completion_controls import (
    batch,
    make_env,
    predictions,
    restore_episode,
    stop_probe,
)
from completion_v2 import layout, save, setup, sha

SOURCES = [
    "scripts/engineering/completion_acceptance.py",
    "scripts/engineering/completion_controls.py",
    "scripts/engineering/completion_v2.py",
    "scripts/engineering/train_visual_completion.py",
    "src/actionstream/completion_labels.py",
    "src/actionstream/llm_vla/confirmation.py",
    "src/actionstream/llm_vla/temporal_completion.py",
    "src/actionstream/llm_vla/completion.py",
    "src/actionstream/lerobot_backend.py",
    "src/actionstream/libero_config.py",
]
HOLD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])


def read(path):
    return json.loads(Path(path).read_text())


def root(args):
    return args.base / "completion-acceptance-v3"


def verify_lock(args, p):
    lock = read(root(args) / "freeze.json")
    assert sha(args.protocol) == lock["protocol_sha256"]
    assert (
        sha(root(args) / "frozen.pt")
        == p["checkpoint_sha256"]
        == lock["checkpoint_sha256"]
    )
    for name, digest in lock["source_sha256"].items():
        assert sha(name) == digest, name
    return lock


def stage(args, p, name):
    verify_lock(args, p)
    out = root(args) / name
    out.mkdir(exist_ok=False)
    (out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (out / "protocol.json").write_bytes(args.protocol.read_bytes())
    return out


def hold_mode(env):
    controller = env.robots[0].controller
    controller.use_delta = True
    controller.update(force=True)
    controller.reset_goal()


def confirmed(rows, p):
    gate = ContinuousConfirmation(p["confirmation_controls"])
    return [(r["control"], gate.update(r["control"], r["decision"])) for r in rows]


def score_episodes(episodes, p):
    g = p["closed_loop_gate"]
    stopped = [e for e in episodes if e["first_claim_control"] is not None]
    delays = [
        e["confirmation_delay_s"]
        for e in episodes
        if e["confirmation_delay_s"] is not None
    ]
    result = dict(
        episodes=len(episodes),
        physically_completed_episodes=sum(
            e["first_truth_control"] is not None for e in episodes
        ),
        premature_stops=sum(e["premature_stop"] for e in episodes),
        missed_completed_episodes=sum(e["missed_completed_event"] for e in episodes),
        no_stop=sum(e["no_stop"] for e in episodes),
        post_stop_failures=sum(e["post_stop_stable"] is False for e in stopped),
        missing_post_stop_probes=sum(e["post_stop_stable"] is None for e in stopped),
        maximum_confirmation_delay_s=max(delays) if delays else None,
        median_confirmation_delay_s=float(np.median(delays)) if delays else None,
        late_confirmations=sum(d > g["max_confirmation_delay_s"] for d in delays),
        exact_seed_coverage=sorted(e["seed"] for e in episodes) == sorted(p["seeds"]),
    )
    result["passed"] = bool(
        result["exact_seed_coverage"]
        and len(episodes) == g["expected_episodes"]
        and result["physically_completed_episodes"]
        >= g["minimum_physically_completed_episodes"]
        and result["premature_stops"] <= g["premature_stops_max"]
        and result["missed_completed_episodes"] <= g["missed_completed_episodes_max"]
        and result["post_stop_failures"] <= g["post_stop_failures_max"]
        and result["missing_post_stop_probes"] <= g["missing_post_stop_probes_max"]
        and result["late_confirmations"] == 0
    )
    return result


def intervals(truth, decisions):
    hits = {i for i, yes in decisions if yes}
    runs, start = [], None
    for i, yes in enumerate(list(truth) + [False]):
        if yes and start is None:
            start = i
        if not yes and start is not None:
            runs.append(
                dict(
                    start=start,
                    end=i - 1,
                    detected=any(j in hits for j in range(start, i)),
                )
            )
            start = None
    return runs


def freeze(args, p):
    out = root(args)
    out.mkdir(exist_ok=False)
    checkpoint = args.base / "development-controls/adaptation-training/best.pt"
    development = args.base / "development-controls/adaptation-training/frozen.json"
    assert sha(checkpoint) == p["checkpoint_sha256"]
    assert read(development)["checkpoint_sha256"] == p["checkpoint_sha256"]
    shutil.copyfile(checkpoint, out / "frozen.pt")
    shutil.copyfile(args.protocol, out / "protocol.json")
    (out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    save(
        out / "freeze.json",
        dict(
            status="FROZEN_BEFORE_HELDOUT_COLLECTION",
            checkpoint_sha256=sha(out / "frozen.pt"),
            protocol_sha256=sha(args.protocol),
            source_sha256={name: sha(name) for name in SOURCES},
            development_freeze_sha256=sha(development),
            development_collection_sha256=sha(
                args.base / "development-controls/collection/manifest.json"
            ),
            consumed_v2_collection_sha256=sha(
                args.base / "completion-v2/fresh_holdout/manifest.json"
            ),
            frozen_unix=time.time(),
            independent_test_seen=False,
        ),
    )
    predictor = TemporalPredictor(out / "frozen.pt", p["checkpoint_sha256"])
    with np.load(
        args.base / "development-controls/collection/validation_2026091808.npz",
        allow_pickle=False,
    ) as d:
        probabilities = predictor.predict(d["rgb"][np.array([0, 5, 10])][None])[0]
    decision = classify(probabilities)
    save(
        out / "smoke_receipt.json",
        dict(
            status="PASS",
            input_source="previously consumed development validation only",
            probabilities=probabilities.tolist(),
            decision=decision,
            checkpoint_sha256=p["checkpoint_sha256"],
        ),
    )
    print("ACCEPTANCE_FROZEN_SMOKE_PASS", flush=True)


def backend_for(args, p, out):
    import torch
    from actionstream.lerobot_backend import LeRobotBackend

    setup(out)
    torch.set_num_threads(8)
    backend = LeRobotBackend(
        task_ids=[5],
        seed=p["seeds"][0],
        suite="libero_object",
        episode_length=400,
        model_id=str(args.base / "assets/xvla"),
        model_revision=p["policy_revision"],
        device="cuda",
    )
    sub = backend._sub_env(5)
    sub.init_states = False
    sub._init_states = None
    return backend, sub


def prior_layouts(args, env):
    import h5py

    used = {
        e["initial_layout_sha256"]
        for e in read(args.base / "development-controls/collection/manifest.json")[
            "entries"
        ]
    }
    used.update(
        e["metadata"]["initial_layout_sha256"]
        for e in read(args.base / "completion-v2/fresh_holdout/manifest.json")[
            "entries"
        ]
    )
    with h5py.File(args.base / "data/tomato-xet.hdf5") as f:
        for name in f["data"]:
            env.set_init_state(f["data"][name]["states"][0])
            used.add(layout(env))
    return used


def run_episode(backend, sub, seed, p, model=None):
    observation, _, instruction = backend.reset_episode(
        task_id=5, seed=seed, initial_state_index=0
    )
    env = sub._env
    assert env.robots[0].controller.use_delta is False
    signature = layout(env)
    raw = env.set_init_state(env.sim.get_state().flatten())
    truth, gate = (
        StableTruth(p["truth"]),
        ContinuousConfirmation(p["confirmation_controls"]),
    )
    rgb, states, facts, actions, journal = [], [], [], [], []
    for control in range(361):
        f = simulator_facts(env)
        f["strict_complete"] = truth.update(control, f)
        facts.append(f)
        states.append(env.sim.get_state().flatten().copy())
        rgb.append(camera_rgb(raw))
        if model is not None and control >= 10:
            pr = model.predict(np.stack([rgb[control - k] for k in (10, 5, 0)])[None])[
                0
            ]
            decision = classify(pr)
            yes = gate.update(control, decision)
            journal.append(
                dict(
                    control=control,
                    probabilities=pr.tolist(),
                    decision=decision,
                    confirmed=yes,
                )
            )
            if yes:
                break
        if control == 360:
            break
        if control < 300:
            if control % 30 == 0:
                chunk = backend.infer_action_chunk(observation, instruction).actions
            action = chunk[control % 30]
        else:
            if control == 300:
                hold_mode(env)
            action = HOLD
        raw, _, _, _ = env.step(action)
        actions.append(action.tolist())
        observation = batch(sub._format_raw_obs(raw))
    post, post_states, post_rgb = [], [], []
    if model is not None and journal[-1]["confirmed"]:
        hold_mode(env)
        for i in range(p["stop_probe"]["controls"]):
            raw, _, _, _ = env.step(HOLD)
            f = simulator_facts(env)
            f["strict_complete"] = truth.update(len(facts) + i, f)
            post.append(f)
            post_states.append(env.sim.get_state().flatten().copy())
            post_rgb.append(camera_rgb(raw))
    return dict(
        seed=seed,
        instruction=instruction,
        initial_layout_sha256=signature,
        xml=env.env.model.get_xml(),
        rgb=np.stack(rgb),
        states=np.stack(states),
        truth=facts,
        actions=actions,
        observations=journal,
        post=post,
        post_states=np.asarray(post_states),
        post_rgb=np.asarray(post_rgb),
    )


def retain_episode(out, name, data):
    (out / (name + ".xml")).write_text(data["xml"])
    np.savez_compressed(
        out / (name + ".npz"),
        rgb=data["rgb"],
        states=data["states"],
        post_states=data["post_states"],
        post_rgb=data["post_rgb"],
    )
    for key in ("truth", "actions", "observations", "post"):
        save(out / (name + "_" + key + ".json"), data[key])
    return dict(
        episode=name,
        seed=data["seed"],
        instruction=data["instruction"],
        initial_layout_sha256=data["initial_layout_sha256"],
        control_records=len(data["truth"]),
        post_control_records=len(data["post"]),
        files_sha256={
            name + suffix: sha(out / (name + suffix))
            for suffix in (
                ".npz",
                ".xml",
                "_truth.json",
                "_actions.json",
                "_observations.json",
                "_post.json",
            )
        },
    )


def collect(args, p):
    out = stage(args, p, "collection")
    backend, sub = backend_for(args, p, out)
    entries = []
    try:
        sub._ensure_env()
        used = prior_layouts(args, sub._env)
        for seed in p["seeds"]:
            data = run_episode(backend, sub, seed, p)
            assert data["initial_layout_sha256"] not in used, (
                "Duplicate initial layout; no seed replacement"
            )
            used.add(data["initial_layout_sha256"])
            name = f"holdout_{seed}"
            entry = retain_episode(out, name, data)
            entries.append(entry)
            save(out / "progress.json", entries)
            print(
                json.dumps(
                    dict(
                        phase="collect",
                        seed=seed,
                        positive_controls=sum(
                            f["strict_complete"] for f in data["truth"]
                        ),
                    )
                ),
                flush=True,
            )
    finally:
        backend.close()
    save(
        out / "manifest.json",
        dict(
            entries=entries,
            freeze_sha256=sha(root(args) / "freeze.json"),
            completed_unix=time.time(),
            model_predictions_used=False,
        ),
    )


def window(facts, predicate):
    return next(
        (
            i
            for i in range(len(facts) - 20)
            if all(predicate(f) for f in facts[i : i + 21])
        ),
        None,
    )


def challenges(args, p):
    out = stage(args, p, "challenges")
    source = root(args) / "collection"
    env = make_env(args, out)
    entries, chosen = [], []
    try:
        for entry in read(source / "manifest.json")["entries"]:
            name = entry["episode"]
            facts = read(source / (name + "_truth.json"))
            held = window(
                facts, lambda f: f["finger_contact"] and not f["strict_complete"]
            )
            stable = window(facts, lambda f: f["strict_complete"])
            if held is None or stable is None:
                continue
            chosen.append(name)
            restore_episode(env, source, name)
            with np.load(source / (name + ".npz"), allow_pickle=False) as d:
                for category in p["challenges"]["categories"]:
                    start = held if category == "held" else stable
                    indices = list(range(start, start + 21))
                    images, states, private = [], [], []
                    oracle = StableTruth(p["truth"])
                    for i in range(max(0, start - 10), start):
                        oracle.update(i, facts[i])
                    for j, i in enumerate(indices):
                        raw = env.set_init_state(d["states"][i])
                        transform = (
                            category
                            if category in ("above", "outside", "wrong_object")
                            else "outside"
                            if category == "recent_drop" and j >= 11
                            else None
                        )
                        if transform:
                            target = env.env.objects_dict["tomato_sauce_1"].joints[0]
                            one = env.sim.data.get_joint_qpos(target).copy()
                            if transform == "wrong_object":
                                other = env.env.objects_dict["bbq_sauce_1"].joints[0]
                                two = env.sim.data.get_joint_qpos(other).copy()
                                one[:3], two[:3] = two[:3].copy(), one[:3].copy()
                                env.sim.data.set_joint_qpos(other, two)
                            elif transform == "above":
                                one[2] += 0.15
                            else:
                                one[0] += 0.30
                            env.sim.data.set_joint_qpos(target, one)
                            raw = env.set_init_state(env.sim.get_state().flatten())
                            assert not env.check_success(), (name, category, i)
                        f = simulator_facts(env)
                        f["strict_complete"] = oracle.update(i, f)
                        private.append(f)
                        rgb = (
                            d["rgb"][i].copy()
                            if category == "held" or category.endswith("hidden")
                            else camera_rgb(raw)
                        )
                        if category in ("external_hidden", "both_hidden"):
                            rgb[0] = 0
                        if category in ("wrist_hidden", "both_hidden"):
                            rgb[1] = 0
                        images.append(rgb)
                        states.append(env.sim.get_state().flatten().copy())
                    case = name + "_" + category
                    np.savez_compressed(
                        out / (case + ".npz"),
                        rgb=np.stack(images),
                        states=np.stack(states),
                    )
                    record = dict(
                        case=case,
                        source_episode=name,
                        category=category,
                        source_controls=indices,
                        private_truth=private,
                        labels=[
                            2
                            if category.endswith("hidden")
                            else int(f["strict_complete"])
                            for f in private
                        ],
                        image_state_sha256=sha(out / (case + ".npz")),
                    )
                    save(out / (case + ".json"), record)
                    entries.append(
                        dict(case=case, record_sha256=sha(out / (case + ".json")))
                    )
            if len(chosen) == p["challenges"]["sources"]:
                break
    finally:
        env.close()
    save(
        out / "manifest.json",
        dict(
            entries=entries,
            source_episodes=chosen,
            selection_uses_model_predictions=False,
            collection_sha256=sha(source / "manifest.json"),
        ),
    )
    print(
        json.dumps(
            dict(phase="challenges", sources=len(chosen), sequences=len(entries))
        ),
        flush=True,
    )


def evaluate(args, p):
    out = stage(args, p, "paired-evaluation")
    source = root(args) / "collection"
    cases = root(args) / "challenges"
    env = make_env(args, out)
    models = [
        (
            "baseline",
            args.base / "completion-v2/training/best.pt",
            p["baseline_sha256"],
        ),
        ("adapted", root(args) / "frozen.pt", p["checkpoint_sha256"]),
    ]
    results = {}
    try:
        for label, checkpoint, digest in models:
            model = TemporalPredictor(checkpoint, digest)
            episodes, stress = [], []
            for entry in read(source / "manifest.json")["entries"]:
                name = entry["episode"]
                restore_episode(env, source, name)
                facts = read(source / (name + "_truth.json"))
                with np.load(source / (name + ".npz"), allow_pickle=False) as d:
                    rows = predictions(model, d["rgb"])
                    decisions = confirmed(rows, p)
                    m = episode_metrics(
                        [f["strict_complete"] for f in facts], decisions
                    )
                    post = stop_probe(env, d["states"], m["first_claim_control"], p)
                save(out / f"{name}_{label}_predictions.json", rows)
                save(out / f"{name}_{label}_post.json", post)
                m = episode_metrics(
                    [f["strict_complete"] for f in facts],
                    decisions,
                    post_stop=[f["strict_complete"] for f in post] if post else None,
                )
                episodes.append(
                    dict(
                        episode=name,
                        seed=entry["seed"],
                        **m,
                        complete_intervals=intervals(
                            [f["strict_complete"] for f in facts], decisions
                        ),
                    )
                )
            for entry in read(cases / "manifest.json")["entries"]:
                case = entry["case"]
                record = read(cases / (case + ".json"))
                with np.load(cases / (case + ".npz"), allow_pickle=False) as d:
                    rows = predictions(model, d["rgb"])
                decisions = confirmed(rows, p)
                false_claims = [
                    i
                    for i, yes in decisions
                    if yes and not record["private_truth"][i]["strict_complete"]
                ]
                hidden_confident = (
                    sum(r["decision"] != "unknown" for r in rows)
                    if record["category"].endswith("hidden")
                    else 0
                )
                save(out / f"{case}_{label}_predictions.json", rows)
                stress.append(
                    dict(
                        case=case,
                        category=record["category"],
                        confirmed_false_complete_controls=false_claims,
                        blackout_confident_decisions=hidden_confident,
                    )
                )
            cg = p["challenges"]["gates"]
            challenge_gate = dict(
                sequences=len(stress),
                confirmed_false_complete=sum(
                    len(r["confirmed_false_complete_controls"]) for r in stress
                ),
                blackout_confident_decisions=sum(
                    r["blackout_confident_decisions"] for r in stress
                ),
            )
            challenge_gate["passed"] = (
                len(stress) == cg["expected_sequences"]
                and challenge_gate["confirmed_false_complete"]
                <= cg["confirmed_false_complete_max"]
                and challenge_gate["blackout_confident_decisions"]
                <= cg["blackout_confident_decisions_max"]
            )
            results[label] = dict(
                episodes=episodes,
                closed_loop=score_episodes(episodes, p),
                challenges=stress,
                challenge_gate=challenge_gate,
                checkpoint_sha256=digest,
            )
            print(
                json.dumps(
                    dict(
                        phase="paired",
                        model=label,
                        closed_loop=results[label]["closed_loop"],
                        challenges=challenge_gate,
                    )
                ),
                flush=True,
            )
            del model
    finally:
        env.close()
    save(
        out / "summary.json",
        dict(
            results=results,
            collection_sha256=sha(source / "manifest.json"),
            challenge_manifest_sha256=sha(cases / "manifest.json"),
            freeze_sha256=sha(root(args) / "freeze.json"),
        ),
    )


def live(args, p):
    out = stage(args, p, "live")
    source = root(args) / "collection"
    refs = {e["seed"]: e for e in read(source / "manifest.json")["entries"]}
    model = TemporalPredictor(root(args) / "frozen.pt", p["checkpoint_sha256"])
    backend, sub = backend_for(args, p, out)
    episodes = []
    try:
        for seed in p["seeds"]:
            name = f"holdout_{seed}"
            data = run_episode(backend, sub, seed, p, model=model)
            assert data["initial_layout_sha256"] == refs[seed]["initial_layout_sha256"]
            entry = retain_episode(out, name, data)
            with np.load(source / (name + ".npz"), allow_pickle=False) as d:
                prefix_rgb_equal = np.array_equal(
                    data["rgb"], d["rgb"][: len(data["rgb"])]
                )
                state_delta = float(
                    np.max(np.abs(data["states"] - d["states"][: len(data["states"])]))
                )
            ref_actions = read(source / (name + "_actions.json"))[
                : len(data["actions"])
            ]
            m = episode_metrics(
                [f["strict_complete"] for f in data["truth"]],
                [(r["control"], r["confirmed"]) for r in data["observations"]],
                post_stop=[f["strict_complete"] for f in data["post"]]
                if data["post"]
                else None,
            )
            episodes.append(
                dict(
                    entry,
                    **m,
                    reference_prefix_rgb_equal=prefix_rgb_equal,
                    reference_prefix_actions_equal=data["actions"] == ref_actions,
                    reference_prefix_max_state_delta=state_delta,
                )
            )
            save(out / "progress.json", episodes)
            print(
                json.dumps(
                    dict(phase="live", seed=seed, **m, prefix_equal=prefix_rgb_equal)
                ),
                flush=True,
            )
    finally:
        backend.close()
    save(
        out / "summary.json",
        dict(
            episodes=episodes,
            closed_loop=score_episodes(episodes, p),
            freeze_sha256=sha(root(args) / "freeze.json"),
            independent_acceptance=True,
            completed_unix=time.time(),
        ),
    )


def verdict(args, p):
    verify_lock(args, p)
    paired = read(root(args) / "paired-evaluation/summary.json")
    actual = read(root(args) / "live/summary.json")
    adapted = paired["results"]["adapted"]
    complete = (
        len(actual["episodes"]) == len(p["seeds"])
        and adapted["challenge_gate"]["sequences"]
        == p["challenges"]["gates"]["expected_sequences"]
    )
    passed = (
        actual["closed_loop"]["passed"]
        and adapted["closed_loop"]["passed"]
        and adapted["challenge_gate"]["passed"]
    )
    result = dict(
        status="GO" if complete and passed else "NO-GO" if complete else "INCOMPLETE",
        live=actual["closed_loop"],
        paired_adapted=adapted["closed_loop"],
        paired_baseline=paired["results"]["baseline"]["closed_loop"],
        challenges=adapted["challenge_gate"],
        freeze_sha256=sha(root(args) / "freeze.json"),
        protocol_sha256=sha(args.protocol),
        checkpoint_sha256=p["checkpoint_sha256"],
        completed_unix=time.time(),
        test_used_for_fitting=False,
        scope=p["scope"],
    )
    save(root(args) / "verdict.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=["freeze", "collect", "challenges", "evaluate", "live", "verdict"],
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    try:
        globals()[args.phase](args, p)
    except Exception as exc:
        if root(args).exists():
            save(
                root(args) / f"failure_{args.phase}_{time.time_ns()}.json",
                dict(
                    phase=args.phase,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    time_unix=time.time(),
                ),
            )
        raise


if __name__ == "__main__":
    main()
