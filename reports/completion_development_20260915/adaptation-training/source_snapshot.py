"""Development adaptation and actual RGB-controlled stopping after paired controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np

from actionstream.completion_labels import StableTruth, simulator_facts
from actionstream.llm_vla.confirmation import ContinuousConfirmation, episode_metrics
from actionstream.llm_vla.temporal_completion import (
    camera_rgb,
    classify,
    make_model,
    TemporalPredictor,
)
from completion_controls import (
    batch,
    make_env,
    predictions,
    restore_episode,
    stop_probe,
    visibility,
)
from completion_v2 import layout, load_development, save, setup, sha


class NetworkPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, clips):
        import torch

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return (
                self.model(torch.from_numpy(np.asarray(clips)).cuda())
                .float()
                .softmax(-1)
                .cpu()
                .numpy()
            )


def folder(args, name):
    out = args.base / "development-controls" / name
    out.mkdir(exist_ok=False)
    (out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (out / "protocol.json").write_bytes(args.protocol.read_bytes())
    return out


def prepare(args, p, parent):
    out = folder(args, "adaptation-data")
    collection = args.base / "development-controls/collection"
    manifest = json.loads((collection / "manifest.json").read_text())
    env = make_env(args, out)
    rows, clips, labels = [], [], []
    try:
        for entry in manifest["entries"]:
            if entry["split"] != "train":
                continue
            name = entry["episode"]
            restore_episode(env, collection, name)
            facts = json.loads((collection / (name + "_truth.json")).read_text())
            with np.load(collection / (name + ".npz"), allow_pickle=False) as d:
                for i in range(10, len(d["rgb"]), p["new_frame_stride"]):
                    env.set_init_state(d["states"][i])
                    pixels = visibility(env)
                    observable = (
                        max(pixels) >= parent["nuisance"]["min_target_pixels_any_view"]
                    )
                    label = int(facts[i]["strict_complete"]) if observable else 2
                    clips.append(d["rgb"][np.array([i - 10, i - 5, i])])
                    labels.append(label)
                    rows.append(
                        dict(
                            episode=name,
                            control=i,
                            source="new_vla",
                            label=label,
                            physical_complete=facts[i]["strict_complete"],
                            target_pixels=pixels,
                        )
                    )
    finally:
        env.close()
    nuisance = args.base / "development-controls/nuisance"
    summary = json.loads((nuisance / "summary.json").read_text())
    effect = any(
        r["eligible"]
        and r["reference_decision"] == "complete"
        and r["intervention_decision"] != "complete"
        for r in summary["pairs"]
    )
    if effect:
        with np.load(nuisance / "clips.npz", allow_pickle=False) as d:
            for row in json.loads((nuisance / "rows.json").read_text()):
                if row.get("physically_valid") and row.get("visibility_proxy"):
                    clips.append(d["rgb"][row["clip_index"]])
                    labels.append(1)
                    rows.append(
                        dict(
                            episode=row["episode"],
                            control=row["source_control"],
                            source="visible_nuisance",
                            label=1,
                            nuisance_clip=row["clip_index"],
                        )
                    )
    np.savez(
        out / "additional_train.npz",
        rgb=np.stack(clips),
        labels=np.asarray(labels, dtype=np.int64),
    )
    save(out / "rows.json", rows)
    save(
        out / "manifest.json",
        dict(
            samples=len(labels),
            counts={str(i): labels.count(i) for i in (0, 1, 2)},
            nuisance_effect_observed=effect,
            nuisance_added=sum(r["source"] == "visible_nuisance" for r in rows),
            source_sha256=sha(__file__),
            protocol_sha256=sha(args.protocol),
            collection_sha256=sha(collection / "manifest.json"),
            nuisance_summary_sha256=sha(nuisance / "summary.json"),
            shard_sha256=sha(out / "additional_train.npz"),
        ),
    )
    print("ADAPTATION_DATA_READY", flush=True)


def gate_summary(episodes, auxiliary, parent):
    gate = parent["development_gate"]
    completed = sum(m["first_truth_control"] is not None for m in episodes)
    late = sum(
        m["confirmation_delay_s"] is not None
        and m["confirmation_delay_s"] > gate["max_confirmation_delay_s"]
        for m in episodes
    )
    delays = [
        m["confirmation_delay_s"]
        for m in episodes
        if m["confirmation_delay_s"] is not None
    ]
    result = dict(
        episodes=len(episodes),
        completed_episodes=completed,
        premature_stops=sum(m["premature_stop"] for m in episodes),
        missed_events=sum(m["missed_completed_event"] for m in episodes),
        post_stop_failures=sum(m["post_stop_stable"] is False for m in episodes),
        late_confirmations=late,
        max_delay_s=max(delays) if delays else None,
        auxiliary=auxiliary,
    )
    result["passed"] = (
        completed >= gate["minimum_completed_validation_episodes"]
        and result["premature_stops"]
        == result["missed_events"]
        == result["post_stop_failures"]
        == late
        == 0
        and auxiliary["physical_false_complete"] == auxiliary["blackout_confident"] == 0
    )
    return result


def selection_key(s):
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


def train(args, p, parent):
    import torch

    out = folder(args, "adaptation-training")
    torch.set_num_threads(8)
    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])
    random.seed(p["seed"])
    x, y, _ = load_development(args.base / "completion-v2/development", "train")
    additional = args.base / "development-controls/adaptation-data"
    assert (
        sha(additional / "additional_train.npz")
        == json.loads((additional / "manifest.json").read_text())["shard_sha256"]
    )
    with np.load(additional / "additional_train.npz", allow_pickle=False) as d:
        x, y = np.concatenate([x, d["rgb"]]), np.concatenate([y, d["labels"]])
    vx, vy, vrows = load_development(
        args.base / "completion-v2/development", "validation"
    )
    del vy
    aux_indices = [
        i
        for i, r in enumerate(vrows)
        if r["category"]
        in {
            "held",
            "above",
            "wrong_object",
            "outside",
            "recent_drop",
            "external_hidden",
            "wrist_hidden",
            "both_hidden",
        }
    ]
    aux_x = vx[aux_indices]
    aux_rows = [vrows[i] for i in aux_indices]
    del vx
    net = make_model().cuda()
    start_checkpoint = args.base / "completion-v2/training/best.pt"
    assert sha(start_checkpoint) == p["warm_start_sha256"]
    net.load_state_dict(
        torch.load(start_checkpoint, map_location="cpu", weights_only=False)["model"]
    )
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"]
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(
            len(y) / (3 * np.bincount(y)), dtype=torch.float32, device="cuda"
        )
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=p["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    collection = args.base / "development-controls/collection"
    entries = [
        e
        for e in json.loads((collection / "manifest.json").read_text())["entries"]
        if e["split"] == "validation"
    ]
    val_data = {}
    for e in entries:
        name = e["episode"]
        with np.load(collection / (name + ".npz"), allow_pickle=False) as d:
            val_data[name] = dict(
                rgb=d["rgb"],
                states=d["states"],
                truth=[
                    f["strict_complete"]
                    for f in json.loads(
                        (collection / (name + "_truth.json")).read_text()
                    )
                ],
            )
    env = make_env(args, out)
    post_cache = {}
    best, selected, steps = None, None, 0
    started = time.time()
    try:
        for epoch in range(1, p["epochs"] + 1):
            net.train()
            loss_sum = 0.0
            for rgb, label in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = criterion(net(rgb.cuda()), label.cuda())
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), 1, error_if_nonfinite=True
                )
                optimizer.step()
                steps += 1
                loss_sum += float(loss) * len(label)
                if steps == 1:
                    torch.save(dict(model=net.state_dict()), out / "smoke.pt")
                    rng = torch.get_rng_state()
                    other = make_model().cuda().eval()
                    other.load_state_dict(
                        torch.load(out / "smoke.pt", weights_only=False)["model"]
                    )
                    net.eval()
                    with torch.inference_mode():
                        torch.testing.assert_close(
                            net(rgb[:1].cuda()), other(rgb[:1].cuda()), rtol=0, atol=0
                        )
                    del other
                    torch.set_rng_state(rng)
                    net.train()
                    save(
                        out / "smoke_receipt.json",
                        dict(
                            status="PASS", backward_optimizer_steps=1, exact_reload=True
                        ),
                    )
            net.eval()
            predictor = NetworkPredictor(net)
            events, journals = [], {}
            for e in entries:
                name = e["episode"]
                d = val_data[name]
                rows = predictions(predictor, d["rgb"])
                journals[name] = rows
                confirmation = ContinuousConfirmation(
                    parent["temporal"]["confirmation_controls"]
                )
                decisions = [
                    (r["control"], confirmation.update(r["control"], r["decision"]))
                    for r in rows
                ]
                metrics = episode_metrics(d["truth"], decisions)
                claim = metrics["first_claim_control"]
                if claim is not None and (name, claim) not in post_cache:
                    restore_episode(env, collection, name)
                    post = stop_probe(env, d["states"], claim, parent)
                    post_cache[name, claim] = post
                    save(out / f"{name}_stop_{claim}.json", post)
                post = post_cache.get((name, claim))
                events.append(
                    dict(
                        episode=name,
                        **episode_metrics(
                            d["truth"],
                            decisions,
                            post_stop=[f["strict_complete"] for f in post]
                            if post
                            else None,
                        ),
                    )
                )
            aux_prob = np.concatenate(
                [predictor.predict(aux_x[i : i + 24]) for i in range(0, len(aux_x), 24)]
            )
            aux_decisions = [classify(pr) for pr in aux_prob]
            aux = dict(
                samples=len(aux_rows),
                physical_false_complete=sum(
                    r["label"] == 0 and d == "complete"
                    for r, d in zip(aux_rows, aux_decisions, strict=True)
                ),
                blackout_confident=sum(
                    r["label"] == 2 and d != "unknown"
                    for r, d in zip(aux_rows, aux_decisions, strict=True)
                ),
            )
            score = gate_summary(events, aux, parent)
            log = dict(
                epoch=epoch,
                steps=steps,
                train_loss=loss_sum / len(y),
                validation=score,
                events=events,
                elapsed_seconds=time.time() - started,
            )
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(log) + "\n")
            key = selection_key(score)
            if best is None or key < best:
                best, selected = key, log
                torch.save(
                    dict(
                        model=net.state_dict(),
                        epoch=epoch,
                        metrics=log,
                        protocol_sha256=sha(args.protocol),
                    ),
                    out / "best.pt",
                )
                save(out / "selected_predictions.json", journals)
                save(
                    out / "selected_auxiliary.json",
                    [
                        dict(r, probabilities=pr.tolist(), decision=d)
                        for r, pr, d in zip(
                            aux_rows, aux_prob, aux_decisions, strict=True
                        )
                    ],
                )
            torch.save(
                dict(
                    model=net.state_dict(),
                    optimizer=optimizer.state_dict(),
                    epoch=epoch,
                ),
                out / "last.pt",
            )
            print(
                json.dumps(dict(epoch=epoch, steps=steps, validation=score)), flush=True
            )
    finally:
        env.close()
    save(
        out / "frozen.json",
        dict(
            status="FROZEN_DEVELOPMENT_CANDIDATE",
            checkpoint_sha256=sha(out / "best.pt"),
            chosen_epoch=selected["epoch"],
            selected_validation=selected,
            train_samples=len(y),
            train_counts={str(i): int((y == i).sum()) for i in (0, 1, 2)},
            protocol_sha256=sha(args.protocol),
            parent_protocol_sha256=sha(args.parent_protocol),
            collection_sha256=sha(collection / "manifest.json"),
            additional_data_sha256=sha(additional / "manifest.json"),
            source_sha256=sha(__file__),
            frozen_unix=time.time(),
            independent_test_seen=False,
        ),
    )
    print("ADAPTATION_FROZEN", flush=True)


def live(args, p, parent):
    import torch
    from actionstream.lerobot_backend import LeRobotBackend

    out = folder(args, "live-development-validation")
    setup(out)
    torch.set_num_threads(8)
    training = args.base / "development-controls/adaptation-training"
    frozen = json.loads((training / "frozen.json").read_text())
    if not frozen["selected_validation"]["validation"]["passed"]:
        save(
            out / "summary.json",
            dict(
                status="NOT_RUN_DEVELOPMENT_GATE_FAILED",
                frozen_sha256=sha(training / "frozen.json"),
            ),
        )
        print("LIVE_NOT_RUN_DEVELOPMENT_GATE_FAILED", flush=True)
        return
    model = TemporalPredictor(training / "best.pt", frozen["checkpoint_sha256"])
    backend = LeRobotBackend(
        task_ids=[5],
        seed=p["seed"],
        suite="libero_object",
        episode_length=400,
        model_id=str(args.base / "assets/xvla"),
        model_revision="12e8783e996944f5c97e490d37d4c145484ed70a",
        device="cuda",
    )
    sub = backend._sub_env(5)
    sub.init_states = False
    sub._init_states = None
    reference = {
        e["seed"]: e
        for e in json.loads(
            (args.base / "development-controls/collection/manifest.json").read_text()
        )["entries"]
    }
    results = []
    try:
        for seed in parent["validation_seeds"]:
            observation, _, instruction = backend.reset_episode(
                task_id=5, seed=seed, initial_state_index=0
            )
            env = sub._env
            assert layout(env) == reference[seed]["initial_layout_sha256"]
            raw = env.set_init_state(env.sim.get_state().flatten())
            gate = ContinuousConfirmation(parent["temporal"]["confirmation_controls"])
            oracle = StableTruth(parent["truth"])
            images, facts, journal, actions, states = [], [], [], [], []
            claim = None
            for control in range(361):
                f = simulator_facts(env)
                f["strict_complete"] = oracle.update(control, f)
                facts.append(f)
                states.append(env.sim.get_state().flatten().copy())
                images.append(camera_rgb(raw))
                if control >= 10:
                    rgb = np.stack([images[control - k] for k in (10, 5, 0)])
                    pr = model.predict(rgb[None])[0]
                    decision = classify(pr)
                    confirmed = gate.update(control, decision)
                    journal.append(
                        dict(
                            control=control,
                            probabilities=pr.tolist(),
                            decision=decision,
                            confirmed=confirmed,
                        )
                    )
                    if confirmed:
                        claim = control
                        break
                if control == 360:
                    break
                if control < 300:
                    if control % 30 == 0:
                        chunk = backend.infer_action_chunk(
                            observation, instruction
                        ).actions
                    action = chunk[control % 30]
                else:
                    if control == 300:
                        env.robots[0].controller.use_delta = True
                        env.robots[0].controller.update(force=True)
                        env.robots[0].controller.reset_goal()
                    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
                raw, _, _, _ = env.step(action)
                observation = batch(sub._format_raw_obs(raw))
                actions.append(action.tolist())
            post = []
            if claim is not None:
                env.robots[0].controller.use_delta = True
                env.robots[0].controller.update(force=True)
                env.robots[0].controller.reset_goal()
                for i in range(parent["stop_probe"]["controls"]):
                    env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
                    f = simulator_facts(env)
                    f["strict_complete"] = oracle.update(claim + i + 1, f)
                    post.append(f)
            metrics = episode_metrics(
                [f["strict_complete"] for f in facts],
                [(r["control"], r["confirmed"]) for r in journal],
                post_stop=[f["strict_complete"] for f in post] if post else None,
            )
            name = f"validation_{seed}"
            save(out / (name + "_observations.json"), journal)
            save(
                out / (name + "_private_truth.json"),
                dict(controls=facts, post_stop=post),
            )
            save(out / (name + "_actions.json"), actions)
            np.savez_compressed(
                out / (name + ".npz"), rgb=np.stack(images), states=np.stack(states)
            )
            results.append(
                dict(
                    episode=name,
                    seed=seed,
                    instruction=instruction,
                    **metrics,
                    image_state_sha256=sha(out / (name + ".npz")),
                    observation_sha256=sha(out / (name + "_observations.json")),
                )
            )
            print(json.dumps(results[-1]), flush=True)
    finally:
        backend.close()
    summary = gate_summary(
        results, frozen["selected_validation"]["validation"]["auxiliary"], parent
    )
    save(
        out / "summary.json",
        dict(
            status="DEVELOPMENT_PASS" if summary["passed"] else "DEVELOPMENT_NO_GO",
            metrics=summary,
            episodes=results,
            checkpoint_sha256=frozen["checkpoint_sha256"],
            frozen_sha256=sha(training / "frozen.json"),
            source_sha256=sha(__file__),
            protocol_sha256=sha(args.protocol),
            independent_acceptance=False,
            limitation="same four development validation seeds; actual VLA/RGB stopping with two-second post-stop physics; no new Qwen evaluation",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "train", "live"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    parent = json.loads(args.parent_protocol.read_text())
    {"prepare": prepare, "train": train, "live": live}[args.phase](args, p, parent)


if __name__ == "__main__":
    main()
