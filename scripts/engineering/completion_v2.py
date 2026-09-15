"""V2 label repair, paired hard negatives, training and a fresh trajectory holdout."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from actionstream.completion_labels import StableTruth, simulator_facts
from actionstream.llm_vla.temporal_completion import (
    camera_rgb,
    classify,
    make_model,
    TemporalPredictor,
)
from actionstream.llm_vla.completion import native_rgb as old_rgb
from train_visual_completion import rebase_demo_assets


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def setup(output):
    from actionstream.libero_config import ensure_isolated_libero_config

    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    ensure_isolated_libero_config(output / "libero-config")


def facts_config(protocol):
    return {
        k: protocol["truth"][k]
        for k in ("window_steps", "linear_speed_max", "angular_speed_max")
    }


def hold_open(env, controls):
    robot = env.robots[0]
    robot.controller.use_delta = True
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    action = np.zeros(7)
    action[-1] = -1
    states = []
    for _ in range(controls):
        env.step(action)
        states.append(env.sim.get_state().flatten().copy())
    return states


def layout(env):
    return sha_bytes(
        np.asarray(
            [
                env.sim.data.get_body_xpos(obj.root_body)
                for name, obj in sorted(env.env.objects_dict.items())
            ]
        )
        .round(4)
        .tobytes()
    )


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sample_metrics(rows):
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    hidden = [r for r in rows if r["label"] == 2]
    return dict(
        samples=len(rows),
        positive=len(pos),
        negative=len(neg),
        unobservable=len(hidden),
        false_complete=sum(r["decision"] == "complete" for r in neg),
        true_complete=sum(r["decision"] == "complete" for r in pos),
        missed_complete=sum(r["decision"] != "complete" for r in pos),
        unknown=sum(r["decision"] == "unknown" for r in rows),
        hidden_confident=sum(r["decision"] != "unknown" for r in hidden),
    )


def build_episode(env, states, name, out, protocol, metadata):
    states = np.asarray(states)
    if not np.allclose(np.diff(states[:, 0]), 0.05, atol=1e-5):
        raise ValueError("Stability history requires 20 Hz simulator states")
    truth = StableTruth(facts_config(protocol))
    facts = []
    for i, state in enumerate(states):
        env.set_state(state)
        env.sim.forward()
        f = simulator_facts(env)
        f["strict_complete"] = truth.update(i, f)
        facts.append(f)
    indices = list(range(10, len(states), protocol["development"]["clip_stride"]))
    cache = {}

    def render(index, transform=None):
        observation = env.set_init_state(states[index])
        if transform is not None:
            target = env.env.objects_dict["tomato_sauce_1"].joints[0]
            other = env.env.objects_dict["bbq_sauce_1"].joints[0]
            one = env.sim.data.get_joint_qpos(target).copy()
            if transform == "wrong_object":
                two = env.sim.data.get_joint_qpos(other).copy()
                one[:3], two[:3] = two[:3].copy(), one[:3].copy()
                env.sim.data.set_joint_qpos(other, two)
            elif transform == "above":
                one[2] += 0.15
            elif transform == "outside":
                one[0] += 0.30
            else:
                raise ValueError(transform)
            env.sim.data.set_joint_qpos(target, one)
            observation = env.set_init_state(env.sim.get_state().flatten())
            if env.check_success():
                raise ValueError("Counterfactual remains inside the basket")
        modern = camera_rgb(observation)
        legacy = old_rgb(
            {
                "image": observation["agentview_image"][None],
                "image2": observation["robot0_eye_in_hand_image"][None],
            }
        )
        return modern, legacy

    def original(index):
        if index not in cache:
            cache[index] = render(index)
        return cache[index]

    clips, legacy_images, rows = [], [], []

    def append(index, category, images, legacy, label):
        clips.append(np.stack(images))
        legacy_images.append(legacy)
        rows.append(
            dict(
                episode=name,
                control_step=index,
                category=category,
                label=int(label),
                rgb_sha256=sha_bytes(clips[-1].tobytes()),
            )
        )

    for index in indices:
        category = (
            "stable"
            if facts[index]["strict_complete"]
            else "held"
            if facts[index]["finger_contact"]
            else "unsettled"
            if facts[index]["inside"]
            else "incomplete"
        )
        sequence = [original(index - offset)[0] for offset in (10, 5, 0)]
        append(
            index,
            category,
            sequence,
            original(index)[1],
            facts[index]["strict_complete"],
        )
    positives = [i for i in indices if facts[i]["strict_complete"]]
    selected = (
        sorted(
            set(
                positives[int(i)]
                for i in np.linspace(0, len(positives) - 1, min(len(positives), 5))
            )
        )
        if positives
        else []
    )
    for index in selected:
        for kind in ("above", "wrong_object", "outside", "recent_drop"):
            images = []
            for offset in (10, 5, 0):
                if kind == "recent_drop" and offset > 0:
                    rgb, legacy = original(index - offset)
                else:
                    rgb, legacy = render(
                        index - offset, "outside" if kind == "recent_drop" else kind
                    )
                images.append(rgb)
            append(index, kind, images, legacy, 0)
        for key, mask in (
            ("external_hidden", [0]),
            ("wrist_hidden", [1]),
            ("both_hidden", [0, 1]),
        ):
            images = np.stack(
                [original(index - offset)[0] for offset in (10, 5, 0)]
            ).copy()
            legacy = original(index)[1].copy()
            images[:, mask] = 0
            legacy[mask] = 0
            append(index, key, images, legacy, 2)
    np.savez(
        out / (name + ".npz"),
        rgb=np.stack(clips),
        legacy_rgb=np.stack(legacy_images),
        labels=np.array([r["label"] for r in rows], dtype=np.int64),
    )
    np.savez_compressed(out / (name + "_states.npz"), states=states)
    save(
        out / (name + "_truth.json"),
        dict(metadata=metadata, per_control=facts, samples=rows),
    )
    receipt = dict(
        episode=name,
        samples=len(rows),
        counts={str(k): sum(r["label"] == k for r in rows) for k in (0, 1, 2)},
        label_changes_from_containment=sum(
            f["inside"] and not f["strict_complete"] for f in facts
        ),
        shard_sha256=sha(out / (name + ".npz")),
        truth_sha256=sha(out / (name + "_truth.json")),
        states_sha256=sha(out / (name + "_states.npz")),
        metadata=metadata,
    )
    print(json.dumps(receipt), flush=True)
    return receipt


def prepare(args, p):
    import h5py

    out = (
        args.base
        / "completion-v2"
        / ("development-smoke" if args.smoke else "development")
    )
    out.mkdir(parents=True, exist_ok=False)
    setup(out)
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils.utils import postprocess_model_xml

    old = json.loads((args.base / "data/rendered-v1/manifest.json").read_text())
    assert sha(args.base / "data/tomato-xet.hdf5") == old["source_sha256"]
    entries = []
    splits = {
        k: old["splits"][k][:1] if args.smoke else old["splits"][k]
        for k in ("train", "validation")
    }
    assert not (set(splits["train"]) | set(splits["validation"])) & set(
        old["splits"]["test"]
    )
    with h5py.File(args.base / "data/tomato-xet.hdf5") as h:
        root = h["data"]
        bddl = str(root.attrs["bddl_file_name"]).split("bddl_files/")[-1]
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / bddl),
            camera_heights=360,
            camera_widths=360,
        )
        env.seed(p["seed"])
        try:
            for split, names in splits.items():
                for name in names:
                    demo = root[name]
                    env.reset()
                    env.reset_from_xml_string(
                        rebase_demo_assets(
                            postprocess_model_xml(str(demo.attrs["model_file"]), {}),
                            get_libero_path("assets"),
                        )
                    )
                    states = list(demo["states"][:])
                    env.set_init_state(states[-1])
                    states += hold_open(
                        env, p["development"]["settle_extension_controls"]
                    )
                    entries.append(
                        build_episode(
                            env,
                            states,
                            name,
                            out,
                            p,
                            dict(
                                split=split,
                                source="original training/validation demo plus uniform open-gripper settling extension",
                                source_frame_count=len(demo["states"]),
                            ),
                        )
                    )
        finally:
            env.close()
    save(
        out / "manifest.json",
        dict(
            protocol_sha256=sha(args.protocol),
            builder_sha256=sha(__file__),
            splits=splits,
            excluded_old_test=old["splits"]["test"],
            entries=entries,
        ),
    )
    print("DEVELOPMENT_DATA_COMPLETE", flush=True)


def load_development(directory, split):
    manifest = json.loads((directory / "manifest.json").read_text())
    entries = {e["episode"]: e for e in manifest["entries"]}
    clips = []
    labels = []
    rows = []
    for name in manifest["splits"][split]:
        path = directory / (name + ".npz")
        assert sha(path) == entries[name]["shard_sha256"]
        with np.load(path, allow_pickle=False) as d:
            clips.append(d["rgb"])
            labels.append(d["labels"])
        rows += json.loads((directory / (name + "_truth.json")).read_text())["samples"]
    return np.concatenate(clips), np.concatenate(labels), rows


def train(args, p):
    import torch

    d = args.base / "completion-v2/development"
    out = args.base / "completion-v2" / ("training-smoke" if args.smoke else "training")
    out.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])
    random.seed(p["seed"])
    x, y, _ = load_development(d, "train")
    vx, vy, vrows = load_development(d, "validation")
    assert set(y) == {0, 1, 2} and set(vy) == {0, 1, 2}
    net = make_model(True).cuda()
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=p["development"]["learning_rate"], weight_decay=0.01
    )
    weights = torch.tensor(
        len(y) / (3 * np.bincount(y)), dtype=torch.float32, device="cuda"
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=p["development"]["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    best = None
    step = 0
    start = time.time()
    for epoch in range(1, (1 if args.smoke else p["development"]["epochs"]) + 1):
        net.train()
        loss_total = 0.0
        n = 0
        for rgb, label in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(net(rgb.cuda()), label.cuda())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            loss_total += float(loss) * len(label)
            n += len(label)
            step += 1
            if args.smoke:
                break
        net.eval()
        prob = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(vx), 24):
                prob.append(
                    net(torch.from_numpy(vx[i : i + 24]).cuda())
                    .float()
                    .softmax(-1)
                    .cpu()
                    .numpy()
                )
        prob = np.concatenate(prob)
        ce = float(-np.log(np.maximum(prob[np.arange(len(vy)), vy], 1e-9)).mean())
        rows = [
            dict(row, decision=classify(pr))
            for row, pr in zip(vrows, prob, strict=True)
        ]
        metrics = sample_metrics(rows)
        key = (metrics["false_complete"], metrics["missed_complete"], ce)
        log = dict(
            epoch=epoch,
            steps=step,
            train_loss=loss_total / n,
            validation_cross_entropy=ce,
            validation=metrics,
            elapsed_seconds=time.time() - start,
        )
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(log) + "\n")
        print(json.dumps(log), flush=True)
        checkpoint = dict(
            model=net.state_dict(),
            epoch=epoch,
            metrics=log,
            protocol_sha256=sha(args.protocol),
            development_manifest_sha256=sha(d / "manifest.json"),
        )
        if best is None or key < best:
            best = key
            torch.save(checkpoint, out / "best.pt")
        torch.save(checkpoint, out / "last.pt")
        if args.smoke:
            copy = make_model().cuda().eval()
            copy.load_state_dict(
                torch.load(out / "last.pt", weights_only=False)["model"]
            )
            with torch.inference_mode():
                torch.testing.assert_close(
                    net(torch.from_numpy(vx[:1]).cuda()),
                    copy(torch.from_numpy(vx[:1]).cuda()),
                    rtol=0,
                    atol=0,
                )
    selected = torch.load(out / "best.pt", weights_only=False, map_location="cpu")
    save(
        out / "frozen.json",
        dict(
            status="SMOKE_PASS" if args.smoke else "FROZEN",
            checkpoint_sha256=sha(out / "best.pt"),
            chosen_epoch=selected["epoch"],
            validation=selected["metrics"],
            protocol_sha256=sha(args.protocol),
            development_manifest_sha256=sha(d / "manifest.json"),
            frozen_unix=time.time(),
            test_seen=False,
            thresholds=p["thresholds"],
            trainer_sha256=sha(__file__),
        ),
    )
    print("TRAINING_SMOKE_PASS" if args.smoke else "CANDIDATE_FROZEN", flush=True)


def fresh(args, p):
    out = args.base / "completion-v2/fresh_holdout"
    freeze = args.base / "completion-v2/training/frozen.json"
    assert json.loads(freeze.read_text())["status"] == "FROZEN"
    out.mkdir(exist_ok=False)
    setup(out)
    from actionstream.lerobot_backend import LeRobotBackend
    import torch
    import h5py

    torch.set_num_threads(8)
    b = LeRobotBackend(
        task_ids=[5],
        seed=p["seed"],
        suite="libero_object",
        episode_length=400,
        model_id=str(args.base / "assets/xvla"),
        model_revision="12e8783e996944f5c97e490d37d4c145484ed70a",
        device="cuda",
    )
    sub = b._sub_env(5)
    sub.init_states = False
    sub._init_states = None
    entries = []
    layouts = set()
    try:
        # Reject exact old initial object layouts; no old test images/labels enter fitting.
        sub._ensure_env()
        env = sub._env
        with h5py.File(args.base / "data/tomato-xet.hdf5") as h:
            for name in h["data"]:
                env.set_init_state(h["data"][name]["states"][0])
                layouts.add(layout(env))
        for n, seed in enumerate(p["fresh_holdout"]["seeds"]):
            observation, _, instruction = b.reset_episode(
                task_id=5, seed=seed, initial_state_index=0
            )
            env = sub._env
            signature = layout(env)
            if signature in layouts:
                raise ValueError("Fresh layout duplicates old or fresh scene")
            layouts.add(signature)
            states = [env.sim.get_state().flatten().copy()]
            actions = []
            for control in range(p["fresh_holdout"]["policy_controls"]):
                if control % 30 == 0:
                    chunk = b.infer_action_chunk(observation, instruction).actions
                action = chunk[control % 30]
                actions.append(action.tolist())
                raw, _, _, _ = env.step(action)

                def batch(v):
                    return (
                        {k: batch(x) for k, x in v.items()}
                        if isinstance(v, dict)
                        else np.asarray(v)[None]
                    )

                observation = batch(sub._format_raw_obs(raw))
                states.append(env.sim.get_state().flatten().copy())
            states += hold_open(env, p["fresh_holdout"]["settle_controls"])
            name = f"fresh_{n:02d}"
            save(out / (name + "_actions.json"), actions)
            entries.append(
                build_episode(
                    env,
                    states,
                    name,
                    out,
                    p,
                    dict(
                        split="fresh_holdout",
                        seed=seed,
                        initial_layout_sha256=signature,
                        init_states=False,
                        policy_controls=len(actions),
                        settle_controls=p["fresh_holdout"]["settle_controls"],
                        native_termination="record-only during fixed-horizon offline collection; no autoreset",
                        action_sha256=sha(out / (name + "_actions.json")),
                    ),
                )
            )
    finally:
        b.close()
    save(
        out / "manifest.json",
        dict(
            protocol_sha256=sha(args.protocol),
            builder_sha256=sha(__file__),
            freeze_sha256=sha(freeze),
            generated_unix=time.time(),
            entries=entries,
            model_predictions_used=False,
            mode="fresh procedural VLA trajectories plus paired state interventions; offline shadow evaluation",
        ),
    )
    print("FRESH_HOLDOUT_GENERATED", flush=True)


def evaluate(args, p):
    import torch
    from actionstream.llm_vla.completion import FrozenCompletion, decision

    torch.set_num_threads(8)
    d = args.base / "completion-v2/fresh_holdout"
    run = args.base / "completion-v2/training"
    frozen = json.loads((run / "frozen.json").read_text())
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["freeze_sha256"] == sha(run / "frozen.json")
    assert (
        manifest["generated_unix"] > frozen["frozen_unix"] and not frozen["test_seen"]
    )
    out = args.base / "completion-v2/evaluation"
    out.mkdir(exist_ok=False)
    candidate = TemporalPredictor(run / "best.pt", frozen["checkpoint_sha256"])
    baseline = FrozenCompletion(args.base / "runs/completion-v1/best.pt")
    all_rows = []
    old_rows = []
    episodes = {}
    with (out / "predictions.jsonl").open("x") as journal:
        for entry in manifest["entries"]:
            name = entry["episode"]
            assert sha(d / (name + ".npz")) == entry["shard_sha256"]
            truth = json.loads((d / (name + "_truth.json")).read_text())["samples"]
            with np.load(d / (name + ".npz"), allow_pickle=False) as data:
                for start in range(0, len(truth), 24):
                    probs = candidate.predict(data["rgb"][start : start + 24])
                    old = baseline.predict(data["legacy_rgb"][start : start + 24])
                    for i, (pr, legacy) in enumerate(zip(probs, old, strict=True)):
                        row = dict(
                            truth[start + i],
                            probabilities=pr.tolist(),
                            decision=classify(pr),
                            baseline_probability=float(legacy),
                            baseline_decision=decision(float(legacy)),
                        )
                        all_rows.append(row)
                        old_rows.append(dict(row, decision=row["baseline_decision"]))
                        journal.write(json.dumps(row) + "\n")
                journal.flush()
            episodes[name] = sample_metrics(
                [r for r in all_rows if r["episode"] == name]
            )
            print(json.dumps(dict(episode=name, **episodes[name])), flush=True)
    modern = sample_metrics(all_rows)
    old = sample_metrics(old_rows)
    recall = modern["true_complete"] / modern["positive"] if modern["positive"] else 0
    passed = (
        modern["false_complete"] == 0
        and recall >= p["acceptance"]["minimum_complete_recall"]
        and modern["hidden_confident"] == 0
        and modern["negative"] > 0
        and modern["positive"] > 0
    )
    categories = {
        k: dict(
            candidate=sample_metrics([r for r in all_rows if r["category"] == k]),
            baseline=sample_metrics([r for r in old_rows if r["category"] == k]),
        )
        for k in sorted({r["category"] for r in all_rows})
    }
    base_categories = {"stable", "held", "unsettled", "incomplete"}
    natural = [r for r in all_rows if r["category"] in base_categories]
    save(
        out / "summary.json",
        dict(
            status="PASS" if passed else "NO_GO",
            candidate=modern,
            baseline=old,
            natural_trajectory=sample_metrics(natural),
            categories=categories,
            episodes=episodes,
            complete_recall=recall,
            false_complete_relative_reduction=(
                old["false_complete"] - modern["false_complete"]
            )
            / old["false_complete"]
            if old["false_complete"]
            else None,
            checkpoint_sha256=frozen["checkpoint_sha256"],
            chosen_epoch=frozen["chosen_epoch"],
            protocol_sha256=sha(args.protocol),
            holdout_manifest_sha256=sha(d / "manifest.json"),
            predictions_sha256=sha(out / "predictions.jsonl"),
            evaluator_sha256=sha(__file__),
            test_used_for_selection=False,
            limitation="20 procedural simulated VLA rollouts and controlled failure/camera interventions; shadow predictions, not a live candidate-controlled closed loop",
        ),
    )
    print(
        json.dumps(
            dict(status="PASS" if passed else "NO_GO", candidate=modern, baseline=old)
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "train", "fresh", "evaluate"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    {"prepare": prepare, "train": train, "fresh": fresh, "evaluate": evaluate}[
        args.phase
    ](args, p)


if __name__ == "__main__":
    main()
