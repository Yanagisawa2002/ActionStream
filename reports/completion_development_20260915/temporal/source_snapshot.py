"""Fresh development collection and paired controls for the frozen v2 verifier."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np

from actionstream.completion_labels import StableTruth, instantaneous, simulator_facts
from actionstream.llm_vla.confirmation import ContinuousConfirmation, episode_metrics
from actionstream.llm_vla.temporal_completion import (
    camera_rgb,
    classify,
    TemporalPredictor,
)
from completion_v2 import hold_open, layout, setup, sha, save


def start_folder(path, protocol):
    path.mkdir(parents=True, exist_ok=False)
    (path / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (path / "protocol.json").write_bytes(protocol.read_bytes())


def batch(value):
    return (
        {k: batch(v) for k, v in value.items()}
        if isinstance(value, dict)
        else np.asarray(value)[None]
    )


def collect(args, p):
    import h5py
    import torch
    from actionstream.lerobot_backend import LeRobotBackend

    out = args.base / "development-controls/collection"
    start_folder(out, args.protocol)
    setup(out)
    torch.set_num_threads(8)
    backend = LeRobotBackend(
        task_ids=[5],
        seed=p["train_seeds"][0],
        suite="libero_object",
        episode_length=400,
        model_id=str(args.base / "assets/xvla"),
        model_revision="12e8783e996944f5c97e490d37d4c145484ed70a",
        device="cuda",
    )
    sub = backend._sub_env(5)
    sub.init_states = False
    sub._init_states = None
    entries = []
    used_layouts = {
        e["metadata"]["initial_layout_sha256"]
        for e in json.loads(
            (args.base / "completion-v2/fresh_holdout/manifest.json").read_text()
        )["entries"]
    }
    try:
        sub._ensure_env()
        with h5py.File(args.base / "data/tomato-xet.hdf5") as h:
            for name in h["data"]:
                sub._env.set_init_state(h["data"][name]["states"][0])
                used_layouts.add(layout(sub._env))
        for split in ("train", "validation"):
            for seed in p[split + "_seeds"]:
                name = f"{split}_{seed}"
                observation, _, instruction = backend.reset_episode(
                    task_id=5, seed=seed, initial_state_index=0
                )
                env = sub._env
                assert env.robots[0].controller.use_delta is False
                signature = layout(env)
                if signature in used_layouts:
                    raise ValueError("Duplicate initial object layout")
                used_layouts.add(signature)
                save(
                    out / (name + "_metadata.json"),
                    dict(
                        seed=seed,
                        split=split,
                        instruction=instruction,
                        initial_layout_sha256=signature,
                        init_states=False,
                    ),
                )
                (out / (name + ".xml")).write_text(env.env.model.get_xml())
                truth = StableTruth(p["truth"])
                states, images, facts, actions = [], [], [], []

                def record(raw):
                    f = simulator_facts(env)
                    f["strict_complete"] = truth.update(len(states), f)
                    states.append(env.sim.get_state().flatten().copy())
                    images.append(camera_rgb(raw))
                    facts.append(f)

                record(env.set_init_state(env.sim.get_state().flatten()))
                for control in range(p["collection"]["policy_controls"]):
                    if control % 30 == 0:
                        chunk = backend.infer_action_chunk(
                            observation, instruction
                        ).actions
                    action = chunk[control % 30]
                    raw, _, _, _ = env.step(action)
                    observation = batch(sub._format_raw_obs(raw))
                    actions.append(action.tolist())
                    record(raw)
                tail = hold_open(env, p["collection"]["settle_controls"])
                for state in tail:
                    record(env.set_init_state(state))
                array = np.stack(states)
                assert np.allclose(np.diff(array[:, 0]), 0.05, atol=1e-5)
                np.savez(out / (name + ".npz"), states=array, rgb=np.stack(images))
                save(out / (name + "_truth.json"), facts)
                save(out / (name + "_actions.json"), actions)
                entry = dict(
                    episode=name,
                    split=split,
                    seed=seed,
                    initial_layout_sha256=signature,
                    controls=len(states) - 1,
                    strict_positive_controls=sum(f["strict_complete"] for f in facts),
                    shard_sha256=sha(out / (name + ".npz")),
                    truth_sha256=sha(out / (name + "_truth.json")),
                    xml_sha256=sha(out / (name + ".xml")),
                    actions_sha256=sha(out / (name + "_actions.json")),
                )
                entries.append(entry)
                save(out / "progress.json", entries)
                print(json.dumps(entry), flush=True)
    finally:
        backend.close()
    save(
        out / "manifest.json",
        dict(
            entries=entries,
            protocol_sha256=sha(args.protocol),
            source_sha256=sha(__file__),
            model_predictions_used=False,
            mode="new procedural development trajectories",
            created_unix=time.time(),
        ),
    )


def make_env(args, out):
    setup(out)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = benchmark.get_benchmark_dict()["libero_object"]().get_task(5)
    return OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ),
        camera_heights=360,
        camera_widths=360,
    )


def restore_episode(env, collection, name):
    env.reset()
    env.reset_from_xml_string((collection / (name + ".xml")).read_text())


def visibility(env):
    """Evaluator-only segmentation; int32 decoding avoids NumPy 2 uint8 overflow."""
    import mujoco

    obj = env.env.objects_dict["tomato_sauce_1"]
    gids = [env.sim.model.geom_name2id(g) for g in obj.visual_geoms + obj.contact_geoms]
    ctx = env.sim._render_context_offscreen
    result = []
    for camera in ("agentview", "robot0_eye_in_hand"):
        ctx.gl_ctx.make_current()
        ctx.render(
            192, 192, camera_id=env.sim.model.camera_name2id(camera), segmentation=True
        )
        raw = np.empty((192, 192, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(
            rgb=raw, depth=None, viewport=mujoco.MjrRect(0, 0, 192, 192), con=ctx.con
        )
        colors = raw.astype(np.int32)
        ids = colors[:, :, 0] + colors[:, :, 1] * 256 + colors[:, :, 2] * 65536
        wanted = [
            g.segid + 1
            for g in list(ctx.scn.geoms)[: ctx.scn.ngeom]
            if g.objtype == int(mujoco.mjtObj.mjOBJ_GEOM)
            and g.objid in gids
            and g.segid >= 0
        ]
        result.append(int(np.isin(ids, wanted).sum()))
    return result


def nuisance(args, p):
    out = args.base / "development-controls/nuisance"
    start_folder(out, args.protocol)
    collection = args.base / "development-controls/collection"
    entries = json.loads((collection / "manifest.json").read_text())["entries"]
    model = TemporalPredictor(
        args.base / "completion-v2/training/best.pt", p["frozen_v2_sha256"]
    )
    env = make_env(args, out)
    rows, clips = [], []
    try:
        for entry in [e for e in entries if e["split"] == "train"][
            : p["nuisance"]["train_episodes"]
        ]:
            name = entry["episode"]
            facts = json.loads((collection / (name + "_truth.json")).read_text())
            positives = [i for i, f in enumerate(facts) if f["strict_complete"]]
            if not positives:
                rows.append(dict(episode=name, skipped="no strictly completed state"))
                continue
            index = positives[min(20, len(positives) - 1)]
            restore_episode(env, collection, name)
            with np.load(collection / (name + ".npz"), allow_pickle=False) as d:
                states = d["states"]
                base = states[index].copy()
                robot = env.robots[0]
                arm_ids = robot._ref_joint_pos_indexes
                grip_ids = robot._ref_gripper_joint_pos_indexes
                qposes = {}
                for key, i in (("reference", index), ("initial", 0), ("late", 300)):
                    env.set_init_state(states[i])
                    qposes[key] = env.sim.data.qpos[arm_ids].copy()
                env.set_init_state(base)
                object_poses = {
                    key: env.sim.data.get_joint_qpos(obj.joints[0]).copy()
                    for key, obj in env.env.objects_dict.items()
                }
                cam_id = env.sim.model.camera_name2id("agentview")
                cam_position = env.sim.model.cam_pos[cam_id].copy()
                for arm, gripper, dx in itertools.product(
                    p["nuisance"]["arm_poses"],
                    p["nuisance"]["gripper"],
                    p["nuisance"]["external_camera_dx_m"],
                ):
                    env.sim.model.cam_pos[cam_id] = cam_position
                    env.set_init_state(base)
                    env.sim.data.qpos[arm_ids] = qposes[arm]
                    env.sim.data.qpos[grip_ids] = (
                        [0.04, -0.04] if gripper == "open" else [0.0, 0.0]
                    )
                    env.sim.data.qvel[:] = 0
                    env.sim.model.cam_pos[cam_id, 0] += dx
                    raw = env.set_init_state(env.sim.get_state().flatten())
                    unchanged = all(
                        np.array_equal(
                            env.sim.data.get_joint_qpos(
                                env.env.objects_dict[key].joints[0]
                            ),
                            pos,
                        )
                        for key, pos in object_poses.items()
                    )
                    f = simulator_facts(env)
                    pixels = visibility(env)
                    valid = unchanged and instantaneous(f, p["truth"])
                    observable = (
                        max(pixels) >= p["nuisance"]["min_target_pixels_any_view"]
                    )
                    rgb = np.stack([camera_rgb(raw)] * 3)
                    pr = model.predict(rgb[None])[0]
                    row = dict(
                        episode=name,
                        source_control=index,
                        arm=arm,
                        gripper=gripper,
                        camera_dx=dx,
                        objects_unchanged=unchanged,
                        physically_valid=bool(valid),
                        visibility_proxy=bool(observable),
                        target_pixels=pixels,
                        facts=f,
                        probabilities=pr.tolist(),
                        decision=classify(pr),
                        clip_index=len(clips),
                    )
                    rows.append(row)
                    clips.append(rgb)
                env.sim.model.cam_pos[cam_id] = cam_position
            print(json.dumps(dict(episode=name, variants=18)), flush=True)
    finally:
        env.close()
    np.savez(out / "clips.npz", rgb=np.stack(clips))
    save(out / "rows.json", rows)
    paired = []
    for name in sorted({r["episode"] for r in rows}):
        subset = [r for r in rows if r["episode"] == name and "arm" in r]
        if not subset:
            continue
        reference = next(
            r
            for r in subset
            if r["arm"] == "reference"
            and r["gripper"] == "open"
            and r["camera_dx"] == 0
        )
        for row in subset:
            factors = [
                row["arm"] != "reference",
                row["gripper"] != "open",
                row["camera_dx"] != 0,
            ]
            if sum(factors) != 1:
                continue
            eligible = all(
                r["physically_valid"] and r["visibility_proxy"]
                for r in (reference, row)
            )
            paired.append(
                dict(
                    episode=name,
                    factor=("arm", "gripper", "camera")[factors.index(True)],
                    arm=row["arm"],
                    gripper=row["gripper"],
                    camera_dx=row["camera_dx"],
                    eligible=eligible,
                    reference_probability=reference["probabilities"][1],
                    intervention_probability=row["probabilities"][1],
                    delta=row["probabilities"][1] - reference["probabilities"][1],
                    reference_decision=reference["decision"],
                    intervention_decision=row["decision"],
                )
            )
    save(
        out / "summary.json",
        dict(
            pairs=paired,
            variants=len(clips),
            valid_visible=sum(
                r.get("physically_valid", False) and r.get("visibility_proxy", False)
                for r in rows
            ),
            source_sha256=sha(__file__),
            protocol_sha256=sha(args.protocol),
            checkpoint_sha256=p["frozen_v2_sha256"],
            limitation="static interventions; visibility is evaluator-only proxy",
        ),
    )


def predictions(model, rgb):
    indices = list(range(10, len(rgb)))
    probs = []
    for start in range(0, len(indices), 24):
        clips = np.stack(
            [rgb[np.array([i - 10, i - 5, i])] for i in indices[start : start + 24]]
        )
        probs.extend(model.predict(clips).tolist())
    return [
        dict(control=i, probabilities=pr, decision=classify(pr))
        for i, pr in zip(indices, probs, strict=True)
    ]


def stop_probe(env, states, claim, p):
    if claim is None:
        return None
    env.set_init_state(states[claim])
    robot = env.robots[0]
    robot.controller.use_delta = True
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    truth = StableTruth(p["truth"])
    for i in range(max(0, claim - 10), claim + 1):
        env.set_state(states[i])
        env.sim.forward()
        truth.update(i, simulator_facts(env))
    env.set_init_state(states[claim])
    action = np.zeros(7)
    action[-1] = -1
    post = []
    for i in range(p["stop_probe"]["controls"]):
        env.step(action)
        f = simulator_facts(env)
        f["strict_complete"] = truth.update(claim + i + 1, f)
        post.append(f)
    return post


def temporal(args, p):
    out = args.base / "development-controls/temporal"
    start_folder(out, args.protocol)
    collection = args.base / "development-controls/collection"
    entries = json.loads((collection / "manifest.json").read_text())["entries"]
    model = TemporalPredictor(
        args.base / "completion-v2/training/best.pt", p["frozen_v2_sha256"]
    )
    env = make_env(args, out)
    results = []
    try:
        for entry in entries:
            name = entry["episode"]
            restore_episode(env, collection, name)
            facts = json.loads((collection / (name + "_truth.json")).read_text())
            truth = [f["strict_complete"] for f in facts]
            with np.load(collection / (name + ".npz"), allow_pickle=False) as d:
                states, rgb = d["states"], d["rgb"]
                rows = predictions(model, rgb)
                save(out / (name + "_predictions.json"), rows)
                modes = {}
                for mode in p["temporal"]["modes"]:
                    gate = ContinuousConfirmation(
                        p["temporal"]["confirmation_controls"]
                    )
                    decisions = []
                    for row in rows:
                        i = row["control"]
                        if mode == "sparse_4hz" and i % 5:
                            continue
                        claim = (
                            gate.update(i, row["decision"])
                            if mode == "dense_20hz_confirm_0.5s"
                            else row["decision"] == "complete"
                        )
                        decisions.append((i, claim))
                    m = episode_metrics(truth, decisions)
                    post = stop_probe(env, states, m["first_claim_control"], p)
                    if post is not None:
                        save(out / (name + "_" + mode + "_post_stop.json"), post)
                    modes[mode] = episode_metrics(
                        truth,
                        decisions,
                        post_stop=[f["strict_complete"] for f in post]
                        if post
                        else None,
                    )
                results.append(dict(episode=name, split=entry["split"], modes=modes))
            print(json.dumps(results[-1]), flush=True)
    finally:
        env.close()
    summary = {}
    for mode in p["temporal"]["modes"]:
        subset = [r["modes"][mode] for r in results if r["split"] == "train"]
        delays = [
            m["confirmation_delay_s"]
            for m in subset
            if m["confirmation_delay_s"] is not None
        ]
        summary[mode] = dict(
            episodes=len(subset),
            premature_stops=sum(m["premature_stop"] for m in subset),
            missed_completed_events=sum(m["missed_completed_event"] for m in subset),
            max_confirmation_delay_s=max(delays) if delays else None,
            post_stop_failures=sum(m["post_stop_stable"] is False for m in subset),
        )
    save(
        out / "summary.json",
        dict(
            results=results,
            train_comparison=summary,
            source_sha256=sha(__file__),
            protocol_sha256=sha(args.protocol),
            checkpoint_sha256=p["frozen_v2_sha256"],
            limitation="same three-frame model at different decision cadences; executed simulator stop branches, not independent live acceptance",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["collect", "nuisance", "temporal"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    {"collect": collect, "nuisance": nuisance, "temporal": temporal}[args.phase](
        args, p
    )


if __name__ == "__main__":
    main()
