"""Predeclared simulator fixtures, with private truth and RGB-only prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from actionstream.llm_vla.completion import FrozenCompletion, decision, native_rgb
from evaluate_visual_completion import digest, metrics


def prepare(args, protocol):
    import h5py
    from actionstream.libero_config import ensure_isolated_libero_config
    from train_visual_completion import rebase_demo_assets

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_bytes(args.protocol.read_bytes())
    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    ensure_isolated_libero_config(args.output / "libero-config")
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils.utils import postprocess_model_xml

    manifest = json.loads((args.data / "manifest.json").read_text())
    names = protocol["stress"]["source_episodes"]
    assert set(names) <= set(manifest["splits"]["train"])
    assert digest(args.hdf5) == manifest["source_sha256"]
    rows, images = [], []
    with h5py.File(args.hdf5, "r") as source:
        root = source["data"]
        bddl = str(root.attrs["bddl_file_name"]).split("bddl_files/")[-1]
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / bddl),
            camera_heights=360,
            camera_widths=360,
        )
        env.seed(protocol["stress"]["seed"])
        try:
            for name in names:
                demo = root[name]
                env.reset()
                env.reset_from_xml_string(
                    rebase_demo_assets(
                        postprocess_model_xml(str(demo.attrs["model_file"]), {}),
                        get_libero_path("assets"),
                    )
                )
                sim = env.sim
                target = env.env.objects_dict["tomato_sauce_1"]
                distractor = env.env.objects_dict["bbq_sauce_1"]
                target_joint, distractor_joint = target.joints[0], distractor.joints[0]

                def grasp():
                    return bool(
                        env.env._check_grasp(
                            env.robots[0].gripper, target.contact_geoms
                        )
                    )

                first_success = last_held = None
                for index, state in enumerate(demo["states"]):
                    env.set_state(state)
                    sim.forward()
                    if env.check_success():
                        first_success = index
                        break
                    if grasp():
                        last_held = index
                if first_success is None:
                    raise ValueError(f"No positive source fixture: {name}")

                def capture(
                    category, state_index, transform=None, mask=None, expected=None
                ):
                    observation = env.set_init_state(demo["states"][state_index])
                    if transform is not None:
                        transform()
                        observation = env.set_init_state(sim.get_state().flatten())
                    truth = bool(env.check_success())
                    if expected is not None and truth != expected:
                        raise ValueError(f"Invalid fixture truth: {name}/{category}")
                    rgb = native_rgb(
                        {
                            "image": observation["agentview_image"][None],
                            "image2": observation["robot0_eye_in_hand_image"][None],
                        }
                    )
                    if mask is not None:
                        rgb[mask] = 0
                    rows.append(
                        dict(
                            case_id=f"{name}/{category}",
                            episode=name,
                            category=category,
                            source_frame=state_index,
                            truth=truth,
                            gripper_contact_grasp=grasp(),
                            target_qpos=sim.data.get_joint_qpos(target_joint).tolist(),
                            distractor_qpos=sim.data.get_joint_qpos(
                                distractor_joint
                            ).tolist(),
                            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                            controlled_intervention=transform is not None
                            or mask is not None,
                        )
                    )
                    images.append(rgb)

                env.set_init_state(demo["states"][0])
                initial_z = float(sim.data.get_joint_qpos(target_joint)[2])
                capture("negative_control", 0, expected=False)
                capture("positive_control", first_success, expected=True)
                if last_held is None:
                    rows.append(
                        dict(
                            case_id=f"{name}/held",
                            episode=name,
                            category="held",
                            status="FIXTURE_UNAVAILABLE",
                        )
                    )
                else:
                    capture("held", last_held, expected=False)
                    assert rows[-1]["gripper_contact_grasp"]

                def above():
                    pose = sim.data.get_joint_qpos(target_joint).copy()
                    pose[2] += 0.20
                    sim.data.set_joint_qpos(target_joint, pose)

                def wrong():
                    one = sim.data.get_joint_qpos(target_joint).copy()
                    two = sim.data.get_joint_qpos(distractor_joint).copy()
                    one[:3], two[:3] = two[:3].copy(), one[:3].copy()
                    sim.data.set_joint_qpos(target_joint, one)
                    sim.data.set_joint_qpos(distractor_joint, two)

                def drop():
                    pose = sim.data.get_joint_qpos(target_joint).copy()
                    pose[0] += 0.30
                    pose[2] = initial_z
                    sim.data.set_joint_qpos(target_joint, pose)

                capture("above_basket", first_success, above, expected=False)
                capture("wrong_object", first_success, wrong, expected=False)
                for label, mask in [
                    ("occluded_external", [0]),
                    ("occluded_wrist", [1]),
                    ("occluded_both", [0, 1]),
                ]:
                    capture(label, first_success, mask=mask, expected=True)
                capture("dropped_after_placement", first_success, drop, expected=False)
                print(
                    json.dumps(
                        {
                            "prepared": name,
                            "first_success": first_success,
                            "last_held": last_held,
                        }
                    ),
                    flush=True,
                )
        finally:
            env.close()
    np.savez_compressed(args.output / "rgb_only.npz", rgb=np.stack(images))
    (args.output / "private_truth.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "build_receipt.json").write_text(
        json.dumps(
            dict(
                protocol_sha256=digest(args.protocol),
                builder_sha256=digest(__file__),
                rgb_sha256=digest(args.output / "rgb_only.npz"),
                private_truth_sha256=digest(args.output / "private_truth.json"),
                fixture_count=len(images),
            ),
            indent=2,
        )
        + "\n"
    )


def score(args):
    # The model receives only RGB. Join labels after all outputs have been saved.
    import torch

    torch.set_num_threads(8)
    output = args.output / "predictions.json"
    if output.exists():
        raise FileExistsError("Preserve the first frozen prediction record")
    model = FrozenCompletion(args.checkpoint)
    with np.load(args.output / "rgb_only.npz", allow_pickle=False) as data:
        probabilities = model.predict(data["rgb"])
    predictions = [
        dict(probability=float(p), decision=decision(float(p))) for p in probabilities
    ]
    output.write_text(json.dumps(predictions, indent=2) + "\n")
    truth = json.loads((args.output / "private_truth.json").read_text())
    available = [r for r in truth if r.get("status") != "FIXTURE_UNAVAILABLE"]
    assert len(available) == len(predictions)
    rows = [dict(a, **b) for a, b in zip(available, predictions, strict=True)]
    categories = {
        key: metrics([r for r in rows if r["category"] == key])
        for key in sorted({r["category"] for r in rows})
    }
    both = [r for r in rows if r["category"] == "occluded_both"]
    false = sum(r["decision"] == "complete" and not r["truth"] for r in rows)
    unobservable = sum(r["decision"] != "unknown" for r in both)
    positive_confirmed = sum(
        r["category"] == "positive_control" and r["decision"] == "complete"
        for r in rows
    )
    summary = dict(
        status="PASS"
        if false == 0
        and unobservable == 0
        and positive_confirmed > 0
        and len(available) == len(truth)
        else "FAIL",
        categories=categories,
        false_complete=false,
        both_hidden_confident_output=unobservable,
        positive_controls_confirmed=positive_confirmed,
        missing_fixtures=[r for r in truth if r.get("status") == "FIXTURE_UNAVAILABLE"],
        cases=rows,
        checkpoint_sha256=digest(args.checkpoint),
        protocol_sha256=digest(args.output / "protocol.json"),
        scorer_sha256=digest(__file__),
        predictions_sha256=digest(output),
        limitation="Controlled state displacements and synthetic camera outages; no claim of spontaneous policy drop or natural occluder coverage",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "score"])
    for name in ["hdf5", "data", "output", "checkpoint", "protocol"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args, json.loads(args.protocol.read_text()))
    else:
        score(args)


if __name__ == "__main__":
    main()
