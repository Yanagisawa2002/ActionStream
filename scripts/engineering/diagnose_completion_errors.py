"""Post-test simulator audit of errors; no checkpoint or threshold adaptation."""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from actionstream.libero_config import ensure_isolated_libero_config
from train_visual_completion import rebase_demo_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base
    output = base / "evaluation-v1/post_test_diagnostics"
    output.mkdir(exist_ok=False)
    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    ensure_isolated_libero_config(output / "libero-config")
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils.utils import postprocess_model_xml

    predictions = [
        json.loads(line)
        for line in (base / "evaluation-v1/heldout/predictions.jsonl")
        .read_text()
        .splitlines()
    ]
    errors = [r for r in predictions if r["decision"] == "complete" and not r["truth"]]
    rows = []
    with h5py.File(base / "data/tomato-xet.hdf5", "r") as source:
        bddl = str(source["data"].attrs["bddl_file_name"]).split("bddl_files/")[-1]
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / bddl),
            camera_heights=128,
            camera_widths=128,
        )
        try:
            for episode in sorted({r["episode"] for r in errors}):
                demo = source["data"][episode]
                env.reset()
                env.reset_from_xml_string(
                    rebase_demo_assets(
                        postprocess_model_xml(str(demo.attrs["model_file"]), {}),
                        get_libero_path("assets"),
                    )
                )
                target = env.env.objects_dict["tomato_sauce_1"]
                basket = env.env.objects_dict["basket_1"]
                indices = sorted(
                    {
                        max(0, min(len(demo["states"]) - 1, r["frame_index"] + offset))
                        for r in errors
                        if r["episode"] == episode
                        for offset in [-6, -3, 0, 3, 6]
                    }
                )
                for index in indices:
                    observation = env.set_init_state(demo["states"][index])
                    truth = bool(env.check_success())
                    grasp = bool(
                        env.env._check_grasp(
                            env.robots[0].gripper, target.contact_geoms
                        )
                    )
                    row = dict(
                        episode=episode,
                        frame_index=index,
                        native_success=truth,
                        gripper_contact_grasp=grasp,
                        target_position=env.sim.data.get_body_xpos(
                            target.root_body
                        ).tolist(),
                        basket_position=env.sim.data.get_body_xpos(
                            basket.root_body
                        ).tolist(),
                        target_basket_contact=bool(
                            env.env.check_contact(
                                target.contact_geoms, basket.contact_geoms
                            )
                        ),
                    )
                    row["prediction"] = next(
                        (
                            r
                            for r in predictions
                            if r["episode"] == episode and r["frame_index"] == index
                        ),
                        None,
                    )
                    rows.append(row)
                    if any(
                        r["episode"] == episode and r["frame_index"] == index
                        for r in errors
                    ):
                        for view, key in enumerate(
                            ["agentview_image", "robot0_eye_in_hand_image"]
                        ):
                            Image.fromarray(observation[key][::-1].copy()).save(
                                output / f"{episode}_{index}_view{view}.png"
                            )
        finally:
            env.close()
    summary = dict(
        scope="Post-test error diagnosis, selected after predictions were frozen; no retuning",
        states=rows,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Save one paired image for each controlled fixture category for human inspection.
    stress = json.loads((base / "evaluation-v1/stress/summary.json").read_text())
    with np.load(base / "evaluation-v1/stress/rgb_only.npz") as data:
        for index, row in enumerate(stress["cases"]):
            if row["episode"] != "demo_32":
                continue
            for view in range(2):
                Image.fromarray(data["rgb"][index, view]).save(
                    output / f"stress_{row['category']}_view{view}.png"
                )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
