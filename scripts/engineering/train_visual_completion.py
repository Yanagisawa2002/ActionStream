"""Finite, dual-camera completion pilot; simulator truth is offline supervision only.

LIBERO demonstration rewards mark the final frame, so they are NOT completion
labels. Render each sampled simulator state and query success at that same state.
This pilot is scoped to tomato sauce -> basket and does not enable runtime PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

import numpy as np


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def episode_splits(names, seed):
    names = sorted(names)
    if len(names) < 10 or len(set(names)) != len(names):
        raise ValueError("Need at least ten distinct episodes")
    random.Random(seed).shuffle(names)
    first, second = int(len(names) * 0.6), int(len(names) * 0.8)
    return {
        "train": names[:first],
        "validation": names[first:second],
        "test": names[second:],
    }


def prepare(args):
    import h5py

    # Select canonical package paths before importing LIBERO (which can prompt).
    from actionstream.libero_config import ensure_isolated_libero_config

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    ensure_isolated_libero_config(args.output / "libero-config")
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils.utils import postprocess_model_xml

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Use a fresh output; preserve existing data and split")
    source_sha256 = sha256(args.hdf5)
    if (
        source_sha256
        != "43c52bdaa78b4c6aff3afb54d932d6a0e065f5c30c8e30a6c17f78749d487ea9"
    ):
        raise ValueError("Input does not match the pinned official demonstration file")
    with h5py.File(args.hdf5, "r") as source:
        root = source["data"]
        bddl = str(root.attrs["bddl_file_name"]).split("bddl_files/")[-1]
        if (
            Path(bddl).name
            != "pick_up_the_tomato_sauce_and_place_it_in_the_basket.bddl"
        ):
            raise ValueError(f"Out-of-scope task: {bddl}")
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / bddl),
            camera_heights=128,
            camera_widths=128,
        )
        env.seed(args.seed)
        splits = episode_splits(list(root.keys()), args.seed)
        manifest = dict(
            schema_version=1,
            task="pick_up_the_tomato_sauce_and_place_it_in_the_basket",
            data_repo="yifengzhu-hf/LIBERO-datasets",
            data_revision="f13aa24a3da8c43c7225569f28c562979fa0e35a",
            source_sha256=source_sha256,
            builder_sha256=sha256(__file__),
            label_source="env.check_success() at the exact rendered simulator state",
            original_rewards_used=False,
            image_keys=["agentview_image", "robot0_eye_in_hand_image"],
            image_transform="vertical flip of simulator RGB, uint8, 128x128",
            seed=args.seed,
            stride=args.stride,
            splits=splits,
            episodes={},
        )
        try:
            for split, names in splits.items():
                for name in names:
                    demo = root[name]
                    env.reset()
                    xml = postprocess_model_xml(str(demo.attrs["model_file"]), {})
                    env.reset_from_xml_string(xml)
                    indices = sorted(
                        set(range(0, len(demo["states"]), args.stride))
                        | {len(demo["states"]) - 1}
                    )
                    images, labels = [], []
                    for index in indices:
                        # Re-render instead of assuming original RGB/state alignment.
                        obs = env.set_init_state(demo["states"][index])
                        labels.append(int(env.check_success()))
                        images.append(
                            np.stack(
                                [
                                    obs[key][::-1].copy()
                                    for key in manifest["image_keys"]
                                ]
                            )
                        )
                    path = args.output / f"{name}.npz"
                    np.savez_compressed(
                        path,
                        rgb=np.stack(images),
                        labels=np.array(labels, dtype=np.int64),
                        frame_indices=np.array(indices),
                    )
                    entry = dict(
                        split=split,
                        frames=len(labels),
                        positive=sum(labels),
                        negative=len(labels) - sum(labels),
                        sha256=sha256(path),
                    )
                    manifest["episodes"][name] = entry
                    print(
                        json.dumps(
                            dict(event="prepared_episode", episode=name, **entry)
                        ),
                        flush=True,
                    )
                    save_json(args.output / "prepare_progress.json", manifest)
        finally:
            env.close()
    for split in splits:
        entries = [
            entry for entry in manifest["episodes"].values() if entry["split"] == split
        ]
        if (
            min(
                sum(e["positive"] for e in entries), sum(e["negative"] for e in entries)
            )
            == 0
        ):
            raise ValueError(
                f"Both labels required in {split}; do not train on terminal markers"
            )
    save_json(manifest_path, manifest)
    print(json.dumps(dict(event="data_ready", manifest=str(manifest_path))), flush=True)


def make_model(pretrained):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    class DualViewCompletion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.encoder.fc = torch.nn.Identity()
            self.head = torch.nn.Sequential(
                torch.nn.Linear(1024, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(256, 1),
            )
            self.register_buffer(
                "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

        def forward(self, rgb):
            if (
                rgb.ndim != 5
                or tuple(rgb.shape[1:]) != (2, 128, 128, 3)
                or rgb.dtype != torch.uint8
            ):
                raise ValueError(
                    "Expected uint8 [batch, 2, 128, 128, 3] RGB in manifest camera order"
                )
            batch = len(rgb)
            x = rgb.flatten(0, 1).permute(0, 3, 1, 2).float() / 255
            x = (x - self.mean) / self.std
            features = self.encoder(x).reshape(batch, 1024)
            return self.head(features).flatten()

    return DualViewCompletion()


def load_split(data_dir, manifest, split):
    images, labels = [], []
    for name in manifest["splits"][split]:
        path = data_dir / f"{name}.npz"
        if sha256(path) != manifest["episodes"][name]["sha256"]:
            raise ValueError(f"Data hash mismatch: {name}")
        with np.load(path, allow_pickle=False) as episode:
            images.append(episode["rgb"])
            labels.append(episode["labels"])
    return np.concatenate(images), np.concatenate(labels)


def completion_metrics(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite completion probability")
    positive, negative = labels == 1, labels == 0
    pred = probabilities >= 0.5
    return dict(
        frames=len(labels),
        positives=int(positive.sum()),
        negatives=int(negative.sum()),
        accuracy=float(np.mean(pred == labels)),
        false_complete_at_0_9=int(np.sum((probabilities >= 0.9) & negative)),
        true_complete_at_0_9=int(np.sum((probabilities >= 0.9) & positive)),
        false_incomplete_at_0_1=int(np.sum((probabilities <= 0.1) & positive)),
        unknown_at_0_1_0_9=int(np.sum((probabilities > 0.1) & (probabilities < 0.9))),
    )


def train(args):
    import torch

    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This training command requires CUDA")
    manifest = json.loads((args.data / "manifest.json").read_text())
    groups = [set(manifest["splits"][key]) for key in ("train", "validation", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Episode leakage between splits")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = dict(vars(args))
    cfg = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in cfg.items()
    }
    cfg.update(
        manifest_sha256=sha256(args.data / "manifest.json"),
        script_sha256=sha256(__file__),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        packages={
            name: importlib.metadata.version(name)
            for name in (
                "torchvision",
                "numpy",
                "hf-libero",
                "mujoco",
                "robosuite",
                "h5py",
            )
        },
        model="shared ImageNet ResNet18; fixed-order dual-view MLP; all weights finetuned",
        input="RGB only; no state, reward, frame index, or terminal flag",
        completion_threshold=0.9,
        incomplete_threshold=0.1,
        runtime_enabled=False,
        test_set_evaluated=False,
    )
    save_json(args.output / "config.json", cfg)
    train_rgb, train_y = load_split(args.data, manifest, "train")
    val_rgb, val_y = load_split(args.data, manifest, "validation")
    if set(np.unique(train_y)) != {0, 1} or set(np.unique(val_y)) != {0, 1}:
        raise ValueError("Both classes required")
    datasets = [
        torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y).float())
        for x, y in ((train_rgb, train_y), (val_rgb, val_y))
    ]
    loaders = [
        torch.utils.data.DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=(index == 0),
            num_workers=0,
            pin_memory=True,
        )
        for index, data in enumerate(datasets)
    ]
    model = make_model(pretrained=True).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    pos_weight = torch.tensor(
        [(train_y == 0).sum() / (train_y == 1).sum()], device="cuda"
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best, step, started = float("inf"), 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_count = 0.0, 0
        for rgb, labels in loaders[0]:
            rgb, labels = rgb.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(rgb), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            step += 1
            total_loss += loss.item() * len(labels)
            total_count += len(labels)
            if step == 1 or step % 10 == 0:
                print(
                    json.dumps(
                        dict(
                            event="train_step",
                            epoch=epoch,
                            step=step,
                            loss=loss.item(),
                            gradient_norm=float(gradient_norm),
                        )
                    ),
                    flush=True,
                )
            if args.smoke:
                break
        model.eval()
        probabilities = []
        with torch.inference_mode():
            for rgb, _ in loaders[1]:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    probabilities.append(
                        model(rgb.cuda(non_blocking=True))
                        .float()
                        .sigmoid()
                        .cpu()
                        .numpy()
                    )
                if args.smoke:
                    break
        probabilities = np.concatenate(probabilities)
        labels = val_y[: len(probabilities)]
        val_loss = float(
            np.mean(
                -(
                    labels * np.log(np.clip(probabilities, 1e-7, 1))
                    + (1 - labels) * np.log(np.clip(1 - probabilities, 1e-7, 1))
                )
            )
        )
        metrics = dict(
            event="epoch",
            epoch=epoch,
            step=step,
            train_loss=total_loss / total_count,
            validation_loss=val_loss,
            validation=completion_metrics(labels, probabilities),
            elapsed_seconds=time.time() - started,
            peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20,
        )
        print(json.dumps(metrics), flush=True)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        checkpoint = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            step=step,
            config=cfg,
            metrics=metrics,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),
        )
        temporary = args.output / "last.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(args.output / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(checkpoint, args.output / "best.pt")
        if args.smoke:
            restored = make_model(pretrained=False).cuda().eval()
            restored.load_state_dict(
                torch.load(
                    args.output / "last.pt", map_location="cuda", weights_only=False
                )["model"]
            )
            sample = torch.from_numpy(val_rgb[:2]).cuda()
            with torch.inference_mode():
                torch.testing.assert_close(
                    model(sample), restored(sample), atol=0, rtol=0
                )
            save_json(
                args.output / "smoke_pass.json",
                dict(metrics, passed=True, checkpoint_reload_equal=True),
            )
            break
    save_json(
        args.output / "finished.json",
        dict(
            status="smoke_pass" if args.smoke else "completed",
            epoch=epoch,
            step=step,
            test_set_evaluated=False,
            runtime_enabled=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--hdf5", required=True, type=Path)
    prepare_parser.add_argument("--stride", default=3, type=int)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data", required=True, type=Path)
    train_parser.add_argument("--epochs", default=30, type=int)
    train_parser.add_argument("--batch-size", default=64, type=int)
    train_parser.add_argument("--lr", default=0.0001, type=float)
    train_parser.add_argument("--smoke", action="store_true")
    for command_parser in (prepare_parser, train_parser):
        command_parser.add_argument("--output", required=True, type=Path)
        command_parser.add_argument("--seed", default=20260915, type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.stride < 1:
            parser.error("stride must be positive")
        prepare(args)
    else:
        if min(args.epochs, args.batch_size, args.lr) <= 0:
            parser.error("epochs, batch size, and learning rate must be positive")
        train(args)


if __name__ == "__main__":
    main()
