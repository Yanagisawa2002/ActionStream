"""Build a guarded development manifest and train the shared completion candidate.

Run from the checkout root. Native collection uses
python -m actionstream.llm_vla.multitask_cli; this script never collects heldout data.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from actionstream.llm_vla.finite_native import digest, save
from actionstream.llm_vla.multitask_data import (
    episode_entry,
    load_clips,
    validate_manifest,
)
from actionstream.llm_vla.task_completion import ARCHITECTURE, ENCODER, make_model
from actionstream.llm_vla.task_registry import DEVELOPMENT, registry_sha256


def build_manifest(args):
    entries = []
    for path in args.episodes:
        purpose = json.loads((path / "purpose.json").read_text())
        entries.append(episode_entry(path, purpose["split"]))
    manifest = dict(registry_sha256=registry_sha256(), episodes=entries)
    validate_manifest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")


def evaluate(model, rgb, vectors, labels, rows, device, batch_size):
    import torch

    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(rgb), batch_size):
            batch = slice(start, start + batch_size)
            values = model(
                torch.from_numpy(rgb[batch]).to(device),
                torch.from_numpy(vectors[batch]).to(device),
            )
            probabilities.append(values.float().softmax(-1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    tasks = {}
    for task in DEVELOPMENT:
        mask = np.array([row["task_key"] == task.key for row in rows])
        values, truth = probabilities[mask], labels[mask]
        tasks[task.key] = dict(
            clips=int(mask.sum()),
            false_complete=int(((values[:, 1] >= 0.95) & (truth != 1)).sum()),
            complete_recall=float((values[truth == 1, 1] >= 0.95).mean()),
            cross_entropy=float(
                -np.log(values[np.arange(len(truth)), truth].clip(1e-8)).mean()
            ),
        )
    return dict(
        tasks=tasks,
        selection_loss=float(np.mean([v["cross_entropy"] for v in tasks.values()])),
    )


def train(args):
    import torch

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epoch and batch counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for native training; CPU is available only for a smoke test"
        )
    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    # Both folds must cover every task, and each task needs positive and negative clips.
    training = load_clips(manifest, "train")
    validation = load_clips(manifest, "validation")
    counts = {}
    for split, data in (("train", training), ("validation", validation)):
        rgb, labels, vectors, rows = data
        counts[split] = {}
        for task in DEVELOPMENT:
            selected = [
                int(y) for y, row in zip(labels, rows) if row["task_key"] == task.key
            ]
            counts[split][task.key] = dict(Counter(selected))
            if not {0, 1}.issubset(selected):
                raise ValueError(
                    f"Missing complete/incomplete coverage: {split}/{task.key}"
                )
    args.output.mkdir(parents=True, exist_ok=False)
    from actionstream.delivery import environment

    runtime = environment(require_cuda=args.device == "cuda")
    save(args.output / "environment.json", runtime)
    if args.device == "cuda" and runtime["status"] != "PASS":
        raise RuntimeError("Training runtime differs from the locked CUDA environment")
    save(args.output / "manifest.json", manifest)
    save(args.output / "data_counts.json", counts)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    rgb, labels, vectors, rows = training
    # Start from pinned torchvision ImageNet weights; record their identity in the run.
    model = make_model(vectors.shape[1], pretrained=True).to(args.device)
    initialization = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        initialization.update(name.encode())
        initialization.update(tensor.cpu().contiguous().numpy().tobytes())
    initialization_sha256 = initialization.hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(rgb), torch.from_numpy(vectors), torch.from_numpy(labels)
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    source_root = Path(__file__).resolve().parents[2]
    source_hashes = {
        str(path.relative_to(source_root)).replace("\\", "/"): digest(path)
        for path in (source_root / "src/actionstream/llm_vla").glob("*.py")
    }
    source_hashes[str(Path(__file__).relative_to(source_root)).replace("\\", "/")] = (
        digest(__file__)
    )
    for name in ("uv.lock", "pyproject.toml", "configs/multitask_development_v1.json"):
        source_hashes[name] = digest(source_root / name)
    best, history = float("inf"), []
    for epoch in range(args.epochs):
        model.train()
        for images, text, truth in loader:
            images, text, truth = (
                images.to(args.device),
                text.to(args.device),
                truth.to(args.device),
            )
            # Explicit unobservable examples; do not relabel arbitrary task swaps.
            blackout = torch.rand(len(images), device=args.device) < 0.15
            images[blackout] = 0
            truth[blackout] = 2
            if args.ablation == "no_task":
                text = torch.zeros_like(text)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images, text), truth)
            loss.backward()
            optimizer.step()
        vrgb, vlabels, vvectors, vrows = validation
        if args.ablation == "no_task":
            vvectors = np.zeros_like(vvectors)
        metrics = evaluate(
            model, vrgb, vvectors, vlabels, vrows, args.device, args.batch_size
        )
        history.append(dict(epoch=epoch + 1, **metrics))
        save(args.output / "history.json", history)
        if metrics["selection_loss"] < best:
            best = metrics["selection_loss"]
            torch.save(
                dict(
                    model=model.state_dict(),
                    architecture=ARCHITECTURE,
                    encoder=ENCODER,
                    registry_sha256=registry_sha256(),
                    text_dim=vectors.shape[1],
                    epoch=epoch + 1,
                    manifest_sha256=digest(args.output / "manifest.json"),
                    sources=source_hashes,
                    seed=args.seed,
                    ablation=args.ablation,
                    acceptance=False,
                    vision_initialization="torchvision.ResNet18_Weights.IMAGENET1K_V1",
                    initialization_sha256=initialization_sha256,
                ),
                args.output / "candidate.pt",
            )
        print(json.dumps(history[-1]), flush=True)
    save(
        args.output / "candidate.json",
        dict(
            checkpoint_sha256=digest(args.output / "candidate.pt"),
            epochs=args.epochs,
            ablation=args.ablation,
            native_acceptance="NOT_RUN",
            status="DEVELOPMENT_ONLY",
            generalization="NOT_RUN",
            data_manifest_sha256=digest(args.output / "manifest.json"),
            source_hashes=source_hashes,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--episodes", type=Path, nargs="+", required=True)
    manifest.add_argument("--output", type=Path, required=True)
    trainer = commands.add_parser("train")
    trainer.add_argument("--manifest", type=Path, required=True)
    trainer.add_argument("--output", type=Path, required=True)
    trainer.add_argument("--epochs", type=int, default=16)
    trainer.add_argument("--batch-size", type=int, default=8)
    trainer.add_argument("--seed", type=int, default=20260916)
    trainer.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    trainer.add_argument(
        "--ablation", choices=("conditioned", "no_task"), default="conditioned"
    )
    args = parser.parse_args()
    (build_manifest if args.command == "manifest" else train)(args)


if __name__ == "__main__":
    main()
