"""Frozen RGB-only completion candidate for the finite tomato-sauce task."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

CHECKPOINT_SHA256 = "35c122539730783bc4c61a65d7515a075385a5ac81a0dd8fa2a34c445a55e7c5"
MANIFEST_SHA256 = "7ae52ff3c456c94368e3c8b1f4b3641982349e615ebd46db2dde1d19253c59c3"
INCOMPLETE_THRESHOLD = 0.1
COMPLETE_THRESHOLD = 0.9


def decision(probability):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid completion probability")
    if probability >= COMPLETE_THRESHOLD:
        return "complete"
    if probability <= INCOMPLETE_THRESHOLD:
        return "incomplete"
    return "unknown"


def native_rgb(pixels):
    """Raw native RGB -> training convention; never use the legacy Qwen rotation."""
    from PIL import Image

    result = []
    for key in ("image", "image2"):
        value = np.asarray(pixels[key])
        if value.shape != (1, 360, 360, 3) or value.dtype != np.uint8:
            raise ValueError("Expected two native uint8 360x360 cameras")
        result.append(
            np.asarray(
                Image.fromarray(value[0, ::-1].copy()).resize(
                    (128, 128), Image.Resampling.BILINEAR
                )
            )
        )
    return np.stack(result)


def make_model():
    # Architecture and parameter names match the frozen training implementation.
    import torch
    from torchvision.models import resnet18

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = resnet18(weights=None)
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
            batch = len(rgb)
            x = rgb.flatten(0, 1).permute(0, 3, 1, 2).float() / 255
            features = self.encoder((x - self.mean) / self.std).reshape(batch, 1024)
            return self.head(features).flatten()

    return Model()


class FrozenCompletion:
    def __init__(self, checkpoint: Path):
        import torch

        with Path(checkpoint).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT_SHA256:
                raise ValueError("Checkpoint differs from the frozen epoch-4 candidate")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["epoch"] != 4 or saved["config"]["manifest_sha256"] != MANIFEST_SHA256:
            raise ValueError("Checkpoint provenance mismatch")
        self.model = make_model().cuda().eval()
        self.model.load_state_dict(saved["model"], strict=True)
        self.torch = torch

    def predict(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.ndim != 5 or rgb.shape[1:] != (2, 128, 128, 3) or rgb.dtype != np.uint8:
            raise ValueError("Expected uint8 [batch,2,128,128,3] RGB only")
        torch = self.torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Match the original training/evaluation math even after VLA setup.
            with torch.backends.cudnn.flags(
                benchmark=False, deterministic=False, allow_tf32=True
            ):
                values = (
                    self.model(torch.from_numpy(rgb.copy()).cuda())
                    .float()
                    .sigmoid()
                    .cpu()
                    .numpy()
                )
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite completion output")
        return values

    def __call__(self, observation):
        from .contracts import CheckDecision

        rgb = native_rgb(observation.pixels)
        probability = float(self.predict(rgb[None])[0])
        return CheckDecision(
            observation.observation_id,
            decision(probability),
            "Frozen dual-view RGB completion probability",
        ), dict(
            probability=probability,
            checkpoint_sha256=CHECKPOINT_SHA256,
            input_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            preprocessing="vertical flip both native cameras; PIL bilinear 360 to 128",
        )
