"""RGB-only temporal completion candidate; no simulator truth imports."""

import hashlib
import numpy as np
from PIL import Image

IMAGE_SIZE = 192
OFFSETS = (10, 5, 0)


def camera_rgb(observation):
    return np.stack(
        [
            np.asarray(
                Image.fromarray(observation[k][::-1].copy()).resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
                )
            )
            for k in ("agentview_image", "robot0_eye_in_hand_image")
        ]
    )


def classify(probabilities):
    p = np.asarray(probabilities)
    if (
        p.shape != (3,)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
        or not np.isclose(p.sum(), 1, atol=1e-5)
    ):
        raise ValueError("Expected three finite normalized class probabilities")
    if p[1] >= 0.95:
        return "complete"
    if p[0] >= 0.9:
        return "incomplete"
    return "unknown"


def make_model(pretrained=False):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    class TemporalCompletion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.encoder.avgpool = torch.nn.AdaptiveAvgPool2d((2, 2))
            self.encoder.fc = torch.nn.Identity()
            self.head = torch.nn.Sequential(
                torch.nn.Linear(3 * 2 * 512 * 4, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(256, 3),
            )
            self.register_buffer(
                "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

        def forward(self, rgb):
            if (
                rgb.shape[1:] != (3, 2, IMAGE_SIZE, IMAGE_SIZE, 3)
                or rgb.dtype != torch.uint8
            ):
                raise ValueError("Expected uint8 [batch,3 times,2 views,192,192,3]")
            batch = len(rgb)
            x = rgb.flatten(0, 2).permute(0, 3, 1, 2).float() / 255
            return self.head(
                self.encoder((x - self.mean) / self.std).reshape(batch, -1)
            )

    return TemporalCompletion()


class TemporalPredictor:
    def __init__(self, checkpoint, expected_sha256):
        import torch

        with open(checkpoint, "rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
                raise ValueError("Frozen v2 checkpoint hash mismatch")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = make_model().cuda().eval()
        self.model.load_state_dict(saved["model"])
        self.torch = torch

    def predict(self, clips):
        torch = self.torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return (
                self.model(torch.from_numpy(np.asarray(clips).copy()).cuda())
                .float()
                .softmax(-1)
                .cpu()
                .numpy()
            )
