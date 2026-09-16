"""Task-conditioned RGB completion candidate; no simulator or truth inputs.

Text is encoded once before dispatch by the pinned, frozen language model.
Only the shared visual encoder and fusion head are trained. No task-ID embedding
or task-specific classifier is used. This architecture alone proves no transfer.
"""

from dataclasses import asdict, dataclass
import hashlib

import numpy as np

from .grounding import contract_hash
from .qwen import MODEL_ID, MODEL_REVISION
from .task_registry import registry_sha256, require_development

ENCODER = dict(
    model_id=MODEL_ID,
    revision=MODEL_REVISION,
    pooling="masked_mean_last_hidden_l2_v1",
    input="canonical_instruction",
)
ARCHITECTURE = "temporal_resnet18_qwen_fusion_v1"


@dataclass(frozen=True)
class TaskCondition:
    task_sha256: str
    instruction: str
    vector: tuple[float, ...]
    encoder_sha256: str

    def __post_init__(self):
        object.__setattr__(self, "vector", tuple(self.vector))

    @property
    def sha256(self):
        return contract_hash(asdict(self))

    def validate(self, task):
        require_development(task)
        values = np.asarray(self.vector, dtype=np.float32)
        if (
            self.task_sha256 != task.sha256
            or self.instruction != task.canonical_instruction
            or self.encoder_sha256 != contract_hash(ENCODER)
            or values.ndim != 1
            or not len(values)
            or not np.isfinite(values).all()
            or not np.isclose(np.linalg.norm(values), 1, atol=1e-4)
        ):
            raise ValueError("Invalid or foreign task text condition")
        return values


def encode_task(llm, task):
    """Text only; runs during startup/collection, never in the 50 ms loop."""
    require_development(task)
    if llm.settings["id"] != MODEL_ID or llm.settings["revision"] != MODEL_REVISION:
        raise ValueError("Task encoder must use the pinned language model")
    torch = llm.torch
    inputs = llm.processor(text=[task.canonical_instruction], return_tensors="pt").to(
        llm.model.device
    )
    with torch.inference_mode():
        outputs = llm.model(
            **inputs, output_hidden_states=True, use_cache=False, return_dict=True
        )
        hidden = outputs.hidden_states[-1].float()
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        vector = torch.nn.functional.normalize(pooled, dim=-1)[0].cpu().tolist()
    result = TaskCondition(
        task.sha256, task.canonical_instruction, tuple(vector), contract_hash(ENCODER)
    )
    result.validate(task)
    return result


def make_model(text_dim, pretrained=False):
    import torch
    from .temporal_completion import make_model as visual_model

    class TaskCompletion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = visual_model(pretrained)
            # Keep spatial/temporal features, then fuse with the language condition.
            self.visual.head = torch.nn.Sequential(
                torch.nn.Linear(3 * 2 * 512 * 4, 256), torch.nn.ReLU()
            )
            self.text = torch.nn.Sequential(
                torch.nn.Linear(text_dim, 128), torch.nn.ReLU()
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(384, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(256, 3),
            )

        def forward(self, rgb, condition):
            if (
                condition.shape != (len(rgb), text_dim)
                or not torch.isfinite(condition).all()
            ):
                raise ValueError("Expected one finite text vector per RGB clip")
            return self.head(
                torch.cat((self.visual(rgb), self.text(condition)), dim=-1)
            )

    return TaskCompletion()


class TaskPredictor:
    def __init__(self, checkpoint, expected_sha256, task, condition, *, device="cuda"):
        import torch

        vector = condition.validate(task)
        with open(checkpoint, "rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
                raise ValueError("Task completion checkpoint hash mismatch")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            saved["architecture"] != ARCHITECTURE
            or saved["encoder"] != ENCODER
            or saved["registry_sha256"] != registry_sha256()
            or saved["text_dim"] != len(vector)
            or saved.get("ablation") != "conditioned"
        ):
            raise ValueError("Task completion checkpoint provenance mismatch")
        self.model = make_model(len(vector)).to(device).eval()
        self.model.load_state_dict(saved["model"])
        self.torch, self.device = torch, device
        self.task, self.condition = task, condition
        self.checkpoint_sha256 = expected_sha256
        self.vector = torch.tensor(vector.copy(), device=device)[None]

    def validate_binding(self, task):
        if task != self.task:
            raise ValueError("RGB verifier is bound to another task")
        self.condition.validate(task)

    def predict(self, clips):
        from contextlib import nullcontext

        torch = self.torch
        rgb = np.asarray(clips)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if str(self.device).startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            return (
                self.model(
                    torch.from_numpy(rgb.copy()).to(self.device),
                    self.vector.expand(len(rgb), -1),
                )
                .float()
                .softmax(-1)
                .cpu()
                .numpy()
            )


def vector_digest(condition):
    return hashlib.sha256(
        np.asarray(condition.vector, dtype="<f4").tobytes()
    ).hexdigest()
