"""Pinned local Qwen; prompts deliberately accept no evaluator or native fields."""

from __future__ import annotations

import hashlib
import inspect
import json
import signal
import time

from .contracts import CAPABILITY

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def parser_messages(system_prompt: str, request_id: str, utterance: str):
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {"request_id": request_id, "utterance": utterance},
                        ensure_ascii=False,
                    ),
                }
            ],
        },
    ]


def checker_messages(system_prompt: str, observation_id: str, images: list):
    if len(images) != 2:
        raise ValueError("Exactly two simultaneous RGB images are required")
    goal = {key: CAPABILITY[key] for key in ("skill", "target", "destination")}
    content = [
        {
            "type": "text",
            "text": json.dumps({"observation_id": observation_id, "goal": goal}),
        }
    ]
    content.extend({"type": "image", "image": item} for item in images)
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


class LocalQwen:
    def __init__(self, directory, settings: dict):
        if settings["id"] != MODEL_ID or settings["revision"] != MODEL_REVISION:
            raise ValueError("Only the frozen model revision is authorized")
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.settings = settings
        started = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                directory,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
                trust_remote_code=False,
            )
            .to("cuda")
            .eval()
        )
        torch.cuda.synchronize()
        self.receipt = dict(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            source=inspect.getfile(type(self.model)),
            processor_source=inspect.getfile(type(self.processor)),
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            dtype=str(next(self.model.parameters()).dtype),
            device=str(next(self.model.parameters()).device),
            load_wall_s=time.monotonic() - started,
            generation={
                "do_sample": False,
                "max_new_tokens": settings["max_new_tokens"],
            },
        )

    def generate(
        self,
        messages: list,
        *,
        structured_slots=None,
        record_tensors=False,
        max_new_tokens=None,
    ) -> dict:
        started = time.monotonic()
        record = dict(status="RUNNING", raw_output=None)

        def timeout(signum, frame):
            raise TimeoutError("Frozen per-call model deadline exceeded")

        previous = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, self.settings["call_timeout_s"])
        try:
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            record.update(
                prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()
            )
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            record["input_shapes"] = {
                key: list(value.shape) for key, value in inputs.items()
            }
            record["image_grid_thw"] = (
                inputs["image_grid_thw"].tolist()
                if "image_grid_thw" in inputs
                else None
            )
            options = {}
            if record_tensors:

                def tensor_digest(tensor):
                    data = tensor.detach().contiguous().view(self.torch.uint8).cpu()
                    return hashlib.sha256(data.numpy().tobytes()).hexdigest()

                record["tensor_sha256"] = {
                    key: tensor_digest(value) for key, value in inputs.items()
                }
                record["tensor_dtypes"] = {
                    key: str(value.dtype) for key, value in inputs.items()
                }
                if "pixel_values" in inputs:
                    offset = 0
                    hashes = []
                    for grid in record["image_grid_thw"]:
                        size = grid[0] * grid[1] * grid[2]
                        hashes.append(
                            tensor_digest(
                                inputs["pixel_values"][offset : offset + size]
                            )
                        )
                        offset += size
                    if offset != inputs["pixel_values"].shape[0]:
                        raise ValueError("Cannot establish per-image tensor boundaries")
                    record["image_tensor_sha256"] = hashes
            if structured_slots is not None:
                from .structured import SlotDecoder

                options["prefix_allowed_tokens_fn"] = SlotDecoder(
                    self.processor.tokenizer,
                    structured_slots,
                    inputs["input_ids"].shape[1],
                )
                record["structured_decoding"] = "finite_token_slots_v1"
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens or self.settings["max_new_tokens"],
                    **options,
                )
            self.torch.cuda.synchronize()
            new = output[:, inputs["input_ids"].shape[1] :]
            record.update(
                status="COMPLETED",
                generated_tokens=new.shape[1],
                raw_output=self.processor.batch_decode(
                    new, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0],
            )
        except Exception as exc:
            record.update(status="ERROR", error_type=type(exc).__name__, error=str(exc))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            record["wall_s"] = time.monotonic() - started
        return record

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()
