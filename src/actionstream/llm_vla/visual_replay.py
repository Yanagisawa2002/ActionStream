"""CPU replay of visual receipts against immutable RGB inputs.

Logged tensor hashes are consistency evidence, not attestation of a model run.
Worker-supplied final decisions are never accepted without recomputation.
"""

import hashlib
import re

from PIL import Image, ImageOps

from .contracts import strict_object
from .gate_runner import sha256
from .visual_evidence import (
    EvidenceHistory,
    interventions,
    mask_targets,
    parse_evidence,
    rgb_hash,
    verify,
)


def _images(root, views):
    root = root.resolve()
    images = []
    for view in views:
        if set(view) != {"path", "sha256"}:
            raise ValueError("Unexpected image metadata")
        path = (root / view["path"]).resolve()
        if not path.is_relative_to(root) or sha256(path) != view["sha256"]:
            raise ValueError("Image identity mismatch")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _record(row, images, references):
    if row.get("status") != "COMPLETED" or not isinstance(row.get("raw_output"), str):
        raise ValueError("Missing or failed model call")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Missing model prompt")
    if hashlib.sha256(prompt.encode()).hexdigest() != row.get("prompt_sha256"):
        raise ValueError("Prompt hash mismatch")
    if row.get("rgb_sha256") != [rgb_hash(im) for im in images]:
        raise ValueError("Model record does not bind expected RGB intervention")
    if row.get("reference_rgb_sha256") != [rgb_hash(im) for im in references]:
        raise ValueError("Appearance reference binding mismatch")
    tensors = row.get("image_tensor_sha256", [])
    if len(tensors) != len(images) + len(references) or not all(
        isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in tensors
    ):
        raise ValueError("Missing per-image tensor identity")


def replay_visual_receipt(receipt, public, input_root):
    """Reconstruct decisions before scoring; fail closed on any broken binding."""
    try:
        if public is None or input_root is None:
            raise ValueError("Independent replay requires public inputs and RGB root")
        if receipt.get("status") != "COMPLETED":
            raise ValueError("Incomplete visual receipt")
        observations = public["observations"]
        lookup = {r["observation_id"]: r for r in observations}
        if len(lookup) != len(observations) or not observations:
            raise ValueError("Missing or duplicate public observations")
        references = _images(input_root, public["references"])
        if len(references) != 2:
            raise ValueError("Exactly two appearance references required")
        images = {
            identity: _images(input_root, r["views"]) for identity, r in lookup.items()
        }
        if any(len(pair) != 2 for pair in images.values()):
            raise ValueError("Exactly two current RGB views required")
        calls = {}
        for row in receipt["cases"]:
            key = row["identity"], row["variant"]
            if key in calls:
                raise ValueError("Duplicate visual call")
            calls[key] = row
        expected = {
            (identity, variant)
            for identity in lookup
            for variant in ("original", "flipped", "masked")
        }
        causal_inputs = {}
        for anchor in public["anchors"]:
            current, earlier = anchor["current"], anchor["earlier"]
            if current not in images or earlier not in images:
                raise ValueError("Unknown causal anchor")
            variants = interventions(images[current], images[earlier][1])
            for variant, pair in variants.items():
                key = current, "causal_" + variant
                if key in causal_inputs:
                    raise ValueError("Duplicate causal anchor")
                causal_inputs[key] = pair
        expected.update(causal_inputs)
        if set(calls) != expected:
            raise ValueError("Visual call schedule differs from public protocol")
        for key, pair in causal_inputs.items():
            row = calls[key]
            _record(row, pair, [])
            status = strict_object(row["raw_output"], {"status"})["status"]
            if status not in ("complete", "incomplete", "unknown"):
                raise ValueError("Invalid causal decision")
            if row["prompt_sha256"] != calls[key[0], "causal_real"]["prompt_sha256"]:
                raise ValueError("Causal prompt changed")
        logged = {r["identity"]: r for r in receipt.get("verified", [])}
        if len(logged) != len(receipt.get("verified", [])) or set(logged) != set(
            lookup
        ):
            raise ValueError("Missing, duplicate or unknown worker verification")
        history, verified = EvidenceHistory(), {}
        for item in observations:
            identity = item["observation_id"]
            pair = images[identity]
            original, flipped, masked = [
                calls[identity, v] for v in ("original", "flipped", "masked")
            ]
            _record(original, pair, references)
            proposal = parse_evidence(original["raw_output"])
            _record(flipped, [ImageOps.mirror(im) for im in pair], references)
            _record(masked, mask_targets(pair, proposal), references)
            parse_evidence(flipped["raw_output"])
            parse_evidence(masked["raw_output"])
            result = verify(
                pair,
                original["raw_output"],
                flipped["raw_output"],
                masked["raw_output"],
                references,
                public["thresholds"],
            )
            changed = all(
                original["image_tensor_sha256"][v] != altered["image_tensor_sha256"][v]
                and original["rgb_sha256"][v] != altered["rgb_sha256"][v]
                for altered in (flipped, masked)
                for v in range(2)
            )
            prompt_same = (
                len({r["prompt_sha256"] for r in (original, flipped, masked)}) == 1
            )
            result["intervention_input_verified"] = bool(changed and prompt_same)
            if not changed or not prompt_same:
                result.update(
                    status="unknown", reason="counterfactual_input_not_established"
                )
            result = history.record(item["stream"], item["observed_ns"], result)
            reconstructed = dict(identity=identity, **result)
            if reconstructed != logged[identity]:
                raise ValueError(
                    "Worker verification differs from independent RGB replay"
                )
            verified[identity] = reconstructed
        return dict(status="PASS", errors=[], verified=verified)
    except (KeyError, ValueError, TypeError, AttributeError, OSError) as exc:
        return dict(status="FAIL", errors=[str(exc)], verified={})
