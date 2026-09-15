"""RGB-only appearance and counterfactual checks of proposed visual regions.

No simulator state, frame number, source filename, evaluator ID or success label
is accepted. These are finite 2D consistency checks, not a proof of 3D physics.
"""

import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .contracts import strict_object


def rgb_hash(image):
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def box_valid(box):
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(type(n) is int for n in box)
        and 0 <= box[0] < box[2] <= 1000
        and 0 <= box[1] < box[3] <= 1000
        and (box[2] - box[0]) * (box[3] - box[1]) >= 100
    )


def parse_evidence(raw):
    data = strict_object(raw, {f"{k}{v}" for v in range(2) for k in "tbijvwr"})
    for view in range(2):
        for key in "tb":
            box = data[f"{key}{view}"]
            if not (
                isinstance(box, list)
                and len(box) == 4
                and all(type(n) is int and 0 <= n <= 1000 for n in box)
            ):
                raise ValueError("Box schema invalid")
        for key, choices in (
            ("i", ("match", "other", "unknown")),
            ("j", ("match", "other", "unknown")),
            ("v", ("visible", "partial", "absent", "unknown")),
            ("w", ("visible", "partial", "absent", "unknown")),
            ("r", ("inside", "outside", "unknown")),
        ):
            if data[f"{key}{view}"] not in choices:
                raise ValueError("Unknown evidence enum")
    return data


def pixel_box(box, image):
    return tuple(
        round(n * (image.width if i % 2 == 0 else image.height) / 1000)
        for i, n in enumerate(box)
    )


def overlap(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    return area


def iou(a, b):
    if not box_valid(a) or not box_valid(b):
        return 0.0
    shared = overlap(a, b)
    return shared / (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared
    )


def histogram(image):
    # Color appearance is independently measured but is not category recognition.
    values = np.asarray(image.convert("HSV"), dtype=np.float64).reshape(-1, 3)
    hist, _ = np.histogramdd(values, bins=(8, 4, 4), range=((0, 256),) * 3)
    hist = hist.ravel()
    return hist / max(1, hist.sum())


def similarity(first, second):
    return float(np.sqrt(histogram(first) * histogram(second)).sum())


def mask_targets(images, proposal):
    output = []
    for view, image in enumerate(images):
        box = proposal.get(f"t{view}", []) if proposal else []
        masked = image.copy()
        if box_valid(box):
            x0, y0, x1, y1 = pixel_box(box, image)
            px, py = max(2, (x1 - x0) // 10), max(2, (y1 - y0) // 10)
            ImageDraw.Draw(masked).rectangle(
                (
                    max(0, x0 - px),
                    max(0, y0 - py),
                    min(image.width - 1, x1 + px),
                    min(image.height - 1, y1 + py),
                ),
                fill=(127,) * 3,
            )
        else:
            masked = Image.new("RGB", image.size, (127,) * 3)
        output.append(masked)
    return output


def interventions(images, earlier_wrist):
    occluded = [image.copy() for image in images]
    for im in occluded:
        ImageDraw.Draw(im).rectangle((90, 90, 270, 270), fill=(127,) * 3)
    return dict(
        real=images,
        blank=[Image.new("RGB", im.size, (127,) * 3) for im in images],
        swapped=list(reversed(images)),
        mismatched=[images[0], earlier_wrist],
        occluded=occluded,
        cropped=[im.crop((60, 60, 300, 300)) for im in images],
    )


def verify(images, original, flipped, masked, references, thresholds):
    """All current views must agree; repeatability alone does not establish truth."""
    try:
        proposals = [parse_evidence(x) for x in (original, flipped, masked)]
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return dict(
            status="unknown", schema="EXPLICIT_FAILURE", reason=str(exc), views=[]
        )
    base, flip, mask = proposals
    rows = []
    for view, image in enumerate(images):
        target, basket = base[f"t{view}"], base[f"b{view}"]
        row = dict(view=view, checks={}, appearance={}, relation="unknown")
        for key, ref in (("t", references[0]), ("b", references[1])):
            box = base[f"{key}{view}"]
            reflected = flip[f"{key}{view}"]
            restored = [
                1000 - reflected[2],
                reflected[1],
                1000 - reflected[0],
                reflected[3],
            ]
            row["checks"][key + "_flip"] = iou(box, restored) >= thresholds["flip_iou"]
            score = (
                similarity(image.crop(pixel_box(box, image)), ref)
                if box_valid(box)
                else 0.0
            )
            row["appearance"][key] = score
            row["checks"][key + "_appearance"] = score >= thresholds[key]
        row["checks"]["identity"] = all(
            p[f"{k}{view}"] == "match" for p in (base, flip) for k in "ij"
        )
        row["checks"]["visibility"] = all(
            p[f"{k}{view}"] in ("visible", "partial")
            for p in (base, flip)
            for k in "vw"
        )
        row["checks"]["mask_response"] = (
            mask[f"v{view}"] in ("absent", "unknown") and mask[f"r{view}"] == "unknown"
        )
        row["checks"]["relation_consistency"] = base[f"r{view}"] == flip[f"r{view}"]
        if box_valid(target) and box_valid(basket):
            fraction = overlap(target, basket) / (
                (target[2] - target[0]) * (target[3] - target[1])
            )
            center = ((target[0] + target[2]) / 2, (target[1] + target[3]) / 2)
            inside = (
                basket[0] <= center[0] <= basket[2]
                and basket[1] <= center[1] <= basket[3]
            )
            row["overlap_fraction"] = fraction
            if all(row["checks"].values()):
                if (
                    fraction >= thresholds["inside_fraction"]
                    and inside
                    and base[f"r{view}"] == "inside"
                ):
                    row["relation"] = "inside"
                elif (
                    fraction <= thresholds["outside_fraction"]
                    and not inside
                    and base[f"r{view}"] == "outside"
                ):
                    row["relation"] = "outside"
        rows.append(row)
    relations = [r["relation"] for r in rows]
    status = (
        "complete"
        if relations == ["inside", "inside"]
        else (
            "incomplete"
            if "outside" in relations and "inside" not in relations
            else "unknown"
        )
    )
    return dict(
        status=status,
        schema="VALID",
        views=rows,
        input_rgb_sha256=[rgb_hash(im) for im in images],
        reflected_rgb_sha256=[rgb_hash(ImageOps.mirror(im)) for im in images],
    )


class EvidenceHistory:
    """Opaque stream and observed timestamps; no source index or outcome access."""

    def __init__(self):
        self.previous = {}

    def record(self, stream, observed_ns, result):
        previous = self.previous.get(stream)
        if not isinstance(observed_ns, int) or (
            previous and observed_ns <= previous[0]
        ):
            return dict(
                result, status="unknown", temporal="missing_or_nonmonotonic_timestamp"
            )
        # Sparse saved frames do not establish continuous tracking. Record only
        # a contradiction check; two simultaneous verified views remain required.
        contradiction = (
            previous and previous[1] == "complete" and result["status"] == "incomplete"
        )
        self.previous[stream] = (observed_ns, result["status"])
        return dict(
            result,
            status="unknown" if contradiction else result["status"],
            temporal="contradiction"
            if contradiction
            else "ordered_sparse_observations",
        )
