"""Build the honest 60--90 second ActionStream v1.1.0 GPU demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = ROOT / "release" / "v1.1.0"
MEDIA_ROOT = RELEASE_ROOT / "media"

WIDTH = 1280
HEIGHT = 720
FPS = 30
SOURCE_FPS = 10
BACKGROUND = (9, 15, 29)
PANEL = (20, 29, 50)
WHITE = (245, 247, 250)
MUTED = (172, 184, 205)
GRID = (61, 75, 101)
BLUE = (0, 114, 178)
LIGHT_BLUE = (86, 180, 233)
GREEN = (0, 158, 115)
ORANGE = (230, 159, 0)
RED = (213, 94, 0)

FONT_REGULAR_PATH = font_manager.findfont(
    font_manager.FontProperties(family="DejaVu Sans", weight="normal")
)
FONT_BOLD_PATH = font_manager.findfont(
    font_manager.FontProperties(family="DejaVu Sans", weight="bold")
)


def executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required executable is unavailable: {name}")
    return path


FFMPEG = executable("ffmpeg")
FFPROBE = executable("ffprobe")


def probe_video(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_REGULAR_PATH, size)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def canvas() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)


def centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    *,
    fill: tuple[int, int, int] = WHITE,
    spacing: int = 7,
) -> None:
    draw.multiline_text(
        xy,
        text,
        font=text_font,
        fill=fill,
        anchor="mm",
        align="center",
        spacing=spacing,
    )


def fade(image: Image.Image, index: int, count: int, frames: int = 10) -> Image.Image:
    alpha = 1.0
    if index < frames:
        alpha = max(0.0, index / frames)
    if index >= count - frames:
        alpha = min(alpha, max(0.0, (count - 1 - index) / frames))
    return image if alpha >= 1.0 else Image.blend(canvas(), image, alpha)


def fit_frame(frame: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(frame, mode="RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image


@dataclass(frozen=True)
class Clip:
    path: Path
    frames: tuple[np.ndarray, ...]

    @classmethod
    def read(cls, path: Path) -> "Clip":
        metadata = probe_video(path)
        source_width = int(metadata["width"])
        source_height = int(metadata["height"])
        ratio = min(420 / source_width, 420 / source_height)
        width = max(2, int(source_width * ratio) // 2 * 2)
        height = max(2, int(source_height * ratio) // 2 * 2)
        result = subprocess.run(
            [
                FFMPEG,
                "-v",
                "error",
                "-i",
                str(path),
                "-vf",
                f"fps={SOURCE_FPS},scale={width}:{height}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        frame_bytes = width * height * 3
        if len(result.stdout) % frame_bytes:
            raise AssertionError(f"Truncated decoded payload in {path}")
        frames = [
            np.frombuffer(
                result.stdout, dtype=np.uint8, count=frame_bytes, offset=offset
            )
            .reshape(height, width, 3)
            .copy()
            for offset in range(0, len(result.stdout), frame_bytes)
        ]
        if not frames:
            raise AssertionError(f"No decoded frames in {path}")
        return cls(path=path, frames=tuple(frames))

    def at(self, output_index: int) -> np.ndarray:
        source_index = min(
            int(output_index * SOURCE_FPS / FPS),
            len(self.frames) - 1,
        )
        return self.frames[source_index]


@dataclass(frozen=True)
class PanelSpec:
    clip: Clip
    label: str
    result: str
    color: tuple[int, int, int]


def title_frame(index: int, count: int) -> Image.Image:
    image = canvas()
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((240, 170, 1040, 184), radius=7, fill=BLUE)
    centered(draw, (WIDTH // 2, 280), "ActionStream v1.1.0", font(62, bold=True))
    centered(
        draw,
        (WIDTH // 2, 385),
        "A pluggable LeRobot inference backend\nwith frozen GPU failure boundaries",
        font(31),
        fill=(220, 227, 239),
        spacing=12,
    )
    centered(
        draw,
        (WIDTH // 2, 535),
        "RTX 5090  ·  X-VLA  ·  3 LIBERO suites  ·  375 paired holdout episodes",
        font(20, bold=True),
        fill=LIGHT_BLUE,
    )
    return fade(image, index, count, frames=12)


def architecture_frame(index: int, count: int) -> Image.Image:
    image = canvas()
    draw = ImageDraw.Draw(image)
    centered(
        draw,
        (WIDTH // 2, 48),
        "Runtime path and auditable telemetry",
        font(28, bold=True),
    )
    boxes = (
        (55, 185, 265, 330, "Observation\nsnapshot", LIGHT_BLUE),
        (310, 185, 520, 330, "Async GPU\nworker", ORANGE),
        (565, 185, 775, 330, "Aligned\naction queue", BLUE),
        (820, 185, 1030, 330, "Bounded hold\n+ fallback", GREEN),
        (1075, 185, 1225, 330, "Robot /\nsimulator", WHITE),
    )
    progress = min(1.0, index / max(1, int(count * 0.65)))
    for box_index, (x0, y0, x1, y1, label, color) in enumerate(boxes):
        active = progress * len(boxes) >= box_index
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=14,
            fill=PANEL,
            outline=color if active else GRID,
            width=3,
        )
        centered(
            draw,
            ((x0 + x1) // 2, (y0 + y1) // 2),
            label,
            font(21, bold=True),
            fill=color,
        )
        if box_index < len(boxes) - 1:
            draw.line(
                (x1 + 8, 257, boxes[box_index + 1][0] - 8, 257), fill=MUTED, width=3
            )
    metrics = (
        ("435", "verified traces"),
        ("75,043", "final 7-D actions"),
        ("12,259 MiB", "peak allocated CUDA"),
        ("p50 / p95", "latency + queue telemetry"),
    )
    for metric_index, (value, label) in enumerate(metrics):
        x0 = 55 + metric_index * 305
        draw.rounded_rectangle(
            (x0, 420, x0 + 270, 610), radius=16, fill=PANEL, outline=GRID, width=2
        )
        centered(draw, (x0 + 135, 486), value, font(29, bold=True), fill=LIGHT_BLUE)
        centered(draw, (x0 + 135, 551), label, font(17), fill=MUTED)
    centered(
        draw,
        (WIDTH // 2, 675),
        "Stale rejection · queue age · depletion · disconnect/recovery · fallback",
        font(17),
        fill=MUTED,
    )
    return fade(image, index, count)


def pair_renderer(
    heading: str,
    subheading: str,
    panels: tuple[PanelSpec, ...],
    footer: str,
) -> Callable[[int, int], Image.Image]:
    if len(panels) not in {2, 3}:
        raise ValueError("paired visual requires two or three panels")

    def render(index: int, count: int) -> Image.Image:
        image = canvas()
        draw = ImageDraw.Draw(image)
        centered(draw, (WIDTH // 2, 38), heading, font(27, bold=True))
        centered(draw, (WIDTH // 2, 77), subheading, font(16), fill=MUTED)
        if len(panels) == 2:
            panel_width, panel_height = 520, 520
            x_values = (95, 665)
        else:
            panel_width, panel_height = 360, 420
            x_values = (55, 460, 865)
        for panel, x0 in zip(panels, x_values, strict=True):
            draw.rounded_rectangle(
                (x0 - 8, 120, x0 + panel_width + 8, 120 + panel_height),
                radius=12,
                fill=PANEL,
                outline=panel.color,
                width=3,
            )
            frame = fit_frame(panel.clip.at(index), (panel_width, panel_height - 70))
            image.paste(
                frame,
                (
                    x0 + (panel_width - frame.width) // 2,
                    130 + (panel_height - 90 - frame.height) // 2,
                ),
            )
            centered(
                draw,
                (x0 + panel_width // 2, 572),
                panel.label,
                font(19, bold=True),
                fill=panel.color,
            )
            centered(
                draw,
                (x0 + panel_width // 2, 608),
                panel.result,
                font(16, bold=True),
                fill=WHITE,
            )
        centered(draw, (WIDTH // 2, 680), footer, font(16), fill=MUTED)
        return fade(image, index, count)

    return render


def results_frame(index: int, count: int) -> Image.Image:
    image = canvas()
    draw = ImageDraw.Draw(image)
    centered(
        draw,
        (WIDTH // 2, 48),
        "Frozen paired result: positive secondary contrast, bounded claim",
        font(27, bold=True),
    )
    draw.rounded_rectangle(
        (65, 115, 620, 600), radius=18, fill=PANEL, outline=GREEN, width=3
    )
    draw.text(
        (105, 155),
        "vs official weighted-average Async",
        font=font(23, bold=True),
        fill=GREEN,
    )
    draw.text(
        (105, 220),
        (
            "600 ± 400 ms jitter\n\n"
            "Success: 8/15 → 14/15\n"
            "+40.0 pp  [13.3, 66.7]\n\n"
            "Mean steps: 207.2 → 144.9\n"
            "−30.1%"
        ),
        font=font(21),
        fill=WHITE,
        spacing=8,
    )
    draw.rounded_rectangle(
        (660, 115, 1215, 600), radius=18, fill=PANEL, outline=ORANGE, width=3
    )
    draw.text(
        (700, 155),
        "Primary latest-only boundary",
        font=font(23, bold=True),
        fill=ORANGE,
    )
    draw.text(
        (700, 220),
        (
            "250 ms / jitter\n"
            "Success tied\n\n"
            "Burst\n"
            "+1/15 success; step CI crosses zero\n\n"
            "Fixed 950 ms\n"
            "Aligned: 10/15\n"
            "Latest-only: 14/15"
        ),
        font=font(21),
        fill=WHITE,
        spacing=8,
    )
    centered(
        draw,
        (WIDTH // 2, 665),
        "Secondary contrast does not replace latest-only as the primary reference.",
        font(17, bold=True),
        fill=LIGHT_BLUE,
    )
    return fade(image, index, count)


def outro_frame(index: int, count: int) -> Image.Image:
    image = canvas()
    draw = ImageDraw.Draw(image)
    centered(draw, (WIDTH // 2, 70), "What v1.1.0 demonstrates", font(35, bold=True))
    claims = (
        ("GO", "Pluggable LeRobot backend + formal X-VLA GPU benchmark", GREEN),
        ("POSITIVE", "Aligned > official Async under frozen jitter", LIGHT_BLUE),
        ("BOUNDARY", "Latest-only remains stronger at fixed 950 ms", ORANGE),
        ("NO CLAIM", "No real-robot safety or completed Arena benchmark", RED),
    )
    for row, (status, claim, color) in enumerate(claims):
        y0 = 145 + row * 105
        draw.rounded_rectangle(
            (105, y0, 1175, y0 + 78), radius=14, fill=PANEL, outline=color, width=2
        )
        draw.text((135, y0 + 22), status, font=font(20, bold=True), fill=color)
        draw.text((335, y0 + 22), claim, font=font(20), fill=WHITE)
    centered(
        draw,
        (WIDTH // 2, 630),
        "Upstream extension boundary: huggingface/lerobot PR #4466",
        font(20, bold=True),
        fill=LIGHT_BLUE,
    )
    centered(
        draw,
        (WIDTH // 2, 676),
        "Frozen evidence · paired videos · replay provenance · explicit NO-GO results",
        font(16),
        fill=MUTED,
    )
    return fade(image, index, count, frames=12)


def write_video(
    output: Path,
    segments: Iterable[tuple[float, Callable[[int, int], Image.Image]]],
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        FFMPEG,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-metadata",
        "title=ActionStream v1.1.0",
        str(output),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    frame_count = 0
    try:
        for seconds, renderer in segments:
            count = int(round(seconds * FPS))
            for index in range(count):
                frame = np.asarray(
                    renderer(index, count).convert("RGB"), dtype=np.uint8
                )
                if frame.shape != (HEIGHT, WIDTH, 3):
                    raise AssertionError(f"Unexpected frame shape: {frame.shape}")
                process.stdin.write(frame.tobytes())
                frame_count += 1
    finally:
        process.stdin.close()
    assert process.stderr is not None
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    if process.wait() != 0:
        raise RuntimeError(stderr)
    return frame_count


def verify_video(path: Path, expected_frames: int) -> dict[str, object]:
    metadata = probe_video(path)
    subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-f", "null", "-"],
        check=True,
        capture_output=True,
    )
    decoded = int(metadata["nb_frames"])
    size = (int(metadata["width"]), int(metadata["height"]))
    if size != (WIDTH, HEIGHT) or decoded != expected_frames:
        raise AssertionError(
            f"Video verification failed: size={size} decoded={decoded} expected={expected_frames}"
        )
    version = subprocess.run(
        [FFMPEG, "-version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    return {
        "codec": metadata["codec_name"],
        "pixel_format": metadata["pix_fmt"],
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frame_count": decoded,
        "duration_seconds": decoded / FPS,
        "ffmpeg_version": version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=ROOT / "artifacts" / "v1_1_raw",
        help="extracted full raw archive root; it remains outside Git",
    )
    args = parser.parse_args()
    evidence_root = args.evidence_root.resolve()
    formal_root = evidence_root / "actionstream_backend_gpu_v1"
    native_object = (
        evidence_root / "learned_isaac_dynamic_official_object0_sync30_dev_20260820"
    )
    native_spatial = (
        evidence_root / "learned_isaac_dynamic_official_spatial2_sync30_dev_20260820"
    )
    native_goal = (
        evidence_root
        / "learned_isaac_dynamic_official_goal2_sync30_dev_20260820_retry2"
    )

    paths = {
        "async_jitter": formal_root
        / "xvla_holdout_object"
        / "videos"
        / "xvla__lerobot_weighted_average__jitter_0600_pm0400_holdout__task5__ep0.mp4",
        "aligned_jitter": formal_root
        / "xvla_holdout_object"
        / "videos"
        / "xvla__actionstream_backend_aligned__jitter_0600_pm0400_holdout__task5__ep0.mp4",
        "latest_950": formal_root
        / "xvla_holdout_goal"
        / "videos"
        / "xvla__lerobot_latest_only__fixed_0950__task2__ep0.mp4",
        "aligned_950": formal_root
        / "xvla_holdout_goal"
        / "videos"
        / "xvla__actionstream_backend_aligned__fixed_0950__task2__ep0.mp4",
        "latest_burst": formal_root
        / "xvla_holdout_object"
        / "videos"
        / "xvla__lerobot_latest_only__burst_0250_to2000_holdout__task5__ep0.mp4",
        "aligned_burst": formal_root
        / "xvla_holdout_object"
        / "videos"
        / "xvla__actionstream_backend_aligned__burst_0250_to2000_holdout__task5__ep0.mp4",
        "guarded_burst": formal_root
        / "xvla_holdout_object"
        / "videos"
        / "xvla__actionstream_backend_guarded__burst_0250_to2000_holdout__task5__ep0.mp4",
        "native_object": native_object / "learned_isaac_smoke.mp4",
        "native_spatial": native_spatial / "learned_isaac_smoke.mp4",
        "native_goal": native_goal / "learned_isaac_smoke.mp4",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    clips = {name: Clip.read(path) for name, path in paths.items()}

    jitter = pair_renderer(
        "Same reset, frozen jitter: aligned recovers official Async failure",
        "LIBERO-Object task 5 · initial state 40 · seed 2026082140 · 600 ± 400 ms",
        (
            PanelSpec(clips["async_jitter"], "Official Async", "FAIL · 300 steps", RED),
            PanelSpec(
                clips["aligned_jitter"],
                "ActionStream aligned",
                "SUCCESS · 134 steps",
                GREEN,
            ),
        ),
        "Aggregate jitter cell: 8/15 → 14/15 success; paired difference +40.0 pp [13.3, 66.7].",
    )
    fixed_950 = pair_renderer(
        "Same reset, fixed 950 ms: the primary failure boundary",
        "LIBERO-Goal task 2 · initial state 40 · seed 2026082340",
        (
            PanelSpec(
                clips["latest_950"],
                "Official latest-only",
                "SUCCESS · 112 steps",
                GREEN,
            ),
            PanelSpec(
                clips["aligned_950"], "ActionStream aligned", "FAIL · 300 steps", RED
            ),
        ),
        "Aggregate fixed-950 cell: latest-only 14/15 vs aligned 10/15; no high-latency robustness claim.",
    )
    burst = pair_renderer(
        "Burst/outage: aligned gains one success, guarded over-corrects",
        "LIBERO-Object task 5 · same reset/state/trace",
        (
            PanelSpec(clips["latest_burst"], "Latest-only", "SUCCESS · 175", GREEN),
            PanelSpec(clips["aligned_burst"], "Aligned", "SUCCESS · 227", LIGHT_BLUE),
            PanelSpec(clips["guarded_burst"], "Guarded", "FAIL · 300", RED),
        ),
        "Aggregate: latest-only 12/15 · aligned 13/15 · guarded 8/15. Guarded is a formal NO-GO.",
    )
    native = pair_renderer(
        "Native learned-policy Isaac development evidence",
        "Real X-VLA CUDA inference drives native Franka; reset 0 only, not a holdout",
        (
            PanelSpec(clips["native_object"], "Object0", "PASS · 0 N", GREEN),
            PanelSpec(clips["native_spatial"], "Spatial2", "PASS · 0 N", GREEN),
            PanelSpec(
                clips["native_goal"], "Goal2", "TASK PASS · SAFETY NO-GO", ORANGE
            ),
        ),
        "Goal2 reached the predicate but hit 96.46 N against the frozen 40 N collision limit.",
    )
    segments = (
        (4.0, title_frame),
        (8.0, architecture_frame),
        (12.0, jitter),
        (12.0, fixed_950),
        (12.0, burst),
        (12.0, native),
        (8.0, results_frame),
        (6.0, outro_frame),
    )
    output = MEDIA_ROOT / "actionstream_v1_1_demo.mp4"
    frame_count = write_video(output, segments)
    video = verify_video(output, frame_count)
    poster = jitter(int(12.0 * FPS * 0.72), int(12.0 * FPS))
    poster_path = MEDIA_ROOT / "actionstream_v1_1_demo_poster.png"
    poster.save(poster_path)

    report_files = [
        ROOT / "reports" / "actionstream_backend_gpu_v1" / "formal_xvla" / name
        for name in (
            "report.md",
            "summary.json",
            "main_table.csv",
            "paired_effects.csv",
            "secondary_paired_effects.csv",
            "latency_success_operating_points.png",
        )
    ] + [
        ROOT
        / "reports"
        / "actionstream_backend_gpu_v1"
        / "smolvla_rtc_gate_v2_v3"
        / name
        for name in ("report.md", "summary.json", "sync_gate_by_suite.png")
    ]
    manifest = {
        "schema_version": 1,
        "release": "v1.1.0",
        "content_boundary": (
            "All comparison footage comes from frozen same-reset result cells. Native Isaac clips are development reset-0 evidence. "
            "Goal2 remains a safety NO-GO; SmolVLA formal RTC was not run after its sync-gate NO-GO; no real-robot or Arena claim is made."
        ),
        "inputs": [
            {
                "role": name,
                "path": path.relative_to(evidence_root).as_posix(),
                "sha256": sha256(path),
            }
            for name, path in sorted(paths.items())
        ],
        "report_inputs": [
            {"path": relative(path), "sha256": sha256(path)} for path in report_files
        ],
        "outputs": [
            {"path": relative(output), "sha256": sha256(output)},
            {"path": relative(poster_path), "sha256": sha256(poster_path)},
        ],
        "video": video,
    }
    manifest_path = RELEASE_ROOT / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Saved {relative(output)}")
    print(f"Saved {relative(poster_path)}")
    print(f"Saved {relative(manifest_path)}")


if __name__ == "__main__":
    main()
