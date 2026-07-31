"""Build the captioned ActionStream v1.0.0 demonstration video."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Callable, Iterable

import imageio_ffmpeg
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = ROOT / "release" / "v1.0.0"
MEDIA_ROOT = RELEASE_ROOT / "media"
SOURCE_ROOT = MEDIA_ROOT / "source"
FIGURE_ROOT = RELEASE_ROOT / "figures"

WIDTH = 1280
HEIGHT = 720
FPS = 30
BACKGROUND = (10, 16, 31)
PANEL = (20, 29, 50)
WHITE = (245, 247, 250)
MUTED = (172, 184, 205)
GRID = (61, 75, 101)
BLUE = (0, 114, 178)
LIGHT_BLUE = (86, 180, 233)
GREEN = (0, 158, 115)
ORANGE = (230, 159, 0)
RED = (213, 94, 0)
GRAY = (107, 114, 128)

FONT_REGULAR_PATH = font_manager.findfont(
    font_manager.FontProperties(family="DejaVu Sans", weight="normal")
)
FONT_BOLD_PATH = font_manager.findfont(
    font_manager.FontProperties(family="DejaVu Sans", weight="bold")
)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_REGULAR_PATH, size)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def new_canvas() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    *,
    fill: tuple[int, int, int] = WHITE,
    spacing: int = 8,
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


def fit_image(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(box, Image.Resampling.LANCZOS)
    return copy


def fade_frame(image: Image.Image, index: int, count: int, fade: int = 10) -> Image.Image:
    alpha = 1.0
    if index < fade:
        alpha = max(0.0, index / fade)
    if index >= count - fade:
        alpha = min(alpha, max(0.0, (count - 1 - index) / fade))
    if alpha >= 1.0:
        return image
    return Image.blend(new_canvas(), image, alpha)


def read_video(path: Path) -> list[np.ndarray]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    frames: list[np.ndarray] = []
    try:
        for payload in reader:
            frame = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
            frames.append(frame.copy())
    finally:
        reader.close()
    if not frames:
        raise AssertionError(f"No frames decoded from {path}")
    return frames


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_episode(path: Path, *, task_id: int, episode_index: int) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["task_id"] == task_id and row["episode_index"] == episode_index:
            return row
    raise AssertionError(f"Episode task={task_id}, index={episode_index} not found in {path}")


def title_frame(index: int, count: int) -> Image.Image:
    image = new_canvas()
    draw = ImageDraw.Draw(image)
    progress = index / max(1, count - 1)
    accent_width = int(240 + 520 * progress)
    draw.rounded_rectangle(
        (WIDTH // 2 - accent_width // 2, 176, WIDTH // 2 + accent_width // 2, 188),
        radius=6,
        fill=BLUE,
    )
    draw_centered(draw, (WIDTH // 2, 285), "ActionStream", font(72, bold=True))
    draw_centered(
        draw,
        (WIDTH // 2, 380),
        "Age-aligned asynchronous action chunks\nunder control latency",
        font(34),
        fill=(220, 227, 239),
        spacing=12,
    )
    draw_centered(
        draw,
        (WIDTH // 2, 520),
        "Frozen X-VLA  ·  LIBERO Object  ·  v1.0.0",
        font(22, bold=True),
        fill=LIGHT_BLUE,
    )
    return fade_frame(image, index, count, fade=12)


def make_task_context_renderer(
    clips: list[list[np.ndarray]],
) -> Callable[[int, int], Image.Image]:
    descriptions = (
        ("Task 0", "alphabet soup → basket"),
        ("Task 1", "cream cheese → basket"),
        ("Task 2", "salad dressing → basket"),
    )
    x_positions = (70, 460, 850)

    def render(index: int, count: int) -> Image.Image:
        image = new_canvas()
        draw = ImageDraw.Draw(image)
        draw_centered(
            draw,
            (WIDTH // 2, 52),
            "Official synchronous baseline: illustrative task context",
            font(27, bold=True),
        )
        draw_centered(
            draw,
            (WIDTH // 2, 92),
            "These are not recordings of the M4 runtime modes.",
            font(17),
            fill=MUTED,
        )
        for panel_index, (clip, (task, description), x) in enumerate(
            zip(clips, descriptions, x_positions, strict=True)
        ):
            source_index = min(int(index * 40 / FPS), len(clip) - 1)
            frame = Image.fromarray(clip[source_index], mode="RGB")
            frame = frame.resize((360, 360), Image.Resampling.LANCZOS)
            draw.rounded_rectangle(
                (x - 6, 151, x + 366, 523),
                radius=12,
                fill=PANEL,
                outline=(BLUE, GREEN, ORANGE)[panel_index],
                width=3,
            )
            image.paste(frame, (x, 157))
            draw_centered(
                draw,
                (x + 180, 563),
                task,
                font(22, bold=True),
                fill=(LIGHT_BLUE, GREEN, ORANGE)[panel_index],
            )
            draw_centered(draw, (x + 180, 598), description, font(18))
        draw_centered(
            draw,
            (WIDTH // 2, 668),
            "Policy: lerobot/xvla-libero  ·  absolute control  ·  final 7D commands",
            font(16),
            fill=MUTED,
        )
        return fade_frame(image, index, count)

    return render


def make_mechanism_renderer(delivery_age: float, chunk_size: int) -> Callable[[int, int], Image.Image]:
    row_y = {"sync_hold": 235, "async_naive": 365, "async_aligned": 495}
    labels = {
        "sync_hold": ("Sync hold", GRAY),
        "async_naive": ("Naive async", ORANGE),
        "async_aligned": ("Aligned async", BLUE),
    }
    cell_width = 22
    gap = 2
    cells_x = 405

    def render(index: int, count: int) -> Image.Image:
        image = new_canvas()
        draw = ImageDraw.Draw(image)
        progress = min(1.0, index / max(1, int(count * 0.70)))
        current_age = delivery_age * progress
        stale_steps = min(chunk_size, int(round(current_age)))

        draw_centered(
            draw,
            (WIDTH // 2, 50),
            "What alignment changes when a chunk arrives late",
            font(28, bold=True),
        )
        draw.line((405, 125, 1120, 125), fill=MUTED, width=3)
        arrow_x = int(405 + 715 * progress)
        draw.ellipse((arrow_x - 8, 117, arrow_x + 8, 133), fill=ORANGE)
        draw.text((405, 90), "Observation captured", font=font(16), fill=MUTED)
        draw.text((1000, 90), "Chunk delivered", font=font(16), fill=MUTED)
        draw_centered(
            draw,
            (WIDTH // 2, 160),
            f"950 ms pressure point  ·  effective age ≈ {current_age:.1f} control steps",
            font(18, bold=True),
            fill=LIGHT_BLUE,
        )

        for mode in ("sync_hold", "async_naive", "async_aligned"):
            y = row_y[mode]
            mode_label, color = labels[mode]
            draw.text((65, y - 9), mode_label, font=font(22, bold=True), fill=color)

            for step in range(chunk_size):
                x0 = cells_x + step * (cell_width + gap)
                box = (x0, y - 17, x0 + cell_width, y + 17)
                if mode == "sync_hold":
                    fill = PANEL
                    outline = RED if progress > 0.15 else GRID
                elif mode == "async_naive":
                    fill = RED if step < stale_steps else ORANGE
                    outline = WHITE
                else:
                    fill = PANEL if step < stale_steps else BLUE
                    outline = RED if step < stale_steps else WHITE
                draw.rounded_rectangle(box, radius=3, fill=fill, outline=outline, width=1)
                if mode == "async_aligned" and step < stale_steps:
                    draw.line((x0 + 4, y - 11, x0 + cell_width - 4, y + 11), fill=RED, width=2)
                    draw.line((x0 + 4, y + 11, x0 + cell_width - 4, y - 11), fill=RED, width=2)

            if mode == "sync_hold":
                status = "queue empty → repeat the previous finite 7D command"
            elif mode == "async_naive":
                status = f"accept all {chunk_size} actions, including {stale_steps} stale-prefix steps"
            else:
                status = f"drop {stale_steps} stale-prefix steps; keep {chunk_size - stale_steps}"
            draw.text((405, y + 32), status, font=font(16), fill=MUTED)

        draw_centered(
            draw,
            (WIDTH // 2, 660),
            "Mechanism illustration; the displayed age comes from the frozen calibration evidence.",
            font(16),
            fill=MUTED,
        )
        return fade_frame(image, index, count)

    return render


def make_figure_renderer(
    figure_path: Path,
    *,
    heading: str,
    footer: str,
    accent: tuple[int, int, int],
) -> Callable[[int, int], Image.Image]:
    source = Image.open(figure_path).convert("RGB")
    fitted = fit_image(source, (1180, 555))

    def render(index: int, count: int) -> Image.Image:
        image = new_canvas()
        draw = ImageDraw.Draw(image)
        draw_centered(draw, (WIDTH // 2, 40), heading, font(25, bold=True))
        x = (WIDTH - fitted.width) // 2
        y = 75 + (555 - fitted.height) // 2
        draw.rounded_rectangle(
            (x - 8, y - 8, x + fitted.width + 8, y + fitted.height + 8),
            radius=12,
            fill=(255, 255, 255),
            outline=accent,
            width=2,
        )
        image.paste(fitted, (x, y))
        draw_centered(draw, (WIDTH // 2, 685), footer, font(17, bold=True), fill=accent)
        return fade_frame(image, index, count)

    return render


def make_telemetry_renderer(telemetry: list[dict]) -> Callable[[int, int], Image.Image]:
    max_steps = 800
    x0, x1 = 275, 1195
    plot_width = x1 - x0
    row_centers = (225, 390, 555)

    def render(index: int, count: int) -> Image.Image:
        image = new_canvas()
        draw = ImageDraw.Draw(image)
        progress_steps = int(max_steps * index / max(1, count - 1))
        draw_centered(
            draw,
            (WIDTH // 2, 44),
            "Representative paired episode: real frozen queue telemetry",
            font(26, bold=True),
        )
        draw_centered(
            draw,
            (WIDTH // 2, 80),
            "Task 0  ·  initial state 4  ·  seed 144  ·  950 ms",
            font(17),
            fill=MUTED,
        )

        for row_index, item in enumerate(telemetry):
            center = row_centers[row_index]
            color = item["color"]
            depths = item["depths"]
            holds = item["holds"]
            row = item["episode"]
            draw.text((55, center - 28), item["label"], font=font(21, bold=True), fill=color)
            result = "success" if row["success"] else "failure / timeout"
            result_color = GREEN if row["success"] else RED
            draw.text((55, center + 8), result, font=font(16, bold=True), fill=result_color)
            draw.text(
                (55, center + 34),
                f'{row["environment_steps"]} steps · {row["wall_clock_episode_seconds"]:.2f} s',
                font=font(14),
                fill=MUTED,
            )

            top = center - 50
            bottom = center + 50
            draw.rectangle((x0, top, x1, bottom), fill=PANEL, outline=GRID, width=1)
            for depth in (0, 10, 20, 30):
                y = bottom - int(depth / 30 * (bottom - top))
                draw.line((x0, y, x1, y), fill=GRID, width=1)
            draw.text((x0 + 5, top + 4), "queue depth", font=font(12), fill=MUTED)

            visible = min(progress_steps, len(depths))
            if visible > 1:
                stride = max(1, visible // 450)
                indices = list(range(0, visible, stride))
                if indices[-1] != visible - 1:
                    indices.append(visible - 1)
                points = [
                    (
                        x0 + int(step / max_steps * plot_width),
                        bottom - int(float(depths[step]) / 30 * (bottom - top)),
                    )
                    for step in indices
                ]
                draw.line(points, fill=color, width=3)

            hold_indices = np.flatnonzero(holds[:visible])
            for step in hold_indices[:: max(1, len(hold_indices) // 120 or 1)]:
                x = x0 + int(int(step) / max_steps * plot_width)
                draw.line((x, bottom - 6, x, bottom), fill=RED, width=2)

            endpoint = min(len(depths), progress_steps)
            if endpoint >= len(depths):
                endpoint_x = x0 + int(len(depths) / max_steps * plot_width)
                draw.ellipse(
                    (endpoint_x - 5, center - 5, endpoint_x + 5, center + 5),
                    fill=result_color,
                )

        draw.line((x0, 665, x1, 665), fill=MUTED, width=2)
        for tick in (0, 200, 400, 600, 800):
            x = x0 + int(tick / max_steps * plot_width)
            draw.line((x, 660, x, 670), fill=MUTED, width=2)
            draw_centered(draw, (x, 692), str(tick), font(13), fill=MUTED)
        draw_centered(draw, ((x0 + x1) // 2, 640), "control steps", font(14), fill=MUTED)
        return fade_frame(image, index, count)

    return render


def make_outro_renderer(report: dict) -> Callable[[int, int], Image.Image]:
    claim = report["claim_statuses"]
    success = claim["success"]
    success_low, success_high = success["paired_mean_difference_95pct_bootstrap_ci"]
    efficiency = claim["efficiency"]["metrics"]
    step_effect = efficiency["environment_steps"]["paired_mean_difference"]
    wall_effect = efficiency["wall_clock_episode_seconds"]["paired_mean_difference"]

    def render(index: int, count: int) -> Image.Image:
        image = new_canvas()
        draw = ImageDraw.Draw(image)
        draw_centered(draw, (WIDTH // 2, 62), "ActionStream v1.0.0", font(38, bold=True))

        draw.rounded_rectangle((80, 125, 620, 590), radius=18, fill=PANEL, outline=GREEN, width=3)
        draw.text((120, 165), "Supported vs naive async", font=font(25, bold=True), fill=GREEN)
        draw.text(
            (120, 235),
            (
                f"Success: +{success['paired_mean_difference'] * 100:.1f} pp\n"
                f"95% CI [{success_low * 100:.1f}, {success_high * 100:.1f}]\n\n"
                f"Observed episode efficiency:\n"
                f"{step_effect:.1f} steps\n"
                f"{wall_effect:.2f} s"
            ),
            font=font(21),
            fill=WHITE,
            spacing=9,
        )

        draw.rounded_rectangle((660, 125, 1200, 590), radius=18, fill=PANEL, outline=ORANGE, width=3)
        draw.text((700, 165), "Boundaries", font=font(25, bold=True), fill=ORANGE)
        draw.text(
            (700, 235),
            (
                "No demonstrated success\n"
                "superiority vs sync hold\n\n"
                "Underrun evidence is mixed\n"
                "by comparator\n\n"
                "Fully stale rejection is tested;\n"
                "no event was observed"
            ),
            font=font(21),
            fill=WHITE,
            spacing=9,
        )
        draw_centered(
            draw,
            (WIDTH // 2, 655),
            "Frozen evidence · reproducible release assets · no training",
            font(18, bold=True),
            fill=LIGHT_BLUE,
        )
        return fade_frame(image, index, count, fade=12)

    return render


def write_video(
    output: Path,
    segments: Iterable[tuple[float, Callable[[int, int], Image.Image]]],
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
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
        "title=ActionStream v1.0.0",
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
                frame = renderer(index, count).convert("RGB")
                array = np.asarray(frame, dtype=np.uint8)
                if array.shape != (HEIGHT, WIDTH, 3):
                    raise AssertionError(f"Unexpected frame shape: {array.shape}")
                process.stdin.write(array.tobytes())
                frame_count += 1
    finally:
        process.stdin.close()
    assert process.stderr is not None
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed ({return_code}): {stderr}")
    return frame_count


def verify_video(path: Path, expected_frames: int) -> dict:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    metadata = next(reader)
    frame_count = 0
    try:
        for _ in reader:
            frame_count += 1
    finally:
        reader.close()
    if metadata["size"] != (WIDTH, HEIGHT):
        raise AssertionError(f"Unexpected video size: {metadata['size']}")
    if abs(float(metadata["fps"]) - FPS) > 0.01:
        raise AssertionError(f"Unexpected video fps: {metadata['fps']}")
    if frame_count != expected_frames:
        raise AssertionError(f"Decoded {frame_count} frames; expected {expected_frames}")
    return {
        "codec": metadata["codec"],
        "pixel_format": metadata["pix_fmt"],
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frame_count": frame_count,
        "duration_seconds": frame_count / FPS,
        "ffmpeg_version": imageio_ffmpeg.get_ffmpeg_version(),
    }


def main() -> None:
    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "outputs" / "m4" / "report" / "m4_report.json"
    plan_path = (
        ROOT / "outputs" / "m4" / "calibration" / "calibration_plan.json"
    )
    selection_path = ROOT / "outputs" / "m4" / "calibration" / "selection.json"
    report = load_json(report_path)
    plan = load_json(plan_path)
    selection = load_json(selection_path)
    assert report["status"] == "validated"
    assert report["validation"]["passed"] is True
    assert selection["status"] == "selected"
    assert selection["selected_pressure_delay_ms"] == 950
    selected_decision = next(
        item
        for item in selection["delay_decisions"]
        if item["injected_delay_ms"] == selection["selected_pressure_delay_ms"]
    )

    source_video_paths = (
        SOURCE_ROOT / "official_baseline_task0_episode2.mp4",
        SOURCE_ROOT / "official_baseline_task1_episode8.mp4",
        SOURCE_ROOT / "official_baseline_task2_episode3.mp4",
    )
    clips = [read_video(path) for path in source_video_paths]

    pressure_root = ROOT / "outputs" / "m4" / "pressure"
    telemetry_specs = (
        (
            "Sync hold",
            GRAY,
            pressure_root / "sync_hold_delay950" / "episodes.jsonl",
            pressure_root
            / "sync_hold_delay950"
            / "traces"
            / "sync_hold_delay950_task0_episode2.npz",
        ),
        (
            "Naive async",
            ORANGE,
            pressure_root / "async_naive_delay950" / "episodes.jsonl",
            pressure_root
            / "async_naive_delay950"
            / "traces"
            / "async_naive_delay950_task0_episode2.npz",
        ),
        (
            "Aligned async",
            BLUE,
            pressure_root / "async_aligned_delay950" / "episodes.jsonl",
            pressure_root
            / "async_aligned_delay950"
            / "traces"
            / "async_aligned_delay950_task0_episode2.npz",
        ),
    )
    telemetry: list[dict] = []
    for label, color, episode_path, trace_path in telemetry_specs:
        episode = find_episode(episode_path, task_id=0, episode_index=2)
        assert episode["initial_state_index"] == 4
        assert episode["seed"] == 144
        trace = np.load(trace_path, allow_pickle=False)
        depths = trace["queue_depth_after_action"]
        holds = trace["queue_hold_mask"]
        assert len(depths) == episode["environment_steps"]
        assert int(np.sum(holds)) == episode["queue_underrun_hold_steps"]
        telemetry.append(
            {
                "label": label,
                "color": color,
                "episode": episode,
                "depths": depths,
                "holds": holds,
                "episode_path": episode_path,
                "trace_path": trace_path,
            }
        )

    calibration_renderer = make_figure_renderer(
        FIGURE_ROOT / "queue_pressure_calibration.png",
        heading="Calibrating a genuine queue-pressure point",
        footer="950 ms was the smallest tested delay that crossed the 20-step headroom.",
        accent=ORANGE,
    )
    headline_renderer = make_figure_renderer(
        FIGURE_ROOT / "m4_pressure_headline.png",
        heading="Full pressure evaluation: 30 paired episodes per mode",
        footer="Aligned async recovered success and observed episode efficiency vs naive async.",
        accent=BLUE,
    )
    outro_renderer = make_outro_renderer(report)
    segments = (
        (2.5, title_frame),
        (3.5, make_task_context_renderer(clips)),
        (
            5.5,
            make_mechanism_renderer(
                selected_decision["pooled_effective_delivery_age_steps"]["median"],
                plan["runtime_configuration"]["chunk_size"],
            ),
        ),
        (6.0, calibration_renderer),
        (8.0, headline_renderer),
        (6.5, make_telemetry_renderer(telemetry)),
        (6.0, outro_renderer),
    )

    output_path = MEDIA_ROOT / "actionstream_v1_demo.mp4"
    frame_count = write_video(output_path, segments)
    video_info = verify_video(output_path, frame_count)

    poster = headline_renderer(int(8.0 * FPS * 0.55), int(8.0 * FPS))
    poster_path = MEDIA_ROOT / "actionstream_v1_demo_poster.png"
    poster.save(poster_path)

    input_paths = [
        report_path,
        plan_path,
        selection_path,
        *source_video_paths,
        *(item["episode_path"] for item in telemetry),
        *(item["trace_path"] for item in telemetry),
        FIGURE_ROOT / "queue_pressure_calibration.png",
        FIGURE_ROOT / "m4_pressure_headline.png",
    ]
    output_paths = [output_path, poster_path]
    artifact_paths = [
        *sorted(FIGURE_ROOT.glob("*.pdf")),
        *sorted(FIGURE_ROOT.glob("*.png")),
        RELEASE_ROOT / "evidence_summary.json",
        RELEASE_ROOT / "RELEASE_NOTES.md",
        RELEASE_ROOT / "frozen_git_objects.json",
        ROOT / "requirements-release.txt",
        ROOT / "THIRD_PARTY_NOTICES.md",
    ]
    manifest = {
        "schema_version": 1,
        "release": "v1.0.0",
        "content_boundary": (
            "Task footage is from the official synchronous baseline and is "
            "illustrative only. M4 charts and telemetry are derived from frozen evidence."
        ),
        "inputs": [
            {"path": relative(path), "sha256": sha256(path)} for path in input_paths
        ],
        "outputs": [
            {"path": relative(path), "sha256": sha256(path)} for path in output_paths
        ],
        "artifacts": [
            {"path": relative(path), "sha256": sha256(path)}
            for path in artifact_paths
        ],
        "video": video_info,
    }
    manifest_path = RELEASE_ROOT / "asset_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Saved {relative(output_path)}")
    print(f"Saved {relative(poster_path)}")
    print(f"Saved {relative(manifest_path)}")


if __name__ == "__main__":
    main()
