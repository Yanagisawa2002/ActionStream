"""Build the 60--90 second ActionStream v1.1 systems-result demo.

The demo deliberately separates the H1-R2 delivery-pipeline mechanism result
from the H2 fixed-budget NO-GO.  Source videos remain outside ordinary Git;
only the compressed demo, poster, and a content-addressed manifest are emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps


WIDTH = 1280
HEIGHT = 720
FPS = 20
BACKGROUND = "#07111f"
PANEL = "#10253a"
TEXT = "#edf5ff"
MUTED = "#a9bdd2"
GREEN = "#35d07f"
RED = "#ff6b6b"
AMBER = "#ffca58"
BLUE = "#58a6ff"


def _tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"Required executable is unavailable: {name}")
    return resolved


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (
        ["C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise RuntimeError("No usable TrueType font found")


def _fit_visual(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#ffffff")
    canvas.paste(
        fitted,
        ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2),
    )
    return canvas


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    font: ImageFont.FreeTypeFont,
    fill: str,
    width_chars: int,
    spacing: int = 8,
) -> int:
    wrapped = "\n".join(textwrap.wrap(text, width=width_chars))
    draw.multiline_text(xy, wrapped, font=font, fill=fill, spacing=spacing)
    box = draw.multiline_textbbox(xy, wrapped, font=font, spacing=spacing)
    return box[3]


def _card(
    path: Path,
    *,
    eyebrow: str,
    title: str,
    bullets: list[tuple[str, str]],
    footer: str,
    visual: Path | None = None,
) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((34, 28, WIDTH - 34, HEIGHT - 28), 28, fill=PANEL)
    draw.text((64, 54), eyebrow.upper(), font=_font(22, bold=True), fill=BLUE)
    draw.text((64, 91), title, font=_font(47, bold=True), fill=TEXT)

    y = 174
    body_font = _font(27)
    for color, line in bullets:
        draw.ellipse((66, y + 8, 82, y + 24), fill=color)
        y = (
            _draw_wrapped(
                draw,
                line,
                (98, y),
                font=body_font,
                fill=TEXT,
                width_chars=31 if visual is not None else 67,
            )
            + 23
        )

    if visual is not None:
        fitted = _fit_visual(visual, (690, 478))
        image.paste(fitted, (530, 158))
        draw.rounded_rectangle((526, 154, 1224, 640), 18, outline=BLUE, width=3)

    draw.line((64, 654, WIDTH - 64, 654), fill="#29445d", width=2)
    draw.text((64, 671), footer, font=_font(19), fill=MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _pair_header(
    path: Path,
    *,
    left: tuple[str, str],
    right: tuple[str, str],
    footer: str,
) -> None:
    image = Image.new("RGB", (WIDTH, 80), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 80), fill=PANEL)
    draw.line((WIDTH // 2, 0, WIDTH // 2, 53), fill="#29445d", width=2)
    draw.text((24, 8), left[0], font=_font(25, bold=True), fill=left[1])
    draw.text((WIDTH // 2 + 24, 8), right[0], font=_font(25, bold=True), fill=right[1])
    draw.text((24, 54), footer, font=_font(17), fill=MUTED)
    image.save(path)


def _still_segment(ffmpeg: str, image: Path, duration: float, output: Path) -> None:
    _run(
        [
            ffmpeg,
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(FPS),
            "-i",
            str(image),
            "-t",
            str(duration),
            "-vf",
            f"scale={WIDTH}:{HEIGHT},fps={FPS},format=yuv420p",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            str(output),
        ]
    )


def _pair_segment(
    ffmpeg: str,
    *,
    left_video: Path,
    right_video: Path,
    header: Path,
    duration: float,
    slow_factor: float,
    output: Path,
) -> None:
    filter_graph = (
        f"[0:v]scale=640:640:force_original_aspect_ratio=decrease,"
        f"pad=640:640:(ow-iw)/2:(oh-ih)/2:black,"
        f"setpts=(PTS-STARTPTS)*{slow_factor},"
        f"tpad=stop_mode=clone:stop_duration=30,trim=duration={duration}[left];"
        f"[1:v]scale=640:640:force_original_aspect_ratio=decrease,"
        f"pad=640:640:(ow-iw)/2:(oh-ih)/2:black,"
        f"setpts=(PTS-STARTPTS)*{slow_factor},"
        f"tpad=stop_mode=clone:stop_duration=30,trim=duration={duration}[right];"
        f"[2:v]scale=1280:80,trim=duration={duration},setpts=PTS-STARTPTS[header];"
        "[left][right]hstack=inputs=2[pair];"
        f"[header][pair]vstack=inputs=2,trim=duration={duration},"
        f"fps={FPS},format=yuv420p[out]"
    )
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(left_video),
            "-i",
            str(right_video),
            "-loop",
            "1",
            "-framerate",
            str(FPS),
            "-i",
            str(header),
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            str(output),
        ]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_duration(ffprobe: str, path: Path) -> float:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h1-serialized-video", type=Path, required=True)
    parser.add_argument("--h1-pipelined-video", type=Path, required=True)
    parser.add_argument("--h1-figure", type=Path, required=True)
    parser.add_argument("--h2-unbounded-video", type=Path, required=True)
    parser.add_argument("--h2-budgeted-video", type=Path, required=True)
    parser.add_argument("--h2-figure", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    sources = {
        "h1_serialized_video": args.h1_serialized_video.resolve(),
        "h1_pipelined_video": args.h1_pipelined_video.resolve(),
        "h1_figure": args.h1_figure.resolve(),
        "h2_unbounded_video": args.h2_unbounded_video.resolve(),
        "h2_budgeted_video": args.h2_budgeted_video.resolve(),
        "h2_figure": args.h2_figure.resolve(),
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing demo inputs: {missing}")

    ffmpeg = _tool("ffmpeg")
    ffprobe = _tool("ffprobe")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "actionstream_v1_1_demo.mp4"
    poster = output_dir / "actionstream_v1_1_demo_poster.png"
    manifest_path = output_dir / "asset_manifest.json"

    with tempfile.TemporaryDirectory(prefix="actionstream-demo-") as tmp:
        work = Path(tmp)
        intro = work / "intro.png"
        h1_header = work / "h1_header.png"
        h1_result = work / "h1_result.png"
        h2_header = work / "h2_header.png"
        h2_result = work / "h2_result.png"
        outro = work / "outro.png"

        _card(
            intro,
            eyebrow="ActionStream v1.1",
            title="Learned-policy GPU runtime under delay",
            bullets=[
                (BLUE, "X-VLA on three LIBERO task families, one RTX 5090"),
                (GREEN, "H1-R2: delivery-pipeline mechanism GO"),
                (RED, "H2: fixed compute-budget replacement NO-GO"),
            ],
            footer="Frozen paired protocols · fixed 950 ms · simulator evidence, not real-robot safety",
        )
        _pair_header(
            h1_header,
            left=("Serialized aligned — FAIL @ 300 steps", RED),
            right=("Pipelined aligned — SUCCESS @ 123 steps", GREEN),
            footer="H1-R2 Spatial · same task/reset/checkpoint · only delivery scheduling changed",
        )
        _card(
            h1_result,
            eyebrow="H1-R2 · mechanism result",
            title="Move delivery wait off the inference worker",
            bullets=[
                (GREEN, "Queue depletion: 46.6% → 0.0%"),
                (GREEN, "Median request supply: 11.43×"),
                (GREEN, "Success: 10/15 → 13/15"),
                (AMBER, "Cost: 7.93× more inference calls"),
                (AMBER, "Object family regressed 5/5 → 3/5"),
            ],
            visual=sources["h1_figure"],
            footer="30 scored episodes · paired success CI crosses zero · claim the pipeline bottleneck, not universal policy gain",
        )
        _pair_header(
            h2_header,
            left=("Unbounded pipeline — representative success", BLUE),
            right=("5-step budget — representative success", AMBER),
            footer="H2 Goal representative pair · decisive state-19 failure has trace only, not video",
        )
        _card(
            h2_result,
            eyebrow="H2 · compute boundary",
            title="Fewer requests did not preserve the success gate",
            bullets=[
                (GREEN, "Inference calls: 1,109 → 426 (−61.6%)"),
                (GREEN, "Queue depletion remained 0.0%"),
                (RED, "Success: 14/15 → 13/15"),
                (RED, "Mean steps: 125.67 → 139.80"),
                (AMBER, "Verdict: strict NO-GO"),
            ],
            visual=sources["h2_figure"],
            footer="The useful result is a measured compute/success frontier, not a superior default scheduler",
        )
        _card(
            outro,
            eyebrow="Evidence boundary",
            title="What the project proves — and does not",
            bullets=[
                (
                    GREEN,
                    "Proven: serialized delivery caused worker starvation at this operating point",
                ),
                (
                    GREEN,
                    "Proven: cancellable pipelining restored queue supply across three task families",
                ),
                (
                    AMBER,
                    "Bounded: a fixed request budget saved compute but lost one paired success",
                ),
                (
                    RED,
                    "Not claimed: RTC, native Isaac holdout, real RPC soak, or real-robot safety",
                ),
            ],
            footer="Reports, paired tables, telemetry, videos, and content hashes are retained with the release",
        )

        segment_specs = [
            ("intro", 6.0),
            ("h1_pair", 15.0),
            ("h1_result", 11.0),
            ("h2_pair", 10.0),
            ("h2_result", 11.0),
            ("outro", 9.0),
        ]
        segments = {
            name: work / f"{index:02d}_{name}.mp4"
            for index, (name, _) in enumerate(segment_specs)
        }
        _still_segment(ffmpeg, intro, 6.0, segments["intro"])
        _pair_segment(
            ffmpeg,
            left_video=sources["h1_serialized_video"],
            right_video=sources["h1_pipelined_video"],
            header=h1_header,
            duration=15.0,
            slow_factor=1.0,
            output=segments["h1_pair"],
        )
        _still_segment(ffmpeg, h1_result, 11.0, segments["h1_result"])
        _pair_segment(
            ffmpeg,
            left_video=sources["h2_unbounded_video"],
            right_video=sources["h2_budgeted_video"],
            header=h2_header,
            duration=10.0,
            slow_factor=2.0,
            output=segments["h2_pair"],
        )
        _still_segment(ffmpeg, h2_result, 11.0, segments["h2_result"])
        _still_segment(ffmpeg, outro, 9.0, segments["outro"])

        concat_list = work / "concat.txt"
        concat_list.write_text(
            "".join(
                f"file '{segments[name].as_posix()}'\n" for name, _ in segment_specs
            ),
            encoding="utf-8",
            newline="\n",
        )
        video_only = work / "video_only.mp4"
        _run(
            [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                str(video_only),
            ]
        )
        _run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video_only),
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-shortest",
                str(output),
            ]
        )

    duration = _probe_duration(ffprobe, output)
    if not 60.0 <= duration <= 90.0:
        raise RuntimeError(f"Demo duration {duration:.3f}s is outside [60, 90]")
    _run(
        [
            ffmpeg,
            "-y",
            "-ss",
            "15",
            "-i",
            str(output),
            "-frames:v",
            "1",
            str(poster),
        ]
    )

    manifest = {
        "schema_version": 1,
        "duration_seconds": duration,
        "frame_size": [WIDTH, HEIGHT],
        "fps": FPS,
        "claim_boundary": (
            "H1-R2 is a delivery-pipeline mechanism GO; H2 is a fixed-budget NO-GO. "
            "No RTC, native-Isaac holdout, remote-RPC soak, or real-robot claim."
        ),
        "sources": {
            role: {"filename": path.name, "sha256": _sha256(path)}
            for role, path in sources.items()
        },
        "outputs": {
            "demo": {
                "filename": output.name,
                "bytes": output.stat().st_size,
                "sha256": _sha256(output),
            },
            "poster": {
                "filename": poster.name,
                "bytes": poster.stat().st_size,
                "sha256": _sha256(poster),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
