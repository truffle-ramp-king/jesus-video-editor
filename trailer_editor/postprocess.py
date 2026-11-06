"""Background composition with an imperceptible deduplication perturbation."""

from __future__ import annotations

import logging
from pathlib import Path

from trailer_editor.preprocessor import run_command

logger = logging.getLogger(__name__)

MAIN_SCALE = 0.85
# Match the known-good VideoDeduplicator: full gray noise blended at 1% opacity.
DEDUP_NOISE_PERCENT = 1.0
BEIGE_OVERLAY_OPACITY = 0.05
BEIGE_OVERLAY_SCALE = 1.5
BEIGE_OVERLAY_PATH = Path(__file__).with_name("assets") / "beige-overlay.mp4"


def ffprobe_value(path: Path, entries: str, *, video_stream: bool = False) -> str:
    command = ["ffprobe", "-v", "error"]
    if video_stream:
        command += ["-select_streams", "v:0"]
    result = run_command(command + [
        "-show_entries", entries,
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    return result.stdout.strip()


def media_duration(path: Path) -> float:
    return float(ffprobe_value(path, "format=duration"))


def video_dimensions(path: Path) -> tuple[int, int]:
    width, height = ffprobe_value(
        path, "stream=width,height", video_stream=True
    ).splitlines()
    return int(width), int(height)


def video_frame_rate(path: Path) -> float:
    numerator, denominator = ffprobe_value(
        path, "stream=avg_frame_rate", video_stream=True
    ).split("/")
    return int(numerator) / int(denominator)


def make_even(value: int) -> int:
    return value - value % 2


def scaled_main_size(
    background_width: int,
    background_height: int,
    scale: float = MAIN_SCALE,
) -> tuple[int, int]:
    return (
        make_even(int(background_width * scale)),
        make_even(int(background_height * scale)),
    )


def build_filtergraph(
    source_width: int,
    source_height: int,
    background_width: int,
    background_height: int,
    *,
    fps: float,
    duration: float,
    noise_percent: float = DEDUP_NOISE_PERCENT,
    beige_opacity: float = BEIGE_OVERLAY_OPACITY,
    beige_scale: float = BEIGE_OVERLAY_SCALE,
) -> str:
    """Add noise, background, and a reusable moving beige overlay."""
    foreground_width, foreground_height = scaled_main_size(
        background_width, background_height
    )
    x = (background_width - foreground_width) // 2
    y = (background_height - foreground_height) // 2
    opacity = noise_percent / 100
    return ";".join([
        "[0:v]setpts=PTS-STARTPTS[foreground]",
        f"color=c=gray:s={source_width}x{source_height}:r={fps:.3f},"
        "noise=alls=100:allf=t+u[noise]",
        f"[foreground][noise]blend=all_mode=overlay:"
        f"all_opacity={opacity:.5f}:shortest=1[deduplicated]",
        f"[deduplicated]scale={foreground_width}:{foreground_height}[main]",
        f"[1:v]fps={fps:.3f},setpts=PTS-STARTPTS[background]",
        f"[background][main]overlay={x}:{y}:shortest=1[composite]",
        f"[2:v]fps={fps:.3f},setpts=PTS-STARTPTS,"
        f"scale='trunc(iw*{beige_scale}/2)*2':'trunc(ih*{beige_scale}/2)*2',"
        "format=rgba,"
        f"colorchannelmixer=aa={beige_opacity:.5f},"
        "rotate=PI/4:ow=rotw(PI/4):oh=roth(PI/4):c=none[beige]",
        f"[composite][beige]overlay="
        f"x='-w+(W+w)*t/{duration:.6f}':"
        f"y='-h+(H+h)*t/{duration:.6f}':shortest=1,"
        f"fps={fps:.3f},setsar=1,format=yuv420p[final]",
    ])


def post_process_video(
    main_path: Path,
    background_path: Path,
    output_path: Path,
    *,
    music_path: Path | None = None,
    beige_path: Path = BEIGE_OVERLAY_PATH,
    fps: float | None = None,
) -> Path:
    """Composite foreground, background, optional music, and subtle noise."""
    duration = media_duration(main_path)
    source_width, source_height = video_dimensions(main_path)
    fps = fps or video_frame_rate(main_path)
    background_width, background_height = video_dimensions(background_path)
    if not beige_path.is_file():
        raise ValueError(f"Reusable beige overlay is missing: {beige_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg", "-y",
        "-i", str(main_path),
        "-stream_loop", "-1", "-i", str(background_path),
        "-stream_loop", "-1", "-i", str(beige_path),
    ]
    graph = build_filtergraph(
        source_width,
        source_height,
        background_width,
        background_height,
        fps=fps,
        duration=duration,
    )
    audio_map = "0:a"
    if music_path:
        command += ["-stream_loop", "-1", "-i", str(music_path)]
        graph += (
            ";[0:a]aresample=async=1[source_audio];"
            "[3:a]aresample=async=1[music];"
            "[source_audio][music]amix=inputs=2:duration=first[mixed]"
        )
        audio_map = "[mixed]"
    command += [
        "-filter_complex", graph,
        "-map", "[final]",
        "-map", audio_map,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-movflags", "+faststart",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    logger.info("Rendering final composition to %s", output_path)
    run_command(command)
    return output_path
