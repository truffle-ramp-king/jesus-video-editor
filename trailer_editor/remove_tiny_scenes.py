"""Remove tiny scenes with a single reference-safe re-encode."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from scenedetect import ContentDetector, SceneManager, open_video

from trailer_editor.preprocessor import run_command

logger = logging.getLogger(__name__)
Scene = tuple[Any, Any]


def filter_scenes_by_duration(
    scenes: Iterable[Scene], min_duration_seconds: float
) -> list[Scene]:
    kept: list[Scene] = []
    for start, end in scenes:
        duration = (end - start).get_seconds()
        if duration >= min_duration_seconds:
            kept.append((start, end))
        else:
            logger.info("Dropping %.3fs scene at %s", duration, start.get_timecode())
    return kept


def build_concat_filter(scenes: list[Scene]) -> str:
    """Build trims from one decoder, preserving valid frame references."""
    filters: list[str] = []
    concat_inputs: list[str] = []
    for index, (start, end) in enumerate(scenes):
        start_seconds = start.get_seconds()
        end_seconds = end.get_seconds()
        filters.extend([
            f"[0:v]trim=start={start_seconds:.6f}:end={end_seconds:.6f},"
            f"setpts=PTS-STARTPTS[v{index}]",
            f"[0:a]atrim=start={start_seconds:.6f}:end={end_seconds:.6f},"
            f"asetpts=PTS-STARTPTS[a{index}]",
        ])
        concat_inputs.append(f"[v{index}][a{index}]")
    filters.append(
        f"{''.join(concat_inputs)}concat=n={len(scenes)}:v=1:a=1[outv][outa]"
    )
    return ";".join(filters)


def process_video(
    input_path: Path, output_path: Path, min_duration_seconds: float = 0.1
) -> Path:
    """Detect and remove tiny scenes without splitting compressed bitstreams."""
    video = open_video(str(input_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(min_scene_len=0))
    manager.detect_scenes(video, show_progress=True)
    scenes = filter_scenes_by_duration(
        manager.get_scene_list(), min_duration_seconds
    )
    if not scenes:
        raise ValueError("No scenes remain after filtering")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg", "-y", "-i", str(input_path),
        "-filter_complex", build_concat_filter(scenes),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ])
    return output_path
