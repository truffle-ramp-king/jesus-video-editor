"""Render an edit decision list with one FFmpeg filter graph."""

from __future__ import annotations

from pathlib import Path

from trailer_editor.preprocessor import has_audio_stream, run_command
from trailer_editor.schemas import TrailerEditDecisionList


def timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def render_edl(
    edl: TrailerEditDecisionList,
    source_video: Path,
    output_path: Path,
    *,
    deduplicate: bool = False,
) -> Path:
    """Trim and concatenate EDL clips without shared output names."""
    if not edl.edit_decision_list:
        raise ValueError("EDL contains no clips")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters: list[str] = []
    inputs: list[str] = []
    command = ["ffmpeg", "-y", "-i", str(source_video)]
    audio = "0:a:0"
    if not has_audio_stream(source_video):
        command += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        audio = "1:a:0"
    for index, clip in enumerate(edl.edit_decision_list):
        start, end = timestamp_seconds(clip.source_start), timestamp_seconds(clip.source_end)
        filters.extend(
            (
                f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{index}]",
                f"[{audio}]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{index}]",
            )
        )
        inputs.append(f"[v{index}][a{index}]")
    joined = "joined" if deduplicate else "outv"
    filters.append(f"{''.join(inputs)}concat=n={len(inputs)}:v=1:a=1[{joined}][outa]")
    if deduplicate:
        # Batch 95 shared grade and rectangle paths on its 1080x1920 canvas.
        filters.extend([
            "[joined]fps=30,scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,scale=1124:1996,crop=1080:1920,hflip,setsar=1,"
            "eq=contrast=1.12:brightness=0.022:saturation=1.1:gamma=1.025[graded]",
            "color=c=white:s=156x1920:r=30,format=rgba,colorchannelmixer=aa=0.025[vertical]",
            "color=c=white:s=1080x180:r=30,format=rgba,colorchannelmixer=aa=0.025[horizontal]",
            "[graded][vertical]overlay=x='mod(t*240+90,1236)-156':y=0:shortest=1:format=auto[pass1]",
            "[pass1][horizontal]overlay=x=0:y='2100-mod(t*300+450,2100)':"
            "shortest=1:format=auto,format=yuv420p[outv]",
        ])
    graph = ";".join(filters)
    run_command(command + [
        "-filter_complex", graph,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output_path),
    ])
    return output_path
