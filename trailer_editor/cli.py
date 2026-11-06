"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from trailer_editor.config import ApiConfig, make_run_config



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jesus-video-edit",
        description="Cut a visual Bible story with Gemini and FFmpeg.",
    )
    parser.add_argument("video", type=Path, help="source video")
    parser.add_argument(
        "edl",
        nargs="?",
        type=Path,
        help="optional edit-plan JSON path (kept for command compatibility)",
    )
    parser.add_argument(
        "background",
        nargs="?",
        type=Path,
        help="opt into the legacy background composition",
    )
    parser.add_argument("--music", type=Path, help="music file (optional)")
    parser.add_argument("--output", type=Path, help="final video path")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--min-scene-duration", type=float, default=0.0)
    parser.add_argument(
        "--duration",
        type=float,
        help=(
            "target output duration in seconds (for example, --duration 60); "
            "invalid targets warn and fall back to automatic duration"
        ),
    )
    parser.add_argument(
        "--keep-source-audio-only",
        action="store_true",
        help="do not mix in music",
    )
    parser.add_argument("--from-plan", type=Path, help="render a saved edit plan locally with the selected effects")
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=False,
                        help="apply batch95 mirror, reframe, color grade and moving rectangles")
    parser.add_argument("--prompt-style", choices=["old", "tough"], default="tough")
    parser.add_argument("--env-file", type=Path, help="credentials file (default: .env in the working directory)")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.env_file and not args.env_file.is_file():
        parser.error(f"Credentials file does not exist: {args.env_file}")
    load_dotenv(args.env_file or Path.cwd() / ".env")
    from trailer_editor.pipeline import run_pipeline

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    background = args.background
    music = None if args.keep_source_audio_only else args.music
    if background and not music and not args.keep_source_audio_only:
        music = _env_path("TRAILER_MUSIC_PATH")
    try:
        config = make_run_config(
            args.video,
            background,
            music=music,
            runs_dir=args.runs_dir,
            edl_path=args.edl,
            output_path=args.output,
            min_scene_duration=args.min_scene_duration,
            requested_duration=args.duration,
            deduplicate=args.deduplicate,
            prompt_style=args.prompt_style,
            from_plan=args.from_plan,
        )
        result = run_pipeline(config, None if args.from_plan else ApiConfig.from_env())
    except ValueError as exc:
        parser.error(str(exc))
    print(result.edl.model_dump_json(indent=2))
    print(f"Video: {result.video_path}")
    print(f"Run artifacts: {result.run_dir}")


def _env_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None
