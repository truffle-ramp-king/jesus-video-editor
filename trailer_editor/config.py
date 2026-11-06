"""Runtime configuration and collision-safe output paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class ApiConfig:
    elevenlabs_key: str
    gemini_key: str
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-4-argon"

    @classmethod
    def from_env(cls) -> "ApiConfig":
        return cls(
            elevenlabs_key=_required_env("ELEVENLABS_API_KEY"),
            gemini_key=_required_env("GEMINI_API_KEY"),
            gemini_base_url=os.getenv(
                "GEMINI_BASE_URL", cls.gemini_base_url
            ),
            gemini_model=os.getenv("GEMINI_MODEL", cls.gemini_model),
        )


@dataclass(frozen=True)
class PipelineConfig:
    video: Path
    background: Path | None
    music: Path | None
    run_dir: Path
    edl_path: Path
    output_path: Path
    min_scene_duration: float = 0.0
    requested_duration: float | None = None
    deduplicate: bool = False
    prompt_style: str = "tough"
    from_plan: Path | None = None


def make_run_config(
    video: Path,
    background: Path | None = None,
    *,
    music: Path | None = None,
    runs_dir: Path = Path("runs"),
    edl_path: Path | None = None,
    output_path: Path | None = None,
    min_scene_duration: float = 0.0,
    requested_duration: float | None = None,
    deduplicate: bool = False,
    prompt_style: str = "tough",
    from_plan: Path | None = None,
) -> PipelineConfig:
    video = video.expanduser().resolve()
    background = background.expanduser().resolve() if background else None
    music = music.expanduser().resolve() if music else None
    for label, path in (("video", video), ("background", background), ("music", music)):
        if path is not None and not path.is_file():
            raise ValueError(f"{label} file does not exist: {path}")
    if music and not background:
        raise ValueError("Music mixing requires explicitly selecting the legacy background mode")
    if min_scene_duration < 0:
        raise ValueError("min_scene_duration cannot be negative")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{video.stem}-{stamp}-{uuid4().hex[:8]}"
    run_dir = (runs_dir / run_id).expanduser().resolve()
    return PipelineConfig(
        video=video,
        background=background,
        music=music,
        run_dir=run_dir,
        edl_path=edl_path.expanduser().resolve() if edl_path else run_dir / "edit-plan.json",
        output_path=output_path.expanduser().resolve() if output_path else run_dir / "final.mp4",
        min_scene_duration=min_scene_duration,
        requested_duration=requested_duration,
        deduplicate=deduplicate,
        prompt_style=prompt_style,
        from_plan=from_plan,
    )


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is not set in .env or the environment")
    return value
