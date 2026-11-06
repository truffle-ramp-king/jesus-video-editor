"""The trailer editing pipeline, independent of command-line parsing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from trailer_editor.agent import _validate_edl, run_editor_agent
from trailer_editor.config import ApiConfig, PipelineConfig
from trailer_editor.duration_policy import requested_duration_issue, target_duration_range
from trailer_editor.postprocess import post_process_video
from trailer_editor.preprocessor import preprocess, probe_duration
from trailer_editor.remove_tiny_scenes import process_video
from trailer_editor.render import render_edl
from trailer_editor.schemas import TrailerEditDecisionList
from trailer_editor.timeline import generate_timeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    edl: TrailerEditDecisionList
    edl_path: Path
    video_path: Path
    run_dir: Path


def run_pipeline(config: PipelineConfig, api: ApiConfig | None) -> PipelineResult:
    """Execute all pipeline phases in an isolated, inspectable run directory."""
    config.run_dir.mkdir(parents=True, exist_ok=False)
    work_dir = config.run_dir / "work"
    work_dir.mkdir()
    config.edl_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    source_duration = probe_duration(config.video)
    duration_issue = requested_duration_issue(source_duration, config.requested_duration)
    if duration_issue is not None:
        automatic_min, automatic_max = target_duration_range(source_duration)
        logger.warning(
            "Ignoring --duration %r because %s; using automatic range %.3f–%.3fs",
            config.requested_duration,
            duration_issue,
            automatic_min,
            automatic_max,
        )
    elif config.requested_duration is not None:
        requested_min, requested_max = target_duration_range(
            source_duration, config.requested_duration
        )
        logger.info(
            "Using requested duration %.3fs with target range %.3f–%.3fs",
            config.requested_duration,
            requested_min,
            requested_max,
        )

    if config.from_plan:
        if not config.from_plan.is_file():
            raise ValueError(f"Edit plan does not exist: {config.from_plan}")
        edl = TrailerEditDecisionList.model_validate_json(config.from_plan.read_text())
        _validate_edl(edl, source_duration=source_duration, requested_duration=config.requested_duration)
    else:
        if api is None:
            raise ValueError("Generating a plan requires API credentials")
        logger.info("Phase 1/3: preprocess %s", config.video.name)
        data = preprocess(
            config.video,
            work_dir,
            api.elevenlabs_key,
            source_duration=source_duration,
        )
        timeline_path = generate_timeline(data, work_dir, work_dir / "timeline.png")

        logger.info("Phase 2/3: generate edit plan")
        edl = run_editor_agent(
            data,
            timeline_path,
            api.gemini_key,
            api.gemini_base_url,
            api.gemini_model,
            requested_duration=config.requested_duration,
            prompt_style=config.prompt_style,
            receipt_path=config.run_dir / "model-receipt.json",
        )
    config.edl_path.write_text(edl.model_dump_json(indent=2), encoding="utf-8")

    logger.info("Rendering source framing and sound at original speed")
    intermediate = bool(config.background or config.min_scene_duration)
    rendered = render_edl(edl, config.video, work_dir / "rendered.mp4" if intermediate else config.output_path,
                          deduplicate=config.deduplicate)
    if config.min_scene_duration:
        rendered = process_video(rendered, work_dir / "filtered.mp4" if config.background else config.output_path,
                                 config.min_scene_duration)
    if config.background:
        post_process_video(rendered, config.background, config.output_path, music_path=config.music)
    return PipelineResult(edl, config.edl_path, config.output_path, config.run_dir)
