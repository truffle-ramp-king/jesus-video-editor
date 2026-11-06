from pathlib import Path

import pytest

from trailer_editor.config import make_run_config


def test_run_paths_are_unique(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    background = tmp_path / "background.mp4"
    video.touch()
    background.touch()

    first = make_run_config(video, background, runs_dir=tmp_path / "runs")
    second = make_run_config(video, background, runs_dir=tmp_path / "runs")

    assert first.run_dir != second.run_dir
    assert first.output_path.parent == first.run_dir
    assert first.edl_path.parent == first.run_dir


def test_missing_input_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="video file does not exist"):
        make_run_config(tmp_path / "missing.mp4", tmp_path / "also-missing.mp4")


def test_negative_scene_duration_is_rejected(tmp_path: Path) -> None:
    video, background = tmp_path / "a.mp4", tmp_path / "b.mp4"
    video.touch()
    background.touch()
    with pytest.raises(ValueError, match="cannot be negative"):
        make_run_config(video, background, min_scene_duration=-0.1)


def test_requested_duration_is_preserved_for_source_aware_validation(tmp_path: Path) -> None:
    video, background = tmp_path / "a.mp4", tmp_path / "b.mp4"
    video.touch()
    background.touch()
    config = make_run_config(video, background, requested_duration=60.0)
    assert config.requested_duration == 60.0


def test_source_only_is_the_default(tmp_path: Path):
    video = tmp_path / "source.mp4"
    video.touch()
    config = make_run_config(video)
    assert config.background is None
    assert config.min_scene_duration == 0
