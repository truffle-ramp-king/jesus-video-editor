import pytest

from trailer_editor.agent import _parse_ts, _validate_edl
from trailer_editor.schemas import EditClip, TrailerEditDecisionList


def make_edl(total: float = 28.0) -> TrailerEditDecisionList:
    return TrailerEditDecisionList(
        narrative_justification="A compact dramatic arc with a strong hook.",
        total_duration_seconds=total,
        edit_decision_list=[
            EditClip(
                clip_index=1,
                source_start="00:00:01.125",
                source_end="00:00:29.125",
                speaker="Speaker_0",
                reason="Carries the central dramatic exchange.",
            )
        ],
    )


def test_parse_timestamp() -> None:
    assert _parse_ts("01:02:03.456") == pytest.approx(3723.456)


def test_validate_edl_accepts_matching_duration() -> None:
    _validate_edl(make_edl(), source_duration=60.0)


def test_validate_edl_rejects_wrong_declared_sum() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _validate_edl(make_edl(29.0), source_duration=60.0)


def test_validate_edl_accepts_shorter_output_for_short_source() -> None:
    edl = TrailerEditDecisionList(
        narrative_justification="A compact dramatic arc using the available short clip.",
        total_duration_seconds=13.6,
        edit_decision_list=[
            EditClip(
                clip_index=1,
                source_start="00:00:01.000",
                source_end="00:00:14.600",
                speaker="Speaker_0",
                reason="Keeps the strongest portion without padding.",
            )
        ],
    )
    _validate_edl(edl, source_duration=17.0)


def test_validate_edl_rejects_overlapping_source_content() -> None:
    edl = TrailerEditDecisionList(
        narrative_justification="An invalid edit that repeats the same source material.",
        total_duration_seconds=14.0,
        edit_decision_list=[
            EditClip(
                clip_index=1,
                source_start="00:00:00.000",
                source_end="00:00:07.000",
                speaker="Speaker_0",
                reason="First use of the source range.",
            ),
            EditClip(
                clip_index=2,
                source_start="00:00:05.000",
                source_end="00:00:12.000",
                speaker="Speaker_0",
                reason="Overlaps the first source range.",
            ),
        ],
    )
    with pytest.raises(ValueError, match="reuse overlapping source content"):
        _validate_edl(edl, source_duration=15.0)


def test_validate_edl_rejects_timestamp_past_source_end() -> None:
    edl = make_edl()
    with pytest.raises(ValueError, match="outside source duration"):
        _validate_edl(edl, source_duration=29.0)


def test_validate_edl_accepts_explicit_one_minute_target() -> None:
    edl = TrailerEditDecisionList(
        narrative_justification="A longer dramatic arc requested by the user.",
        total_duration_seconds=60.0,
        edit_decision_list=[
            EditClip(
                clip_index=1,
                source_start="00:00:10.000",
                source_end="00:01:10.000",
                speaker="Speaker_0",
                reason="Uses a unique one-minute source range.",
            )
        ],
    )
    _validate_edl(edl, source_duration=220.0, requested_duration=60.0)


def test_visual_only_full_length_plan():
    edl = make_edl(12.0)
    clip = edl.edit_decision_list[0]
    clip.source_start, clip.source_end = "00:00:00.000", "00:00:12.000"
    clip.speaker = None
    _validate_edl(edl, source_duration=12.0)


def test_reverse_order_is_rejected():
    edl = make_edl(29.0)
    edl.edit_decision_list.append(edl.edit_decision_list[0].model_copy(update={
        "clip_index": 2, "source_start": "00:00:00.000", "source_end": "00:00:01.000",
    }))
    with pytest.raises(ValueError, match="chronological"):
        _validate_edl(edl, source_duration=60.0)
