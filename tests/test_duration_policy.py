import pytest

from trailer_editor.duration_policy import requested_duration_issue, target_duration_range


def test_long_source_can_use_its_full_runtime() -> None:
    assert target_duration_range(120.0) == (1.0, 120.0)


def test_29_second_source_is_never_padded_to_30_seconds() -> None:
    assert target_duration_range(29.0) == (1.0, 29.0)


def test_short_source_uses_duration_aware_range() -> None:
    assert target_duration_range(17.0) == pytest.approx((1.0, 17.0))


def test_too_short_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="too short"):
        target_duration_range(0.999)


def test_explicit_one_minute_target_uses_two_second_tolerance() -> None:
    assert target_duration_range(220.0, 60.0) == (58.0, 60.0)


def test_target_longer_than_source_falls_back_to_automatic_policy() -> None:
    assert requested_duration_issue(17.0, 60.0) is not None
    assert target_duration_range(17.0, 60.0) == pytest.approx((1.0, 17.0))


@pytest.mark.parametrize("requested", [0.0, -5.0, float("nan"), float("inf")])
def test_nonsensical_target_falls_back_to_automatic_policy(requested: float) -> None:
    assert requested_duration_issue(220.0, requested) is not None
    assert target_duration_range(220.0, requested) == (1.0, 220.0)
