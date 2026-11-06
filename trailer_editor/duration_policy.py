"""Output-duration policy shared by prompting and validation."""

from __future__ import annotations

import math

MIN_SOURCE_DURATION_SECONDS = 1.0
REQUESTED_DURATION_TOLERANCE_SECONDS = 2.0


def requested_duration_issue(
    source_duration: float, requested_duration: float | None
) -> str | None:
    """Explain why an explicit target cannot be honored, if applicable."""
    if requested_duration is None:
        return None
    if not math.isfinite(requested_duration):
        return "it must be a finite number"
    if requested_duration < MIN_SOURCE_DURATION_SECONDS:
        return f"it must be at least {MIN_SOURCE_DURATION_SECONDS:.1f}s"
    if requested_duration > source_duration:
        return f"it exceeds the {source_duration:.3f}s source duration"
    return None


def target_duration_range(
    source_duration: float, requested_duration: float | None = None
) -> tuple[float, float]:
    """Return the allowed output-duration range for a source video."""
    if source_duration < MIN_SOURCE_DURATION_SECONDS:
        raise ValueError(
            f"Source duration {source_duration:.3f}s is too short; "
            f"minimum is {MIN_SOURCE_DURATION_SECONDS:.1f}s"
        )

    if requested_duration_issue(source_duration, requested_duration) is None and (
        requested_duration is not None
    ):
        maximum = round(requested_duration, 3)
        minimum = round(
            max(
                MIN_SOURCE_DURATION_SECONDS,
                requested_duration - REQUESTED_DURATION_TOLERANCE_SECONDS,
            ),
            3,
        )
        return minimum, maximum

    return MIN_SOURCE_DURATION_SECONDS, round(source_duration, 3)
