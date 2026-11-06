"""Strict Pydantic schemas for the edit decision list output."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class EditClip(BaseModel):
    """A single clip selected for the trailer."""

    clip_index: int = Field(
        ...,
        ge=1,
        description="Sequential index starting from 1.",
    )
    source_start: str = Field(
        ...,
        pattern=r"^\d{2}:[0-5]\d:[0-5]\d\.\d{3}$",
        description="Timestamp HH:MM:SS.mmm where the clip begins in the source file.",
    )
    source_end: str = Field(
        ...,
        pattern=r"^\d{2}:[0-5]\d:[0-5]\d\.\d{3}$",
        description="Timestamp HH:MM:SS.mmm where the clip ends in the source file.",
    )
    speaker: str | None = Field(
        ...,
        description="Speaker if identifiable, or null for a visual or music-only beat.",
    )
    reason: str = Field(
        ...,
        description=(
            "Short direct explanation connecting the cut to the visual timeline "
            "features (peaks, silence zones, shot frames) or script data."
        ),
    )


class TrailerEditDecisionList(BaseModel):
    """Top-level response envelope — the complete edit decision list."""

    narrative_justification: str = Field(
        ...,
        min_length=20,
        description="Describe the setup, action and visible outcome. Explain what can be omitted, or why the complete source earns its runtime.",
    )
    total_duration_seconds: float = Field(
        ...,
        gt=0.0,
        description="Exact mathematical total duration of all selected clips combined.",
    )
    edit_decision_list: List[EditClip] = Field(
        ...,
        min_length=1,
        description="Chronological, disjoint source intervals. A single continuous interval including the entire source is valid.",
    )
