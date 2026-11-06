"""Single-pass Editor Agent — sends multimodal payload to Gemini via
the OpenAI Python SDK and returns a structured Edit Decision List."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from pathlib import Path

from openai import OpenAI

from trailer_editor.duration_policy import target_duration_range
from trailer_editor.preprocessor import DialogueBlock, PreprocessedData, Word
from trailer_editor.storyboard import make_storyboards
from trailer_editor.schemas import TrailerEditDecisionList

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transcript formatter
# ---------------------------------------------------------------------------


def _format_transcript(dialogue: list[DialogueBlock | Word]) -> str:
    lines: list[str] = []
    for blk in dialogue:
        lines.append(
            f"[{blk.speaker}] [{_fmt_ts(blk.start)}] [{_fmt_ts(blk.end)}] {blk.text}"
        )
    return "\n".join(lines)


def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ORIGINAL_PROMPT = """\
Edit a self-contained visual story for a first-time viewer of a Jesus or Bible short. Use only evidence from this source. Work out who faces what problem, what the decisive action changes, and how the viewer sees its result. Preserve the actual sequence, meaning and tone, including quiet or reverent moments. Never assume a miracle, identity, or event from familiarity with a Bible story.

Read the timestamped frame sheets in row order. Use shot boundaries and the waveform in the timeline to locate cuts. Frame samples describe moments, not exact transitions. Preserve gestures, object movement, changes of state, and the reaction long enough to register. Keep a lead-in when the action would otherwise jump. Read visible captions and let a phrase or title finish. Use word times to preserve a complete thought if there is narration. Lyrics, music and sound-effect tags are context, not dialogue driving the plot. Silence may carry the payoff.

Open at the earliest frame needed to understand the action. For a compact single beat, keep it continuous, even the entire source if it already works. Trim only identifiable idle holds, repeated illustrations, redundant action, branding or an unrelated outro. For a long retelling, choose one complete arc and omit whole redundant beats. Favor a clean shot boundary or complete spoken phrase. Do not splice unrelated images into a false cause and effect. Do not impose a cut count, duration quota, teaser cliffhanger, or invented call to action.

Return chronological, non-overlapping ranges in HH:MM:SS.mmm, inside the source. Consolidate adjoining ranges. Keep source speed, no loops or padding. For each interval explain the observable action and why the in/out points work. A speaker may be null. Sum the interval lengths for the declared duration. Allowed bounds are {min_duration:.3f} to {max_duration:.3f} seconds; the upper bound is a limit, never a goal.
"""


TOUGH_PROMPT = """You are cutting a vertical Jesus or Bible story for a first-time viewer. Deliver the shortest cut in which the viewer can understand the predicament, see the decisive action, and recognize the result. Build from an empty timeline. Every retained second needs a job.

Pick the one change that makes this worth watching. Find its strongest visible payoff first. Work backwards: which exact gesture, look, or spoken clause makes that payoff intelligible? Open on the predicament already in motion. Keep one instance of each necessary idea. Once we know the person is afraid, walking, praying, explaining, or reacting, jump to the next change. Lose repeated angles, approaches, scenic holds, second reactions, recaps, sermons after the resolution, credits and calls to subscribe. End as soon as the result lands.

Make a paper cut inside the body of the footage. On a multi-shot source, actively bridge from an early setup to a later action, then to its consequence. On a continuous shot look for a match on pose, motion or object position that skips waiting. A single prefix with the outro removed is generally a rough assembly. Try dropping each retained interval and shortening both its ends. Restore only the precise moment whose loss would confuse a new viewer. Do not fake a jump by splitting adjoining ranges. A genuinely unbroken transformation can stay continuous when a jump would hide how it happens.

For a 10–30 second source, first audition a cut using roughly half its runtime. For a multi-minute retelling, first audition one compact sequence using roughly a quarter. These are editing passes, not final quotas. Use more time only where a viewer needs it. If an explicit requested duration is given, honor its bounds.

Use shot edges and timestamped frame sheets to identify the action. Thumbnails sample motion; leave enough frames on each side for a gesture to read. At dialogue cuts use word timestamps: preserve complete intelligible clauses, their speaker and meaning. Keep cause before effect, protect a meaningful silence, and cut music on a plausible beat. Never invent a biblical event absent from the footage. Do not speed up, loop, or pad.

Return chronological, disjoint source intervals in HH:MM:SS.mmm, within {min_duration:.3f}–{max_duration:.3f} total seconds. Merge adjoining ranges. Explain the specific action in each interval and why it begins and ends there. In the rationale, name the largest omitted internal beat and why the story works across that cut. If there is no internal omission, identify the exact continuous action and why its middle must remain. Sum the actual intervals for total_duration_seconds.
"""


def build_evidence(data: PreprocessedData, timeline_image_path: Path, storyboards: list[Path]):
    transcript_text = _format_transcript(data.dialogue)
    transcript_words = _format_transcript(data.words)

    user_content = [
        {
            "type": "text",
            "text": (
                f"## SOURCE VIDEO METADATA\n"
                f"- Total duration: {_fmt_ts(data.duration)} ({data.duration:.2f}s)\n"
                f"- Frame rate: {data.fps:.2f} fps\n"
                f"- Detected shots: {len(data.shots)}\n"
                f"- Dialogue blocks: {len(data.dialogue)}\n\n"
                f"##DIALOGUE TRANSCRIPT\n"
                f"{transcript_text if transcript_text else '(No dialogue detected — use visual cues only.)'}\n\n"
                f"- Word blocks: {len(data.words)}\n\n"
                f"##WORD TRANSCRIPT\n"
                f"{transcript_words if transcript_words else '(No dialogue detected — use visual cues only.)'}\n\n"
                f"## VISUAL TIMELINE IMAGE\n"
                f"Study the image below carefully. Use the ruler timestamps to align with the transcript times above. "
                f"Shot edges in seconds: {[(round(s.start, 3), round(s.end, 3)) for s in data.shots]}. "
                "The first image is the timeline. The following sheets show source frames with times in seconds."
            ),
        },
    ]
    for path in [timeline_image_path, *storyboards]:
        mime = "png" if path.suffix == ".png" else "jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        user_content.append({"type": "image_url", "image_url": {
            "url": f"data:image/{mime};base64,{encoded}", "detail": "high"}})
    logger.debug("Agent text payload: %s", user_content[0]["text"])

    return user_content


def plan_from_evidence(user_content, source_duration, gemini_api_key, gemini_base_url,
                       gemini_model, *, prompt_style="tough", requested_duration=None,
                       receipt_path=None):
    minimum, maximum = target_duration_range(source_duration, requested_duration)
    template = {"old": ORIGINAL_PROMPT, "tough": TOUGH_PROMPT}[prompt_style]
    system = template.format(min_duration=minimum, max_duration=maximum)
    client = OpenAI(
        api_key=gemini_api_key,
        max_retries=1,
        base_url=gemini_base_url,
    )

    logger.info("Calling %s via OpenAI-compatible endpoint …", gemini_model)

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
    for attempt in range(2):
        response = client.beta.chat.completions.parse(
            model=gemini_model, messages=messages, response_format=TrailerEditDecisionList,
        )
        reported = response.model.removeprefix("models/")
        if reported != gemini_model and not reported.startswith(gemini_model + "-"):
            raise RuntimeError(f"Requested {gemini_model}, provider reported {reported}")
        edl = response.choices[0].message.parsed
        if edl is None:
            raise RuntimeError("Gemini returned no parsed edit plan")
        try:
            _validate_edl(edl, source_duration=source_duration, requested_duration=requested_duration)
        except ValueError as exc:
            if attempt:
                raise
            messages += [{"role": "assistant", "content": edl.model_dump_json()},
                         {"role": "user", "content": f"Correct this validation error: {exc}"}]
            continue
        break
    if receipt_path:
        receipt_path.write_text(json.dumps({
            "requested_model": gemini_model, "returned_model": response.model, "response_id": response.id,
            "prompt_style": prompt_style, "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "evidence_sha256": hashlib.sha256(json.dumps(user_content).encode()).hexdigest(),
            "validation_attempts": attempt + 1,
        }, indent=2))
    logger.info(
        "Agent returned %d clips, total=%.3fs. Narrative: %s",
        len(edl.edit_decision_list),
        edl.total_duration_seconds,
        edl.narrative_justification[:80],
    )
    return edl


def run_editor_agent(data, timeline_image_path, gemini_api_key, gemini_base_url, gemini_model,
                     *, requested_duration=None, prompt_style="tough", receipt_path=None):
    sheets = make_storyboards(data, timeline_image_path.parent)
    return plan_from_evidence(build_evidence(data, timeline_image_path, sheets), data.duration,
                              gemini_api_key, gemini_base_url, gemini_model, prompt_style=prompt_style,
                              requested_duration=requested_duration, receipt_path=receipt_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})$")


def _parse_ts(ts: str) -> float:
    m = _TS_RE.match(ts)
    if not m:
        raise ValueError(f"Invalid timestamp format: {ts!r}")
    h, mn, s, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return h * 3600 + mn * 60 + s + ms / 1000.0


def _validate_edl(
    edl: TrailerEditDecisionList,
    *,
    source_duration: float,
    requested_duration: float | None = None,
) -> None:
    """Raise if the EDL violates any structural or mathematical constraint."""
    min_duration, max_duration = target_duration_range(
        source_duration, requested_duration
    )
    if not (min_duration <= edl.total_duration_seconds <= max_duration):
        raise ValueError(
            f"total_duration_seconds={edl.total_duration_seconds} is outside "
            f"[{min_duration:.3f}, {max_duration:.3f}]"
        )

    computed_total = 0.0
    source_ranges: list[tuple[float, float, int]] = []
    for i, clip in enumerate(edl.edit_decision_list, start=1):
        if clip.clip_index != i:
            raise ValueError(f"clip_index={clip.clip_index} expected {i}")
        s = _parse_ts(clip.source_start)
        e = _parse_ts(clip.source_end)
        if e <= s:
            raise ValueError(
                f"Clip {i}: source_end ({clip.source_end}) <= source_start ({clip.source_start})"
            )
        if s < 0 or e > source_duration + 0.001:
            raise ValueError(
                f"Clip {i}: range [{s:.3f}, {e:.3f}] is outside source duration "
                f"[0.000, {source_duration:.3f}]"
            )
        computed_total += e - s
        source_ranges.append((s, e, i))

    if source_ranges != sorted(source_ranges):
        raise ValueError("Source ranges must be in chronological order")
    for previous, current in zip(source_ranges, source_ranges[1:]):
        previous_start, previous_end, previous_index = previous
        current_start, current_end, current_index = current
        if current_start < previous_end:
            raise ValueError(
                f"Clips {previous_index} and {current_index} reuse overlapping source "
                f"content [{previous_start:.3f}, {previous_end:.3f}] and "
                f"[{current_start:.3f}, {current_end:.3f}]"
            )

    if abs(computed_total - edl.total_duration_seconds) > 0.05:
        raise ValueError(
            f"Declared total_duration_seconds={edl.total_duration_seconds} "
            f"does not match computed sum={computed_total:.3f}"
        )
