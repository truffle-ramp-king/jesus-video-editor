"""Generate the Unified Visual Timeline image for multimodal LLM input."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from trailer_editor.preprocessor import (
    AudioAnalysis,
    DialogueBlock,
    PreprocessedData,
    extract_frame,
)

logger = logging.getLogger(__name__)

# Layout constants
WIDTH = 3600
HEIGHT = 1200
RULER_H = 50
WAVE_TOP = RULER_H
WAVE_H = 340
SHOT_TOP = WAVE_TOP + WAVE_H + 20
SHOT_H = 370
DIAL_TOP = SHOT_TOP + SHOT_H + 20
DIAL_H = HEIGHT - DIAL_TOP - 40
LEGEND_TOP = HEIGHT - 40

# Colours
BG = (13, 17, 23)
RULER_TEXT = (160, 170, 180)
WAVE_LOW = (0, 255, 136)
WAVE_HIGH = (255, 68, 68)
SILENCE_FILL = (40, 40, 50, 140)
SHOT_COLORS = [(38, 70, 120), (120, 80, 38)]
BORDER_COL = (60, 65, 75)
DIVIDER_COL = (45, 50, 60)

SPEAKER_PALETTE = [
    (70, 180, 220),
    (220, 130, 70),
    (180, 80, 200),
    (80, 200, 120),
    (200, 200, 70),
    (200, 70, 100),
]

THUMB_W = 108
THUMB_H = 192
MAX_THUMBS = 80  # cap to keep image manageable


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _t_to_x(t: float, duration: float) -> float:
    return (t / duration) * WIDTH


def _draw_ruler(draw: ImageDraw.ImageDraw, duration: float, font: ImageFont.FreeTypeFont) -> None:
    """Time ruler at the very top."""
    # Choose tick interval based on duration
    if duration <= 60:
        interval = 5.0
    elif duration <= 300:
        interval = 15.0
    else:
        interval = 30.0

    t = 0.0
    while t <= duration:
        x = _t_to_x(t, duration)
        draw.line([(x, RULER_H - 12), (x, RULER_H)], fill=RULER_TEXT, width=1)
        label = f"{int(t//60):02d}:{int(t%60):02d}"
        draw.text((x + 3, 5), label, fill=RULER_TEXT, font=font)
        t += interval


def _draw_waveform(
    draw: ImageDraw.ImageDraw,
    overlay: Image.Image,
    aa: AudioAnalysis,
    duration: float,
) -> None:
    """Draw RMS waveform, highlight peaks, shade silence."""
    wave_bottom = WAVE_TOP + WAVE_H
    mid_y = WAVE_TOP + WAVE_H // 2

    # Pre-compute x positions for each RMS window
    xs = _t_to_x(aa.rms_times, duration)
    n = len(aa.rms)

    # Draw waveform bars
    for i in range(n):
        val = float(aa.rms[i])
        x = int(xs[i])
        if x < 0 or x >= WIDTH:
            continue
        half_h = int(val * (WAVE_H // 2 - 4))
        if half_h < 1:
            continue
        # Colour ramp: green → red
        r = int(min(255, val * 2 * 255))
        g = int(min(255, (1 - val) * 2 * 255))
        col = (r, g, 60)
        draw.rectangle(
            [x, mid_y - half_h, x + 1, mid_y + half_h],
            fill=col,
        )

    # Shade silence intervals
    overlay_rgba = overlay.convert("RGBA")
    ov_draw = ImageDraw.Draw(overlay_rgba)
    for s, e in aa.silence_intervals:
        x1 = int(_t_to_x(s, duration))
        x2 = int(_t_to_x(e, duration))
        ov_draw.rectangle([x1, WAVE_TOP, x2, wave_bottom], fill=SILENCE_FILL)
    overlay.paste(overlay_rgba, (0, 0), overlay_rgba)

    # Draw peak zone markers (thin red lines at top)
    for s, e in aa.peak_intervals:
        x1 = int(_t_to_x(s, duration))
        x2 = int(_t_to_x(e, duration))
        draw.rectangle([x1, WAVE_TOP, x2, WAVE_TOP + 6], fill=WAVE_HIGH)


def _draw_shots(
    draw: ImageDraw.ImageDraw,
    data: PreprocessedData,
    work_dir: Path,
    font_small: ImageFont.FreeTypeFont,
) -> None:
    """Draw shot blocks with embedded frame thumbnails."""
    duration = data.duration
    shots = data.shots

    # Decide which shots get thumbnails (cap at MAX_THUMBS, pick evenly)
    if len(shots) > MAX_THUMBS:
        step = len(shots) / MAX_THUMBS
        thumb_indices = {int(i * step) for i in range(MAX_THUMBS)}
    else:
        thumb_indices = set(range(len(shots)))

    for idx, shot in enumerate(shots):
        x1 = int(_t_to_x(shot.start, duration))
        x2 = int(_t_to_x(shot.end, duration))
        x2 = max(x2, x1 + 2)  # minimum visible width
        col = SHOT_COLORS[idx % len(SHOT_COLORS)]
        draw.rectangle([x1, SHOT_TOP, x2, SHOT_TOP + SHOT_H], fill=col, outline=BORDER_COL)

        # Vertical boundary marker through waveform area
        draw.line([(x1, WAVE_TOP), (x1, SHOT_TOP)], fill=BORDER_COL, width=1)

        # Thumbnail
        if idx in thumb_indices:
            thumb_path = work_dir / f"frame_{shot.mid_frame}.jpg"
            if not thumb_path.exists():
                try:
                    extract_frame(data.video_path, shot.mid, thumb_path)
                except Exception as exc:
                    logger.warning("Failed to extract frame %d: %s", shot.mid_frame, exc)
                    continue
            try:
                thumb = Image.open(thumb_path).convert("RGB")
                thumb = thumb.resize((THUMB_W, THUMB_H), Image.LANCZOS)
                tx = (x1 + x2) // 2 - THUMB_W // 2
                ty = SHOT_TOP + (SHOT_H - THUMB_H) // 2
                # Clamp so thumbnail stays inside its block if possible
                tx = max(x1 + 1, min(tx, max(x2 - THUMB_W - 1, x1 + 1)))
                draw._image.paste(thumb, (tx, ty))
                # Draw border around thumbnail
                draw.rectangle(
                    [tx, ty, tx + THUMB_W, ty + THUMB_H],
                    outline=(255, 255, 255, 180),
                    width=1,
                )
            except Exception as exc:
                logger.warning("Failed to paste thumbnail %d: %s", shot.mid_frame, exc)

    # Label: "SHOT FRAMES" on left margin
    draw.text((6, SHOT_TOP + 4), "SHOTS", fill=(200, 200, 210), font=font_small)


# def _draw_dialogue(
#     draw: ImageDraw.ImageDraw,
#     dialogue: List[DialogueBlock],
#     duration: float,
#     font_small: ImageFont.FreeTypeFont,
#     font_tiny: ImageFont.FreeTypeFont,
# ) -> None:
#     """Draw all dialogue blocks using local Y-offset collision tracking."""
#     # Tracks the ending X-coordinate (pixel) for each row layer
#     row_ends: List[int] = []

#     # Styling requirements
#     bg_color = (10, 25, 47)       # Dark Blue
#     text_color = (255, 255, 255)  # White
#     row_height = 32               # Vertical height allocated per line row

#     for blk in dialogue:
#         x1 = int(_t_to_x(blk.start, duration))
#         x2 = int(_t_to_x(blk.end, duration))
#         x2 = max(x2, x1 + 4)

#         # Format the text string cleanly
#         label = f"{blk.speaker}: {blk.text}"

#         # Calculate exact text width to ensure the background box never clips the text
#         text_bbox = font_tiny.getbbox(label)
#         text_w = text_bbox[2] - text_bbox[0]
#         box_end_x = max(x2, x1 + text_w + 12)  # Box extends to either block end or text end

#         # Find the first available row track where this block won't overlap horizontally
#         assigned_row = 0
#         for row_idx, end_x in enumerate(row_ends):
#             if x1 >= end_x:  # Clear path found
#                 assigned_row = row_idx
#                 row_ends[row_idx] = box_end_x
#                 break
#         else:
#             # All existing rows are busy at this timestamp, spin up a new row track
#             assigned_row = len(row_ends)
#             row_ends.append(box_end_x)

#         # Calculate the final vertical placement using the assigned row index
#         current_dial_top = DIAL_TOP + (assigned_row * row_height)

#         # Safety check: Prevent drawing lower than the available DIAL_H area
#         if (current_dial_top + row_height - 4) > (DIAL_TOP + DIAL_H):
#             continue

#         # Draw dark blue background box
#         draw.rectangle(
#             [x1, current_dial_top, box_end_x, current_dial_top + row_height - 4],
#             fill=bg_color,
#             outline=(255, 255, 255, 30),
#         )

#         # Draw white dialogue text
#         draw.text((x1 + 6, current_dial_top + 4), label, fill=text_color, font=font_tiny)

#     # Label: "DIALOGUE" on left margin
#     draw.text((6, DIAL_TOP + 4), "DIALOGUE", fill=(200, 200, 210), font=font_small)

def _draw_dialogue(
    draw: ImageDraw.ImageDraw,
    dialogue: list[DialogueBlock],
    duration: float,
    font_small: ImageFont.FreeTypeFont,
    font_tiny: ImageFont.FreeTypeFont,
) -> None:
    """Draw all dialogue blocks with strict horizontal boundaries and dynamic vertical wrapping."""

    bg_color = (10, 25, 47)
    text_color = (255, 255, 255)
    row_height = 32
    pad_x = 6
    pad_y = 4

    def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        """Word-wrap text to fit within max_width pixels."""
        if max_width < 20:
            return []

        words = text.split(' ')
        lines = []
        current = ""

        for word in words:
            test = f"{current} {word}".strip()
            test_w = font.getbbox(test)[2] - font.getbbox(test)[0]

            if test_w <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)

                word_w = font.getbbox(word)[2] - font.getbbox(word)[0]
                if word_w > max_width:
                    # Single word too wide - truncate with ellipsis
                    trunc = word
                    while trunc and font.getbbox(trunc + "…")[2] - font.getbbox(trunc + "…")[0] > max_width:
                        trunc = trunc[:-1]
                    if trunc:
                        lines.append(trunc + "…")
                    current = ""
                else:
                    current = word

        if current:
            lines.append(current)

        return lines

    row_ends: list[int] = []  # Tracks end X-coordinate for each row

    for blk in dialogue:
        x1 = int(_t_to_x(blk.start, duration))
        x2 = int(_t_to_x(blk.end, duration))
        x2 = max(x2, x1 + 4)

        # Skip very short blocks - not enough space for text
        if x2 - x1 < 20:
            continue

        label = f"{blk.speaker}: {blk.text}"
        avail_w = (x2 - x1) - (pad_x * 2)
        lines = wrap_text(label, font_tiny, avail_w)

        if not lines:
            continue

        num_lines = len(lines)

        # Find consecutive available rows where all `num_lines` rows are free at x1
        assigned_row = None
        for start in range(len(row_ends) + 1):
            fits = True
            for off in range(num_lines):
                r = start + off
                if r < len(row_ends) and x1 < row_ends[r]:
                    fits = False
                    break
            if fits:
                assigned_row = start
                break

        # Fallback (shouldn't happen - we can always add new rows)
        if assigned_row is None:
            assigned_row = len(row_ends)

        # Extend row_ends and mark all used rows with our end x2
        while len(row_ends) <= assigned_row + num_lines - 1:
            row_ends.append(0)
        for off in range(num_lines):
            row_ends[assigned_row + off] = x2

        # Calculate vertical position
        top = DIAL_TOP + (assigned_row * row_height)
        height = num_lines * row_height - 4  # -4 for visual gap

        # Skip if exceeds available vertical space
        if top + height > DIAL_TOP + DIAL_H:
            continue

        # Draw background box - STRICTLY bounded by x1/x2 (no overflow)
        draw.rectangle(
            [x1, top, x2, top + height],
            fill=bg_color,
            outline=(255, 255, 255, 30),
        )

        # Draw each line of wrapped text
        for i, line in enumerate(lines):
            draw.text(
                (x1 + pad_x, top + i * row_height + pad_y),
                line,
                fill=text_color,
                font=font_tiny,
            )

    # Label
    draw.text((6, DIAL_TOP + 4), "DIALOGUE", fill=(200, 200, 210), font=font_small)


def _draw_legend(draw: ImageDraw.ImageDraw, font_tiny: ImageFont.FreeTypeFont) -> None:
    y = LEGEND_TOP + 10
    x = 10
    items = [
        (WAVE_LOW, "Waveform (normal)"),
        (WAVE_HIGH, "Peak / Impact zone"),
        ((80, 80, 90), "Silence / Dead air"),
    ]
    for col, label in items:
        draw.rectangle([x, y, x + 14, y + 14], fill=col)
        draw.text((x + 20, y), label, fill=RULER_TEXT, font=font_tiny)
        x += len(label) * 7 + 44


def generate_timeline(
    data: PreprocessedData,
    work_dir: Path,
    output_path: Path,
) -> Path:
    """Produce the unified visual timeline image."""
    logger.info("Generating timeline image (%dx%d) …", WIDTH, HEIGHT)

    font_sm = _load_font(16)
    font_tiny = _load_font(11)

    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # Dividers
    draw.line([(0, SHOT_TOP - 10), (WIDTH, SHOT_TOP - 10)], fill=DIVIDER_COL, width=1)
    draw.line([(0, DIAL_TOP - 10), (WIDTH, DIAL_TOP - 10)], fill=DIVIDER_COL, width=1)

    _draw_ruler(draw, data.duration, font_sm)
    _draw_waveform(draw, overlay, data.audio, data.duration)
    # Re-draw after overlay paste
    canvas.paste(overlay, (0, 0), overlay)
    draw = ImageDraw.Draw(canvas)  # re-create after paste
    _draw_shots(draw, data, work_dir, font_sm)
    _draw_dialogue(draw, data.dialogue, data.duration, font_sm, font_tiny)
    _draw_legend(draw, font_tiny)

    canvas.save(str(output_path), quality=92)
    logger.info("Timeline saved to %s", output_path)

    return output_path
