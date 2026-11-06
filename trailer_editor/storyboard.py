"""Show progress within a shot, including slow reveals and silent action."""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from trailer_editor.preprocessor import PreprocessedData, extract_frame
from trailer_editor.timeline import _load_font


def make_storyboards(data: PreprocessedData, work: Path) -> list[Path]:
    # Include both ends, sampling more densely on short sources.
    count = max(4, min(48, math.ceil(data.duration / 0.9)))
    last = max(0, data.duration - max(0.16, 2 / data.fps))
    paths = []
    for offset in range(0, count, 16):
        n = min(16, count - offset)
        sheet = Image.new("RGB", (4 * 176, math.ceil(n / 4) * 288), "#14181d")
        draw = ImageDraw.Draw(sheet)
        font = _load_font(16)
        for i in range(n):
            t = last * (offset + i) / (count - 1)
            frame = work / f"sample_{offset + i:02d}.jpg"
            extract_frame(data.video_path, t, frame)
            with Image.open(frame) as im:
                thumb = ImageOps.contain(im, (172, 261))
                x, y = (i % 4) * 176, (i // 4) * 288
                sheet.paste(thumb, (x + (176 - thumb.width) // 2, y))
            draw.text((x + 5, y + 263), f"{t:.3f} s", font=font, fill="white")
        path = work / f"storyboard_{offset // 16 + 1}.jpg"
        sheet.save(path, quality=89)
        paths.append(path)
    return paths
