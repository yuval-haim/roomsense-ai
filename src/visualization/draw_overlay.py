from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from src.reasoning.report_schema import DetectedObject, RiskItem


def _font(size: int = 22):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_overlay(image_path: str | Path, objects: list[DetectedObject], risks: list[RiskItem], out_path: str | Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(20)

    for obj in objects:
        x1, y1, x2, y2 = obj.bbox
        draw.rectangle([x1, y1, x2, y2], outline=(255, 40, 40, 230), width=4)
        label = f"{obj.label} {obj.confidence:.2f}"
        tw = draw.textlength(label, font=font)
        draw.rectangle([x1, max(0, y1 - 28), x1 + tw + 10, y1], fill=(255, 40, 40, 180))
        draw.text((x1 + 5, max(0, y1 - 26)), label, fill=(255, 255, 255, 255), font=font)

    panel_h = 145
    draw.rectangle([0, 0, image.width, panel_h], fill=(0, 0, 0, 135))
    title = "RoomSense AI - Grounded Indoor Safety Report"
    draw.text((24, 18), title, fill=(255, 255, 255, 255), font=_font(28))
    for i, risk in enumerate(risks[:3], start=1):
        draw.text(
            (28, 48 + 28 * i),
            f"{i}. {risk.type.replace('_', ' ')} [{risk.severity}]",
            fill=(255, 255, 255, 255),
            font=font,
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)
