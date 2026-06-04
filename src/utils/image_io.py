from __future__ import annotations

from pathlib import Path
from PIL import Image


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
