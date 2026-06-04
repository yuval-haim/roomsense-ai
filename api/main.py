from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File, Query
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pipeline import RoomSensePipeline

app = FastAPI(title="RoomSense AI API", version="0.2.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze-room")
async def analyze_room(
    file: UploadFile = File(...),
    detector: str = Query(default=os.getenv("ROOMSENSE_DETECTOR", "annotation"), pattern="^(annotation|grounding-dino)$"),
) -> dict:
    suffix = Path(file.filename or "room.jpg").suffix or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    pipe = RoomSensePipeline(
        annotations_dir=ROOT / "data/annotations",
        detector_backend=detector,
        prompts_path=ROOT / "configs/object_prompts.yaml",
    )
    try:
        report = pipe.analyze(tmp_path, ROOT / "outputs/api")
        return report.model_dump()
    except FileNotFoundError as exc:
        return {
            "error": "No annotation exists for this uploaded image in demo mode.",
            "detail": str(exc),
            "next_step": "Use ?detector=grounding-dino and install requirements-grounding.txt for arbitrary image inference.",
        }
