from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pipeline import RoomSensePipeline


def evaluate() -> dict:
    pipe = RoomSensePipeline(ROOT / "data/annotations")
    total = 0
    grounded = 0
    for ann_path in sorted((ROOT / "data/annotations").glob("*.json")):
        ann = json.loads(ann_path.read_text())
        report = pipe.analyze(ROOT / ann["image_path"], ROOT / "outputs/eval")
        detected = {obj.label for obj in report.detected_objects}
        for risk in report.risks:
            total += 1
            if set(risk.grounded_objects).issubset(detected):
                grounded += 1
    rate = grounded / total if total else 0
    return {"grounded_claim_rate": round(rate, 4), "claims": total, "grounded_claims": grounded}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
