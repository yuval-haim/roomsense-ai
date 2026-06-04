from __future__ import annotations

import json
from pathlib import Path
import sys
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pipeline import RoomSensePipeline

LABELS = ["narrow_path", "trip_hazard", "cluttered_surface", "healthy_layout_positive"]


def evaluate() -> dict:
    pipe = RoomSensePipeline(ROOT / "data/annotations")
    y_true, y_pred = [], []
    for ann_path in sorted((ROOT / "data/annotations").glob("*.json")):
        ann = json.loads(ann_path.read_text())
        report = pipe.analyze(ROOT / ann["image_path"], ROOT / "outputs/eval")
        true = set(ann.get("risk_labels", []))
        pred = {r.type for r in report.risks if r.severity != "positive"}
        for label in LABELS[:-1]:
            y_true.append(1 if label in true else 0)
            y_pred.append(1 if label in pred else 0)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "note": "Small demo risk-label smoke test. Expand with a manually labeled evaluation set before reporting as benchmark.",
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
