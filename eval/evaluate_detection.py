from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.detector import AnnotationDetector
from src.utils.geometry import iou


def evaluate(annotations_dir: Path, images_dir: Path, iou_threshold: float = 0.5) -> dict:
    detector = AnnotationDetector(annotations_dir)
    total_gt = 0
    total_pred = 0
    matched = 0
    for ann_path in annotations_dir.glob("*.json"):
        data = json.loads(ann_path.read_text())
        image_path = ROOT / data["image_path"]
        gt = data["objects"]
        pred = [p.model_dump() for p in detector.predict(image_path)]
        total_gt += len(gt)
        total_pred += len(pred)
        used = set()
        for g in gt:
            best_i = -1
            best = 0.0
            for idx, p in enumerate(pred):
                if idx in used or p["label"] != g["label"]:
                    continue
                score = iou(g["bbox"], p["bbox"])
                if score > best:
                    best = score
                    best_i = idx
            if best >= iou_threshold:
                used.add(best_i)
                matched += 1
    precision = matched / total_pred if total_pred else 0
    recall = matched / total_gt if total_gt else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {
        "note": "Demo-mode metric over checked-in annotations. Replace AnnotationDetector with a model for real mAP.",
        "iou_threshold": iou_threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "gt_boxes": total_gt,
        "pred_boxes": total_pred,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(ROOT / "data/annotations", ROOT / "data/demo_images"), indent=2))
