from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml
from PIL import Image

from src.reasoning.report_schema import DetectedObject


class AnnotationDetector:
    """Deterministic detector used for the lightweight demo and tests.

    It reads checked-in annotation JSON files, which makes the repository runnable
    without model downloads. For arbitrary images, use ``GroundingDinoDetector``.
    """

    def __init__(self, annotations_dir: str | Path = "data/annotations") -> None:
        self.annotations_dir = Path(annotations_dir)

    def predict(self, image_path: str | Path) -> List[DetectedObject]:
        image_id = Path(image_path).stem
        ann_path = self.annotations_dir / f"{image_id}.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"No demo annotation found for {image_id}. Add {ann_path} or run with "
                "--detector grounding-dino for arbitrary image inference."
            )
        data = json.loads(ann_path.read_text())
        return [DetectedObject(**obj) for obj in data["objects"]]


class GroundingDinoDetector:
    """Open-vocabulary detector backed by Hugging Face Grounding DINO.

    This is the production-style detector for RoomSense AI. It accepts natural
    language indoor-object prompts and returns normalized ``DetectedObject``
    records used by the rest of the pipeline.

    The implementation keeps heavy dependencies optional. Install them with:

        pip install -r requirements-grounding.txt

    Example:

        detector = GroundingDinoDetector()
        objects = detector.predict("data/demo_images/living_room_01.jpg")
    """

    def __init__(
        self,
        prompts_path: str | Path = "configs/object_prompts.yaml",
        model_id: str = "IDEA-Research/grounding-dino-base",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        device: str | None = None,
        max_detections: int = 80,
        nms_iou_threshold: float = 0.55,
    ) -> None:
        self.prompts_path = Path(prompts_path)
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_detections = max_detections
        self.nms_iou_threshold = nms_iou_threshold
        self.prompts = self._load_prompts(self.prompts_path)
        self.text_prompt = self._build_text_prompt(self.prompts)

        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise ImportError(
                "GroundingDinoDetector requires optional model dependencies. "
                "Install them with: pip install -r requirements-grounding.txt"
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @staticmethod
    def _load_prompts(path: Path) -> List[str]:
        if not path.exists():
            raise FileNotFoundError(f"Object prompt config not found: {path}")
        data = yaml.safe_load(path.read_text()) or {}
        prompts = data.get("indoor_objects", [])
        if not prompts:
            raise ValueError(f"No indoor_objects found in {path}")
        return [str(p).strip() for p in prompts if str(p).strip()]

    @staticmethod
    def _build_text_prompt(prompts: Sequence[str]) -> str:
        # Grounding DINO works best with period-separated phrases.
        return ". ".join(prompts) + "."

    def predict(self, image_path: str | Path) -> List[DetectedObject]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=self.text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [image.size[::-1]]  # HF expects (height, width)
        result = self._post_process(outputs, inputs, target_sizes)
        return self._format_detections(result)

    def _post_process(self, outputs, inputs, target_sizes):
        # Transformers versions have used slightly different signatures. Support
        # both to make the repo less fragile across environments.
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            try:
                processed = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.get("input_ids"),
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=target_sizes,
                )
            except TypeError:
                processed = self.processor.post_process_grounded_object_detection(
                    outputs,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=target_sizes,
                )
            return processed[0]

        if hasattr(self.processor, "post_process_object_detection"):
            return self.processor.post_process_object_detection(
                outputs,
                threshold=self.box_threshold,
                target_sizes=target_sizes,
            )[0]

        raise RuntimeError("The loaded processor does not expose a supported Grounding DINO post-process method.")

    def _format_detections(self, result) -> List[DetectedObject]:
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("text_labels", result.get("labels", []))

        raw: List[DetectedObject] = []
        for box, score, label in zip(boxes, scores, labels):
            conf = float(score.item() if hasattr(score, "item") else score)
            if conf < self.box_threshold:
                continue
            xyxy = [int(round(float(v))) for v in box.tolist()]
            clean_label = self._normalize_label(label)
            raw.append(DetectedObject(label=clean_label, confidence=conf, bbox=xyxy))

        return self._nms(raw)[: self.max_detections]

    def _normalize_label(self, label) -> str:
        # HF can return int IDs for some processors; map safely when possible.
        if isinstance(label, int):
            if 0 <= label < len(self.prompts):
                label = self.prompts[label]
            else:
                label = str(label)
        label = str(label).lower().strip()
        label = re.sub(r"[^a-z0-9_ /-]+", "", label)
        label = label.replace(" ", "_")
        aliases = {
            "tv": "television",
            "television_screen": "television",
            "console": "tv_console",
            "tv_stand": "tv_console",
            "couch": "sofa",
            "plant_pot": "plant",
            "potted_plant": "plant",
            "coffee_table": "coffee_table",
        }
        return aliases.get(label, label)

    def _nms(self, detections: Iterable[DetectedObject]) -> List[DetectedObject]:
        items = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[DetectedObject] = []
        for det in items:
            duplicate = False
            for prev in kept:
                same_label = det.label == prev.label
                high_overlap = self._iou(det.bbox, prev.bbox) >= self.nms_iou_threshold
                if same_label and high_overlap:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept

    @staticmethod
    def _iou(a: Sequence[int], b: Sequence[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union else 0.0


def build_detector(
    detector: str = "annotation",
    annotations_dir: str | Path = "data/annotations",
    prompts_path: str | Path = "configs/object_prompts.yaml",
    model_id: str = "IDEA-Research/grounding-dino-base",
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    device: str | None = None,
):
    """Factory used by the pipeline and CLI."""
    detector = detector.lower().strip()
    if detector in {"annotation", "annotations", "demo"}:
        return AnnotationDetector(annotations_dir)
    if detector in {"grounding-dino", "grounding_dino", "dino"}:
        return GroundingDinoDetector(
            prompts_path=prompts_path,
            model_id=model_id,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
    raise ValueError(f"Unknown detector backend: {detector}")
