from __future__ import annotations

import json
from pathlib import Path

from src.models.detector import build_detector
from src.models.depth_estimator import HeuristicDepthEstimator
from src.models.vlm_reasoner import GroundedTextReasoner
from src.reasoning.spatial_rules import evaluate_spatial_risks
from src.reasoning.risk_scoring import overall_score
from src.reasoning.report_schema import RoomAnalysisReport
from src.utils.image_io import load_rgb, ensure_dir
from src.visualization.draw_overlay import draw_overlay


class RoomSensePipeline:
    def __init__(
        self,
        annotations_dir: str | Path = "data/annotations",
        detector_backend: str = "annotation",
        prompts_path: str | Path = "configs/object_prompts.yaml",
        model_id: str = "IDEA-Research/grounding-dino-base",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        device: str | None = None,
    ) -> None:
        self.detector = build_detector(
            detector=detector_backend,
            annotations_dir=annotations_dir,
            prompts_path=prompts_path,
            model_id=model_id,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
        self.depth = HeuristicDepthEstimator()
        self.reasoner = GroundedTextReasoner()

    def analyze(self, image_path: str | Path, output_dir: str | Path = "outputs/examples") -> RoomAnalysisReport:
        image_path = Path(image_path)
        output_dir = ensure_dir(output_dir)
        objects = self.detector.predict(image_path)
        risks = evaluate_spatial_risks(objects)
        labels = [obj.label for obj in objects]
        summary = self.reasoner.summarize(labels, risks)
        report = RoomAnalysisReport(
            image_id=image_path.stem,
            detected_objects=objects,
            risks=risks,
            overall_score=overall_score(risks),
            summary=summary,
        )

        depth = self.depth.predict(load_rgb(image_path))
        self.depth.save_depth_png(depth, str(output_dir / f"{image_path.stem}_depth.png"))
        draw_overlay(image_path, objects, risks, output_dir / f"{image_path.stem}_overlay.jpg")
        (output_dir / f"{image_path.stem}_report.json").write_text(report.model_dump_json(indent=2))
        return report


def analyze_room(image_path: str | Path, output_dir: str | Path = "outputs/examples") -> dict:
    report = RoomSensePipeline().analyze(image_path, output_dir)
    return json.loads(report.model_dump_json())
