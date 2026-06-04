from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pipeline import RoomSensePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RoomSense AI on one image")
    parser.add_argument("--image", required=True, help="Path to room image")
    parser.add_argument("--output_dir", default="outputs/examples")
    parser.add_argument("--annotations_dir", default="data/annotations")
    parser.add_argument("--detector", default="annotation", choices=["annotation", "grounding-dino"], help="Detection backend")
    parser.add_argument("--prompts_path", default="configs/object_prompts.yaml")
    parser.add_argument("--model_id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument("--device", default=None, help="cuda, cpu, mps, or leave empty for auto")
    args = parser.parse_args()

    pipe = RoomSensePipeline(
        annotations_dir=args.annotations_dir,
        detector_backend=args.detector,
        prompts_path=args.prompts_path,
        model_id=args.model_id,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )
    report = pipe.analyze(args.image, args.output_dir)
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
