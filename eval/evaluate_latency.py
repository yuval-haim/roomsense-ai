from __future__ import annotations

import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pipeline import RoomSensePipeline


def evaluate() -> dict:
    pipe = RoomSensePipeline(ROOT / "data/annotations")
    times = []
    for image in sorted((ROOT / "data/demo_images").glob("*.jpg")):
        t0 = time.perf_counter()
        pipe.analyze(image, ROOT / "outputs/latency")
        times.append(time.perf_counter() - t0)
    return {
        "images": len(times),
        "avg_latency_sec": round(sum(times) / len(times), 4),
        "max_latency_sec": round(max(times), 4),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
