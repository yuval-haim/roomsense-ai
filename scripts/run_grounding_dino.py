from __future__ import annotations

"""Convenience entry point for real open-vocabulary detection.

Install optional dependencies first:
    pip install -r requirements-grounding.txt

Then run:
    python scripts/run_grounding_dino.py --image data/demo_images/living_room_01.jpg
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_inference.py"),
        "--detector",
        "grounding-dino",
        *sys.argv[1:],
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
