from __future__ import annotations

import numpy as np
from PIL import Image


class HeuristicDepthEstimator:
    """Small deterministic depth-like map for the demo.

    Real deployment can replace this with Depth Anything V2. This fallback gives
    a visual depth prior where lower image regions are closer to the camera.
    """

    def predict(self, image: Image.Image) -> np.ndarray:
        w, h = image.size
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        depth = np.repeat(y, w, axis=1)
        return depth

    def save_depth_png(self, depth: np.ndarray, path: str) -> None:
        arr = (255 * (depth - depth.min()) / (np.ptp(depth) + 1e-6)).astype("uint8")
        Image.fromarray(arr).save(path)
