from __future__ import annotations

from typing import Sequence

BBox = Sequence[float]


def area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return area((x1, y1, x2, y2))


def iou(a: BBox, b: BBox) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def center(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def horizontal_gap(a: BBox, b: BBox) -> float:
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    if ax2 < bx1:
        return bx1 - ax2
    if bx2 < ax1:
        return ax1 - bx2
    return 0.0


def vertical_overlap_ratio(a: BBox, b: BBox) -> float:
    _, ay1, _, ay2 = a
    _, by1, _, by2 = b
    inter = max(0.0, min(ay2, by2) - max(ay1, by1))
    denom = min(ay2 - ay1, by2 - by1)
    return inter / denom if denom > 0 else 0.0
