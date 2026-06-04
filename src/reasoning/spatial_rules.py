from __future__ import annotations

from collections import defaultdict
from src.reasoning.report_schema import DetectedObject, RiskItem
from src.utils.geometry import horizontal_gap, vertical_overlap_ratio


def by_label(objects: list[DetectedObject]) -> dict[str, list[DetectedObject]]:
    out: dict[str, list[DetectedObject]] = defaultdict(list)
    for obj in objects:
        out[obj.label].append(obj)
    return out


def evaluate_spatial_risks(objects: list[DetectedObject]) -> list[RiskItem]:
    groups = by_label(objects)
    risks: list[RiskItem] = []

    sofas = groups.get("sofa", []) + groups.get("chaise_lounge", [])
    tables = groups.get("coffee_table", [])
    narrow_path_added = False
    for sofa in sofas:
        for table in tables:
            gap = horizontal_gap(sofa.bbox, table.bbox)
            v_overlap = vertical_overlap_ratio(sofa.bbox, table.bbox)
            if gap < 90 and v_overlap > 0.15 and not narrow_path_added:
                risks.append(
                    RiskItem(
                        type="narrow_path",
                        severity="medium",
                        evidence=(
                            "The coffee table is visually close to the sofa/chaise, "
                            "leaving a relatively narrow walking channel."
                        ),
                        grounded_objects=[sofa.label, table.label],
                        recommendation=(
                            "Move the coffee table slightly toward the room center or away from the sofa "
                            "to improve walking clearance."
                        ),
                    )
                )
                narrow_path_added = True

    if groups.get("cable"):
        risks.append(
            RiskItem(
                type="trip_hazard",
                severity="medium",
                evidence="A visible cable is detected near the TV console and open floor area.",
                grounded_objects=["cable", "tv_console"],
                recommendation="Route the cable behind the TV console or attach it along the wall.",
            )
        )

    if groups.get("tv_console") and len(groups.get("plant", [])) >= 2:
        risks.append(
            RiskItem(
                type="cluttered_surface",
                severity="low",
                evidence="The TV-console area contains several small objects and plants.",
                grounded_objects=["tv_console", "plant"],
                recommendation="Keep the console surface minimal to make cleaning and visual scanning easier.",
            )
        )

    if groups.get("large_window") and len(groups.get("plant", [])) >= 2:
        risks.append(
            RiskItem(
                type="healthy_layout_positive",
                severity="positive",
                evidence="Plants are placed near the large window with good natural light.",
                grounded_objects=["plant", "large_window"],
                recommendation="Keep light-loving plants near the window while preserving access to the window track.",
            )
        )

    return risks
