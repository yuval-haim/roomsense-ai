from __future__ import annotations

from src.reasoning.report_schema import RiskItem

WEIGHTS = {"positive": -5, "low": 8, "medium": 18, "high": 32}


def overall_score(risks: list[RiskItem]) -> float:
    penalty = sum(WEIGHTS.get(r.severity, 10) for r in risks)
    return max(0.0, min(100.0, 100.0 - penalty))
