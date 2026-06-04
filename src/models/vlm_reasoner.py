from __future__ import annotations

from typing import Iterable
from src.reasoning.report_schema import RiskItem


class GroundedTextReasoner:
    """Creates a natural-language summary from grounded objects and rule outputs."""

    def summarize(self, objects: Iterable[str], risks: list[RiskItem]) -> str:
        labels = sorted(set(objects))
        if not risks:
            return f"The room appears organized. Detected {len(labels)} object categories."
        risk_names = ", ".join(r.type.replace("_", " ") for r in risks)
        return (
            f"The room is a bright living room with {len(labels)} detected object categories. "
            f"The main issues found are: {risk_names}. All recommendations are tied to detected objects."
        )
