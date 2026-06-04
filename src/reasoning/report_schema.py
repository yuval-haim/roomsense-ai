from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class DetectedObject(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    bbox: List[int]
    mask_id: Optional[str] = None


class RiskItem(BaseModel):
    type: str
    severity: str
    evidence: str
    grounded_objects: List[str]
    recommendation: str


class RoomAnalysisReport(BaseModel):
    image_id: str
    room_type: str = "living room"
    detected_objects: List[DetectedObject]
    risks: List[RiskItem]
    overall_score: float = Field(ge=0, le=100)
    summary: str
