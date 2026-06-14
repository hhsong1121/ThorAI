from pydantic import BaseModel
from typing import List, Optional


class ClinicalData(BaseModel):
    post_op_day: int
    sbp: int
    bt: float
    wbc: float
    hb: float
    crp_trend: List[Optional[float]]
    drain_trend: List[Optional[int]]
    fluid_type: str
    air_leak_present: bool


class Alert(BaseModel):
    level: str
    message: str


class Report(BaseModel):
    decision: str
    reasons: List[str]
    alerts: List[dict]
    ai_raw_scores: dict
    vindr_detections: List[str] = []
    guideline_evidence: str
    heatmap_base64: str = ""
    progress_note: str = ""
    agent_insight: str = ""
