from pydantic import BaseModel
from typing import List, Optional

# 환자 입력 데이터 모델
class ClinicalData(BaseModel):
    post_op_day: int
    sbp: int
    bt: float
    wbc: float
    hb: float
    # 🚨 [변경됨] 단일 float/int에서 List로 변경 (None은 미검사 처리)
    crp_trend: List[Optional[float]]
    drain_trend: List[Optional[int]]
    fluid_type: str
    air_leak_present: bool

class EsophagectomyData(BaseModel):
    post_op_day: int
    sbp: int
    bt: float
    wbc: float
    hb: float
    # 🚨 [변경됨] 단일 float/int에서 List로 변경
    crp_trend: List[Optional[float]]
    drain_trend: List[Optional[int]]
    fluid_type: str
    drain_amylase: float
    esophagogram_leak: bool

# 알림 및 결과 응답 모델
class Alert(BaseModel):
    level: str  # "INFO", "ACTION", "CRITICAL"
    message: str

# Report 모델의 필드에 객체 탐지 결과를 담을 리스트 추가
class Report(BaseModel):
    decision: str
    reasons: List[str]
    alerts: List[dict]
    ai_raw_scores: dict
    vindr_detections: List[str] = []  # 🚨 [NEW] 바운딩 박스 탐지 결과
    guideline_evidence: str
    heatmap_base64: str = ""
    progress_note: str = ""
    agent_insight: str = ""