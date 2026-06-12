from schemas import ClinicalData, Alert

class LobectomyEngine:
    def __init__(self):
        self.MAX_DRAINAGE = 400
        self.MAX_CRP = 10.0
        self.FEVER_BT = 37.8
        self.AI_PNEUMONIA_THRES = 0.6

    def evaluate(self, clinical: ClinicalData, ai_scores: dict):
        decision = "REMOVE"
        reasons = []
        alerts = []

        # 🚨 [NEW] 시계열 데이터 추출 ([D-2, D-1, Today])
        drain_today = clinical.drain_trend[-1] if clinical.drain_trend[-1] is not None else 0
        drain_yest = clinical.drain_trend[-2]
        crp_today = clinical.crp_trend[-1] if clinical.crp_trend[-1] is not None else 0
        crp_yest = clinical.crp_trend[-2]

        # 1. 발관 기준 검토
        if clinical.air_leak_present:
            decision = "KEEP"
            reasons.append("공기 누출(Air leak)이 관찰됩니다.")
            
        # 절대량 초과
        if drain_today > self.MAX_DRAINAGE:
            decision = "KEEP"
            reasons.append(f"오늘 배액량이 {self.MAX_DRAINAGE}ml를 초과합니다 ({drain_today}ml).")
            
        # 🚨 [NEW] 시계열 추세 로직: 배액량이 전일 대비 20% 이상 증가 (튀는 경우)
        if drain_yest is not None and drain_today > 200 and drain_today >= drain_yest * 1.2:
            decision = "KEEP"
            reasons.append(f"배액량이 어제({drain_yest}ml) 대비 20% 이상 증가 추세입니다. 출혈 등 확인 요망.")

        if clinical.fluid_type in ["purulent", "chylous"]:
            decision = "KEEP"
            reasons.append(f"배액 성상이 비정상적입니다 ({clinical.fluid_type}).")

        # 2. 임상 경고(Alert) 생성
        if clinical.bt >= self.FEVER_BT or crp_today > self.MAX_CRP:
            alerts.append(Alert(level="ACTION", message="발열 또는 CRP 기준치 초과. 감염 여부 확인 요망."))
            
        # 🚨 [NEW] 시계열 추세 로직: CRP 급상승 (전일 대비 30% 이상)
        if crp_yest is not None and crp_today >= crp_yest * 1.3:
            alerts.append(Alert(level="CRITICAL", message=f"CRP가 전일 대비 30% 이상 급상승 중입니다 ({crp_yest} -> {crp_today}). 숨은 감염원 탐색 필수."))

        if ai_scores.get("Pneumonia", 0.0) > self.AI_PNEUMONIA_THRES:
            alerts.append(Alert(level="CRITICAL", message=f"AI 판독 결과 폐렴 의심 (확률 {ai_scores.get('Pneumonia', 0.0):.1%}). 항생제 프로토콜 검토."))

        return decision, reasons, [alert.model_dump() for alert in alerts]