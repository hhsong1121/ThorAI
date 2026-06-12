from schemas import EsophagectomyData, Alert

class EsophagectomyEngine:
    def __init__(self):
        self.MAX_DRAINAGE = 300
        self.MAX_CRP = 12.0
        self.FEVER_BT = 37.8
        self.AMYLASE_LEAK_THRES = 250.0

    def evaluate(self, clinical: EsophagectomyData, ai_scores: dict):
        drain_decision = "REMOVE (발관 권장)"
        diet_decision = "ADVANCE DIET (식이 진행)"
        reasons = []
        alerts = []

        # 🚨 [NEW] 시계열 데이터 추출
        drain_today = clinical.drain_trend[-1] if clinical.drain_trend[-1] is not None else 0
        drain_yest = clinical.drain_trend[-2]
        crp_today = clinical.crp_trend[-1] if clinical.crp_trend[-1] is not None else 0
        crp_yest = clinical.crp_trend[-2]

        # 1. 초응급 합병증 필터링
        is_leaking = clinical.esophagogram_leak or clinical.drain_amylase > self.AMYLASE_LEAK_THRES
        is_chylothorax = clinical.fluid_type == "chylous"
        
        # 🚨 [NEW] 시계열 추세: CRP 50% 이상 폭증 시 Leak 강력 의심
        crp_spiked = (crp_yest is not None and crp_today >= crp_yest * 1.5 and crp_today > 10.0)

        if is_leaking or crp_spiked:
            drain_decision = "KEEP (유지)"
            diet_decision = "NPO (절대 금식)"
            if is_leaking:
                reasons.append("🛑 문합부 누출(Leak) 소견: 조영술 양성 또는 아밀라아제 비정상 상승.")
                alerts.append(Alert(level="CRITICAL", message="문합부 누출(Anastomotic Leak) 의심. 즉각적인 금식 및 영상 검사 요망."))
            if crp_spiked:
                reasons.append(f"🛑 CRP 전일 대비 50% 이상 폭증 ({crp_yest} -> {crp_today}). 누출 위험으로 식이를 전면 보류합니다.")
                if not is_leaking:
                    alerts.append(Alert(level="CRITICAL", message="CRP 폭증 소견. 숨겨진 문합부 누출이나 중증 패혈증이 강력히 의심됩니다!"))
                    
        elif is_chylothorax:
            drain_decision = "KEEP (유지)"
            diet_decision = "NPO or MCT DIET (금식/중쇄지방산)"
            reasons.append("🛑 배액 성상이 유미(Chylous) 형태입니다.")
            alerts.append(Alert(level="CRITICAL", message="유미흉 발생 의심. 일반 식이 중단 및 TPN 전환 검토."))

        # 2. 일반 흉관 발관 로직
        if not (is_leaking or is_chylothorax or crp_spiked):
            if drain_today > self.MAX_DRAINAGE:
                drain_decision = "KEEP (유지)"
                reasons.append(f"배액량이 안전 기준 초과입니다 ({drain_today}ml).")
            # 🚨 [NEW] 배액량 증가 추세 모니터링
            if drain_yest is not None and drain_today > 200 and drain_today >= drain_yest * 1.2:
                drain_decision = "KEEP (유지)"
                reasons.append(f"배액량이 어제({drain_yest}ml) 대비 증가 추세입니다. 관찰이 더 필요합니다.")
            
            if clinical.fluid_type == "purulent":
                drain_decision = "KEEP (유지)"
                reasons.append("농흉(Empyema) 의심 성상(purulent)입니다.")

        # 3. 단계적 식이 진행 로직
        if not (is_leaking or is_chylothorax or crp_spiked):
            if clinical.bt >= self.FEVER_BT or crp_today > self.MAX_CRP:
                diet_decision = "HOLD DIET (식이 보류)"
                reasons.append("발열 또는 염증 수치 상승으로 식이를 보류합니다.")
            else:
                if clinical.post_op_day <= 1:
                    diet_decision = "NPO (금식)"
                    reasons.append(f"POD {clinical.post_op_day}: 안정화 대기.")
                elif clinical.post_op_day == 2:
                    diet_decision = "SIPS OF WATER (소량 물 섭취)"
                    reasons.append(f"POD {clinical.post_op_day}: 물(Sips) 섭취 시도.")
                elif clinical.post_op_day <= 4:
                    diet_decision = "LIQUID DIET (유동식)"
                    reasons.append(f"POD {clinical.post_op_day}: 미음 시작. 사레들림 주의.")
                else:
                    diet_decision = "SOFT DIET (죽식)"
                    reasons.append(f"POD {clinical.post_op_day}: 연식으로 진행.")

        if ai_scores.get("Pneumonia", 0.0) > 0.5:
            alerts.append(Alert(level="ACTION", message=f"AI 폐렴 징후({ai_scores.get('Pneumonia', 0.0)*100:.1f}%). 흡인성 폐렴 주의."))

        final_decision = f"Drain: {drain_decision}  |  Diet: {diet_decision}"
        return final_decision, reasons, [alert.model_dump() for alert in alerts]