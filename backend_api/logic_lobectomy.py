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

        drain_today = clinical.drain_trend[-1] if clinical.drain_trend[-1] is not None else 0
        drain_yest = clinical.drain_trend[-2]
        crp_today = clinical.crp_trend[-1] if clinical.crp_trend[-1] is not None else 0
        crp_yest = clinical.crp_trend[-2]

        if clinical.air_leak_present:
            decision = "KEEP"
            reasons.append("Persistent air leak observed.")

        if drain_today > self.MAX_DRAINAGE:
            decision = "KEEP"
            reasons.append(
                f"Today's drain output exceeds {self.MAX_DRAINAGE} mL ({drain_today} mL)."
            )

        if drain_yest is not None and drain_today > 200 and drain_today >= drain_yest * 1.2:
            decision = "KEEP"
            reasons.append(
                f"Drain output increased by at least 20% compared with yesterday "
                f"({drain_yest} mL -> {drain_today} mL). Evaluate for bleeding."
            )

        if clinical.fluid_type in ["purulent", "chylous"]:
            decision = "KEEP"
            reasons.append(f"Abnormal drain fluid type ({clinical.fluid_type}).")

        if clinical.bt >= self.FEVER_BT or crp_today > self.MAX_CRP:
            alerts.append(
                Alert(
                    level="ACTION",
                    message="Fever or elevated CRP. Evaluate for infection.",
                )
            )

        if crp_yest is not None and crp_today >= crp_yest * 1.3:
            alerts.append(
                Alert(
                    level="CRITICAL",
                    message=(
                        f"CRP rose by at least 30% from the prior day "
                        f"({crp_yest} -> {crp_today}). Search for occult infection."
                    ),
                )
            )

        if ai_scores.get("Pneumonia", 0.0) > self.AI_PNEUMONIA_THRES:
            alerts.append(
                Alert(
                    level="CRITICAL",
                    message=(
                        f"AI suggests possible pneumonia "
                        f"({ai_scores.get('Pneumonia', 0.0):.1%}). Review antibiotic protocol."
                    ),
                )
            )

        return decision, reasons, [alert.model_dump() for alert in alerts]
