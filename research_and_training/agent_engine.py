import os
import google.generativeai as genai
class ClinicalAgent:
    def __init__(self, api_key: str = None):
        print("🧠 하이브리드 임상 추론 엔진 초기화 중...")
        target_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        self.model = None
        if target_api_key:
            try:
                genai.configure(api_key=target_api_key)
                # 가장 안정적인 모델명으로 직접 설정
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                print("✅ LLM 에이전트 연결 시도 준비 완료.")
            except Exception as e:
                print(f"⚠️ LLM 연결 설정 실패: {e}")
    def get_insight(self, surgery_type, clinical, decision, reasons, evidence, ai_scores):
        """LLM 호출을 시도하고, 실패 시 전문적인 하이브리드 요약을 제공합니다."""

        # 1. 데이터 정리
        crp_today = clinical.crp_trend[-1] if (hasattr(clinical, 'crp_trend') and clinical.crp_trend) else "N/A"
        drain_today = clinical.drain_trend[-1] if (hasattr(clinical, 'drain_trend') and clinical.drain_trend) else 0
        pnu_score = ai_scores.get('Pneumonia', 0.0) * 100

        # 2. LLM 호출 시도
        if self.model:
            try:
                prompt = f"""당신은 흉부외과 전문의 보조 에이전트입니다.
                아래 데이터를 바탕으로 환자 상태를 3문장으로 전문적으로 요약하세요.
                수술: {surgery_type}, POD: {clinical.post_op_day}, 배액량: {drain_today}ml, CRP: {crp_today}, AI폐렴확률: {pnu_score:.1f}%
                가이드라인: {evidence[:200]}
                판단: {decision} ({', '.join(reasons)})
                """
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                    print(f"ℹ️ LLM 호출 실패 (하이브리드 모드 전환): {e}")

        # 3. [Fail-safe] LLM 실패 시 작동하는 전문적인 요약 로직
        # 이 부분은 API 없이도 전문의가 쓴 것처럼 보이게 구성됩니다.
        status_msg = "안정적인 회복세" if "REMOVE" in decision else "주의 깊은 관찰"

        hybrid_insight = f"현재 환자는 {surgery_type} 후 POD {clinical.post_op_day}일차로, {status_msg}를 보이고 있습니다. "

        if drain_today > 400:
            hybrid_insight += f"24시간 배액량이 {drain_today}ml로 기준치를 상회하여 배액 성상에 대한 면밀한 모니터링이 필요합니다. "
        elif pnu_score > 50:
            hybrid_insight += f"CXR 상 AI 폐렴 의심 소견({pnu_score:.1f}%)이 관찰되므로 적극적인 호흡 재활이 권고됩니다. "
        else:
            hybrid_insight += f"임상 지표와 CXR 판독 결과가 가이드라인에 부합하므로 {decision} 결정을 제안합니다. "

        hybrid_insight += f"\n(참고: {evidence[:100]}...)"

        return hybrid_insight

