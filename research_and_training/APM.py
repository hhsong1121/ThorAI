from fastapi import FastAPI, File, UploadFile, Form
from schemas import ClinicalData, EsophagectomyData, Report
from logic_lobectomy import LobectomyEngine
from logic_esophagectomy import EsophagectomyEngine
from zotero_rag import GuidelineRAG
from ai_engine import CXREngine # 🚨 1. 우리가 만든 AI 엔진 불러오기
import json
import os
from dotenv import load_dotenv
from agent_engine import ClinicalAgent

app = FastAPI(title="Thoracic CDSS API")

# 1. .env 파일의 내용을 시스템 환경 변수로 불러옵니다.
load_dotenv()

# 2. os.getenv를 통해 값을 가져옵니다.
ZOTERO_ID = os.getenv("ZOTERO_ID")
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # 추가

rag_system = GuidelineRAG(library_id=ZOTERO_ID, api_key=ZOTERO_API_KEY, target_tag="CDSS")
ai_engine = CXREngine() # 🚨 2. 서버가 켜질 때 AI 모델 장착

# API 키를 인자로 전달하여 생성 (에러 방지)
llm_agent = ClinicalAgent(api_key=GOOGLE_API_KEY)

engines = {
    "lobectomy": LobectomyEngine(),
    "esophagectomy": EsophagectomyEngine()
}

@app.on_event("startup")
def startup_event():
    rag_system.sync_and_build_knowledge_base()

def generate_soap_note(surgery_type, clinical, decision, reasons, ai_scores):
    """임상 데이터를 바탕으로 EMR 복사-붙여넣기용 SOAP 노트를 생성합니다."""
    # 오늘자 수치 추출
    crp_today = clinical.crp_trend[-1] if clinical.crp_trend[-1] is not None else 'Not checked'
    drain_today = clinical.drain_trend[-1] if clinical.drain_trend[-1] is not None else 0
    
    # [S]ubjective: 주관적 증상 (보통 회진 시 확인하므로 공란 처리)
    s = "S: No specific complaints."
    
    # [O]bjective: 객관적 수치 정리
    o_lines = [
        f"O: POD #{clinical.post_op_day} ({'Esophagectomy' if surgery_type == 'esophagectomy' else 'Lobectomy'})",
        f"   - V/S: SBP {clinical.sbp}mmHg, BT {clinical.bt}℃",
        f"   - Lab: WBC {clinical.wbc}, Hb {clinical.hb}, CRP {crp_today}",
        f"   - Drain: {drain_today}ml / 24h ({clinical.fluid_type})"
    ]
    if surgery_type == "esophagectomy":
        o_lines.append(f"   - Drain Amylase: {clinical.drain_amylase} U/L")
        if clinical.esophagogram_leak: o_lines.append("   - Esophagogram: Leak (+) 🛑")
    else:
        if clinical.air_leak_present: o_lines.append("   - Air leak: Present (+)")
        
    o_lines.append(f"   - CXR (AI): Pneumonia probability {ai_scores.get('Pneumonia', 0.0):.1%}")
    o = "\n".join(o_lines)
    
    # [A]ssessment: 환자 상태 평가
    a = "A: Post-operative management.\n   - " + ("Stable state." if "REMOVE" in decision and "ADVANCE" in decision else "Needs close observation.")
    
    # [P]lan: CDSS의 권고안을 그대로 계획으로 전환
    p = f"P: \n   - {decision}"
    for reason in reasons:
        p += f"\n   - {reason}"
        
    return f"{s}\n{o}\n{a}\n{p}"

@app.post("/api/v1/evaluate-full")
async def evaluate_full(image: UploadFile = File(...), data: str = Form(...)):
    # 1. 공통 데이터 파싱
    raw_dict = json.loads(data)
    s_type = raw_dict.get("surgery_type", "lobectomy")
    
    if s_type == "esophagectomy":
        clinical = EsophagectomyData(**raw_dict)
        engine = engines["esophagectomy"]
        # 🚨 [변경됨] 리스트에서 오늘 배액량을 꺼냅니다
        drain_today = clinical.drain_trend[-1] if clinical.drain_trend[-1] else 0 
        query = (f"Criteria for chest tube removal and diet initiation after esophagectomy. "
                 f"Drain amylase: {clinical.drain_amylase}, Drainage: {drain_today}ml, "
                 f"POD: {clinical.post_op_day}.")
    else:
        clinical = ClinicalData(**raw_dict)
        engine = engines["lobectomy"]
        # 🚨 [변경됨] 리스트에서 오늘 배액량을 꺼냅니다
        drain_today = clinical.drain_trend[-1] if clinical.drain_trend[-1] else 0
        query = (f"Criteria for chest tube removal after lung surgery lobectomy. "
                 f"Drainage: {drain_today}ml, Air leak: {clinical.air_leak_present}.")
    # 🚨 3. 진짜 AI 판독 및 히트맵 생성
    image_bytes = await image.read()
    ai_scores, heatmap_b64 = ai_engine.analyze(image_bytes) # 두 개를 받도록 수정!
    
    # 4. 선택된 엔진으로 로직 실행
    decision, reasons, alerts = engine.evaluate(clinical, ai_scores)
    
    # 5. Zotero RAG 근거 검색
    evidence = rag_system.get_clinical_evidence(query)
    # 🚨 [NEW] 5.5 SOAP 노트 자동 생성
    soap_note = generate_soap_note(s_type, clinical, decision, reasons, ai_scores)
    
    # 🚨 [NEW] 5.8 대망의 LLM 에이전트 동적 추론 실행!
    insight = llm_agent.get_insight(s_type, clinical, decision, reasons, evidence, ai_scores)
    
    # 6. 결과 반환 (agent_insight 추가)
    report = Report(
        decision=decision,
        reasons=reasons,
        alerts=alerts,
        ai_raw_scores=ai_scores,
        guideline_evidence=evidence,
        heatmap_base64=heatmap_b64,
        progress_note=soap_note,
        agent_insight=insight # 🚨 프론트엔드로 전송
    )
    return {"report": report.model_dump()}