import os
import streamlit as st
import requests
import json
import pandas as pd
import datetime
import altair as alt
from PIL import Image
import base64
from io import BytesIO

# 설정 및 상수
API_BASE = "http://127.0.0.1:8000"
API_URL = f"{API_BASE}/api/v1/evaluate-full"
HEALTH_URL = f"{API_BASE}/api/v1/health"
SAMPLE_CXR_PATH = os.path.join(os.path.dirname(__file__), "demo_assets", "sample_cxr.png")

# ==========================================
# 🚀 1. 가상 FHIR EMR 연동 시뮬레이터
# ==========================================
def fetch_mock_fhir_data(patient_id):
    """실제로는 FHIR 서버(예: /Patient/123/Observation)에 API 요청을 보내 데이터를 가져옵니다."""
    # 시뮬레이션을 위해 "식도 절제술 후 정상 회복 중인 환자"의 데이터를 반환합니다.
    return {
        "surgery_type": "식도 절제술 (Esophagectomy)",
        "pod": 3,
        "sbp": 115, "bt": 36.9, "wbc": 8.2, "hb": 11.5,
        "crp_0": 14.5, "crp_1": 10.2, "crp_2": 4.1,   # CRP 뚜렷한 감소 추세
        "drain_0": 450, "drain_1": 350, "drain_2": 180, # 배액량 뚜렷한 감소 추세
        "fluid_type": "serous",
        "drain_amylase": 42.0,
        "esoph_leak": False,
        "air_leak": False
    }

def fetch_backend_status():
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


def init_page():
    st.set_page_config(page_title="Thoracic CDSS", page_icon="🫁", layout="wide")
    st.title("🫁 흉부외과 수술 후 통합 모니터링 CDSS")
    st.markdown("수술 후 트렌드와 AI 분석을 결합하여 최적의 발관 및 처치 시점을 권고합니다.")

    status = fetch_backend_status()
    if status and status.get("demo_mode"):
        st.info(
            "포트폴리오 데모 모드 — SLM·RAG·VinDr 객체 탐지는 로컬 전용입니다. "
            "NIH Grad-CAM, 룰 엔진, UI 플로우는 실제로 동작합니다.",
            icon="ℹ️",
        )

    # 🚨 [NEW] 세션 상태(Session State) 초기화: 값들을 저장해두는 메모리 공간
    defaults = {
        "surgery_type": "폐엽 절제술 (Lobectomy)", "pod": 2, "sbp": 110, "bt": 36.8, "wbc": 7.5, "hb": 13.2,
        "crp_0": 15.2, "crp_1": 0.0, "crp_2": 2.1,
        "drain_0": 650, "drain_1": 420, "drain_2": 250, "fluid_type": "serous",
        "drain_amylase": 45.0, "esoph_leak": False, "air_leak": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def render_sidebar():
    with st.sidebar:
        # ==========================================
        # 🚀 2. EMR 연동 UI 패널
        # ==========================================
        st.header("🏥 가상 EMR 연동 (FHIR)")
        st.caption("환자 번호를 입력하고 데이터를 불러오세요.")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            patient_id = st.text_input("Patient ID", value="PT-10293", label_visibility="collapsed")
        with c2:
            if st.button("데이터 로드", type="primary", use_container_width=True):
                # 버튼을 누르면 FHIR 데이터를 가져와서 session_state 메모리에 덮어씌웁니다.
                fetched_data = fetch_mock_fhir_data(patient_id)
                for k, v in fetched_data.items():
                    st.session_state[k] = v
                st.success("로드 완료!")
        
        st.divider()
        st.header("📋 수동 입력 / 확인")
        
        # 🚨 [NEW] 모든 입력 위젯에 key="이름"을 달아 session_state와 동기화시킵니다.
        surgery_type = st.selectbox("수술 종류", ["폐엽 절제술 (Lobectomy)", "식도 절제술 (Esophagectomy)"], key="surgery_type")
        pod = st.number_input("수술 후 일차 (POD)", min_value=0, key="pod")
        
        with st.expander("Vital Signs & Labs", expanded=False):
            sbp = st.number_input("SBP (mmHg)", key="sbp")
            bt = st.number_input("BT (°C)", step=0.1, key="bt")
            wbc = st.number_input("WBC (x10^3)", key="wbc")
            hb = st.number_input("Hb (g/dL)", key="hb")

        st.subheader("CRP Trend")
        crp_vals = [
            st.number_input("D-2 CRP", key="crp_0"),
            st.number_input("D-1 CRP", key="crp_1"),
            st.number_input("Today CRP", key="crp_2")
        ]

        st.subheader("Drainage Trend")
        drain_vals = [
            st.number_input("D-2 Drain (ml)", key="drain_0"),
            st.number_input("D-1 Drain (ml)", key="drain_1"),
            st.number_input("Today Drain (ml)", key="drain_2")
        ]
        
        fluid_type = st.selectbox("성상", ["serous", "bloody", "chylous", "purulent"], key="fluid_type")
        
        drain_amylase = 0.0
        esophagogram_leak = False
        air_leak = False
        
        if "식도" in surgery_type:
            st.subheader("Esophagectomy Specifics")
            drain_amylase = st.number_input("배액관 아밀라아제", step=1.0, key="drain_amylase")
            esophagogram_leak = st.toggle("식도조영술 상 누출", key="esoph_leak")
        else:
            air_leak = st.toggle("Air Leak 존재", key="air_leak")
            
        return {
            "surgery_type": "esophagectomy" if "식도" in surgery_type else "lobectomy",
            "post_op_day": pod, "sbp": sbp, "bt": bt, "wbc": wbc, "hb": hb,
            "crp_trend": [None if c == 0.0 else c for c in crp_vals], # 0.0은 편의상 None(미검사) 처리
            "drain_trend": drain_vals,
            "fluid_type": fluid_type, "air_leak_present": air_leak,
            "drain_amylase": drain_amylase, "esophagogram_leak": esophagogram_leak
        }

def render_charts(data):
    """중앙 상단: 트렌드 차트 시각화"""
    st.subheader("📈 Clinical Trends")
    dates = [(datetime.date.today() - datetime.timedelta(days=i)).strftime("%m-%d") for i in [2, 1, 0]]

    c1, c2 = st.columns(2)
    with c1:
        df_drain = pd.DataFrame({"Date": dates, "Drainage": data["drain_trend"]})
        chart = alt.Chart(df_drain).mark_line(point=True, color="#4A90E2").encode(
            x=alt.X('Date:O', sort=None), y='Drainage:Q')
        st.altair_chart(chart, use_container_width=True)
    with c2:
        df_crp = pd.DataFrame({"Date": dates, "CRP": data["crp_trend"]})
        chart = alt.Chart(df_crp).mark_line(point=True, color="#E24A4A").encode(
            x=alt.X('Date:O', sort=None), y='CRP:Q')
        st.altair_chart(chart, use_container_width=True)

def process_analysis(inputs):
    """분석 실행 및 결과 출력"""
    c1, c2 = st.columns([1, 1])

    if "demo_cxr_bytes" not in st.session_state:
        st.session_state.demo_cxr_bytes = None
    if "demo_cxr_name" not in st.session_state:
        st.session_state.demo_cxr_name = None

    with c1:
        st.subheader("📷 CXR Image")
        if os.path.exists(SAMPLE_CXR_PATH):
            if st.button("데모 샘플 CXR 불러오기", use_container_width=True):
                with open(SAMPLE_CXR_PATH, "rb") as sample_file:
                    st.session_state.demo_cxr_bytes = sample_file.read()
                    st.session_state.demo_cxr_name = "sample_cxr.png"
                st.rerun()

        file = st.file_uploader("Upload CXR", type=['png', 'jpg', 'jpeg'])
        if file:
            st.session_state.demo_cxr_bytes = file.getvalue()
            st.session_state.demo_cxr_name = file.name

        image_bytes = st.session_state.demo_cxr_bytes
        image_name = st.session_state.demo_cxr_name
        if image_bytes:
            st.image(Image.open(BytesIO(image_bytes)), use_container_width=True)
            heatmap_placeholder = st.empty()
        else:
            image_bytes = None
            image_name = None
            heatmap_placeholder = st.empty()

    with c2:
        st.subheader("🤖 Analysis Report")
        if image_bytes and st.button("Run Full Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                payload_data = inputs.copy()
                files = {"image": (image_name or "chest_xray.jpg", image_bytes, "image/jpeg")}
                try:
                    response = requests.post(
                        API_URL,
                        files=files,
                        data={"data": json.dumps(payload_data)},
                        timeout=300,
                    )
                except requests.exceptions.ConnectionError:
                    st.error(
                        "백엔드 API에 연결할 수 없습니다. "
                        "프로젝트 루트에서 `./run_backend.sh`를 실행했는지 확인하세요. "
                        "`.venv`(Python 3.11)를 사용해야 합니다."
                    )
                    return
                except requests.exceptions.Timeout:
                    st.error("분석 요청 시간이 초과되었습니다. 백엔드 로그를 확인하세요.")
                    return

                if response.status_code == 200:
                    payload = response.json()
                    report = payload["report"]
                    if payload.get("demo_mode"):
                        st.caption("응답: Portfolio Demo Mode")
                    display_report(report)
                    if report.get("heatmap_base64"):
                        heatmap_data = base64.b64decode(report["heatmap_base64"])
                        heatmap_placeholder.image(Image.open(BytesIO(heatmap_data)), use_container_width=True, caption="🔍 AI 시선 추적 (Grad-CAM)")
                else:
                    st.error(f"Server Error ({response.status_code}): {response.text}")

def display_report(report):
    """결과 리포트 UI 렌더링"""
    st.info(f"💡 **AI 수석 에이전트의 임상 브리핑**\n\n{report.get('agent_insight', '')}", icon="👨‍⚕️")
    st.divider()
    
    if report["decision"].startswith("REMOVE"):
        st.success(f"✅ **RECOMMENDATION: {report['decision']}**")
    else:
        st.error(f"🛑 **RECOMMENDATION: {report['decision']}**")
        
    for reason in report["reasons"]:
        st.write(f"• {reason}")

    st.divider()
    
    st.write("### 🚨 Clinical Alerts")
    if not report["alerts"]:
        st.info("특이 소견 없음. 안정적입니다.")
    else:
        for alert in report["alerts"]:
            if alert["level"] == "CRITICAL": st.error(alert["message"])
            else: st.warning(alert["message"])

    st.divider()

    st.write("### 🧠 AI 영상 판독 결과")
    ai = report.get("ai_raw_scores", {})
    if ai:
        st.progress(ai.get("Atelectasis", 0.0), text=f"무기폐 (Atelectasis): {ai.get('Atelectasis', 0.0):.1%}")
        st.progress(ai.get("Effusion", 0.0), text=f"흉수 (Effusion): {ai.get('Effusion', 0.0):.1%}")
        st.progress(ai.get("Pneumonia", 0.0), text=f"폐렴 (Pneumonia): {ai.get('Pneumonia', 0.0):.1%}")
        st.progress(ai.get("Pneumothorax", 0.0), text=f"기흉 (Pneumothorax): {ai.get('Pneumothorax', 0.0):.1%}")
    st.divider()
    
    with st.expander("📚 Guideline Evidence (RAG)", expanded=True):
        st.markdown(report["guideline_evidence"])

    # 🚨 [NEW] 경과기록지 출력 섹션 추가
    st.divider()
    st.write("### 📝 EMR 경과기록지 초안 (SOAP)")
    st.caption("우측 상단의 복사 버튼을 눌러 EMR에 바로 붙여넣으세요.")
    # st.code를 사용하면 자동으로 예쁜 복사 버튼이 생깁니다!
    st.code(report.get("progress_note", "기록지 생성 실패"), language="markdown")

if __name__ == "__main__":
    init_page()
    user_inputs = render_sidebar()
    render_charts(user_inputs)
    st.divider()
    process_analysis(user_inputs)