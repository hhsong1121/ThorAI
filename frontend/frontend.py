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

API_BASE = "http://127.0.0.1:8000"
API_URL = f"{API_BASE}/api/v1/evaluate-full"
HEALTH_URL = f"{API_BASE}/api/v1/health"
SAMPLE_CXR_PATH = os.path.join(os.path.dirname(__file__), "demo_assets", "sample_cxr.png")


def fetch_mock_fhir_data(patient_id):
    return {
        "pod": 2,
        "sbp": 110,
        "bt": 36.8,
        "wbc": 7.5,
        "hb": 13.2,
        "crp_0": 15.2,
        "crp_1": 0.0,
        "crp_2": 2.1,
        "drain_0": 650,
        "drain_1": 420,
        "drain_2": 250,
        "fluid_type": "serous",
        "air_leak": False,
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
    st.title("🫁 Thoracic Post-Operative CDSS")
    st.markdown(
        "Lobectomy recovery support: combines post-operative trends and AI analysis to recommend "
        "optimal chest tube management and clinical next steps."
    )

    status = fetch_backend_status()
    if status and status.get("demo_mode"):
        st.info(
            "Portfolio demo mode — NIH Grad-CAM, rule engine, and template briefing are live.",
            icon="ℹ️",
        )


def render_sidebar():
    defaults = {
        "pod": 2,
        "sbp": 110,
        "bt": 36.8,
        "wbc": 7.5,
        "hb": 13.2,
        "crp_0": 15.2,
        "crp_1": 0.0,
        "crp_2": 2.1,
        "drain_0": 650,
        "drain_1": 420,
        "drain_2": 250,
        "fluid_type": "serous",
        "air_leak": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    with st.sidebar:
        st.header("🏥 Mock EMR (FHIR)")
        st.caption("Enter a patient ID and load simulated chart data.")

        c1, c2 = st.columns([2, 1])
        with c1:
            patient_id = st.text_input("Patient ID", value="PT-10293", label_visibility="collapsed")
        with c2:
            if st.button("Load Data", type="primary", use_container_width=True):
                fetched_data = fetch_mock_fhir_data(patient_id)
                for key, value in fetched_data.items():
                    st.session_state[key] = value
                st.success("Loaded.")

        st.divider()
        st.header("📋 Manual Input")
        st.caption("Lobectomy post-operative assessment")

        pod = st.number_input("Post-Operative Day (POD)", min_value=0, key="pod")

        with st.expander("Vital Signs & Labs", expanded=False):
            sbp = st.number_input("SBP (mmHg)", key="sbp")
            bt = st.number_input("BT (°C)", step=0.1, key="bt")
            wbc = st.number_input("WBC (×10³)", key="wbc")
            hb = st.number_input("Hb (g/dL)", key="hb")

        st.subheader("CRP Trend")
        crp_vals = [
            st.number_input("D-2 CRP", key="crp_0"),
            st.number_input("D-1 CRP", key="crp_1"),
            st.number_input("Today CRP", key="crp_2"),
        ]

        st.subheader("Drainage Trend")
        drain_vals = [
            st.number_input("D-2 Drain (mL)", key="drain_0"),
            st.number_input("D-1 Drain (mL)", key="drain_1"),
            st.number_input("Today Drain (mL)", key="drain_2"),
        ]

        fluid_type = st.selectbox(
            "Drain Fluid Type",
            ["serous", "bloody", "chylous", "purulent"],
            key="fluid_type",
        )

        air_leak = st.toggle("Air Leak Present", key="air_leak")

        return {
            "surgery_type": "lobectomy",
            "post_op_day": pod,
            "sbp": sbp,
            "bt": bt,
            "wbc": wbc,
            "hb": hb,
            "crp_trend": [None if c == 0.0 else c for c in crp_vals],
            "drain_trend": drain_vals,
            "fluid_type": fluid_type,
            "air_leak_present": air_leak,
        }


def render_charts(data):
    st.subheader("📈 Clinical Trends")
    dates = [
        (datetime.date.today() - datetime.timedelta(days=i)).strftime("%m-%d")
        for i in [2, 1, 0]
    ]

    c1, c2 = st.columns(2)
    with c1:
        df_drain = pd.DataFrame({"Date": dates, "Drainage (mL)": data["drain_trend"]})
        chart = alt.Chart(df_drain).mark_line(point=True, color="#4A90E2").encode(
            x=alt.X("Date:O", sort=None),
            y="Drainage (mL):Q",
        )
        st.altair_chart(chart, use_container_width=True)
    with c2:
        df_crp = pd.DataFrame({"Date": dates, "CRP": data["crp_trend"]})
        chart = alt.Chart(df_crp).mark_line(point=True, color="#E24A4A").encode(
            x=alt.X("Date:O", sort=None),
            y="CRP:Q",
        )
        st.altair_chart(chart, use_container_width=True)


def process_analysis(inputs):
    c1, c2 = st.columns([1, 1])

    if "demo_cxr_bytes" not in st.session_state:
        st.session_state.demo_cxr_bytes = None
    if "demo_cxr_name" not in st.session_state:
        st.session_state.demo_cxr_name = None

    with c1:
        st.subheader("📷 CXR Image")
        if os.path.exists(SAMPLE_CXR_PATH):
            if st.button("Load Demo Sample CXR", use_container_width=True):
                with open(SAMPLE_CXR_PATH, "rb") as sample_file:
                    st.session_state.demo_cxr_bytes = sample_file.read()
                    st.session_state.demo_cxr_name = "sample_cxr.png"
                st.rerun()

        file = st.file_uploader("Upload CXR", type=["png", "jpg", "jpeg"])
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
                        "Cannot connect to the backend API. "
                        "Run `./run_demo.sh` from the project root and ensure `.venv` is active."
                    )
                    return
                except requests.exceptions.Timeout:
                    st.error("The analysis request timed out. Check the backend logs.")
                    return

                if response.status_code == 200:
                    payload = response.json()
                    report = payload["report"]
                    if payload.get("demo_mode"):
                        st.caption("Response: Portfolio Demo Mode")
                    display_report(report)
                    if report.get("heatmap_base64"):
                        heatmap_data = base64.b64decode(report["heatmap_base64"])
                        heatmap_placeholder.image(
                            Image.open(BytesIO(heatmap_data)),
                            use_container_width=True,
                            caption="AI attention overlay (Grad-CAM)",
                        )
                else:
                    st.error(f"Server Error ({response.status_code}): {response.text}")


def display_report(report):
    st.info(
        f"**AI Attending Briefing**\n\n{report.get('agent_insight', '')}",
        icon="👨‍⚕️",
    )
    st.divider()

    if report["decision"].startswith("REMOVE"):
        st.success(f"**RECOMMENDATION: {report['decision']}**")
    else:
        st.error(f"**RECOMMENDATION: {report['decision']}**")

    for reason in report["reasons"]:
        st.write(f"• {reason}")

    st.divider()
    st.write("### Clinical Alerts")
    if not report["alerts"]:
        st.info("No alerts. Patient appears stable.")
    else:
        for alert in report["alerts"]:
            if alert["level"] == "CRITICAL":
                st.error(alert["message"])
            else:
                st.warning(alert["message"])

    st.divider()
    st.write("### AI Chest X-ray Scores")
    ai = report.get("ai_raw_scores", {})
    if ai:
        st.progress(
            ai.get("Atelectasis", 0.0),
            text=f"Atelectasis: {ai.get('Atelectasis', 0.0):.1%}",
        )
        st.progress(
            ai.get("Effusion", 0.0),
            text=f"Pleural Effusion: {ai.get('Effusion', 0.0):.1%}",
        )
        st.progress(
            ai.get("Pneumonia", 0.0),
            text=f"Pneumonia: {ai.get('Pneumonia', 0.0):.1%}",
        )
        st.progress(
            ai.get("Pneumothorax", 0.0),
            text=f"Pneumothorax: {ai.get('Pneumothorax', 0.0):.1%}",
        )

    st.divider()
    with st.expander("Guideline Evidence (RAG)", expanded=True):
        st.markdown(report["guideline_evidence"])

    st.divider()
    st.write("### EMR Progress Note Draft (SOAP)")
    st.caption("Use the copy button to paste this note into the EMR.")
    st.code(report.get("progress_note", "Failed to generate progress note."), language="markdown")


if __name__ == "__main__":
    init_page()
    user_inputs = render_sidebar()
    render_charts(user_inputs)
    st.divider()
    process_analysis(user_inputs)
