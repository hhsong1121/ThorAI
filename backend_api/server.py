from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import json

from app_config import DEMO_MODE
from schemas import ClinicalData, Report
from logic_lobectomy import LobectomyEngine
from ai_engine import CXREngine
from demo_assets import get_demo_guideline, generate_demo_insight

app = FastAPI(title="Thoracic CDSS - Portfolio Demo API")

print("Loading CXR AI engine...")
ai_engine = CXREngine()
lobectomy_engine = LobectomyEngine()
print("Server ready.")


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "demo_mode": DEMO_MODE,
    }


def generate_soap_note(clinical, decision, reasons, ai_scores):
    crp_today = clinical.crp_trend[-1] if clinical.crp_trend[-1] is not None else 'Not checked'
    drain_today = clinical.drain_trend[-1] if clinical.drain_trend[-1] is not None else 0

    s = "S: No specific complaints."
    o_lines = [
        f"O: POD #{clinical.post_op_day} (Lobectomy)",
        f"   - V/S: SBP {clinical.sbp}mmHg, BT {clinical.bt}℃",
        f"   - Lab: WBC {clinical.wbc}, Hb {clinical.hb}, CRP {crp_today}",
        f"   - Drain: {drain_today}ml / 24h ({clinical.fluid_type})",
    ]
    if clinical.air_leak_present:
        o_lines.append("   - Air leak: Present (+)")

    o_lines.append(f"   - CXR (AI): Pneumonia probability {ai_scores.get('Pneumonia', 0.0):.1%}")
    o = "\n".join(o_lines)
    a = "A: Post-operative management.\n   - " + (
        "Stable state." if "REMOVE" in decision else "Needs close observation."
    )
    p = f"P: \n   - {decision}"
    for reason in reasons:
        p += f"\n   - {reason}"

    return f"{s}\n{o}\n{a}\n{p}"


@app.post("/api/v1/evaluate-full")
async def evaluate_full(image: UploadFile = File(...), data: str = Form(...)):
    raw_dict = json.loads(data)
    clinical = ClinicalData(**raw_dict)
    drain_today = clinical.drain_trend[-1] if clinical.drain_trend[-1] else 0
    search_query = (
        f"Criteria for chest tube removal after lung surgery lobectomy. "
        f"Drainage {drain_today}ml, Air leak {clinical.air_leak_present}."
    )

    image_bytes = await image.read()
    ai_scores, heatmap_b64, detected_boxes = ai_engine.analyze(image_bytes)

    decision, reasons, alerts = lobectomy_engine.evaluate(clinical, ai_scores)

    retrieved_text = get_demo_guideline(search_query)

    soap_note = generate_soap_note(clinical, decision, reasons, ai_scores)
    insight = generate_demo_insight(
        clinical, decision, reasons, alerts,
        ai_scores, detected_boxes, retrieved_text,
    )

    report = Report(
        decision=decision,
        reasons=reasons,
        alerts=alerts,
        ai_raw_scores=ai_scores,
        vindr_detections=detected_boxes,
        guideline_evidence=retrieved_text[:500] + ("..." if len(retrieved_text) > 500 else ""),
        heatmap_base64=heatmap_b64,
        progress_note=soap_note,
        agent_insight=insight,
    )

    return {"demo_mode": DEMO_MODE, "report": report.model_dump()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
