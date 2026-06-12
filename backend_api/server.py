from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import uvicorn
import json

from config import DEMO_MODE, MODEL_PATH, ADAPTER_PATH, QDRANT_DB_PATH, EMBED_MODEL_NAME
from schemas import ClinicalData, EsophagectomyData, Report
from logic_lobectomy import LobectomyEngine
from logic_esophagectomy import EsophagectomyEngine
from ai_engine import CXREngine
from demo_assets import get_demo_guideline, generate_demo_insight

app = FastAPI(title="Thoracic CDSS - Local SLM & RAG API")

model = None
tokenizer = None
retriever = None

if DEMO_MODE:
    print("Portfolio Demo Mode enabled — MLX, RAG, and VinDr detection use fallbacks.")
else:
    from mlx_lm import load, generate
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    import qdrant_client

    print("1/3: Loading fine-tuned MLX model...")
    try:
        model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
    except Exception as e:
        print(f"MLX load failed: {e}")
        model, tokenizer = None, None

    print("2/3: Connecting to local Qdrant vector DB...")
    try:
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        Settings.embed_model = embed_model
        Settings.llm = None

        client = qdrant_client.QdrantClient(path=QDRANT_DB_PATH)
        vector_store = QdrantVectorStore(client=client, collection_name="thoracic_guidelines")
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        retriever = index.as_retriever(similarity_top_k=2)
    except Exception as e:
        print(f"DB connection failed: {e}")
        retriever = None

print("3/3: Loading CXR AI engine...")
ai_engine = CXREngine()

engines = {
    "lobectomy": LobectomyEngine(),
    "esophagectomy": EsophagectomyEngine()
}

print("Server ready.")


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "demo_mode": DEMO_MODE,
        "mlx_loaded": model is not None,
        "rag_loaded": retriever is not None,
    }


def generate_soap_note(surgery_type, clinical, decision, reasons, ai_scores):
    crp_today = clinical.crp_trend[-1] if clinical.crp_trend[-1] is not None else 'Not checked'
    drain_today = clinical.drain_trend[-1] if clinical.drain_trend[-1] is not None else 0

    s = "S: No specific complaints."
    o_lines = [
        f"O: POD #{clinical.post_op_day} ({'Esophagectomy' if surgery_type == 'esophagectomy' else 'Lobectomy'})",
        f"   - V/S: SBP {clinical.sbp}mmHg, BT {clinical.bt}℃",
        f"   - Lab: WBC {clinical.wbc}, Hb {clinical.hb}, CRP {crp_today}",
        f"   - Drain: {drain_today}ml / 24h ({clinical.fluid_type})"
    ]
    if surgery_type == "esophagectomy":
        o_lines.append(f"   - Drain Amylase: {clinical.drain_amylase} U/L")
        if clinical.esophagogram_leak:
            o_lines.append("   - Esophagogram: Leak (+)")
    else:
        if clinical.air_leak_present:
            o_lines.append("   - Air leak: Present (+)")

    o_lines.append(f"   - CXR (AI): Pneumonia probability {ai_scores.get('Pneumonia', 0.0):.1%}")
    o = "\n".join(o_lines)
    a = "A: Post-operative management.\n   - " + (
        "Stable state." if "REMOVE" in decision and "ADVANCE" in decision else "Needs close observation."
    )
    p = f"P: \n   - {decision}"
    for reason in reasons:
        p += f"\n   - {reason}"

    return f"{s}\n{o}\n{a}\n{p}"


def _generate_agent_insight(
    s_type,
    clinical,
    drain_today,
    decision,
    reasons,
    alerts,
    ai_scores,
    detected_boxes,
    retrieved_text,
):
    if DEMO_MODE or model is None or tokenizer is None:
        return generate_demo_insight(
            s_type, clinical, decision, reasons, alerts,
            ai_scores, detected_boxes, retrieved_text,
        )

    from mlx_lm import generate

    patient_status = (
        f"POD {clinical.post_op_day} patient ({s_type}). Drain {drain_today}ml. "
        f"Rule recommendation: {decision}. Alerts: {len(alerts)}."
    )
    ai_vision_status = (
        f"VinDr detections: {', '.join(detected_boxes) if detected_boxes else 'none'}."
    )
    instruction = (
        "You are a thoracic surgery AI agent. Summarize clinical status, rule output, "
        "and guideline context for the care team in concise medical language."
    )
    full_content = (
        f"{instruction}\n\n"
        f"[Patient summary]\n{patient_status}\n"
        f"[AI vision]\n{ai_vision_status}\n\n"
        f"[Rationale]\n{'; '.join(reasons)}\n\n"
        f"[Guidelines]\n{retrieved_text[:1000]}"
    )
    messages = [{"role": "user", "content": full_content}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        return generate(model, tokenizer, prompt=prompt_text, max_tokens=400, verbose=False)
    except Exception as e:
        print(f"LLM inference error: {e}")
        return generate_demo_insight(
            s_type, clinical, decision, reasons, alerts,
            ai_scores, detected_boxes, retrieved_text,
        )


@app.post("/api/v1/evaluate-full")
async def evaluate_full(image: UploadFile = File(...), data: str = Form(...)):
    if not DEMO_MODE and model is None:
        raise HTTPException(status_code=500, detail="SLM is not loaded. Set DEMO_MODE=1 for portfolio demo.")

    raw_dict = json.loads(data)
    s_type = raw_dict.get("surgery_type", "lobectomy")

    if s_type == "esophagectomy":
        clinical = EsophagectomyData(**raw_dict)
        engine = engines["esophagectomy"]
        drain_today = clinical.drain_trend[-1] if clinical.drain_trend[-1] else 0
        search_query = (
            f"Criteria for chest tube removal and diet initiation after esophagectomy. "
            f"POD {clinical.post_op_day}, Drain {drain_today}ml."
        )
    else:
        clinical = ClinicalData(**raw_dict)
        engine = engines["lobectomy"]
        drain_today = clinical.drain_trend[-1] if clinical.drain_trend[-1] else 0
        search_query = (
            f"Criteria for chest tube removal after lung surgery lobectomy. "
            f"Drainage {drain_today}ml, Air leak {clinical.air_leak_present}."
        )

    image_bytes = await image.read()
    ai_scores, heatmap_b64, detected_boxes = ai_engine.analyze(image_bytes)

    decision, reasons, alerts = engine.evaluate(clinical, ai_scores)

    retrieved_text = "No guideline evidence retrieved."
    if DEMO_MODE:
        retrieved_text = get_demo_guideline(s_type, search_query)
    elif retriever:
        try:
            nodes = retriever.retrieve(search_query)
            retrieved_text = "\n".join([n.node.get_content() for n in nodes])
        except Exception as e:
            print(f"RAG search error: {e}")

    soap_note = generate_soap_note(s_type, clinical, decision, reasons, ai_scores)
    insight = _generate_agent_insight(
        s_type, clinical, drain_today, decision, reasons, alerts,
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
