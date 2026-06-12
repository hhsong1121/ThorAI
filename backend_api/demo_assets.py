"""Portfolio demo fallbacks for RAG and agent insight (no proprietary PDFs or weights)."""

LOBECTOMY_GUIDELINES = [
    (
        "Chest tube removal after lobectomy is generally considered when daily drainage "
        "is below 200–250 mL, the output is serous, and there is no persistent air leak. "
        "Clinical stability and radiographic lung expansion should be confirmed."
    ),
    (
        "Persistent air leak beyond POD 5–7 may require continued water-seal management "
        "or specialist review. Monitor for signs of tension pneumothorax or infection."
    ),
    (
        "Post-lobectomy fever with rising inflammatory markers and purulent drainage "
        "warrants evaluation for pneumonia or empyema, including culture studies and "
        "repeat chest imaging."
    ),
]

ESOPHAGECTOMY_GUIDELINES = [
    (
        "After esophagectomy, chest tube removal is typically considered when drainage "
        "is low-volume and serous, drain amylase is not elevated, and there is no "
        "clinical or radiographic evidence of anastomotic leak."
    ),
    (
        "Diet advancement after esophagectomy depends on hemodynamic stability, "
        "decreasing drain output, and absence of leak on esophagogram when indicated."
    ),
    (
        "Elevated drain amylase, fever, or positive esophagogram leak should prompt "
        "hold on oral intake, broad workup, and surgical consultation."
    ),
]


def get_demo_guideline(surgery_type: str, search_query: str) -> str:
    snippets = ESOPHAGECTOMY_GUIDELINES if surgery_type == "esophagectomy" else LOBECTOMY_GUIDELINES
    header = f"[Demo guideline summary — query: {search_query[:80]}...]\n"
    return header + "\n\n".join(f"- {s}" for s in snippets)


def generate_demo_insight(
    surgery_type: str,
    clinical,
    decision: str,
    reasons: list,
    alerts: list,
    ai_scores: dict,
    detected_boxes: list,
    retrieved_text: str,
) -> str:
    pneumonia = ai_scores.get("Pneumonia", 0.0)
    effusion = ai_scores.get("Effusion", 0.0)
    pneumothorax = ai_scores.get("Pneumothorax", 0.0)

    vision_line = (
        "VinDr object detection is disabled in demo mode; rely on NIH Grad-CAM scores."
        if not detected_boxes
        else f"Detected findings: {', '.join(detected_boxes)}."
    )

    alert_summary = (
        f"{len(alerts)} alert(s) flagged — review urgently."
        if alerts
        else "No critical alerts at this time."
    )

    lines = [
        "[Portfolio Demo Mode — template insight, not from a fine-tuned SLM]",
        "",
        f"POD {clinical.post_op_day} {surgery_type} patient: rule-based recommendation is {decision}.",
        alert_summary,
        "",
        "Imaging (NIH pretrained model):",
        f"- Pneumonia {pneumonia:.1%}, Effusion {effusion:.1%}, Pneumothorax {pneumothorax:.1%}.",
        vision_line,
        "",
        "Key rationale:",
    ]
    for reason in reasons[:4]:
        lines.append(f"- {reason}")

    lines.extend([
        "",
        "Guideline context (paraphrased demo snippets):",
        retrieved_text[:400] + ("..." if len(retrieved_text) > 400 else ""),
    ])
    return "\n".join(lines)
