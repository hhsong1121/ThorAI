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


def get_demo_guideline(search_query: str) -> str:
    header = f"[Demo guideline summary — query: {search_query[:80]}...]\n"
    return header + "\n\n".join(f"- {s}" for s in LOBECTOMY_GUIDELINES)


def generate_demo_insight(
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

    alert_summary = (
        f"{len(alerts)} alert(s) flagged — review urgently."
        if alerts
        else "No critical alerts at this time."
    )

    lines = [
        "[Portfolio Demo Mode — template insight]",
        "",
        f"POD {clinical.post_op_day} lobectomy patient: rule-based recommendation is {decision}.",
        alert_summary,
        "",
        "Imaging (NIH pretrained model):",
        f"- Pneumonia {pneumonia:.1%}, Effusion {effusion:.1%}, Pneumothorax {pneumothorax:.1%}.",
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
