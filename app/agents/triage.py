"""Lead Triage agent: scikit-learn conversion probability -> tier, with an explicit uncertainty band."""
from app.config import COLD_THRESHOLD, HOT_THRESHOLD, UNCERTAINTY_BAND
from app.ml import model


def lead_triage(state):
    result = model.score(state["lead_id"])
    p = result["p_convert"]
    tier = "Hot" if p >= HOT_THRESHOLD else "Cold" if p < COLD_THRESHOLD else "Warm"
    # Near the Hot boundary a small calibration error flips the decision (sales time vs nurture),
    # so the agent declines to decide and escalates instead of guessing.
    distance = abs(p - HOT_THRESHOLD)
    uncertain = distance < UNCERTAINTY_BAND
    triage = {
        "p_convert": p,
        "tier": tier,
        "uncertain": uncertain,
        "margin": round(distance, 4),
        "thresholds": {"hot": HOT_THRESHOLD, "cold": COLD_THRESHOLD, "band": UNCERTAINTY_BAND},
        "reasons": result["reasons"],
    }
    why = "; ".join(f"{r['factor']} ({'+' if r['impact'] > 0 else ''}{r['impact']})" for r in result["reasons"])
    summary = f"p(convert)={p:.2f} -> {tier}{' [UNCERTAIN: within band of hot threshold]' if uncertain else ''}. Drivers: {why}"
    return {"triage": triage, "_summary": summary}
