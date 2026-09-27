"""Decision Router: joins the parallel triage + optimizer branches and picks the path."""


def decision_router(state):
    t, c = state["triage"], state["campaign"]
    if t["uncertain"]:
        route = "human_review"
        reason = (f"p(convert)={t['p_convert']:.2f} is within {t['thresholds']['band']} of the hot threshold "
                  f"{t['thresholds']['hot']} -- too close to auto-decide between sales handoff and nurture")
    elif t["tier"] == "Hot":
        route, reason = "sales_handoff", f"Hot lead (p={t['p_convert']:.2f}) -> meeting-focused outreach + SDR task"
    elif t["tier"] == "Warm":
        route, reason = "engage", f"Warm lead (p={t['p_convert']:.2f}) -> personalised outreach"
    else:
        route, reason = "nurture", f"Cold lead (p={t['p_convert']:.2f}) -> low-cost nurture track"
    if c.get("reassigned_to") and route != "nurture":
        reason += f"; using {c['reassigned_to']} because {c['primary']['campaign_id']} is flagged Pause"
    return {"route": route, "route_reason": reason, "_summary": f"{route}: {reason}"}
