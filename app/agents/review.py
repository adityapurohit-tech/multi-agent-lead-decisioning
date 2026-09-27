"""Human-in-the-loop review using LangGraph's interrupt().

The graph pauses here, its full state is checkpointed, and execution resumes only when a
reviewer submits a decision (dashboard button or API) via Command(resume=...).
"""
from langgraph.types import interrupt


def human_review(state):
    critique = state.get("critique") or {}
    if critique.get("escalate"):
        reason = f"Quality critic rejected the draft {critique['attempt']} times"
    else:
        reason = state.get("route_reason", "Escalated for review")
    payload = {
        "lead_id": state["lead_id"],
        "reason": reason,
        "p_convert": state["triage"]["p_convert"],
        "tier": state["triage"]["tier"],
        "reasons": state["triage"]["reasons"],
        "draft": state.get("draft"),
        "options": ["approve", "nurture", "reject"],
    }
    decision = interrupt(payload)          # <- execution pauses here until resumed
    action = decision.get("action", "reject")
    review = {"action": action, "note": decision.get("note", ""), "reviewer": decision.get("reviewer", "reviewer"),
              "reason": reason}
    return {"review": review, "_summary": f"Reviewer chose '{action}'" + (f": {review['note']}" if review["note"] else "")}
