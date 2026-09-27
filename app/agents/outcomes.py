"""Terminal agents: nurture enrolment, commit (persist + record), and suppression."""
import json
import threading
import time

from app.config import DECISIONS_LOG
from app.services import services

_log_lock = threading.Lock()


def _audit(record: dict):
    with _log_lock, DECISIONS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def nurture(state):
    permitted = state["compliance"]["permitted_channels"]
    channel = "Email" if "Email" in permitted else permitted[0]
    interests = state["memory"]["long_term"].get("interests") or ["case_studies"]
    plan = {"track": f"{interests[0]}-nurture", "channel": channel, "touches": 3, "cadence_days": 7,
            "rescore_in_days": 14, "campaign_id": state["campaign"]["active_campaign_id"]}
    return {"nurture_plan": plan,
            "_summary": f"Enrolled in {plan['track']} via {channel}: 3 touches / 7 days, re-score in 14 days"}


def commit(state):
    lead_id, run_id = state["lead_id"], state["run_id"]
    review = state.get("review") or {}
    if state.get("nurture_plan"):
        action, channel = "nurture", state["nurture_plan"]["channel"]
        campaign_id = state["nurture_plan"]["campaign_id"]
    else:
        draft = state["draft"]
        action, channel, campaign_id = draft["mode"], draft["channel"], draft["campaign_id"]

    store, graph = services.store, services.graph
    store.record_contact(lead_id, channel, run_id)                  # arms the frequency cap
    store.update_stm(lead_id, last_route=action, last_run=run_id, last_channel=channel, at=time.time())
    decision = {"run_id": run_id, "route": action, "channel": channel, "campaign_id": campaign_id,
                "p_convert": state["triage"]["p_convert"], "tier": state["triage"]["tier"],
                "human_reviewed": bool(review), "reviewer_action": review.get("action")}
    graph.record_decision(lead_id, decision)
    store.incr("routes", action)
    outcome = {**decision, "status": "committed", "lead_id": lead_id,
               "revisions": state.get("attempts", 0)}
    _audit(outcome)
    return {"outcome": outcome,
            "_summary": f"Committed '{action}' via {channel}; frequency cap armed; decision written to graph"}


def suppress(state):
    lead_id, run_id = state["lead_id"], state["run_id"]
    review = state.get("review") or {}
    reason = ("rejected by reviewer" if review.get("action") == "reject"
              else (state.get("compliance") or {}).get("reason", "suppressed"))
    decision = {"run_id": run_id, "route": "suppressed", "reason": reason,
                "campaign_id": state["lead"]["campaign_id"], "human_reviewed": bool(review)}
    services.graph.record_decision(lead_id, decision)
    services.store.incr("routes", "suppressed")
    outcome = {**decision, "status": "suppressed", "lead_id": lead_id}
    _audit(outcome)
    return {"outcome": outcome, "_summary": f"Suppressed: {reason}"}
