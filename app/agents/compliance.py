"""Compliance gate: hard rules that run BEFORE any model or outreach logic.

A lead that fails here is suppressed -- the ML model is never even consulted, the same way
a RAG system should return "not found" before calling the LLM when retrieval is weak.
"""
from app.services import services


def compliance_gate(state):
    lead, memory = state["lead"], state["memory"]
    checks = []

    consent = bool(lead["gdpr_consent"])
    checks.append({"check": "marketing_consent", "passed": consent,
                   "detail": "consent on record" if consent else "no GDPR marketing consent"})

    last = services.store.last_contact(lead["lead_id"])
    checks.append({"check": "frequency_cap", "passed": last is None,
                   "detail": "no recent contact" if last is None
                   else f"contacted via {last['channel']} recently; cap clears in {last['ttl_hours']}h"})

    # Channel-level opt-out: if the most recent touch ended in "unsubscribe", that channel is off-limits.
    latest = memory.get("latest_interaction") or {}
    opted_out = latest.get("channel") if latest.get("outcome") == "unsubscribe" else None

    preferred = [lead["preferred_channel"]] + memory["long_term"].get("preferred_channels", [])
    permitted = [c for c in dict.fromkeys(preferred) if c and c != opted_out]
    checks.append({"check": "permitted_channel", "passed": bool(permitted),
                   "detail": f"permitted: {', '.join(permitted) or 'none'}"
                             + (f" (opted out of {opted_out})" if opted_out else "")})

    failed = [c for c in checks if not c["passed"]]
    allowed = not failed
    compliance = {"allowed": allowed, "checks": checks, "permitted_channels": permitted,
                  "opted_out_channel": opted_out,
                  "reason": None if allowed else failed[0]["detail"]}
    summary = ("PASS - permitted channels: " + ", ".join(permitted)) if allowed else f"BLOCKED - {compliance['reason']}"
    return {"compliance": compliance, "_summary": summary}
