"""Quality Critic: independent verification of every draft before it can be sent.

The engagement agent is never trusted to police itself -- the same principle as a
hallucination-detection layer sitting after an LLM. Every check is deterministic and
returns a machine-readable issue code that the engagement agent can act on.
"""
from app.config import MAX_REVISIONS

MEETING_CTAS = {"Book Demo", "Talk to Sales"}
BODY_LIMITS = {"SMS": 160, "Social": 280}
SUBJECT_LIMITS = {"Email": 70, "Ads": 40, "Web": 60}


def evaluate(draft: dict, lead: dict, permitted: list[str]) -> list[dict]:
    issues = []
    ch, body, subject = draft["channel"], draft["body"], draft["subject"]
    if ch not in permitted:
        issues.append({"code": "CHANNEL_NOT_PERMITTED", "detail": f"{ch} not in permitted {permitted}"})
    if ch in BODY_LIMITS and len(body) > BODY_LIMITS[ch]:
        issues.append({"code": "TOO_LONG", "field": "body", "detail": f"{ch} body {len(body)} > {BODY_LIMITS[ch]} chars"})
    if ch in SUBJECT_LIMITS and len(subject) > SUBJECT_LIMITS[ch]:
        issues.append({"code": "TOO_LONG", "field": "subject",
                       "detail": f"{ch} headline {len(subject)} > {SUBJECT_LIMITS[ch]} chars"})
    if lead["industry"].lower() not in body.lower():
        issues.append({"code": "NOT_PERSONALIZED", "detail": "body does not reference the lead's industry"})
    if draft["mode"] == "sales_handoff" and draft["cta"] not in MEETING_CTAS:
        issues.append({"code": "CTA_MISMATCH", "detail": f"hot lead needs a meeting CTA, got '{draft['cta']}'"})
    if ch in ("Email", "SMS") and "opt out" not in body.lower() and "unsubscribe" not in body.lower():
        issues.append({"code": "MISSING_OPT_OUT", "detail": f"{ch} requires an opt-out instruction"})
    if "{" in body or "}" in body:
        issues.append({"code": "UNRESOLVED_PLACEHOLDER", "detail": "template placeholder left in copy"})
    return issues


N_CHECKS = 6


def quality_critic(state):
    draft = state["draft"]
    issues = evaluate(draft, state["lead"], state["compliance"]["permitted_channels"])
    attempts = state.get("attempts", 0)
    passed = not issues
    escalate = not passed and attempts >= MAX_REVISIONS
    critique = {"passed": passed, "issues": issues, "attempt": attempts, "revision": draft["revision"],
                "score": round(1 - len({i["code"] for i in issues}) / N_CHECKS, 2), "escalate": escalate}
    if passed:
        summary = f"PASS on revision v{draft['revision']} ({N_CHECKS} checks)"
    elif escalate:
        summary = f"FAIL after {attempts} revisions -> escalating to human: " + "; ".join(i["detail"] for i in issues)
    else:
        summary = "FAIL -> send back for revision: " + "; ".join(i["detail"] for i in issues)
    return {"critique": critique, "_summary": summary}
