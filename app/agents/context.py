"""Context / Memory agent: assembles everything the other agents need to know about a lead.

    long-term memory   seeds (durable preferences, RFM)
    short-term memory  Redis (what happened in recent runs)
    relational memory  Neo4j (how similar leads converted, prior decisions)
"""
from app import data
from app.services import services


def _mask_email(email: str) -> str:
    if not isinstance(email, str) or "@" not in email:
        return ""
    name, domain = email.split("@", 1)
    return f"{name[0]}***@{domain}"


def _mask_phone(phone: str) -> str:
    return f"***-{str(phone)[-4:]}" if isinstance(phone, str) and len(phone) >= 4 else ""


def load_context(state):
    lead_id = state["lead_id"]
    raw = data.get_lead(lead_id)
    if raw is None:
        raise ValueError(f"Lead {lead_id} not found")

    # PII is masked at the boundary: nothing downstream (traces, UI, logs) sees raw contact details.
    lead = {k: raw[k] for k in ("lead_id", "source", "campaign_id", "lead_status", "lead_score", "company_size",
                                "industry", "persona", "region", "preferred_channel", "gdpr_consent")}
    lead["email"], lead["phone"] = _mask_email(raw["email"]), _mask_phone(raw["phone"])
    lead["has_phone"] = bool(raw.get("phone"))

    inter = data.lead_interactions(lead_id)
    recent = inter.tail(5)[["timestamp", "channel", "event_type", "outcome"]].to_dict("records")[::-1]
    ltm = data.long_term_memory(lead_id)
    stm = services.store.get_stm(lead_id)
    peers = services.graph.peer_insights(lead)
    history = services.graph.lead_history(lead_id, limit=3)

    memory = {
        "long_term": ltm,
        "short_term": stm,
        "recent_interactions": recent,
        "latest_interaction": recent[0] if recent else None,
        "peers": peers,
        "prior_decisions": history,
    }
    summary = (f"{lead['industry']} {lead['persona']} ({lead['region']}), score {lead['lead_score']}; "
               f"{len(inter)} past touches; LTM {'found' if ltm else 'missing'}; "
               f"{len(history)} prior decision(s) in graph; peer conversion {peers['peer_conversion_rate']:.0%}")
    return {"lead": lead, "memory": memory, "_summary": summary}
