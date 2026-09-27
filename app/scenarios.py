"""Curated demo leads, each exercising a different path through the graph (all HealthTech)."""

SCENARIOS = [
    {"id": "hot", "lead_id": "L0000003", "title": "Hot lead -> sales handoff",
     "path": "triage Hot -> engagement -> critic rejects CTA -> revise -> commit",
     "shows": "parallel agents, critic revision loop, reason codes"},
    {"id": "warm_sms", "lead_id": "L0000391", "title": "Warm lead on SMS",
     "path": "engagement -> critic: SMS over 160 chars -> compact copy -> commit",
     "shows": "generator / critic loop on a hard channel constraint"},
    {"id": "review", "lead_id": "L0000173", "title": "Uncertain -> human review",
     "path": "p(convert) inside the uncertainty band -> interrupt() -> you decide",
     "shows": "human-in-the-loop with checkpointed state"},
    {"id": "reassign", "lead_id": "L0000107", "title": "Paused campaign",
     "path": "KMeans puts CMP0008 in the Pause cluster -> lead reassigned to the best Scale campaign",
     "shows": "campaign classification driving the decision"},
    {"id": "cold", "lead_id": "L0000011", "title": "Cold lead -> nurture",
     "path": "triage Cold -> low-cost nurture track, no personalised outreach",
     "shows": "conditional routing"},
    {"id": "consent", "lead_id": "L0000006", "title": "No consent -> suppressed",
     "path": "compliance gate blocks before the model is called",
     "shows": "hard rules before ML (fail closed)"},
    {"id": "optout", "lead_id": "L0000125", "title": "Channel opt-out",
     "path": "latest touch was an unsubscribe on the only preferred channel -> suppressed",
     "shows": "channel-level consent from interaction memory"},
]
FREQUENCY_CAP_TIP = "Run any committed lead a second time: the Redis frequency cap blocks it at the compliance gate."
