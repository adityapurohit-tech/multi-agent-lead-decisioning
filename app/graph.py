"""LangGraph workflow definition.

                         START
                           |
                     load_context          (Redis STM + seeds LTM + Neo4j peers/history)
                           |
                    compliance_gate ------------------------------+
                     /           \\   (parallel fan-out)            | blocked
             lead_triage    campaign_optimizer                     |
                     \\           /    (join: waits for both)        |
                    decision_router                                |
            /             |               \\                        |
     engagement <---+   nurture       human_review  (interrupt)    |
         |          | revise              |                        |
     quality_critic-+                     +--> engagement / nurture / commit / suppress
         |    \\ escalate -> human_review                          |
       commit <------------------------------------------------ suppress
         |                                                         |
        END                                                       END
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.compliance import compliance_gate
from app.agents.context import load_context
from app.agents.critic import quality_critic
from app.agents.engagement import engagement
from app.agents.optimizer import campaign_optimizer
from app.agents.outcomes import commit, nurture, suppress
from app.agents.review import human_review
from app.agents.router import decision_router
from app.agents.state import LeadState
from app.agents.triage import lead_triage
from app.observability import traced
from app.services import services

NODES = {
    "load_context": load_context,
    "compliance_gate": compliance_gate,
    "lead_triage": lead_triage,
    "campaign_optimizer": campaign_optimizer,
    "decision_router": decision_router,
    "engagement": engagement,
    "quality_critic": quality_critic,
    "human_review": human_review,
    "nurture": nurture,
    "commit": commit,
    "suppress": suppress,
}


# ---- conditional edges -------------------------------------------------------
def after_compliance(state):
    # Returning a list fans out: both agents run concurrently in the same super-step.
    return ["lead_triage", "campaign_optimizer"] if state["compliance"]["allowed"] else "suppress"


def after_router(state):
    return {"sales_handoff": "engagement", "engage": "engagement",
            "nurture": "nurture", "human_review": "human_review"}[state["route"]]


def after_critic(state):
    c = state["critique"]
    if c["passed"]:
        return "commit"
    return "human_review" if c["escalate"] else "engagement"     # the revision loop


def after_review(state):
    action = state["review"]["action"]
    if action == "reject":
        return "suppress"
    if action == "nurture":
        return "nurture"
    # approve: an escalated draft is sent as-is; an escalated triage decision proceeds to engagement
    return "commit" if (state.get("critique") or {}).get("escalate") else "engagement"


def build_graph(checkpointer=None):
    g = StateGraph(LeadState)
    for name, fn in NODES.items():
        g.add_node(name, traced(lambda: services.tracer, name)(fn))

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "compliance_gate")
    g.add_conditional_edges("compliance_gate", after_compliance, ["lead_triage", "campaign_optimizer", "suppress"])
    g.add_edge(["lead_triage", "campaign_optimizer"], "decision_router")
    g.add_conditional_edges("decision_router", after_router, ["engagement", "nurture", "human_review"])
    g.add_edge("engagement", "quality_critic")
    g.add_conditional_edges("quality_critic", after_critic, ["engagement", "commit", "human_review"])
    g.add_conditional_edges("human_review", after_review, ["engagement", "nurture", "commit", "suppress"])
    g.add_edge("nurture", "commit")
    g.add_edge("commit", END)
    g.add_edge("suppress", END)
    # Checkpointing is what makes human-in-the-loop possible: state survives the pause.
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def graph_structure(compiled) -> dict:
    gg = compiled.get_graph()
    return {
        "nodes": [n for n in gg.nodes],
        "edges": [{"source": e.source, "target": e.target, "conditional": e.conditional} for e in gg.edges],
        "mermaid": gg.draw_mermaid(),
    }
