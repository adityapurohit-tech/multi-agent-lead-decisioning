"""End-to-end tests of every path through the LangGraph workflow."""
import pytest

from app.agents import critic
from app.graph import after_critic, after_review
from app.orchestrator import get_orchestrator
from app.services import services


@pytest.fixture(autouse=True)
def clean_state():
    services.store.reset()
    yield


def run(lead_id):
    o = get_orchestrator()
    run_id = o.start(lead_id, background=False, source="test")
    return run_id, services.store.get_run(run_id), o.state(run_id)["values"]


def path(state):
    return [t["node"] for t in state["trace"]]


def test_hot_lead_goes_through_critic_loop_to_sales_handoff():
    _, meta, state = run("L0000003")
    assert meta["status"] == "completed" and meta["route"] == "sales_handoff"
    p = path(state)
    assert {"lead_triage", "campaign_optimizer"} <= set(p)       # both parallel branches ran
    assert p.count("engagement") == 2 and p.count("quality_critic") == 2   # one revision
    assert state["draft"]["cta"] in critic.MEETING_CTAS


def test_sms_draft_is_compacted_to_fit_160_chars():
    _, meta, state = run("L0000391")
    assert state["draft"]["channel"] == "SMS"
    assert len(state["draft"]["body"]) <= 160
    assert state["critique"]["passed"]


def test_no_consent_is_suppressed_before_model_runs():
    _, meta, state = run("L0000006")
    assert meta["route"] == "suppressed"
    assert "lead_triage" not in path(state)


def test_frequency_cap_blocks_second_contact():
    run("L0000003")
    _, meta, state = run("L0000003")
    assert meta["route"] == "suppressed"
    assert "cap clears" in state["compliance"]["reason"]


def test_paused_campaign_reassigns_lead():
    _, _, state = run("L0000107")
    c = state["campaign"]
    assert c["primary"]["health"] == "Pause" and c["reassigned_to"]
    assert state["draft"]["campaign_id"] == c["reassigned_to"]


def test_cold_lead_is_nurtured():
    _, meta, state = run("L0000011")
    assert meta["route"] == "nurture" and "engagement" not in path(state)


def test_uncertain_lead_pauses_for_human_and_resumes():
    o = get_orchestrator()
    run_id, meta, _ = run("L0000173")
    assert meta["status"] == "awaiting_review"
    o.resume(run_id, "reject", "not a fit", background=False)
    meta = services.store.get_run(run_id)
    assert meta["status"] == "completed" and meta["route"] == "suppressed"


def test_decisions_are_written_to_graph():
    run("L0000011")
    history = services.graph.lead_history("L0000011")
    assert history and history[0]["route"] == "nurture"


def test_critic_escalates_after_max_revisions():
    assert after_critic({"critique": {"passed": False, "escalate": True}}) == "human_review"
    assert after_critic({"critique": {"passed": False, "escalate": False}}) == "engagement"
    assert after_critic({"critique": {"passed": True, "escalate": False}}) == "commit"


def test_review_routing():
    assert after_review({"review": {"action": "approve"}, "critique": {"escalate": True}}) == "commit"
    assert after_review({"review": {"action": "approve"}}) == "engagement"
    assert after_review({"review": {"action": "nurture"}}) == "nurture"
    assert after_review({"review": {"action": "reject"}}) == "suppress"


def test_critic_flags_each_issue_code():
    lead = {"industry": "HealthTech"}
    draft = {"channel": "SMS", "body": "x" * 200, "subject": "s", "mode": "sales_handoff", "cta": "Learn More"}
    codes = {i["code"] for i in critic.evaluate(draft, lead, ["Email"])}
    assert codes == {"CHANNEL_NOT_PERMITTED", "TOO_LONG", "NOT_PERSONALIZED", "CTA_MISMATCH", "MISSING_OPT_OUT"}
