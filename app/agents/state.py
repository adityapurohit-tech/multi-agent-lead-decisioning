"""Shared state that flows through the LangGraph workflow.

Each agent reads what it needs and writes only its own keys, which is what lets
lead_triage and campaign_optimizer run in parallel without conflicts. `trace` uses an
additive reducer so concurrent agents can both append to it.
"""
import operator
from typing import Annotated, TypedDict


class LeadState(TypedDict, total=False):
    run_id: str
    lead_id: str
    lead: dict            # context agent: profile (PII masked)
    memory: dict          # context agent: long-term prefs, short-term memory, recent interactions, graph peers
    compliance: dict      # compliance gate: allowed?, permitted channels, checks
    triage: dict          # lead triage: p(convert), tier, uncertainty, reason codes
    campaign: dict        # campaign optimizer: health class, KPIs, chosen creative variant
    route: str            # decision router: sales_handoff | engage | nurture | human_review
    route_reason: str
    draft: dict           # engagement: proposed outreach
    critique: dict        # quality critic: pass/fail + issues
    attempts: int         # engagement revisions so far
    review: dict          # human review decision
    nurture_plan: dict
    outcome: dict         # final committed decision
    trace: Annotated[list, operator.add]
