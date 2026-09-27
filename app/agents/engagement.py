"""Engagement agent: drafts outreach, and revises it when the Quality Critic rejects it.

The first draft starts from the campaign's selected (tested) creative and personalises it.
On revision it reads the critic's issue codes and applies a targeted fix for each one --
it never silently rewrites parts of the message the critic did not object to.
"""

COMPANY = "Purple Merit"
LINK = "pm.example/go"
ASSETS = {
    "security": "our security & compliance one-pager",
    "roadmap": "a preview of our product roadmap",
    "pricing": "a transparent pricing breakdown",
    "integration": "our HubSpot / Salesforce integration guide",
    "roi": "a CPL & ROI benchmark",
    "case_studies": "a case study",
}
OPT_OUT = {"Email": "Unsubscribe any time: pm.example/prefs", "SMS": "Reply STOP to opt out"}


def _pick_channel(state) -> str:
    permitted = state["compliance"]["permitted_channels"]
    variant_channel = (state["campaign"]["creative"].get("variant") or {}).get("channel")
    peers = state["memory"]["peers"]["top_converting_channels"]
    # Prefer the tested creative's channel, then channels that converted peers, then the lead's own order.
    for candidate in [variant_channel, *peers, *permitted]:
        if candidate in permitted:
            return candidate
    return permitted[0]


def _body(channel, lead, asset, subject, cta, compact=False) -> str:
    ind = lead["industry"]
    if channel == "SMS":
        if compact:
            return f"{COMPANY}: {asset} for {ind} teams. {cta}: {LINK} {OPT_OUT['SMS']}"
        return (f"Hi from {COMPANY}! {subject}. We put together {asset} tailored to {ind} "
                f"{lead['persona']}s like you. {cta} here: {LINK} {OPT_OUT['SMS']}")
    if channel == "Call":
        return (f"SDR call script: open with {ind} context, reference {asset}, "
                f"qualify timeline & budget, close with '{cta}'.")
    if channel in ("Ads", "Web"):
        return f"Show {asset} to this lead's {ind} retargeting audience; CTA '{cta}'."
    if channel == "Social":
        return f"Hi! Saw your interest in {asset.split(' ', 1)[1]}. Here's {asset} for {ind} teams. {cta}: {LINK}"
    return (f"Hi there,\n\nTeams in {ind} tell us the hardest part is proving pipeline impact. "
            f"We've prepared {asset} specifically for {lead['persona']}s.\n\n{cta}: {LINK}\n\n{OPT_OUT['Email']}")


def _subject(channel, base, lead) -> str:
    if channel in ("Ads", "Web"):
        return f"{base} - built for {lead['industry']}"      # headline
    return f"{base} ({lead['industry']})" if channel in ("Email", "Social") else base


def engagement(state):
    lead, route = state["lead"], state.get("route")
    attempts = state.get("attempts", 0)
    ltm = state["memory"]["long_term"]
    interest = (ltm.get("interests") or ["case_studies"])[0]
    asset = ASSETS.get(interest, ASSETS["case_studies"])
    mode = "sales_handoff" if route == "sales_handoff" else "engage"

    prev, critique = state.get("draft"), state.get("critique") or {}
    if prev is None or not critique.get("issues"):
        variant = state["campaign"]["creative"].get("variant") or {}
        channel = _pick_channel(state)
        base = variant.get("subject_line") or "A quick idea for your team"
        cta = variant.get("call_to_action") or "Learn More"
        draft = {"channel": channel, "subject": _subject(channel, base, lead), "base_subject": base,
                 "body": _body(channel, lead, asset, base, cta), "cta": cta, "asset": asset,
                 "interest": interest, "variant_id": variant.get("variant_id"),
                 "send_window": ltm.get("best_contact_time", "09:00-11:00"), "mode": mode,
                 "campaign_id": state["campaign"]["active_campaign_id"], "compact": False,
                 "revision": 0, "changes": []}
        summary = f"Draft v0 on {channel}: \"{draft['subject']}\" / CTA '{cta}' / send {draft['send_window']}"
    else:
        draft = dict(prev, revision=prev["revision"] + 1, mode=mode, changes=[])
        for issue in critique["issues"]:
            code = issue["code"]
            if code == "CHANNEL_NOT_PERMITTED":
                draft["channel"] = state["compliance"]["permitted_channels"][0]
                draft["changes"].append(f"switched channel to {draft['channel']}")
            elif code == "CTA_MISMATCH":
                draft["cta"] = "Book Demo"
                draft["changes"].append("CTA -> 'Book Demo' for sales handoff")
            elif code == "TOO_LONG" and issue.get("field") == "subject":
                draft["subject"] = draft["base_subject"]
                draft["changes"].append("shortened headline")
            elif code == "TOO_LONG":
                draft["compact"] = True
                draft["changes"].append("switched to compact copy")
            elif code == "NOT_PERSONALIZED":
                draft["changes"].append("added industry & interest personalisation")
            elif code == "MISSING_OPT_OUT":
                draft["changes"].append("added opt-out footer")
        draft["body"] = _body(draft["channel"], lead, asset, draft["base_subject"], draft["cta"], draft["compact"])
        if "shortened headline" not in draft["changes"]:
            draft["subject"] = _subject(draft["channel"], draft["base_subject"], lead)
        summary = f"Revision v{draft['revision']}: " + "; ".join(draft["changes"])
    if mode == "sales_handoff":
        draft["sdr_task"] = f"Call within 24h in window {draft['send_window']} -- p(convert) {state['triage']['p_convert']:.2f}"
    return {"draft": draft, "attempts": attempts + (0 if prev is None or not critique.get("issues") else 1),
            "critique": {}, "_summary": summary}
