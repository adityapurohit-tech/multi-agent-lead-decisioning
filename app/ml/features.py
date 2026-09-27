"""Feature engineering for lead scoring: profile attributes + engagement history + long-term memory."""
from functools import lru_cache

import pandas as pd

from app import data
from app.config import REFERENCE_DATE

CATEGORICAL = ["source", "company_size", "industry", "persona", "region", "preferred_channel"]
NUMERIC = [
    "lead_score", "rfm_score", "n_touches", "n_positive", "n_callback",
    "n_negative", "n_unsubscribe", "n_replies", "n_clicks", "days_since_active",
]
FEATURES = NUMERIC + CATEGORICAL

REPLY_EVENTS = ["reply", "sms_reply", "social_comment"]
CLICK_EVENTS = ["email_click", "site_visit"]


def build_features(leads: pd.DataFrame, interactions: pd.DataFrame, ltm: pd.DataFrame) -> pd.DataFrame:
    outcomes = pd.crosstab(interactions["lead_id"], interactions["outcome"])
    events = pd.crosstab(interactions["lead_id"], interactions["event_type"])
    eng = pd.DataFrame(index=outcomes.index)
    eng["n_touches"] = outcomes.sum(axis=1)
    eng["n_positive"] = outcomes.get("positive", 0)
    eng["n_callback"] = outcomes.get("callback_requested", 0)
    eng["n_negative"] = outcomes.get("negative", 0)
    eng["n_unsubscribe"] = outcomes.get("unsubscribe", 0)
    eng["n_replies"] = events.reindex(columns=REPLY_EVENTS, fill_value=0).sum(axis=1)
    eng["n_clicks"] = events.reindex(columns=CLICK_EVENTS, fill_value=0).sum(axis=1)

    f = leads.set_index("lead_id")[["lead_score", "last_active_at"] + CATEGORICAL].copy()
    f = f.join(eng).join(ltm.set_index("lead_id")[["rfm_score"]])
    f[eng.columns] = f[eng.columns].fillna(0)
    # Leads without long-term memory get the population median rather than 0 (0 would read as "worst").
    f["rfm_score"] = f["rfm_score"].fillna(ltm["rfm_score"].median())
    ref = pd.Timestamp(REFERENCE_DATE)
    f["days_since_active"] = (ref - pd.to_datetime(f["last_active_at"])).dt.days.clip(lower=0)
    return f[FEATURES]


@lru_cache(maxsize=None)
def feature_table() -> pd.DataFrame:
    return build_features(data.table("leads"), data.table("interactions"), data.table("memory_long_term"))


def features_for(lead_id: str) -> pd.DataFrame:
    return feature_table().loc[[lead_id]]
