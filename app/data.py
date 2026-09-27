"""Loads the seed CSVs once and exposes lookups used by the agents."""
import json
from functools import lru_cache

import pandas as pd

from app.config import SEEDS_DIR


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # Seed files use a non-breaking hyphen (U+2011) in values like "E‑commerce".
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.replace("‑", "-", regex=False)
    return df


@lru_cache(maxsize=None)
def table(name: str) -> pd.DataFrame:
    return _clean(pd.read_csv(SEEDS_DIR / f"{name}.csv"))


@lru_cache(maxsize=None)
def _leads_indexed() -> pd.DataFrame:
    return table("leads").set_index("lead_id", drop=False)


@lru_cache(maxsize=None)
def _ltm_indexed() -> pd.DataFrame:
    return table("memory_long_term").set_index("lead_id")


@lru_cache(maxsize=None)
def _interactions_by_lead():
    return {k: g.sort_values("timestamp") for k, g in table("interactions").groupby("lead_id")}


def get_lead(lead_id: str) -> dict | None:
    df = _leads_indexed()
    if lead_id not in df.index:
        return None
    row = df.loc[lead_id].to_dict()
    row["gdpr_consent"] = str(row.get("gdpr_consent")).lower() == "true"
    return row


def long_term_memory(lead_id: str) -> dict:
    df = _ltm_indexed()
    if lead_id not in df.index:
        return {}
    row = df.loc[lead_id]
    prefs = json.loads(row["preferences_json"]) if isinstance(row["preferences_json"], str) else {}
    return {"rfm_score": float(row["rfm_score"]), "last_updated_at": row["last_updated_at"], **prefs}


def lead_interactions(lead_id: str) -> pd.DataFrame:
    return _interactions_by_lead().get(lead_id, table("interactions").iloc[0:0])


def campaign(campaign_id: str) -> dict | None:
    df = table("campaigns")
    row = df[df["campaign_id"] == campaign_id]
    return None if row.empty else row.iloc[0].to_dict()


def search_leads(q: str = "", limit: int = 25) -> list[dict]:
    df = table("leads")
    if q:
        ql = q.lower()
        mask = (
            df["lead_id"].str.lower().str.contains(ql)
            | df["industry"].str.lower().str.contains(ql)
            | df["persona"].str.lower().str.contains(ql)
            | df["region"].str.lower().str.contains(ql)
        )
        df = df[mask]
    cols = ["lead_id", "industry", "persona", "region", "lead_score", "campaign_id", "preferred_channel"]
    return df[cols].head(limit).to_dict(orient="records")
