"""Campaign Optimizer agent.

1. Classifies campaign health (Scale / Maintain / Optimize / Pause) by clustering campaign KPIs
   (ROAS, CPL, CTR, week-over-week trend) with scikit-learn KMeans.
2. Picks the creative variant to use. A variant is only declared the winner if a chi-square
   test says so (p < 0.05); otherwise it falls back to Thompson sampling, which keeps exploring
   instead of over-trusting a noisy leader.
3. If the lead's campaign should be paused, reassigns the lead to the best healthy campaign.

Analyses are cached in Redis because every lead in a campaign would otherwise recompute them.
"""
from functools import lru_cache

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app import data
from app.services import services

POSITIVE = {"positive", "callback_requested"}

CLUSTER_FEATURES = ["roas", "cpl_usd", "ctr", "trend"]
# Direction each KPI contributes to "health" when ranking cluster centroids.
HEALTH_WEIGHTS = {"roas": 1.0, "cpl_usd": -0.5, "ctr": 0.3, "trend": 0.5}
HEALTH_ORDER = ["Scale", "Maintain", "Optimize", "Pause"]      # best centroid -> worst centroid
LABELS = {"roas": "ROAS", "cpl_usd": "CPL", "ctr": "CTR", "trend": "WoW trend"}


def _kpis(sub) -> dict:
    cost, rev = sub["cost_usd"].sum(), sub["revenue_usd"].sum()
    leads, conv = sub["leads_created"].sum(), sub["conversions"].sum()
    clicks, impr = sub["clicks"].sum(), sub["impressions"].sum()
    return {"spend_usd": round(float(cost), 2), "revenue_usd": round(float(rev), 2),
            "roas": round(float(rev / cost), 2) if cost else 0.0,
            "cpl_usd": round(float(cost / leads), 2) if leads else None,
            "ctr": round(float(clicks / impr), 4) if impr else 0.0,
            "lead_to_conversion": round(float(conv / leads), 3) if leads else 0.0,
            "days": int(len(sub))}


def _trend(sub) -> float:
    """Week-over-week ROAS change: pooled ROAS of the last 7 days vs the 7 days before.
    Pooling revenue and cost (rather than averaging daily ROAS) keeps low-spend days from dominating."""
    s = sub.sort_values("date")
    if len(s) < 14:
        return 0.0
    last, prev = s.tail(7), s.iloc[-14:-7]
    r_last = last["revenue_usd"].sum() / max(last["cost_usd"].sum(), 1e-9)
    r_prev = prev["revenue_usd"].sum() / max(prev["cost_usd"].sum(), 1e-9)
    return round(float(r_last / r_prev - 1), 3) if r_prev else 0.0


@lru_cache(maxsize=1)
def portfolio() -> dict:
    """Campaign classification with scikit-learn.

    Each campaign becomes a KPI vector (ROAS, CPL, CTR, week-over-week trend), standardised and
    clustered with KMeans (k=4). Clusters are unlabeled, so their centroids are ranked by a
    weighted health score and named Scale > Maintain > Optimize > Pause. The explanation for a
    campaign is the pair of KPIs that deviate most from the portfolio average.
    """
    daily = data.table("campaign_daily")
    rows = {cid: {**_kpis(g), "trend": _trend(g)} for cid, g in daily.groupby("campaign_id")}
    ids = list(rows)
    X = np.array([[rows[c]["roas"], rows[c]["cpl_usd"] or 0, rows[c]["ctr"],
                   float(np.clip(rows[c]["trend"], -1, 1))] for c in ids])
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    km = KMeans(n_clusters=4, n_init=20, random_state=0).fit(Z)
    w = np.array([HEALTH_WEIGHTS[f] for f in CLUSTER_FEATURES])
    names = {int(c): HEALTH_ORDER[rank] for rank, c in enumerate(np.argsort(-(km.cluster_centers_ @ w)))}
    for cid, z, label in zip(ids, Z, km.labels_):
        r = rows[cid]
        r["health"], r["cluster"] = names[int(label)], int(label)
        top = np.argsort(-np.abs(z))[:2]
        drivers = ", ".join(f"{LABELS[CLUSTER_FEATURES[i]]} {'above' if z[i] > 0 else 'below'} avg ({z[i]:+.1f} sd)"
                            for i in top)
        r["health_reason"] = f"KMeans cluster '{r['health']}': {drivers}"
    centroids = scaler.inverse_transform(km.cluster_centers_)
    return {"campaigns": rows,
            "cluster_sizes": {names[c]: int((km.labels_ == c).sum()) for c in range(4)},
            "centroids": {names[c]: dict(zip(CLUSTER_FEATURES, np.round(centroids[c], 3).tolist())) for c in range(4)}}


@lru_cache(maxsize=1)
def _variant_stats():
    """Portfolio-wide response rate per creative variant (variants are reused across campaigns)."""
    inter = data.table("interactions")
    inter = inter.assign(pos=inter["outcome"].isin(POSITIVE).astype(int))
    g = inter.groupby("variant_id")["pos"].agg(["sum", "count"])
    return {vid: (int(r["sum"]), int(r["count"])) for vid, r in g.iterrows()}


def choose_variant(campaign_id: str, lead_id: str) -> dict:
    variants = data.table("ab_variants")
    own = variants[variants["campaign_id"] == campaign_id]
    stats = _variant_stats()
    rows = []
    for v in own.to_dict("records"):
        pos, n = stats.get(v["variant_id"], (0, 0))
        rows.append({**v, "positives": pos, "n": n, "rate": round(pos / n, 4) if n else 0.0})
    rows.sort(key=lambda r: -r["rate"])
    if not rows:
        return {"variant": None, "method": "no variants for campaign"}

    p_value = None
    if len(rows) >= 2 and rows[1]["n"]:
        a, b = rows[0], rows[1]
        table = [[a["positives"], a["n"] - a["positives"]], [b["positives"], b["n"] - b["positives"]]]
        p_value = round(float(chi2_contingency(table)[1]), 4)

    if p_value is not None and p_value < 0.05:
        chosen, method = rows[0], f"significant winner (chi-square p={p_value})"
    else:
        # Deterministic per lead so a re-run reproduces the same choice.
        rng = np.random.default_rng(int(lead_id[1:]))
        draws = [rng.beta(1 + r["positives"], 1 + r["n"] - r["positives"]) for r in rows]
        chosen = rows[int(np.argmax(draws))]
        method = (f"no significant winner (chi-square p={p_value}) -> Thompson sampling"
                  if p_value is not None else "single variant")
    keep = ("variant_id", "channel", "creative_type", "subject_line", "call_to_action", "tone", "rate", "n")
    return {"variant": {k: chosen[k] for k in keep}, "method": method, "p_value": p_value,
            "candidates": [{k: r[k] for k in ("variant_id", "channel", "rate", "n")} for r in rows]}


def _analysis(campaign_id: str) -> dict:
    cached = services.store.cache_get(f"campaign:{campaign_id}")
    if cached:
        return {**cached, "cache": "hit"}
    k = portfolio()["campaigns"].get(campaign_id)
    meta = data.campaign(campaign_id) or {}
    result = {"campaign_id": campaign_id, "name": meta.get("name"), "objective": meta.get("objective"),
              "kpis": {x: k[x] for x in ("roas", "cpl_usd", "ctr", "lead_to_conversion", "spend_usd", "days")},
              "trend": k["trend"], "health": k["health"], "health_reason": k["health_reason"]}
    services.store.cache_set(f"campaign:{campaign_id}", result)
    return {**result, "cache": "miss"}


def _best_alternative(exclude: str, objective: str | None) -> str:
    camps = portfolio()["campaigns"]
    meta = data.table("campaigns").set_index("campaign_id")["objective"].to_dict()
    healthy = [(cid, k) for cid, k in camps.items() if cid != exclude and k["health"] in ("Scale", "Maintain")]
    same = [x for x in healthy if meta.get(x[0]) == objective] or healthy
    return max(same, key=lambda x: x[1]["roas"])[0]


def campaign_optimizer(state):
    lead = state["lead"]
    primary = _analysis(lead["campaign_id"])
    active, reassigned = primary, None
    if primary["health"] == "Pause":
        alt = _best_alternative(primary["campaign_id"], primary["objective"])
        active, reassigned = _analysis(alt), alt
    creative = choose_variant(active["campaign_id"], lead["lead_id"])
    campaign = {"primary": primary, "active_campaign_id": active["campaign_id"],
                "reassigned_to": reassigned, "active": active, "creative": creative}
    v = creative["variant"] or {}
    summary = (f"{primary['campaign_id']} health={primary['health']} ({primary['health_reason']}); "
               + (f"lead reassigned to {reassigned} ({active['health']}); " if reassigned else "")
               + f"creative {v.get('variant_id')} via {creative['method']}; cache {primary['cache']}")
    return {"campaign": campaign, "_summary": summary}
