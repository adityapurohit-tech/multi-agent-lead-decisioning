"""Inference wrapper: calibrated probability + human-readable reason codes."""
import json
from functools import lru_cache

import joblib
import numpy as np

from app.config import MODEL_CARD_PATH, MODEL_PATH
from app.ml.features import CATEGORICAL, NUMERIC, features_for

READABLE = {
    "lead_score": "lead score", "rfm_score": "RFM score", "n_touches": "touch count",
    "n_positive": "positive responses", "n_callback": "callback requests",
    "n_negative": "negative responses", "n_unsubscribe": "unsubscribe events",
    "n_replies": "replies", "n_clicks": "clicks / site visits",
    "days_since_active": "days since last active",
}


@lru_cache(maxsize=1)
def load():
    if not MODEL_PATH.exists():
        raise RuntimeError("No trained model found. Run: python -m app.ml.train")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def model_card() -> dict:
    return json.loads(MODEL_CARD_PATH.read_text()) if MODEL_CARD_PATH.exists() else {}


def _reason_codes(model, x, top_k=3) -> list[dict]:
    """Per-feature contribution to the log-odds (coefficient x standardized value).
    One-hot columns are folded back into their source feature."""
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []
    pre = model.named_steps["pre"]
    z = pre.transform(x)
    z = z.toarray()[0] if hasattr(z, "toarray") else np.asarray(z)[0]
    contrib = clf.coef_[0] * z
    names = pre.get_feature_names_out()
    raw = x.iloc[0]
    totals: dict[str, float] = {}
    for name, c in zip(names, contrib):
        kind, col = name.split("__", 1)
        if kind == "cat":
            feat = next(f for f in CATEGORICAL if col.startswith(f + "_"))
            if col != f"{feat}_{raw[feat]}":
                continue   # only the active category explains this lead
        else:
            feat = col
        totals[feat] = totals.get(feat, 0.0) + float(c)
    ranked = sorted(totals.items(), key=lambda kv: -abs(kv[1]))[:top_k]
    means = dict(zip(NUMERIC, pre.named_transformers_["num"].mean_))
    out = []
    for feat, c in ranked:
        if feat in NUMERIC:
            label = f"{READABLE.get(feat, feat)} {raw[feat]:g} (avg {means[feat]:.1f})"
        else:
            label = f"{feat.replace('_', ' ')}: {raw[feat]}"
        out.append({"factor": label, "impact": round(c, 3), "direction": "up" if c > 0 else "down"})
    return out


def score(lead_id: str) -> dict:
    model = load()
    x = features_for(lead_id)
    p = float(model.predict_proba(x)[0, 1])
    return {"p_convert": round(p, 4), "reasons": _reason_codes(model, x),
            "features": {k: (v.item() if hasattr(v, "item") else v) for k, v in x.iloc[0].to_dict().items()}}
