"""Simulated conversion outcomes.

Why this exists: the seed dataset's own labels (conversions.csv, triage_category) are
generated independently of every feature -- cross-validated ROC-AUC is ~0.50 for any
model (see `seed_label_sanity_check` in the model card). Training on them would produce a
model that looks like it works but has learned nothing.

Instead, outcomes are simulated from a documented, business-plausible process: engagement
quality, fit and recency drive intent, plus irreducible noise. The pipeline, evaluation and
decision policy are identical to what would run on real CRM outcomes -- only the label source
differs, and it is stated openly.
"""
import numpy as np
import pandas as pd

SOURCE_EFFECT = {"Referral": 0.8, "Event": 0.5, "Partner": 0.4, "Website": 0.2,
                 "LinkedIn": 0.1, "Google Ads": 0.0, "Outbound": -0.5}
PERSONA_EFFECT = {"CMO": 0.35, "Marketing Manager": 0.3, "Founder": 0.2,
                  "Sales Lead": 0.0, "Engineer": -0.2, "Procurement": -0.3}
SIZE_EFFECT = {"1-10": -0.4, "11-50": -0.1, "51-200": 0.2, "201-1000": 0.35,
               "1001-5000": 0.25, "5000+": 0.1}


def simulate_outcomes(f: pd.DataFrame, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    z = (
        -1.6
        + 0.040 * (f["lead_score"] - 60)
        + 0.55 * f["n_positive"]
        + 0.70 * f["n_callback"]
        - 0.45 * f["n_negative"]
        - 0.90 * f["n_unsubscribe"]
        + 0.25 * f["n_replies"]
        + 1.20 * (f["rfm_score"] - 0.5)
        - 0.012 * f["days_since_active"]
        + f["source"].map(SOURCE_EFFECT).fillna(0)
        + f["persona"].map(PERSONA_EFFECT).fillna(0)
        + f["company_size"].map(SIZE_EFFECT).fillna(0)
        + rng.normal(0, 0.6, len(f))   # unobserved factors
    )
    p = 1 / (1 + np.exp(-z))
    return pd.Series((rng.random(len(f)) < p).astype(int), index=f.index, name="converted")
