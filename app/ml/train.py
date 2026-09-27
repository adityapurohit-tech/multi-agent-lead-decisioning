"""Train the lead-scoring model and write an honest model card.

    python -m app.ml.train
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app import data
from app.config import COLD_THRESHOLD, HOT_THRESHOLD, MODEL_CARD_PATH, MODEL_PATH
from app.ml.features import CATEGORICAL, FEATURES, NUMERIC, feature_table
from app.ml.labels import simulate_outcomes


def make_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", estimator)])


def seed_label_sanity_check(X: pd.DataFrame) -> float:
    """CV ROC-AUC against the seed dataset's own conversion labels."""
    seed_y = X.index.isin(data.table("conversions")["lead_id"]).astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(make_pipeline(LogisticRegression(max_iter=2000)), X, seed_y,
                                 cv=cv, scoring="roc_auc").mean())


def calibration_table(y, p, bins=10) -> list[dict]:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = df.groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return [{"predicted": round(r.predicted, 3), "observed": round(r.observed, 3), "n": int(r.n)}
            for r in out.itertuples()]


def main():
    X = feature_table()
    y = simulate_outcomes(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    candidates = {
        "logistic_regression": make_pipeline(LogisticRegression(max_iter=2000, C=1.0)),
        "hist_gradient_boosting": make_pipeline(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300)),
    }
    comparison = {}
    for name, pipe in candidates.items():
        comparison[name] = float(cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean())
        print(f"  CV ROC-AUC {name:24s} {comparison[name]:.3f}")

    # Logistic regression is chosen unless boosting is clearly better: per-lead reason codes
    # from coefficients make every automated decision explainable and auditable.
    chosen = "logistic_regression"
    if comparison["hist_gradient_boosting"] - comparison["logistic_regression"] > 0.02:
        chosen = "hist_gradient_boosting"
    model = candidates[chosen].fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]

    baseline_auc = roc_auc_score(y_te, X_te["lead_score"])
    tiers = np.where(p >= HOT_THRESHOLD, "Hot", np.where(p < COLD_THRESHOLD, "Cold", "Warm"))
    tier_stats = {t: {"share": float((tiers == t).mean()), "observed_conversion": float(y_te[tiers == t].mean())}
                  for t in ["Hot", "Warm", "Cold"] if (tiers == t).any()}
    tn, fp, fn, tp = confusion_matrix(y_te, (p >= HOT_THRESHOLD).astype(int)).ravel()

    imp = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=0)
    importance = sorted(({"feature": f, "importance": round(float(v), 4)}
                         for f, v in zip(FEATURES, imp.importances_mean)), key=lambda d: -d["importance"])

    card = {
        "model": chosen,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": FEATURES,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "base_rate": float(y.mean()),
        "label_source": "simulated (see app/ml/labels.py) -- seed labels carry no signal",
        "seed_label_sanity_check_auc": round(seed_label_sanity_check(X), 3),
        "cv_auc_comparison": {k: round(v, 3) for k, v in comparison.items()},
        "test_metrics": {
            "roc_auc": round(float(roc_auc_score(y_te, p)), 3),
            "pr_auc": round(float(average_precision_score(y_te, p)), 3),
            "brier": round(float(brier_score_loss(y_te, p)), 4),
            "baseline_lead_score_roc_auc": round(float(baseline_auc), 3),
            "hot_threshold": HOT_THRESHOLD,
            "hot_precision": round(float(tp / max(tp + fp, 1)), 3),
            "hot_recall": round(float(tp / max(tp + fn, 1)), 3),
            "confusion_at_hot_threshold": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "tier_stats": tier_stats,
        "calibration": calibration_table(y_te.values, p),
        "permutation_importance": importance,
    }
    joblib.dump(model, MODEL_PATH)
    MODEL_CARD_PATH.write_text(json.dumps(card, indent=2))
    tm = card["test_metrics"]
    print(f"Chosen: {chosen} | test ROC-AUC {tm['roc_auc']} | PR-AUC {tm['pr_auc']} | Brier {tm['brier']}")
    print(f"Baseline (lead_score only) ROC-AUC {tm['baseline_lead_score_roc_auc']}")
    print(f"Seed-label sanity check ROC-AUC {card['seed_label_sanity_check_auc']} (0.5 = no signal)")
    print(f"Tiers: {json.dumps(tier_stats)}")
    print(f"Saved {MODEL_PATH.name} and {MODEL_CARD_PATH.name}")


if __name__ == "__main__":
    main()
