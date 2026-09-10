"""Step 7 - Positive-Unlabeled model (Elkan-Noto) for household PV.

There are essentially no confirmed PV-negatives (see docs/data_problems.md), so
this is a PU problem: P = known positives (GIGI PV/battery + silver feed-in),
U = every other household (a contaminated negative, contamination ~= the PV
base rate).

Method (Elkan & Noto, 2008, "case-control" variant):
  1. train g(x) = P(labelled | x) on P vs U with cross-validation
  2. c = E[g(x) | x is a known positive]   (label frequency)
  3. calibrated score  p(x) = g(x) / c , clipped to [0, 1]
  4. shift p so its population mean matches the ElPA-derived base rate

Outputs under artifacts/model/: model.pkl, meta.json, oof_predictions.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from . import config as C

DROP = {"gp_nr", "mp_id", "plz", "pv_label", "label_source",
        "pv_commissioning_date", "kanton", "first_year", "last_year"}
N_SPLITS = 5
TARGET_BASE_RATE = 0.12   # canton-AG rooftop PV share (ElPA lower bound ~11%)


def _feature_matrix(gp: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in gp.columns
            if c not in DROP and pd.api.types.is_numeric_dtype(gp[c])]
    return gp[cols].astype(float), cols


def _new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=4, max_iter=300, learning_rate=0.05,
        l2_regularization=1.0, class_weight="balanced",
        random_state=C.SEED,
    )


def train() -> dict:
    gp = pd.read_pickle(C.FEATURES_GP)
    gp = gp[gp["n_months"] >= 3].reset_index(drop=True)   # need some history
    s = (gp["pv_label"] == 1).astype(int).to_numpy()      # labelled indicator
    X, feat_cols = _feature_matrix(gp)

    print(f"households: {len(gp)}   known positives P: {s.sum()}   unlabelled U: {(s==0).sum()}")

    # ---- cross-validated g(x) ----
    g_oof = np.zeros(len(gp))
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=C.SEED)
    for k, (tr, te) in enumerate(skf.split(X, s), 1):
        mdl = _new_model().fit(X.iloc[tr], s[tr])
        g_oof[te] = mdl.predict_proba(X.iloc[te])[:, 1]
        print(f"  fold {k}: AUC(s vs U) = {roc_auc_score(s[te], g_oof[te]):.3f}")

    # ---- one-feature rule baseline (for context) ----
    for feat in ("daytime_zero_summer", "midday_clear_summer"):
        col = X[feat].fillna(X[feat].median()).to_numpy()
        sign = 1 if roc_auc_score(s, col) >= 0.5 else -1
        print(f"  baseline AUC [{feat}]: {roc_auc_score(s, sign * col):.3f}")

    # ---- Elkan-Noto label frequency ----
    c = float(g_oof[s == 1].mean())
    p_oof = np.clip(g_oof / c, 0, 1)

    # ---- shift to the population base rate ----
    u_mean = float(p_oof[s == 0].mean())
    gamma = _match_base_rate(p_oof[s == 0], TARGET_BASE_RATE)
    p_oof_cal = _apply_gamma(p_oof, gamma)

    print(f"\nElkan-Noto c (label frequency)     : {c:.3f}")
    print(f"raw PU score mean over U            : {u_mean:.3f}")
    print(f"calibrated mean over U (target {TARGET_BASE_RATE:.0%}) : "
          f"{p_oof_cal[s==0].mean():.3f}   (gamma={gamma:.3f})")
    print(f"recall @ p>=0.5 on known positives  : {(p_oof_cal[s==1] >= 0.5).mean():.3f}")
    print(f"PU ROC-AUC (P vs U, proxy)          : {roc_auc_score(s, p_oof_cal):.3f}")
    print(f"PU PR-AUC  (P vs U, proxy)          : {average_precision_score(s, p_oof_cal):.3f}")

    # ---- final model on everything ----
    final = _new_model().fit(X, s)

    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(final, C.MODEL_DIR / "model.pkl")
    oof = gp[["gp_nr", "pv_label", "label_source", "plz", "kanton",
              "pv_commissioning_date", "n_meters", "n_months"]].copy()
    oof["g"] = g_oof
    oof["pu_score"] = p_oof
    oof["pv_probability"] = p_oof_cal
    oof.to_csv(C.MODEL_DIR / "oof_predictions.csv", index=False)

    meta = dict(features=feat_cols, c=c, gamma=gamma,
                target_base_rate=TARGET_BASE_RATE, n_splits=N_SPLITS,
                n_positives=int(s.sum()), n_unlabelled=int((s == 0).sum()))
    (C.MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {C.MODEL_DIR}/(model.pkl, meta.json, oof_predictions.csv)")
    return meta


def _match_base_rate(p_u: np.ndarray, target: float) -> float:
    """Find gamma so that mean(sigmoid(logit(p)+gamma)) == target over U."""
    from scipy.optimize import brentq

    lo, hi = -10.0, 10.0
    f = lambda g: _apply_gamma(p_u, g).mean() - target
    try:
        return float(brentq(f, lo, hi))
    except ValueError:
        return 0.0


def _apply_gamma(p: np.ndarray, gamma: float) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) + gamma
    return 1.0 / (1.0 + np.exp(-z))


if __name__ == "__main__":
    train()
