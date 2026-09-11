"""Step 5 - Positive-Unlabeled model (Elkan-Noto) for behind-the-meter battery
storage, mirroring models/pv/modeling/train.py.

P = years of a known battery household AFTER its commissioning date (or the
most recent complete year, if undated). U = every other in-universe household-
year, including the SAME household's own pre-install years. The 82 GIGI
audit-negative households never enter P or U -- see _select_rows.

pi (the assumed population base rate) is swept over {0.20, 0.30, 0.40}
(config.PI_VALUES); one calibrated score per pi is written to
oof_predictions.csv and meta.json, with 0.30 (config.DEFAULT_PI) as the
column used for the p>=0.5 operating point elsewhere in the pipeline.

Outputs: artifacts/battery/model/{model.pkl, meta.json, oof_predictions.csv}
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from . import config as C

DROP = {"gp_nr", "mp_id", "year", "plz", "kanton", "battery_positive",
        "audit_negative", "battery_commissioning_date", "in_universe",
        "pu_label", "cp_year"}
N_SPLITS = 5


def _new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=4, max_iter=300, learning_rate=0.05,
        l2_regularization=1.0, class_weight="balanced",
        random_state=C.SEED,
    )


def _feature_matrix(df: pd.DataFrame, only: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in df.columns
            if c not in DROP and pd.api.types.is_numeric_dtype(df[c])]
    if only:
        cols = [c for c in cols if c in set(only)]
    return df[cols].astype(float), cols


def select_rows(gp_year: pd.DataFrame, labels: pd.DataFrame,
                universe: pd.DataFrame) -> pd.DataFrame:
    """Household-years usable for PU training, with the P/U/dropped rule from
    battery_implementation_spec.md S4.5. Label-blind up to this point (the
    universe/quality filters never consult a label); the P/U assignment
    below is exactly where labels legitimately enter -- training itself."""
    hh = labels.groupby("gp_nr", as_index=False).agg(
        battery_positive=("battery_positive", "max"),
        audit_negative=("audit_negative", "max"),
        battery_commissioning_date=("battery_commissioning_date", "first"),
    )
    df = gp_year.merge(hh, on="gp_nr", how="left")
    df = df.merge(universe[["gp_nr", "in_universe"]], on="gp_nr", how="left")
    df["battery_positive"] = df["battery_positive"].fillna(False)
    df["audit_negative"] = df["audit_negative"].fillna(False)
    df["in_universe"] = df["in_universe"].fillna(False)

    df = df[(df["n_months"] >= C.MIN_MONTHS_PER_YEAR) & df["in_universe"]].copy()
    df = df[~df["audit_negative"]].copy()   # held out entirely, never P or U

    comm_year = pd.to_datetime(df["battery_commissioning_date"], errors="coerce").dt.year
    label = pd.Series("U", index=df.index, dtype=object)

    dated = df["battery_positive"] & comm_year.notna()
    label[dated & (df["year"] > comm_year)] = "P"
    label[dated & (df["year"] < comm_year)] = "U"
    label[dated & (df["year"] == comm_year)] = "DROP"

    undated = df["battery_positive"] & comm_year.isna()
    if undated.any():
        recent = df.loc[undated].groupby("gp_nr")["year"].transform("max")
        label.loc[undated & (df["year"] == recent)] = "P"
        label.loc[undated & (df["year"] < recent)] = "U"

    df["pu_label"] = label
    return df[df["pu_label"] != "DROP"].reset_index(drop=True)


def _match_base_rate(p_u: np.ndarray, target: float) -> float:
    from scipy.optimize import brentq
    f = lambda g: _apply_gamma(p_u, g).mean() - target
    try:
        return float(brentq(f, -10.0, 10.0))
    except ValueError:
        return 0.0


def _apply_gamma(p: np.ndarray, gamma: float) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) + gamma
    return 1.0 / (1.0 + np.exp(-z))


def _single_rule_baseline(df: pd.DataFrame, s: np.ndarray, groups: np.ndarray) -> dict:
    """evening_zero_share_summer > tau, tau chosen on each training fold by
    Youden's J, reported out-of-fold -- the single-feature baseline the PV
    deck also reports next to the full model."""
    if "evening_zero_share_summer" not in df:
        return {}
    x = df["evening_zero_share_summer"].fillna(df["evening_zero_share_summer"].median()).to_numpy()
    pred_oof = np.zeros(len(df))
    gkf = GroupKFold(N_SPLITS)
    for tr, te in gkf.split(x, s, groups):
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(s[tr], x[tr])
        tau = thr[np.argmax(tpr - fpr)]
        pred_oof[te] = (x[te] >= tau).astype(float)
    return {
        "recall": float(pred_oof[s == 1].mean()) if (s == 1).any() else None,
        "flag_rate": float(pred_oof[s == 0].mean()) if (s == 0).any() else None,
        "roc_auc": float(roc_auc_score(s, x)),
    }


def train(only: list[str] | None = None, out_dir=None) -> dict:
    out_dir = out_dir or C.MODEL_DIR
    gp_year = pd.read_pickle(C.FEATURES_GP_YEAR)
    labels = pd.read_csv(C.LABELS, dtype={"gp_nr": str})
    universe = pd.read_csv(C.UNIVERSE, dtype={"gp_nr": str})

    df = select_rows(gp_year, labels, universe)
    s = (df["pu_label"] == "P").astype(int).to_numpy()
    groups = df["gp_nr"].to_numpy()
    X, feat_cols = _feature_matrix(df, only)
    X = X.fillna(X.median(numeric_only=True))
    print(f"features ({len(feat_cols)}): {feat_cols}")
    print(f"household-years: {len(df)}   households: {df['gp_nr'].nunique()}   "
          f"P: {s.sum()}   U: {(s == 0).sum()}")

    g_oof = np.zeros(len(df))
    gkf = GroupKFold(N_SPLITS)
    for k, (tr, te) in enumerate(gkf.split(X, s, groups), 1):
        mdl = _new_model().fit(X.iloc[tr], s[tr])
        g_oof[te] = mdl.predict_proba(X.iloc[te])[:, 1]
        print(f"  fold {k}: AUC(s vs U) = {roc_auc_score(s[te], g_oof[te]):.3f}")

    baseline = _single_rule_baseline(df, s, groups)
    if baseline:
        print(f"  baseline [evening_zero_share_summer > tau]: {baseline}")

    c = float(g_oof[s == 1].mean())
    p_oof = np.clip(g_oof / c, 0, 1)
    print(f"\nElkan-Noto c (label frequency): {c:.3f}")

    pi_results: dict[str, dict] = {}
    gammas: dict[str, float] = {}
    for pi in C.PI_VALUES:
        gamma = _match_base_rate(p_oof[s == 0], pi)
        p_cal = _apply_gamma(p_oof, gamma)
        gammas[f"{pi:.2f}"] = gamma
        pi_results[f"{pi:.2f}"] = dict(
            gamma=gamma,
            calibrated_u_mean=float(p_cal[s == 0].mean()),
            recall_at_0_5=float((p_cal[s == 1] >= 0.5).mean()),
            flag_rate_at_0_5=float((p_cal[s == 0] >= 0.5).mean()),
            roc_auc=float(roc_auc_score(s, p_cal)),
            pr_auc=float(average_precision_score(s, p_cal)),
        )
        print(f"  pi={pi:.2f}: gamma={gamma:+.3f}  recall@0.5={pi_results[f'{pi:.2f}']['recall_at_0_5']:.3f}  "
              f"flag_rate@0.5={pi_results[f'{pi:.2f}']['flag_rate_at_0_5']:.3f}  "
              f"ROC-AUC={pi_results[f'{pi:.2f}']['roc_auc']:.3f}")

    default_gamma = gammas[f"{C.DEFAULT_PI:.2f}"]
    p_default = _apply_gamma(p_oof, default_gamma)

    final = _new_model().fit(X, s)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(final, out_dir / "model.pkl")

    oof = df[["gp_nr", "mp_id", "year", "plz", "kanton", "pu_label", "n_meters",
              "n_months"]].copy()
    oof["g"] = g_oof
    oof["pu_score"] = p_oof
    oof["battery_probability"] = p_default
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)

    meta = dict(
        features=feat_cols, c=c, gamma=default_gamma, default_pi=C.DEFAULT_PI,
        pi_values=list(C.PI_VALUES), pi_results=pi_results, gammas=gammas,
        n_splits=N_SPLITS, n_positives=int(s.sum()), n_unlabelled=int((s == 0).sum()),
        n_households=int(df["gp_nr"].nunique()), baseline_single_rule=baseline,
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out_dir}/(model.pkl, meta.json, oof_predictions.csv)")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", metavar="FEATURE")
    a = ap.parse_args()
    if a.only:
        tag = "_".join(a.only)[:40]
        train(only=a.only, out_dir=C.ARTIFACTS / f"model_{tag}")
    else:
        train()
