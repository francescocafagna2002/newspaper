"""Step 6 - evaluation against the two audit sets.

Audit set A (the headline): PV-owning known-battery households vs PV-owning
GIGI audit-negative households. The audit negatives never entered training in
any role (not as negatives, not in the unlabelled pool), so they are a genuine
held-out control; the positives are scored out-of-fold via GroupKFold on
gp_nr. The comparison is deliberately PV-owner vs PV-owner -- 96% of battery
households also have PV, so a battery-vs-general-population number would just
be measuring "has PV" (battery_mvp_plan.md D3).

Audit set B: the paired retrofit test on households commissioned inside the
measurement window -- probability in the last pre-install year vs the first
full post-install year, on the same household. Each household is its own
control, so it cancels the household-level confounders audit set A cannot.

Accuracy on audit set A is never reported without its no-skill baseline
(always predict "battery"), and PR-AUC is average_precision_score, never a
trapezoid over a PR curve.

Outputs: artifacts/battery/metrics.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score)

from . import config as C
from .train import _apply_gamma

THRESHOLD = 0.5
N_BOOTSTRAP = 2000


def _score_heldout(df: pd.DataFrame, meta: dict, model) -> np.ndarray:
    """Calibrated battery probability from the final model, for households
    that were never in training (the audit negatives)."""
    X = df[meta["features"]].astype(float)
    X = X.fillna(X.median(numeric_only=True))
    g = model.predict_proba(X)[:, 1]
    pu = np.clip(g / meta["c"], 0, 1)
    return _apply_gamma(pu, meta["gamma"])


def _bootstrap_ci(y: np.ndarray, p: np.ndarray, fn, n: int = N_BOOTSTRAP) -> list[float]:
    rng = np.random.default_rng(C.SEED)
    stats = []
    idx = np.arange(len(y))
    for _ in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[take])) < 2:
            continue
        stats.append(fn(y[take], p[take]))
    if not stats:
        return [float("nan"), float("nan")]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def _balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    pred = (p >= THRESHOLD).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); fn_ = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    sens = tp / (tp + fn_) if (tp + fn_) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return (sens + spec) / 2


def audit_set_a(oof: pd.DataFrame, labels: pd.DataFrame, gp_year: pd.DataFrame,
                meta: dict, model) -> dict:
    hh = labels.groupby("gp_nr", as_index=False).agg(
        battery_positive=("battery_positive", "max"),
        audit_negative=("audit_negative", "max"),
        has_pv_row=("has_pv_row", "max"),
        has_meter=("has_meter", "max"),
    )
    pos_ids = set(hh.loc[hh.battery_positive & hh.has_pv_row & hh.has_meter, "gp_nr"])
    neg_ids = set(hh.loc[hh.audit_negative & hh.has_pv_row & hh.has_meter, "gp_nr"])

    # positives: their most recent P-year out-of-fold score
    pos = oof[oof.gp_nr.isin(pos_ids) & (oof.pu_label == "P")]
    pos = pos.sort_values("year").groupby("gp_nr").tail(1)

    # negatives: never trained on, so score their most recent usable year with
    # the final model
    neg_rows = gp_year[gp_year.gp_nr.isin(neg_ids)
                       & (gp_year["n_months"] >= C.MIN_MONTHS_PER_YEAR)]
    neg_rows = neg_rows.sort_values("year").groupby("gp_nr").tail(1)
    neg_scores = _score_heldout(neg_rows, meta, model) if len(neg_rows) else np.array([])

    y = np.r_[np.ones(len(pos)), np.zeros(len(neg_rows))].astype(int)
    p = np.r_[pos["battery_probability"].to_numpy(), neg_scores]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"error": "audit set A unavailable", "n_positives": len(pos),
                "n_negatives": len(neg_rows)}

    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn_) if (tp + fn_) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / len(y)
    no_skill = max(y.mean(), 1 - y.mean())   # always predict the majority class
    return {
        "n": int(len(y)), "n_positives": int(y.sum()), "n_negatives": int((y == 0).sum()),
        "threshold": THRESHOLD,
        "confusion_matrix": {"tp": int(tp), "fn": int(fn_), "fp": int(fp), "tn": int(tn)},
        "sensitivity": float(sens), "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
        "accuracy": float(acc),
        "accuracy_no_skill_baseline": float(no_skill),
        "accuracy_vs_baseline": f"{acc:.1%} vs {no_skill:.1%} no-skill baseline",
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_average_precision": float(average_precision_score(y, p)),
        "ci95": {
            "balanced_accuracy": _bootstrap_ci(y, p, _balanced_accuracy),
            "roc_auc": _bootstrap_ci(y, p, lambda a, b: float(roc_auc_score(a, b))),
            "pr_auc_average_precision": _bootstrap_ci(
                y, p, lambda a, b: float(average_precision_score(a, b))),
        },
    }


def audit_set_b(oof: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Paired retrofit test: last pre-install year vs first full post-install
    year, same household."""
    hh = labels.groupby("gp_nr", as_index=False).agg(
        battery_positive=("battery_positive", "max"),
        battery_commissioning_date=("battery_commissioning_date", "first"),
    )
    hh["comm_year"] = pd.to_datetime(hh["battery_commissioning_date"],
                                     errors="coerce").dt.year
    in_window = hh["battery_commissioning_date"].between("2023-04-01", "2026-03-01")
    hh = hh[hh.battery_positive & in_window & hh.comm_year.notna()]

    merged = oof.merge(hh[["gp_nr", "comm_year"]], on="gp_nr", how="inner")
    pre = merged[merged.year < merged.comm_year].sort_values("year").groupby("gp_nr").tail(1)
    post = merged[merged.year > merged.comm_year].sort_values("year").groupby("gp_nr").head(1)
    pair = pre[["gp_nr", "battery_probability"]].merge(
        post[["gp_nr", "battery_probability"]], on="gp_nr", suffixes=("_pre", "_post"))
    if len(pair) < 5:
        return {"n_pairs": int(len(pair)), "note": "too few paired households"}

    delta = pair["battery_probability_post"] - pair["battery_probability_pre"]
    try:
        stat, pval = wilcoxon(pair["battery_probability_post"],
                              pair["battery_probability_pre"])
        pval = float(pval)
    except ValueError:
        pval = float("nan")
    return {
        "n_pairs": int(len(pair)),
        "win_rate": float((delta > 0).mean()),
        "median_jump": float(delta.median()),
        "mean_pre": float(pair["battery_probability_pre"].mean()),
        "mean_post": float(pair["battery_probability_post"].mean()),
        "wilcoxon_p": pval,
    }


def pu_metrics(oof: pd.DataFrame, meta: dict) -> dict:
    s = (oof["pu_label"] == "P").to_numpy()
    p = oof["battery_probability"].to_numpy()
    return {
        "n_household_years": int(len(oof)),
        "n_P_years": int(s.sum()),
        "n_U_years": int((~s).sum()),
        "elkan_noto_c": meta["c"],
        "recall@0.5_on_P": float((p[s] >= THRESHOLD).mean()),
        "population_flag_rate@0.5_on_U": float((p[~s] >= THRESHOLD).mean()),
        "roc_auc_P_vs_U": float(roc_auc_score(s.astype(int), p)),
        "pr_auc_P_vs_U": float(average_precision_score(s.astype(int), p)),
    }


def by_year(oof: pd.DataFrame) -> dict:
    out = {}
    for year, g in oof.groupby("year"):
        s = (g["pu_label"] == "P").to_numpy()
        if s.sum() < 5 or (~s).sum() < 5:
            continue
        out[str(int(year))] = {
            "n": int(len(g)), "n_P": int(s.sum()),
            "recall@0.5": float((g.loc[s, "battery_probability"] >= THRESHOLD).mean()),
            "flag_rate@0.5": float((g.loc[~s, "battery_probability"] >= THRESHOLD).mean()),
        }
    return out


def main() -> dict:
    meta = json.loads((C.MODEL_DIR / "meta.json").read_text())
    model = pd.read_pickle(C.MODEL_DIR / "model.pkl")
    oof = pd.read_csv(C.MODEL_DIR / "oof_predictions.csv", dtype={"gp_nr": str})
    labels = pd.read_csv(C.LABELS, dtype={"gp_nr": str})
    gp_year = pd.read_pickle(C.FEATURES_GP_YEAR)

    metrics = {
        "audit_set_a": audit_set_a(oof, labels, gp_year, meta, model),
        "audit_set_b_paired_retrofit": audit_set_b(oof, labels),
        "pu_in_loop": pu_metrics(oof, meta),
        "by_year": by_year(oof),
        "pi_sweep": meta.get("pi_results", {}),
        "baseline_single_rule": meta.get("baseline_single_rule", {}),
    }
    C.METRICS.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nwrote {C.METRICS}")
    return metrics


if __name__ == "__main__":
    main()
