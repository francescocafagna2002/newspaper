"""Step 9 - PU-adjusted evaluation, base-rate check, evidence.

Without real negatives we cannot compute ordinary precision. Using the known
recall on positives and the ElPA base rate pi we reconstruct:

    precision(t) ~= recall_P(t) * pi / frac_flagged(t)

and integrate a PU-PR curve. We also report recall on GIGI vs silver positives
separately, the population flag-rate vs the per-PLZ base rate, and the
change-point feature against the PV commissioning date.

Outputs: artifacts/metrics.json  and  artifacts/figures/*.png
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from .. import config as C

FIG = C.ARTIFACTS / "figures"
PI = 0.12  # canton-AG rooftop-PV base rate (ElPA lower bound ~0.11)


def _pu_pr_curve(score, is_pos, pi):
    ts = np.quantile(score, np.linspace(0, 1, 201))
    rec, prec, flag = [], [], []
    for t in ts:
        r = float((score[is_pos] >= t).mean())
        f = float((score >= t).mean())
        rec.append(r)
        flag.append(f)
        prec.append(min(1.0, r * pi / f) if f > 0 else 1.0)
    rec, prec = np.array(rec), np.array(prec)
    order = np.argsort(rec)
    pr_auc = float(np.trapezoid(prec[order], rec[order]))
    # PU ROC: FPR ~= (flag - recall*pi) / (1 - pi)
    flag = np.array(flag)
    fpr = np.clip((flag - rec * pi) / (1 - pi), 0, 1)
    o2 = np.argsort(fpr)
    roc_auc = float(np.trapezoid(rec[o2], fpr[o2]))
    return dict(ts=ts, rec=rec, prec=prec, fpr=fpr, pr_auc=pr_auc, roc_auc=roc_auc)


def main(model_dir=None, metrics_path=None) -> dict:
    model_dir = model_dir or C.MODEL_DIR
    metrics_path = metrics_path or (C.ARTIFACTS / "metrics.json")
    FIG.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(model_dir / "oof_predictions.csv", dtype={"gp_nr": str, "plz": str})
    gp = pd.read_pickle(C.FEATURES_GP)[["gp_nr", "midday_max_yoy_drop", "midday_clear_summer",
                                       "daytime_zero_summer", "midday_seasonality"]]
    oof = oof.merge(gp, on="gp_nr", how="left")

    is_pos = (oof["pv_label"] == 1).to_numpy()
    score = oof["pv_probability"].to_numpy()
    gigi = oof["label_source"].isin(["gigi_pv", "gigi_battery"]).to_numpy()
    silver = (oof["label_source"] == "silver_feedin").to_numpy()

    # trustworthy evaluation: recall axis = GIGI positives only (silver labels
    # are derived from feed-in and would make P-vs-U circular)
    cur = _pu_pr_curve(score, gigi, PI)
    cur_all = _pu_pr_curve(score, is_pos, PI)
    metrics = {
        "n_households": int(len(oof)),
        "n_positives": int(is_pos.sum()),
        "n_gigi_positives": int(gigi.sum()),
        "pu_pr_auc_gigi": round(cur["pr_auc"], 3),
        "pu_roc_auc_gigi": round(cur["roc_auc"], 3),
        "pu_pr_auc_all_positives": round(cur_all["pr_auc"], 3),
        "pu_roc_auc_all_positives": round(cur_all["roc_auc"], 3),
        "roc_auc_gigi_vs_rest": round(float(roc_auc_score(gigi, score)), 3),
        "avg_precision_gigi_vs_rest": round(float(average_precision_score(gigi, score)), 3),
        "roc_auc_allpos_vs_rest": round(float(roc_auc_score(is_pos, score)), 3),
        "recall@0.5_all_positives": round(float((score[is_pos] >= 0.5).mean()), 3),
        "recall@0.5_gigi_positives": round(float((score[gigi] >= 0.5).mean()), 3),
        "recall@0.5_silver_positives": round(float((score[silver] >= 0.5).mean()), 3),
        "population_flag_rate@0.5": round(float((score[~is_pos] >= 0.5).mean()), 3),
        "population_mean_probability": round(float(score[~is_pos].mean()), 3),
        "target_base_rate": PI,
    }

    # flag-rate vs per-PLZ base rate
    try:
        br = pd.read_csv(C.PV_BASERATE_PLZ, dtype={"plz": str})
        pl = (oof[~is_pos].assign(flag=score[~is_pos] >= 0.5)
              .groupby("plz")["flag"].mean().rename("pred_rate").reset_index()
              .merge(br[["plz", "pv_rate"]], on="plz"))
        rho = spearmanr(pl["pred_rate"], pl["pv_rate"]).statistic
        metrics["plz_flagrate_vs_baserate_spearman"] = round(float(rho), 3)
        plt.figure(figsize=(4, 4))
        plt.scatter(pl["pv_rate"], pl["pred_rate"], s=12, alpha=0.6)
        plt.plot([0, .6], [0, .6], "k--", lw=.8)
        plt.xlabel("ElPA PV rate (per PLZ)"); plt.ylabel("model flag rate (per PLZ)")
        plt.title(f"Spearman {rho:.2f}"); plt.tight_layout()
        plt.savefig(FIG / "flagrate_vs_baserate.png", dpi=110); plt.close()
    except FileNotFoundError:
        pass

    # change-point vs commissioning date (known positives)
    cp = oof[gigi].copy()
    cp["comm"] = pd.to_datetime(cp["pv_commissioning_date"], errors="coerce")
    cp = cp[cp["comm"].between("2023-06-01", "2026-01-01") & cp["midday_max_yoy_drop"].notna()]
    if len(cp) > 5:
        rho = spearmanr(cp["comm"].astype("int64"), cp["midday_max_yoy_drop"]).statistic
        metrics["changepoint_vs_commissioning_spearman"] = round(float(rho), 3)
        metrics["n_midwindow_installs_checked"] = int(len(cp))

    # PU-PR / ROC figure
    plt.figure(figsize=(8, 3.4))
    plt.subplot(1, 2, 1)
    plt.plot(cur["rec"], cur["prec"]); plt.xlabel("recall (on positives)")
    plt.ylabel("PU precision"); plt.title(f"PU-PR  AUC={cur['pr_auc']:.3f}")
    plt.ylim(0, 1)
    plt.subplot(1, 2, 2)
    plt.plot(cur["fpr"], cur["rec"]); plt.plot([0, 1], [0, 1], "k--", lw=.7)
    plt.xlabel("PU FPR"); plt.ylabel("TPR"); plt.title(f"PU-ROC  AUC={cur['roc_auc']:.3f}")
    plt.tight_layout(); plt.savefig(FIG / "pu_pr_roc.png", dpi=110); plt.close()

    # score distribution
    plt.figure(figsize=(5, 3))
    plt.hist(score[~is_pos], bins=50, alpha=.6, density=True, label="unlabelled")
    plt.hist(score[is_pos], bins=50, alpha=.6, density=True, label="known positive")
    plt.legend(); plt.xlabel("pv_probability"); plt.tight_layout()
    plt.savefig(FIG / "score_hist.png", dpi=110); plt.close()

    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nfigures -> {FIG}")
    return metrics


if __name__ == "__main__":
    main()
