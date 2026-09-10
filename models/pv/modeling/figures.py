"""Figures for the slide deck / report.

    python -m newspaper.models.pv.modeling.figures

Writes PNGs to artifacts/figures/:
  fig_architecture.png     pipeline flow diagram
  fig_feature_sep.png      how well each feature separates PV from the rest
  fig_top_feature.png      distribution of the dominant feature, by class
  fig_importance.png       permutation importance (trained model)
  fig_results.png          ROC-AUC + recall vs baselines
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

from .. import config as C

FIG = C.ARTIFACTS / "figures"

PV = "#2a78d6"       # categorical slot 1 - "has PV"
REST = "#eb6834"     # categorical slot 2 - "rest of population"
INK = "#0b0b0b"
SUB = "#52514e"
SURF = "#fcfcfb"
GRID = "#e7e7e3"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": SUB, "ytick.color": SUB,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "font.size": 11,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
})

KEY_FEATURES = {
    "daytime_zero_summer":   "summer daytime near-zero-import share",
    "midday_clear_summer":   "midday load ÷ daily load, sunny days",
    "midday_seasonality":    "midday dip: winter − summer",
    "load_irr_slope":        "midday load vs irradiance (slope)",
    "evening_midday_summer": "evening peak ÷ midday load",
    "summer_winter_kwh_ratio": "summer ÷ winter daily kWh",
    "night_kwh_mean":        "night base load (size control)",
}


def _load():
    gp = pd.read_pickle(C.FEATURES_GP)
    gp = gp[gp["n_months"] >= 3].reset_index(drop=True)
    gp["is_pv"] = gp["pv_label"] == 1
    return gp


# --------------------------------------------------------------------------- #
def fig_architecture():
    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

    def box(x, y, w, h, title, lines, fc="#ffffff", ec=GRID):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="top", fontsize=11,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 7.6, "\n".join(lines), ha="center", va="top",
                fontsize=8.6, color=SUB, linespacing=1.5)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=16, color=SUB, lw=1.3))

    box(1, 33, 20, 20, "INPUTS", [
        "42 monthly exports (~77 GB)", "15-min grid-import, OBIS 1.29",
        "GIGI subsidy labels", "MeteoSwiss radiation", "BFE ElPA PV register"])
    box(1, 3, 20, 22, "LABELS  (prep/)", [
        "GIGI PV/battery  → 288 pos", "clear summer feed-in (OBIS 2.29,",
        "  offline)        → +6 188 silver pos", "everything else   → unlabelled",
        "no confirmed negatives"], fc="#eef4fb")

    box(27, 20, 22, 20, "FEATURES  (features/)", [
        "one streaming pass →", "per-(meter, month) table",
        "→ 23 per-household features:", "midday depression · weather–solar",
        "coupling · seasonality · change-pt"], fc="#eef4fb")

    box(55, 20, 21, 20, "MODEL  (modeling/)", [
        "Positive–Unlabeled", "(Elkan–Noto, c≈0.95)",
        "HistGradientBoosting", "5-fold CV, 1 row/household",
        "calibrated to 12% base rate"], fc="#eef4fb")

    box(82, 20, 17, 20, "OUTPUT", [
        "pv_probability", "per household (GP-Nr)",
        "+ pv_pred @ 0.5", "+ plain-language", "  evidence"], fc="#fdefe8")

    arrow(21, 40, 27, 33)          # inputs  -> features
    arrow(21, 15, 27, 27)          # labels  -> features
    arrow(49, 30, 55, 30)          # features -> model
    arrow(76, 30, 82, 30)          # model   -> output

    ax.text(50, 52, "SolarPrint — pipeline", ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig_architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_feature_sep(gp):
    """Standardised mean difference (Cohen's d) per feature — how far PV sits
    from the rest, in pooled-SD units."""
    rows = []
    for f, label in KEY_FEATURES.items():
        a, b = gp.loc[gp.is_pv, f].dropna(), gp.loc[~gp.is_pv, f].dropna()
        sd = np.sqrt(((a.var() * (len(a) - 1)) + (b.var() * (len(b) - 1))) / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / sd if sd else 0.0
        rows.append((label, d))
    rows.sort(key=lambda r: abs(r[1]))
    labels, d = zip(*rows)
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.barh(labels, d, color=PV, height=0.62)
    ax.axvline(0, color=SUB, lw=1)
    for y, x in enumerate(d):
        ax.text(x + (0.15 if x > 0 else -0.15), y, f"{x:+.1f}", va="center",
                ha="left" if x > 0 else "right", fontsize=9.5, color=INK)
    ax.set_xlabel("standardised mean difference  (PV − rest, pooled-SD units;\n"
                  "negative = feature is lower for PV households)")
    ax.set_xlim(min(d) - 1.2, max(d) + 1.2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Each feature: how far PV households sit from the rest", loc="left",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_feature_sep.png", dpi=150)
    plt.close(fig)


def fig_top_feature(gp):
    f = "daytime_zero_summer"
    a = gp.loc[gp.is_pv, f].dropna().clip(0, 1)
    b = gp.loc[~gp.is_pv, f].dropna().clip(0, 1)
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    bins = np.linspace(0, 1, 41)
    ax.hist(b, bins=bins, density=True, color=REST, alpha=0.85, label=f"rest  (n={len(b):,})")
    ax.hist(a, bins=bins, density=True, color=PV, alpha=0.85, label=f"has PV  (n={len(a):,})")
    ax.set_xlabel("summer daytime near-zero-import share")
    ax.set_ylabel("density")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("The dominant feature separates the classes on its own", loc="left",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_top_feature.png", dpi=150)
    plt.close(fig)


def fig_importance(gp):
    from sklearn.inspection import permutation_importance
    meta = json.loads((C.MODEL_DIR / "meta.json").read_text())
    model = pd.read_pickle(C.MODEL_DIR / "model.pkl")
    s = gp["is_pv"].astype(int).to_numpy()
    X = gp[meta["features"]].astype(float)
    r = permutation_importance(model, X, s, n_repeats=5, random_state=0, scoring="roc_auc")
    imp = pd.Series(r.importances_mean, index=meta["features"]).sort_values().tail(8)
    nice = {**KEY_FEATURES,
            "n_meters": "# metering points", "n_months": "# months of history",
            "pv_rate": "per-PLZ PV base rate", "load_irr_corr": "load–irradiance corr.",
            "daytime_zero_winter": "winter daytime near-zero share",
            "midday_std_summer": "midday load day-to-day scatter",
            "below_night_summer": "daytime min below night base"}
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.barh([nice.get(i, i) for i in imp.index], imp.values, color=PV, height=0.6)
    ax.set_xlabel("permutation importance  (ROC-AUC drop when shuffled)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Trained model: feature importance", loc="left", fontsize=12,
                 fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_importance.png", dpi=150)
    plt.close(fig)


def fig_results():
    full = json.loads((C.ARTIFACTS / "metrics.json").read_text())
    sfp = C.ARTIFACTS / "model_daytime_zero_summer" / "metrics.json"
    single = json.loads(sfp.read_text()) if sfp.exists() else None

    models = ["Random\nassignment", "1 feature\n(daytime-zero)", "Full model\n(23 features)"]
    roc = [0.50,
           single["roc_auc_gigi_vs_rest"] if single else np.nan,
           full["roc_auc_gigi_vs_rest"]]
    rec = [0.12,
           single["recall@0.5_gigi_positives"] if single else np.nan,
           full["recall@0.5_gigi_positives"]]

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    y = np.arange(len(models))
    h = 0.34
    gap = 0.03
    ax.barh(y + h / 2 + gap, roc, height=h, color=PV, label="ROC-AUC (GIGI vs rest)")
    ax.barh(y - h / 2 - gap, rec, height=h, color=REST, label="Recall @0.5 on known PV")
    for yy, v in zip(y + h / 2 + gap, roc):
        ax.text(v + 0.012, yy, f"{v:.2f}", va="center", fontsize=10, color=INK)
    for yy, v in zip(y - h / 2 - gap, rec):
        ax.text(v + 0.012, yy, f"{v:.2f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(models)
    ax.set_xlim(0, 1.05); ax.set_xlabel("score")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.set_title("SolarPrint vs baselines  (5-fold out-of-fold)", loc="left",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_results.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    gp = _load()
    fig_architecture()
    fig_feature_sep(gp)
    fig_top_feature(gp)
    fig_importance(gp)
    fig_results()
    print(f"wrote 5 figures -> {FIG}")


if __name__ == "__main__":
    main()
