"""Standalone chart: what GWR code 7610 actually contains, and the
like-for-like comparison against the detector's flag rate.

    python -m models.heat_pump_boiler.wpb_comparison_chart
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C

OUT = C.ARTIFACTS / "figures" / "wpb_gwr_expected_comparison.png"
INK, ACCENT, MUTED, WARN = "#1f2937", "#0f766e", "#9ca3af", "#b45309"
MIN_METERS = 30


def build() -> str:
    df = pd.read_csv(C.ARTIFACTS / "wpb_gwr_comparison.csv")
    sub = df[df.n_meters >= MIN_METERS]
    corr = pd.read_csv(C.ARTIFACTS / "wpb_gwr_correlations.csv")
    prov = json.loads((C.ARTIFACTS / "wpb_gwr_provenance.json").read_text())

    n_meters, n_flagged = int(sub.n_meters.sum()), int(sub.n_flagged.sum())
    n_buildings = int(sub.n_buildings.sum())
    n_sep = int(sub.n_separate_wpb.sum())
    n_all_hp = int(sub.n_hotwater_hp.sum())
    flag_rate, sep_rate = n_flagged / n_meters, n_sep / n_buildings
    all_rate = n_all_hp / n_buildings

    combined = prov["n_hotwater_hp"] - prov["n_separate_wpb"]
    pct_combined = 100 * prov["share_of_7610_that_is_combined"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.4),
                                   gridspec_kw={"width_ratios": [1.2, 1]})

    # --- Panel A: what the register code actually contains -------------
    ax1.barh([0], [combined], color=MUTED, height=0.42)
    ax1.barh([0], [prov["n_separate_wpb"]], left=[combined], color=ACCENT, height=0.42)
    ax1.set_yticks([])
    ax1.set_ylim(-0.75, 0.95)
    ax1.set_xlim(0, prov["n_hotwater_hp"] * 1.04)
    ax1.set_xlabel("AG buildings coded GWAERZW1 = 7610 'Wärmepumpe'",
                   fontsize=10, color=INK, labelpad=8)
    ax1.set_title("GWR has no Wärmepumpenboiler code", fontsize=12.5,
                  color=INK, fontweight="bold", pad=26)
    ax1.text(combined * 0.5, 0, f"{combined:,}  ({pct_combined:.0f}%)",
             ha="center", va="center", fontsize=11.5, color="white", fontweight="bold")
    ax1.text(combined * 0.5, -0.30,
             "space heating is ALSO a heat pump → one machine, not a WPB",
             ha="center", va="top", fontsize=9.5, color=INK)
    ax1.annotate(f"{prov['n_separate_wpb']:,} ({100 - pct_combined:.0f}%) separate\nhot-water heat pump",
                 xy=(combined + prov["n_separate_wpb"] * 0.5, 0.22),
                 xytext=(prov["n_hotwater_hp"] * 0.86, 0.80),
                 ha="center", va="top", fontsize=9.5, color=ACCENT, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
    for sp in ["top", "right", "left"]:
        ax1.spines[sp].set_visible(False)

    # --- Panel B: like-for-like rate comparison -------------------------
    labels = [f"Detector\nWPB candidates\n{n_flagged:,} of {n_meters:,} meters",
              f"GWR separate\nhot-water heat pump\n{n_sep:,} of {n_buildings:,} buildings"]
    values = [flag_rate * 100, sep_rate * 100]
    bars = ax2.bar(labels, values, color=[ACCENT, WARN], width=0.5)
    for b, v in zip(bars, values):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}%",
                 ha="center", fontsize=13, color=INK, fontweight="bold")
    ax2.set_ylim(0, max(values) * 1.38)
    ax2.set_ylabel("Share of population (%)", fontsize=10, color=INK)
    ax2.set_title(f"Like-for-like: same order of magnitude\n"
                  f"(the broad 7610 rate would be {all_rate * 100:.0f}%)",
                  fontsize=12.5, color=INK, fontweight="bold", pad=14)
    ax2.tick_params(labelsize=9.5)
    ax2.spines[["top", "right"]].set_visible(False)

    r30 = corr[(corr.min_meters == 30) & (corr.reference == "gwr_separate_wpb_rate")].iloc[0]
    r100 = corr[(corr.min_meters == 100) & (corr.reference == "gwr_separate_wpb_rate")].iloc[0]

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.text(0.5, 0.045,
             f"Postcode rank correlation vs. this rate:  ρ = {r30.rho:+.2f} "
             f"(p = {r30.pvalue:.2f}, n = {int(r30.n_postcodes)})   ·   "
             f"≥100 meters: ρ = {r100.rho:+.2f} (p = {r100.pvalue:.3f}, "
             f"n = {int(r100.n_postcodes)})",
             ha="center", va="center", fontsize=8.8, color=MUTED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    return str(OUT)


if __name__ == "__main__":
    print("wrote", build())
