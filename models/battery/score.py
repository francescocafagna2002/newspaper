"""Step 7 - score every household in the universe and write the deliverable.

Households outside the universe get ``battery_probability = NaN`` and a
``reason`` string -- never a fabricated 0. The base-rate shift is refitted on
this model's own predictions over U, because the stored gamma was fitted on
out-of-fold scores that the refit final model no longer matches (the same
pattern as models/pv/modeling/score.py).

Output: artifacts/battery/battery_predictions.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .train import _apply_gamma, select_rows

# interpretable drivers: (feature, direction meaning "more likely battery", label)
EVIDENCE = [
    ("evening_zero_share_summer", "high",
     "grid import sits at ~zero through the evening in summer"),
    ("both_zero_share_summer", "high",
     "neither import nor export for long stretches (net pinned at zero)"),
    ("flat_interval_share_summer", "high",
     "long constant-power intervals in net load (charge/discharge plateaus)"),
    ("evening_zero_share_contrast", "high",
     "the evening-zero pattern is summer-specific, not year-round"),
    ("zero_run_after_sunset_summer", "high",
     "import stops after sunset and resumes in a step"),
    ("exp_peak_flatness_summer", "high",
     "feed-in peak is flattened / clipped rather than bell-shaped"),
]


def _evidence_strings(df: pd.DataFrame) -> list[str]:
    z = {}
    for feat, _d, _l in EVIDENCE:
        if feat not in df:
            continue
        col = df[feat].astype(float)
        sd = col.std(ddof=0)
        z[feat] = (col - col.median()) / (sd if sd and np.isfinite(sd) else 1.0)
    out = []
    for i in range(len(df)):
        contribs = []
        for feat, direction, label in EVIDENCE:
            if feat not in z:
                continue
            zi = z[feat].iat[i]
            if not np.isfinite(zi):
                continue
            signed = -zi if direction == "low" else zi
            if signed > 0.7:
                contribs.append((signed, label))
        contribs.sort(reverse=True)
        out.append("; ".join(l for _, l in contribs[:3]) or "weak / mixed signal")
    return out


def main() -> None:
    meta = json.loads((C.MODEL_DIR / "meta.json").read_text())
    model = pd.read_pickle(C.MODEL_DIR / "model.pkl")
    gp_year = pd.read_pickle(C.FEATURES_GP_YEAR)
    labels = pd.read_csv(C.LABELS, dtype={"gp_nr": str})
    universe = pd.read_csv(C.UNIVERSE, dtype={"gp_nr": str})

    scored = select_rows(gp_year, labels, universe)
    X = scored[meta["features"]].astype(float)
    X = X.fillna(X.median(numeric_only=True))
    g = model.predict_proba(X)[:, 1]
    pu = np.clip(g / meta["c"], 0, 1)

    # refit the shift on THIS model's unlabelled predictions
    from scipy.optimize import brentq
    u = (scored["pu_label"] != "P").to_numpy()
    try:
        gamma = float(brentq(
            lambda gm: _apply_gamma(pu[u], gm).mean() - meta["default_pi"], -10, 10))
    except ValueError:
        gamma = meta["gamma"]
    prob = _apply_gamma(pu, gamma)
    scored = scored.assign(battery_probability=prob)

    # one row per household: its most recent scored year
    latest = scored.sort_values("year").groupby("gp_nr").tail(1).copy()
    latest["top_evidence"] = _evidence_strings(latest)

    hh = labels.groupby("gp_nr", as_index=False).agg(
        battery_positive=("battery_positive", "max"),
        audit_negative=("audit_negative", "max"),
    )
    out = latest[["gp_nr", "plz", "kanton", "n_meters", "battery_probability",
                  "pv_probability_bin", "top_evidence"]].copy()
    out["in_universe"] = True
    out["reason"] = "scored"
    out["battery_pred"] = (out["battery_probability"] >= 0.5).astype(int)
    out["n_years"] = latest["gp_nr"].map(scored.groupby("gp_nr")["year"].nunique())
    out = out.merge(hh, on="gp_nr", how="left")

    # households known to the universe table but not scorable: NaN, with a reason
    missing = universe[~universe["gp_nr"].isin(out["gp_nr"])].copy()
    if len(missing):
        reason = np.where(
            missing["in_universe"].fillna(False),
            "in universe but no year with >=" + str(C.MIN_MONTHS_PER_YEAR) + " months of data",
            "outside universe: " + missing["admission_rule"].fillna("excluded").astype(str))
        pad = pd.DataFrame({
            "gp_nr": missing["gp_nr"], "plz": np.nan, "kanton": np.nan,
            "n_meters": np.nan, "battery_probability": np.nan,
            "pv_probability_bin": np.nan, "top_evidence": "not scored",
            "in_universe": missing["in_universe"].fillna(False).to_numpy(),
            "reason": reason, "battery_pred": np.nan, "n_years": 0,
            "battery_positive": np.nan, "audit_negative": np.nan,
        })
        out = pd.concat([out, pad], ignore_index=True)

    out = out.sort_values("battery_probability", ascending=False)
    out.to_csv(C.PREDICTIONS, index=False)

    scored_only = out[out["battery_probability"].notna()]
    known = scored_only[scored_only["battery_positive"] == True]  # noqa: E712
    print(f"wrote {C.PREDICTIONS}  ({len(out)} households)")
    print(f"  scored                    : {len(scored_only)}")
    print(f"  not scored (NaN + reason) : {len(out) - len(scored_only)}")
    print(f"  predicted battery (p>=0.5): {scored_only['battery_pred'].mean():.1%}")
    print(f"  mean probability          : {scored_only['battery_probability'].mean():.3f}")
    if len(known):
        print(f"  known positives recovered : {known['battery_pred'].mean():.1%} "
              f"(n={len(known)})")


if __name__ == "__main__":
    main()
