"""Step 10 - score every household and write the deliverable.

Loads the final PU model, applies the Elkan-Noto c and the base-rate shift,
and writes one calibrated PV probability per GP-Nr with a short plain-language
evidence string.

Output: artifacts/pv_predictions.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C

# interpretable drivers: (feature, direction that means "more likely PV", label)
EVIDENCE = [
    ("midday_clear_summer", "low", "low midday consumption on sunny summer days"),
    ("daytime_zero_summer", "high", "many summer daytime intervals at ~zero import"),
    ("midday_seasonality", "high", "midday consumption drops sharply in summer"),
    ("midday_std_summer", "high", "midday load swings day-to-day with cloud cover"),
    ("summer_winter_kwh_ratio", "low", "much lower net import in summer than winter"),
    ("midday_max_yoy_drop", "high", "midday self-consumption appeared partway through the record"),
]


def _evidence_strings(df: pd.DataFrame) -> list[str]:
    z = {}
    for feat, _d, _l in EVIDENCE:
        col = df[feat].astype(float)
        z[feat] = (col - col.median()) / (col.std(ddof=0) or 1.0)
    out = []
    for i in range(len(df)):
        contribs = []
        for feat, direction, label in EVIDENCE:
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
    gp = pd.read_pickle(C.FEATURES_GP)
    gp = gp[gp["n_months"] >= 3].reset_index(drop=True)

    X = gp[meta["features"]].astype(float)
    g = model.predict_proba(X)[:, 1]
    pu = np.clip(g / meta["c"], 0, 1)
    # recalibrate the base-rate shift on THIS model's unlabelled predictions so
    # the population mean matches the ElPA base rate (the stored gamma was fit
    # on out-of-fold scores, which the refit model no longer matches)
    from scipy.optimize import brentq
    u = ~(gp["pv_label"] == 1).to_numpy()
    def _shift(p, gm):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + gm)))
    try:
        gamma = float(brentq(lambda gm: _shift(pu[u], gm).mean() - meta["target_base_rate"],
                             -10, 10))
    except ValueError:
        gamma = meta["gamma"]
    prob = _shift(pu, gamma)

    out = pd.DataFrame({
        "gp_nr": gp["gp_nr"],
        "n_meters": gp["n_meters"].astype(int),
        "plz": gp["plz"],
        "kanton": gp["kanton"].fillna("AG"),
        "pv_probability": prob.round(4),
        "pv_pred": (prob >= 0.5).astype(int),
        "known_label": gp["label_source"].where(gp["pv_label"] == 1),
        "top_evidence": _evidence_strings(gp),
    })
    out = out.sort_values("pv_probability", ascending=False)
    out.to_csv(C.PREDICTIONS, index=False)

    real = ~out["gp_nr"].str.startswith("MP:")
    print(f"wrote {C.PREDICTIONS}  ({len(out)} households)")
    print(f"  real GP-Nr households      : {real.sum()}")
    print(f"  predicted PV (p>=0.5)      : {out['pv_pred'].mean():.1%}")
    print(f"  mean probability           : {out['pv_probability'].mean():.3f}")
    print(f"  known positives recovered  : "
          f"{(out.loc[out.known_label.notna(),'pv_pred']).mean():.1%}")
    print(out.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
