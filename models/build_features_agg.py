"""Step 6 - aggregate the monthly per-meter table into per-household features.

monthly (mp_id, year, month)  ->  per-meter seasonal / change-point features
                              ->  consumption-weighted rollup to GP-Nr

Consumption channel only. Production meters (from meter_flags.csv) are dropped.
Meters with no household in the join chain are kept under a synthetic
gp_nr = "MP:<mp_id>" so the full population can still be scored.

Output: artifacts/features_gp.pkl
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import gp_to_mpid

MIN_DAYS_PER_MONTH = 20
_CLEAR = "midday_ratio_clear"


def _load_monthly() -> pd.DataFrame:
    m = pd.read_pickle(C.FEATURES_MONTHLY)
    m = m[m["n_days"] >= MIN_DAYS_PER_MONTH].copy()
    for c in ["day_kwh_mean", "day_kwh_p10", "day_kwh_p50", "day_kwh_p90",
              "night_kwh_mean", "day_kwh_weekday", "day_kwh_weekend"]:
        m[c] = m[c].clip(lower=0)
    for c in ["midday_ratio_mean", "midday_ratio_clear", "evening_midday_ratio_mean"]:
        m[c] = m[c].clip(0, 5)
    m["season"] = np.select(
        [m["month"].isin(C.SUMMER_MONTHS), m["month"].isin(C.WINTER_MONTHS)],
        ["summer", "winter"], default="shoulder",
    )
    try:
        flags = pd.read_csv(C.ARTIFACTS / "meter_flags.csv", dtype={"mp_id": str})
        prod = set(flags.loc[flags["is_production_meter"], "mp_id"])
        m = m[~m["mp_id"].isin(prod)]
    except FileNotFoundError:
        print("  (meter_flags.csv missing - not excluding production meters)")
    return m


def _seasonal(m: pd.DataFrame) -> pd.DataFrame:
    """Vectorised per-meter seasonal + change-point features."""
    su = m[m.season == "summer"]
    wi = m[m.season == "winter"]

    def agg(df, spec):
        return df.groupby("mp_id").agg(**spec)

    a = agg(su, dict(
        midday_clear_summer=(_CLEAR, "mean"),
        midday_mean_summer=("midday_ratio_mean", "mean"),
        midday_std_summer=("midday_ratio_std", "mean"),
        daytime_zero_summer=("frac_daytime_zero_mean", "mean"),
        below_night_summer=("below_night_frac", "mean"),
        evening_midday_summer=("evening_midday_ratio_mean", "mean"),
        day_kwh_summer=("day_kwh_mean", "mean"),
        n_summers=("year", "nunique"),
    ))
    b = agg(wi, dict(
        midday_clear_winter=(_CLEAR, "mean"),
        daytime_zero_winter=("frac_daytime_zero_mean", "mean"),
        day_kwh_winter=("day_kwh_mean", "mean"),
    ))
    c = agg(m, dict(
        day_kwh_mean=("day_kwh_mean", "mean"),
        night_kwh_mean=("night_kwh_mean", "mean"),
        n_months=("month", "size"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    ))
    plz = m.groupby("mp_id")["plz"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else "")

    out = c.join(a, how="left").join(b, how="left").join(plz.rename("plz"))
    out["midday_seasonality"] = out["midday_clear_winter"] - out["midday_clear_summer"]
    out["summer_winter_kwh_ratio"] = np.where(
        out["day_kwh_winter"] > 0, out["day_kwh_summer"] / out["day_kwh_winter"], np.nan)
    out["n_summers"] = out["n_summers"].fillna(0).astype(int)
    out["rollout_era"] = (out["first_year"] <= 2023).astype(int)

    # ---- change-point on per-summer-year clear midday ratio ----
    sy = (su.groupby(["mp_id", "year"])[_CLEAR].mean()
          .rename("v").reset_index().sort_values(["mp_id", "year"]))
    sy["drop"] = sy.groupby("mp_id")["v"].shift(1) - sy["v"]
    max_drop = sy.groupby("mp_id")["drop"].max().rename("midday_max_yoy_drop")
    # slope via grouped least squares: cov(year, v) / var(year)
    g = sy.groupby("mp_id")
    n = g["v"].size()
    sx, sy_, sxx, sxy = g["year"].sum(), g["v"].sum(), (sy["year"]**2).groupby(sy["mp_id"]).sum(), (sy["year"]*sy["v"]).groupby(sy["mp_id"]).sum()
    denom = (n * sxx - sx**2)
    slope = ((n * sxy - sx * sy_) / denom.where(denom != 0)).rename("midday_yoy_slope")
    out = out.join(max_drop).join(slope)

    return out.reset_index()


def _rollup_to_gp(meter: pd.DataFrame) -> pd.DataFrame:
    """Consumption-weighted mean of every feature over a household's meters."""
    chain = gp_to_mpid()[["gp_nr", "mp_id"]].dropna().drop_duplicates()
    meter = meter.merge(chain, on="mp_id", how="left")
    meter["gp_nr"] = meter["gp_nr"].fillna("MP:" + meter["mp_id"])

    id_cols = {"mp_id", "gp_nr", "plz", "first_year", "last_year"}
    feat_cols = [c for c in meter.columns if c not in id_cols]
    w = meter["day_kwh_mean"].clip(lower=0.1).to_numpy(float)

    g = meter.groupby("gp_nr")
    parts = {"n_meters": g.size()}
    for c in feat_cols:
        v = meter[c].to_numpy(float)
        ok = np.isfinite(v)
        num = pd.Series(np.where(ok, w * v, 0.0), index=meter.index).groupby(meter["gp_nr"]).sum()
        den = pd.Series(np.where(ok, w, 0.0), index=meter.index).groupby(meter["gp_nr"]).sum()
        parts[c] = num / den.where(den > 0)
    out = pd.DataFrame(parts)
    out["plz"] = g["plz"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "")
    out["first_year"] = g["first_year"].min().astype(int)
    out["last_year"] = g["last_year"].max().astype(int)
    return out.reset_index()


def main() -> None:
    m = _load_monthly()
    meter = _seasonal(m)
    print(f"meters after production filter / min-days: {len(meter)}")

    gp = _rollup_to_gp(meter)

    lab = pd.read_csv(C.GIGI_AUGMENTED, dtype={"gp_nr": str, "mp_id": str, "plz": str})
    def _src(s):
        real = [x for x in s if x != "unlabeled"]
        return real[0] if real else "unlabeled"
    lab_gp = (
        lab.groupby("gp_nr")
        .agg(pv_label=("pv_label", "max"), label_source=("label_source", _src),
             pv_commissioning_date=("pv_commissioning_date", "first"),
             kanton=("kanton", "first"))
        .reset_index()
    )
    gp = gp.merge(lab_gp, on="gp_nr", how="left")

    try:
        br = pd.read_csv(C.PV_BASERATE_PLZ, dtype={"plz": str})
        gp = gp.merge(br[["plz", "pv_rate"]], on="plz", how="left")
    except FileNotFoundError:
        gp["pv_rate"] = np.nan

    C.FEATURES_GP.parent.mkdir(parents=True, exist_ok=True)
    gp.to_pickle(C.FEATURES_GP)

    n_pos = (gp["pv_label"] == 1).sum()
    n_synth = gp["gp_nr"].str.startswith("MP:").sum()
    print(f"wrote {C.FEATURES_GP}")
    print(f"  households (GP-Nr)     : {len(gp)}  ({n_synth} synthetic MP:*)")
    print(f"  positives (any source): {n_pos}")
    print(f"  positives by source   : "
          f"{gp.loc[gp.pv_label == 1, 'label_source'].value_counts().to_dict()}")
    print(f"  midday_clear_summer   : pos={gp.loc[gp.pv_label==1,'midday_clear_summer'].mean():.3f}"
          f"  rest={gp.loc[gp.pv_label!=1,'midday_clear_summer'].mean():.3f}")


if __name__ == "__main__":
    main()
