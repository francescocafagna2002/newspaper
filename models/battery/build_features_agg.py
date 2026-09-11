"""Step 4 - aggregate the monthly per-meter table into per-(household, year)
features for the battery PU model.

monthly (mp_id, year, month)
    -> per-(mp_id, year): seasonal means, summer-winter contrasts,
       Pettitt change-point on the daily evening_zero_share series
    -> consumption-weighted rollup to (gp_nr, year)

Contrast features (summer - winter) matter more than the levels: they cancel
household size and occupancy, which the levels don't (battery_implementation_
spec.md S4.4). Annual export energy is deliberately NOT attached as a
feature-level control -- a battery reduces it, so it is a mediator, not a
covariate; it may only appear inside signal features (e.g. the contrasts
above, which already use it).

Output: artifacts/battery/features_gp_year.pkl, one row per (gp_nr, year).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from ..pv.io import gp_to_mpid

CONTRAST_FEATURES = [
    "evening_zero_share", "both_zero_share", "flat_interval_share",
    "midday_import_share",
]


def pettitt_test(x: np.ndarray) -> tuple[int, float, float]:
    """Pettitt's (1979) non-parametric single change-point test.

    Returns (change_index, K, p): change_index is the 0-based index of the
    last point before the shift; K is the test statistic; p is the standard
    large-sample approximation. Rank-based O(n log n) reformulation:
    U_t = 2*sum(rank_1..t) - t*(n+1), K = max_t |U_t|.
    """
    n = len(x)
    if n < 4:
        return -1, 0.0, 1.0
    ranks = pd.Series(x).rank().to_numpy()
    u = 2 * np.cumsum(ranks) - np.arange(1, n + 1) * (n + 1)
    t = int(np.argmax(np.abs(u)))
    k = float(np.abs(u[t]))
    p = float(min(1.0, 2 * np.exp(-6 * k ** 2 / (n ** 3 + n ** 2))))
    return t, k, p


def _change_points(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per mp_id: cp_year (year of the detected shift, or NaN) and
    cp_strength (1 - p, higher = more significant) on the daily
    evening_zero_share series. Run on the DAILY series deliberately -- the
    PV model's change-point feature differenced year-aggregated values and
    only reached Spearman 0.05 against true commissioning dates."""
    rows = []
    daily = daily.dropna(subset=["evening_zero_share"]).sort_values(["mp_id", "date"])
    for mp_id, g in daily.groupby("mp_id"):
        x = g["evening_zero_share"].to_numpy()
        if len(x) < 30:
            rows.append({"mp_id": mp_id, "cp_year": np.nan, "cp_strength": 0.0})
            continue
        idx, _k, p = pettitt_test(x)
        cp_year = int(g["date"].iloc[idx].year) if idx >= 0 else np.nan
        rows.append({"mp_id": mp_id, "cp_year": cp_year, "cp_strength": 1 - p})
    return pd.DataFrame(rows)


def _seasonal_meter_year(m: pd.DataFrame) -> pd.DataFrame:
    """Per-(mp_id, year): seasonal means, summer-winter contrasts, the
    PV-size control, and the weight column for the household rollup."""
    m = m.copy()
    m["season"] = np.select(
        [m["month"].isin(C.SUMMER_MONTHS), m["month"].isin(C.WINTER_MONTHS)],
        ["summer", "winter"], default="shoulder",
    )

    def seasonal(df: pd.DataFrame, season: str) -> pd.DataFrame:
        cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_clear")]
        g = df[df.season == season].groupby(["mp_id", "year"])[cols].mean()
        g.columns = [f"{c.replace('_mean', '')}_{season}" for c in g.columns]
        return g

    su = seasonal(m, "summer")
    wi = seasonal(m, "winter")
    out = su.join(wi, how="outer")

    for feat in CONTRAST_FEATURES:
        a, b = f"{feat}_summer", f"{feat}_winter"
        if a in out and b in out:
            out[f"{feat}_contrast"] = out[a] - out[b]

    # PV-size control: 99th percentile of summer export peak power, not
    # annual export energy (that is a mediator of the battery effect itself).
    peak = (m[m.season == "summer"].groupby(["mp_id", "year"])["exp_peak_kw_mean"]
            .quantile(0.99).rename("export_peak_kw_summer_p99"))
    out = out.join(peak, how="left")

    ann = m.groupby(["mp_id", "year"]).agg(
        day_kwh_mean=("day_kwh_mean", "mean"),
        n_months=("month", "nunique"),
        plz=("plz", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
    )
    out = out.join(ann, how="outer")
    return out.reset_index()


def _rollup_to_gp_year(meter_year: pd.DataFrame) -> pd.DataFrame:
    """Consumption-weighted mean of every feature over a household's meters,
    per year -- the household-year is the modelling unit (label hygiene for
    mid-window retrofits; see battery_mvp_plan.md D4)."""
    chain = gp_to_mpid()[["gp_nr", "mp_id"]].dropna().drop_duplicates()
    mm = meter_year.merge(chain, on="mp_id", how="left")
    mm["gp_nr"] = mm["gp_nr"].fillna("MP:" + mm["mp_id"])

    id_cols = {"mp_id", "gp_nr", "year", "plz"}
    feat_cols = [c for c in mm.columns if c not in id_cols]
    w = mm["day_kwh_mean"].clip(lower=0.1).to_numpy(float)

    g = mm.groupby(["gp_nr", "year"])
    parts = {"n_meters": g["mp_id"].nunique()}
    for c in feat_cols:
        v = mm[c].to_numpy(float)
        ok = np.isfinite(v)
        num = pd.Series(np.where(ok, w * v, 0.0), index=mm.index).groupby(
            [mm["gp_nr"], mm["year"]]).sum()
        den = pd.Series(np.where(ok, w, 0.0), index=mm.index).groupby(
            [mm["gp_nr"], mm["year"]]).sum()
        parts[c] = num / den.where(den > 0)
    out = pd.DataFrame(parts)
    out["plz"] = g["plz"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "")
    return out.reset_index()


def main() -> None:
    m = pd.read_pickle(C.FEATURES_MONTHLY)
    m = m[m["n_days"] >= C.MIN_DAYS_PER_MONTH].copy()
    try:
        flags = pd.read_csv(C.PV_METER_FLAGS, dtype={"mp_id": str})
        prod = set(flags.loc[flags["is_production_meter"], "mp_id"])
        m = m[~m["mp_id"].isin(prod)]
    except FileNotFoundError:
        print("  (meter_flags.csv missing - not excluding production meters)")
    print(f"meter-months after production filter / min-days: {len(m)}")

    meter_year = _seasonal_meter_year(m)
    print(f"meter-years: {len(meter_year)}")

    daily = pd.read_pickle(C.DAILY_FEATURES)
    cp = _change_points(daily)
    meter_year = meter_year.merge(cp, on="mp_id", how="left")

    gp_year = _rollup_to_gp_year(meter_year)

    pv = pd.read_csv(C.PV_OOF, dtype={"gp_nr": str})[["gp_nr", "pv_probability"]] \
        if C.PV_OOF.exists() else pd.DataFrame(columns=["gp_nr", "pv_probability"])
    gp_year = gp_year.merge(pv, on="gp_nr", how="left")
    gp_year["pv_probability_bin"] = np.select(
        [gp_year["pv_probability"] < 0.2, gp_year["pv_probability"] <= 0.8],
        [0, 1], default=2,
    ).astype(int)
    gp_year.loc[gp_year["pv_probability"].isna(), "pv_probability_bin"] = 1

    try:
        br = pd.read_csv(C.PV_BASERATE_PLZ, dtype={"plz": str})
        gp_year = gp_year.merge(br[["plz", "pv_rate"]], on="plz", how="left")
    except FileNotFoundError:
        gp_year["pv_rate"] = np.nan

    labels = pd.read_csv(C.LABELS, dtype={"gp_nr": str})
    kanton = labels.groupby("gp_nr")["kanton"].first()
    gp_year = gp_year.merge(kanton.rename("kanton"), on="gp_nr", how="left")

    C.FEATURES_GP_YEAR.parent.mkdir(parents=True, exist_ok=True)
    gp_year.to_pickle(C.FEATURES_GP_YEAR)

    n_synth = gp_year["gp_nr"].str.startswith("MP:").sum()
    print(f"wrote {C.FEATURES_GP_YEAR}")
    print(f"  household-years         : {len(gp_year)}  ({n_synth} synthetic MP:*)")
    print(f"  households              : {gp_year['gp_nr'].nunique()}")
    print(f"  with >=6 months, this yr: "
          f"{(gp_year['n_months'] >= C.MIN_MONTHS_PER_YEAR).sum()}")
    print(f"  pv_probability_bin      : {gp_year['pv_probability_bin'].value_counts().to_dict()}")
    print(f"  cp_year present         : {gp_year['cp_year'].notna().sum()}")


if __name__ == "__main__":
    main()
