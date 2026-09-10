"""Per-household auxiliary features that survive meter aggregation.

``load_property_daily.csv`` sums a household's meters into one series, which is
right -- the GIGI label describes the customer -- but it hides *which* meter
carried the load.  Two cases matter:

  * a dedicated **heat-pump meter**: winter-heavy, near-zero in summer.  Summed
    into the household total it is only a mild winter bump; on its own it is
    the cleanest heat-pump example in the dataset.
  * a dedicated **PV production point**: zero import, pure export.

This step reads ``meter_month_labelled.csv`` (per-meter monthly) and emits one
row per household with the per-meter extremes preserved, plus the household
size flags the aggregate cannot express.

Output: ``data_addition/property_aux_features.csv``
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import read_gigi, gp_to_mpid

IN = C.DATA_ADDITION / "meter_month_labelled.csv"
OUT = C.DATA_ADDITION / "property_aux_features.csv"

# a meter is "heat-pump-like" if winter draw dwarfs summer draw
HP_RATIO = 3.0
HP_MAX_SUMMER_SHARE = 0.20
# a meter is a production point if it exports but essentially never imports
PROD_MAX_IMPORT_KWH = 50.0
MIN_MONTHS = 3          # need some history before the ratio means anything
MULTI_UNIT_MIN_METERS = 4


def _meter_seasonal(mm: pd.DataFrame) -> pd.DataFrame:
    """Per-meter winter/summer import totals and export totals."""
    imp = mm[mm["direction"] == "bezug"].copy()
    exp = mm[mm["direction"] == "einsp"].copy()

    imp["season"] = np.select(
        [imp["month"].isin(C.SUMMER_MONTHS), imp["month"].isin(C.WINTER_MONTHS)],
        ["summer", "winter"], default="shoulder",
    )
    # mean kWh per day, so months with different coverage stay comparable
    imp["kwh_per_day"] = imp["kwh"] / imp["n_days"].clip(lower=1)

    piv = (imp.pivot_table(index="mp_id", columns="season",
                           values="kwh_per_day", aggfunc="mean")
             .rename(columns={"summer": "summer_kwh_d",
                              "winter": "winter_kwh_d",
                              "shoulder": "shoulder_kwh_d"}))
    for c in ("summer_kwh_d", "winter_kwh_d", "shoulder_kwh_d"):
        if c not in piv:
            piv[c] = np.nan

    tot = imp.groupby("mp_id").agg(imp_kwh=("kwh", "sum"),
                                   n_months=("kwh", "size"))
    ex = exp.groupby("mp_id").agg(exp_kwh=("kwh", "sum"))

    m = piv.join(tot, how="outer").join(ex, how="outer")
    m["imp_kwh"] = m["imp_kwh"].fillna(0.0)
    m["exp_kwh"] = m["exp_kwh"].fillna(0.0)
    m["n_months"] = m["n_months"].fillna(0).astype(int)

    denom = m["summer_kwh_d"].where(m["summer_kwh_d"] > 0.01)
    m["winter_summer_ratio"] = m["winter_kwh_d"] / denom
    total_season = m[["summer_kwh_d", "winter_kwh_d"]].sum(axis=1)
    m["summer_share"] = m["summer_kwh_d"] / total_season.where(total_season > 0)

    m["is_heat_pump_like"] = (
        (m["winter_summer_ratio"] >= HP_RATIO)
        & (m["summer_share"] <= HP_MAX_SUMMER_SHARE)
        & (m["n_months"] >= MIN_MONTHS)
    ).fillna(False)
    m["is_production_only"] = (m["exp_kwh"] > 0) & (m["imp_kwh"] <= PROD_MAX_IMPORT_KWH)
    return m.reset_index()


def _rollup(meter: pd.DataFrame) -> pd.DataFrame:
    chain = gp_to_mpid()[["gp_nr", "mp_id"]].dropna().drop_duplicates()
    gigi = set(read_gigi(annotated=True)["gp_nr"].dropna()) - {""}
    chain = chain[chain["gp_nr"].isin(gigi)]

    df = meter.merge(chain, on="mp_id", how="inner")
    g = df.groupby("gp_nr")

    out = pd.DataFrame({
        "n_meters": g["mp_id"].nunique(),
        "n_meters_with_data": g["n_months"].apply(lambda s: int((s > 0).sum())),
        # the extremes are what aggregation destroys
        "meter_max_winter_summer_ratio": g["winter_summer_ratio"].max(),
        "meter_min_summer_share": g["summer_share"].min(),
        "n_heat_pump_like_meters": g["is_heat_pump_like"].sum(),
        "n_production_only_meters": g["is_production_only"].sum(),
        "meter_max_imp_kwh": g["imp_kwh"].max(),
        "meter_min_imp_kwh": g["imp_kwh"].min(),
        "household_imp_kwh": g["imp_kwh"].sum(),
        "household_exp_kwh": g["exp_kwh"].sum(),
    })
    # "winter-dominant" describes the *household*; on a single-meter household
    # the max-over-meters ratio is simply that household's own seasonality.
    out["is_winter_dominant"] = (out["n_heat_pump_like_meters"] > 0).astype(int)
    # The signal aggregation actually destroys: a *dedicated* heating meter,
    # i.e. one winter-only meter sitting next to a normal-looking one.
    n_other = out["n_meters"] - out["n_heat_pump_like_meters"]
    out["has_dedicated_hp_meter"] = (
        (out["n_meters"] > 1) & (out["n_heat_pump_like_meters"] > 0) & (n_other > 0)
    ).astype(int)
    out["has_production_only_meter"] = (out["n_production_only_meters"] > 0).astype(int)
    out["is_multi_meter"] = (out["n_meters"] > 1).astype(int)
    out["is_multi_unit"] = (out["n_meters"] >= MULTI_UNIT_MIN_METERS).astype(int)
    # share of the household total carried by its single biggest meter
    out["meter_concentration"] = (
        out["meter_max_imp_kwh"] / out["household_imp_kwh"].where(out["household_imp_kwh"] > 0)
    )
    return out.reset_index()


TRAIN_IN = C.DATA_ADDITION / "train_property_samples.csv"
TRAIN_OUT = C.DATA_ADDITION / "train_property_samples_aux.csv"

AUX_COLS = [
    "n_meters", "n_meters_with_data", "is_multi_meter", "is_multi_unit",
    "meter_max_winter_summer_ratio", "meter_min_summer_share",
    "meter_concentration", "n_heat_pump_like_meters", "is_winter_dominant", "has_dedicated_hp_meter",
    "n_production_only_meters", "has_production_only_meter",
]


def merge_into_train(prop: pd.DataFrame) -> None:
    """Attach the aux columns to the existing cross-sectional training table."""
    if not TRAIN_IN.exists():
        print(f"  ({TRAIN_IN.name} missing - skipping merge)")
        return
    tr = pd.read_csv(TRAIN_IN, sep=C.CSV_SEP, dtype={"gp_nr": str, "plz": str})
    before = len(tr)
    merged = tr.merge(prop[["gp_nr"] + AUX_COLS], on="gp_nr", how="left")
    assert len(merged) == before, "merge changed the row count"

    # single-meter households: the aggregate *is* the meter
    merged["n_meters"] = merged["n_meters"].fillna(1).astype(int)
    for c in ("is_multi_meter", "is_multi_unit", "is_winter_dominant",
              "has_dedicated_hp_meter",
              "has_production_only_meter", "n_heat_pump_like_meters",
              "n_production_only_meters"):
        merged[c] = merged[c].fillna(0).astype(int)

    merged.to_csv(TRAIN_OUT, sep=C.CSV_SEP, index=False)
    matched = merged["meter_max_winter_summer_ratio"].notna().sum()
    print(f"\nwrote {TRAIN_OUT}  ({len(merged)} samples, +{len(AUX_COLS)} cols)")
    print(f"  samples with aux features     : {matched}/{len(merged)}")
    print(f"  samples from multi-meter hh   : {int(merged.is_multi_meter.sum())}")
    print(f"  samples w/ dedicated hp meter : {int(merged.has_dedicated_hp_meter.sum())}")
    # NOTE: has_Waermepumpe == 0 is UNLABELLED, not a verified negative, so the
    # contrast below is a floor on separation, not an accuracy estimate.
    hp = merged[merged["has_Waermepumpe"] == 1]
    rest = merged[merged["has_Waermepumpe"] == 0]
    for c in ("has_dedicated_hp_meter", "is_winter_dominant"):
        print(f"  {c:24s}: HP-labelled {hp[c].mean():.3f} vs unlabelled {rest[c].mean():.3f}")
    r = "meter_max_winter_summer_ratio"
    print(f"  {r} median: HP-labelled {hp[r].median():.2f} "
          f"vs unlabelled {rest[r].median():.2f}")


def main() -> None:
    mm = pd.read_csv(IN, sep=C.CSV_SEP, dtype={"mp_id": str, "plz": str})
    print(f"read {IN.name}: {len(mm)} rows, {mm.mp_id.nunique()} meters")

    meter = _meter_seasonal(mm)
    prop = _rollup(meter)
    prop.to_csv(OUT, sep=C.CSV_SEP, index=False)

    print(f"\nwrote {OUT}  ({len(prop)} households)")
    print(f"  multi-meter households        : {int(prop.is_multi_meter.sum())}")
    print(f"  multi-unit (>={MULTI_UNIT_MIN_METERS} meters)      : {int(prop.is_multi_unit.sum())}")
    print(f"  winter-dominant households    : {int(prop.is_winter_dominant.sum())}")
    print(f"  with a DEDICATED hp meter     : {int(prop.has_dedicated_hp_meter.sum())}")
    print(f"  with a production-only meter  : {int(prop.has_production_only_meter.sum())}")
    print("\nper-meter extremes (multi-meter households only):")
    mm_only = prop[prop.is_multi_meter == 1]
    print(mm_only[["meter_max_winter_summer_ratio", "meter_min_summer_share",
                   "meter_concentration"]].describe().round(3).to_string())

    merge_into_train(prop)


if __name__ == "__main__":
    main()
