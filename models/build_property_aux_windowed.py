"""Window the per-meter auxiliary features to each training sample's period.

``build_property_aux`` computes one aux row per household over each meter's
*whole* history. That makes the features identical for a household's
``augmented_before`` and ``current`` samples, so they add nothing to exactly the
comparison the before/after design exists to make.

This step recomputes the same features per **sample**, restricted to the
months overlapping ``[first_day, last_day]``, so a household that gained a
dedicated heat-pump meter mid-window looks different before and after.

A month counts toward a sample when it overlaps the window at all; months are
the finest resolution ``meter_month_labelled.csv`` carries, so a sample shorter
than ~2 months gets too few points for a seasonal ratio and is left NaN.

Output: ``data_addition/train_property_samples_auxw.csv``
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import read_gigi, gp_to_mpid
from .build_property_aux import (
    HP_RATIO, HP_MAX_SUMMER_SHARE, PROD_MAX_IMPORT_KWH, MULTI_UNIT_MIN_METERS,
)

MM_IN = C.DATA_ADDITION / "meter_month_labelled.csv"
TRAIN_IN = C.DATA_ADDITION / "train_property_samples.csv"
OUT = C.DATA_ADDITION / "train_property_samples_auxw.csv"
MANIFEST_IN = C.DATA_ADDITION / "augmentation_manifest.csv"
MANIFEST_OUT = C.DATA_ADDITION / "augmentation_manifest_auxw.csv"

MIN_MONTHS_WINDOWED = 2   # lower than the static build: windows are shorter

AUX_COLS = [
    "w_n_meters", "w_n_meters_with_data", "w_is_multi_meter", "w_is_multi_unit",
    "w_meter_max_winter_summer_ratio", "w_meter_min_summer_share",
    "w_meter_concentration", "w_n_heat_pump_like_meters",
    "w_is_winter_dominant", "w_has_dedicated_hp_meter",
    "w_n_production_only_meters", "w_has_production_only_meter",
    "w_n_months_in_window",
]


def _prepare_meter_months() -> pd.DataFrame:
    mm = pd.read_csv(MM_IN, sep=C.CSV_SEP, dtype={"mp_id": str, "plz": str})
    mm["period"] = pd.PeriodIndex.from_fields(
        year=mm["year"], month=mm["month"], freq="M")
    mm["kwh_per_day"] = mm["kwh"] / mm["n_days"].clip(lower=1)
    mm["season"] = np.select(
        [mm["month"].isin(C.SUMMER_MONTHS), mm["month"].isin(C.WINTER_MONTHS)],
        ["summer", "winter"], default="shoulder")

    chain = gp_to_mpid()[["gp_nr", "mp_id"]].dropna().drop_duplicates()
    gigi = set(read_gigi(annotated=True)["gp_nr"].dropna()) - {""}
    chain = chain[chain["gp_nr"].isin(gigi)]
    return mm.merge(chain, on="mp_id", how="inner")


def _aux_for_window(sub: pd.DataFrame, n_meters_total: int) -> dict:
    """Same feature definitions as build_property_aux, on a windowed slice."""
    imp = sub[sub["direction"] == "bezug"]
    exp = sub[sub["direction"] == "einsp"]
    if imp.empty:
        return {}

    per = imp.pivot_table(index="mp_id", columns="season",
                          values="kwh_per_day", aggfunc="mean")
    for c in ("summer", "winter"):
        if c not in per:
            per[c] = np.nan
    tot = imp.groupby("mp_id").agg(imp_kwh=("kwh", "sum"), n_months=("kwh", "size"))
    ex = exp.groupby("mp_id")["kwh"].sum().rename("exp_kwh")
    m = per.join(tot).join(ex)
    m["exp_kwh"] = m["exp_kwh"].fillna(0.0)

    denom = m["summer"].where(m["summer"] > 0.01)
    m["ratio"] = m["winter"] / denom
    season_tot = m[["summer", "winter"]].sum(axis=1)
    m["summer_share"] = m["summer"] / season_tot.where(season_tot > 0)

    hp = ((m["ratio"] >= HP_RATIO) & (m["summer_share"] <= HP_MAX_SUMMER_SHARE)
          & (m["n_months"] >= MIN_MONTHS_WINDOWED)).fillna(False)
    prod = (m["exp_kwh"] > 0) & (m["imp_kwh"] <= PROD_MAX_IMPORT_KWH)

    n_meters = int(m.index.nunique())
    hh_imp = float(m["imp_kwh"].sum())
    n_hp = int(hp.sum())
    return {
        "w_n_meters": n_meters_total,
        "w_n_meters_with_data": n_meters,
        "w_is_multi_meter": int(n_meters_total > 1),
        "w_is_multi_unit": int(n_meters_total >= MULTI_UNIT_MIN_METERS),
        "w_meter_max_winter_summer_ratio": float(m["ratio"].max())
            if m["ratio"].notna().any() else np.nan,
        "w_meter_min_summer_share": float(m["summer_share"].min())
            if m["summer_share"].notna().any() else np.nan,
        "w_meter_concentration": float(m["imp_kwh"].max() / hh_imp) if hh_imp > 0 else np.nan,
        "w_n_heat_pump_like_meters": n_hp,
        "w_is_winter_dominant": int(n_hp > 0),
        "w_has_dedicated_hp_meter": int(n_meters > 1 and n_hp > 0 and (n_meters - n_hp) > 0),
        "w_n_production_only_meters": int(prod.sum()),
        "w_has_production_only_meter": int(prod.sum() > 0),
        "w_n_months_in_window": int(m["n_months"].max()),
    }


def manifest_aux(mm: pd.DataFrame) -> None:
    """Aux features on the *install-anchored* before/after windows.

    ``train_property_samples.csv`` pairs an ``augmented_before`` sample with the
    generic cross-sectional window, so its "after" is not the manifest's after.
    The manifest carries the real pairing (10 households, both periods anchored
    on ``install_date``), which is the correct like-for-like comparison.
    """
    if not MANIFEST_IN.exists():
        print(f"  ({MANIFEST_IN.name} missing - skipping)")
        return
    man = pd.read_csv(MANIFEST_IN, sep=C.CSV_SEP, dtype={"gp_nr": str})
    meters_per_gp = mm.groupby("gp_nr")["mp_id"].nunique()
    by_gp = dict(tuple(mm.groupby("gp_nr")))

    rows = []
    for r in man.itertuples():
        rec = {"sample_id": r.sample_id}
        sub = by_gp.get(r.gp_nr)
        if sub is not None:
            lo = pd.Period(pd.Timestamp(r.first_day), freq="M")
            hi = pd.Period(pd.Timestamp(r.last_day), freq="M")
            win = sub[(sub["period"] >= lo) & (sub["period"] <= hi)]
            rec.update(_aux_for_window(win, int(meters_per_gp.get(r.gp_nr, 1))))
        rows.append(rec)

    aux = pd.DataFrame(rows)
    for c in AUX_COLS:
        if c not in aux:
            aux[c] = np.nan
    out = man.merge(aux[["sample_id"] + AUX_COLS], on="sample_id", how="left")
    out.to_csv(MANIFEST_OUT, sep=C.CSV_SEP, index=False)
    print(f"\nwrote {MANIFEST_OUT}  ({len(out)} periods)")

    # like-for-like: does the seasonality ratio move after installation?
    key = "w_meter_max_winter_summer_ratio"
    piv = out.pivot_table(index=["gp_nr"], columns="period", values=key)
    lab = out[out.period == "after"].set_index("gp_nr")["label"]
    print(f"\n  {'gp_nr':>8} {'label':<12} {'before':>8} {'after':>8}   direction")
    moved = []
    for gp in piv.index:
        b = piv.at[gp, "before"] if "before" in piv else np.nan
        a = piv.at[gp, "after"] if "after" in piv else np.nan
        d = "n/a" if (pd.isna(b) or pd.isna(a)) else ("UP" if a > b else "down")
        if d != "n/a":
            moved.append((lab.get(gp, "?"), a > b))
        print(f"  {gp:>8} {str(lab.get(gp,'?'))[:11]:<12} {b:8.2f} {a:8.2f}   {d}")
    if moved:
        hp = [u for l, u in moved if l == "Wärmepumpe"]
        print(f"\n  comparable pairs: {len(moved)}/{len(piv)}"
              f"   ratio up: {sum(u for _, u in moved)}/{len(moved)}")
        if hp:
            print(f"  heat-pump installs with ratio up: {sum(hp)}/{len(hp)}")


def main() -> None:
    mm = _prepare_meter_months()
    tr = pd.read_csv(TRAIN_IN, sep=C.CSV_SEP, dtype={"gp_nr": str, "plz": str})
    print(f"samples: {len(tr)}   meter-months: {len(mm)}")

    meters_per_gp = mm.groupby("gp_nr")["mp_id"].nunique()
    by_gp = dict(tuple(mm.groupby("gp_nr")))

    rows = []
    for r in tr.itertuples():
        sub = by_gp.get(r.gp_nr)
        rec = {"sample_id": r.sample_id}
        if sub is not None:
            lo = pd.Period(pd.Timestamp(r.first_day), freq="M")
            hi = pd.Period(pd.Timestamp(r.last_day), freq="M")
            win = sub[(sub["period"] >= lo) & (sub["period"] <= hi)]
            rec.update(_aux_for_window(win, int(meters_per_gp.get(r.gp_nr, 1))))
        rows.append(rec)

    aux = pd.DataFrame(rows)
    for c in AUX_COLS:
        if c not in aux:
            aux[c] = np.nan
    merged = tr.merge(aux[["sample_id"] + AUX_COLS], on="sample_id", how="left")
    assert len(merged) == len(tr), "merge changed the row count"

    merged["w_n_meters"] = merged["w_n_meters"].fillna(1).astype(int)
    for c in ("w_is_multi_meter", "w_is_multi_unit", "w_is_winter_dominant",
              "w_has_dedicated_hp_meter", "w_has_production_only_meter",
              "w_n_heat_pump_like_meters", "w_n_production_only_meters",
              "w_n_meters_with_data", "w_n_months_in_window"):
        merged[c] = merged[c].fillna(0).astype(int)
    merged.to_csv(OUT, sep=C.CSV_SEP, index=False)

    print(f"\nwrote {OUT}  ({len(merged)} samples, +{len(AUX_COLS)} cols)")
    print(f"  samples with a windowed ratio : "
          f"{int(merged.w_meter_max_winter_summer_ratio.notna().sum())}/{len(merged)}")
    print(f"  median months in window       : {merged.w_n_months_in_window.median():.0f}")

    # does it now vary within a household's before/after pair?
    pairs = merged[merged.group_id.isin(
        merged.loc[merged.sample_type == "augmented_before", "group_id"])]
    print(f"\nbefore/after households: {pairs.group_id.nunique()}")
    for c in ("w_meter_max_winter_summer_ratio", "w_is_winter_dominant",
              "w_has_dedicated_hp_meter"):
        varies = pairs.groupby("group_id")[c].nunique(dropna=False)
        print(f"  {c:34s} varies within pair: {int((varies > 1).sum())}"
              f"/{pairs.group_id.nunique()}")

    manifest_aux(mm)


if __name__ == "__main__":
    main()
