"""Steps 2 & 3 (post-processing) - silver PV labels + production-meter flags.

Reads ``feedin_meter_stats.csv`` (written by ``build_features``) and:

* flags **production meters** - export-dominant metering points with no
  night base load, i.e. a dedicated PV meter rather than a household supply
  meter. These are excluded from consumption-feature aggregation (Step 6).
* derives **silver positives** - meters with a clear, sustained, midday-centred
  summer feed-in pattern => the household has PV. Households that are currently
  unlabelled and silver-positive are appended to ``gigi_augmented.csv`` with
  ``label_source = silver_feedin``. No silver negatives (a meter with no
  feed-in may still host a fully self-consuming PV system).

Outputs: ``artifacts/meter_flags.csv`` and an updated ``gigi_augmented.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import gp_to_mpid

MOD_FLAGS = C.ARTIFACTS / "meter_flags.csv"

# production meter: lots of export, almost no import, no night base load
PROD_MIN_EXP_KWH = 100.0
PROD_MAX_IMP_RATIO = 0.10
PROD_MAX_NIGHT_KWH_PER_DAY = 0.02

# silver positive: a full-ish summer of frequent midday feed-in, peak near noon
SILVER_MIN_SUMMER_DAYS = 80
SILVER_MIN_MIDDAY_DAYS = 40
SILVER_PEAK_SLOT_LO, SILVER_PEAK_SLOT_HI = 38, 58   # ~09:45 .. 14:45 local


def classify_meters() -> pd.DataFrame:
    fi = pd.read_csv(C.FEEDIN_METER_STATS, dtype={"mp_id": str})
    imp_night_per_day = fi["imp_night_kwh"] / fi["imp_days"].clip(lower=1)

    fi["is_production_meter"] = (
        (fi["exp_kwh"] > PROD_MIN_EXP_KWH)
        & (fi["imp_kwh"] < PROD_MAX_IMP_RATIO * fi["exp_kwh"])
        & (imp_night_per_day < PROD_MAX_NIGHT_KWH_PER_DAY)
    )
    fi["is_silver_pv"] = (
        (fi["best_summer_days"] >= SILVER_MIN_SUMMER_DAYS)
        & (fi["best_summer_midday_days"] >= SILVER_MIN_MIDDAY_DAYS)
        & fi["peak_slot_med"].between(SILVER_PEAK_SLOT_LO, SILVER_PEAK_SLOT_HI)
        & ~fi["is_production_meter"]
    )
    return fi


def silver_positive_households(fi: pd.DataFrame) -> pd.DataFrame:
    chain = gp_to_mpid()[["gp_nr", "mp_id"]].dropna().drop_duplicates()
    plz = (
        pd.read_pickle(C.FEATURES_MONTHLY)
        .groupby("mp_id")["plz"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "")
    )
    silver_mp = fi.loc[fi["is_silver_pv"], "mp_id"]
    gp = chain[chain["mp_id"].isin(silver_mp)].copy()
    gp["plz"] = gp["mp_id"].map(plz).fillna("")
    agg = (
        gp.groupby("gp_nr")
        .agg(mp_id=("mp_id", "first"), plz=("plz", lambda s: s.mode().iat[0] if not s.mode().empty else ""))
        .reset_index()
    )
    return agg


def main() -> None:
    fi = classify_meters()
    fi[["mp_id", "is_production_meter", "is_silver_pv", "exp_kwh", "imp_kwh",
        "best_summer_midday_days", "peak_slot_med"]].to_csv(MOD_FLAGS, index=False)
    print(f"wrote {MOD_FLAGS}")
    print(f"  production meters : {int(fi['is_production_meter'].sum())}")
    print(f"  silver-PV meters  : {int(fi['is_silver_pv'].sum())}")

    labels = pd.read_csv(C.GIGI_AUGMENTED, dtype={"mp_id": str, "gp_nr": str, "plz": str})
    labels = labels[labels["label_source"] != "silver_feedin"]  # idempotent
    labelled_gp = set(labels.loc[labels["pv_label"].notna(), "gp_nr"])

    silver = silver_positive_households(fi)
    new = silver[~silver["gp_nr"].isin(labelled_gp)].copy()
    new["pv_label"] = 1
    new["label_source"] = "silver_feedin"
    new["has_meter"] = True
    for col in labels.columns:
        if col not in new.columns:
            new[col] = pd.NA

    out = pd.concat([labels, new[labels.columns]], ignore_index=True)
    out.to_csv(C.GIGI_AUGMENTED, index=False)

    gigi_pos = labels.loc[labels["pv_label"] == 1, "gp_nr"].nunique()
    print(f"\nlabels: {gigi_pos} GIGI positive households"
          f"  + {new['gp_nr'].nunique()} new silver-positive households")
    print(f"  silver positives also in GIGI (already labelled): "
          f"{silver['gp_nr'].isin(labelled_gp).sum()}")
    print(f"wrote {C.GIGI_AUGMENTED}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
