"""Step 5 - single streaming pass over all 42 monthly exports.

Produces two artifacts in one ~77 GB read:

* ``features_monthly.pkl`` - one row per ``(mp_id, year, month)`` of
  consumption-channel (OBIS 1.29) features. The feed-in channel is **never**
  used here.
* ``feedin_meter_stats.csv`` - per ``mp_id`` export + import totals and
  per-summer feed-in counts, used offline by ``extract_feedin_labels`` for
  silver positives and the production-meter filter.

Run ``python -m newspaper.models.build_features --months 2025-07 2025-01`` for a
quick smoke test before the full pass.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import config as C
from .stream import iter_chunks, hour_band_sum

EPS = 1e-6
_MIDDAY_H = len(list(C.MIDDAY_HOURS))     # 3
_NIGHT_H = len(list(C.NIGHT_HOURS))       # 3
_EVENING_H = len(list(C.EVENING_HOURS))   # 3
_SOLAR_SLOTS = [s for h in C.SOLAR_HOURS for s in range(h * 4, h * 4 + 4)]
_FEEDIN_MIDDAY = range(10, 15)            # wider window for export peak


# --------------------------------------------------------------------------- #
# per-chunk day-level feature frames
# --------------------------------------------------------------------------- #
def _import_day_frame(meta: pd.DataFrame, v: np.ndarray) -> pd.DataFrame:
    day = np.nansum(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        midday_p = hour_band_sum(v, C.MIDDAY_HOURS) / _MIDDAY_H
        night_p = hour_band_sum(v, C.NIGHT_HOURS) / _NIGHT_H
        evening_p = hour_band_sum(v, C.EVENING_HOURS) / _EVENING_H
        day_p = day / 24.0
        midday_ratio = np.where(day_p > EPS, midday_p / day_p, np.nan)
        evening_midday = np.where(midday_p > EPS, evening_p / midday_p, np.nan)
    solar = v[:, _SOLAR_SLOTS]
    daytime_min = np.nanmin(np.where(np.isnan(solar), np.inf, solar), axis=1)
    daytime_min[~np.isfinite(daytime_min)] = np.nan
    frac_daytime_zero = np.nanmean((solar < 0.01).astype(np.float32), axis=1)
    below_night = (daytime_min < (night_p / 4.0)).astype(np.float32)  # per-qh base

    d = pd.to_datetime(meta["date"], format="%d.%m.%Y", errors="coerce")
    return pd.DataFrame(
        {
            "mp_id": meta["mp_id"].to_numpy(),
            "plz": meta["plz"].to_numpy(),
            "date": d.to_numpy(),
            "is_weekend": (d.dt.dayofweek >= 5).to_numpy(),
            "day_kwh": day,
            "midday_ratio": midday_ratio,
            "evening_midday_ratio": evening_midday,
            "night_kwh": hour_band_sum(v, C.NIGHT_HOURS),
            "solar_kwh": hour_band_sum(v, C.SOLAR_HOURS),
            "frac_daytime_zero": frac_daytime_zero,
            "below_night": below_night,
        }
    )


def _export_day_frame(meta: pd.DataFrame, v: np.ndarray) -> pd.DataFrame:
    day = np.nansum(v, axis=1)
    midday = hour_band_sum(v, _FEEDIN_MIDDAY)
    allnan = np.all(np.isnan(v), axis=1)
    filled = np.where(np.isnan(v), -np.inf, v)
    peak_slot = np.where(allnan | (day <= EPS), np.nan, np.argmax(filled, axis=1))
    return pd.DataFrame(
        {
            "mp_id": meta["mp_id"].to_numpy(),
            "exp_day_kwh": day,
            "exp_midday_kwh": midday,
            "peak_slot": peak_slot,
        }
    )


# --------------------------------------------------------------------------- #
# per-file aggregation
# --------------------------------------------------------------------------- #
def _agg_import_month(df: pd.DataFrame, year: int, month: int,
                      clear_days: set) -> pd.DataFrame:
    df = df[df["mp_id"].notna()].copy()
    df["is_clear"] = df["date"].isin(clear_days)
    g = df.groupby("mp_id")
    q = g["day_kwh"].quantile([0.1, 0.5, 0.9]).unstack()
    clear = df[df["is_clear"]].groupby("mp_id")["midday_ratio"].mean()
    wk = df.groupby(["mp_id", "is_weekend"])["day_kwh"].mean().unstack()
    out = pd.DataFrame({
        "year": year,
        "month": month,
        "n_days": g.size(),
        "day_kwh_mean": g["day_kwh"].mean(),
        "day_kwh_p10": q[0.1],
        "day_kwh_p50": q[0.5],
        "day_kwh_p90": q[0.9],
        "midday_ratio_mean": g["midday_ratio"].mean(),
        "midday_ratio_std": g["midday_ratio"].std(),
        "midday_ratio_clear": clear,
        "evening_midday_ratio_mean": g["evening_midday_ratio"].mean(),
        "night_kwh_mean": g["night_kwh"].mean(),
        "frac_daytime_zero_mean": g["frac_daytime_zero"].mean(),
        "below_night_frac": g["below_night"].mean(),
        "day_kwh_weekday": wk.get(False),
        "day_kwh_weekend": wk.get(True),
        "plz": g["plz"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else ""),
    })
    return out.reset_index()


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(months: list[str] | None = None) -> None:
    weather = pd.read_csv(C.WEATHER_DAILY, parse_dates=["date"])
    clear_days = set(weather.loc[weather["is_clear_sky"], "date"].to_numpy())

    exports = C.discover_monthly_exports()
    if months:
        exports = [e for e in exports if e.ym in set(months)]

    monthly_rows: list[pd.DataFrame] = []
    feedin: dict[str, dict] = {}      # mp_id -> running export/import stats
    feedin_year: dict[tuple, dict] = {}  # (mp_id, year) -> summer feed-in counts

    for e in exports:
        t0 = time.time()
        imp_parts, exp_parts = [], []
        for meta, v in iter_chunks(e.path):
            is_imp = meta["obis"] == C.OBIS_IMPORT
            is_exp = meta["obis"] == C.OBIS_EXPORT
            if is_imp.any():
                imp_parts.append(_import_day_frame(meta[is_imp].reset_index(drop=True),
                                                   v[is_imp.to_numpy()]))
            if is_exp.any():
                exp_parts.append(_export_day_frame(meta[is_exp].reset_index(drop=True),
                                                   v[is_exp.to_numpy()]))

        # ---- monthly import features ----
        if imp_parts:
            imp = pd.concat(imp_parts, ignore_index=True)
            monthly_rows.append(_agg_import_month(imp, e.year, e.month, clear_days))
            # feed the import side of the meter-type stats
            gi = imp.groupby("mp_id").agg(imp_kwh=("day_kwh", "sum"),
                                          imp_days=("day_kwh", "size"),
                                          imp_night_kwh=("night_kwh", "sum"))
            for mp, r in gi.iterrows():
                d = feedin.setdefault(mp, {})
                d["imp_kwh"] = d.get("imp_kwh", 0.0) + r.imp_kwh
                d["imp_days"] = d.get("imp_days", 0) + int(r.imp_days)
                d["imp_night_kwh"] = d.get("imp_night_kwh", 0.0) + r.imp_night_kwh

        # ---- feed-in (export) stats ----
        if exp_parts:
            exp = pd.concat(exp_parts, ignore_index=True)
            ge = exp.groupby("mp_id").agg(
                exp_kwh=("exp_day_kwh", "sum"),
                exp_days=("exp_day_kwh", "size"),
                exp_midday_days=("exp_midday_kwh", lambda s: int((s > 0.5).sum())),
                peak_slot_med=("peak_slot", "median"),
            )
            for mp, r in ge.iterrows():
                d = feedin.setdefault(mp, {})
                d["exp_kwh"] = d.get("exp_kwh", 0.0) + r.exp_kwh
                d["exp_days"] = d.get("exp_days", 0) + int(r.exp_days)
                d["exp_midday_days"] = d.get("exp_midday_days", 0) + int(r.exp_midday_days)
                d.setdefault("peak_slots", []).append(r.peak_slot_med)
                if e.month in C.SUMMER_MONTHS:
                    y = feedin_year.setdefault((mp, e.year),
                                               {"days": 0, "midday_days": 0})
                    y["days"] += int(r.exp_days)
                    y["midday_days"] += int(r.exp_midday_days)

        print(f"  {e.ym}: {time.time() - t0:5.1f}s  "
              f"imp_rows={sum(len(p) for p in imp_parts):>8}  "
              f"exp_rows={sum(len(p) for p in exp_parts):>8}")

    # ---- write monthly features ----
    feats = pd.concat(monthly_rows, ignore_index=True)
    feats.to_pickle(C.FEATURES_MONTHLY)
    print(f"\nwrote {C.FEATURES_MONTHLY}  "
          f"({len(feats)} rows, {feats['mp_id'].nunique()} meters)")

    # ---- write feed-in / meter-type stats ----
    best_summer = {}
    for (mp, _y), y in feedin_year.items():
        cur = best_summer.get(mp, (0, 0))
        if y["midday_days"] > cur[1]:
            best_summer[mp] = (y["days"], y["midday_days"])
    rows = []
    for mp, d in feedin.items():
        bs = best_summer.get(mp, (0, 0))
        peaks = [p for p in d.get("peak_slots", []) if p == p]
        rows.append({
            "mp_id": mp,
            "imp_kwh": d.get("imp_kwh", 0.0),
            "imp_days": d.get("imp_days", 0),
            "imp_night_kwh": d.get("imp_night_kwh", 0.0),
            "exp_kwh": d.get("exp_kwh", 0.0),
            "exp_days": d.get("exp_days", 0),
            "exp_midday_days": d.get("exp_midday_days", 0),
            "best_summer_days": bs[0],
            "best_summer_midday_days": bs[1],
            "peak_slot_med": float(np.median(peaks)) if peaks else np.nan,
        })
    fi = pd.DataFrame(rows)
    fi.to_csv(C.FEEDIN_METER_STATS, index=False)
    print(f"wrote {C.FEEDIN_METER_STATS}  ({len(fi)} meters)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", help="e.g. 2025-07 2025-01 (default: all)")
    run(ap.parse_args().months)
