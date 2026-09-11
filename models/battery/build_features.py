"""Single-pass daily and monthly battery feature extraction.

Rows are parsed by position through the unchanged PV stream reader.  To keep
the pass tractable, the completed PV scan supplies a label-blind candidate
superset (any measured export, plus the documented missing-register fallback).
Exact positive-export day counts are recomputed here.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config as C
from .solar import sun_hours_vec
from ..pv.features.stream import iter_chunks
from ..pv.io import gp_to_mpid


def _candidate_meters() -> set[str] | None:
    prior = C.PV_METER_FLAGS.parent / "feedin_meter_stats.csv"
    if not prior.exists():
        return None
    s = pd.read_csv(prior, dtype={"mp_id": str})
    ids = set(s.loc[s["exp_kwh"] > 0, "mp_id"])
    if C.PV_OOF.exists():
        pv = pd.read_csv(C.PV_OOF, dtype={"gp_nr": str})
        gps = set(pv.loc[pv["pv_probability"] >= 0.8, "gp_nr"])
        missing_register = set(s.loc[s["exp_days"] == 0, "mp_id"])
        ids.update(
            gp_to_mpid().loc[lambda x: x.gp_nr.isin(gps) & x.mp_id.isin(missing_register), "mp_id"].dropna()
        )
    return ids


def _longest_true(mask: np.ndarray) -> tuple[int, int]:
    best_start = best_len = start = length = 0
    for i, value in enumerate(mask):
        if value:
            if length == 0:
                start = i
            length += 1
            if length > best_len:
                best_start, best_len = start, length
        else:
            length = 0
    return best_start, best_len


def _flat_run(v: np.ndarray, slots: np.ndarray | None = None) -> tuple[float, float]:
    if slots is None:
        slots = np.arange(96)
    x = v[slots]
    good = np.isfinite(x)
    stable = np.zeros(len(x), dtype=bool)
    if len(x) > 1:
        tol = np.maximum(0.05, 0.10 * np.maximum(np.abs(x[1:]), np.abs(x[:-1])))
        stable[1:] = good[1:] & good[:-1] & (np.abs(np.diff(x)) <= tol)
    covered = np.zeros(len(x), dtype=bool)
    i = 1
    while i < len(x):
        if not stable[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(x) and stable[j + 1]:
            j += 1
        if j - i + 2 >= 4:
            covered[i - 1:j + 1] = True
        i = j + 1
    vals = x[covered]
    mode = np.nan
    if len(vals):
        q = np.round((4 * vals) / 0.25) * 0.25
        mode = float(pd.Series(q).mode().iloc[0])
    return float(covered.mean()), mode


_SUN_CACHE: dict[pd.Timestamp, tuple[float, float]] = {}


def _safe_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.std(x)) if len(x) else np.nan


def _day_record(mp_id: str, plz: str, date: pd.Timestamp,
                imp: np.ndarray | None, exp: np.ndarray | None,
                is_clear: bool) -> dict:
    if date not in _SUN_CACHE:
        values = sun_hours_vec([date])
        _SUN_CACHE[date] = (float(values[0][0]), float(values[1][0]))
    rise, setting = _SUN_CACHE[date]
    rec = {"mp_id": mp_id, "plz": plz, "date": date, "year": date.year,
           "month": date.month, "is_clear": bool(is_clear),
           "has_both": imp is not None and exp is not None}
    if imp is not None:
        day = float(np.nansum(imp)); night = float(np.nansum(imp[4:16]))
        ss = min(95, max(0, int(np.ceil(setting * 4))))
        eve = imp[ss:]
        zero = np.isfinite(eve) & (eve < C.EPS)
        run = 0
        while run < len(zero) and zero[run]:
            run += 1
        step = np.nan
        threshold = 0.5 * night / 12
        for idx in range(ss + run, 96):
            if np.isfinite(imp[idx]) and imp[idx] > threshold:
                step = idx
                break
        rec.update(
            day_kwh=day, night_kwh=night,
            evening_zero_share=float(zero.mean()) if len(zero) else np.nan,
            zero_run_after_sunset=run / 4 if date.month in C.RUNLEN_MONTHS else np.nan,
            step_slot=step if date.month in C.RUNLEN_MONTHS else np.nan,
            step_magnitude=float(imp[int(step)]) if np.isfinite(step) and date.month in C.RUNLEN_MONTHS else np.nan,
            midday_import_share=float(np.nansum(imp[44:56]) / day) if day > 0 else np.nan,
            ramp_std_midday=_safe_std(np.diff(imp[44:60])),
            ramp_std_morning=_safe_std(np.diff(imp[28:40])),
        )
        if date.month in C.RUNLEN_MONTHS:
            night_slots = np.r_[88:96, 0:24]
            share, power = _flat_run(imp, night_slots)
            rec["night_flat_block_share"], rec["night_block_power"] = share, power
        else:
            rec["night_flat_block_share"] = rec["night_block_power"] = np.nan
    if exp is not None:
        total = float(np.nansum(exp)); positive = np.isfinite(exp) & (exp > C.EPS)
        inds = np.flatnonzero(positive)
        peak = float(np.nanmax(exp)) if np.isfinite(exp).any() else np.nan
        rec.update(
            exp_day_kwh=total, exp_peak_kw=4 * peak,
            exp_start_slot=float(inds[0]) if len(inds) else np.nan,
            exp_end_slot=float(inds[-1]) if len(inds) else np.nan,
            exp_peak_flatness=float(np.mean(exp[inds] >= 0.95 * peak)) if len(inds) and peak > 0 else np.nan,
        )
    if imp is not None and exp is not None:
        zi = np.isfinite(imp) & (imp < C.EPS)
        ze = np.isfinite(exp) & (exp < C.EPS)
        daylight = np.zeros(96, dtype=bool)
        daylight[max(0, int(np.floor(rise * 4))):min(96, int(np.ceil(setting * 4)))] = True
        both = zi & ze
        flat, mode = _flat_run(imp - exp) if date.month in C.RUNLEN_MONTHS else (np.nan, np.nan)
        rec.update(
            both_zero_share=float(both.mean()),
            both_zero_share_daylight=float(both[daylight].mean()) if daylight.any() else np.nan,
            flat_interval_share=flat, flat_interval_power_mode=mode,
            export_share_of_daylight_energy=float(np.nansum(exp[daylight]) / (np.nansum(exp[daylight]) + np.nansum(imp[44:56]))) if (np.nansum(exp[daylight]) + np.nansum(imp[44:56])) > 0 else np.nan,
        )
    return rec


def _aggregate_month(days: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c in days.select_dtypes(include="number").columns if c not in {"year", "month"}]
    spec = {f"{c}_mean": (c, "mean") for c in numeric}
    spec.update({f"{c}_std": (c, "std") for c in numeric})
    spec["n_days"] = ("date", "nunique")
    out = days.groupby(["mp_id", "year", "month"], as_index=False).agg(**spec)
    clear_cols = [c for c in ["evening_zero_share", "both_zero_share", "flat_interval_share", "exp_start_slot"] if c in days]
    if clear_cols:
        cl = days[days.is_clear].groupby(["mp_id", "year", "month"])[clear_cols].mean()
        cl.columns = [f"{c}_clear" for c in cl]
        out = out.merge(cl.reset_index(), on=["mp_id", "year", "month"], how="left")
    plz = days.groupby(["mp_id", "year", "month"])["plz"].first().rename("plz").reset_index()
    return out.merge(plz, on=["mp_id", "year", "month"], how="left")


def run(months: list[str] | None = None) -> None:
    weather = pd.read_csv(C.WEATHER_DAILY, parse_dates=["date"])
    clear_days = set(weather.loc[weather.is_clear_sky, "date"])
    exports = C.discover_monthly_exports()
    if months:
        exports = [e for e in exports if e.ym in set(months)]
    candidates = _candidate_meters()
    print(f"label-blind candidate meters: {len(candidates) if candidates is not None else 'all'}")
    monthly, daily_cp = [], []
    export_days: dict[str, set[pd.Timestamp]] = defaultdict(set)
    export_seen: set[str] = set()
    # Seed exact-zero/seen information from the already completed all-meter pass.
    prior_path = C.PV_METER_FLAGS.parent / "feedin_meter_stats.csv"
    prior = pd.read_csv(prior_path, dtype={"mp_id": str}) if prior_path.exists() else None
    date_cache: dict[str, pd.Timestamp] = {}
    def parsed(ds: str) -> pd.Timestamp:
        if ds not in date_cache:
            date_cache[ds] = pd.to_datetime(ds, format="%d.%m.%Y")
        return date_cache[ds]
    for e in exports:
        t0 = time.time(); pending: dict[tuple[str, str], dict] = {}
        for meta, values in iter_chunks(e.path):
            keep = meta.obis.isin([C.OBIS_IMPORT, C.OBIS_EXPORT])
            if candidates is not None:
                keep &= meta.mp_id.isin(candidates)
            for row, vec in zip(meta.loc[keep].itertuples(index=False), values[keep.to_numpy()]):
                key = (row.mp_id, row.date)
                slot = pending.setdefault(key, {"plz": row.plz})
                channel = "imp" if row.obis == C.OBIS_IMPORT else "exp"
                slot[channel] = vec.copy()
                if channel == "exp":
                    export_seen.add(row.mp_id)
                    if np.nansum(vec) > 0:
                        export_days[row.mp_id].add(parsed(row.date))
                if "imp" in slot and "exp" in slot:
                    date = parsed(row.date)
                    rec = _day_record(row.mp_id, slot["plz"], date, slot["imp"], slot["exp"], date in clear_days)
                    daily_cp.append({"mp_id": row.mp_id, "date": date, "evening_zero_share": rec.get("evening_zero_share")})
                    slot["record"] = rec
                    del slot["imp"], slot["exp"]
        records = []
        for (mp_id, ds), slot in pending.items():
            if "record" in slot:
                records.append(slot["record"])
            elif "imp" in slot or "exp" in slot:
                date = parsed(ds)
                records.append(_day_record(mp_id, slot["plz"], date, slot.get("imp"), slot.get("exp"), date in clear_days))
        days = pd.DataFrame(records)
        if len(days):
            monthly.append(_aggregate_month(days))
        print(f"{e.ym}: {len(days)} meter-days in {time.time()-t0:.1f}s")
    out = pd.concat(monthly, ignore_index=True)
    out.to_pickle(C.FEATURES_MONTHLY)
    pd.DataFrame(daily_cp).to_pickle(C.DAILY_FEATURES)
    rows = []
    all_ids = set(prior.mp_id) if prior is not None else (export_seen | set(export_days))
    prior_seen = set(prior.loc[prior.exp_days > 0, "mp_id"]) if prior is not None else set()
    for mp_id in all_ids:
        rows.append({"mp_id": mp_id, "exp_days_positive": len(export_days.get(mp_id, set())),
                     "exp_rows_seen": mp_id in export_seen or mp_id in prior_seen})
    pd.DataFrame(rows).to_csv(C.METER_EXPORT_DAYS, index=False)
    print(f"wrote {C.FEATURES_MONTHLY} ({len(out)} rows), {C.METER_EXPORT_DAYS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*")
    run(ap.parse_args().months)


if __name__ == "__main__":
    main()
