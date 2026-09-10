"""Step 4 - MeteoSwiss daily weather for canton Aargau + a clear-sky-day flag.

Source: MeteoSwiss SwissMetNet open data (data.geo.admin.ch, CC-BY, no auth),
STAC collection ``ch.meteoschweiz.ogd-smn``. We use the five automatic stations
in canton AG and average them into one daily series - for a canton this small
the day-to-day radiation signal (what the PV midday-dip feature needs) barely
varies with location, so a single AG series is enough for the MVP. Per-PLZ
interpolation is a documented stretch goal.

Parameters: ``gre000d0`` global radiation daily mean [W/m2],
``tre200d0/dn/dx`` air temperature 2 m daily mean/min/max [degC].

Output: ``data_addition/weather_daily.csv``.
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from . import config as C

BASE = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
AG_STATIONS = ["bus", "bez", "lei", "moe", "psi"]  # all canton AG
PARAMS = ["gre000d0", "tre200d0", "tre200dn", "tre200dx"]
START, END = "2023-01-01", "2026-08-01"

_RAW_DIR = C.ARTIFACTS / "weather_raw"


def _download(station: str, period: str) -> pd.DataFrame:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = _RAW_DIR / f"ogd-smn_{station}_d_{period}.csv"
    if not cache.exists():
        url = f"{BASE}/{station}/ogd-smn_{station}_d_{period}.csv"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    df = pd.read_csv(cache, sep=";")
    df["date"] = pd.to_datetime(
        df["reference_timestamp"], format="%d.%m.%Y %H:%M", errors="coerce"
    ).dt.normalize()
    keep = ["date"] + [p for p in PARAMS if p in df.columns]
    return df[keep]


def load_station_series() -> pd.DataFrame:
    frames = []
    for st in AG_STATIONS:
        parts = [_download(st, p) for p in ("historical", "recent")]
        s = pd.concat(parts, ignore_index=True).dropna(subset=["date"])
        s = s[(s["date"] >= START) & (s["date"] < END)]
        s["station"] = st
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def build_daily() -> pd.DataFrame:
    s = load_station_series()
    daily = (
        s.groupby("date")
        .agg(
            global_radiation=("gre000d0", "mean"),
            temp_mean=("tre200d0", "mean"),
            temp_min=("tre200dn", "min"),
            temp_max=("tre200dx", "max"),
            n_stations=("station", "nunique"),
        )
        .reset_index()
        .sort_values("date")
    )
    # clear-sky flag: radiation in the top quartile of its calendar month
    daily["month"] = daily["date"].dt.month
    thr = daily.groupby("month")["global_radiation"].transform(lambda x: x.quantile(0.75))
    p95 = daily.groupby("month")["global_radiation"].transform(lambda x: x.quantile(0.95))
    daily["clear_sky_threshold"] = thr
    daily["is_clear_sky"] = daily["global_radiation"] >= thr
    daily["clearness"] = (daily["global_radiation"] / p95).clip(0, 1.5)
    return daily.drop(columns="month")


def main() -> None:
    daily = build_daily()
    C.WEATHER_DAILY.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(C.WEATHER_DAILY, index=False)
    print(f"wrote {C.WEATHER_DAILY}  ({len(daily)} days, "
          f"{daily['date'].min().date()}..{daily['date'].max().date()})")
    print(f"  stations/day: {daily['n_stations'].min()}-{daily['n_stations'].max()}")
    print(f"  clear-sky days: {daily['is_clear_sky'].mean():.1%}")
    print(daily.groupby(daily['date'].dt.month)[['global_radiation', 'is_clear_sky']]
          .mean().round(1).to_string())


if __name__ == "__main__":
    main()
