"""Step 8 - per-PLZ rooftop-PV base rate, for PU calibration and sanity checks.

Numerator:  BFE ElPA register (data.geo.admin.ch, CC-BY, Pronovo-fed) -
            photovoltaic plants in canton AG with TotalPower <= 30 kW
            (residential scale), counted per postcode.
Denominator: distinct metering points per postcode in our own population
            (one recent monthly export), i.e. the share of *AEW-metered*
            households with a registered PV plant. This is the right
            denominator for calibrating predictions on this population and
            sidesteps a separate BFS households table.

Caveat: ElPA misses small PV without a guarantee of origin -> the rate is a
lower bound. Downstream code sensitivity-tests x1.0-1.5.

Output: ``data_addition/pv_baserate_plz.csv``.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from . import config as C
from .stream import iter_chunks

ELPA_URL = (
    "https://data.geo.admin.ch/ch.bfe.elektrizitaetsproduktionsanlagen/"
    "elektrizitaetsproduktionsanlagen/elektrizitaetsproduktionsanlagen_2056.csv.zip"
)
_ELPA_CACHE = C.ARTIFACTS / "elpa_raw.zip"
MAX_RESIDENTIAL_KWP = 30.0
DENOM_EXPORT_YM = "2026-07"  # most complete month (all ~90k meters)


def load_elpa_pv_ag() -> pd.DataFrame:
    if not _ELPA_CACHE.exists():
        resp = requests.get(ELPA_URL, timeout=300)
        resp.raise_for_status()
        _ELPA_CACHE.write_bytes(resp.content)
    with zipfile.ZipFile(_ELPA_CACHE) as z:
        df = pd.read_csv(z.open("ElectricityProductionPlant.csv"))
    pv = df[(df["SubCategory"] == "subcat_2") & (df["Canton"] == "AG")].copy()
    pv["PostCode"] = pv["PostCode"].astype("Int64").astype(str)
    pv["kwp"] = pd.to_numeric(pv["TotalPower"], errors="coerce")
    return pv


def meters_per_plz(ym: str = DENOM_EXPORT_YM) -> pd.Series:
    export = next(e for e in C.discover_monthly_exports() if e.ym == ym)
    seen: dict[str, set[str]] = {}
    for meta, _vals in iter_chunks(export.path, obis=C.OBIS_IMPORT):
        for plz, mp in zip(meta["plz"], meta["mp_id"]):
            seen.setdefault(str(plz), set()).add(mp)
    return pd.Series({plz: len(s) for plz, s in seen.items()}, name="n_meters")


def build() -> pd.DataFrame:
    pv = load_elpa_pv_ag()
    n_plants = (
        pv[pv["kwp"] <= MAX_RESIDENTIAL_KWP]
        .groupby("PostCode")
        .size()
        .rename("n_pv_plants")
    )
    n_meters = meters_per_plz()

    out = pd.concat([n_plants, n_meters], axis=1).fillna(0)
    out.index.name = "plz"
    out = out.reset_index()
    out = out[out["n_meters"] > 0].copy()

    ag_rate = out["n_pv_plants"].sum() / out["n_meters"].sum()
    # Beta-Binomial shrinkage toward the AG mean for thinly-metered postcodes
    prior_strength = 50.0
    out["pv_rate_raw"] = out["n_pv_plants"] / out["n_meters"]
    out["pv_rate"] = (
        (out["n_pv_plants"] + prior_strength * ag_rate)
        / (out["n_meters"] + prior_strength)
    ).clip(0, 0.95)
    out.attrs["ag_rate"] = ag_rate
    return out


def main() -> None:
    out = build()
    C.PV_BASERATE_PLZ.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.PV_BASERATE_PLZ, index=False)
    ag = out.attrs["ag_rate"]
    print(f"wrote {C.PV_BASERATE_PLZ}  ({len(out)} postcodes)")
    print(f"  canton-AG PV rate (plants / metered households): {ag:.1%}")
    print(f"  per-PLZ pv_rate: p10={out['pv_rate'].quantile(.1):.1%} "
          f"median={out['pv_rate'].median():.1%} p90={out['pv_rate'].quantile(.9):.1%}")
    print(f"  total plants={int(out['n_pv_plants'].sum())} "
          f"total meters={int(out['n_meters'].sum())}")


if __name__ == "__main__":
    main()
