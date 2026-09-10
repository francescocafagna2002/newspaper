"""Per-meter monthly aggregates for the *labelled* households only.

Aggregating a household's meters to GP-Nr (see ``build_features_agg`` and
``load_property_daily.csv``) is correct -- the GIGI label describes the
customer, not the meter -- but it dilutes two signals that live in a single
meter of a multi-meter household:

  * a **separate heat-pump meter** (Wärmepumpentarif): winter-only, near-zero
    in summer.  Summed into the household total it becomes a mild winter bump.
  * a **dedicated PV production point**: zero import, pure export.

This step keeps the per-meter view for the ~414 labelled meters so
``build_property_aux`` can carry those signals alongside the aggregate.

Only labelled meters are read, but every monthly export is streamed, so this
is one full pass over the 77 GB (~15-20 min).

Output: ``data_addition/meter_month_labelled.csv``
        one row per (mp_id, year, month, direction)
"""
from __future__ import annotations

import collections
import time

import numpy as np
import pandas as pd

from . import config as C
from .stream import iter_chunks

OUT = C.DATA_ADDITION / "meter_month_labelled.csv"

_DIRECTION = {C.OBIS_IMPORT: "bezug", C.OBIS_EXPORT: "einsp"}


def labelled_meters() -> set[str]:
    """MP IDs of the **GIGI** households (the ones with technology labels).

    ``gigi_augmented.csv`` also carries ~6.2k ``silver_feedin`` households that
    were weak-labelled from measured export; those are all single-meter, so the
    per-meter view adds nothing for them and they are excluded here.
    """
    from .io_utils import read_gigi, gp_to_mpid

    gigi = set(read_gigi(annotated=True)["gp_nr"].dropna())
    gigi.discard("")
    chain = gp_to_mpid()
    mp = chain.loc[chain["gp_nr"].isin(gigi) & chain["mp_id"].notna(), "mp_id"]
    return set(mp.str.strip()) - {""}


def run() -> pd.DataFrame:
    want = labelled_meters()
    print(f"labelled meters to extract: {len(want)}")

    exports = C.discover_monthly_exports()
    print(f"monthly exports found: {len(exports)}")

    # (mp_id, year, month, direction) -> [kwh, n_days, midday_kwh, night_kwh]
    acc: dict[tuple, np.ndarray] = collections.defaultdict(
        lambda: np.zeros(4, dtype=np.float64)
    )
    plz_of: dict[str, str] = {}

    for ex in exports:
        if ex.ym in C.KNOWN_PARTIAL_MONTHS:
            print(f"  {ex.ym}: SKIPPED (known partial month)")
            continue
        t0 = time.time()
        n_rows = 0
        for meta, values in iter_chunks(ex.path):
            keep = meta["mp_id"].isin(want).to_numpy()
            if not keep.any():
                continue
            meta = meta[keep]
            values = values[keep]
            n_rows += len(meta)

            day = np.nansum(values, axis=1)
            midday = np.nansum(
                values[:, [s for h in C.MIDDAY_HOURS for s in range(h * 4, h * 4 + 4)]],
                axis=1)
            night = np.nansum(
                values[:, [s for h in C.NIGHT_HOURS for s in range(h * 4, h * 4 + 4)]],
                axis=1)

            for mp, ob, plz, d, mid, ni in zip(
                meta["mp_id"].to_numpy(), meta["obis"].to_numpy(),
                meta["plz"].to_numpy(), day, midday, night,
            ):
                direction = _DIRECTION.get(ob)
                if direction is None:
                    continue
                acc[(mp, ex.year, ex.month, direction)] += (d, 1.0, mid, ni)
                if mp not in plz_of and isinstance(plz, str):
                    plz_of[mp] = plz
        print(f"  {ex.ym}: {n_rows:7d} labelled rows  ({time.time() - t0:5.1f}s)")

    recs = [
        dict(mp_id=mp, year=y, month=mo, direction=dr,
             kwh=v[0], n_days=int(v[1]), midday_kwh=v[2], night_kwh=v[3],
             plz=plz_of.get(mp, ""))
        for (mp, y, mo, dr), v in acc.items()
    ]
    df = pd.DataFrame.from_records(recs).sort_values(
        ["mp_id", "year", "month", "direction"]).reset_index(drop=True)
    df.to_csv(OUT, sep=C.CSV_SEP, index=False)
    print(f"\nwrote {OUT}  ({len(df)} rows, {df.mp_id.nunique()} meters)")
    return df


def main() -> None:
    run()


if __name__ == "__main__":
    main()
