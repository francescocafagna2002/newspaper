"""Streaming reader for the monthly smart-meter exports.

The files are 1-3 GB each; we never load one whole. ``iter_chunks`` yields
``(meta, values)`` where ``meta`` is a small DataFrame (mp_id, obis, date, plz)
and ``values`` is a float32 ``(n, 96)`` array of quarter-hour kWh readings.

We parse **by column position**, not by header name: the ``April 2023`` export
ships a header that is missing the ``OBIS-Code`` label, which would shift every
column if we trusted the names. Every data row is
``MP ID ; OBIS-Code ; Datum ; PLZ ; <96 values> [; trailing empty]``.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from .. import config as C

_NAMES = ["mp_id", "obis", "date", "plz", *C.QUARTER_HOUR_COLS]
_USECOLS = list(range(100))  # 0..3 meta + 4..99 the 96 quarter-hours


def iter_chunks(
    path, chunksize: int = 150_000, obis: str | None = None
) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
    reader = pd.read_csv(
        path,
        sep=C.CSV_SEP,
        encoding="utf-8",
        header=0,               # skip the (sometimes malformed) header row
        names=_NAMES,
        usecols=_USECOLS,
        index_col=False,
        dtype={"mp_id": str, "obis": str, "date": str, "plz": str},
        chunksize=chunksize,
    )
    for chunk in reader:
        if obis is not None:
            chunk = chunk[chunk["obis"] == obis]
            if chunk.empty:
                continue
        meta = chunk[["mp_id", "obis", "date", "plz"]].reset_index(drop=True)
        values = (
            chunk[C.QUARTER_HOUR_COLS]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        yield meta, values


def hour_band_sum(values: np.ndarray, hours) -> np.ndarray:
    """Sum the quarter-hour slots inside the given clock hours.

    Clock hour ``h`` is the 0-based slot range ``[4h, 4h+4)`` (slot k, 1-based,
    holds energy for the 15 min ending at ``k*15`` minutes past midnight).
    """
    idx = [s for h in hours for s in range(h * 4, h * 4 + 4) if s < 96]
    return np.nansum(values[:, idx], axis=1)
