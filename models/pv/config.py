"""Paths, constants and monthly-export discovery.

Everything downstream imports from here so there is a single place that knows
where the data lives and which files are canonical.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Roots
# --------------------------------------------------------------------------- #
WORK = Path(os.environ.get("PV_WORK_ROOT", "/home/renku/work"))
NEWSPAPER = WORK / "newspaper"
INPUT_DATA = WORK / "store" / "input_data"

DATA_ADDITION = NEWSPAPER / "data_addition"
DOCS = NEWSPAPER / "docs"
ARTIFACTS = NEWSPAPER / "models" / "pv" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reference tables
# --------------------------------------------------------------------------- #
GIGI_RAW = INPUT_DATA / "HackDays2026 - GIGI.csv"
GIGI_ANNOTATED = DATA_ADDITION / "HackDays2026 - GIGI - annotated.csv"
MPID_MAPPING = INPUT_DATA / "mpid_zähler_mapping.csv"
ZAEHLER_GP = INPUT_DATA / "Zähler-GP.csv"

# Labels / intermediate artifacts written by the pipeline
GIGI_AUGMENTED = DATA_ADDITION / "gigi_augmented.csv"
WEATHER_DAILY = DATA_ADDITION / "weather_daily.csv"
PV_BASERATE_PLZ = DATA_ADDITION / "pv_baserate_plz.csv"

FEATURES_MONTHLY = ARTIFACTS / "features_monthly.pkl"
FEATURES_GP = ARTIFACTS / "features_gp.pkl"
FEEDIN_METER_STATS = ARTIFACTS / "feedin_meter_stats.csv"
MODEL_DIR = ARTIFACTS / "model"
PREDICTIONS = ARTIFACTS / "pv_predictions.csv"

CSV_SEP = ";"
CSV_ENCODING = "utf-8-sig"  # handles the BOM on GIGI / Zähler-GP

SEED = 20260910

# --------------------------------------------------------------------------- #
# Monthly smart-meter exports
# --------------------------------------------------------------------------- #
GERMAN_MONTHS = {
    "Januar": 1, "Februar": 2, "März": 3, "April": 4, "Mai": 5, "Juni": 6,
    "Juli": 7, "August": 8, "September": 9, "Oktober": 10, "November": 11,
    "Dezember": 12,
}
_MONTH_DIR_RE = re.compile(r"^(" + "|".join(GERMAN_MONTHS) + r")\s+(\d{4})$")

# OBIS registers (one row each per meter-day)
OBIS_IMPORT = "1-1:1.29.0*255"   # active energy drawn from the grid (consumption)
OBIS_EXPORT = "1-1:2.29.0*255"   # active energy fed into the grid (PV / battery)

# 96 quarter-hour columns as they appear in the export header:
# "00:15", "00:30", "00:45", "01:00", ... "23:45", "00:00" (the last = 24:00).
# Column k (1-based) holds the energy for the 15 min ending at 00:00 + k*15 min.
QUARTER_HOUR_COLS = [
    f"{(k * 15 // 60) % 24:02d}:{k * 15 % 60:02d}" for k in range(1, 97)
]
assert len(QUARTER_HOUR_COLS) == 96
assert QUARTER_HOUR_COLS[:4] == ["00:15", "00:30", "00:45", "01:00"]
assert QUARTER_HOUR_COLS[-1] == "00:00"
# clock hour h (0-23) is the 0-based slot range [h*4, h*4+4)

SUMMER_MONTHS = (5, 6, 7, 8)
WINTER_MONTHS = (11, 12, 1, 2)
SOLAR_HOURS = range(9, 16)     # 09:00-16:00 local, feed-in / dip window
MIDDAY_HOURS = range(11, 14)   # 11:00-14:00, core midday-dip window
NIGHT_HOURS = range(1, 4)      # 01:00-04:00, base-load reference
EVENING_HOURS = range(18, 21)  # 18:00-21:00, duck-curve peak


@dataclass(frozen=True)
class MonthlyExport:
    path: Path
    year: int
    month: int

    @property
    def ym(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def discover_monthly_exports(
    root: Path = INPUT_DATA, include_stalled_uploads: bool = True
) -> list[MonthlyExport]:
    """Return the canonical monthly exports, one per (year, month).

    Only files whose *parent directory* is a German "<Month> <Year>" folder are
    kept; a flat copy such as ``2023/LG_...csv`` (same basename, no month dir)
    is ignored. If both variants exist the month-dir version wins.

    ``include_stalled_uploads`` also accepts ``*.csv.tmp.*`` files for months
    that have no ``.csv`` yet -- the share is written to during the event and an
    upload can stall mid-rename. Verify such a file before trusting it.
    """
    found: dict[tuple[int, int], MonthlyExport] = {}
    stalled: dict[tuple[int, int], MonthlyExport] = {}
    for dirpath, _dirs, files in os.walk(root):
        m = _MONTH_DIR_RE.match(os.path.basename(dirpath))
        if not m:
            continue
        month = GERMAN_MONTHS[m.group(1)]
        year = int(m.group(2))
        for fn in files:
            if not fn.startswith("LG_AIM2Hackerdays_kWh_"):
                continue
            if fn.endswith(".csv"):
                found[(year, month)] = MonthlyExport(Path(dirpath) / fn, year, month)
            elif include_stalled_uploads and ".csv.tmp." in fn:
                # An abandoned server-side upload. 2023-03 sat like this for
                # hours; the file is complete (uniform 101-field rows, both
                # OBIS registers, all 31 days) but was never renamed to .csv.
                stalled[(year, month)] = MonthlyExport(Path(dirpath) / fn, year, month)

    for key, export in stalled.items():
        found.setdefault(key, export)   # a real .csv always wins
    return sorted(found.values(), key=lambda e: (e.year, e.month))


# Known data-quality caveats (see docs/data_problems.md)
KNOWN_MISSING_MONTHS: set[str] = set()      # 2023-03 recovered from a stalled
                                            # .csv.tmp upload (see discover fn)
KNOWN_PARTIAL_MONTHS = {"2024-10"}          # ~32k meters vs ~47k neighbours
KNOWN_BAD_HEADER_MONTHS = {"2023-04"}       # header missing the OBIS-Code label
                                            # -> stream.py parses by position


if __name__ == "__main__":
    exports = discover_monthly_exports()
    print(f"{len(exports)} monthly exports")
    for e in exports:
        size_gb = e.path.stat().st_size / 1e9
        print(f"  {e.ym}  {size_gb:5.2f} GB  {e.path.relative_to(INPUT_DATA)}")
    missing = KNOWN_MISSING_MONTHS - {e.ym for e in exports}
    print("expected-missing:", KNOWN_MISSING_MONTHS, "| not found:", missing)
