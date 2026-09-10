"""Split the feed-in channel out of the training tables.

``docs/data_problems.md`` block ④ records the decision: the challenge asks to
infer behind-the-meter assets from the **net load**, so the export register
(OBIS ``1-1:2.29.0*255``) must not reach model features. It stays available
offline for silver-positive labels, the export-dominant meter filter, and
evaluation.

``train_property_samples*.csv`` violated that -- it carries ``einsp_*``,
``frac_days_feedin`` and the ``he00``-``he23`` export hour-bands, which
separate PV almost perfectly (mean 27.5 vs 12.4 kWh/day). Training on them
yields a near-perfect classifier that answers the wrong question.

This step writes, for each input table:
  * ``*_netload.csv``  -- safe to train on: identifiers, labels, net-load features
  * ``*_feedin.csv``   -- the quarantined export columns, keyed by ``sample_id``

Output names are derived from the input, e.g.
``train_property_samples_auxw.csv`` -> ``..._auxw_netload.csv`` / ``..._auxw_feedin.csv``
"""
from __future__ import annotations

import pandas as pd

from . import config as C

INPUTS = [
    C.DATA_ADDITION / "train_property_samples_auxw.csv",
    C.DATA_ADDITION / "train_property_samples_aux.csv",
]

# Columns computed from the export register. Keep this list explicit: a silent
# addition here is a silent leak.
FEEDIN_PREFIXES = ("einsp_", "he")
FEEDIN_EXACT = {
    "frac_days_feedin",
    "n_production_only_meters", "has_production_only_meter",
    "w_n_production_only_meters", "w_has_production_only_meter",
    "household_exp_kwh",
}
# Identifiers and labels are copied into BOTH outputs so each stands alone.
ID_COLS = ["sample_id", "group_id", "gp_nr", "plz", "sample_type",
           "window_anchored_to_install", "n_days", "first_day", "last_day"]
LABEL_PREFIX = "has_"
# ...but these has_* columns are features, not labels.
NOT_LABELS = {"has_dedicated_hp_meter", "has_production_only_meter",
              "w_has_dedicated_hp_meter", "w_has_production_only_meter"}


def is_feedin(col: str) -> bool:
    if col in FEEDIN_EXACT:
        return True
    if col.startswith("einsp_"):
        return True
    # he00..he23 are export hour-bands; hb00..hb23 are import and are fine
    return len(col) == 4 and col.startswith("he") and col[2:].isdigit()


def split_one(path) -> None:
    if not path.exists():
        print(f"  ({path.name} missing - skipping)")
        return
    df = pd.read_csv(path, sep=C.CSV_SEP, dtype={"gp_nr": str, "plz": str})

    feedin = [c for c in df.columns if is_feedin(c)]
    ids = [c for c in ID_COLS if c in df.columns]
    labels = [c for c in df.columns
              if c.startswith(LABEL_PREFIX) and c not in NOT_LABELS]
    safe = [c for c in df.columns if c not in feedin]

    stem = path.stem
    net_path = path.with_name(f"{stem}_netload.csv")
    fin_path = path.with_name(f"{stem}_feedin.csv")

    df[safe].to_csv(net_path, sep=C.CSV_SEP, index=False)
    df[ids + labels + feedin].to_csv(fin_path, sep=C.CSV_SEP, index=False)

    print(f"  {path.name}")
    print(f"    -> {net_path.name:44s} {len(df):4d} x {len(safe):3d} cols  (train on this)")
    print(f"    -> {fin_path.name:44s} {len(df):4d} x {len(ids+labels+feedin):3d} cols  "
          f"({len(feedin)} quarantined)")
    return feedin


def main() -> None:
    print("quarantining the export register (docs/data_problems.md block 4)\n")
    removed = None
    for p in INPUTS:
        r = split_one(p)
        removed = removed or r
    if removed:
        print(f"\nquarantined columns ({len(removed)}):")
        print("  " + ", ".join(removed))
        print("\nUse the *_feedin.csv tables for silver labels and evaluation only,\n"
              "never as model input.")


if __name__ == "__main__":
    main()
