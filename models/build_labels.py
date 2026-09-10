"""Step 1 - build household-level PV labels and the GP-Nr -> MP ID join.

Key decisions (see ../docs/data_problems.md):
  * GIGI is keyed by subsidy *application*, not household -> aggregate to GP-Nr.
  * ``PV = "-"`` means "this row is about another asset", NOT "no PV".
  * positive  = any row has PV in {x, X}  OR  any row marks a battery.
  * everything else = UNLABELLED (not negative) -> Positive-Unlabeled problem.
  * blank PV rows and blank GP-Nr rows are dropped.
  * whole-record labels: we keep the PV commissioning date for the change-point
    feature / leakage checks but do not window features per household.

Output: ``data_addition/gigi_augmented.csv`` with one row per (gp_nr, mp_id).
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from .io_utils import read_gigi, gp_to_mpid

POS_TOKENS = {"x", "X"}


def _is_pos(series: pd.Series) -> bool:
    return series.isin(POS_TOKENS).any()


def _commissioning_date(rows: pd.DataFrame) -> pd.Timestamp | pd.NaT:
    """Earliest InBetrieb-Datum whose annotation refers to PV (fallback: any)."""
    dates = pd.to_datetime(rows["date_commissioned"], format="%d.%m.%Y", errors="coerce")
    refers = rows.get("commissioned_refers_to", pd.Series("", index=rows.index)).fillna("")
    pv_mask = refers.str.contains("PV", case=False, na=False)
    picked = dates[pv_mask]
    if picked.notna().any():
        return picked.min()
    return dates.min()


def build_household_labels() -> pd.DataFrame:
    gigi = read_gigi(annotated=True)
    n_raw = len(gigi)
    gigi = gigi[gigi["gp_nr"].notna() & (gigi["gp_nr"] != "")].copy()
    n_blank_gp = n_raw - len(gigi)

    recs = []
    for gp_nr, rows in gigi.groupby("gp_nr"):
        pv_pos = _is_pos(rows["pv"])
        batt_pos = _is_pos(rows["battery"])
        pv_values = set(rows["pv"].str.lower())
        has_blank_pv = "" in pv_values or rows["pv"].isna().any()

        if pv_pos:
            label, source = 1, "gigi_pv"
        elif batt_pos:
            label, source = 1, "gigi_battery"
        else:
            label, source = pd.NA, "unlabeled"

        # first non-empty context values
        def first(col: str) -> str:
            s = rows[col][rows[col].astype(bool)]
            return s.iloc[0] if len(s) else ""

        recs.append(
            dict(
                gp_nr=gp_nr,
                pv_label=label,
                label_source=source,
                pv_commissioning_date=_commissioning_date(rows),
                has_blank_pv_row=has_blank_pv,
                n_gigi_rows=len(rows),
                plz=first("plz"),
                ort=first("ort"),
                kanton=first("kanton"),
                pv_kwp=pd.to_numeric(
                    rows["pv_kwp"].str.replace(",", ".", regex=False), errors="coerce"
                ).max(),
            )
        )
    hh = pd.DataFrame.from_records(recs)
    hh.attrs["n_blank_gp_rows"] = n_blank_gp
    return hh


def attach_meters(hh: pd.DataFrame) -> pd.DataFrame:
    chain = gp_to_mpid()  # gp_nr, zpb, anlage, mp_id
    merged = hh.merge(chain[["gp_nr", "zpb", "anlage", "mp_id"]], on="gp_nr", how="left")
    merged["has_meter"] = merged["mp_id"].notna()
    return merged


def main() -> None:
    hh = build_household_labels()
    out = attach_meters(hh)

    C.GIGI_AUGMENTED.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "gp_nr", "mp_id", "zpb", "anlage", "pv_label", "label_source",
        "pv_commissioning_date", "has_meter", "has_blank_pv_row", "n_gigi_rows",
        "pv_kwp", "plz", "ort", "kanton",
    ]
    out[cols].to_csv(C.GIGI_AUGMENTED, index=False)

    # ---- verification printout (targets from the plan) ----
    n_hh = hh["gp_nr"].nunique()
    labelled = hh[hh["pv_label"].notna()]
    pos_pv = hh[hh["label_source"] == "gigi_pv"]
    pos_batt = hh[hh["label_source"] == "gigi_battery"]
    with_meter = out[out["has_meter"]].groupby("gp_nr")
    hh_with_meter = out.loc[out["has_meter"], "gp_nr"].nunique()
    multi = (
        out[out["has_meter"]].groupby("gp_nr")["mp_id"].nunique().pipe(lambda s: s[s > 1])
    )
    pos_with_meter = out[(out["pv_label"] == 1) & out["has_meter"]]["gp_nr"].nunique()

    print(f"GIGI households (unique GP-Nr)      : {n_hh}")
    print(f"  dropped blank-GP-Nr rows          : {hh.attrs['n_blank_gp_rows']}")
    print(f"  positives (gigi_pv)               : {pos_pv['gp_nr'].nunique()}")
    print(f"  positives (gigi_battery)          : {pos_batt['gp_nr'].nunique()}")
    print(f"  total positives                   : {labelled['gp_nr'].nunique()}")
    print(f"  unlabeled households              : {n_hh - labelled['gp_nr'].nunique()}")
    print(f"households reaching meter data      : {hh_with_meter}")
    print(f"  positives with meter data         : {pos_with_meter}")
    print(f"  multi-meter households            : {multi.size} (max {int(multi.max()) if multi.size else 0})")
    print(f"\nwrote {C.GIGI_AUGMENTED}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
