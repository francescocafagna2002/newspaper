"""Build battery-positive and audit-only household labels from GIGI."""
from __future__ import annotations

import pandas as pd

from . import config as C
from ..pv.io import gp_to_mpid, read_gigi


def _first(rows: pd.DataFrame, col: str) -> str:
    s = rows[col].fillna("").astype(str).str.strip()
    s = s[s != ""]
    return s.iloc[0] if len(s) else ""


def build_household_labels() -> pd.DataFrame:
    gigi = read_gigi(annotated=True)
    gigi = gigi[gigi["gp_nr"].fillna("").str.strip() != ""].copy()
    records = []
    for gp_nr, rows in gigi.groupby("gp_nr", sort=False):
        batt = rows["battery"].fillna("").str.strip().str.lower()
        pv = rows["pv"].fillna("").str.strip().str.lower()
        positive = bool((batt == "x").any())
        audit_negative = bool(not positive and len(batt) and (batt == "-").all())
        refs = rows.get("commissioned_refers_to", pd.Series("", index=rows.index))
        ref_mask = refs.fillna("").str.contains("batterie", case=False, na=False)
        dates = pd.to_datetime(
            rows.loc[ref_mask, "date_commissioned"], format="%d.%m.%Y", errors="coerce"
        )
        records.append({
            "gp_nr": gp_nr,
            "battery_positive": positive,
            "audit_negative": audit_negative,
            "battery_commissioning_date": dates.min(),
            "has_pv_row": bool((pv == "x").any()),
            "n_gigi_rows": len(rows),
            "plz": _first(rows, "plz"),
            "kanton": _first(rows, "kanton"),
        })
    return pd.DataFrame(records)


def attach_meters(households: pd.DataFrame) -> pd.DataFrame:
    chain = gp_to_mpid()[["gp_nr", "mp_id", "zpb", "anlage"]].drop_duplicates()
    out = households.merge(chain, on="gp_nr", how="left")
    # A mapping row is not evidence of usable load history.  The completed PV
    # scan's OOF table is the canonical inventory of households with sufficient
    # meter data (>=3 valid months), and is what the plan's measured counts use.
    if C.PV_OOF.exists():
        observed = set(pd.read_csv(C.PV_OOF, dtype={"gp_nr": str})["gp_nr"])
        out["has_meter"] = out["mp_id"].notna() & out["gp_nr"].isin(observed)
    else:
        out["has_meter"] = out["mp_id"].notna()
    return out


def _verification(hh: pd.DataFrame, out: pd.DataFrame) -> dict[str, int]:
    metered = set(out.loc[out["has_meter"], "gp_nr"])
    in_window = hh["battery_commissioning_date"].between("2023-04-01", "2026-03-01")
    return {
        "households": hh.gp_nr.nunique(),
        "positive": int(hh.battery_positive.sum()),
        "positive_metered": int(hh.loc[hh.battery_positive, "gp_nr"].isin(metered).sum()),
        "positive_pv": int((hh.battery_positive & hh.has_pv_row).sum()),
        "positive_pv_metered": int(hh.loc[hh.battery_positive & hh.has_pv_row, "gp_nr"].isin(metered).sum()),
        "positive_no_pv": int((hh.battery_positive & ~hh.has_pv_row).sum()),
        "positive_no_pv_metered": int(hh.loc[hh.battery_positive & ~hh.has_pv_row, "gp_nr"].isin(metered).sum()),
        "audit_negative": int(hh.audit_negative.sum()),
        "audit_negative_metered": int(hh.loc[hh.audit_negative, "gp_nr"].isin(metered).sum()),
        "audit_negative_pv": int((hh.audit_negative & hh.has_pv_row).sum()),
        "audit_negative_pv_metered": int(hh.loc[hh.audit_negative & hh.has_pv_row, "gp_nr"].isin(metered).sum()),
        "dated": int(hh.battery_commissioning_date.notna().sum()),
        "in_window": int(in_window.sum()),
        "in_window_metered": int(hh.loc[in_window, "gp_nr"].isin(metered).sum()),
    }


EXPECTED = {
    "households": 878, "positive": 608, "positive_metered": 222,
    "positive_pv": 582, "positive_pv_metered": 212,
    "positive_no_pv": 26, "positive_no_pv_metered": 10,
    "audit_negative": 214, "audit_negative_metered": 82,
    "audit_negative_pv": 118, "audit_negative_pv_metered": 53,
    "dated": 469, "in_window": 276, "in_window_metered": 105,
}


def main() -> None:
    hh = build_household_labels()
    out = attach_meters(hh)
    cols = ["gp_nr", "mp_id", "zpb", "anlage", "battery_positive", "audit_negative",
            "battery_commissioning_date", "has_pv_row", "has_meter", "n_gigi_rows",
            "plz", "kanton"]
    out[cols].to_csv(C.LABELS, index=False)
    got = _verification(hh, out)
    for key, value in got.items():
        print(f"{key:28s}: {value}")
    print(f"wrote {C.LABELS} ({len(out)} household-meter rows)")
    if got != EXPECTED:
        delta = {k: (EXPECTED[k], got[k]) for k in EXPECTED if EXPECTED[k] != got[k]}
        raise RuntimeError(f"battery label acceptance gate failed: {delta}")


if __name__ == "__main__":
    main()
