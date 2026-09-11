"""Define the label-blind PV-likely universe from measured export days."""
from __future__ import annotations

import pandas as pd

from . import config as C
from ..pv.io import gp_to_mpid


def build_universe() -> pd.DataFrame:
    """Admit households that demonstrably feed in (>=MIN_EXPORT_DAYS days of
    positive export), which is a validated label-blind proxy for "has PV".

    Do not weaken this to "the export register was read at all": every meter
    in the export files has 2.29 rows (mostly exact zeros), so that test is
    true for 100% of meters and would admit the whole population -- which
    makes the model relearn "has PV" instead of the battery (plan D3).
    Measured: positive export covers 11.3% of meters, matching the ~12% ElPA
    PV base rate, and pv_probability separates exporters from non-exporters
    at AUC 0.904.
    """
    stats = pd.read_csv(C.METER_EXPORT_DAYS, dtype={"mp_id": str})
    flags = pd.read_csv(C.PV_METER_FLAGS, dtype={"mp_id": str})
    stats = stats.merge(flags[["mp_id", "is_production_meter"]], on="mp_id", how="left")
    stats = stats[~stats["is_production_meter"].fillna(False)].copy()
    per_gp = gp_to_mpid()[["gp_nr", "mp_id"]].dropna().drop_duplicates().merge(
        stats, on="mp_id", how="left"
    )
    agg = per_gp.groupby("gp_nr", as_index=False).agg(
        exp_days_positive=("exp_days_positive", "sum"),
        exp_rows_seen=("exp_rows_seen", "max"),
    )
    # Label-blind fallback for genuinely absent export registers only.
    pv = pd.read_csv(C.PV_OOF, dtype={"gp_nr": str})[["gp_nr", "pv_probability"]]
    agg = agg.merge(pv, on="gp_nr", how="outer")
    measured = agg["exp_rows_seen"].fillna(False).astype(bool)
    fallback = ~measured & (agg["pv_probability"].fillna(0) >= 0.8)
    agg["admission_rule"] = "excluded"
    agg.loc[measured & (agg["exp_days_positive"].fillna(0) >= C.MIN_EXPORT_DAYS), "admission_rule"] = "measured_export"
    agg.loc[fallback, "admission_rule"] = "pv_probability_fallback"
    agg["in_universe"] = agg["admission_rule"] != "excluded"
    return agg


def retention_upper_bound() -> tuple[int, int, float]:
    """Upper bound for the gate from the prior complete all-meter scan.

    A household with zero export energy over the record cannot have ten
    positive-export days.  This check can therefore reject an impossible gate
    without repeating 77 GB of I/O.

    Checked 2026-09-11 and left as-is: relaxing ``exp_kwh > 0`` to "the
    export register was read at all" does NOT fix the gate, it destroys it.
    Every one of the 93,279 meters has 2.29 rows, so that test is true for
    everyone and admits the whole population. The 14 households it would
    "rescue" also score pv_probability 0.012-0.482 on the import side, i.e.
    they show no PV signature either -- they are unobservable in this data,
    not mismeasured.
    """
    prior = pd.read_csv(
        C.PV_METER_FLAGS.parent / "feedin_meter_stats.csv", dtype={"mp_id": str}
    )
    ever_exported = set(prior.loc[prior["exp_kwh"] > 0, "mp_id"])
    labels = pd.read_csv(C.LABELS, dtype={"gp_nr": str, "mp_id": str})
    positives = labels[labels.battery_positive & labels.has_meter]
    gps = positives.groupby("gp_nr")["mp_id"].apply(lambda s: s.isin(ever_exported).any())
    kept, total = int(gps.sum()), int(len(gps))
    return kept, total, kept / total


def main() -> None:
    upper_kept, upper_total, upper = retention_upper_bound()
    print(f"known battery retention upper bound: {upper_kept}/{upper_total} = {upper:.1%}")
    if upper < 0.95:
        raise RuntimeError(
            "universe acceptance gate is mathematically impossible: even the "
            f"weaker any-export rule retains only {upper:.1%}; >=10 days cannot retain more"
        )
    out = build_universe()
    out.to_csv(C.UNIVERSE, index=False)
    labels = pd.read_csv(C.LABELS, dtype={"gp_nr": str})
    hh = labels.groupby("gp_nr", as_index=False).agg(
        battery_positive=("battery_positive", "max"), has_meter=("has_meter", "max")
    )
    checked = hh[hh.battery_positive & hh.has_meter].merge(
        out[["gp_nr", "in_universe", "admission_rule"]], on="gp_nr", how="left"
    )
    kept = checked["in_universe"].fillna(False)
    retention = float(kept.mean())
    dropped = checked.loc[~kept, ["gp_nr", "admission_rule"]]
    dropped.to_csv(C.ARTIFACTS / "universe_dropped.csv", index=False)
    n_missing = int((~out["exp_rows_seen"].fillna(False).astype(bool)).sum())
    n_fallback = int((out["admission_rule"] == "pv_probability_fallback").sum())
    print(f"households in universe: {int(out.in_universe.sum())} / {len(out)}")
    print(f"missing export register: {n_missing} ({n_missing / len(out):.2%}); fallback admitted: {n_fallback}")
    print(f"known battery retention: {int(kept.sum())}/{len(kept)} = {retention:.1%}")
    print(f"dropped list: {C.ARTIFACTS / 'universe_dropped.csv'}")
    if retention < 0.95:
        raise RuntimeError(
            f"universe acceptance gate failed: {retention:.1%} < 95%; threshold unchanged"
        )


if __name__ == "__main__":
    main()
