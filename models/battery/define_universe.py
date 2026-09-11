"""Define the label-blind PV-likely universe from measured export days."""
from __future__ import annotations

import pandas as pd

from . import config as C
from ..pv.io import gp_to_mpid


def build_universe() -> pd.DataFrame:
    """Admit every household whose export register is confirmed present, not
    only those that ever read a positive value.

    A working register that reads exactly zero on every measured day is a
    real, physically meaningful class: a battery run under a greedy
    self-consumption or feed-in-limiting strategy can eliminate export
    entirely (battery_evidence_base.md S1, S C). The previous rule required
    >=MIN_EXPORT_DAYS *positive* days even for meters with a confirmed
    working register, which silently excluded that class as if its register
    were missing. It admits it as ``measured_zero_export`` instead.
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
    has_export_days = agg["exp_days_positive"].fillna(0) >= C.MIN_EXPORT_DAYS
    fallback = ~measured & (agg["pv_probability"].fillna(0) >= 0.8)
    agg["admission_rule"] = "excluded"
    agg.loc[measured & ~has_export_days, "admission_rule"] = "measured_zero_export"
    agg.loc[measured & has_export_days, "admission_rule"] = "measured_export"
    agg.loc[fallback, "admission_rule"] = "pv_probability_fallback"
    agg["in_universe"] = agg["admission_rule"] != "excluded"
    return agg


def retention_upper_bound() -> tuple[int, int, float]:
    """Upper bound for the gate from the prior complete all-meter scan.

    A household whose export register was never read at all (no reading on
    any day) truly has no export evidence. One that was read every day but
    always logged zero has a *confirmed working* register -- that is the
    real, admissible ``measured_zero_export`` class build_universe() now
    keeps, so the bound checks register presence (>=1 reading, any value),
    not a positive reading. This check rejects an impossible gate without
    repeating 77 GB of I/O.
    """
    prior = pd.read_csv(
        C.PV_METER_FLAGS.parent / "feedin_meter_stats.csv", dtype={"mp_id": str}
    )
    register_present = set(prior.loc[prior["exp_days"] > 0, "mp_id"])
    labels = pd.read_csv(C.LABELS, dtype={"gp_nr": str, "mp_id": str})
    positives = labels[labels.battery_positive & labels.has_meter]
    gps = positives.groupby("gp_nr")["mp_id"].apply(lambda s: s.isin(register_present).any())
    kept, total = int(gps.sum()), int(len(gps))
    return kept, total, kept / total


def main() -> None:
    upper_kept, upper_total, upper = retention_upper_bound()
    print(f"known battery retention upper bound: {upper_kept}/{upper_total} = {upper:.1%}")
    if upper < 0.95:
        raise RuntimeError(
            "universe acceptance gate is mathematically impossible: even "
            f"admitting every household with a confirmed export register "
            f"retains only {upper:.1%}"
        )
    if not C.METER_EXPORT_DAYS.exists():
        raise RuntimeError(
            f"gate pre-check passed ({upper_kept}/{upper_total} = {upper:.1%}); "
            f"{C.METER_EXPORT_DAYS} is missing, so build_universe() cannot run yet "
            "-- run `python -m models.battery.build_features` (Step 3, the full "
            "streaming pass) first"
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
