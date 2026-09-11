"""Rule-based presence heuristics for PV and EV — the benchmark the ML models must beat.

Reads `data_addition/train_property_sequences.csv` (15-min series for the 342 labelled
samples), applies transparent physical rules, scores them against the GIGI labels, and
writes per-sample scores + a metrics table to `models/artifacts/heuristics/`.

Run:  python -m newspaper.models.heuristics
See:  docs/heuristics_baseline.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEQ = ROOT / "data_addition" / "train_property_sequences.csv"
NETLOAD = ROOT / "data_addition" / "train_property_samples_auxw_netload.csv"
OUT = HERE / "artifacts" / "heuristics"

QH = [f"t{v // 60:02d}{v % 60:02d}" for v in range(15, 24 * 60 + 15, 15)]
QH[-1] = "t0000"
SLOTS = 96
KW = 4.0  # 15-min kWh -> kW

# --- EV -----------------------------------------------------------------------
# Headline rule: the largest 15-min ramp on the import channel. At quarter-hour
# resolution the charger's switch-on step is all that survives -- the plateau, the
# power band and the matched switch-off add nothing measurable (see docs), so the
# detector below is kept only to reproduce that comparison, not as the rule.
EV_RAMP_KW = 9.0   # selected out-of-fold: 9.2 kW in ~100/100 StratifiedGroupKFold folds
EV_SPEC_KW = 5.8   # the Large-L2 minimum from the challenge spec -- unfitted variant
EV_RAMP_SWEEP = [3.6, 5.8, 7.2, 9.0, 11.0]

# name, min kW, max kW, min samples as specified, min samples tuned on this set
EV_CLASSES = [
    ("L1", 1.4, 1.8, 24, 24),        # 6 h  / 6 h
    ("L2_small", 3.6, 4.8, 4, 12),   # 1 h  / 3 h
    ("L2_large", 5.8, np.inf, 2, 8),  # 30 min / 2 h
]
SCAN_DMIN = 2    # detect once at the shortest duration, filter per variant afterwards
EDGE_TOL = 0.8   # a step of >= 0.8 * p_lo counts as switch-on
BASE_LAG = 4     # quarter-hours before the step used as the pre-event baseline

# --- EV mode 2 ("granny cable") -----------------------------------------------
# A Schuko domestic socket delivers 230 V x 10 A = 2.3 kW. That sits in the dead
# zone between the spec's L1 (1.4-1.8 kW) and L2_small (3.6-4.8 kW) bands, so
# EV_CLASSES cannot see it by construction -- a household charging without a
# wallbox is invisible to every rule above. The band edges below are those two
# spec bands' inner edges, not constants fitted here; the duration reuses
# L2_small's 1 h minimum.
MODE2_LO, MODE2_HI = 1.8, 3.6
MODE2_DMIN = 4                                        # 1 h, as L2_small
MODE2_WINDOW = set(range(80, 96)) | set(range(0, 24))  # run starts 20:00-06:00

# --- PV -----------------------------------------------------------------------
# No invented constants. The three windows are astronomical, not tuned: SUN_MONTHS is
# equinox to equinox, MIDDAY is solar noon (13:00 in this fixed-UTC+1 series) +/- 2.5 h,
# NIGHT is the repo's existing base-load window (`night_share`). A 36-way sweep over
# plausible alternatives moves PV-B's AUC by at most 0.02 - see docs.
SUN_MONTHS = (3, 4, 5, 6, 7, 8, 9, 10)   # March-October
MIDDAY = slice(39, 60)                   # 10:00-15:00 (interval-end labelled)
NIGHT = slice(0, 24)                     # 00:15-06:00

# --- Battery ------------------------------------------------------------------
# 96% of labelled batteries sit in a PV household, and `has_PV` used on its own already
# scores AUC 0.700 against `has_Batterie`. Every battery rule is therefore evaluated
# WITHIN the PV households (n=281, 77% battery) -- otherwise it just re-detects PV.
#
# Mechanism: a PV house without storage exports the moment generation exceeds load, so
# its export profile is symmetric about solar noon. A battery absorbs the morning
# surplus and only lets export resume once it is full, pushing export into the
# afternoon. No tuned constants: the split point is solar noon and the decision
# threshold is 0.50, the symmetric-solar-day null.
SOLAR_NOON = 51                          # index of the 13:00 quarter-hour
SHOULDER = (3, 4, 9, 10)                 # battery saturates in high summer - see docs
BAT_PM_THR = 0.50                        # symmetric solar day; not fitted
EVENING = slice(67, 92)                  # 17:00-23:00, the discharge window

# PV-B deliberately has NO threshold. Parity (ratio < 1) is not a separating point --
# 38 of 61 non-PV households also draw less at midday than at night, because the house
# is empty during the day. Any cut would be invented, so PV-B is reported threshold-free
# (ROC-AUC / PR-AUC) and used only as the net-load-only comparison against PV-A.

EV_THR = 3.3    # >= 1 detected session per month, per 100 days (comparison rules only)


# ------------------------------------------------------------------ loading ---
def load_samples() -> dict:
    """sample_id -> dict(net, bezug, dates, labels). Gap days become NaN so that no
    detected event can straddle a hole in the record."""
    df = pd.read_csv(SEQ, sep=";", parse_dates=["date"], low_memory=False)
    meta = ["sample_id", "group_id", "gp_nr", "sample_type",
            "has_PV", "has_Waermepumpe", "has_Batterie", "has_EV"]
    out = {}
    for sid, g in df.groupby("sample_id", sort=False):
        bez = g[g.direction == "bezug"].set_index("date")[QH].sort_index()
        ein = g[g.direction == "einspeisung"].set_index("date")[QH].sort_index()
        full = pd.date_range(bez.index.min(), bez.index.max(), freq="D")
        bez = bez.reindex(full)
        ein = ein.reindex(full).fillna(0.0).where(bez.notna())
        row = g.iloc[0]
        out[sid] = dict(
            net=(bez.to_numpy(float) - ein.to_numpy(float)) * KW,
            bezug=bez.to_numpy(float) * KW,
            dates=full,
            n_days=int(bez.notna().all(axis=1).sum()),
            **{c: row[c] for c in meta},
        )
    return out


# --------------------------------------------------------------- EV rules ----
def detect_events(series: np.ndarray, p_lo: float, p_hi: float, d_min: int):
    """Maximal runs that (a) start on a switch-on step, (b) stay >= 0.8*p_lo above the
    pre-step baseline for >= d_min samples, and (c) whose median excess sits in the
    class power band. Returns [(start_index, length, median_excess_kW), ...].

    NOTE: pass the IMPORT channel, not raw net load. On raw net load the falling limb
    of a PV afternoon is itself a sustained in-band step and the rule becomes a PV
    detector (see docs/heuristics_baseline.md)."""
    thr = EDGE_TOL * p_lo
    step = np.diff(series, prepend=np.nan)
    n = series.size
    events, i = [], 1
    while i < n:
        if not (step[i] >= thr):
            i += 1
            continue
        win = series[max(0, i - BASE_LAG):i]
        base = np.nanmedian(win) if np.isfinite(win).any() else np.nan
        if not np.isfinite(base):
            i += 1
            continue
        j = i
        while j < n and np.isfinite(series[j]) and (series[j] - base) >= thr:
            j += 1
        length = j - i
        if length >= d_min:
            excess = float(np.nanmedian(series[i:j]) - base)
            if p_lo <= excess <= p_hi:
                events.append((i, length, excess))
                i = j
                continue
        i += 1
    return events


def ev_scores(s: dict) -> dict:
    """EV: the largest step the import channel takes in one quarter-hour. A house
    without an EV has a stable daily peak; a charger adds a single large switch-on
    step that nothing else in a dwelling matches. Detection runs on the import
    channel (net clipped at 0) so that a PV afternoon cannot masquerade as a load.

    The spec-table detector is also scored, purely as the published comparison."""
    net = s["net"]
    valid = np.isfinite(net).all(axis=1)
    imp_d = np.clip(net[valid], 0, None)
    ramp = np.diff(imp_d, axis=1)          # kW per 15 min, never across a day boundary
    n_days = max(imp_d.shape[0], 1)
    res = dict(
        ev_ramp_max=float(ramp.max()) if ramp.size else 0.0,
        ev_ramp_days100=100.0 * float((ramp >= EV_RAMP_KW).any(axis=1).mean()),
        ev_ramp_spec_days100=100.0 * float((ramp >= EV_SPEC_KW).any(axis=1).mean()),
    )
    # --- comparison only: the three-class NILM detector as specified --------------
    imp = np.clip(net.ravel(), 0, None)
    spec_days, tuned_n = set(), 0
    spec_n = 0
    for name, p_lo, p_hi, d_spec, d_tuned in EV_CLASSES:
        ev = detect_events(imp, p_lo, p_hi, SCAN_DMIN)
        for tag, dm in (("spec", d_spec), ("tuned", d_tuned)):
            hit = [e for e in ev if e[1] >= dm]
            res[f"ev_{name}_{tag}_rate100"] = 100.0 * len(hit) / n_days
            if tag == "spec":
                spec_n += len(hit)
                spec_days |= {e[0] // SLOTS for e in hit}
            elif name != "L1":      # L1 carries no EV signal; excluded from the blend
                tuned_n += len(hit)
    res["ev_spec_rate100"] = 100.0 * spec_n / n_days
    res["ev_spec_daysfrac"] = len(spec_days) / n_days
    res["ev_tuned_rate100"] = 100.0 * tuned_n / n_days
    res.update(ev_mode2_scores(imp, n_days))
    return res


def ev_mode2_scores(imp: np.ndarray, n_days: int) -> dict:
    """EV charging from a domestic socket: a multi-hour flat plateau at ~2.3 kW
    beginning in the overnight window.

    NEGATIVE RESULT (measured 2026-09-11, kept as the closed question).
    The rule does not work: AUC 0.490 against ``has_EV``, i.e. chance, flagging
    64% of households for a 21% base rate. Adding it to the headline ramp rule
    makes that rule worse (AUC 0.686 -> 0.566, MCC 0.342 -> 0.123).

    The reason is the confound this was built to survive. A night-tariff hot
    water boiler also draws ~2 kW for hours after midnight, and most of this
    population has one. The intended discriminator was the *spread* of
    delivered energy -- a boiler reheats the same tank to the same setpoint
    every night, a car takes back whatever the day's driving used -- but
    ``energy_cv`` scores only AUC 0.556, so it does not separate them either.

    Keep this here so the question is not reopened: at 15-minute resolution,
    mode-2 charging is not distinguishable from a controlled overnight load.
    ``energy_cv`` is NaN below 3 qualifying nights, where it means nothing.
    """
    ev = detect_events(imp, MODE2_LO, MODE2_HI, MODE2_DMIN)
    night = [e for e in ev if (e[0] % SLOTS) in MODE2_WINDOW]
    kwh = np.array([e[1] * e[2] / KW for e in night])
    return dict(
        ev_mode2_nights100=100.0 * len({e[0] // SLOTS for e in night}) / n_days,
        ev_mode2_med_hours=float(np.median([e[1] for e in night]) / 4) if night else 0.0,
        ev_mode2_energy_cv=float(kwh.std() / kwh.mean())
            if len(kwh) >= 3 and kwh.mean() > 0 else np.nan,
    )


# --------------------------------------------------------------- PV rules ----
def pv_scores(s: dict) -> dict:
    """PV-A: the meter exported at all (you cannot export what you do not generate).
    PV-B: midday grid draw relative to night grid draw, over the sunlit half-year.
    A house without PV draws about the same at both (ratio ~ 1); PV collapses the
    midday term. PV-B never reads the export register."""
    net, bez, dates = s["net"], s["bezug"], s["dates"]
    valid = np.isfinite(net).all(axis=1)
    n_days = max(int(valid.sum()), 1)
    export = np.clip(-net[valid], 0, None)
    exp_day = (export > 0).any(axis=1)

    sun = valid & np.isin(dates.month, SUN_MONTHS)
    if sun.any():
        night = float(bez[sun][:, NIGHT].mean())
        ratio = float(bez[sun][:, MIDDAY].mean() / night) if night > 0 else np.nan
    else:
        ratio = np.nan
    return dict(
        pv_export_any=int(exp_day.any()),
        pv_export_days_frac=float(exp_day.mean()),
        pv_export_kwh_d=float(export.sum() / KW / n_days),
        pv_midday_night_ratio=ratio,
        n_sun_days=int(sun.sum()),
    )


# ---------------------------------------------------------- Battery rules ----
def bat_scores(s: dict) -> dict:
    """BAT-A: share of shoulder-season export falling after solar noon. A PV house
    without storage is symmetric about noon (~0.50); a battery eats the morning
    surplus and pushes export past noon. BAT-B: evening grid draw relative to night
    draw -- the battery covers the evening peak. BAT-B never reads export.

    Undefined (NaN) for a house that never exports; such houses are excluded from the
    battery evaluation rather than scored as zero."""
    net = s["net"]
    valid = np.isfinite(net).all(axis=1)
    net, dates = net[valid], s["dates"][valid]
    exp, imp = np.clip(-net, 0, None), np.clip(net, 0, None)

    def pm_share(mask):
        e = exp[mask]
        return float(e[:, SOLAR_NOON:].sum() / e.sum()) if e.sum() > 0 else np.nan

    sun = np.isin(dates.month, SUN_MONTHS)
    i = imp[sun]
    night = float(i[:, NIGHT].mean()) if sun.any() else 0.0
    return dict(
        bat_pm_share_shoulder=pm_share(np.isin(dates.month, SHOULDER)),
        bat_pm_share_sun=pm_share(sun),
        bat_eve_night=float(i[:, EVENING].mean() / night) if night > 0 else np.nan,
    )


# -------------------------------------------------------------- evaluation ---
def _ops(score, y, thr, name, note, mask=None):
    """thr=None -> threshold-free: report ranking metrics only, no operating point.
    mask -> evaluate on a subset (used to score battery rules within PV households)."""
    score, y = np.asarray(score, float), np.asarray(y)
    if mask is not None:
        keep = np.asarray(mask) & np.isfinite(score)
        score, y = score[keep], y[keep]
    score = np.nan_to_num(score)
    base = dict(rule=name, threshold=thr, auc=roc_auc_score(y, score),
                pr_auc=average_precision_score(y, score), note=note)
    if thr is None:
        return {**base, **{k: np.nan for k in
                ("precision", "recall", "f1", "mcc", "flag_rate")},
                **{k: -1 for k in ("TP", "FP", "FN", "TN")}}
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(rule=name, threshold=thr, auc=roc_auc_score(y, score),
                pr_auc=average_precision_score(y, score),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), mcc=matthews_corrcoef(y, pred),
                flag_rate=float(pred.mean()), TP=int(tp), FP=int(fp), FN=int(fn),
                TN=int(tn), note=note)


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    y_pv, y_ev = df.has_PV.to_numpy(), df.has_EV.to_numpy()
    y_bat, in_pv = df.has_Batterie.to_numpy(), df.has_PV.to_numpy() == 1
    ref = (pd.read_csv(NETLOAD, sep=";").set_index("sample_id")
           .reindex(df.sample_id))
    rows = [
        _ops(df.pv_export_any, y_pv, 1, "PV-A  meter exported at all",
             "no tuned constants"),
        _ops(-df.pv_midday_night_ratio.fillna(9), y_pv, None,
             "PV-B  midday/night draw ratio", "net-load only, threshold-free"),
    ]
    rows += [
        _ops(df.ev_ramp_max, y_ev, EV_RAMP_KW, "EV    max 15-min ramp >= 9 kW",
             "HEADLINE - threshold selected out-of-fold"),
        _ops(df.ev_ramp_max, y_ev, EV_SPEC_KW, "EV    max 15-min ramp >= 5.8 kW",
             "unfitted - 5.8 kW is the spec's Large-L2 minimum"),
        _ops(df.ev_ramp_days100, y_ev, None, "EV    ramp-day rate, threshold-free",
             "robust form: share of days with a >= 9 kW ramp"),
        _ops(df.ev_mode2_nights100, y_ev, EV_THR, "EV-m2 mode-2 night plateau rate",
             "NEGATIVE - chance-level; flags 64% at a 21% base rate (boilers)"),
        _ops(df.ev_mode2_nights100, y_ev, None, "EV-m2 mode-2 rate, threshold-free",
             "NEGATIVE - ranking only, confirms the operating point is not the problem"),
        _ops(df.ev_mode2_energy_cv, y_ev, None,
             "EV-m2 per-night energy spread (>=3 nights)",
             "NEGATIVE - intended boiler discriminator, does not separate either",
             mask=df.ev_mode2_energy_cv.notna().to_numpy()),
        _ops(np.maximum((df.ev_ramp_max >= EV_RAMP_KW).astype(float),
                        (df.ev_mode2_nights100 >= EV_THR).astype(float)),
             y_ev, 1.0, "EV    ramp >= 9 kW OR mode-2 plateau",
             "NEGATIVE - union is worse than the headline rule alone"),
    ]
    for name, *_ in EV_CLASSES:
        rows.append(_ops(df[f"ev_{name}_spec_rate100"], y_ev, EV_THR,
                         f"EV-cmp {name} as specified", "comparison"))
    rows += [
        _ops(df.ev_spec_rate100, y_ev, EV_THR, "EV-cmp  all three classes",
             "comparison"),
        _ops(df.ev_L2_large_tuned_rate100, y_ev, EV_THR, "EV-cmp  >=5.8kW for >=2h",
             "comparison, duration tuned on this set"),
        _ops(ref.bezug_peak_kw, y_ev, 6.0, "EV-ref  peak power >= 6 kW",
             "trivial reference"),
        # --- battery: scored WITHIN PV households; has_PV alone scores 0.700 ---
        _ops(df.bat_pm_share_shoulder, y_bat, BAT_PM_THR,
             "BAT-A  export after noon, Mar/Apr/Sep/Oct",
             "within PV; no fitted constant", mask=in_pv),
        _ops(df.bat_pm_share_sun, y_bat, BAT_PM_THR,
             "BAT-A' export after noon, Mar-Oct",
             "within PV; zero-selection variant", mask=in_pv),
        _ops(-df.bat_eve_night, y_bat, None, "BAT-B  evening/night grid draw",
             "within PV; net-load only, threshold-free", mask=in_pv),
        _ops(df.has_PV.astype(float), y_bat, 1.0, "BAT-ref has_PV used as the predictor",
             "whole set; the trap BAT-A/B are measured against"),
    ]
    return pd.DataFrame(rows)


def extra_checks(df: pd.DataFrame) -> dict:
    """The check the headline metrics cannot give: the only verified negatives in the
    dataset, plus the EV threshold sweep."""
    y_ev = df.has_EV.to_numpy()
    pre = df[df.sample_type == "augmented_before"]
    ev_sweep = []
    for col, grid in [("ev_ramp_max", EV_RAMP_SWEEP),
                      ("ev_spec_rate100", [1.0, 3.3, 7.1, 14.3]),
                      ("ev_L2_large_tuned_rate100", [1.0, 3.3, 7.1, 14.3])]:
        for thr in grid:
            pred = (df[col].to_numpy() >= thr).astype(int)
            ev_sweep.append(dict(score=col, threshold=thr, flag_rate=float(pred.mean()),
                                 precision=precision_score(y_ev, pred, zero_division=0),
                                 recall=recall_score(y_ev, pred, zero_division=0),
                                 f1=f1_score(y_ev, pred, zero_division=0),
                                 mcc=matthews_corrcoef(y_ev, pred)))
    pd.DataFrame(ev_sweep).to_csv(OUT / "ev_threshold_sweep.csv", sep=";", index=False)
    return dict(
        confirmed_negatives_n=int(len(pre)),
        confirmed_neg_pv_flagged=int(pre.pv_export_any.sum()),
        confirmed_neg_ev_flagged=int((pre.ev_ramp_max >= EV_RAMP_KW).sum()),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    rows = []
    for sid, s in samples.items():
        r = {k: s[k] for k in ("sample_id", "group_id", "gp_nr", "sample_type",
                               "has_PV", "has_EV", "has_Batterie", "has_Waermepumpe")}
        r["n_days"] = s["n_days"]
        r.update(pv_scores(s))
        r.update(ev_scores(s))
        r.update(bat_scores(s))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "heuristic_scores.csv", sep=";", index=False)

    metrics = evaluate(df)
    metrics.to_csv(OUT / "heuristic_metrics.csv", sep=";", index=False)
    (OUT / "heuristic_metrics.json").write_text(
        json.dumps(metrics.to_dict("records"), indent=2, default=float))
    checks = extra_checks(df)
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2))
    pd.set_option("display.width", 200)
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nchecks:", json.dumps(checks, indent=2))
    print(f"\nwrote {OUT}/heuristic_scores.csv  {df.shape}")


if __name__ == "__main__":
    main()
