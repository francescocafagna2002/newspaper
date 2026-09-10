"""Wärmepumpenboiler detection pipeline - I/O, staging and figures.

Detector logic lives in :mod:`wpb_core` (unit-tested without data). This
module is the part that touches the 77 GB exports.

Stages, in the order they should be run - each one is a stopping point that
produces something worth showing:

  0 ``noise``     go/no-go. Measures the night-load noise floor against the
                  0.125 kWh/slot a 0.5 kW WPB would add. If the floor is at
                  or above that, stop: amplitude-based detection cannot work
                  and nothing below is worth running.
  1 ``profiles``  one streaming pass per month -> per-slot p10/p50/p90 across
                  days, per meter. Everything downstream reads this instead
                  of the exports, so the detector can be retuned in seconds.
  2 ``labelled``  plots the day profiles of the four labelled WPB households.
                  Decides whether the plateau is at night or at midday, i.e.
                  whether a night-only search window is viable at all.
  3 ``detect``    scores every meter from a summer/winter profile pair.
  4 ``inject``    donor-based detectability curve (see ``cmd_inject``).
  5 ``figures``   the slide figures.

Usage::

    python -m newspaper.models.wpb noise    --month 2025-07
    python -m newspaper.models.wpb profiles --month 2025-07 --month 2025-01
    python -m newspaper.models.wpb labelled --summer 2025-07 --winter 2025-01
    python -m newspaper.models.wpb detect   --summer 2025-07 --winter 2025-01
    python -m newspaper.models.wpb inject   --month 2025-07
    python -m newspaper.models.wpb figures  --summer 2025-07 --winter 2025-01
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import wpb_core as W
from .stream import iter_chunks

PROFILES = C.ARTIFACTS / "wpb_profiles_{ym}.npz"
RAW = C.ARTIFACTS / "wpb_raw_{ym}.npz"
PREDICTIONS = C.ARTIFACTS / "wpb_predictions.csv"
EVIDENCE = C.ARTIFACTS / "wpb_evidence.json"
NOISE_CSV = C.ARTIFACTS / "wpb_noise_floor.csv"
INJECT_CSV = C.ARTIFACTS / "wpb_detectability.csv"
FIGDIR = C.ARTIFACTS / "figures"

# A 0.5 kW WPB adds this much to each 15-minute slot. The number the whole
# approach lives or dies on.
WPB_SLOT_KWH = 0.5 / W.KW_PER_KWH_SLOT


# --------------------------------------------------------------------------- #
# Labelled cohort
# --------------------------------------------------------------------------- #
def labelled_wpb_meters() -> pd.DataFrame:
    """The GIGI households flagged ``Wärmepumpenboiler = x``, with their meters.

    Only 10 households carry the flag and only 4 of those reach meter data, so
    this is evidence to inspect, not a training set. ``-`` in the GIGI column
    means "this row is about a different asset", never "no WPB" (see
    docs/data_problems.md ③) - so there are no confirmed negatives here.
    """
    gigi = pd.read_csv(C.GIGI_ANNOTATED, sep=C.CSV_SEP, encoding=C.CSV_ENCODING,
                       dtype=str).fillna("")
    col = next(c for c in gigi.columns if c.strip().lower() == "wärmepumpenboiler")
    pos = gigi[gigi[col].str.strip().str.lower() == "x"]
    gp = {g.strip() for g in pos["GP-Nr"] if g.strip()}

    aug = pd.read_csv(C.GIGI_AUGMENTED, dtype=str).fillna("")
    out = aug[aug["gp_nr"].isin(gp) & (aug["mp_id"] != "")][
        ["gp_nr", "mp_id", "plz", "ort", "kanton"]
    ].copy()
    out["label"] = "wpb"
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage 1 - profiles
# --------------------------------------------------------------------------- #
def _collect_days(
    path: Path, meters: set[str] | None, sample_every: int
) -> dict[str, list]:
    """Stream one monthly export into ``{mp_id: [day arrays]}`` (import only).

    ``sample_every`` sub-samples the population deterministically by a hash of
    the meter id, so the same meters are picked in every month - a sample that
    changed between months would break the summer/winter pairing.
    """
    days: dict[str, list] = defaultdict(list)
    plz: dict[str, str] = {}
    for meta, values in iter_chunks(path, obis=C.OBIS_IMPORT):
        ids = meta["mp_id"].to_numpy()
        pz = meta["plz"].to_numpy()
        for i, mp in enumerate(ids):
            if meters is not None:
                if mp not in meters:
                    continue
            elif sample_every > 1 and (hash(mp) % sample_every):
                continue
            days[mp].append(values[i])
            plz.setdefault(mp, pz[i])
    return days, plz


def build_profiles(ym: str, meters=None, sample_every: int = 1,
                   keep_raw: int = 400) -> Path:
    export = {e.ym: e for e in C.discover_monthly_exports()}[ym]
    days, plz = _collect_days(export.path, meters, sample_every)

    mp_ids, p10s, p50s, p90s, n_days, noise, plzs = [], [], [], [], [], [], []
    raw_ids, raw_days = [], []
    for mp, lst in days.items():
        arr = np.vstack(lst)
        if arr.shape[0] < 10:          # too few days to trust a p10 envelope
            continue
        q = W.day_quantile_profiles(arr)
        mp_ids.append(mp)
        p10s.append(q[0]); p50s.append(q[1]); p90s.append(q[2])
        n_days.append(arr.shape[0])
        noise.append(W.night_noise(arr))
        plzs.append(plz.get(mp, ""))
        if len(raw_ids) < keep_raw:    # kept for the injection test
            raw_ids.append(mp)
            raw_days.append(arr.astype(np.float32))

    out = Path(str(PROFILES).format(ym=ym))
    np.savez_compressed(
        out, mp_id=np.array(mp_ids), plz=np.array(plzs),
        p10=np.array(p10s, dtype=np.float32), p50=np.array(p50s, dtype=np.float32),
        p90=np.array(p90s, dtype=np.float32),
        n_days=np.array(n_days), noise=np.array(noise, dtype=np.float32),
    )
    if raw_ids:
        rawout = Path(str(RAW).format(ym=ym))
        np.savez_compressed(
            rawout, mp_id=np.array(raw_ids),
            **{f"d{i}": a for i, a in enumerate(raw_days)},
        )
    print(f"{ym}: {len(mp_ids)} meters -> {out.name}  (raw kept for {len(raw_ids)})")
    return out


def load_profiles(ym: str):
    return np.load(Path(str(PROFILES).format(ym=ym)), allow_pickle=False)


def load_raw(ym: str):
    z = np.load(Path(str(RAW).format(ym=ym)), allow_pickle=False)
    return list(z["mp_id"]), [z[f"d{i}"] for i in range(len(z["mp_id"]))]


# --------------------------------------------------------------------------- #
# Stage 0 - noise floor (go/no-go)
# --------------------------------------------------------------------------- #
def cmd_noise(args):
    z = load_profiles(args.month)
    noise = z["noise"][~np.isnan(z["noise"])]
    frac = float((noise < WPB_SLOT_KWH / 2).mean())
    rows = [{
        "month": args.month, "n_meters": len(noise),
        "wpb_slot_kwh": round(WPB_SLOT_KWH, 4),
        "noise_p25": round(float(np.percentile(noise, 25)), 4),
        "noise_median": round(float(np.median(noise)), 4),
        "noise_p75": round(float(np.percentile(noise, 75)), 4),
        "frac_meters_below_half_wpb": round(frac, 4),
    }]
    pd.DataFrame(rows).to_csv(NOISE_CSV, index=False)
    print(json.dumps(rows[0], indent=2))
    print(
        f"\nGO/NO-GO: {frac:.1%} of meters have a night noise floor below half a "
        f"0.5 kW WPB step.\n"
        + ("PROCEED - the plateau clears the floor on most meters."
           if frac > 0.5 else
           "STOP or restrict to the quiet subset - on most meters a 0.5 kW "
           "plateau is inside the noise and amplitude detection will not work.")
    )


# --------------------------------------------------------------------------- #
# Stage 3 - detect
# --------------------------------------------------------------------------- #
def cmd_detect(args):
    zs, zw = load_profiles(args.summer), load_profiles(args.winter)
    wi = {mp: i for i, mp in enumerate(zw["mp_id"])}
    labelled = set(labelled_wpb_meters()["mp_id"]) if args.mark_labelled else set()

    rows, evidence = [], {}
    for i, mp in enumerate(zs["mp_id"]):
        s = W.find_plateaus(zs["p10"][i], zs["p50"][i])
        j = wi.get(mp)
        w = W.find_plateaus(zw["p10"][j], zw["p50"][j]) if j is not None else []
        score, ev = W.score_household(s, w)
        best = next((p for p in s if W.classify(p) == "wpb_candidate"), None)
        rows.append({
            "mp_id": mp, "plz": zs["plz"][i], "wpb_score": score,
            "night_noise_kwh_slot": round(float(zs["noise"][i]), 4),
            "amplitude_kw": round(best.amplitude_kw, 3) if best else "",
            "duration_h": best.duration_h if best else "",
            "start_clock": best.start_clock if best else "",
            "daily_kwh": round(best.energy_kwh, 3) if best else "",
            "has_winter_month": j is not None,
            "reason": ev["reason"],
            "is_labelled_wpb": mp in labelled,
        })
        evidence[str(mp)] = ev

    df = pd.DataFrame(rows).sort_values("wpb_score", ascending=False)
    df.to_csv(PREDICTIONS, index=False)
    EVIDENCE.write_text(json.dumps(evidence, indent=1), encoding="utf-8")
    flagged = (df["wpb_score"] >= args.threshold).mean()
    print(f"{len(df)} meters scored -> {PREDICTIONS.name}")
    print(f"flag rate at score>={args.threshold}: {flagged:.2%}")
    if labelled:
        print("\nlabelled WPB households:")
        print(df[df["is_labelled_wpb"]].to_string(index=False))


# --------------------------------------------------------------------------- #
# Stage 4 - donor-based injection
# --------------------------------------------------------------------------- #
def _retrofit(host: np.ndarray, donor: np.ndarray, plateau: W.Plateau,
              target_kw: float, rng) -> np.ndarray:
    """Add the donor's hot-water event to the host as if it ran on a heat pump.

    Per donor day we take the real excess over the donor's standing load in a
    window around its plateau, keep that day's energy ``E`` unchanged, and
    spread it over ``E / (target_kw/4)`` slots. Power drops, duration grows,
    kWh stay put - the physical signature of swapping a resistance element
    for a heat pump on the same tank.
    """
    pad = W.SLOTS_PER_HOUR  # widen by 1 h so each day's own jitter is captured
    win = (np.arange(plateau.start_slot - pad,
                     plateau.start_slot + plateau.n_slots + pad)) % W.SLOTS_PER_DAY
    base = float(np.nanmin(np.nanmean(donor, axis=0)))
    per_slot = target_kw / W.KW_PER_KWH_SLOT

    out = host.copy()
    for k in range(host.shape[0]):
        ex = np.clip(donor[rng.integers(donor.shape[0])][win] - base, 0, None)
        energy = float(np.nansum(ex))
        n_new = int(round(energy / per_slot))
        if n_new < 1 or energy <= 0:
            continue
        # Resample the day's real shape to the stretched length, then restore
        # its energy so the retrofit is strictly energy-conserving.
        stretched = np.interp(
            np.linspace(0, len(ex) - 1, n_new), np.arange(len(ex)), ex
        )
        total = float(stretched.sum())
        if total <= 0:
            continue
        stretched *= energy / total
        idx = (np.arange(win[0], win[0] + n_new)) % W.SLOTS_PER_DAY
        out[k, idx] += stretched
    return out


def cmd_inject(args):
    """Detectability curve from manufactured ground truth.

    We have 4 positives and no confirmed negatives, so recall cannot be
    measured the usual way. Instead we manufacture it: take a household's
    real day matrix, add a plateau we know is there, and see whether the
    detector finds it.

    The plateau is not a synthetic rectangle. It is lifted from a *donor* -
    a meter with an unambiguous >=1.5 kW resistance-boiler plateau - and
    converted into the WPB the same household would have after a retrofit.
    That conversion holds **daily energy fixed** and trades power for
    duration: the same tank still has to be heated, a heat pump just does it
    at roughly 1/COP of the power over roughly COP times as long. Rescaling
    only the amplitude would be wrong - it produces a 0.5 kW plateau lasting
    45 minutes, which is not a WPB and which the classifier rightly rejects.

    Because the shape comes from a real device it carries real start-time
    jitter, real duration variation and real day-to-day irregularity.
    Sweeping the target power against the host's own noise gives:

        "at X kW, on hosts with night noise below N, the plateau is
         recovered P% of the time"

    That is a physical sensitivity curve, not an accuracy claim: hosts are
    unlabelled and some may already own a WPB, so it says what the detector
    *can* see, not how often it is right.
    """
    ids, mats = load_raw(args.month)
    donors, hosts = [], []
    for mp, arr in zip(ids, mats):
        p10, p50, _ = W.day_quantile_profiles(arr)
        ps = W.find_plateaus(p10, p50)
        if ps and ps[0].amplitude_kw >= W.RESISTANCE_KW_MIN:
            donors.append((mp, arr, ps[0]))
        elif not any(W.classify(p) == "wpb_candidate" for p in ps):
            hosts.append((mp, arr, W.night_noise(arr)))
    print(f"{len(donors)} donors, {len(hosts)} candidate hosts")
    if not donors:
        print("no resistance-boiler donors found - falling back to a synthetic "
              "rectangle; report the curve as optimistic.")

    rng = np.random.default_rng(C.SEED)
    rows = []
    for target_kw in args.amplitudes:
        for mp, arr, noise in hosts[: args.n_hosts]:
            if donors:
                _, darr, dp = donors[int(rng.integers(len(donors)))]
                injected = _retrofit(arr, darr, dp, target_kw, rng)
            else:
                idx = (np.arange(91, 91 + 16)) % 96
                injected = arr.copy()
                injected[:, idx] += target_kw / W.KW_PER_KWH_SLOT
            p10, p50, _ = W.day_quantile_profiles(injected)
            hit = any(
                W.classify(p) == "wpb_candidate" for p in W.find_plateaus(p10, p50)
            )
            rows.append({"target_kw": target_kw, "mp_id": mp,
                         "night_noise_kwh_slot": round(float(noise), 4),
                         "recovered": int(hit), "donor_based": bool(donors)})

    df = pd.DataFrame(rows)
    df.to_csv(INJECT_CSV, index=False)
    print(f"\n{len(df)} injections -> {INJECT_CSV.name}\n")
    q = df["night_noise_kwh_slot"].quantile([0.33, 0.66]).to_list()
    df["noise_band"] = pd.cut(df["night_noise_kwh_slot"],
                              [-np.inf, *q, np.inf], labels=["quiet", "mid", "noisy"])
    print(df.pivot_table(index="target_kw", columns="noise_band",
                         values="recovered", aggfunc="mean", observed=False)
            .round(3).to_string())


# --------------------------------------------------------------------------- #
# Stages 2 & 5 - figures
# --------------------------------------------------------------------------- #
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGDIR.mkdir(parents=True, exist_ok=True)
    return plt


def cmd_labelled(args):
    """Plot all 96 slots for the labelled WPB households.

    The point of this stage is to answer one question before any window is
    hard-coded: is the plateau at night, or has PV self-consumption control
    moved it to midday? GIGI is ~9:1 PV-skewed, so a night-only detector
    could miss exactly these households.
    """
    plt = _mpl()
    lab = labelled_wpb_meters()
    zs, zw = load_profiles(args.summer), load_profiles(args.winter)
    si = {mp: i for i, mp in enumerate(zs["mp_id"])}
    wi = {mp: i for i, mp in enumerate(zw["mp_id"])}
    hits = [mp for mp in lab["mp_id"] if mp in si or mp in wi]
    if not hits:
        print("none of the labelled WPB meters appear in these months")
        return

    fig, axes = plt.subplots(len(hits), 1, figsize=(11, 2.6 * len(hits)),
                             squeeze=False, sharex=True)
    hours = np.arange(96) / 4
    for ax, mp in zip(axes[:, 0], hits):
        for z, idx, name, colour in ((zs, si, args.summer, "#c2410c"),
                                     (zw, wi, args.winter, "#1d4ed8")):
            if mp not in idx:
                continue
            i = idx[mp]
            ax.fill_between(hours, z["p10"][i], z["p90"][i], alpha=0.18, color=colour)
            ax.plot(hours, z["p50"][i], color=colour, lw=1.4, label=f"{name} p50")
            ax.plot(hours, z["p10"][i], color=colour, lw=1.0, ls="--",
                    label=f"{name} p10 (always on)")
        ax.axhline(WPB_SLOT_KWH, color="#6b7280", lw=0.8, ls=":",
                   label="0.5 kW step")
        row = lab[lab["mp_id"] == mp].iloc[0]
        ax.set_title(f"labelled WPB - GP {row['gp_nr']} / meter {mp} "
                     f"({row['plz']} {row['ort']})", fontsize=9, loc="left")
        ax.set_ylabel("kWh / 15 min")
        ax.margins(x=0)
    axes[-1, 0].set_xlabel("hour of day")
    axes[-1, 0].set_xticks(range(0, 25, 3))
    axes[0, 0].legend(fontsize=7, ncol=3)
    fig.tight_layout()
    out = FIGDIR / "wpb_labelled_profiles.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  ({len(hits)} of {len(lab)} labelled meters present)")


def cmd_figures(args):
    plt = _mpl()
    zs = load_profiles(args.summer)

    # Amplitude histogram - is the population bimodal at ~0.5 and ~3 kW?
    amps = []
    for i in range(len(zs["mp_id"])):
        ps = W.find_plateaus(zs["p10"][i], zs["p50"][i])
        if ps:
            amps.append(ps[0].amplitude_kw)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.clip(amps, 0, 5), bins=80, color="#0f766e")
    ax.axvspan(W.WPB_KW_MIN, W.WPB_KW_MAX, color="#f59e0b", alpha=0.25,
               label="WPB band")
    ax.axvline(W.RESISTANCE_KW_MIN, color="#b91c1c", ls="--",
               label="resistance floor")
    ax.set(xlabel="plateau amplitude (kW, p10 over baseline)", ylabel="meters",
           title=f"Plateau amplitude across the population, {args.summer}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "wpb_amplitude_hist.png", dpi=150)

    # Noise floor vs the signal we need to see
    fig, ax = plt.subplots(figsize=(8, 4))
    n = zs["noise"][~np.isnan(zs["noise"])]
    ax.hist(np.clip(n, 0, 0.4), bins=80, color="#334155")
    ax.axvline(WPB_SLOT_KWH, color="#b91c1c",
               label="0.5 kW WPB step (0.125 kWh/slot)")
    ax.set(xlabel="night noise floor (kWh / 15 min)", ylabel="meters",
           title="Can a 0.5 kW plateau be seen at all?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "wpb_noise_floor.png", dpi=150)

    if INJECT_CSV.exists():
        d = pd.read_csv(INJECT_CSV)
        fig, ax = plt.subplots(figsize=(8, 4))
        for band, g in d.groupby(pd.qcut(d["night_noise_kwh_slot"], 3,
                                         labels=["quiet", "mid", "noisy"],
                                         duplicates="drop"), observed=False):
            m = g.groupby("target_kw")["recovered"].mean()
            ax.plot(m.index, m.values, marker="o", label=f"{band} hosts")
        ax.set(xlabel="injected plateau amplitude (kW)",
               ylabel="fraction recovered", ylim=(0, 1.02),
               title="Detectability curve (donor-based injection)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGDIR / "wpb_detectability.png", dpi=150)
    print(f"figures -> {FIGDIR}")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profiles")
    p.add_argument("--month", action="append", required=True)
    p.add_argument("--sample-every", type=int, default=1,
                   help="keep 1 in N meters (hash-stable across months)")
    p.add_argument("--keep-raw", type=int, default=400)
    p.add_argument("--labelled-only", action="store_true")
    p.set_defaults(func=lambda a: [
        build_profiles(m,
                       meters=set(labelled_wpb_meters()["mp_id"])
                       if a.labelled_only else None,
                       sample_every=a.sample_every, keep_raw=a.keep_raw)
        for m in a.month])

    p = sub.add_parser("noise"); p.add_argument("--month", required=True)
    p.set_defaults(func=cmd_noise)

    for name, fn in (("labelled", cmd_labelled), ("detect", cmd_detect),
                     ("figures", cmd_figures)):
        p = sub.add_parser(name)
        p.add_argument("--summer", required=True)
        p.add_argument("--winter", required=True)
        if name == "detect":
            p.add_argument("--threshold", type=float, default=0.8)
            p.add_argument("--mark-labelled", action="store_true", default=True)
        p.set_defaults(func=fn)

    p = sub.add_parser("inject")
    p.add_argument("--month", required=True)
    p.add_argument("--amplitudes", type=float, nargs="+",
                   default=[0.3, 0.5, 0.8, 1.2])
    p.add_argument("--n-hosts", type=int, default=150)
    p.set_defaults(func=cmd_inject)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
