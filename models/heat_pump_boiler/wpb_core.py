"""Wärmepumpenboiler (heat-pump water heater) plateau detector - pure logic.

Kept free of I/O so it can be unit-tested without the 77 GB exports.

Physics we are exploiting
-------------------------
A Wärmepumpenboiler (WPB) heats a 200-300 L tank with a small compressor:

    electrical power   ~0.3-1.0 kW      (vs ~2-4 kW for a resistance Elektroboiler)
    daily energy       ~1.5-2.5 kWh     (vs ~6-8 kWh; COP ~3 explains the ratio)
    cycle              ~1 per day, 3-8 h long, usually timer/tariff-locked
    seasonality        weak - tracks cold-water inlet temperature, NOT heating
                       degree days (that is what separates it from a space heat pump)

At 15-minute resolution the exports are **kWh per slot**, so
``kW = kWh_per_slot * 4`` and a 0.5 kW WPB shows up as 0.125 kWh/slot -
close to the household noise floor. Measuring that floor is the go/no-go
check before anything else here is worth running (see ``night_noise``).

The detector therefore does *not* look at single days. It works on
per-slot quantiles taken across all days of a meter-month:

    p10 profile - the "always on" envelope. A load has to be present on
                  >=90% of days to lift it, which encodes the once-a-day
                  regularity of a timer-driven boiler and rejects
                  occasional loads without a separate regularity test.
    p50 profile - the typical day, used for reporting amplitude.

A plateau is a contiguous run of slots where the p10 profile sits a
threshold above the meter's own baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

SLOTS_PER_DAY = 96
SLOTS_PER_HOUR = 4
KW_PER_KWH_SLOT = 4.0  # kWh in a 15-min slot -> average kW over that slot

# Amplitude bands in kW (electrical), converted to kWh/slot where used.
WPB_KW_MIN, WPB_KW_MAX = 0.25, 1.20
RESISTANCE_KW_MIN = 1.50
# A WPB tank reheat is long; a resistance element dumps the same energy fast.
WPB_MIN_HOURS = 1.5
WPB_MAX_HOURS = 9.0
# Search floor - short enough to also catch a jitter-eroded resistance boiler.
SEARCH_MIN_HOURS = 0.5


def slot_to_clock(slot: int) -> str:
    """0-based slot index -> the clock time that slot *ends* at.

    Slot k (0-based) covers the 15 min ending at (k+1)*15 minutes past
    midnight, matching ``config.QUARTER_HOUR_COLS``.
    """
    minutes = (slot + 1) * 15
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def day_quantile_profiles(
    days: np.ndarray, quantiles: tuple[float, ...] = (10.0, 50.0, 90.0)
) -> np.ndarray:
    """``(n_days, 96)`` of kWh -> ``(len(quantiles), 96)`` per-slot quantiles.

    Days with any missing slot still contribute their present slots, so a
    partially-delivered day does not drop out entirely.
    """
    days = np.asarray(days, dtype=np.float64)
    if days.ndim != 2 or days.shape[1] != SLOTS_PER_DAY:
        raise ValueError(f"expected (n_days, {SLOTS_PER_DAY}), got {days.shape}")
    if days.shape[0] == 0 or np.all(np.isnan(days)):
        return np.full((len(quantiles), SLOTS_PER_DAY), np.nan)
    return np.nanpercentile(days, quantiles, axis=0)


def baseline(profile: np.ndarray, window_hours: float = 1.0) -> float:
    """The meter's own standing load: the quietest hour of the typical day.

    Taking the minimum of a rolling window rather than the global minimum
    keeps a single anomalous slot from setting the reference too low.
    """
    w = max(1, int(round(window_hours * SLOTS_PER_HOUR)))
    prof = np.asarray(profile, dtype=np.float64)
    if np.all(np.isnan(prof)):
        return float("nan")
    # Wrap around midnight - the quiet hour often straddles 00:00.
    padded = np.concatenate([prof, prof[: w - 1]]) if w > 1 else prof
    windows = np.lib.stride_tricks.sliding_window_view(padded, w)
    return float(np.nanmin(np.nanmean(windows, axis=1)))


def night_noise(days: np.ndarray, night_slots: np.ndarray | None = None) -> float:
    """Go/no-go statistic: residual spread of night slots, in kWh/slot.

    For each night slot we take the median across days, then measure how far
    individual days sit from it (median absolute deviation, scaled to a
    standard-deviation equivalent). This is the floor a plateau has to clear:
    if it comes back at or above 0.125 kWh/slot, a 0.5 kW WPB is not
    separable from ordinary household variation and amplitude-based detection
    cannot work on this meter.
    """
    if night_slots is None:
        night_slots = np.arange(0, 6 * SLOTS_PER_HOUR)  # 00:00-06:00
    sub = np.asarray(days, dtype=np.float64)[:, night_slots]
    if sub.size == 0 or np.all(np.isnan(sub)):
        return float("nan")
    resid = sub - np.nanmedian(sub, axis=0)
    return float(1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid))))


@dataclass
class Plateau:
    """One contiguous elevated run in a meter-month's p10 profile."""

    start_slot: int
    end_slot: int          # exclusive
    n_slots: int
    duration_h: float
    start_clock: str
    end_clock: str
    amplitude_kw: float    # p10 elevation over baseline -> the "every day" power
    p50_amplitude_kw: float
    energy_kwh: float      # per day, from the p10 elevation
    baseline_kwh_slot: float
    wraps_midnight: bool

    def as_row(self) -> dict:
        return asdict(self)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs in a circular mask, as (start, end_exclusive).

    Circular because a boiler started at 22:00 and running 5 h crosses
    midnight and would otherwise be reported as two short plateaus.
    """
    if not mask.any():
        return []
    if mask.all():
        return [(0, len(mask))]
    padded = np.concatenate([mask, mask])
    out: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if padded[i] and not padded[i - 1]:
            j = i
            while j < i + n and padded[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def find_plateaus(
    p10: np.ndarray,
    p50: np.ndarray,
    min_kw: float = WPB_KW_MIN,
    min_hours: float = SEARCH_MIN_HOURS,
    base: float | None = None,
) -> list[Plateau]:
    """Elevated runs in the p10 (always-on) profile, longest first.

    Both thresholds are deliberately permissive rather than WPB-specific:
    ``classify`` applies the WPB band afterwards. Searching with the WPB
    duration floor would miss resistance boilers entirely - they dump the
    same daily energy in ~1 h, and once start-time jitter is eroded by the
    p10 envelope their always-on core is only a few slots long. We need them
    found, both as the contrast class that makes the amplitude split
    defensible and as donors for the injection test.
    """
    p10 = np.asarray(p10, dtype=np.float64)
    p50 = np.asarray(p50, dtype=np.float64)
    if np.all(np.isnan(p10)):
        return []
    base = baseline(p10) if base is None else base
    thresh = min_kw / KW_PER_KWH_SLOT  # kW -> kWh per 15-min slot
    min_slots = max(1, int(round(min_hours * SLOTS_PER_HOUR)))

    elevation = np.nan_to_num(p10 - base, nan=0.0)
    found: list[Plateau] = []
    for start, end in _runs(elevation >= thresh):
        n = end - start
        if n < min_slots:
            continue
        idx = np.arange(start, end) % SLOTS_PER_DAY
        amp = float(np.nanmean(elevation[idx]))
        found.append(
            Plateau(
                start_slot=int(start % SLOTS_PER_DAY),
                end_slot=int(end % SLOTS_PER_DAY),
                n_slots=int(n),
                duration_h=n / SLOTS_PER_HOUR,
                start_clock=slot_to_clock(int(start % SLOTS_PER_DAY) - 1),
                end_clock=slot_to_clock(int((end - 1) % SLOTS_PER_DAY)),
                amplitude_kw=amp * KW_PER_KWH_SLOT,
                p50_amplitude_kw=float(
                    np.nanmean(np.nan_to_num(p50 - base, nan=0.0)[idx])
                ) * KW_PER_KWH_SLOT,
                energy_kwh=amp * n,
                baseline_kwh_slot=float(base),
                wraps_midnight=bool(end > SLOTS_PER_DAY),
            )
        )
    return sorted(found, key=lambda p: (p.n_slots, p.amplitude_kw), reverse=True)


def classify(p: Plateau) -> str:
    """Label one plateau by amplitude and duration.

    The split is physical, not fitted: a WPB and a resistance element deliver
    a comparable amount of hot water per day, so they differ by roughly the
    COP in power and inversely in duration.
    """
    if p.amplitude_kw >= RESISTANCE_KW_MIN:
        return "resistance_boiler"
    if (
        WPB_KW_MIN <= p.amplitude_kw <= WPB_KW_MAX
        and WPB_MIN_HOURS <= p.duration_h <= WPB_MAX_HOURS
    ):
        return "wpb_candidate"
    return "other"


def score_household(
    summer: list[Plateau], winter: list[Plateau]
) -> tuple[float, dict]:
    """Combine a summer and a winter month into one WPB score in [0, 1].

    Year-round persistence is the discriminator that does the real work here.
    A space heat pump, bathroom floor heating and a dehumidifier are all
    strongly seasonal; a WPB is not, because it tracks cold-water inlet
    temperature rather than heating demand. Summer alone would confuse all
    of them; requiring a matching plateau in both months does not.
    """
    s = next((p for p in summer if classify(p) == "wpb_candidate"), None)
    w = next((p for p in winter if classify(p) == "wpb_candidate"), None)
    ev = {
        "summer_plateau": s.as_row() if s else None,
        "winter_plateau": w.as_row() if w else None,
    }
    if s is None and w is None:
        return 0.0, ev | {"reason": "no WPB-shaped plateau in either month"}
    if s is None or w is None:
        # Seasonal: consistent with floor heating (winter) or a dehumidifier
        # (summer), so it is evidence but not much.
        ev["reason"] = "plateau in one season only - seasonal load not excluded"
        return 0.35, ev

    amp_ratio = min(s.amplitude_kw, w.amplitude_kw) / max(
        s.amplitude_kw, w.amplitude_kw
    )
    start_shift = abs(s.start_slot - w.start_slot)
    start_shift = min(start_shift, SLOTS_PER_DAY - start_shift) / SLOTS_PER_HOUR

    score = 0.5 + 0.3 * amp_ratio + 0.2 * max(0.0, 1.0 - start_shift / 3.0)
    ev |= {
        "amplitude_ratio": round(amp_ratio, 3),
        "start_shift_h": round(start_shift, 2),
        "reason": "WPB-shaped plateau present in both summer and winter",
    }
    return round(min(score, 1.0), 3), ev
