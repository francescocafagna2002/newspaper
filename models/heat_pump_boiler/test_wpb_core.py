"""Synthetic-household tests for the WPB plateau detector.

These run without the 77 GB exports: every household here is generated, so
the tests pin the detector's *logic* (does it find a plateau it is given,
does it reject one it should) rather than its accuracy on real data.

Run:  python -m pytest tests/test_wpb_core.py -q
      (or just execute this file - it asserts on import)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import wpb_core as W  # noqa: E402


def make_household(
    n_days: int = 30,
    base_kw: float = 0.25,
    plateau_kw: float | None = None,
    start_slot: int = 92,
    duration_slots: int = 16,
    start_jitter: int = 2,
    noise: float = 0.02,
    plateau_prob: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """A synthetic ``(n_days, 96)`` kWh matrix with standing load, an evening
    cooking bump on most days, and optionally a recurring plateau."""
    r = np.random.default_rng(seed)
    days = np.full((n_days, 96), base_kw / W.KW_PER_KWH_SLOT)
    days += r.normal(0, noise, (n_days, 96))
    for i in range(n_days):
        if r.random() < 0.7:  # cooking - irregular, must not read as a plateau
            days[i, 72:80] += r.uniform(0.1, 0.4)
    if plateau_kw:
        for i in range(n_days):
            if r.random() > plateau_prob:
                continue
            s = (start_slot + int(r.integers(-start_jitter, start_jitter + 1))) % 96
            length = duration_slots + int(r.integers(-2, 3))
            days[i, (np.arange(s, s + length)) % 96] += plateau_kw / W.KW_PER_KWH_SLOT
    return np.clip(days, 0, None)


def _plateaus(days):
    p10, p50, _ = W.day_quantile_profiles(days)
    return W.find_plateaus(p10, p50)


def test_slot_clock_mapping():
    assert W.slot_to_clock(0) == "00:15"
    assert W.slot_to_clock(95) == "00:00"


def test_finds_wpb_across_midnight():
    """0.5 kW from 23:00, wrapping past midnight - the common timer setting."""
    ps = _plateaus(make_household(plateau_kw=0.5, start_slot=91, duration_slots=16))
    hits = [p for p in ps if W.classify(p) == "wpb_candidate"]
    assert hits, "0.5 kW nightly plateau not found"
    assert hits[0].wraps_midnight
    assert 0.4 <= hits[0].amplitude_kw <= 0.65


def test_finds_resistance_boiler():
    """3 kW for ~1.5 h. Start jitter erodes the p10 core to well under the
    WPB duration floor, so the *search* floor has to be shorter than it."""
    ps = _plateaus(
        make_household(plateau_kw=3.0, start_slot=92, duration_slots=6, start_jitter=1)
    )
    assert any(W.classify(p) == "resistance_boiler" for p in ps)


def test_no_plateau_in_clean_household():
    assert not any(W.classify(p) == "wpb_candidate" for p in _plateaus(make_household()))


def test_p10_envelope_rejects_occasional_load():
    """The design claim: a load on only half the days must not lift p10.

    This is what makes a separate regularity test unnecessary, and what keeps
    an EV charged a few nights a week out of the WPB class.
    """
    days = make_household(
        plateau_kw=0.5, start_slot=40, duration_slots=16, plateau_prob=0.5, seed=5
    )
    assert not any(W.classify(p) == "wpb_candidate" for p in _plateaus(days))


def test_year_round_scores_above_seasonal():
    summer = _plateaus(make_household(plateau_kw=0.5, start_slot=91, duration_slots=16))
    winter = _plateaus(
        make_household(plateau_kw=0.5, start_slot=91, duration_slots=16, seed=9)
    )
    both, _ = W.score_household(summer, winter)
    one_only, ev = W.score_household(summer, [])
    assert both > 0.8
    assert one_only < 0.5
    assert "seasonal" in ev["reason"]


def test_noise_floor_statistic_is_below_wpb_amplitude():
    """Go/no-go: a 0.5 kW WPB is 0.125 kWh/slot. On a quiet synthetic
    household the floor must sit well under that, or nothing else works."""
    floor = W.night_noise(make_household(plateau_kw=None))
    assert floor < 0.125 / 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL PASS")
