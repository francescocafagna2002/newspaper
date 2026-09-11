from __future__ import annotations

import numpy as np
import pandas as pd

from . import build_labels
from .build_features import _day_record, _flat_run
from .define_universe import retention_upper_bound
from .solar import sun_hours, sun_hours_vec


def test_solar_geometry_and_vectorisation() -> None:
    dates = pd.to_datetime(["2025-06-21", "2025-12-21"])
    rise, setting = sun_hours_vec(dates)
    assert rise[0] < rise[1]
    assert setting[0] > setting[1]
    assert np.allclose(sun_hours(dates[0]), (rise[0], setting[0]))


def test_flat_run_detects_constant_net_power() -> None:
    v = np.linspace(0, 1, 96)
    v[20:28] = 0.25
    share, mode = _flat_run(v)
    assert share >= 8 / 96
    assert mode == 1.0


def test_joint_day_features_are_bounded() -> None:
    imp = np.full(96, 0.05, dtype=np.float32)
    exp = np.zeros(96, dtype=np.float32)
    imp[80:88] = 0
    rec = _day_record("1", "5000", pd.Timestamp("2025-07-15"), imp, exp, True)
    assert 0 <= rec["evening_zero_share"] <= 1
    assert 0 <= rec["both_zero_share"] <= 1
    assert rec["day_kwh"] > 0


def test_label_acceptance_counts() -> None:
    hh = build_labels.build_household_labels()
    out = build_labels.attach_meters(hh)
    assert build_labels._verification(hh, out) == build_labels.EXPECTED


def test_universe_gate_upper_bound_is_below_required_retention() -> None:
    # Investigated 2026-09-11: this is a real data limitation, not a bug in
    # the rule. Relaxing "ever exported" to "has any 2.29 row" passes the
    # gate only because every meter has 2.29 rows -- it admits the whole
    # population and destroys the PV-likely universe. The 14 excluded
    # households score pv_probability 0.012-0.482, so they show no PV
    # signature on the import side either. Do not "fix" this test.
    kept, total, rate = retention_upper_bound()
    assert (kept, total) == (208, 222)
    assert rate < 0.95
