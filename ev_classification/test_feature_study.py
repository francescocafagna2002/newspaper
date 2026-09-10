import numpy as np
import pandas as pd
import pytest

from ev_classification.classification_core import LOADS
from ev_classification.feature_study import (
    aggregate_customer_features,
    check_customer_features,
    time_window_mask,
    weekly_distribution_features,
    weekly_timed_features,
)


def sample_week():
    frame = pd.DataFrame(np.zeros((1, 672)), columns=LOADS)
    frame.index = pd.Index([7])
    # Monday 20:00: 3 kW step, held for two hours, then paired shutoff.
    frame.iloc[0, 80:88] = 0.75
    return frame


def test_wrapping_time_window():
    hours = np.array([5.75, 6.0, 17.75, 18.0, 23.75])
    assert time_window_mask(hours, 18, 6).tolist() == [True, False, False, True, True]


def test_timed_ramp_persistence_and_units():
    features, events = weekly_timed_features(
        sample_week(), window="evening_overnight", ramp_kw=2, persistence_hours=1.5,
        return_events=True,
    )
    assert features.loc[7, "timed_sustained_count"] == 1
    assert features.loc[7, "timed_paired_count"] == 1
    assert events.iloc[0].rise_kw == pytest.approx(3.0)


def test_distribution_and_multiscale_features_are_finite():
    features = weekly_distribution_features(sample_week(), power_floor_kw=3.0)
    assert features.loc[7, "dist_window_high_hours"] == pytest.approx(2.0)
    assert features.filter(like="multi_").shape[1] == 16
    assert np.isfinite(features.to_numpy()).all()


def test_customer_aggregation_and_prohibited_check():
    weeks = pd.DataFrame({
        "gp_nr": ["a", "a", "b", "b"],
        "week_start": pd.date_range("2025-01-06", periods=4, freq="7D"),
        "temporal_target": [1, 1, 0, 0],
    }, index=[2, 3, 8, 9])
    weekly = pd.DataFrame({"timed_ramp_count": [1.0, 3.0, 0.0, 2.0]}, index=weeks.index)
    result = aggregate_customer_features(weeks, weekly)
    assert result.gp_nr.tolist() == ["a", "b"]
    assert result.loc[0, "customer_mean_timed_ramp_count"] == 2.0
    with pytest.raises(AssertionError):
        check_customer_features(pd.DataFrame({"weeks": [2], "ev_label": [1]}))
