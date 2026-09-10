"""Signal-only feature primitives for the post-test EV behavior study."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .classification_core import LOADS, PROHIBITED, compact_features, kw
except ImportError:
    from classification_core import LOADS, PROHIBITED, compact_features, kw


WINDOWS = {
    "late_afternoon_night": (16, 2),
    "evening_overnight": (18, 6),
    "night": (20, 8),
}


def time_window_mask(hours, start_hour, end_hour):
    """Return a possibly midnight-wrapping half-open time window."""
    hours = np.asarray(hours, dtype=float)
    if start_hour < end_hour:
        return (hours >= start_hour) & (hours < end_hour)
    return (hours >= start_hour) | (hours < end_hour)


def _max_run(values):
    padded = np.r_[False, np.asarray(values, dtype=bool), False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int((ends - starts).max()) if len(starts) else 0


def weekly_timed_features(
    frame,
    window="evening_overnight",
    ramp_kw=2.0,
    persistence_hours=1.5,
    return_events=False,
):
    """Extract timed positive ramps, persistence, plateaus, and paired edges."""
    start_hour, end_hour = WINDOWS[window]
    power = kw(frame[LOADS])
    hours = np.tile(np.arange(96) / 4, 7)
    allowed = time_window_mask(hours, start_hour, end_hour)
    persistence = int(round(persistence_hours * 4))
    names = [
        "timed_ramp_count", "timed_all_ramp_count", "timed_sustained_count",
        "timed_paired_count", "timed_active_days", "timed_cross_midnight_count",
        "timed_max_rise_kw", "timed_median_rise_kw", "timed_mean_level_kw",
        "timed_mean_plateau_cv", "timed_mean_edge_symmetry",
        "timed_start_concentration", "timed_start_sin", "timed_start_cos",
    ]
    values = np.zeros((len(frame), len(names)), dtype=float)
    records = []
    for position, (row_index, series) in enumerate(zip(frame.index, power)):
        delta = np.diff(series, prepend=series[0])
        all_candidates = np.flatnonzero(delta >= ramp_kw)
        candidates = all_candidates[allowed[all_candidates]]
        events = []
        for start in candidates:
            stop = start + persistence
            if start < 4 or stop > len(series):
                continue
            baseline = float(np.median(series[start - 4:start]))
            segment = series[start:stop]
            rise = float(delta[start])
            required = baseline + 0.6 * rise
            interruptions = int(np.count_nonzero(segment < required))
            sustained = interruptions <= 1
            level = float(np.median(segment) - baseline)
            plateau_cv = float(np.median(np.abs(segment - np.median(segment))) / max(abs(level), 0.1))
            search_end = min(len(series), stop + 32)
            later = delta[stop:search_end]
            fall = float(max(0.0, -later.min())) if len(later) else 0.0
            symmetry = min(rise, fall) / max(rise, fall, 0.1)
            paired = sustained and fall >= 0.5 * rise
            start_time = float(hours[start])
            crosses = (start % 96) + persistence > 96
            event = {
                "row": row_index,
                "start_interval": int(start),
                "start_hour": start_time,
                "day": int(start // 96),
                "rise_kw": rise,
                "level_kw": level,
                "persistence_hours": persistence_hours,
                "interruptions": interruptions,
                "sustained": bool(sustained),
                "paired": bool(paired),
                "edge_symmetry": symmetry,
                "plateau_cv": plateau_cv,
                "cross_midnight": bool(crosses),
            }
            records.append(event)
            events.append(event)
        rises = np.array([event["rise_kw"] for event in events], dtype=float)
        levels = np.array([event["level_kw"] for event in events], dtype=float)
        cvs = np.array([event["plateau_cv"] for event in events], dtype=float)
        symmetries = np.array([event["edge_symmetry"] for event in events], dtype=float)
        angles = np.array([event["start_hour"] * 2 * np.pi / 24 for event in events])
        mean_sin = float(np.sin(angles).mean()) if len(angles) else 0.0
        mean_cos = float(np.cos(angles).mean()) if len(angles) else 0.0
        values[position] = [
            len(events), len(all_candidates), sum(event["sustained"] for event in events),
            sum(event["paired"] for event in events),
            len({event["day"] for event in events if event["sustained"]}),
            sum(event["cross_midnight"] for event in events),
            float(rises.max()) if len(rises) else 0.0,
            float(np.median(rises)) if len(rises) else 0.0,
            float(levels.mean()) if len(levels) else 0.0,
            float(cvs.mean()) if len(cvs) else 0.0,
            float(symmetries.mean()) if len(symmetries) else 0.0,
            float(np.hypot(mean_sin, mean_cos)), mean_sin, mean_cos,
        ]
    features = pd.DataFrame(values, index=frame.index, columns=names)
    if return_events:
        return features, pd.DataFrame.from_records(records)
    return features


def weekly_distribution_features(
    frame,
    window="evening_overnight",
    power_floor_kw=3.0,
    level_tolerance_kw=0.5,
):
    """Extract time-of-use, power-level, load-tail, and multiscale features."""
    start_hour, end_hour = WINDOWS[window]
    power = kw(frame[LOADS])
    hours = np.tile(np.arange(96) / 4, 7)
    selected = time_window_mask(hours, start_hour, end_hour)
    high = power >= power_floor_kw
    selected_high = high[:, selected]
    positive_excess = np.maximum(power - power_floor_kw, 0.0)
    selected_excess = positive_excess[:, selected]
    all_high_hours = high.sum(axis=1) / 4
    selected_high_hours = selected_high.sum(axis=1) / 4
    daily = power.reshape(-1, 7, 96)
    peak_slot = daily.argmax(axis=2)
    peak_angles = peak_slot * 2 * np.pi / 96
    peak_sin = np.sin(peak_angles).mean(axis=1)
    peak_cos = np.cos(peak_angles).mean(axis=1)
    profile = np.median(daily, axis=1)
    profile_centered = profile - profile.mean(axis=1, keepdims=True)
    day_centered = daily - daily.mean(axis=2, keepdims=True)
    numerator = (day_centered * profile_centered[:, None, :]).sum(axis=2)
    denominator = np.linalg.norm(day_centered, axis=2) * np.linalg.norm(profile_centered, axis=1)[:, None]
    day_consistency = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0).mean(axis=1)
    output = {
        "dist_all_high_hours": all_high_hours,
        "dist_window_high_hours": selected_high_hours,
        "dist_window_high_fraction": selected_high_hours / np.maximum(all_high_hours, 0.25),
        "dist_all_excess_kwh": positive_excess.sum(axis=1) / 4,
        "dist_window_excess_kwh": selected_excess.sum(axis=1) / 4,
        "dist_window_excess_fraction": selected_excess.sum(axis=1) / np.maximum(positive_excess.sum(axis=1), 0.1),
        "dist_active_days": selected_high.reshape(-1, 7, selected.sum() // 7).any(axis=2).sum(axis=1),
        "dist_peak_time_concentration": np.hypot(peak_sin, peak_cos),
        "dist_peak_time_sin": peak_sin,
        "dist_peak_time_cos": peak_cos,
        "dist_day_profile_consistency": day_consistency,
        "dist_midday_negative_share": (power[:, np.tile((np.arange(96) >= 40) & (np.arange(96) < 60), 7)] < 0).mean(axis=1),
    }
    output["dist_window_longest_high_run_hours"] = np.array([
        _max_run(row[selected]) / 4 for row in high
    ], dtype=float)
    concentration = np.zeros(len(frame)); modes = np.zeros(len(frame)); entropy = np.zeros(len(frame))
    for i, row in enumerate(power):
        values = row[row >= power_floor_kw]
        if not len(values):
            continue
        quantized = np.round(values / level_tolerance_kw).astype(int)
        _, counts = np.unique(quantized, return_counts=True)
        probabilities = counts / counts.sum()
        concentration[i] = probabilities.max()
        modes[i] = len(counts)
        entropy[i] = -(probabilities * np.log(probabilities)).sum()
    output.update(
        dist_level_concentration=concentration,
        dist_level_mode_count=modes,
        dist_level_entropy=entropy,
    )
    sorted_power = np.sort(power, axis=1)
    output["dist_tail_mean_kw"] = sorted_power[:, -67:].mean(axis=1)
    output["dist_tail_area_kwh"] = np.maximum(sorted_power[:, -67:] - power_floor_kw, 0).sum(axis=1) / 4
    for lag, minutes in ((1, 15), (2, 30), (4, 60), (8, 120)):
        difference = power[:, lag:] - power[:, :-lag]
        output[f"multi_pos_q95_{minutes}m"] = np.quantile(difference, 0.95, axis=1)
        output[f"multi_neg_q95_{minutes}m"] = np.quantile(-difference, 0.95, axis=1)
        output[f"multi_abs_mean_{minutes}m"] = np.abs(difference).mean(axis=1)
        output[f"multi_energy_{minutes}m"] = np.square(difference).mean(axis=1)
    result = pd.DataFrame(output, index=frame.index)
    return result.replace([np.inf, -np.inf], 0).fillna(0)


def aggregate_customer_features(weeks, weekly_features):
    """Aggregate capped state-appropriate week features to one customer row."""
    assert weeks.index.equals(weekly_features.index)
    rows = []
    for gp_nr, group in weeks.groupby("gp_nr", sort=True):
        values = weekly_features.loc[group.index].to_numpy(dtype=float)
        top = np.sort(values, axis=0)[-min(3, len(values)):].mean(axis=0)
        record = {"gp_nr": gp_nr, "target": int(group.temporal_target.iloc[0]), "weeks": len(group)}
        for column, mean, maximum, top3, std in zip(
            weekly_features.columns,
            values.mean(axis=0), values.max(axis=0), top, values.std(axis=0),
        ):
            record[f"customer_mean_{column}"] = mean
            record[f"customer_max_{column}"] = maximum
            record[f"customer_top3_{column}"] = top3
            record[f"customer_std_{column}"] = std
        rows.append(record)
    result = pd.DataFrame(rows)
    check_customer_features(result.drop(columns=["gp_nr", "target"]))
    return result


def build_customer_features(weeks, family, config):
    """Build one of the predeclared customer-level feature families."""
    parts = []
    if family in ("timed", "combined"):
        timed = config["timed"] if family == "combined" else config
        parts.append(weekly_timed_features(weeks, **timed))
    if family in ("distribution", "combined"):
        distribution = config["distribution"] if family == "combined" else config
        parts.append(weekly_distribution_features(weeks, **distribution))
    if family == "combined":
        parts.insert(0, compact_features(weeks))
    weekly = pd.concat(parts, axis=1)
    return aggregate_customer_features(weeks, weekly), weekly


def check_customer_features(features):
    names = set(features.columns)
    assert not names.intersection(PROHIBITED), "Prohibited feature column"
    assert "weeks" in names
    signal = names - {"weeks"}
    assert signal and all(name.startswith("customer_") for name in signal), "Non-signal customer feature"
    assert np.isfinite(features.to_numpy(dtype=float)).all(), "Non-finite customer feature"
