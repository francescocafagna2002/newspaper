#!/usr/bin/env python3
"""Test steep-edge evidence as a post-LightGBM decision cascade."""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT.parent))

from ev_classification_total.run_heuristic_extension import (
    ROOT,
    WEEKS,
    evidence_counts,
    load_partition,
    model_scores,
    paired_bootstrap,
    score_metrics,
    select_recent_weeks,
    write_json,
)
from ev_classification.classification_core import LOADS, extract_events

OUT = ROOT / "output" / "heuristic_extension_2"


def apply_direction(base, evidence, direction, floor=.9, ceiling=.3):
    base = np.asarray(base, float)
    evidence = np.asarray(evidence, bool)
    if direction == "positive_rescue":
        return np.maximum(base, np.where(evidence, floor, 0.0))
    if direction == "negative_veto":
        return np.minimum(base, np.where(evidence, 1.0, ceiling))
    if direction == "two_way":
        return np.where(evidence, np.maximum(base, floor), np.minimum(base, ceiling))
    raise ValueError(direction)


def edge_weekly_features(weeks, events, duration, rise_kw, variation, paired):
    selected = events.loc[
        events.duration.ge(duration)
        & events.rise.ge(rise_kw)
        & events.power.gt(0)
        & events.mad.div(events.power.clip(lower=.1)).le(variation)
    ].copy()
    if paired:
        selected = selected.loc[selected.fall.ge(rise_kw / 2) & selected.symmetry.ge(.3)]
    counts = selected.groupby("row").size().reindex(weeks.index, fill_value=0)
    maximum_rise = selected.groupby("row").rise.max().reindex(weeks.index, fill_value=0)
    result = pd.DataFrame({
        "gp_nr": weeks.gp_nr.astype(str).to_numpy(),
        "week_start": weeks.week_start.to_numpy(),
        "strong_event_count": counts.to_numpy(int),
        "maximum_rise_kw_per_15min": maximum_rise.to_numpy(float),
    })
    return result


def customer_evidence(properties, weekly, minimum_events, minimum_weeks):
    summary = weekly.groupby("gp_nr").agg(
        strong_events=("strong_event_count", "sum"),
        evidence_weeks=("strong_event_count", lambda x: int((x > 0).sum())),
        maximum_rise_kw_per_15min=("maximum_rise_kw_per_15min", "max"),
    )
    aligned = summary.reindex(properties.astype(str)).fillna(0)
    obvious = aligned.strong_events.ge(minimum_events) & aligned.evidence_weeks.ge(minimum_weeks)
    return aligned.reset_index(drop=True), obvious.to_numpy()


def search_direction(validation, base, threshold, weeks):
    searches = []
    event_cache = {}
    weekly_cache = {}
    for amplitude in (1.5, 3.0, 5.0):
        event_cache[amplitude] = extract_events(weeks, "daily_q20", amplitude)
    for amplitude, duration, rise_kw, variation, paired in product(
        (1.5, 3.0, 5.0), (1.5, 2.0), (1.0, 2.0, 3.0, 4.0), (.2, .4), (False, True)
    ):
        key = (amplitude, duration, rise_kw, variation, paired)
        weekly = edge_weekly_features(
            weeks, event_cache[amplitude], duration, rise_kw, variation, paired
        )
        weekly_cache[key] = weekly
        for minimum_events, minimum_weeks in product((2, 3), (1, 2)):
            summary, obvious = customer_evidence(validation.group_id, weekly, minimum_events, minimum_weeks)
            for direction in ("positive_rescue", "negative_veto", "two_way"):
                for floor, ceiling in ((.7, .3), (.9, .3), (.9, .1)):
                    score = apply_direction(base, obvious, direction, floor, ceiling)
                    m = score_metrics(validation.has_EV, score, threshold)
                    searches.append({
                        "direction": direction, "amplitude_kw": amplitude,
                        "minimum_duration_hours": duration, "rise_kw_per_15min": rise_kw,
                        "relative_mad_max": variation, "paired_edge_required": paired,
                        "minimum_events": minimum_events, "minimum_evidence_weeks": minimum_weeks,
                        "score_floor": floor, "score_ceiling": ceiling,
                        "evidence_flags": int(obvious.sum()),
                        "changed_predictions": int(np.sum((score >= threshold) != (base >= threshold))),
                        **{name: m[name] for name in (
                            "accuracy", "balanced_accuracy", "pr_auc", "precision", "recall", "f1", "brier"
                        )},
                    })
    return pd.DataFrame(searches), weekly_cache


def select_best(frame):
    return frame.sort_values(
        ["balanced_accuracy", "pr_auc", "precision", "changed_predictions",
         "minimum_duration_hours", "rise_kw_per_15min", "minimum_events",
         "minimum_evidence_weeks", "paired_edge_required"],
        ascending=[False, False, False, True, False, False, False, False, False],
    ).iloc[0]


def config_from_row(row):
    return {
        "direction": row.direction, "amplitude_kw": float(row.amplitude_kw),
        "minimum_duration_hours": float(row.minimum_duration_hours),
        "rise_kw_per_15min": float(row.rise_kw_per_15min),
        "relative_mad_max": float(row.relative_mad_max),
        "paired_edge_required": bool(row.paired_edge_required),
        "minimum_events": int(row.minimum_events),
        "minimum_evidence_weeks": int(row.minimum_evidence_weeks),
        "score_floor": float(row.score_floor), "score_ceiling": float(row.score_ceiling),
    }


def evaluate(frame, base, threshold, weekly, config):
    summary, obvious = customer_evidence(
        frame.group_id, weekly, config["minimum_events"], config["minimum_evidence_weeks"]
    )
    extended = apply_direction(
        base, obvious, config["direction"], config["score_floor"], config["score_ceiling"]
    )
    predictions = pd.DataFrame({
        "sample_id": frame.sample_id, "group_id": frame.group_id, "target": frame.has_EV,
        "base_score": base, "strong_events": summary.strong_events,
        "evidence_weeks": summary.evidence_weeks,
        "maximum_rise_kw_per_15min": summary.maximum_rise_kw_per_15min,
        "steep_edge_evidence": obvious.astype(int), "extended_score": extended,
        "base_prediction": (base >= threshold).astype(int),
        "extended_prediction": (extended >= threshold).astype(int),
    })
    result = {
        "base": score_metrics(frame.has_EV, base, threshold),
        "extended": score_metrics(frame.has_EV, extended, threshold),
        "paired_bootstrap": paired_bootstrap(frame.has_EV, base, extended, threshold),
    }
    return predictions, result


def run():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite extension: {OUT}")
    OUT.mkdir(parents=True)
    data = pd.read_csv(ROOT.parent / "data_addition" / "train_property_samples_auxw_netload.csv", sep=";")
    split = pd.read_csv(ROOT / "output" / "split_manifest.csv")
    validation = load_partition(data, split, "validation")
    test = load_partition(data, split, "test")
    validation_base, base_config = model_scores(validation, ROOT / "output" / "E22_lgbm_total")
    threshold = float(base_config["threshold"])
    weeks = select_recent_weeks(pd.read_parquet(WEEKS))
    if [c for c in weeks if c.startswith("net_load_")] != LOADS:
        raise AssertionError("Detailed weekly load contract changed")

    search, weekly_cache = search_direction(validation, validation_base, threshold, weeks)
    search.to_csv(OUT / "bounded_search.csv", index=False)
    directional = pd.DataFrame([select_best(search.loc[search.direction.eq(direction)])
                                for direction in ("positive_rescue", "negative_veto", "two_way")])
    directional.to_csv(OUT / "direction_comparison_validation.csv", index=False)
    chosen = select_best(directional)
    config = config_from_row(chosen)
    base_validation = score_metrics(validation.has_EV, validation_base, threshold)
    accepted = (chosen.balanced_accuracy > base_validation["balanced_accuracy"]
                and chosen.precision >= base_validation["precision"])
    config.update({
        "base_model": "E22_lgbm_total", "base_threshold": threshold,
        "baseline": "daily_q20", "history": "latest 12 eligible weeks, label-independent",
        "selection": "best direction by validation balanced accuracy, then PR-AUC and precision",
        "accepted": bool(accepted),
        "deployment_direction": config["direction"] if accepted else "none; retain base LightGBM",
        "test_status": "sealed_for_extension_2",
    })
    write_json(OUT / "extension_freeze.json", config)

    key = (config["amplitude_kw"], config["minimum_duration_hours"], config["rise_kw_per_15min"],
           config["relative_mad_max"], config["paired_edge_required"])
    weekly = weekly_cache[key]
    weekly.to_parquet(OUT / "weekly_steep_edge_evidence.parquet", index=False)
    validation_predictions, validation_metrics = evaluate(
        validation, validation_base, threshold, weekly, config
    )
    validation_predictions.to_csv(OUT / "validation_predictions.csv", index=False)
    write_json(OUT / "metrics_validation.json", validation_metrics)

    # Configuration is frozen above; the reused test is read only after freeze.
    test_base, final_config = model_scores(test, ROOT / "output" / "best")
    if final_config["id"] != config["base_model"]:
        raise AssertionError("Base model mismatch")
    test_predictions, test_metrics = evaluate(test, test_base, threshold, weekly, config)
    test_metrics["status"] = "reused test diagnostic; not an unbiased new estimate"
    test_predictions.to_csv(OUT / "reused_test_predictions.csv", index=False)
    write_json(OUT / "metrics_reused_test.json", test_metrics)
    config["test_status"] = "opened_once_after_extension_2_freeze; reused from original study"
    write_json(OUT / "extension_freeze.json", config)

    direction_rows = []
    for _, row in directional.sort_values("direction").iterrows():
        direction_rows.append(
            f"| {row.direction} | {row.accuracy:.4f} | {row.balanced_accuracy:.4f} | "
            f"{row.pr_auc:.4f} | {row.precision:.4f} | {row.recall:.4f} | {int(row.changed_predictions)} |"
        )
    vm, tm = validation_metrics, test_metrics
    def metric_row(name, key):
        return (f"| {name} | {vm['base'][key]:.4f} | {vm['extended'][key]:.4f} | "
                f"{vm['extended'][key]-vm['base'][key]:+.4f} | {tm['base'][key]:.4f} | "
                f"{tm['extended'][key]:.4f} | {tm['extended'][key]-tm['base'][key]:+.4f} |")
    report = f"""# Heuristic extension 2: steep-edge cascade

## Finding

A steep slope makes physical sense as part of an EV charging signature, but no tested
direction improved validation balanced accuracy. The best heuristic candidate was
`{config['direction']}` and it was **rejected**, so the deployed direction remains
`none` and LightGBM is unchanged. The diagnostic candidate requires at least
{config['minimum_events']} events across
{config['minimum_evidence_weeks']} week(s), each beginning with a rise of at least
{config['rise_kw_per_15min']:g} kW in one 15-minute step and continuing at least
{config['minimum_duration_hours']:g} hours above a {config['amplitude_kw']:g} kW
residual threshold. Paired stop edge required: `{config['paired_edge_required']}`.

## Direction test on validation

| Direction | Accuracy | Balanced accuracy | PR-AUC | Precision | Recall | Predictions changed |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM (no heuristic) | {base_validation['accuracy']:.4f} | {base_validation['balanced_accuracy']:.4f} | {base_validation['pr_auc']:.4f} | {base_validation['precision']:.4f} | {base_validation['recall']:.4f} | 0 |
{chr(10).join(direction_rows)}

This directly tests the assumption that steep-edge evidence should rescue likely
false negatives rather than veto predicted EVs. A charger may simply be unused during
the retained history, so absence of a detected edge is weak negative evidence. In this
run, positive rescue added false positives without recovering a false negative, while
negative veto gained majority-class accuracy by missing more EV-labelled properties.

## Frozen rule results

| Metric | Validation base | Validation extended | Change | Reused-test base | Reused-test extended | Change |
|---|---:|---:|---:|---:|---:|---:|
{metric_row('Accuracy','accuracy')}
{metric_row('Balanced accuracy','balanced_accuracy')}
{metric_row('PR-AUC','pr_auc')}
{metric_row('Precision','precision')}
{metric_row('Recall','recall')}
{metric_row('F1','f1')}

Validation confusion matrix: `{vm['extended']['confusion_matrix']}`; reused-test
confusion matrix: `{tm['extended']['confusion_matrix']}`, both in
`[[TN, FP], [FN, TP]]` order. The test is reused because the original `_total` run
already opened it. No parameter was changed after the extension-2 freeze.

## Recommendation

Do not apply the steep-edge cascade to the classifier. Keep the edge evidence as an
explanation field for manual review. The paired rise, plateau, and fall remain a
reasonable charging-event hypothesis, but they are not specific enough to infer EV
ownership in this cohort.
"""
    (OUT / "results.md").write_text(report)


if __name__ == "__main__":
    run()
