#!/usr/bin/env python3
"""Add a research-grounded positive EV-event override to frozen LightGBM."""
from __future__ import annotations

import json
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / "output" / "heuristic_extension"
SOURCE = REPO / "data_addition" / "train_property_samples_auxw_netload.csv"
WEEKS = REPO / "ev_classification" / "output" / "customer_week_table.parquet"
SEED = 20260911

sys.path.insert(0, str(REPO))
from ev_classification.classification_core import LOADS, event_features, extract_events  # noqa: E402


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def positive_override(base, obvious, floor):
    """A conservative second stage: evidence can only raise a base score."""
    return np.maximum(np.asarray(base, float), np.where(np.asarray(obvious, bool), floor, 0.0))


def select_recent_weeks(frame: pd.DataFrame, cap: int = 12) -> pd.DataFrame:
    """Select history without labels, model scores, or project-date boundaries."""
    eligible = frame.loc[frame.eligible_for_supervised_training.eq(1)].copy()
    return (eligible.sort_values(["gp_nr", "week_start"])
            .groupby("gp_nr", sort=False).tail(cap).reset_index(drop=True))


def score_metrics(y, score, threshold) -> dict:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    pred = score >= threshold
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, score)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
    }


def paired_bootstrap(y, base, extended, threshold, draws=2000) -> dict:
    y = np.asarray(y, int)
    base = np.asarray(base, float)
    extended = np.asarray(extended, float)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        values.append([
            accuracy_score(y[idx], extended[idx] >= threshold) - accuracy_score(y[idx], base[idx] >= threshold),
            balanced_accuracy_score(y[idx], extended[idx] >= threshold) - balanced_accuracy_score(y[idx], base[idx] >= threshold),
            average_precision_score(y[idx], extended[idx]) - average_precision_score(y[idx], base[idx]),
        ])
    values = np.asarray(values)
    return {
        "draws": int(len(values)),
        "delta_accuracy_ci95": np.quantile(values[:, 0], [.025, .975]).tolist(),
        "delta_balanced_accuracy_ci95": np.quantile(values[:, 1], [.025, .975]).tolist(),
        "delta_pr_auc_ci95": np.quantile(values[:, 2], [.025, .975]).tolist(),
    }


def load_partition(data, split, name):
    return (data.merge(split[["group_id", "split"]], on="group_id", validate="many_to_one")
            .loc[lambda d: d.split.eq(name) & d.sample_type.eq("cross_sectional")]
            .reset_index(drop=True))


def model_scores(frame, artifact):
    config = json.loads((artifact / "config.json").read_text())
    with (artifact / ("pipeline.pkl" if artifact.name == "best" else "model.pkl")).open("rb") as stream:
        model = pickle.load(stream)
    return model.predict_proba(frame[config["columns"]])[:, 1], config


def weekly_candidate_features(weeks, amplitude, duration, family, cache):
    key = (amplitude,)
    if key not in cache:
        cache[key] = extract_events(weeks, "daily_q20", amplitude)
    config = {
        "family": family, "duration": duration, "min_events": 3,
        "variation": .2, "edge": .5, "symmetry": .3,
        "tolerance": .5, "repeat_days": 3,
    }
    features, scores, _ = event_features(weeks, cache[key], config)
    result = features.copy()
    result["gp_nr"] = weeks.gp_nr.astype(str).to_numpy()
    result["week_start"] = weeks.week_start.to_numpy()
    result["heuristic_score"] = scores
    return result


def evidence_counts(properties, weekly, score_cutoff):
    counts = weekly.assign(strong=weekly.heuristic_score.ge(score_cutoff)).groupby("gp_nr").strong.sum()
    return properties.astype(str).map(counts).fillna(0).astype(int).to_numpy()


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite extension: {OUT}")
    OUT.mkdir(parents=True)
    data = pd.read_csv(SOURCE, sep=";")
    split = pd.read_csv(ROOT / "output" / "split_manifest.csv")
    validation = load_partition(data, split, "validation")
    test = load_partition(data, split, "test")
    validation_base, base_config = model_scores(validation, ROOT / "output" / "E22_lgbm_total")
    threshold = float(base_config["threshold"])
    base_validation = score_metrics(validation.has_EV, validation_base, threshold)

    weeks = pd.read_parquet(WEEKS)
    if [c for c in weeks if c.startswith("net_load_")] != LOADS:
        raise AssertionError("The detailed weekly load contract changed")
    weeks = select_recent_weeks(weeks)
    cache = {}
    weekly_cache = {}
    search = []
    for amplitude, duration, family in product((1.5, 3.0, 5.0), (1.5, 2.0), ("block", "plateau", "repeat")):
        key = (amplitude, duration, family)
        weekly_cache[key] = weekly_candidate_features(weeks, amplitude, duration, family, cache)
        for score_cutoff, minimum_weeks, floor in product((.85, .90, .95), (1, 2, 3, 4), (.7, .9)):
            counts = evidence_counts(validation.group_id, weekly_cache[key], score_cutoff)
            obvious = counts >= minimum_weeks
            extended = positive_override(validation_base, obvious, floor)
            m = score_metrics(validation.has_EV, extended, threshold)
            search.append({
                "amplitude_kw": amplitude, "minimum_duration_hours": duration,
                "family": family, "weekly_score_cutoff": score_cutoff,
                "minimum_evidence_weeks": minimum_weeks, "score_floor": floor,
                "flagged": int(obvious.sum()), "new_positive_overrides": int((obvious & (validation_base < threshold)).sum()),
                **{k: m[k] for k in ("accuracy", "balanced_accuracy", "pr_auc", "precision", "recall", "f1", "brier")},
            })
    search_frame = pd.DataFrame(search)
    search_frame.to_csv(OUT / "bounded_search.csv", index=False)
    eligible = search_frame.loc[search_frame.precision.ge(base_validation["precision"])]
    if eligible.empty:
        eligible = search_frame
    selected_row = eligible.sort_values(
        ["balanced_accuracy", "pr_auc", "precision", "flagged", "minimum_duration_hours",
         "minimum_evidence_weeks", "amplitude_kw", "weekly_score_cutoff", "score_floor"],
        ascending=[False, False, False, True, False, False, False, False, True],
    ).iloc[0]
    selected = {
        "base_model": "E22_lgbm_total", "base_threshold": threshold,
        "baseline": "daily_q20", "amplitude_kw": float(selected_row.amplitude_kw),
        "minimum_duration_hours": float(selected_row.minimum_duration_hours),
        "family": selected_row.family, "plateau_relative_mad_max": .2,
        "edge_kw": .5, "edge_symmetry": .3, "repeat_tolerance_kw": .5,
        "repeat_days": 3, "weekly_score_cutoff": float(selected_row.weekly_score_cutoff),
        "minimum_evidence_weeks": int(selected_row.minimum_evidence_weeks),
        "score_floor": float(selected_row.score_floor),
        "selection": "validation balanced accuracy, then PR-AUC, precision, fewer flags; precision may not fall",
        "history": "latest 12 eligible weeks selected without labels or project dates",
        "test_status": "sealed_for_extension",
    }
    write_json(OUT / "extension_freeze.json", selected)

    key = (selected["amplitude_kw"], selected["minimum_duration_hours"], selected["family"])
    selected_weekly = weekly_cache[key]
    selected_weekly.to_parquet(OUT / "weekly_evidence.parquet", index=False)
    validation_counts = evidence_counts(validation.group_id, selected_weekly, selected["weekly_score_cutoff"])
    validation_flag = validation_counts >= selected["minimum_evidence_weeks"]
    validation_extended = positive_override(validation_base, validation_flag, selected["score_floor"])
    validation_metrics = {
        "base": base_validation,
        "extended": score_metrics(validation.has_EV, validation_extended, threshold),
        "paired_bootstrap": paired_bootstrap(validation.has_EV, validation_base, validation_extended, threshold),
    }
    validation_out = pd.DataFrame({
        "sample_id": validation.sample_id, "group_id": validation.group_id, "target": validation.has_EV,
        "base_score": validation_base, "evidence_weeks": validation_counts,
        "obvious_ev_override": validation_flag.astype(int), "extended_score": validation_extended,
        "base_prediction": (validation_base >= threshold).astype(int),
        "extended_prediction": (validation_extended >= threshold).astype(int),
    })
    validation_out.to_csv(OUT / "validation_predictions.csv", index=False)
    write_json(OUT / "metrics_validation.json", validation_metrics)

    # The extension is frozen above. Only now score the previously opened test.
    test_base, final_config = model_scores(test, ROOT / "output" / "best")
    if final_config["id"] != selected["base_model"]:
        raise AssertionError("Frozen base model does not match final export")
    test_counts = evidence_counts(test.group_id, selected_weekly, selected["weekly_score_cutoff"])
    test_flag = test_counts >= selected["minimum_evidence_weeks"]
    test_extended = positive_override(test_base, test_flag, selected["score_floor"])
    test_metrics = {
        "status": "reused test diagnostic; not an unbiased new estimate",
        "base": score_metrics(test.has_EV, test_base, threshold),
        "extended": score_metrics(test.has_EV, test_extended, threshold),
        "paired_bootstrap": paired_bootstrap(test.has_EV, test_base, test_extended, threshold),
    }
    pd.DataFrame({
        "sample_id": test.sample_id, "group_id": test.group_id, "target": test.has_EV,
        "base_score": test_base, "evidence_weeks": test_counts,
        "obvious_ev_override": test_flag.astype(int), "extended_score": test_extended,
        "base_prediction": (test_base >= threshold).astype(int),
        "extended_prediction": (test_extended >= threshold).astype(int),
    }).to_csv(OUT / "reused_test_predictions.csv", index=False)
    write_json(OUT / "metrics_reused_test.json", test_metrics)
    selected["test_status"] = "opened_once_after_extension_freeze; reused from original study"
    write_json(OUT / "extension_freeze.json", selected)

    vm, tm = validation_metrics, test_metrics
    accepted = vm["extended"]["balanced_accuracy"] > vm["base"]["balanced_accuracy"] and vm["extended"]["precision"] >= vm["base"]["precision"]
    report = f"""# LightGBM plus obvious-EV heuristic

## Frozen extension

The selected positive-only override uses `{selected['family']}` events at least
{selected['amplitude_kw']:g} kW above a daily local baseline, lasting at least
{selected['minimum_duration_hours']:g} hours. Plateau events must have relative MAD
≤0.2, paired edges ≥0.5 kW, and edge symmetry ≥0.3. A property is overridden only
when weekly evidence reaches {selected['weekly_score_cutoff']:.2f} in at least
{selected['minimum_evidence_weeks']} of its latest 12 eligible weeks. Its LightGBM
score is then raised to at least {selected['score_floor']:.1f}; all other scores are
unchanged. The rule was selected from the bounded grid on validation only.

## Validation

| Metric | LightGBM | LightGBM + heuristic | Change |
|---|---:|---:|---:|
| Accuracy | {vm['base']['accuracy']:.4f} | {vm['extended']['accuracy']:.4f} | {vm['extended']['accuracy']-vm['base']['accuracy']:+.4f} |
| Balanced accuracy | {vm['base']['balanced_accuracy']:.4f} | {vm['extended']['balanced_accuracy']:.4f} | {vm['extended']['balanced_accuracy']-vm['base']['balanced_accuracy']:+.4f} |
| PR-AUC | {vm['base']['pr_auc']:.4f} | {vm['extended']['pr_auc']:.4f} | {vm['extended']['pr_auc']-vm['base']['pr_auc']:+.4f} |
| Precision | {vm['base']['precision']:.4f} | {vm['extended']['precision']:.4f} | {vm['extended']['precision']-vm['base']['precision']:+.4f} |
| Recall | {vm['base']['recall']:.4f} | {vm['extended']['recall']:.4f} | {vm['extended']['recall']-vm['base']['recall']:+.4f} |

Validation confusion matrices changed from `{vm['base']['confusion_matrix']}` to
`{vm['extended']['confusion_matrix']}`. The extension is **{'accepted' if accepted else 'not accepted'}**
under the predeclared validation rule.

## Reused-test diagnostic

The test partition had already been opened by the original `_total` run. These numbers
are useful as a stability check, but they are not a fresh unbiased estimate.

| Metric | LightGBM | LightGBM + heuristic | Change |
|---|---:|---:|---:|
| Accuracy | {tm['base']['accuracy']:.4f} | {tm['extended']['accuracy']:.4f} | {tm['extended']['accuracy']-tm['base']['accuracy']:+.4f} |
| Balanced accuracy | {tm['base']['balanced_accuracy']:.4f} | {tm['extended']['balanced_accuracy']:.4f} | {tm['extended']['balanced_accuracy']-tm['base']['balanced_accuracy']:+.4f} |
| PR-AUC | {tm['base']['pr_auc']:.4f} | {tm['extended']['pr_auc']:.4f} | {tm['extended']['pr_auc']-tm['base']['pr_auc']:+.4f} |
| Precision | {tm['base']['precision']:.4f} | {tm['extended']['precision']:.4f} | {tm['extended']['precision']-tm['base']['precision']:+.4f} |
| Recall | {tm['base']['recall']:.4f} | {tm['extended']['recall']:.4f} | {tm['extended']['recall']-tm['base']['recall']:+.4f} |

Reused-test confusion matrices changed from `{tm['base']['confusion_matrix']}` to
`{tm['extended']['confusion_matrix']}`. Because zeros are unlabelled rather than
confirmed negatives, even a strong plateau is evidence for review rather than proof of
EV ownership. A future accuracy claim needs a newly labelled untouched cohort.
"""
    (OUT / "results.md").write_text(report)


if __name__ == "__main__":
    run()

