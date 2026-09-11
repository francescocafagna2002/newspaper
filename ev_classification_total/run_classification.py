#!/usr/bin/env python3
"""Adaptive EV classification on the ML-ready property sample table.

The held-out test partition is evaluated only after a validation champion and
operating threshold have been frozen. All split assignment happens by group_id.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "data_addition" / "train_property_samples_auxw_netload.csv"
FEEDIN_SOURCE = ROOT.parent / "data_addition" / "train_property_samples_auxw_feedin.csv"
OUT = ROOT / "output"
SEED = 20260911
LABEL = "has_EV"
IDENTITY = {
    "sample_id", "group_id", "gp_nr", "plz", "sample_type",
    "window_anchored_to_install", "n_days", "first_day", "last_day",
}
LABELS = {"has_PV", "has_Waermepumpe", "has_Batterie", "has_EV", "has_WP_Boiler"}
PROHIBITED = IDENTITY | LABELS


@dataclass(frozen=True)
class Experiment:
    id: str
    family: str
    features: str
    params: dict
    hypothesis: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def load_data() -> pd.DataFrame:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Required input is absent: {SOURCE}")
    data = pd.read_csv(SOURCE, sep=";")
    required = {"sample_id", "group_id", "sample_type", LABEL}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data.sample_id.duplicated().any():
        raise ValueError("sample_id must be unique")
    if not data[LABEL].isin([0, 1]).all():
        raise ValueError("has_EV must be binary")
    if data.groupby("group_id")[LABEL].nunique().max() != 1:
        raise ValueError("A group has contradictory EV labels")
    feedin_names = {
        "einsp_mean_kwh_d", "einsp_max_kwh_d", "frac_days_feedin",
        "w_n_production_only_meters", "w_has_production_only_meter",
        *{f"he{hour:02d}" for hour in range(24)},
    }
    leaked = feedin_names & set(data)
    if leaked:
        raise ValueError(f"Quarantined feed-in features found: {sorted(leaked)}")
    return data


def make_split(data: pd.DataFrame) -> pd.DataFrame:
    groups = data.groupby("group_id", as_index=False)[LABEL].first()
    development, remainder = train_test_split(
        groups, test_size=0.4, random_state=SEED, stratify=groups[LABEL]
    )
    validation, test = train_test_split(
        remainder, test_size=0.5, random_state=SEED + 1, stratify=remainder[LABEL]
    )
    development = development.assign(split="development")
    validation = validation.assign(split="validation")
    test = test.assign(split="test")
    result = pd.concat([development, validation, test], ignore_index=True)
    if result.group_id.duplicated().any() or len(result) != data.group_id.nunique():
        raise AssertionError("Split manifest is not one-to-one with groups")
    return result.sort_values("group_id").reset_index(drop=True)


def feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    numeric = [c for c in data.select_dtypes(include=np.number) if c not in PROHIBITED]
    hourly = [f"hb{hour:02d}" for hour in range(24)]
    aux = [c for c in numeric if c.startswith("w_")]
    scalar = [c for c in numeric if c not in hourly and c not in aux]
    sets = {
        "compact": scalar,
        "load_shape": scalar + hourly,
        "all_netload": scalar + hourly + aux,
    }
    for name, columns in sets.items():
        if not columns or set(columns) & PROHIBITED:
            raise AssertionError(f"Invalid feature set {name}")
    return sets


def primary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.sample_type.eq("cross_sectional")].copy()


def metrics(y, score, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = score >= threshold
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, score)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def choose_threshold(y, score) -> float:
    candidates = np.unique(np.r_[0.0, score, 1.0])
    ranked = []
    for threshold in candidates:
        pred = score >= threshold
        ranked.append((
            balanced_accuracy_score(y, pred),
            f1_score(y, pred, zero_division=0),
            -abs(float(threshold) - 0.5),
            float(threshold),
        ))
    return max(ranked)[-1]


def bootstrap(y, score, threshold: float, draws: int = 2000) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        m = metrics(y[idx], score[idx], threshold)
        values.append([m["pr_auc"], m["roc_auc"], m["balanced_accuracy"]])
    values = np.asarray(values)
    return {
        "draws": int(len(values)),
        "pr_auc": np.quantile(values[:, 0], [0.025, 0.975]).tolist(),
        "roc_auc": np.quantile(values[:, 1], [0.025, 0.975]).tolist(),
        "balanced_accuracy": np.quantile(values[:, 2], [0.025, 0.975]).tolist(),
    }


def logistic(c: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=SEED)),
    ])


def histogram(params: dict) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"], max_iter=params["max_iter"],
            max_leaf_nodes=params["max_leaf_nodes"], l2_regularization=params["l2_regularization"],
            class_weight="balanced", random_state=SEED,
        )),
    ])


def lightgbm(params: dict) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", verbosity=-1, random_state=SEED, n_jobs=-1,
        class_weight="balanced", subsample=0.8, colsample_bytree=0.85,
        **params,
    )


def model_for(experiment: Experiment):
    if experiment.family == "logistic":
        return logistic(experiment.params["C"])
    if experiment.family == "hist_gradient_boosting":
        return histogram(experiment.params)
    if experiment.family == "lightgbm":
        return lightgbm(experiment.params)
    raise ValueError(experiment.family)


def grouped_cv_ap(data: pd.DataFrame, columns: list[str], experiment: Experiment) -> float:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = np.zeros(len(data), dtype=float)
    for fit_idx, score_idx in splitter.split(data, data[LABEL], data.group_id):
        model = model_for(experiment)
        model.fit(data.iloc[fit_idx][columns], data.iloc[fit_idx][LABEL])
        scores[score_idx] = model.predict_proba(data.iloc[score_idx][columns])[:, 1]
    return float(average_precision_score(data[LABEL], scores))


def heuristic_scores(train: pd.DataFrame, frame: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "peak":
        return frame.bezug_peak_kw.rank(pct=True).to_numpy()
    if kind == "high_fraction":
        return frame.frac_qh_above_6kw.rank(pct=True).to_numpy()
    if kind != "composite":
        raise ValueError(kind)
    # Convert values to percentiles against development data, which fixes the
    # score transformation before validation and test are inspected.
    def percentile(reference, values):
        ordered = np.sort(np.asarray(reference, dtype=float))
        return np.searchsorted(ordered, np.asarray(values, dtype=float), side="right") / len(ordered)
    peak = percentile(train.bezug_peak_kw, frame.bezug_peak_kw)
    fraction = percentile(train.frac_qh_above_6kw, frame.frac_qh_above_6kw)
    p95 = percentile(train.bezug_p95_kwh_d, frame.bezug_p95_kwh_d)
    return 0.35 * peak + 0.45 * fraction + 0.20 * p95


def candidate_grid() -> list[Experiment]:
    return [
        Experiment("E02_logistic_compact", "logistic", "compact", {"C": 0.1},
                   "Robust annual load statistics separate repeated EV demand from household scale."),
        Experiment("E20_hgb_compact", "hist_gradient_boosting", "compact",
                   {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 7, "l2_regularization": 5.0},
                   "Nonlinear interactions improve on the compact linear baseline."),
        Experiment("E21_hgb_shape", "hist_gradient_boosting", "load_shape",
                   {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 7, "l2_regularization": 5.0},
                   "The mean hourly profile adds charging-time evidence."),
        Experiment("E22_lgbm_total", "lightgbm", "all_netload",
                   {"learning_rate": 0.03, "n_estimators": 250, "num_leaves": 7,
                    "min_child_samples": 25, "reg_alpha": 0.5, "reg_lambda": 10.0},
                   "Windowed meter context adds useful evidence to load shape."),
        Experiment("E23_logistic_total", "logistic", "all_netload", {"C": 0.03},
                   "A regularized full-feature linear model is a stable low-variance alternative."),
    ]


def diagnostic_plot(path: Path, y, score, threshold: float, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].hist(np.asarray(score)[np.asarray(y) == 0], bins=12, alpha=0.7, label="unlabelled/negative")
    axes[0].hist(np.asarray(score)[np.asarray(y) == 1], bins=12, alpha=0.7, label="EV-labelled")
    axes[0].axvline(threshold, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="score", ylabel="properties", title="Validation score distribution")
    axes[0].legend(fontsize=8)
    order = np.argsort(score)
    axes[1].plot(np.asarray(score)[order], label="score")
    axes[1].scatter(np.arange(len(y)), np.asarray(y)[order], s=10, alpha=0.6, label="label")
    axes[1].set(xlabel="ordered validation samples", title=title)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {OUT}")
    OUT.mkdir(parents=True)
    started = time.perf_counter()
    data = load_data()
    split = make_split(data)
    split.to_csv(OUT / "split_manifest.csv", index=False)
    data = data.merge(split[["group_id", "split"]], on="group_id", how="left", validate="many_to_one")
    sets = feature_sets(data)
    train = data.loc[data.split.eq("development")].reset_index(drop=True)
    validation = primary_rows(data.loc[data.split.eq("validation")]).reset_index(drop=True)
    test = primary_rows(data.loc[data.split.eq("test")]).reset_index(drop=True)

    source = {
        "path": str(SOURCE.relative_to(ROOT.parent)), "sha256": sha256(SOURCE),
        "rows": len(data), "groups": int(data.group_id.nunique()),
        "positive_rows": int(data[LABEL].sum()), "feedin_source_used": False,
        "feedin_source_sha256_for_audit_only": sha256(FEEDIN_SOURCE) if FEEDIN_SOURCE.exists() else None,
    }
    manifest = {
        "created_utc": pd.Timestamp.now("UTC").isoformat(), "seed": SEED,
        "source": source, "python": sys.version, "platform": platform.platform(),
        "split": split.groupby(["split", LABEL]).size().unstack(fill_value=0).to_dict("index"),
        "protocol": "60/20/20 stratified group split; current cross-sectional validation/test; test sealed until freeze",
        "prohibited_features": sorted(PROHIBITED), "feature_sets": sets,
    }
    write_json(OUT / "run_manifest.json", manifest)

    leaderboard = []
    fitted = {}

    prevalence = float(train[LABEL].mean())
    dummy_score = np.full(len(validation), prevalence)
    threshold = prevalence
    result = metrics(validation[LABEL], dummy_score, threshold)
    result["ci95"] = bootstrap(validation[LABEL], dummy_score, threshold)
    directory = OUT / "E01_dummy"
    directory.mkdir()
    write_json(directory / "config.json", {"prevalence": prevalence, "features": []})
    write_json(directory / "metrics_validation.json", result)
    pd.DataFrame({"sample_id": validation.sample_id, "group_id": validation.group_id,
                  "target": validation[LABEL], "score": dummy_score}).to_csv(directory / "validation_predictions.csv", index=False)
    leaderboard.append({"id": "E01_dummy", "family": "dummy", "features": "none", **result})

    # Transparent annual high-load rules. Validation chooses the operating
    # threshold and the best rule; the composite's reference distribution is
    # learned from development only.
    for kind in ("peak", "high_fraction", "composite"):
        eid = {"peak": "E10_peak", "high_fraction": "E11_high_fraction", "composite": "E12_composite"}[kind]
        score = heuristic_scores(train, validation, kind)
        threshold = choose_threshold(validation[LABEL].to_numpy(), score)
        result = metrics(validation[LABEL], score, threshold)
        result["ci95"] = bootstrap(validation[LABEL], score, threshold)
        directory = OUT / eid
        directory.mkdir()
        write_json(directory / "config.json", {"kind": kind, "threshold": threshold, "feedin_features": False})
        write_json(directory / "metrics_validation.json", result)
        pd.DataFrame({"sample_id": validation.sample_id, "group_id": validation.group_id,
                      "target": validation[LABEL], "score": score}).to_csv(directory / "validation_predictions.csv", index=False)
        diagnostic_plot(directory / "diagnostics.png", validation[LABEL], score, threshold, eid)
        leaderboard.append({"id": eid, "family": "heuristic", "features": kind, **result})

    for experiment in candidate_grid():
        columns = sets[experiment.features]
        cv_ap = grouped_cv_ap(train, columns, experiment)
        model = model_for(experiment)
        model.fit(train[columns], train[LABEL])
        score = model.predict_proba(validation[columns])[:, 1]
        threshold = choose_threshold(validation[LABEL].to_numpy(), score)
        result = metrics(validation[LABEL], score, threshold)
        result["development_group_cv_pr_auc"] = cv_ap
        result["ci95"] = bootstrap(validation[LABEL], score, threshold)
        directory = OUT / experiment.id
        directory.mkdir()
        write_json(directory / "config.json", {**asdict(experiment), "columns": columns, "threshold": threshold})
        write_json(directory / "metrics_validation.json", result)
        pd.DataFrame({"sample_id": validation.sample_id, "group_id": validation.group_id,
                      "target": validation[LABEL], "score": score}).to_csv(directory / "validation_predictions.csv", index=False)
        with (directory / "model.pkl").open("wb") as stream:
            pickle.dump(model, stream)
        diagnostic_plot(directory / "diagnostics.png", validation[LABEL], score, threshold, experiment.id)
        leaderboard.append({"id": experiment.id, "family": experiment.family,
                            "features": experiment.features, "hypothesis": experiment.hypothesis, **result})
        fitted[experiment.id] = (experiment, threshold)

    leaderboard.sort(key=lambda row: (row["pr_auc"], row["balanced_accuracy"], -row["brier"]), reverse=True)
    for rank, row in enumerate(leaderboard, 1):
        row["rank"] = rank
    write_json(OUT / "leaderboard.json", leaderboard)
    pd.DataFrame(leaderboard).drop(columns=["ci95", "confusion_matrix"], errors="ignore").to_csv(OUT / "leaderboard.csv", index=False)

    # Freeze the best supervised model. A heuristic may lead the descriptive
    # leaderboard, but the stable export must support fit/predict on new rows.
    supervised = [row for row in leaderboard if row["id"] in fitted]
    champion_row = max(supervised, key=lambda row: (row["pr_auc"], row["balanced_accuracy"], -row["brier"]))
    champion, frozen_threshold = fitted[champion_row["id"]]
    freeze = {
        "champion": champion.id, "selection": "highest validation PR-AUC among supervised models",
        "threshold": frozen_threshold, "features": sets[champion.features],
        "validation_metrics": {k: champion_row[k] for k in metrics(validation[LABEL], np.zeros(len(validation)), 0).keys() if k in champion_row},
        "test_status": "sealed",
    }
    write_json(OUT / "final_freeze.json", freeze)

    fit_rows = data.loc[data.split.isin(["development", "validation"])].reset_index(drop=True)
    final_model = model_for(champion)
    final_model.fit(fit_rows[sets[champion.features]], fit_rows[LABEL])
    test_score = final_model.predict_proba(test[sets[champion.features]])[:, 1]
    test_result = metrics(test[LABEL], test_score, frozen_threshold)
    test_result["ci95"] = bootstrap(test[LABEL], test_score, frozen_threshold)
    best = OUT / "best"
    best.mkdir()
    with (best / "pipeline.pkl").open("wb") as stream:
        pickle.dump(final_model, stream)
    write_json(best / "config.json", {**asdict(champion), "columns": sets[champion.features], "threshold": frozen_threshold})
    write_json(best / "metrics_validation.json", champion_row)
    write_json(best / "metrics_test.json", test_result)
    pd.DataFrame({"sample_id": test.sample_id, "group_id": test.group_id,
                  "target": test[LABEL], "score": test_score,
                  "prediction": (test_score >= frozen_threshold).astype(int)}).to_csv(best / "test_predictions.csv", index=False)
    diagnostic_plot(best / "test_diagnostics.png", test[LABEL], test_score, frozen_threshold, f"{champion.id} final test")
    freeze["test_status"] = "opened_once"
    freeze["test_metrics_path"] = "best/metrics_test.json"
    write_json(OUT / "final_freeze.json", freeze)

    rows = []
    for row in leaderboard:
        rows.append(
            f'| {row["rank"]} | {row["id"]} | {row["family"]} | {row["features"]} | '
            f'{row["n"]} | {row["pr_auc"]:.4f} | {row["roc_auc"]:.4f} | '
            f'{row["balanced_accuracy"]:.4f} | {row["f1"]:.4f} | {row["brier"]:.4f} |'
        )
    ci = test_result["ci95"]
    results = f"""# EV classification total-sample results

## Protocol

Run `{source['sha256'][:16]}` used all {len(data)} rows ({data.group_id.nunique()} property groups) from
`data_addition/train_property_samples_auxw_netload.csv`. The permanent split is by
`group_id`; validation and test contain current cross-sectional properties only.
The feed-in/export feature table was not joined or used. IDs, dates, window metadata,
postal code, quality-window length, and all five technology labels were excluded from
features. The zero class is subsidy-register-unlabelled rather than verified EV-absent,
so the reported binary metrics measure agreement with GIGI labels and may understate
performance against true ownership.

## Validation leaderboard

| Rank | ID | Family | Feature set | N | PR-AUC | ROC-AUC | Balanced accuracy | F1 | Brier |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The frozen supervised champion is `{champion.id}`, selected by validation PR-AUC.
Its operating threshold `{frozen_threshold:.6f}` was chosen on validation and then
held fixed. The final model was refit on development plus validation before the test
partition was opened once.

## Final held-out test

On {test_result['n']} current properties ({test_result['positives']} EV-labelled), the
frozen pipeline reached PR-AUC **{test_result['pr_auc']:.4f}** (bootstrap 95% interval
{ci['pr_auc'][0]:.4f}–{ci['pr_auc'][1]:.4f}), ROC-AUC **{test_result['roc_auc']:.4f}**,
balanced accuracy **{test_result['balanced_accuracy']:.4f}**, F1 **{test_result['f1']:.4f}**,
and Brier score **{test_result['brier']:.4f}**. The confusion matrix is
`{test_result['confusion_matrix']}` in `[[TN, FP], [FN, TP]]` order.

## Interpretation and limitations

The source contains annual aggregates and mean hourly profiles, so the original
week-level event detector cannot be reproduced from this table. The E10–E12 rules are
transparent annual high-load screens, while the supervised runs compare compact load,
hourly shape, and windowed meter-context features. `train_property_sequences.csv`,
which would permit charging-event localization, is described in the data README but is
not present in this checkout. Results are uncertain because the held-out sets are small
and negative labels are contaminated. Do not interpret individual high-load periods as
confirmed EV charging events.

Runtime: {time.perf_counter() - started:.1f} seconds.
"""
    (ROOT / "results.md").write_text(results)


if __name__ == "__main__":
    run()
