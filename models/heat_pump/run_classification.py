#!/usr/bin/env python3
"""Heat-pump (`has_Waermepumpe`) classification leaderboard.

Follows `ev_classification_total/run_classification.py`'s protocol (group-safe
split, sealed test) but reports only accuracy and the confusion matrix, per
the reviewer's instruction - no ROC-AUC/PR-AUC/F1/Brier here.

The held-out test partition is evaluated only after a validation champion and
operating threshold have been frozen. All split assignment happens by group_id.
"""
from __future__ import annotations

import pickle
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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

OUT = C.ARTIFACTS
SEED = C.SEED
LABEL = "has_Waermepumpe"
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


def write_json(path: Path, value) -> None:
    import json
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def load_data() -> pd.DataFrame:
    if not C.FEATURES.exists():
        raise FileNotFoundError(f"Run `python -m models.heat_pump.build_features` first: {C.FEATURES}")
    data = pd.read_csv(C.FEATURES, sep=C.CSV_SEP, dtype={"gp_nr": str, "plz": str})
    if not data[LABEL].isin([0, 1]).all():
        raise ValueError(f"{LABEL} must be binary")
    # A group_id legitimately carries two labels when its `augmented_before`
    # row is the pre-install snapshot for *this* technology (all-zero) and its
    # `cross_sectional` row is the post-install state (label 1) - see
    # data_addition/Readme.md on `window_anchored_to_install`. Only flag a
    # contradiction *within* the same sample_type, which would be a real bug.
    mixed = data.groupby(["group_id", "sample_type"])[LABEL].nunique()
    if (mixed > 1).any():
        raise ValueError("A group has contradictory heat-pump labels within one sample_type")
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
    scalar = [c for c in numeric if c not in hourly]
    sets = {"compact": scalar, "load_shape": scalar + hourly}
    for name, columns in sets.items():
        if not columns or set(columns) & PROHIBITED:
            raise AssertionError(f"Invalid feature set {name}")
    return sets


def primary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.sample_type.eq("cross_sectional")].copy()


def metrics(y, score, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }


def choose_threshold(y, score) -> float:
    candidates = np.unique(np.r_[0.0, score, 1.0])
    best = max(candidates, key=lambda t: (accuracy_score(y, score >= t), -abs(float(t) - 0.5)))
    return float(best)


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


def heuristic_scores(frame: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "wpb_score":
        return frame["wpb_score"].to_numpy()
    if kind == "dedicated_hp_meter":
        return frame["w_has_dedicated_hp_meter"].to_numpy(dtype=float)
    if kind == "winter_summer_ratio":
        return frame["winter_summer_ratio"].rank(pct=True).to_numpy()
    raise ValueError(kind)


def candidate_grid() -> list[Experiment]:
    return [
        Experiment("H20_logistic_compact", "logistic", "compact", {"C": 0.1},
                   "Regularized linear model on scalar seasonality/shape/meter-aux features."),
        Experiment("H21_hgb_compact", "hist_gradient_boosting", "compact",
                   {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 7, "l2_regularization": 5.0},
                   "Nonlinear interactions between winter-heavy load and dedicated-meter flags."),
        Experiment("H22_hgb_shape", "hist_gradient_boosting", "load_shape",
                   {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 7, "l2_regularization": 5.0},
                   "The mean hourly profile adds heating-cycle timing evidence."),
        Experiment("H23_lgbm_shape", "lightgbm", "load_shape",
                   {"learning_rate": 0.03, "n_estimators": 250, "num_leaves": 7,
                    "min_child_samples": 25, "reg_alpha": 0.5, "reg_lambda": 10.0},
                   "Gradient-boosted trees over the full compact+shape feature set."),
    ]


def diagnostic_plot(path: Path, y, score, threshold: float, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].hist(np.asarray(score)[np.asarray(y) == 0], bins=12, alpha=0.7, label="unlabelled/negative")
    axes[0].hist(np.asarray(score)[np.asarray(y) == 1], bins=12, alpha=0.7, label="HP-labelled")
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


def feature_importance(model, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Out-of-sample permutation importance in accuracy points.

    Model-agnostic, so logistic coefficients, HistGradientBoosting (which has
    no `feature_importances_`) and LightGBM gains stay comparable, and the
    unit is the metric this project reports.
    """
    result = permutation_importance(
        model, frame[columns], frame[LABEL],
        scoring="accuracy", n_repeats=20, random_state=SEED,
    )
    return (pd.DataFrame({"feature": columns,
                          "importance": result.importances_mean,
                          "importance_std": result.importances_std})
              .sort_values("importance", ascending=False).reset_index(drop=True))


def run() -> None:
    if (OUT / "leaderboard.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {OUT / 'leaderboard.json'}")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = load_data()
    split = make_split(data)
    split.to_csv(OUT / "split_manifest.csv", index=False)
    data = data.merge(split[["group_id", "split"]], on="group_id", how="left", validate="many_to_one")
    sets = feature_sets(data)
    train = data.loc[data.split.eq("development")].reset_index(drop=True)
    validation = primary_rows(data.loc[data.split.eq("validation")]).reset_index(drop=True)
    test = primary_rows(data.loc[data.split.eq("test")]).reset_index(drop=True)

    manifest = {
        "created_utc": pd.Timestamp.now("UTC").isoformat(), "seed": SEED,
        "source": str(C.FEATURES.relative_to(C.NEWSPAPER)), "rows": len(data),
        "groups": int(data.group_id.nunique()), "positive_rows": int(data[LABEL].sum()),
        "python": sys.version,
        "split": split.groupby(["split", LABEL]).size().unstack(fill_value=0).to_dict("index"),
        "protocol": "60/20/20 stratified group split; current cross-sectional validation/test; test sealed until freeze",
        "prohibited_features": sorted(PROHIBITED), "feature_sets": sets,
        "metrics_policy": "accuracy + confusion matrix only, per reviewer instruction",
    }
    write_json(OUT / "run_manifest.json", manifest)

    leaderboard = []
    fitted = {}

    prevalence = float(train[LABEL].mean())
    majority = 1 if prevalence >= 0.5 else 0
    dummy_score = np.full(len(validation), float(majority))
    result = metrics(validation[LABEL], dummy_score, 0.5)
    leaderboard.append({"id": "E00_dummy_majority", "family": "dummy", "features": "none", **result})

    for kind in ("wpb_score", "dedicated_hp_meter", "winter_summer_ratio"):
        eid = {"wpb_score": "H10_wpb_score", "dedicated_hp_meter": "H11_dedicated_meter",
               "winter_summer_ratio": "H12_winter_summer_ratio"}[kind]
        score = heuristic_scores(validation, kind)
        threshold = choose_threshold(validation[LABEL].to_numpy(), score)
        result = metrics(validation[LABEL], score, threshold)
        directory = OUT / eid
        directory.mkdir()
        write_json(directory / "config.json", {"kind": kind, "threshold": threshold})
        write_json(directory / "metrics_validation.json", result)
        diagnostic_plot(directory / "diagnostics.png", validation[LABEL], score, threshold, eid)
        leaderboard.append({"id": eid, "family": "heuristic", "features": kind, **result})

    for experiment in candidate_grid():
        columns = sets[experiment.features]
        model = model_for(experiment)
        model.fit(train[columns], train[LABEL])
        score = model.predict_proba(validation[columns])[:, 1]
        threshold = choose_threshold(validation[LABEL].to_numpy(), score)
        result = metrics(validation[LABEL], score, threshold)
        directory = OUT / experiment.id
        directory.mkdir()
        write_json(directory / "config.json", {**asdict(experiment), "columns": columns, "threshold": threshold})
        write_json(directory / "metrics_validation.json", result)
        with (directory / "model.pkl").open("wb") as stream:
            pickle.dump(model, stream)
        diagnostic_plot(directory / "diagnostics.png", validation[LABEL], score, threshold, experiment.id)
        leaderboard.append({"id": experiment.id, "family": experiment.family,
                            "features": experiment.features, "hypothesis": experiment.hypothesis, **result})
        fitted[experiment.id] = (experiment, threshold, model)

    leaderboard.sort(key=lambda row: row["accuracy"], reverse=True)
    for rank, row in enumerate(leaderboard, 1):
        row["rank"] = rank
    write_json(OUT / "leaderboard.json", leaderboard)
    pd.DataFrame(leaderboard).drop(columns=["confusion_matrix"], errors="ignore").to_csv(
        OUT / "leaderboard.csv", index=False)

    # Freeze the best supervised model (a heuristic may lead the descriptive
    # leaderboard, but the stable export must support fit/predict on new rows).
    supervised = [row for row in leaderboard if row["id"] in fitted]
    champion_row = max(supervised, key=lambda row: row["accuracy"])
    champion, frozen_threshold, champion_model = fitted[champion_row["id"]]
    columns = sets[champion.features]

    # Importance is measured out-of-sample: the development-fit champion
    # scored on validation, before the final refit sees those rows.
    importances = feature_importance(champion_model, validation, columns)
    importances.to_csv(OUT / "feature_importance.csv", index=False)

    fit_rows = data.loc[data.split.isin(["development", "validation"])].reset_index(drop=True)
    final_model = model_for(champion)
    final_model.fit(fit_rows[columns], fit_rows[LABEL])
    test_score = final_model.predict_proba(test[columns])[:, 1]
    test_result = metrics(test[LABEL], test_score, frozen_threshold)

    best = OUT / "best"
    best.mkdir()
    with (best / "pipeline.pkl").open("wb") as stream:
        pickle.dump(final_model, stream)
    write_json(best / "config.json", {**asdict(champion), "columns": columns, "threshold": frozen_threshold})
    write_json(best / "metrics_validation.json", champion_row)
    write_json(best / "metrics_test.json", test_result)
    pd.DataFrame({"sample_id": test.sample_id, "group_id": test.group_id,
                  "target": test[LABEL], "score": test_score,
                  "prediction": (test_score >= frozen_threshold).astype(int)}).to_csv(
        best / "test_predictions.csv", index=False)
    diagnostic_plot(best / "test_diagnostics.png", test[LABEL], test_score, frozen_threshold,
                     f"{champion.id} final test")
    freeze = {
        "champion": champion.id, "selection": "highest validation accuracy among supervised models",
        "threshold": frozen_threshold, "features": columns,
        "validation_metrics": champion_row, "test_metrics": test_result,
        "top_feature": importances.iloc[0]["feature"], "test_status": "opened_once",
    }
    write_json(OUT / "final_freeze.json", freeze)

    rows = [
        f'| {row["rank"]} | {row["id"]} | {row["family"]} | {row["features"]} | '
        f'{row["n"]} | {row["accuracy"]:.4f} | {row["confusion_matrix"]} |'
        for row in leaderboard
    ]
    results = f"""# Heat-pump (`has_Waermepumpe`) classification results

## Protocol

{len(data)} rows ({data.group_id.nunique()} property groups) from
`models/heat_pump/artifacts/heat_pump_features.csv`
(`train_property_samples_auxw_netload.csv` + WPB heuristic score +
`winter_share_over_night_share`). Permanent 60/20/20 split by `group_id`;
validation and test contain current cross-sectional properties only. IDs,
dates, window metadata, postal code, and all five technology labels were
excluded from features. `has_Waermepumpe = 0` is subsidy-register-unlabelled,
not verified heat-pump-absent, so accuracy is a floor estimate, not a true
accuracy measurement (`docs/data_problems.md`).

Per the reviewing instruction, only **accuracy** and the **confusion matrix**
(`[[TN, FP], [FN, TP]]` order) are used to compare and select models here -
no ROC-AUC, PR-AUC, F1, or Brier score.

## Validation leaderboard

| Rank | ID | Family | Feature set | N | Accuracy | Confusion matrix |
|---:|---|---|---|---:|---:|---|
{chr(10).join(rows)}

The frozen supervised champion is `{champion.id}`, selected by validation
accuracy. Its operating threshold `{frozen_threshold:.6f}` was chosen on
validation and held fixed. The final model was refit on development plus
validation before the test partition was opened once. Its most important
feature is `{importances.iloc[0]['feature']}`.

## Final held-out test

On {test_result['n']} current properties ({test_result['positives']}
HP-labelled), the frozen pipeline reached accuracy
**{test_result['accuracy']:.4f}** with confusion matrix
`{test_result['confusion_matrix']}` (`[[TN, FP], [FN, TP]]`).

## Interpretation and limitations

`data_addition/Readme.md` already found the generic feature table near-chance
(ROC-AUC 0.565) for this label and suspected the heat-pump load often sits on
a separate switched-tariff meter absent from the aggregate. This run adds the
`models.heat_pump_boiler` plateau detector's score and the dedicated-meter aux
columns, but the WPB detector targets hot-water boilers, not space heating, and
flags only 2 of 342 households here, so it contributes little signal. A true
weather/temperature-coupling feature (the PV-model analogue: regressing daily
import against heating-degree-days) needs per-household daily series
(`train_property_sequences.csv` / `load_property_daily.csv`), neither of which
is present in this checkout, and is therefore not implemented. Do not treat
this model's accuracy as a validated heat-pump detector; treat it as evidence
that, absent daily-resolution data or a metering split by tariff, this label is
hard to separate from the unlabelled population.

`population_results.md` carries the external check: how many heat pumps the
score implies across the unlabelled population, against the BFS GWR register.

Runtime: {time.perf_counter() - started:.1f} seconds.
"""
    (C.NEWSPAPER / "models" / "heat_pump" / "results.md").write_text(results)
    print(results)


if __name__ == "__main__":
    run()
