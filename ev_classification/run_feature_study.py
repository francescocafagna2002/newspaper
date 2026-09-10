#!/usr/bin/env python3
"""Run the post-test, development-tuned EV charging-behavior feature study."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .classification_core import (
        bootstrap, choose_threshold, metrics, primary_cohort, temporal_labels,
    )
    from .feature_study import build_customer_features
except ImportError:
    from classification_core import (
        bootstrap, choose_threshold, metrics, primary_cohort, temporal_labels,
    )
    from feature_study import build_customer_features


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "classification_output"
OUT = ROOT / "feature_study_output"
SEED = 20260910
FOLDS = 5
BOOTSTRAPS = 1000
LOGISTIC_C = (0.01, 0.1, 1.0, 10.0)
CLASS_WEIGHTS = (None, "balanced")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x), allow_nan=False) + "\n")


def model_columns(customers):
    columns = [column for column in customers if column not in {"gp_nr", "target", "weeks"}]
    assert columns and all(column.startswith("customer_") for column in columns)
    return columns


class FeatureStudy:
    def __init__(self):
        OUT.mkdir(exist_ok=True)
        self.original_manifest = read_json(ORIGINAL / "run_manifest.json")
        assert self.original_manifest["test_status"] == "evaluated_once", "Original study must be complete"
        self.source_hash = self.original_manifest["source"]["hash"]
        self.data = pd.read_parquet(ROOT / "output/customer_week_table.parquet").sort_values(["gp_nr", "week_start"]).reset_index(drop=True)
        self.data["temporal_target"], _, _ = temporal_labels(self.data)
        split = pd.read_parquet(ORIGINAL / "split_manifest.parquet")
        assert split.gp_nr.is_unique
        self.data["split"] = self.data.gp_nr.map(split.set_index("gp_nr").split)
        assert self.data.split.notna().all()
        # The original final-test files are never read. Feature computation is limited here.
        self.data = self.data.loc[self.data.split.isin(["development", "validation"])].copy()
        self.dev_weeks, self.dev_cohort = primary_cohort(self.data.loc[self.data.split.eq("development")], cap=12, minimum=2)
        self.val_weeks, self.val_cohort = primary_cohort(self.data.loc[self.data.split.eq("validation")], cap=12, minimum=2)
        self.dev_cohort = self.dev_cohort.loc[self.dev_cohort.status.eq("ok")].reset_index(drop=True)
        self.val_cohort = self.val_cohort.loc[self.val_cohort.status.eq("ok")].reset_index(drop=True)
        self.champion = pd.read_parquet(ORIGINAL / "E02_logistic/customer_validation.parquet").sort_values("gp_nr").reset_index(drop=True)
        assert self.val_cohort.gp_nr.tolist() == self.champion.gp_nr.tolist(), "Validation cohort differs from E02"
        assert self.val_cohort.target.tolist() == self.champion.target.tolist(), "Validation targets differ from E02"
        self.feature_cache = {}
        self._initialize()

    def _initialize(self):
        manifest_path = OUT / "run_manifest.json"
        code_paths = [ROOT / "feature_study.py", ROOT / "run_feature_study.py", ROOT / "plan_classify_features.md"]
        code_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}
        identity = {
            "created_utc": datetime.now(UTC).isoformat(),
            "source_identity": self.source_hash,
            "original_run_identity": self.original_manifest["run_identity"],
            "original_split_hash": hashlib.sha256((ORIGINAL / "split_manifest.parquet").read_bytes()).hexdigest(),
            "seed": SEED,
            "inner_folds": FOLDS,
            "history_cap": 12,
            "minimum_history": 2,
            "code_hashes": code_hashes,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "python": sys.version,
            "evaluation_status": "post-test validation reuse; original final test prohibited",
        }
        identity["run_identity"] = digest(identity)
        if manifest_path.exists():
            existing = read_json(manifest_path)
            for key in ("source_identity", "original_run_identity", "original_split_hash", "seed", "inner_folds", "history_cap", "minimum_history", "code_hashes"):
                assert existing[key] == identity[key], f"Feature-study identity changed: {key}"
        else:
            write_json(manifest_path, identity)
        folds_path = OUT / "development_folds.parquet"
        if folds_path.exists():
            self.folds = pd.read_parquet(folds_path).sort_values("gp_nr").reset_index(drop=True)
        else:
            splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
            fold = np.empty(len(self.dev_cohort), dtype=int)
            for number, (_, validation) in enumerate(splitter.split(self.dev_cohort.gp_nr, self.dev_cohort.target)):
                fold[validation] = number
            self.folds = self.dev_cohort[["gp_nr", "target"]].copy()
            self.folds["fold"] = fold
            self.folds.to_parquet(folds_path, index=False)
        assert self.folds.gp_nr.tolist() == self.dev_cohort.gp_nr.tolist()
        journal = ROOT / "results_feature_study.md"
        if not journal.exists():
            journal.write_text(
                "# EV charging-behavior feature study\n\n"
                "## Protocol\n\n"
                f"Run `{read_json(manifest_path)['run_identity']}` reuses source `{self.source_hash}` and the original permanent split. "
                f"The development cohort contains {len(self.dev_cohort)} customers ({int(self.dev_cohort.target.sum())} positives); "
                f"the unchanged validation cohort contains {len(self.val_cohort)} customers ({int(self.val_cohort.target.sum())} positives). "
                "All feature and model parameters are selected by five-fold customer-level cross-validation inside development. "
                "Each locked candidate is compared with frozen `E02_logistic` on reused validation customers. The original final test is not read or rescored.\n\n"
                "Literature basis: [Vavouris et al.](https://doi.org/10.3390/en15062200), "
                "[Li et al.](https://doi.org/10.1016/j.epsr.2024.110789), "
                "[Neubert et al.](https://doi.org/10.3390/en15134922), "
                "[Hangawatta et al.](https://doi.org/10.1016/j.segan.2025.101903), and "
                "[Hoffmann et al.](https://hdl.handle.net/11250/2618624).\n\n"
                "## Validation-reuse leaderboard\n\n<!-- LEADERBOARD -->\nPending.\n<!-- END LEADERBOARD -->\n\n"
                "## Experiment log\n"
            )
        self.journal = journal

    def customer_features(self, split, family, config):
        key = (split, family, digest(config))
        if key not in self.feature_cache:
            weeks = self.dev_weeks if split == "development" else self.val_weeks
            customers, weekly = build_customer_features(weeks, family, config)
            expected = self.dev_cohort if split == "development" else self.val_cohort
            assert customers.gp_nr.tolist() == expected.gp_nr.tolist()
            assert customers.target.tolist() == expected.target.tolist()
            self.feature_cache[key] = (customers, weekly)
        return self.feature_cache[key]

    def logistic_search(self, family, feature_configs, c_values=LOGISTIC_C):
        search = []
        winner = None
        for feature_config in feature_configs:
            customers, _ = self.customer_features("development", family, feature_config)
            columns = model_columns(customers)
            x = customers[columns]
            for c_value, class_weight in itertools.product(c_values, CLASS_WEIGHTS):
                oof = np.zeros(len(customers), dtype=float)
                fold_ap = []
                for fold in range(FOLDS):
                    validation = self.folds.fold.eq(fold).to_numpy()
                    training = ~validation
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=c_value, class_weight=class_weight, max_iter=3000, random_state=SEED),
                    )
                    model.fit(x.loc[training], customers.target.loc[training])
                    oof[validation] = model.predict_proba(x.loc[validation])[:, 1]
                    fold_ap.append(float(average_precision_score(customers.target.loc[validation], oof[validation])))
                threshold = choose_threshold(customers.target, oof)
                record = {
                    "feature_config": feature_config,
                    "C": c_value,
                    "class_weight": class_weight,
                    "features": len(columns),
                    "fold_pr_auc": fold_ap,
                    "mean_fold_pr_auc": float(np.mean(fold_ap)),
                    "std_fold_pr_auc": float(np.std(fold_ap)),
                    "pooled_oof_metrics": metrics(customers.target, oof, threshold),
                    "threshold": threshold,
                }
                search.append(record)
                key = (record["mean_fold_pr_auc"], -record["std_fold_pr_auc"], -len(columns), -c_value, class_weight is None)
                if winner is None or key > winner[0]:
                    winner = (key, record, customers, x, oof)
        return winner[1:], search

    def finish_logistic(self, experiment, family, hypothesis, feature_configs, c_values=LOGISTIC_C):
        directory = OUT / experiment
        if directory.exists():
            return read_json(directory / "config.json")
        started = time.perf_counter()
        (choice, development, x_dev, oof), search = self.logistic_search(family, feature_configs, c_values)
        validation, weekly_validation = self.customer_features("validation", family, choice["feature_config"])
        columns = model_columns(development)
        assert columns == model_columns(validation)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=choice["C"], class_weight=choice["class_weight"], max_iter=3000, random_state=SEED),
        )
        model.fit(x_dev, development.target)
        score = model.predict_proba(validation[columns])[:, 1]
        config = {
            "experiment": experiment,
            "kind": "logistic",
            "family": family,
            "feature_config": choice["feature_config"],
            "C": choice["C"],
            "class_weight": choice["class_weight"],
            "threshold": choice["threshold"],
            "selection": "mean five-fold development customer AP; threshold from pooled development OOF",
        }
        self._write_experiment(directory, config, hypothesis, development, validation, weekly_validation, columns, model, score, search, started, oof)
        coefficients = model.named_steps["logisticregression"].coef_[0]
        pd.DataFrame({"feature": columns, "coefficient": coefficients, "absolute_coefficient": np.abs(coefficients)}).sort_values("absolute_coefficient", ascending=False).to_csv(directory / "coefficients.csv", index=False)
        return config

    def finish_lightgbm(self, timed_config, distribution_config):
        experiment = "F43_combined_lgbm"
        directory = OUT / experiment
        if directory.exists():
            return read_json(directory / "config.json")
        started = time.perf_counter()
        family = "combined"
        feature_config = {"timed": timed_config, "distribution": distribution_config}
        development, _ = self.customer_features("development", family, feature_config)
        validation, weekly_validation = self.customer_features("validation", family, feature_config)
        columns = model_columns(development)
        x_dev = development[columns]
        grid = [
            {"num_leaves": 3, "learning_rate": 0.03, "n_estimators": 120, "min_child_samples": 20, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 10.0},
            {"num_leaves": 5, "learning_rate": 0.03, "n_estimators": 160, "min_child_samples": 15, "colsample_bytree": 0.7, "reg_alpha": 2.0, "reg_lambda": 15.0},
            {"num_leaves": 7, "learning_rate": 0.02, "n_estimators": 200, "min_child_samples": 10, "colsample_bytree": 0.6, "reg_alpha": 3.0, "reg_lambda": 20.0},
        ]
        search = []
        winner = None
        for parameters, class_weight in itertools.product(grid, CLASS_WEIGHTS):
            oof = np.zeros(len(development), dtype=float)
            fold_ap = []
            for fold in range(FOLDS):
                valid = self.folds.fold.eq(fold).to_numpy(); train = ~valid
                model = LGBMClassifier(objective="binary", random_state=SEED, n_jobs=2, verbosity=-1, class_weight=class_weight, **parameters)
                model.fit(x_dev.loc[train], development.target.loc[train])
                oof[valid] = model.predict_proba(x_dev.loc[valid])[:, 1]
                fold_ap.append(float(average_precision_score(development.target.loc[valid], oof[valid])))
            threshold = choose_threshold(development.target, oof)
            record = {
                "parameters": parameters, "class_weight": class_weight,
                "fold_pr_auc": fold_ap, "mean_fold_pr_auc": float(np.mean(fold_ap)),
                "std_fold_pr_auc": float(np.std(fold_ap)), "threshold": threshold,
                "pooled_oof_metrics": metrics(development.target, oof, threshold),
            }
            search.append(record)
            key = (record["mean_fold_pr_auc"], -record["std_fold_pr_auc"], -parameters["num_leaves"], class_weight is None)
            if winner is None or key > winner[0]:
                winner = (key, record, oof)
        choice, oof = winner[1:]
        model = LGBMClassifier(objective="binary", random_state=SEED, n_jobs=2, verbosity=-1, class_weight=choice["class_weight"], **choice["parameters"])
        model.fit(x_dev, development.target)
        score = model.predict_proba(validation[columns])[:, 1]
        config = {
            "experiment": experiment, "kind": "lightgbm", "family": family,
            "feature_config": feature_config, "parameters": choice["parameters"],
            "class_weight": choice["class_weight"], "threshold": choice["threshold"],
            "selection": "mean five-fold development customer AP; threshold from pooled development OOF",
        }
        self._write_experiment(directory, config, "Nonlinear interactions among locked behavior features improve customer EV ranking", development, validation, weekly_validation, columns, model, score, search, started, oof)
        pd.DataFrame({
            "feature": columns,
            "gain": model.booster_.feature_importance("gain"),
            "split": model.booster_.feature_importance("split"),
        }).sort_values("gain", ascending=False).to_csv(directory / "importance.csv", index=False)
        model.booster_.save_model(str(directory / "model.txt"))
        return config

    def _write_experiment(self, directory, config, hypothesis, development, validation, weekly_validation, columns, model, score, search, started, oof):
        directory.mkdir()
        assert validation.gp_nr.tolist() == self.champion.gp_nr.tolist()
        threshold = config["threshold"]
        validation_predictions = validation[["gp_nr", "target", "weeks"]].copy()
        validation_predictions["score"] = score
        validation_predictions["decision"] = validation_predictions.score.ge(threshold).astype(int)
        paired = bootstrap(validation.target, score, threshold, seed=SEED, n=BOOTSTRAPS, other=self.champion.score)
        result = {
            "status": "validation reuse",
            "customer": metrics(validation.target, score, threshold),
            "ci95": bootstrap(validation.target, score, threshold, seed=SEED, n=BOOTSTRAPS),
            "paired_vs_E02_logistic": paired,
            "development_oof": metrics(development.target, oof, threshold),
            "runtime_seconds": time.perf_counter() - started,
            "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "promoted_over_E02": paired["ci95"][0] > 0,
        }
        write_json(directory / "config.json", config)
        write_json(directory / "cv_search.json", search)
        write_json(directory / "metrics_validation.json", result)
        write_json(directory / "feature_manifest.json", {
            "features": columns,
            "feature_hash": digest(columns),
            "signal_only_check_passed": all(column.startswith("customer_") for column in columns),
            "excluded_columns": ["gp_nr", "target", "weeks"],
        })
        development.to_parquet(directory / "customer_features_development.parquet", index=False)
        validation.to_parquet(directory / "customer_features_validation.parquet", index=False)
        weekly_validation.to_parquet(directory / "weekly_features_validation.parquet", index=False)
        validation_predictions.to_parquet(directory / "customer_validation.parquet", index=False)
        with (directory / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        self._record(config["experiment"], config["kind"], hypothesis, config, result)

    def _record(self, experiment, family, hypothesis, config, result):
        leaderboard_path = OUT / "leaderboard.json"
        leaderboard = read_json(leaderboard_path) if leaderboard_path.exists() else []
        entry = {
            "id": experiment, "family": family, "hypothesis": hypothesis,
            "metrics": result["customer"], "ci95": result["ci95"],
            "paired_vs_E02_logistic": result["paired_vs_E02_logistic"],
            "development_oof": result["development_oof"],
            "runtime_seconds": result["runtime_seconds"],
            "promoted_over_E02": result["promoted_over_E02"],
        }
        leaderboard.append(entry); write_json(leaderboard_path, leaderboard)
        metric = result["customer"]; interval = result["ci95"]["pr_auc"]; paired = result["paired_vs_E02_logistic"]
        with self.journal.open("a") as handle:
            handle.write(
                f"\n### {experiment}\n\n"
                f"- Hypothesis: {hypothesis}.\n"
                f"- Development tuning: {config['selection']}; selected parameters `{json.dumps(config, default=str)}`.\n"
                f"- Result (validation reuse): AP {metric['pr_auc']:.4f} [{interval[0]:.4f}, {interval[1]:.4f}], "
                f"balanced accuracy {metric['balanced_accuracy']:.4f}, F1 {metric['f1']:.4f}, Brier {metric['brier']:.4f}.\n"
                f"- Paired AP versus E02: delta {paired['delta_pr_auc']:.4f} [{paired['ci95'][0]:.4f}, {paired['ci95'][1]:.4f}].\n"
                f"- Decision: {'promoted beyond paired uncertainty' if result['promoted_over_E02'] else 'not promoted; paired uncertainty includes zero'}.\n"
                f"- Artifacts: [configuration](feature_study_output/{experiment}/config.json), "
                f"[CV search](feature_study_output/{experiment}/cv_search.json), "
                f"[metrics](feature_study_output/{experiment}/metrics_validation.json).\n"
            )
        self._update_leaderboard(leaderboard)
        print(experiment, json.dumps({"validation_ap": metric["pr_auc"], "paired": paired, "promoted": result["promoted_over_E02"]}), flush=True)

    def _update_leaderboard(self, entries):
        e02 = read_json(ORIGINAL / "E02_logistic/metrics_validation.json")["customer"]
        rows = [{
            "id": "E02_logistic", "family": "frozen baseline", "metrics": e02,
            "paired_vs_E02_logistic": {"delta_pr_auc": 0.0, "ci95": [0.0, 0.0]},
            "promoted_over_E02": False,
        }, *entries]
        lines = [
            "| Rank | ID | Family | Customers | PR-AUC | Balanced accuracy | F1 | Brier | AP delta vs E02 | Paired 95% interval | Promoted |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for rank, entry in enumerate(sorted(rows, key=lambda value: -value["metrics"]["pr_auc"]), 1):
            metric = entry["metrics"]; paired = entry["paired_vs_E02_logistic"]
            lines.append(
                f"| {rank} | {entry['id']} | {entry['family']} | {metric['customers_or_weeks']} | {metric['pr_auc']:.4f} | "
                f"{metric['balanced_accuracy']:.4f} | {metric['f1']:.4f} | {metric['brier']:.4f} | "
                f"{paired['delta_pr_auc']:.4f} | [{paired['ci95'][0]:.4f}, {paired['ci95'][1]:.4f}] | "
                f"{'yes' if entry['promoted_over_E02'] else 'no'} |"
            )
        text = self.journal.read_text(); before, rest = text.split("<!-- LEADERBOARD -->"); _, after = rest.split("<!-- END LEADERBOARD -->")
        self.journal.write_text(before + "<!-- LEADERBOARD -->\n" + "\n".join(lines) + "\n<!-- END LEADERBOARD -->" + after)

    @staticmethod
    def timed_grid():
        return [
            {"window": window, "ramp_kw": ramp, "persistence_hours": persistence}
            for window, ramp, persistence in itertools.product(
                ("late_afternoon_night", "evening_overnight", "night"),
                (1.0, 2.0, 3.0),
                (1.0, 1.5, 2.0),
            )
        ]

    @staticmethod
    def distribution_grid():
        return [
            {"window": window, "power_floor_kw": floor, "level_tolerance_kw": tolerance}
            for window, floor, tolerance in itertools.product(
                ("late_afternoon_night", "evening_overnight", "night"),
                (1.5, 3.0, 5.0),
                (0.5, 1.0),
            )
        ]

    def run(self):
        f40 = self.finish_logistic(
            "F40_timed_ramp_logistic", "timed",
            "Evening/night positive ramps followed by persistent load and a paired shutoff distinguish EV charging",
            self.timed_grid(),
        )
        f41 = self.finish_logistic(
            "F41_power_multiscale_logistic", "distribution",
            "Time-localized high-load distributions, recurring levels, and multiscale changes distinguish EV homes",
            self.distribution_grid(),
        )
        combined = {"timed": f40["feature_config"], "distribution": f41["feature_config"]}
        self.finish_logistic(
            "F42_combined_logistic", "combined",
            "Locked charging-behavior features add customer-history evidence to compact load statistics",
            [combined],
        )
        self.finish_lightgbm(combined["timed"], combined["distribution"])
        self.conclude()

    def conclude(self):
        marker = "## Recommendation"
        if marker in self.journal.read_text():
            return
        leaderboard = read_json(OUT / "leaderboard.json")
        best = max(leaderboard, key=lambda row: row["metrics"]["pr_auc"])
        promoted = [row for row in leaderboard if row["promoted_over_E02"]]
        if promoted:
            recommendation = max(promoted, key=lambda row: row["metrics"]["pr_auc"])["id"]
            statement = f"Promote `{recommendation}` as the new validation champion because its paired AP interval is wholly above zero."
        else:
            recommendation = "E02_logistic"
            statement = "Retain `E02_logistic`; no new feature classifier established a paired AP improvement beyond uncertainty."
        with self.journal.open("a") as handle:
            handle.write(
                f"\n## Recommendation\n\n{statement} The largest new validation AP point estimate is "
                f"`{best['id']}` at {best['metrics']['pr_auc']:.4f}. These are reused-validation results after the original test was opened, "
                "so they support feature selection and a future study, not a new unbiased performance claim. A future final estimate requires a genuinely untouched customer cohort.\n"
            )
        write_json(OUT / "recommendation.json", {"recommended": recommendation, "best_new_point_estimate": best["id"], "validation_reuse": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    FeatureStudy().run()


if __name__ == "__main__":
    main()
