# EV charging-behavior feature study

## Protocol

Run `4d4d77f8d2262cb267044cd7e16bb806b72609fdd58603d826b3f35b5ffff377` reuses source `69a33762f866aeca210389e6866fedd3aa01c65af74cfe72b4225c92f2658fbc` and the original permanent split. The development cohort contains 183 customers (26 positives); the unchanged validation cohort contains 61 customers (9 positives). All feature and model parameters are selected by five-fold customer-level cross-validation inside development. Each locked candidate is compared with frozen `E02_logistic` on reused validation customers. The original final test is not read or rescored.

Literature basis: [Vavouris et al.](https://doi.org/10.3390/en15062200), [Li et al.](https://doi.org/10.1016/j.epsr.2024.110789), [Neubert et al.](https://doi.org/10.3390/en15134922), [Hangawatta et al.](https://doi.org/10.1016/j.segan.2025.101903), and [Hoffmann et al.](https://hdl.handle.net/11250/2618624).

## Validation-reuse leaderboard

<!-- LEADERBOARD -->
| Rank | ID | Family | Customers | PR-AUC | Balanced accuracy | F1 | Brier | AP delta vs E02 | Paired 95% interval | Promoted |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | F41_power_multiscale_logistic | logistic | 61 | 0.5156 | 0.8120 | 0.5833 | 0.1132 | 0.1007 | [-0.2769, 0.3937] | no |
| 2 | F42_combined_logistic | logistic | 61 | 0.4171 | 0.7105 | 0.4762 | 0.1124 | 0.0021 | [-0.3261, 0.2849] | no |
| 3 | E02_logistic | frozen baseline | 61 | 0.4150 | 0.8291 | 0.5517 | 0.1157 | 0.0000 | [0.0000, 0.0000] | no |
| 4 | F43_combined_lgbm | lightgbm | 61 | 0.3979 | 0.7393 | 0.5556 | 0.1821 | -0.0171 | [-0.3376, 0.3409] | no |
| 5 | F40_timed_ramp_logistic | logistic | 61 | 0.3438 | 0.6838 | 0.4706 | 0.1184 | -0.0711 | [-0.3661, 0.1799] | no |
<!-- END LEADERBOARD -->

## Experiment log

### F40_timed_ramp_logistic

- Hypothesis: Evening/night positive ramps followed by persistent load and a paired shutoff distinguish EV charging.
- Development tuning: mean five-fold development customer AP; threshold from pooled development OOF; selected parameters `{"experiment": "F40_timed_ramp_logistic", "kind": "logistic", "family": "timed", "feature_config": {"window": "evening_overnight", "ramp_kw": 2.0, "persistence_hours": 1.5}, "C": 0.01, "class_weight": null, "threshold": 0.17130060976044925, "selection": "mean five-fold development customer AP; threshold from pooled development OOF"}`.
- Result (validation reuse): AP 0.3438 [0.1513, 0.7013], balanced accuracy 0.6838, F1 0.4706, Brier 0.1184.
- Paired AP versus E02: delta -0.0711 [-0.3661, 0.1799].
- Decision: not promoted; paired uncertainty includes zero.
- Artifacts: [configuration](feature_study_output/F40_timed_ramp_logistic/config.json), [CV search](feature_study_output/F40_timed_ramp_logistic/cv_search.json), [metrics](feature_study_output/F40_timed_ramp_logistic/metrics_validation.json).

### F41_power_multiscale_logistic

- Hypothesis: Time-localized high-load distributions, recurring levels, and multiscale changes distinguish EV homes.
- Development tuning: mean five-fold development customer AP; threshold from pooled development OOF; selected parameters `{"experiment": "F41_power_multiscale_logistic", "kind": "logistic", "family": "distribution", "feature_config": {"window": "late_afternoon_night", "power_floor_kw": 5.0, "level_tolerance_kw": 1.0}, "C": 0.01, "class_weight": null, "threshold": 0.16358906506530152, "selection": "mean five-fold development customer AP; threshold from pooled development OOF"}`.
- Result (validation reuse): AP 0.5156 [0.2376, 0.8051], balanced accuracy 0.8120, F1 0.5833, Brier 0.1132.
- Paired AP versus E02: delta 0.1007 [-0.2769, 0.3937].
- Decision: not promoted; paired uncertainty includes zero.
- Artifacts: [configuration](feature_study_output/F41_power_multiscale_logistic/config.json), [CV search](feature_study_output/F41_power_multiscale_logistic/cv_search.json), [metrics](feature_study_output/F41_power_multiscale_logistic/metrics_validation.json).

### F42_combined_logistic

- Hypothesis: Locked charging-behavior features add customer-history evidence to compact load statistics.
- Development tuning: mean five-fold development customer AP; threshold from pooled development OOF; selected parameters `{"experiment": "F42_combined_logistic", "kind": "logistic", "family": "combined", "feature_config": {"timed": {"window": "evening_overnight", "ramp_kw": 2.0, "persistence_hours": 1.5}, "distribution": {"window": "late_afternoon_night", "power_floor_kw": 5.0, "level_tolerance_kw": 1.0}}, "C": 0.01, "class_weight": null, "threshold": 0.1769400162621682, "selection": "mean five-fold development customer AP; threshold from pooled development OOF"}`.
- Result (validation reuse): AP 0.4171 [0.1830, 0.7567], balanced accuracy 0.7105, F1 0.4762, Brier 0.1124.
- Paired AP versus E02: delta 0.0021 [-0.3261, 0.2849].
- Decision: not promoted; paired uncertainty includes zero.
- Artifacts: [configuration](feature_study_output/F42_combined_logistic/config.json), [CV search](feature_study_output/F42_combined_logistic/cv_search.json), [metrics](feature_study_output/F42_combined_logistic/metrics_validation.json).

### F43_combined_lgbm

- Hypothesis: Nonlinear interactions among locked behavior features improve customer EV ranking.
- Development tuning: mean five-fold development customer AP; threshold from pooled development OOF; selected parameters `{"experiment": "F43_combined_lgbm", "kind": "lightgbm", "family": "combined", "feature_config": {"timed": {"window": "evening_overnight", "ramp_kw": 2.0, "persistence_hours": 1.5}, "distribution": {"window": "late_afternoon_night", "power_floor_kw": 5.0, "level_tolerance_kw": 1.0}}, "parameters": {"num_leaves": 3, "learning_rate": 0.03, "n_estimators": 120, "min_child_samples": 20, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 10.0}, "class_weight": "balanced", "threshold": 0.5419654036958813, "selection": "mean five-fold development customer AP; threshold from pooled development OOF"}`.
- Result (validation reuse): AP 0.3979 [0.1899, 0.7424], balanced accuracy 0.7393, F1 0.5556, Brier 0.1821.
- Paired AP versus E02: delta -0.0171 [-0.3376, 0.3409].
- Decision: not promoted; paired uncertainty includes zero.
- Artifacts: [configuration](feature_study_output/F43_combined_lgbm/config.json), [CV search](feature_study_output/F43_combined_lgbm/cv_search.json), [metrics](feature_study_output/F43_combined_lgbm/metrics_validation.json).

## Recommendation

Retain `E02_logistic`; no new feature classifier established a paired AP improvement beyond uncertainty. The largest new validation AP point estimate is `F41_power_multiscale_logistic` at 0.5156, using a late-afternoon-through-night window, a 5 kW floor, and 1 kW power-level tolerance. Its development out-of-fold AP was only 0.1650 (ROC-AUC 0.4993), so the higher reused-validation result is unstable and should be treated as a hypothesis for new data rather than evidence of reliable generalization. These are reused-validation results after the original test was opened, so they support feature selection and a future study, not a new unbiased performance claim. A future final estimate requires a genuinely untouched customer cohort.
