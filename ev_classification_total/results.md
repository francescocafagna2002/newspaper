# EV classification total-sample results

## Protocol

Run `9b03123512952f20` used all 342 rows (335 property groups) from
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
| 1 | E22_lgbm_total | lightgbm | all_netload | 67 | 0.5884 | 0.7601 | 0.7513 | 0.5500 | 0.1699 |
| 2 | E21_hgb_shape | hist_gradient_boosting | load_shape | 67 | 0.5778 | 0.7102 | 0.6947 | 0.4783 | 0.1640 |
| 3 | E20_hgb_compact | hist_gradient_boosting | compact | 67 | 0.5351 | 0.6927 | 0.6765 | 0.5000 | 0.1845 |
| 4 | E10_peak | heuristic | peak | 67 | 0.3273 | 0.5883 | 0.6496 | 0.4348 | 0.3085 |
| 5 | E11_high_fraction | heuristic | high_fraction | 67 | 0.3030 | 0.5910 | 0.6213 | 0.4082 | 0.3075 |
| 6 | E12_composite | heuristic | composite | 67 | 0.3025 | 0.5829 | 0.6233 | 0.4091 | 0.3211 |
| 7 | E02_logistic_compact | logistic | compact | 67 | 0.2951 | 0.6348 | 0.6947 | 0.4783 | 0.2416 |
| 8 | E23_logistic_total | logistic | all_netload | 67 | 0.2495 | 0.5674 | 0.6611 | 0.4500 | 0.2351 |
| 9 | E01_dummy | dummy | none | 67 | 0.2090 | 0.5000 | 0.5000 | 0.3457 | 0.1653 |

The frozen supervised champion is `E22_lgbm_total`, selected by validation PR-AUC.
Its operating threshold `0.391998` was chosen on validation and then
held fixed. The final model was refit on development plus validation before the test
partition was opened once.

## Final held-out test

On 67 current properties (15 EV-labelled), the
frozen pipeline reached PR-AUC **0.3890** (bootstrap 95% interval
0.2137–0.6546), ROC-AUC **0.6679**,
balanced accuracy **0.6410**, F1 **0.4444**,
and Brier score **0.2044**. The confusion matrix is
`[[32, 20], [5, 10]]` in `[[TN, FP], [FN, TP]]` order.

## Interpretation and limitations

The source contains annual aggregates and mean hourly profiles, so the original
week-level event detector cannot be reproduced from this table. The E10–E12 rules are
transparent annual high-load screens, while the supervised runs compare compact load,
hourly shape, and windowed meter-context features. `train_property_sequences.csv`,
which would permit charging-event localization, is described in the data README but is
not present in this checkout. Results are uncertain because the held-out sets are small
and negative labels are contaminated. Do not interpret individual high-load periods as
confirmed EV charging events.

Runtime: 171.3 seconds.
