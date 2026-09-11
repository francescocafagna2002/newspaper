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

### LightGBM confusion matrices

Rows are the actual labels. Percentages are calculated within each actual class.

| Cohort | Actual class | Correctly predicted | Incorrectly predicted |
|---|---|---:|---:|
| Validation | Non-EV/unlabelled (53) | **71.7% non-EV** (38) | 28.3% EV (15 false alarms) |
| Validation | EV-labelled (14) | **78.6% EV** (11) | 21.4% non-EV (3 missed EVs) |
| Held-out test | Non-EV/unlabelled (52) | **61.5% non-EV** (32) | 38.5% EV (20 false alarms) |
| Held-out test | EV-labelled (15) | **66.7% EV** (10) | 33.3% non-EV (5 missed EVs) |

On the held-out test, LightGBM detected about two-thirds of the EV-labelled
properties and falsely flagged about two-fifths of the non-EV/unlabelled properties.

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

## Post-model heuristic extension

A research-grounded positive-only override was tested after LightGBM using repeated
3 kW, two-hour plateau events with paired start/end edges. On validation it increased
accuracy from 0.7313 to 0.7463 and balanced accuracy from 0.7513 to 0.7871 by recovering
one false negative. On the already-open test cohort it instead decreased accuracy from
0.6269 to 0.6119 by adding one false positive and no true positives. The base LightGBM
therefore remains the recommended classifier; the heuristic is retained as review
evidence. See `output/heuristic_extension/results.md` for the frozen rule, bounded
search, paired uncertainty, and reused-test caveat.

Combined LightGBM-plus-heuristic confusion matrices (rows are actual labels and
columns are predictions):

| Cohort | Actual class | Correctly predicted | Incorrectly predicted |
|---|---|---:|---:|
| Validation | Non-EV/unlabelled (53) | **71.7% non-EV** (38) | 28.3% EV (15 false alarms) |
| Validation | EV-labelled (14) | **85.7% EV** (12) | 14.3% non-EV (2 missed EVs) |
| Reused test | Non-EV/unlabelled (52) | **59.6% non-EV** (31) | 40.4% EV (21 false alarms) |
| Reused test | EV-labelled (15) | **66.7% EV** (10) | 33.3% non-EV (5 missed EVs) |

Thus, on the reused test data the combined model found about two-thirds of the
EV-labelled properties, but incorrectly flagged about two-fifths of the
non-EV/unlabelled properties.

## Heuristic extension 2: steep slopes

Research supports a sharp start edge followed by a sustained plateau as part of an EV
charging signature. The extension tested positive rescue, negative veto, and two-way
correction after LightGBM. None improved validation balanced accuracy. Positive rescue
changed eight validation predictions, all into false positives, and recovered no EV.
Negative veto increased ordinary accuracy to 76.1% by favoring the majority class, but
EV detection fell from 78.6% to 50.0% and balanced accuracy fell from 75.1% to 66.5%.

The steep-slope extension is therefore rejected. The deployed result remains the base
LightGBM; slope evidence is retained only for review. The detailed direction comparison
and reused-test diagnostic are in `output/heuristic_extension_2/results.md`.
