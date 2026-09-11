# Heat-pump (`has_Waermepumpe`) classification results

## Protocol

342 rows (335 property groups) from
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
| 1 | H21_hgb_compact | hist_gradient_boosting | compact | 67 | 0.7164 | [[43, 3], [16, 5]] |
| 2 | H12_winter_summer_ratio | heuristic | winter_summer_ratio | 67 | 0.7015 | [[46, 0], [20, 1]] |
| 3 | H20_logistic_compact | logistic | compact | 67 | 0.7015 | [[43, 3], [17, 4]] |
| 4 | E00_dummy_majority | dummy | none | 67 | 0.6866 | [[46, 0], [21, 0]] |
| 5 | H10_wpb_score | heuristic | wpb_score | 67 | 0.6866 | [[46, 0], [21, 0]] |
| 6 | H22_hgb_shape | hist_gradient_boosting | load_shape | 67 | 0.6866 | [[44, 2], [19, 2]] |
| 7 | H23_lgbm_shape | lightgbm | load_shape | 67 | 0.6866 | [[46, 0], [21, 0]] |
| 8 | H11_dedicated_meter | heuristic | dedicated_hp_meter | 67 | 0.6716 | [[42, 4], [18, 3]] |

The frozen supervised champion is `H21_hgb_compact`, selected by validation
accuracy. Its operating threshold `0.717173` was chosen on
validation and held fixed. The final model was refit on development plus
validation before the test partition was opened once. Its most important
feature is `evening_share`.

## Final held-out test

On 67 current properties (21
HP-labelled), the frozen pipeline reached accuracy
**0.7612** with confusion matrix
`[[44, 2], [14, 7]]` (`[[TN, FP], [FN, TP]]`).

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

Runtime: 17.4 seconds.
