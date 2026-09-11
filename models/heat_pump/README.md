# Heat-pump classifier (`has_Waermepumpe`)

Predicts, per property, whether a space-heating heat pump is present, from the
GIGI-labelled property-sample table. Distinct from
[`../heat_pump_boiler/`](../heat_pump_boiler/), which detects hot-water
heat-pump *boilers* with a physics-based plateau rule.

- **[`results.md`](results.md)** — protocol, leaderboard, held-out test result
- **[`population_results.md`](population_results.md)** — predicted population
  count vs the BFS GWR register
- `artifacts/heat_pump_slide.pptx` — one slide: top feature, confusion matrix,
  accuracy

Per the reviewing instruction, models are compared and selected on **accuracy
and the confusion matrix only** — no ROC-AUC, PR-AUC, F1 or Brier score.

## Run

```bash
python -m models.heat_pump.build_features        # + WPB score, size-controlled seasonality
python -m models.heat_pump.run_classification    # leaderboard -> frozen champion -> sealed test
python -m models.heat_pump.population_baserate   # population count vs GWR (downloads ag.zip once)
python -m models.heat_pump.slides                # artifacts/heat_pump_slide.pptx
python -m unittest models.heat_pump.test_classification
```

`run_classification.py` refuses to overwrite a completed run; delete
`artifacts/leaderboard.json` and the experiment directories to re-run.

## Layout

| File | Role |
|---|---|
| `config.py` | Paths, seed; reuses the PV pipeline's I/O contract |
| `build_features.py` | Joins the WPB detector score and `winter_share_over_night_share` onto `train_property_samples_auxw_netload.csv` |
| `run_classification.py` | Group-safe split, experiment grid, frozen champion, sealed test |
| `population_baserate.py` | Population-scoreable surrogate + GWR postcode comparison |
| `slides.py` | Builds the single-slide deck |
| `test_classification.py` | Split integrity, leakage, metric contracts |

## Results in one paragraph

The champion (`H21_hgb_compact`, HistGradientBoosting on scalar features)
reaches **76.1 % accuracy** on the sealed test set against a 68.7 %
majority-class baseline, recovering 7 of 21 labelled heat pumps at 2 false
positives. The most important feature by out-of-sample permutation importance
is `evening_share`. On the population, the score ranks postcodes in agreement
with the GWR register (Spearman ρ = 0.39 at matched prevalence) but flags
2.7× too many households at the accuracy-optimal threshold.

## Limitations

- `has_Waermepumpe = 0` is subsidy-register-unlabelled, not verified
  heat-pump-absent (`docs/data_problems.md`), so accuracy is a floor estimate.
- The lift over the majority-class baseline is small and rests on 67 test
  properties. This is consistent with `data_addition/Readme.md`, which found
  the generic feature table near-chance for this label and suspected much
  heat-pump load sits on a separate switched-tariff meter.
- The planned temperature-coupling feature (daily import regressed on
  heating-degree-days, the PV weather-coupling analogue) is **not**
  implemented: it needs per-household daily series, and neither
  `train_property_sequences.csv` nor `load_property_daily.csv` is present in
  this checkout. `weather_daily.csv` alone gives one canton-wide series, which
  cannot discriminate between households.
- The WPB detector score contributes little here — it targets hot-water
  boilers and flags only 2 of 342 households in this sample.
