# models/ — rooftop-PV detection MVP

Predicts, per household (`GP-Nr`), the probability of a rooftop PV system from
consumption-channel smart-meter data. See `REPORT.md` for results and
`../docs/pv_mvp_plan.md` / `../docs/data_problems.md` for background.

## Run

```bash
python -m newspaper.models.pipeline              # full (~40 min, one 77 GB pass)
python -m newspaper.models.pipeline --skip-scan  # reuse the monthly feature table
```

Individual steps: `python -m newspaper.models.<step>` where `<step>` is
`config, build_labels, build_weather, build_baserate, build_features,
extract_feedin_labels, build_features_agg, train, evaluate, score`.

## Layout

| module | role |
|---|---|
| `config.py` | paths, the 42-file list, constants, data-quality caveats |
| `stream.py` | positional CSV reader for the 1–3 GB monthly exports |
| `io_utils.py` | reference-table loaders, feature-table (pickle) IO |
| `build_labels.py` | GIGI → household PV labels + join chain |
| `extract_feedin_labels.py` | silver positives + production-meter flags (OBIS 2.29, offline) |
| `build_weather.py` | MeteoSwiss daily radiation/temperature + clear-sky flag |
| `build_baserate.py` | per-PLZ PV base rate from the BFE ElPA register |
| `build_features.py` | single streaming pass → `features_monthly.pkl` + `feedin_meter_stats.csv` |
| `build_features_agg.py` | seasonal / change-point features, meter → `GP-Nr` |
| `train.py` | Positive-Unlabeled model (Elkan-Noto) + base-rate calibration |
| `evaluate.py` | PU metrics, base-rate check, figures |
| `score.py` | `artifacts/pv_predictions.csv` (the deliverable) |

Artifacts (git-ignored) land in `artifacts/`. Environment: Python 3.13 +
pandas / numpy / scikit-learn / scipy / matplotlib (no pyarrow → pickle).
