# SolarPrint — rooftop-PV detection MVP

Predicts, per household (`GP-Nr`), the probability of a rooftop PV system from
consumption-channel smart-meter data.

- **`REPORT.md`** — approach, results, limitations, runtime
- **`MODEL_CARD.md`** — architecture, features + sources, performance
- background: `../../docs/pv_mvp_plan.md`, `../../docs/data_problems.md`

## Run

```bash
python -m newspaper.models.pv.pipeline              # full (~37 min, one 77 GB pass)
python -m newspaper.models.pv.pipeline --skip-scan  # reuse the monthly feature table
```

Any single stage: `python -m newspaper.models.pv.<stage>.<module>` (e.g.
`...pv.modeling.train`, `...pv.features.extract`).

## Layout

```
pv/
├── config.py            paths, the 42-file list, constants, data-quality caveats
├── io.py                reference-table loaders, feature-table (pickle) IO
├── pipeline.py          runs every stage in order
├── prep/
│   ├── labels.py        GIGI → household PV labels + join chain
│   ├── silver_labels.py silver positives + production-meter flags (OBIS 2.29, offline)
│   ├── weather.py       MeteoSwiss daily radiation/temperature + clear-sky flag
│   └── base_rate.py     per-PLZ PV base rate from the BFE ElPA register
├── features/
│   ├── stream.py        positional CSV reader for the 1–3 GB monthly exports
│   ├── extract.py       one streaming pass → features_monthly.pkl + feedin_meter_stats.csv
│   └── aggregate.py     seasonal / weather-coupling / change-point features, meter → GP-Nr
├── modeling/
│   ├── train.py         Positive-Unlabeled model (Elkan–Noto) + base-rate calibration
│   ├── evaluate.py      PU metrics, base-rate check, figures
│   ├── score.py         artifacts/pv_predictions.csv  (the deliverable)
│   └── slides.py        artifacts → ../../docs/pv_model_slides.pptx
└── artifacts/           run outputs (large .pkl/.zip git-ignored, see .gitignore)
```

Environment: Python 3.13 + pandas / numpy / scikit-learn / scipy / matplotlib
/ python-pptx (no pyarrow → pickle).
