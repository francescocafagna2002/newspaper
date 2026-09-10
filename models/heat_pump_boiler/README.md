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

---

## Wärmepumpenboiler (heat-pump water heater) detector

A separate, rule-based detector — **not** part of the PV pipeline. Only 4
labelled WPB households have meter data and there are no confirmed negatives,
so this is a physical detector validated without labels. Plan and limitations:
[`../docs/wpb_detector_plan.md`](../docs/wpb_detector_plan.md).

```bash
python -m newspaper.models.wpb noise    --month 2025-07          # go/no-go first
python -m newspaper.models.wpb profiles --month 2025-07 --month 2025-01
python -m newspaper.models.wpb labelled --summer 2025-07 --winter 2025-01
python -m newspaper.models.wpb detect   --summer 2025-07 --winter 2025-01
python -m newspaper.models.wpb inject   --month 2025-07
python -m newspaper.models.wpb figures  --summer 2025-07 --winter 2025-01
python -m newspaper.models.wpb_slides
```

`--sample-every N` on `profiles` keeps 1 meter in N (hash-stable across months,
so the summer/winter pairing survives) if a full pass is too slow.

| module | role |
| --- | --- |
| `wpb_core.py` | detector logic, no I/O — unit-tested in `../tests/` |
| `wpb.py` | streaming, staging, figures |
| `wpb_slides.py` | builds `artifacts/wpb_slides.pptx` |

Run `noise` before anything else: a 0.5 kW WPB adds 0.125 kWh per 15-minute
slot, and if the night-load noise floor is above that, the rest cannot work.
