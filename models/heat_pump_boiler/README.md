# Wärmepumpenboiler detector

This package detects heat-pump water-heater candidates from 15-minute grid-import
data. It uses an interpretable physical rule: a low-power, long-duration plateau
must appear in both a January and July p10 daily profile. The result is a candidate
score with plateau evidence, not a calibrated probability.

The completed real-data run and its limitations are documented in
[`WPB_REPORT.md`](WPB_REPORT.md). The design rationale is in
[`wpb_detector_plan.md`](wpb_detector_plan.md).

## Run

From the `newspaper/` directory:

```bash
python -m models.heat_pump_boiler.wpb profiles --month 2025-07 --sample-every 20
python -m models.heat_pump_boiler.wpb noise --month 2025-07
python -m models.heat_pump_boiler.wpb labelled --summer 2025-07 --winter 2025-01
python -m models.heat_pump_boiler.wpb detect --summer 2025-07 --winter 2025-01
python -m models.heat_pump_boiler.wpb inject --month 2025-07
python -m models.heat_pump_boiler.wpb figures --summer 2025-07 --winter 2025-01
python -m models.heat_pump_boiler.wpb_slides
```

The resumable full-export run is:

```bash
python -m models.heat_pump_boiler.pipeline --workers 4
python -m models.heat_pump_boiler.pipeline --finish-only  # reuse cached profiles
```

## Layout

| File | Role |
|---|---|
| `wpb_core.py` | Pure plateau detection and scoring logic |
| `wpb.py` | Profile extraction, staged commands, injection study, figures |
| `pipeline.py` | Resumable 43-month population sweep and household rollup |
| `build_baserate.py` | GWR hot-water heat-pump comparison by postcode |
| `wpb_slides.py` | Builds `artifacts/wpb_slides.pptx` |
| `test_wpb_core.py`, `test_wpb_pipeline.py` | Unit and isolated end-to-end tests |

Large intermediate profiles and raw-day samples are git-ignored. CSV results,
figures, the report, and slide deck are the reviewable outputs.
