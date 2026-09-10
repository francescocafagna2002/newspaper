# SolarPrint — rooftop-PV detection from smart-meter data

Energy Data Hackdays 2026 · AEW "Energy Fingerprints" · 2026-09-10

SolarPrint predicts, per household, the probability of a behind-the-meter PV
system from the grid-import load shape alone.

## What it does

Given a household's four years of 15-minute **grid-import** readings (OBIS
1.29 only), the pipeline outputs a calibrated probability that the household
has a rooftop PV system, for every household in the exports (~77k). Run it
with `python -m newspaper.models.pv.pipeline`.

## Method (one line each)

| Step | Module | Result |
|---|---|---|
| Labels | `prep.labels` | GIGI is subsidy applications; positive = any row `PV=x` **or** a battery. 878 households, **288** positive with meter data. No usable negatives. |
| Silver labels | `prep.silver_labels` | Feed-in register (2.29, offline only) → **6,188** extra "clear summer feed-in ⇒ PV" households. Recovers 258/294 GIGI positives (88%). No silver negatives. |
| Production-meter filter | `prep.silver_labels` | 683 export-dominant meters dropped from consumption features. |
| Weather | `prep.weather` | MeteoSwiss 5 AG stations → daily radiation + a clear-sky-day flag (25% of days). |
| Base rate | `prep.base_rate` | BFE ElPA PV register ÷ metered households → **canton-AG rate ≈ 11%** (national ref 12–14%). |
| Features | `features.extract` (1 pass, 42 files) | per `(meter, month)` consumption features; midday-dip on clear-sky days, daytime near-zero share, seasonality, day-to-day scatter, change-point. |
| Aggregate | `features.aggregate` | meter → household (consumption-weighted). 80k households. |
| Model | `train` | **Positive-Unlabeled** (Elkan-Noto) on `HistGradientBoostingClassifier`, 5-fold. `c = 0.95`. Score shifted so population mean = 12%. |
| Score | `score` | `artifacts/pv_predictions.csv`: `gp_nr, pv_probability, pv_pred, top_evidence`. |

## Results (5-fold out-of-fold)

| Metric | Value | Note |
|---|---|---|
| ROC-AUC, GIGI positives vs rest | **0.92** | independent labels; the honest headline |
| ROC-AUC, all positives vs rest | 0.995 | inflated — silver labels ~ circular with the signal |
| Recall @ p≥0.5 on held-out **GIGI** positives | **0.92** | |
| Recall @ p≥0.5 on silver positives | 0.99 | |
| Population flag rate @ p≥0.5 | 6.6% | vs 11–12% ElPA base rate → model is conservative |
| One-feature rule baseline AUC (`daytime_zero_summer`) | 0.96 | the task has a very strong single signal |
| PLZ flag-rate vs ElPA rate (Spearman) | 0.11 | weak / noisy geographic agreement |
| Change-point feature vs commissioning date (Spearman) | 0.05 | **not working** — see limitations |

`avg_precision` against GIGI positives is low (0.04) only because GIGI
positives are 0.37% of the population and sit *below* the 6k silver positives;
it is not a meaningful population metric here (no representative labelled
sample, unknown true prevalence). Figures: `artifacts/figures/`.

## Features

Feature choice follows the behind-the-meter PV-detection literature (duck-curve
midday depression; SunDance weather–solar coupling; two-stage net-load +
weather-status ML detectors) and the challenge's label-first recommendation —
see `MODEL_CARD.md` for the mapping from each source to each feature.
MeteoSwiss global radiation enters as `load_irr_slope` / `load_irr_corr` (daily
summer midday-load vs irradiance) plus clear-sky-day conditioning.

## Top features (permutation AUC drop)

1. **`daytime_zero_summer`** (0.047) — share of summer daytime quarter-hours at
   ~zero net import. PV self-consumption zeroes the meter. PV mean 0.73 vs 0.11.
2. `evening_midday_summer` (0.003) — evening-peak / midday ratio. 4.4 vs 1.9.
3. `midday_clear_summer` (0.002) — midday/daily load ratio on sunny days. 0.24 vs 1.02.

Essentially a one-feature model; the GBDT adds ~3 AUC points over the rule.

**MeteoSwiss irradiance features** (`load_irr_slope`, `load_irr_corr` — daily
summer midday-load vs global-radiation clearness) behave exactly as the
SunDance "weather–solar effect" predicts — PV slope **−0.47** vs **+0.02** for
non-PV — but are **redundant** given `daytime_zero_summer` and do not move the
held-out AUC (0.924 → 0.920). Kept for interpretability and robustness (they
carry independent physical signal if the dominant feature degrades), not for a
score gain.

### MeteoSwiss data — what is used, what could be

| Field | Status |
|---|---|
| **Global radiation** `gre000d0` | **used** — (1) clear-sky-day flag → the `midday_ratio_clear` features (midday dip measured only on sunny days); (2) `load_irr_slope` / `load_irr_corr` (SunDance load–irradiance coupling). |
| **Air temperature** `tre200d0/dn/dx` | downloaded & cached, **not yet a feature**. Intended use: a winter *load-vs-temperature* slope as a **heat-pump control** (heat pumps raise winter import as it gets colder, which can mask a PV home's low summer-relative import) and heating-degree-day normalisation of winter consumption. |
| **Per-PLZ interpolation** | canton-level series only for the MVP; nearest-3-station inverse-distance is a stretch. |

Given `daytime_zero_summer` alone already scores ROC-AUC 0.90, extra
weather-derived features have low marginal value *for this binary classifier*.
They matter more for the subtle self-consumption cases, for separating PV from
heat pumps in a multi-asset model, and for kWp capacity estimation.

## Ablation — single-feature model

Trained the identical PU pipeline (Elkan–Noto + GBDT + base-rate calibration,
5-fold) on **`daytime_zero_summer` alone** — the share of summer daytime
quarter-hours at ~zero net import:

| Model | ROC-AUC (GIGI vs rest) | Recall@0.5 GIGI | Recall@0.5 silver | Pop. flag-rate | PU ROC-AUC (GIGI) |
|---|---:|---:|---:|---:|---:|
| Full (23 features) | **0.92** | **0.92** | 0.99 | 0.066 | 0.96 |
| `daytime_zero_summer` only | 0.90 | 0.87 | 0.89 | 0.084 | 0.95 |

The one feature carries ~90 % of the model. The other 22 add ~2–3 ROC points
and ~5 pp recall, and make the scores sharper (Elkan–Noto `c` 0.95 vs 0.85 —
i.e. with the full set almost every known positive scores high, with one
feature many self-consuming PV homes sit mid-range). Reproduce:
`python -m newspaper.models.pv.modeling.train --only daytime_zero_summer`
(writes `artifacts/model_daytime_zero_summer/`).

## Runtime / compute

Single machine, Python 3.13, no GPU. `HistGradientBoostingClassifier` uses a
few cores during training; everything else is single-core.

| Stage | Time | Notes |
|---|---|---|
| **Feature build** (`features.extract`, one 77 GB streaming pass, 42 files) | **~33 min** | one-time; ~47 s/file, peak RSS ~1.6 GB → `features_monthly.pkl` (2.3 M rows) |
| `prep.labels` / `prep.weather` / `prep.base_rate` | ~2 s / ~1 s (15 s first download) / ~40 s | ElPA download + one export scan |
| `prep.silver_labels` + `features.aggregate` | ~20 s + ~38 s | silver labels, meter→household rollup |
| **Model training** (`train`: 5-fold CV + final fit + Elkan–Noto + calibration) | **~26 s** | 76.7 k households × 23 features |
| Evaluation (`evaluate`) | ~2 s | |
| **Inference** (`score`, all 76,689 households) | **~3 s** (~45 µs/household) | `predict_proba` is milliseconds; rest is I/O + evidence strings |
| Full pipeline from raw data | **~37 min** | dominated entirely by the one-time feature build |
| Pipeline with cached features (`--skip-scan`) | **~2 min** | retrain + rescore |

## Limitations

- **No real negatives.** PU learning + base-rate calibration; the absolute
  probability scale rests on the ElPA-derived 12% and is uncertain. Ranking is
  reliable, calibration is not.
- **Silver-positive bias.** Silver labels are net-exporting systems, so the
  model is sharpest on those and weakest on high-self-consumption / small PV —
  exactly the households the challenge cares most about. GIGI-only recall
  (0.92) partly controls for this.
- **Change-point feature is ineffective** (Spearman 0.05 vs commissioning
  date). Year-over-year midday-dip differences are too noisy at this
  aggregation; needs a proper per-meter changepoint test on the daily series.
- **Weather is canton-level**, not per-PLZ.
- **Coverage**: only 288 of 724 GIGI PV households link to meter data; a
  per-`GP-Nr` pooled-meter dataset (pending) would enlarge and de-bias this.
- `2023-03` export missing; `2023-04` header malformed (handled); `2024-10`
  export partial. See `../../docs/data_problems.md`.

## Files

- `../../data_addition/gigi_augmented.csv` — labels (GIGI + battery + silver)
- `../../data_addition/weather_daily.csv`, `pv_baserate_plz.csv`
- `artifacts/features_monthly.pkl` → `features_gp.pkl`
- `artifacts/model/` — `model.pkl`, `meta.json`, `oof_predictions.csv`
- `artifacts/pv_predictions.csv` — **the deliverable**
- `artifacts/metrics.json`, `artifacts/figures/`
