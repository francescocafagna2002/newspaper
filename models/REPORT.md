# MVP: predicting rooftop-PV ownership from smart-meter data

Energy Data Hackdays 2026 · AEW "Energy Fingerprints" · 2026-09-10

## What it does

Given a household's four years of 15-minute **grid-import** readings (OBIS
1.29 only), the pipeline outputs a calibrated probability that the household
has a rooftop PV system, for every household in the exports (~77k). Run it
with `python -m newspaper.models.pipeline`.

## Method (one line each)

| Step | Module | Result |
|---|---|---|
| Labels | `build_labels` | GIGI is subsidy applications; positive = any row `PV=x` **or** a battery. 878 households, **288** positive with meter data. No usable negatives. |
| Silver labels | `extract_feedin_labels` | Feed-in register (2.29, offline only) → **6,188** extra "clear summer feed-in ⇒ PV" households. Recovers 258/294 GIGI positives (88%). No silver negatives. |
| Production-meter filter | `extract_feedin_labels` | 683 export-dominant meters dropped from consumption features. |
| Weather | `build_weather` | MeteoSwiss 5 AG stations → daily radiation + a clear-sky-day flag (25% of days). |
| Base rate | `build_baserate` | BFE ElPA PV register ÷ metered households → **canton-AG rate ≈ 11%** (national ref 12–14%). |
| Features | `build_features` (1 pass, 42 files) | per `(meter, month)` consumption features; midday-dip on clear-sky days, daytime near-zero share, seasonality, day-to-day scatter, change-point. |
| Aggregate | `build_features_agg` | meter → household (consumption-weighted). 80k households. |
| Model | `train` | **Positive-Unlabeled** (Elkan-Noto) on `HistGradientBoostingClassifier`, 5-fold. `c = 0.95`. Score shifted so population mean = 12%. |
| Score | `score` | `artifacts/pv_predictions.csv`: `gp_nr, pv_probability, pv_pred, top_evidence`. |

## Results (5-fold out-of-fold)

| Metric | Value | Note |
|---|---|---|
| ROC-AUC, GIGI positives vs rest | **0.92** | independent labels; the honest headline |
| ROC-AUC, all positives vs rest | 0.995 | inflated — silver labels ~ circular with the signal |
| Recall @ p≥0.5 on held-out **GIGI** positives | **0.92** | |
| Recall @ p≥0.5 on silver positives | 0.99 | |
| Population flag rate @ p≥0.5 | 7% | vs 11–12% ElPA base rate → model is conservative |
| One-feature rule baseline AUC (`daytime_zero_summer`) | 0.96 | the task has a very strong single signal |
| PLZ flag-rate vs ElPA rate (Spearman) | 0.22 | weak geographic agreement |
| Change-point feature vs commissioning date (Spearman) | 0.05 | **not working** — see limitations |

`avg_precision` against GIGI positives is low (0.04) only because GIGI
positives are 0.37% of the population and sit *below* the 6k silver positives;
it is not a meaningful population metric here (no representative labelled
sample, unknown true prevalence). Figures: `artifacts/figures/`.

## Top features (permutation AUC drop)

1. **`daytime_zero_summer`** (0.058) — share of summer daytime quarter-hours at
   ~zero net import. PV self-consumption zeroes the meter. PV mean 0.73 vs 0.11.
2. `midday_clear_summer` (0.006) — midday/daily load ratio on sunny days. 0.24 vs 1.02.
3. `evening_midday_summer` (0.003) — evening-peak / midday ratio. 4.4 vs 1.9.

Essentially a one-feature model; the GBDT adds ~3 AUC points over the rule.

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
  export partial. See `../docs/data_problems.md`.

## Files

- `../data_addition/gigi_augmented.csv` — labels (GIGI + battery + silver)
- `../data_addition/weather_daily.csv`, `pv_baserate_plz.csv`
- `artifacts/features_monthly.pkl` → `features_gp.pkl`
- `artifacts/model/` — `model.pkl`, `meta.json`, `oof_predictions.csv`
- `artifacts/pv_predictions.csv` — **the deliverable**
- `artifacts/metrics.json`, `artifacts/figures/`
