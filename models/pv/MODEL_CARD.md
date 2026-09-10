# Model card — SolarPrint

**Model.** SolarPrint — household rooftop-PV classifier (MVP), Energy Data
Hackdays 2026 / AEW "Energy Fingerprints".
**Task.** Binary: does a household have a rooftop PV system?
**Unit.** One prediction per household (`GP-Nr`); meters aggregated.
**Output.** Calibrated probability `pv_probability` ∈ [0,1] + `pv_pred` at 0.5,
with a plain-language `top_evidence` string.
**Population.** ~77k AEW-metered households (canton Aargau).

## Architecture

- **Estimator:** scikit-learn `HistGradientBoostingClassifier`
  (`max_depth=4`, `max_iter=300`, `learning_rate=0.05`, `l2_regularization=1.0`,
  `class_weight="balanced"`, `random_state=20260910`).
- **Learning setup:** Positive–Unlabeled (Elkan–Noto case-control variant).
  There are almost no confirmed PV-negatives, so the model is trained to
  separate known positives `P` from all other households `U`, then scores are
  corrected by the estimated label frequency `c ≈ 0.95`:
  `p = clip(g(x) / c, 0, 1)`.
- **Calibration:** a single logit shift `γ` applied so the mean probability
  over unlabelled households equals the external base rate (0.12); refit at
  scoring time on the scored population.
- **Validation:** 5-fold stratified CV on the P/U indicator; one row per
  household so no household straddles folds.
- **Baseline for context:** a threshold on `daytime_zero_summer` alone reaches
  ROC-AUC ≈ 0.96 (vs ≈ 0.92–0.97 for the GBDT on held-out GIGI positives).

## Why these features (sources)

The feature set operationalises the established **behind-the-meter PV
disaggregation / detection** literature, which all points to the same net-load
signatures:

- **Midday net-load depression / "duck curve".** PV self-consumption cuts
  daytime grid draw; the deeper the midday dip relative to the daily profile,
  the more likely PV. (NREL duck-curve analysis, Denholm et al. 2015; the
  challenge brief itself names *"reduced midday consumption associated with
  solar generation"* as expected evidence.)
  → `midday_clear_summer`, `midday_seasonality`, `daytime_zero_summer`,
  `evening_midday_summer`.
- **Weather–solar coupling.** For a PV household, net load co-varies with solar
  irradiance day-to-day; for a non-PV household it does not. SunDance calls
  this the "Universal Weather–Solar Effect" and disaggregates PV from net
  metering on that basis alone (Chen & Irwin, *SunDance*, ACM e-Energy 2017).
  → `midday_std_summer` (cloud-driven scatter), `load_irr_slope`,
  `load_irr_corr`, plus clear-sky-day conditioning of the midday features.
- **Two-stage detection-then-capacity from net-load curve features + weather
  status.** ML detectors in the literature use exactly this family — seasonal
  load-shape ratios, summer/winter contrast, base-load normalisation
  (Wang et al., *Detection and estimation of behind-the-meter PV generation
  based on smart meter data analytics*, 2022).
  → `summer_winter_kwh_ratio`, `day_kwh_summer/winter`, `night_kwh_mean`
  (size control), `pv_rate` (spatial prior).
- **New-install detection.** A PV install shows as a step change in the midday
  dip partway through a multi-year record (challenge question 3).
  → `midday_max_yoy_drop`, `midday_yoy_slope`.

Approach follows the challenge's recommended **label-first** method: compare
households with and without the asset, learn the recurring distinguishing
features, build a classifier.

## Features (23, consumption channel only — OBIS 1.29)

Derived from a single streaming pass over 42 monthly exports (2023-01 … 2026-07),
aggregated per meter then consumption-weighted to the household.

| Group | Features |
|---|---|
| Midday depression (core PV signal) | `midday_clear_summer`, `midday_clear_winter`, `midday_mean_summer`, `midday_std_summer` (day-to-day cloud scatter), `midday_seasonality` (winter−summer), `daytime_zero_summer`, `daytime_zero_winter` (share of daytime quarter-hours at ~0 net import) |
| Weather–solar coupling (MeteoSwiss) | `load_irr_slope`, `load_irr_corr` — OLS slope / Pearson r of daily summer midday-load-ratio vs daily global-radiation clearness; strongly negative for PV |
| Load level / shape | `day_kwh_mean`, `day_kwh_summer`, `day_kwh_winter`, `summer_winter_kwh_ratio`, `night_kwh_mean` (base-load / size control), `below_night_summer`, `evening_midday_summer` (duck-curve) |
| Change over time | `midday_max_yoy_drop`, `midday_yoy_slope` (year-on-year summer midday-dip — intended to catch mid-record installs; **weak in practice**) |
| Coverage / context | `n_months`, `n_summers`, `rollout_era`, `n_meters`, `pv_rate` (per-PLZ base rate from the BFE ElPA register) |

**Most important (permutation AUC drop):** `daytime_zero_summer` (0.047) ≫
`evening_midday_summer` (0.003) > `midday_clear_summer` (0.002). Effectively a
one-feature model. The MeteoSwiss `load_irr_slope` / `load_irr_corr` separate
the classes strongly on their own (PV slope −0.47 vs +0.02) but are redundant
given `daytime_zero_summer` and do not improve held-out AUC — kept for
interpretability / robustness.

**Not used as features:** the feed-in register (OBIS 2.29) — used only offline
to build silver labels and flag production meters; weather beyond a
canton-level clear-sky-day flag; `PV-Leistung in kWp`.

## Training data

- **Positives (6,468):** GIGI subsidy records with `PV=x` or a battery (288
  with meter data) + **6,188 silver** positives (clear sustained midday summer
  feed-in ⇒ PV). No negative labels.
- **Unlabelled (70,221):** every other household, treated as a contaminated
  negative (contamination ≈ the PV base rate).
- **Labels file:** `../../data_addition/gigi_augmented.csv`.

## Performance (5-fold out-of-fold)

| Metric | Value |
|---|---|
| ROC-AUC — held-out **GIGI** positives vs rest | **0.92** |
| ROC-AUC — all positives vs rest | 0.995 (silver labels partly circular) |
| Recall @ 0.5 — held-out GIGI positives | **0.92** |
| Recall @ 0.5 — silver positives | 0.99 |
| Population flag-rate @ 0.5 | 6.6% (vs 11–12% ElPA base rate) |
| PLZ flag-rate vs ElPA rate (Spearman) | 0.11 |

## Runtime / compute

Single machine, Python 3.13, no GPU.

| Stage | Time |
|---|---|
| Feature build — one 77 GB streaming pass, 42 files (one-time) | **~33 min** |
| Model training — 5-fold CV + final fit + Elkan–Noto + calibration | **~26 s** |
| Inference — score all 76,689 households | **~3 s** (≈45 µs/household) |
| Full pipeline from raw data | ~37 min (feature build dominates) |
| Pipeline with cached features (`--skip-scan`) | ~2 min |

## Known limitations

- No real negatives → ranking is reliable, **absolute probability calibration
  is not** (rests on the ElPA-derived 12%).
- Silver positives are net-exporting systems → weakest on small /
  high-self-consumption PV, which is where it matters most.
- Change-point feature does not track commissioning dates (Spearman 0.05).
- Weather is canton-level, not per-PLZ.
- Only 288 / 724 GIGI PV households link to meter data.

## Figures

`artifacts/figures/` — `fig_architecture.png`, `fig_feature_sep.png` (class separation per feature), `fig_top_feature.png`, `fig_importance.png`, `fig_results.png`. Regenerate: `python -m newspaper.models.pv.modeling.figures`. The slide deck (`../../docs/pv_model_slides.pptx`) is built from these.

## Reproduce

`python -m newspaper.models.pv.pipeline` (full) or `--skip-scan`.
Artifacts: `artifacts/model/{model.pkl, meta.json, oof_predictions.csv}`,
`artifacts/pv_predictions.csv`, `artifacts/metrics.json`. See `REPORT.md`.

## References

- Chen & Irwin, *SunDance: Black-box Behind-the-Meter Solar Disaggregation*,
  ACM e-Energy 2017 — the weather–solar coupling / "Universal Weather–Solar
  Effect". http://www.ecs.umass.edu/irwin/e-energy17.pdf
- *Detection and estimation of behind-the-meter photovoltaic generation based
  on smart meter data analytics*, Electricity Journal 2022 — two-stage ML
  detection from net-load-curve + weather-status features.
  https://www.sciencedirect.com/science/article/abs/pii/S1040619022000586
- Denholm et al., *Overgeneration from Solar Energy in California: A Field
  Guide to the Duck Chart*, NREL/TP-6A20-65023, 2015 — midday net-load
  depression.
- Energy Data Hackdays 2026 challenge brief (`../../CLAUDE.md`) — names
  "reduced midday consumption associated with solar generation" as expected
  evidence and recommends the label-first approach.
- MeteoSwiss SwissMetNet open data (`ch.meteoschweiz.ogd-smn`) — daily global
  radiation `gre000d0`, air temperature `tre200d0`.
- BFE ElPA electricity-production-plant register
  (`ch.bfe.elektrizitaetsproduktionsanlagen`) — per-PLZ PV base rate.
