# Feature Evaluation Plan for Household Asset Detection

## Goal

For a target technology (initially PV, then EV, heat pump, and battery), train a model on labelled households and return, for a fixed, previously unseen meter-point ID (`MP ID`):

1. the probability that the technology is present;
2. a calibrated present/absent decision at a documented operating threshold; and
3. the feature evidence behind the decision.

The initial candidate model is LightGBM because it handles nonlinear, correlated, missing, tabular features and yields useful feature-attribution outputs. It is a baseline, not an assumption: logistic regression and random forest should be retained as comparators.

## Data and label contract

* Construct the label table through `MP ID -> mpid_zähler_mapping -> Zähler-GP -> HackDays2026 - GIGI` as documented in `DATASETS.md`.
* Treat the household/meter point as the observational and split unit. All 15-minute readings belonging to one `MP ID` must stay in the same fold.
* Build one row per `(MP ID, analysis window)`. Start with seasonal windows (winter, spring, summer, autumn) and a full-year aggregate, so the model sees both stable presence signals and their seasonal expression.
* Define labels at the window end date. If GIGI provides an installation/project date, readings before the installation must be excluded from positive windows (or labelled absent); otherwise this is label leakage. Record unknown installation dates as a limitation.
* Audit each device label before modelling: positive count, missing/ambiguous values, co-occurrence with other devices, labels per postcode, and label prevalence by year. Never infer absence merely because a field is blank until the source convention is verified.
* Stream one monthly CSV at a time and persist only household-window feature tables. The raw exports are too large to load together.

## Feature catalogue

Implement every feature as a deterministic function with its name, units, time window, missing-data rule and source in `main.md`. Produce features at multiple granularities: raw 15-minute summaries, day profiles, month/season summaries, and year summaries.

### PV candidates

* Export frequency, export energy and lower-tail net-load statistics (`share(P < 0)`, `P_min`, 1st/5th percentiles).
* Midday (11:00–15:00) versus night (01:00–05:00) load ratios, midday skewness, and monthly solar-shaped profiles.
* Day-to-day/24-hour autocorrelation, sunny-day ramp rates, and the stability of these effects across summers.
* If weather is added, irradiance-to-net-load correlation during daylight, evaluated only with weather available at prediction time.

### EV candidates

* Counts, amplitudes, durations and energy of sustained upward plateaus/steps above candidate thresholds (for example 3.0, 3.7 and 7.4 kW); tune thresholds inside training folds only.
* Night/off-peak (19:00–06:00) high quantiles, volatility, session frequency, weekday/weekend contrast and seasonal persistence.
* Daily min-to-median/peak spreads and high-power duration distributions.

### Heat-pump candidates

* Winter-to-summer energy ratio, seasonal profile shifts, daily quantiles and baseload ratios.
* With temperature data: load-versus-temperature slope below 15 °C, heating-degree-day correlation and operation by temperature band.
* Temperature-free seasonal proxies are retained as a separate feature block, because they may be less portable across locations and years.

### Battery candidates

* Daylight near-zero net-load duration, intraday variance/peak suppression, charge/discharge-like ramps and abrupt transition counts.
* Evaluate these both alone and conditional on PV. Battery signatures are likely confounded by PV, so report their incremental value beyond PV features rather than interpreting a raw importance score as causal evidence.

### Shared controls and safeguards

* Include data-quality controls: observed interval share, longest missing run, duplicate/zero-heavy interval rates and meter scale checks. Do not let missingness masquerade as an asset signature.
* Use postcode only as a controlled context variable and report performance with and without it. It can encode local adoption patterns and artificially inflate random-split scores.
* Never use household ID, customer/installation IDs, label fields, installation dates, or post-window metadata as predictors.

## Model design: separate binary models first

Use one binary classifier per device: `PV present`, `EV present`, `heat pump present`, and `battery present`. Each model has its own label, feature subset, class weighting, calibration and decision threshold.

This is the preferred initial design because device fingerprints, prevalence, label quality and error costs differ substantially. It also supports a clean question—“which features improve PV detection?”—without a feature helping a common label while hiding harm to a rare one. The outputs are independent probabilities, allowing valid co-occurrence (for example PV + battery or EV + heat pump).

Potential disadvantages are ignored label correlation and reduced effective sample size for rare devices. Therefore evaluate a secondary shared multi-label baseline (one shared feature representation with one output per device, or a classifier-chain/stacked model). It is adopted only if it improves out-of-household calibrated performance for a device without degrading interpretability or leakage controls. Do not use mutually exclusive multiclass classification: devices can coexist.

For each device compare:

| Candidate | Purpose |
| --- | --- |
| Regularized logistic regression on scaled features | Transparent linear sanity baseline |
| LightGBM | Main nonlinear tabular model |
| Random forest | Robust nonlinear comparison |
| Shared multi-label baseline | Tests whether cross-device information helps |

## Evaluation protocol

1. Freeze an initial feature catalogue and preprocessing code before inspecting test results.
2. Make an untouched final test set by grouping on `MP ID`; stratify approximately by target label, year coverage and postcode where feasible. Do not use it for feature selection or threshold choice.
3. On the remaining households, use repeated grouped, stratified cross-validation (`GroupKFold`/`StratifiedGroupKFold`, group = `MP ID`). All fitting steps—imputation, scaling, threshold tuning, feature selection and calibration—occur within the training side of each fold.
4. Add a temporal robustness test: train on earlier calendar periods and score later periods for households not seen in training. This tests feature stability rather than merely household memorisation.
5. For location robustness, hold out postcode groups (or larger geographic groups if positive counts are small) as an additional diagnostic, not necessarily the sole model-selection score.
6. Choose the probability threshold from cross-validated training predictions based on the agreed operating objective (for example high recall for outreach, high precision for asset verification). Keep it fixed for the final test.
7. Calibrate final probabilities with a fold-safe calibration method (isotonic if enough positives; sigmoid otherwise) and assess calibration separately from ranking.

Report ROC-AUC and PR-AUC, but prioritize precision, recall, F1/F-beta, specificity, confusion matrix, Brier score and calibration curve at the selected threshold. Give bootstrap confidence intervals by resampling households, not rows/windows.

## How a feature earns its place

Evaluate feature groups before individual features; this reduces false conclusions from correlated variables such as several versions of export depth.

1. **Baseline:** quality controls plus basic aggregate load statistics.
2. **Forward block ablation:** add each domain block (PV export/shape/weather; EV sessions/night profile; HP thermal/seasonal; battery daylight/ramps) and measure the grouped-CV change in PR-AUC, recall/precision at the operating threshold, Brier score and geographic/temporal robustness.
3. **Drop-column ablation:** from the best full model, remove one block and then one feature at a time. A useful feature should show a repeatable loss across folds, not a single-fold gain.
4. **Permutation importance on validation folds:** permute a feature within fold, preferably within season/year strata, and quantify score degradation. Use this as corroboration rather than the only criterion because features are correlated.
5. **Stability and redundancy:** record the fraction of folds/bootstrap samples in which the feature is selected or ranks in the top set. Cluster highly correlated features and retain the simplest, most stable representative.
6. **Explanation review:** inspect SHAP values and partial-dependence/ICE plots for the final candidate features. Confirm the direction and shape make physical sense (for example, more repeated daylight export should not lower PV probability without a documented interaction).
7. **Error slices:** compare false positives and false negatives by season, postcode, PV co-occurrence, data completeness and household load scale. Reject features whose apparent benefit is concentrated in a leakage-prone or poorly represented slice.

Accept a feature/block only when it improves the primary cross-validated metric with a practically meaningful, confidence-supported change; is stable across folds; does not materially worsen calibration or robustness; and has an intelligible physical interpretation. Keep a feature decision log with experiment ID, data cut, split, metric deltas, uncertainty and decision.

## Fixed-household inference and evidence

After selecting and refitting a device model on all approved labelled training households, compute the same feature set for the requested fixed `MP ID` and analysis window. Return the calibrated probability, thresholded result, data-coverage warning, top positive and negative SHAP contributions, and the raw supporting measurements (for example export share and midday/night ratio for PV). If the household is outside the training feature range or has insufficient coverage, mark the result as low confidence rather than forcing a conclusion.

## Deliverables and stopping criteria

* Versioned label audit, feature table schema and feature dictionary.
* Reproducible split assignments keyed by `MP ID` and a run configuration with dates, thresholds and random seeds.
* Device-specific scorecard containing baseline/full/ablation results, calibration, uncertainty and error slices.
* Feature decision log and a compact per-household evidence report.

The first milestone is a PV-only, feature-block experiment with LightGBM and logistic-regression baselines. Expand to the other separate device models only after the PV pipeline passes leakage checks and produces stable grouped-CV results; then run the shared multi-label comparison.
