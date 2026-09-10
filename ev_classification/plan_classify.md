# EV presence classification notebook plan

## Objective

Create `ev_classification/classify_ev.ipynb`, a reproducible notebook that reads
the customer-week table produced by `ev_classification/build_customer_week_table.py`
and answers two related questions:

1. Can transparent, literature-informed rules identify whether an EV charger is
   present from a household's 15-minute net-load profile?
2. Can a supervised tabular model, initially LightGBM, improve on those rules
   while retaining interpretable evidence?

The notebook must separate exploratory/tuning work from final testing, respect
the uncertain installation-date interval, split by customer rather than by
week, and never use rows marked as ineligible for supervised training. It should
produce both customer-week probabilities and customer-level EV-presence scores;
the weekly evidence must remain inspectable.

This is a plan for a notebook, not a request to rebuild the source table or to
claim that an individual high-load interval is definitively an EV charge.

## Required input and contract

Read the partitioned Parquet dataset:

```text
ev_classification/output/customer_week_table.parquet/
```

This is the direct output of the code described in `ev_classification/plan.md`.
Optionally read `run_metadata.json` and `run_summary.csv` from the same output
directory for provenance and display-only checks. Do not go back to the raw
monthly CSV exports in this notebook.

Require and validate:

- one unique row per `(gp_nr, week_start)`;
- exactly 672 ordered `net_load_*_kwh` columns;
- no null load values in retained weeks;
- `week_start`, `ev_label`, `eligible_for_supervised_training`,
  `training_exclusion_reason`, and the four project-date columns;
- the load unit is kWh per 15-minute interval and the source register convention
  is import minus export;
- no customer appears with contradictory canonical EV labels.

Convert interval energy to average power only where physics-based thresholds
need it:

```text
net_load_kw = 4 * net_load_kwh
```

Keep the original 672 kWh columns unchanged. Never apply a kW threshold directly
to the kWh values.

## Reproducibility and leakage controls

At the start of the notebook define a small configuration object containing the
input/output paths, random seed, split fractions, date-boundary convention,
heuristic search ranges, LightGBM search space, customer aggregation rule, and
decision metric. Print it and save it with the results.

Before inspecting class-specific profiles or choosing thresholds, assign each
`gp_nr` once to a permanent, seeded customer split:

- 60% development/train;
- 20% validation;
- 20% final test.

Use a deterministic stratified group allocation based on the canonical
customer-level `ev_label`, balancing positive and negative customers with usable
labelled weeks as closely as possible. Customers without usable temporal targets
still receive a split for later scoring but do not count toward class balance.
All weeks of a customer must remain in the same split. Assert that the three
customer sets are disjoint and persist the split manifest. If the positive-customer count is too small for a stable
60/20/20 split, use grouped outer folds and reserve one entire fold as the final
test; record the deviation. A row-wise random split is prohibited because it
would put nearly identical weeks from one household on both sides.

Use the splits as follows:

- development: exploratory plots, candidate feature design, fitting, and search;
- validation: select heuristic thresholds, model hyperparameters, probability
  calibration, and operating threshold;
- final test: evaluate each frozen pipeline once after all choices are fixed.

For the final model, refit the selected feature pipeline and LightGBM parameters
on development plus validation, but do not retune anything after viewing test
results. Preserve a pre-refit validation result so the selection process is
auditable.

## Temporal weak-label policy

The four available dates describe a project span, not a confirmed
EV-commissioning date. Nevertheless, following the requested conservative
working assumption, derive for each customer:

```text
earliest_known_date = minimum non-null value among
  datum_unterschrift_earliest,
  geplanter_baustart_earliest,
  uebergabe_latest,
  inbetrieb_datum_latest

latest_known_date = maximum non-null value among the same four columns
```

Use half-open week bounds `[week_start, week_start + 7 days)` and assign
`temporal_target` only as follows:

| Customer metadata | Week relation | Target | Use |
|---|---|---:|---|
| `ev_label == 1`, both bounds known | week end `<= earliest_known_date` | 0 | train/validate/test |
| `ev_label == 1`, both bounds known | week start `>= latest_known_date` | 1 | train/validate/test |
| `ev_label == 1` | between the bounds or straddles either bound | unknown | score/display only |
| `ev_label == 1`, either bound unavailable | cannot prove temporal state | unknown | score/display only |
| `ev_label == 0` | any eligible week | 0 | train/validate/test |

This yields conservative week-level labels: certainly pre-project, certainly
post-project, or unknown. Never turn an unknown week into a negative. Report
counts by target, split, customer, date-source completeness, and calendar year.
The notebook must visibly state that these are weak labels, because the source
dates have not been confirmed as EV-specific.

Apply the source training gate before forming any supervised or threshold-tuning
dataset:

```python
supervised = weeks.loc[
    weeks["eligible_for_supervised_training"].eq(1)
    & weeks["temporal_target"].notna()
].copy()
```

Rows marked `eligible_for_supervised_training == 0` (the user's "no train"
rows) must never influence heuristic tuning, feature selection, model fitting,
calibration, threshold selection, or reported supervised metrics. They may only
be counted in an exclusions table and optionally receive clearly marked
exploratory predictions after the pipelines are frozen.

Also run two sensitivity analyses without changing the primary result:

1. compare the primary definition with boundaries made only from early fields
   (`Datum Unterschrift`, `geplanter Baustart`) and late fields (`Übergabe`,
   `InBetrieb-Datum`);
2. evaluate post-project positives only after a 7-day and a 28-day washout.

Large changes must be highlighted as date-label uncertainty, not presented as
model instability alone.

## Literature basis for candidate heuristics

The notebook should summarize, in its own words, the following evidence and link
the sources in a Markdown cell:

- Hoffmann et al., *Automated Detection of Electric Vehicles in Hourly Smart
  Meter Data* (CIRED 2019), observes that charging may appear as several hours
  of kW-level demand and also documents substantial false-positive risk from
  ordinary household loads. This motivates duration, amplitude, repetition,
  and explicit confounder checks rather than a peak-only rule:
  https://sintef.brage.unit.no/sintef-xmlui/bitstream/handle/11250/2618624/Hoffmann%2BAutomated%2Bdetection%2Bof%2Belectric%2Bvehicles%2Bin.pdf?sequence=2
- Li et al., *Inferring electric vehicle charging patterns from smart meter
  data for impact studies* (Electric Power Systems Research 235, 2024), detects
  events from 15-minute AMI data using contextual constraints on start time,
  duration, and power level. This supports constructing event-shaped evidence
  rather than using weekly totals alone:
  https://doi.org/10.1016/j.epsr.2024.110789
- Hangawatta et al., *A novel method for electrical vehicle charging load
  extraction from low sampling rate data* (Sustainable Energy, Grids and
  Networks 43, 2025), combines amplitude filtering, a minimum duration of about
  1.5 hours, high-pass/wavelet information, and clustering/template matching for
  30-minute data. These are candidate starting points, not fixed thresholds for
  the AEW population:
  https://doi.org/10.1016/j.segan.2025.101903
- Figueiredo et al., *Non-intrusive load disaggregation solutions for very
  low-rate smart meter data* (Applied Energy 268, 2020), emphasizes the limits
  of NILM at 15--60-minute resolution and the need to validate on genuine
  aggregate household traces:
  https://doi.org/10.1016/j.apenergy.2020.114949
- The 15-minute residential study *Behind-the-Meter Disaggregation of
  Residential Electric Vehicle Charging Load* jointly treats EV presence,
  charging-rate estimation, and charging-period detection. Use it to motivate
  consistency across those three kinds of evidence:
  https://doi.org/10.1109/SmartGridComm52983.2022.9961024

Do not copy a numerical threshold from another country or sampling interval and
call it validated. Use literature values only to define plausible search ranges;
tune all final cutoffs on the development/validation customers and report the
chosen values.

## Part A: interpretable heuristic pipeline

### Baseline and candidate event extraction

Work day by day while retaining events that cross midnight. Estimate a robust,
household-local baseline so PV export, heating, and general household scale do
not turn one global absolute threshold into the result. Candidate baselines to
compare on development data are a rolling low quantile, a time-of-week median
from other weeks of the same customer, and a robust daily low quantile. Fit a
baseline using only past data when evaluating a chronological variant.

From `residual_kw = net_load_kw - baseline_kw`, form candidate charging runs and
merge gaps of at most one 15-minute interval. For each run calculate at least:

- median and upper-quantile residual power;
- duration, energy above baseline, and number of intervals;
- within-run median absolute deviation, slope, and plateau fraction;
- size of the start rise and end fall, and their symmetry;
- start/end hour and whether the event crosses midnight;
- overlap with negative net load or strong midday PV export;
- nearby competing spikes and interrupted/stepped charging indicators.

Candidate rule families should include:

1. **sustained block:** residual exceeds a tunable kW threshold for a tunable
   minimum duration;
2. **plateau with paired edges:** sustained low-variation block with a positive
   start edge and similarly sized negative end edge;
3. **repeated charging level:** event powers cluster around one or a small number
   of household-specific levels on multiple days/weeks;
4. **template similarity:** robust distance or correlation to a rectangular or
   gently tapering charging template;
5. **optional high-pass/wavelet candidate:** use only if it gives a measurable
   validation improvement over the simpler event detector.

Time of day may be evidence but must not be mandatory: workplace schedules,
tariffs, solar-aware charging, and weekends can all produce daytime events.
Likewise, do not require a perfectly flat trace because aggregate household
loads sit on top of charging.

### Weekly and customer heuristic scores

Convert detected events into transparent weekly features: event count, total
candidate energy, number of distinct event days, longest event, median event
power, repeated-level concentration, paired-edge fraction, plateau quality, and
night/day shares. Define a monotonically interpretable heuristic score and
retain the individual component values.

Aggregate weekly evidence to a customer without averaging away rare charging.
Compare frozen candidate rules such as maximum weekly score, top-`k` mean, and
"at least `n` qualifying events on `d` distinct days." Select the aggregation
and threshold on validation customers. Require a minimum number of observable
weeks and emit `insufficient_history` rather than a confident negative when the
requirement is not met.

For the primary one-prediction-per-customer evaluation, aggregate only weeks
that describe the evaluation state: post-`latest_known_date` weeks for
EV-labelled customers and eligible weeks for non-EV customers. Cap both groups
to the same configured number of most recent labelled weeks so history length
alone cannot reveal the class. Never mix a positive customer’s pre-project
negative weeks into its post-project presence score; use those weeks only for
week-level training and the separate within-customer transition analysis.

Tune only a small predeclared grid of amplitude, duration, plateau tolerance,
edge symmetry, event repetition, and customer-decision thresholds. Select by
validation customer-level macro F1 or balanced accuracy (choose one in config),
with recall, precision, false-positive rate, and PR-AUC reported alongside it.
Freeze the complete rule before final-test evaluation.

Include plots for representative true positives, true negatives, false
positives, and false negatives. Show the original net load, estimated baseline,
candidate residual, detected event bounds, and which rule components fired.

## Part B: supervised LightGBM pipeline

### Features

Build features inside sklearn-compatible transformers or pure functions whose
parameters are fitted on development data only. Begin with interpretable
weekly features rather than treating customer metadata or dates as predictors:

- distribution and load-shape statistics (quantiles, peak-to-baseline range,
  load factor, ramp quantiles, autocorrelation);
- duration-above-threshold features across a small threshold grid in kW;
- counts and summaries from the frozen candidate-event extractor;
- day/night, weekday/weekend, and time-of-day block summaries;
- repetition/regularity of high-load plateaus across days;
- optional 672-position normalized profile or compact Fourier/wavelet features,
  admitted only through validation comparison;
- data-quality controls such as imputed count, used for robustness analysis and
  included as model features only if explicitly justified.

Do not use `gp_nr`, `ev_label`, `temporal_target`, split assignment,
`eligible_for_supervised_training`, exclusion reasons, any of the four source
dates, `label_date_ambiguous`, or post-outcome GIGI fields as predictors. Do not
use other asset labels (`pv_label`, heat-pump labels, battery label) as primary
features because they would not be available when predicting unlabeled
customers. They may be used only for stratified error analysis. `PLZ`, `Ort`,
and `Kanton` should also be excluded from the primary model to avoid geographic
memorization; report any later geographic experiment separately.

Fit normalization or household baselines without using validation/test values.
For household-relative features that require history, define a transform that
uses only the permitted earlier weeks; do not compute a full-customer statistic
and then split it across time.

### Training and selection

Start with two baselines:

- a prevalence-only or dummy classifier;
- logistic regression on the compact engineered features.

Train `lightgbm.LGBMClassifier(objective="binary")` on labelled development
weeks. Address imbalance with sample weights calculated from development
customers, but also cap each customer's total weight so customers with many
weeks do not dominate. Do not simultaneously use incompatible oversampling and
class-weight strategies.

Search a deliberately small validation-driven space over number of leaves,
learning rate, number of trees with early stopping, minimum child samples,
feature fraction, bagging fraction, and L1/L2 regularization. The validation
objective is the predeclared customer-level metric; weekly metrics are secondary.
Where hyperparameter tooling expects cross-validation, use grouped folds by
`gp_nr`, never ordinary K-fold CV.

Calibrate probabilities on held-out validation customers (compare uncalibrated,
sigmoid, and isotonic only when validation size supports isotonic). Select the
classification operating threshold from validation data according to the
declared metric or an explicit operational precision/recall constraint. Do not
assume 0.5 is optimal.

Aggregate weekly model probabilities with the same candidate family used for
the heuristic (maximum, top-`k` mean, or at-least-`n` evidence), chosen on
validation. Return both weekly probabilities and the final customer score,
number of weeks, contributing top weeks, and the aggregation rule.

### Interpretation

Report LightGBM gain/split importance and permutation importance on validation
or test customers. Add SHAP only if available and computationally practical.
For each customer prediction, show the weeks and feature values that most
strongly contributed. Interpretation must not use the final test set to redesign
features.

## Evaluation

Primary evaluation unit: one current or post-project state per customer, using
the state-specific, history-capped aggregation above. Secondary unit: labelled
customer-week. Report uncertainty with a customer-level bootstrap so all weeks
of a sampled customer move together. A positive customer without a post-project
week is not evaluable for the primary metric and must not become a negative.

For heuristic, dummy/logistic, and LightGBM results report:

- customer counts and week counts by class and split;
- confusion matrix, precision, recall, specificity, balanced accuracy, F1,
  ROC-AUC, and especially PR-AUC for imbalance;
- Brier score and reliability plot for probabilistic models;
- 95% customer-bootstrap confidence intervals for primary metrics;
- performance versus number of available weeks (1, 2--4, 5--12, 13+);
- error slices by season, year, PV label, heat-pump label, battery label,
  imputation, and pre/post temporal state, with small-cell suppression for
  privacy;
- paired heuristic-versus-model comparison on exactly the same final-test
  customers.

Test the intended change signal separately for EV-labelled customers that have
both pre- and post-project weeks: compare score distributions within customer
and report how often post scores exceed pre scores. This is strong supporting
evidence but not proof of commissioning timing.

Guard against optimistic metrics caused by many negative weeks. Give every
customer equal influence in customer metrics and weighting; report both
micro/week-level and macro/customer-level results with clear labels.

## Notebook structure

Implement the notebook in the following executable order:

1. Title, question, assumptions, and literature references.
2. Imports, versions, seed, and configuration.
3. Load source Parquet and metadata; schema/unit assertions.
4. Build temporal weak labels and exclusion audit.
5. Create and save the customer-level 60/20/20 split manifest.
6. Blind development-only exploratory analysis.
7. Implement event extraction and visualize development examples.
8. Tune heuristics on development/validation and freeze them.
9. Build leakage-safe weekly features.
10. Fit dummy, logistic, and LightGBM models; tune and calibrate on validation.
11. Freeze weekly-to-customer aggregation and operating thresholds.
12. Run final test once and compare pipelines.
13. Run robustness, temporal-transition, and error-slice analyses.
14. Score unknown and no-train rows separately, clearly marked as exploratory.
15. Save artifacts, concise conclusions, limitations, and next steps.

Keep expensive searches behind configuration flags and cache only artifacts
whose cache keys include the source metadata hash and full configuration. The
default notebook should run top to bottom without manual cell reordering.

## Outputs

Write notebook artifacts below `ev_classification/classification_output/`:

- `split_manifest.parquet`: one row per customer with permanent split;
- `temporal_label_audit.parquet`: week bounds, derived date bounds, temporal
  target, eligibility gate, and exclusion reason;
- `weekly_features.parquet`: identifiers, split, target, and model features;
- `heuristic_events.parquet`: candidate events and interpretable components;
- `weekly_predictions.parquet`: heuristic/model scores and labelled/unknown
  status;
- `customer_predictions.parquet`: aggregate scores, decision, evidence weeks,
  and history sufficiency;
- `metrics.json` and `metrics_by_slice.csv`;
- `heuristic_config.json`, `feature_manifest.json`, `model.txt`, calibration
  artifact, and complete run configuration;
- diagnostic figures with no raw customer identifiers in titles or filenames.

Store sensitive identifiers only in Parquet artifacts inside the existing
ignored output area. Hash or replace `gp_nr` in displayed examples, suppress
small groups in tables, and do not export raw load profiles outside the project.

## Acceptance checks

Before handoff, the notebook must demonstrate that:

1. the source schema and 672-column ordering pass;
2. no customer occurs in more than one split;
3. all supervised rows have `eligible_for_supervised_training == 1` and a
   non-null conservative temporal target;
4. no ambiguous-date or boundary-straddling week entered fitting or metric
   computation;
5. no prohibited label, date, identifier, geography, or asset metadata column
   entered the primary feature matrix;
6. every preprocessing fit used development data only, and any grouped CV used
   `gp_nr` as its group;
7. heuristic and model hyperparameters and thresholds were frozen before the
   final test was read;
8. reported customer metrics give each customer one prediction and final-test
   comparisons use identical customers;
9. kWh-to-kW conversion is covered by a small assertion or unit test;
10. saved predictions reconcile to input, exclusion, labelled, and unknown row
    counts;
11. rerunning with the same source/configuration yields the same split and
    materially identical metrics;
12. conclusions explicitly distinguish charger presence, observed charging
    evidence, and uncertain project-date timing.

## Known limitations to state

- A charger can be present without charging in an observed week, so absence of
  an event is not proof of absence.
- The four project dates are not confirmed charger commissioning dates; the
  temporal targets are conservative working weak labels and need owner review.
- Aggregate load can confuse EV charging with heat pumps, electric heating,
  boilers, cooking, batteries, or coincident appliances.
- PV and batteries can partially mask charging in net load.
- Fifteen-minute sampling blurs edges and short or interrupted sessions.
- Labels describe a charger, not necessarily an EV or its usage, and customer
  equipment can change outside the recorded project span.
- Results on labelled AEW customers may not transfer unchanged to the full
  customer population; predicted prevalence must not be presented as audited
  ownership without further validation.
