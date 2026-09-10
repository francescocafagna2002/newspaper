# Adaptive agent plan for EV classification experiments

## Mission

Build and run an adaptive sequence of EV-presence classification experiments on
the Parquet data created by `ev_classification/build_customer_week_table.py`.
Try several transparent heuristics and supervised models, score every valid run,
use validation results and error analysis to decide what to try next, and finish
with one frozen final-test evaluation.

Maintain `ev_classification/results.md` throughout the work. It must give a
brief, reproducible description of every experiment, its results, what was
learned, and why the next experiment was chosen. Record failed and discarded
experiments as well as successful ones.

The execution agent must first read:

- `CLAUDE.md` and invoke the repository-local `$grill-me` skill, as required
  before executing a plan;
- `ev_classification/plan.md` for the source-table contract;
- `ev_classification/plan_classify.md` for labels, leakage controls, literature
  basis, evaluation units, privacy, outputs, and acceptance checks;
- `ev_classification/README.md` and the table-building code for actual paths and
  column names.

This document governs experiment order and adaptive decisions. If it conflicts
with a data-integrity or leakage rule in `plan_classify.md`, the stricter rule
wins.

## Deliverables

The agent must create or update:

- `ev_classification/classify_ev.ipynb`: readable analysis and final comparison;
- reusable Python helpers under `ev_classification/` when notebook-only code
  would be hard to test or rerun;
- tests for temporal labels, split isolation, feature exclusion, kWh-to-kW
  conversion, aggregation, and scoring;
- `ev_classification/results.md`: experiment journal, validation leaderboard,
  final test result, conclusion, and recommended pipeline;
- run artifacts under `ev_classification/classification_output/`, with one
  subdirectory per experiment and a stable `best/` export for the selected
  pipeline.

Do not edit or regenerate the input Parquet data as part of classification.

## Input discovery and immutable run identity

Default input:

```text
ev_classification/output/customer_week_table.parquet/
```

Read the companion `run_metadata.json` and `run_summary.csv` if present. Fail
with a clear message if the Parquet dataset is absent; do not silently substitute
raw CSV files or another dataset.

Before experimentation, compute a run identity from:

- sorted Parquet filenames, sizes, and modification times, plus a content hash
  of metadata files;
- code commit or working-tree identifier;
- random seed and complete experiment configuration;
- Python and package versions.

Persist this in `classification_output/run_manifest.json` and include it near
the top of `results.md`. Never compare leaderboard rows produced from different
source identities without marking them as non-comparable.

## Non-negotiable evaluation protocol

### Labels and exclusions

Implement the conservative temporal weak-label policy from
`plan_classify.md` exactly:

- an EV-labelled customer's week wholly before the earliest known date is 0;
- a week wholly after the latest known date is 1;
- intervening, boundary-straddling, or undatable EV-labelled weeks are unknown;
- eligible weeks of customers with `ev_label == 0` are 0.

Rows with `eligible_for_supervised_training != 1`, including every "no train"
row, must not affect fitting, heuristic tuning, calibration, threshold choice,
model selection, or supervised metrics. They may be scored only after the final
pipeline is frozen and must be labelled exploratory.

### Permanent customer split

Create one seeded 60/20/20 development/validation/test split at customer level,
stratified as closely as possible among customers with usable temporal labels.
All weeks and states of a `gp_nr` stay together. Save the manifest before the
first class-specific analysis, test for zero overlap, and reuse it unchanged in
every experiment.

The experiment loop may inspect development and validation outcomes. It must
not calculate, display, or write final-test predictions or metrics until the
pipeline, features, hyperparameters, calibration method, aggregation rule, and
decision threshold are frozen. If the test is accidentally opened early, record
the breach in `results.md` and stop claiming that split as an unbiased final
test. Continue only if a genuinely untouched customer pool can be reserved
without using the observed test outcomes to redesign the pipeline; otherwise
report validation results and the absence of an unbiased final estimate.

### Comparable scoring set

Build a fixed validation comparison cohort once and save its customer IDs. For
the primary customer score:

- use post-latest-date weeks for positive customers;
- use eligible negative weeks for non-EV customers;
- take the same configured maximum number of most recent labelled weeks per
  customer;
- exclude positive customers without a post-date week from the primary metric
  and report their count;
- require the configured minimum history or return `insufficient_history`.

Every leaderboard experiment must use exactly this cohort and the same history
cap. Pre-project weeks from positive customers remain useful as negative
week-level training examples and for the separate change analysis, but must not
enter that customer's post-project presence score.

### Metrics

Use customer-level average precision/PR-AUC as the primary model-selection
metric. Use balanced accuracy and F1 at a validation-selected threshold as
secondary metrics, followed by recall, precision, specificity, ROC-AUC, and
Brier score. Also report week-level metrics, but never rank pipelines by a
week-level metric alone.

Estimate 95% intervals by bootstrapping customers, not rows. Use identical
bootstrap draws for paired comparisons. Treat changes smaller than bootstrap
uncertainty as ties; prefer the simpler and more interpretable tied pipeline.

## Experiment infrastructure before modelling

Implement one experiment runner so all approaches receive the same split,
labels, aggregation, and metrics. Each run gets an immutable ID such as
`E01_dummy`, `E10_sustained_block`, or `E21_lgbm_heuristic`. Its directory must
contain:

- `config.json` with every effective parameter;
- `metrics_validation.json`;
- customer and weekly validation predictions in Parquet;
- runtime and peak-memory summary where practical;
- selected diagnostic figures;
- serialized model or heuristic configuration;
- a feature manifest and explicit prohibited-column check for trained models.

Add a smoke mode operating on a deterministic customer subset, but never place
smoke-run numbers on the main leaderboard. A full validation run is required
before an experiment can influence the next decision.

Initialize `results.md` from the template below before E00. After each full run,
append its experiment entry immediately and update the validation leaderboard
and decision log. Do not rely on memory or postpone documentation until the end.

## Sequential experiment programme

### Stage 0 — audit and baselines

Run these first:

#### E00: data and label audit

Validate shape, units, load-column order, duplicates, missingness, exclusion
counts, date-bound completeness, temporal-target counts, class counts, split
balance, histories per customer, and primary comparison cohort. This is not a
classifier, but its findings go into `results.md`.

Stop and repair only classification code if an assertion fails. If the source
Parquet violates `plan.md`, report the blocker and do not mutate the source.

#### E01: dummy classifier

Predict development prevalence for all validation examples and use the fixed
customer aggregation. This establishes prevalence, PR-AUC floor, Brier score,
and thresholded-metric floors.

#### E02: compact logistic regression

Use a small predeclared feature set: weekly quantiles, peak-to-baseline range,
ramp quantiles, load factor, day/night energy, weekday/weekend energy, and
duration above several physically interpretable kW levels. Fit preprocessing on
development only and use customer-balanced sample weights.

Decision after Stage 0:

- If E02 cannot beat E01 beyond bootstrap uncertainty, inspect label counts,
  date sensitivity, units, and feature distributions before adding complexity.
- If there are too few positive validation customers for meaningful comparison,
  switch to grouped outer cross-validation while preserving one untouched final
  fold; document the protocol change and restart the leaderboard.
- Otherwise continue to the heuristic stage.

### Stage 1 — heuristic families

Use literature to set broad starting ranges, but tune thresholds only with
development and validation customers. Convert kWh/interval to kW correctly.
Run each family as its own scored experiment so its contribution is visible.

#### E10: sustained high-residual block

Estimate a robust local baseline, detect runs above an amplitude threshold,
merge at most one missing/low interval, and require a minimum duration. Search a
small grid over baseline method, residual power, duration, and minimum event
count. Save detected event bounds and component measurements.

#### E11: plateau plus paired edges

Extend E10 with within-event variation, start-rise, end-fall, and edge-symmetry
conditions. Compare it directly with E10 on the identical cohort.

#### E12: repeated charging level

Extend the stronger of E10/E11 with evidence that candidate events recur near
one or a few customer-specific power levels across distinct days or weeks.
Tune level tolerance and repetition count.

#### E13: template similarity

Score event candidates against rectangular and gently tapering charging
templates using robust correlation or normalized distance. Combine with the
best simple family only if the template score has incremental validation value.

#### E14: optional high-pass or wavelet detector

Run only when E10--E13 show that ramps are obscured by drifting household load,
or error plots reveal missed long blocks. Compare against the best simpler
detector. Skip it, with a recorded reason, if added complexity is unlikely to
address observed errors or the needed dependency is unavailable.

For every heuristic, compare `max weekly score`, `top-k mean`, and a transparent
`at least n events on d days` customer aggregation. Tune the final cutoff only
on validation. Record representative anonymized true positives, false
positives, and false negatives.

Decision after each heuristic:

- Keep it as the current heuristic champion only if primary PR-AUC improves
  beyond uncertainty, or if tied while materially improving interpretability,
  recall at the required precision, runtime, or robustness.
- If false positives concentrate in heat-pump/PV/battery slices, add signal-based
  confounder features or event rules; do not add those ground-truth asset labels
  as predictors.
- If false negatives are mostly low-power charging, widen amplitude ranges and
  strengthen duration/repetition evidence rather than simply lowering every
  threshold.
- If errors are driven by sparse history, tune abstention/minimum-history rules;
  do not label abstentions as negatives.
- Stop adding heuristic complexity after two consecutive full experiments fail
  to improve the champion meaningfully.

### Stage 2 — supervised training families

All models use grouped customer handling, customer-balanced weights, and the
same frozen validation cohort. Keep dates, identifiers, split fields, labels,
other asset labels, and geography out of the primary feature matrix.

#### E20: LightGBM on compact statistical features

Train LightGBM on the E02 features. Search a bounded set of leaves, learning
rate, tree count with early stopping, minimum child samples, row/feature
subsampling, and L1/L2 regularization. Record search size and winning settings.

#### E21: LightGBM plus heuristic evidence

Add event counts, duration, energy above baseline, plateau quality, paired-edge
fraction, repetition concentration, template similarity, and day/night event
shares from the frozen heuristic extractor. This tests whether the heuristic
features provide incremental value beyond generic load statistics.

#### E22: LightGBM with temporal-shape features

Add compact time-of-week summaries, Fourier/wavelet coefficients, or a carefully
normalized 672-position profile. Run ablations so any improvement can be
attributed to a feature family. Reject the raw-profile variant if it appears to
memorize household scale or calendar effects, is unstable across seasons, or
only improves week-level metrics.

#### E23: optional alternative tree model

Run HistGradientBoosting, XGBoost, or CatBoost only if LightGBM is unavailable,
clearly unstable, or a concrete error pattern suggests the alternative. This is
not a mandatory model zoo. Use the same features and scoring contract.

#### E24: probability calibration and aggregation

For the best one or two model variants, compare uncalibrated, sigmoid, and—only
with enough calibration customers—isotonic probabilities. Compare customer
aggregation rules and select the decision threshold on validation. Generate
grouped out-of-fold predictions when fitting a calibrator for the eventual
development-plus-validation refit.

Decision after each training run:

- Promote a new champion using customer PR-AUC first, then balanced accuracy,
  calibration, stability, simplicity, and runtime as tie-breakers.
- If E21 does not beat E20, retain heuristic evidence for explanations but omit
  it from the trained feature matrix.
- If E22 gains only marginally within uncertainty, prefer E20/E21.
- If training PR-AUC is high but validation PR-AUC drops substantially, reduce
  tree capacity, strengthen regularization, remove unstable feature families,
  and run at most two targeted overfitting corrections.
- If both heuristic and model fail on the same customers, inspect label timing,
  PV masking, heating-like plateaus, and history length before trying another
  algorithm.

### Stage 3 — targeted error-driven experiments

Run no more than three experiments in this stage. Each must begin with a written
hypothesis based on the current champion's validation errors and change exactly
one feature family or modelling choice. Valid examples include:

- a PV-masking-aware residual baseline when errors align with midday export;
- seasonal normalization when heat-pump winter loads dominate false positives;
- charger-level clustering when multiple stable power modes are visible;
- separate recent-history and long-history aggregations when evidence dilution
  is measurable;
- a 7-day or 28-day post-date washout when near-boundary label noise dominates.

Do not run an experiment merely because a library or model is available. Reject
post-hoc customer-specific rules, direct use of metadata labels, or repeated
searches over many random seeds.

### Stage 4 — freeze and final test

End iterative development when any of these occurs:

- two consecutive targeted experiments do not meaningfully improve the
  validation champion;
- the champion meets the predeclared operational requirement and remaining
  errors are irreducible from available data;
- the major heuristic and LightGBM families plus relevant error-driven branches
  have been evaluated;
- the limit of 12 full modelling experiments, excluding E00, is reached.

Before opening test data, write a `FINAL FREEZE` entry in `results.md` containing
the exact experiment ID, feature manifest hash, hyperparameters, calibration,
aggregation, minimum history, and threshold. Refit the frozen pipeline on
development plus validation. Then evaluate exactly once on the final test
customers and save `metrics_test.json` and predictions.

Do not modify the selected pipeline after seeing final-test results. Any later
idea belongs to a new study with a new held-out test set. Report bootstrap
confidence intervals and compare the selected model with E01, E02, and the best
heuristic on identical test customers.

After the final test, score unknown and no-train rows only as explicitly marked
exploratory outputs. These predictions must not be folded into performance
metrics.

## `results.md` contract

Use this structure:

```markdown
# EV classification experiment results

## Run identity and evaluation protocol
Source identity, date, code state, seed, label policy, split counts, primary
cohort, history cap, primary metric, and test-blinding status.

## Data audit
Brief counts and important data limitations from E00.

## Validation leaderboard
| Rank | ID | Family | Key change | Customers | PR-AUC | Balanced accuracy | F1 | Brier | Runtime | Status |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|

## Experiment log

### E10 — Sustained high-residual block
- Hypothesis:
- Inputs/split:
- Method and parameters:
- Result: primary and important secondary metrics with intervals.
- Error slices/observations:
- Decision: promoted/rejected/failed, with reason.
- Next experiment and why:
- Artifacts:

## Decision log
Chronological one-line decisions linking each completed run to the next.

## Final freeze
Exact immutable selected configuration, written before test evaluation.

## Final test
One-time test metrics with intervals and paired baseline comparisons.

## Recommendation and limitations
What should be used, expected operating behavior, evidence shown to users, and
remaining uncertainty.
```

Keep descriptions brief but concrete. Never write only "better" or "worse";
include the metric delta, uncertainty, and relevant trade-off. Mark results as
`validation`, `final test`, `smoke`, `failed`, or `exploratory`. Link artifact
paths relative to `results.md`. Do not expose raw `gp_nr` values in Markdown or
figure names.

## Agent autonomy and decision rules

The agent is authorized to implement, run, diagnose, and revise classification
experiments within this plan. It may choose exact search ranges and the next
targeted experiment from observed development/validation evidence. It must:

1. state a hypothesis before each non-mandatory run;
2. change the smallest set of factors needed to test it;
3. score on the fixed validation cohort;
4. append the result and decision to `results.md` before proceeding;
5. preserve all prior results and artifacts;
6. prefer a simpler tied solution;
7. stop when the stopping criteria are satisfied;
8. never use final-test feedback for another iteration.

If a dependency installation, large compute expansion, external data source, or
change to the source-table builder is required, pause and request user approval.
Ordinary fixes to experiment code, bounded model searches, notebook execution,
and creation of the listed local artifacts are in scope.

## Completion criteria

The task is complete only when:

- source and label audits pass or a genuine source blocker is documented;
- E01, E02, at least three distinct heuristic families, E20, and E21 have full
  comparable validation results, unless a recorded data/dependency reason makes
  one impossible;
- adaptive follow-up experiments have explicit hypotheses and decisions;
- every full run appears in `results.md` and has reproducible artifacts;
- one champion is frozen before the final test;
- the final test is evaluated once with customer-level uncertainty;
- the selected approach is compared with the dummy, logistic, and heuristic
  baselines on identical customers;
- unknown/no-train predictions are separated from supervised evaluation;
- the notebook runs top to bottom using the selected cached experiment outputs;
- `results.md` ends with a concise recommendation and honest limitations.

## Post-test extension: explicit charging-behavior features

After the original frozen evaluation, execute
`ev_classification/plan_classify_features.md` as a separate study. It adds
late-afternoon/evening/night ramp, persistence, paired-edge, time-of-use power
distribution, repeated-level, multiscale-change, and customer-history features
supported by the literature cited there.

The original final test is permanently closed for model development. Tune all
new feature parameters and model hyperparameters with customer-level folds
inside the original development split. Compare locked candidates with
`E02_logistic` only on the unchanged validation cohort, label the result as
validation reuse, and never recalculate original final-test predictions or
metrics. Store the new study under `feature_study_output/` and journal it in
`results_feature_study.md`; preserve every original classification artifact.

Run F40 timed-ramp logistic, F41 power-distribution/multiscale logistic, F42
combined regularized logistic, and F43 combined low-capacity LightGBM in that
order, subject to the extension plan's stopping rule. Promotion still requires
a paired customer-bootstrap AP improvement beyond uncertainty, with the simpler
model preferred in a tie.
