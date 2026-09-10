# EV classification feature study plan

## Objective

Test whether explicit charging-behavior features improve EV-presence
classification over the frozen `E02_logistic` champion. The main hypothesis is
that a home charger creates a steep positive load transition in the late
afternoon, evening, or night, followed by a sustained, comparatively stable
load and often a paired negative transition. Repetition at similar power levels
on several days or weeks strengthens that evidence.

This is a post-final-test study. The previous test split has already been
evaluated and must not be opened, rescored, or used for decisions. Tune every
new feature parameter and model hyperparameter using only original development
customers. Compare each locked candidate once with the saved champion on the
unchanged validation cohort. Report these comparisons as validation reuse, not
as a new unbiased final estimate.

## Literature basis

The feature families are grounded in primary literature:

- Vavouris et al. report that residential EV demand is prominent in
  late-afternoon and early-evening hours, and show that longer windows help
  reject high-power appliances that operate for shorter periods. Their
  post-processing uses expected charging duration and time information, and
  their 15-minute experiments use short correction gaps to join interrupted
  detections. This motivates time-windowed ramps, persistence, duration, and
  small-gap tolerance:
  https://doi.org/10.3390/en15062200
- Li et al. infer events using charging start time, duration, and power level.
  This motivates retaining all three as separate, inspectable features rather
  than collapsing them into weekly energy:
  https://doi.org/10.1016/j.epsr.2024.110789
- Neubert et al. find that power-balance distribution above roughly 5 kW is
  influential for EV detection and that longer observation spans help when a
  week contains no charging event. This motivates high-load histograms,
  load-duration features, and recurrence across weeks:
  https://doi.org/10.3390/en15134922
- Hangawatta et al. combine amplitude and duration filters with discrete
  wavelet/high-pass information and clustering at low sampling rates. This
  motivates multiscale difference energy and repeated-level concentration,
  while leaving thresholds to AEW development data:
  https://doi.org/10.1016/j.segan.2025.101903
- Hoffmann et al. use clustered charging signatures and matched filtering in
  hourly data and observe expected night/evening charging patterns, but also
  substantial baseline-consumption confounding. This motivates shape
  consistency and household-relative features instead of peak-only rules:
  https://hdl.handle.net/11250/2618624
- Vavouris et al. also demonstrate performance loss when transferring among
  houses, regions, charger powers, and 15-minute data. Model capacity must
  therefore remain low and customer-level validation uncertainty must govern
  promotion.

Literature values define broad search ranges only. No published threshold is
treated as valid for AEW customers without development-only tuning.

## Data and evaluation contract

Reuse the immutable source identity, temporal weak labels, permanent customer
split, 12-week history cap, two-week minimum history, and comparison cohorts
from `classification_output/`. Assert exact source identity and exact customer
order against the saved `E02_logistic/customer_validation.parquet` artifact.

Use only signal-derived features. Prohibit identifiers, dates, split fields,
geography, EV labels, all other asset labels, eligibility fields, and project
metadata from the model matrix. Convert kWh per 15 minutes to average kW before
using physical thresholds.

For positive customers, customer features use only post-latest-date weeks. For
non-EV customers, they use eligible negative weeks. Use the same most-recent
12-week cap and minimum history as the current comparison cohort. Pre-project
negative weeks may be used by a separate week-level study later, but do not mix
them into a positive customer's presence vector.

Tune with deterministic five-fold stratified cross-validation across
development customers. Each customer contributes one row, so no customer can
cross folds. Rank tuning candidates by mean fold customer average precision;
break ties with lower standard deviation, then fewer features and stronger
regularization. Select the operating threshold from pooled development
out-of-fold scores by balanced accuracy, then F1.

After tuning an experiment, fit it on all development customers and evaluate it
once on the fixed validation customers. Use paired customer bootstraps against
`E02_logistic`. Treat differences inside the paired 95% interval as ties and
prefer the simpler model. Do not read any file below
`classification_output/final_test/` or use `best/metrics_test.json`.

## Feature families

### Timed transitions and persistence

For each day, measure positive 15-minute ramps in candidate windows spanning
late afternoon through night. For each candidate ramp, measure:

- rise magnitude and its rank relative to that household-week's other ramps;
- persistence for 1, 1.5, or 2 hours above the pre-event local baseline;
- median and variation of the post-rise plateau;
- presence, size, and symmetry of a later negative edge;
- interruption count, permitting at most one low interval;
- event start hour and whether it crosses midnight;
- distinct active days and number/fraction of weeks with qualifying evidence.

Tune the start/end window, minimum rise, and persistence duration on development
customers. Include an unrestricted-time count so the model can learn whether
time localization adds information.

### Time-of-use high-load distribution

Measure hours and energy above interpretable power levels in evening/night and
all-day windows; ratios of evening/night to daytime high-load energy; daily peak
hour concentration; positive-power histogram occupancy; load-duration-curve
tail area; and concentration near recurring power levels. Tune the time window,
power floor, and histogram tolerance using development folds.

### Multiscale changes and shape

Measure absolute and signed changes over 15, 30, 60, and 120 minutes; high-pass
energy at those scales; positive/negative edge balance; autocorrelation at
daily and weekly lags; and consistency of the high-load time-of-day profile
across days. These features are descriptive signal summaries and do not use a
wavelet dependency or external data.

### Customer-history aggregation

Aggregate each weekly feature using mean, maximum, and top-three mean. Add
fraction of active weeks, total distinct active days, between-week coefficient
of variation for event power, circular concentration of event start time, and
recurrence at a similar power level. Keep individual weekly features and event
evidence in Parquet for inspection.

## Experiments

### F40 — timed ramp logistic regression

Tune candidate time windows, minimum positive ramp (1, 2, or 3 kW per
15-minute transition), persistence (1, 1.5, or 2 hours), and logistic `C` on
development folds. Use only timed-transition, persistence, paired-edge, and
history-recurrence features.

### F41 — power distribution and multiscale logistic regression

Tune time window, power floor (1.5, 3, or 5 kW), power-level tolerance (0.5 or
1 kW), and logistic `C`. Use time-of-use high-load distribution, load-duration,
repeated-level, and multiscale-change features.

### F42 — combined regularized logistic regression

Combine the locked F40 and F41 feature configurations with the existing compact
weekly statistics. Tune only logistic regularization. This measures the
incremental value of the new families in a low-capacity model.

### F43 — combined low-capacity LightGBM

Use the same frozen combined feature matrix. Search a small development-only
grid with 3--7 leaves, at least 10 customer rows per leaf, feature subsampling,
and L1/L2 regularization. Run only after F42 so nonlinear value is directly
attributable. Prefer F42 if the paired validation difference is uncertain.

## Artifacts and journal

Write all outputs under `feature_study_output/`, never under the frozen original
experiment directories. Persist:

- `run_manifest.json` with source, split, plan, and code hashes;
- development and validation customer feature matrices;
- the fixed inner-fold manifest;
- one immutable directory per experiment containing effective configuration,
  CV search results, feature manifest, fitted model, validation predictions,
  metrics, paired bootstrap comparison, and feature coefficients/importances;
- `leaderboard.json` and `results_feature_study.md`;
- weekly feature and anonymized event-evidence tables for the selected new
  candidate.

Do not place raw `gp_nr` values in Markdown or figure names. Parquet prediction
artifacts may retain `gp_nr` for reconciliation.

## Promotion and stopping

Call a new classifier the validation champion only when its paired AP interval
against `E02_logistic` is wholly above zero. If tied, prefer the smaller
regularized logistic model. Report useful secondary trade-offs, but do not
promote on balanced accuracy alone. Stop after F43 or after two consecutive new
families fail to improve AP beyond paired uncertainty.

Completion requires all four experiments unless the stopping rule is reached,
all tests passing, identical validation customers across comparisons, no access
to final-test artifacts, and a concise recommendation that clearly labels the
study as validation reuse.
