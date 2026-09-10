# EV classification experiment results

## Run identity and evaluation protocol

Run `58bd932a3b44b9669cfd358fa2160175774da912711031976d3f792238b783c6`; source `69a33762f866aeca210389e6866fedd3aa01c65af74cfe72b4225c92f2658fbc`. Created 2026-09-10T22:19:16.191652+00:00; commit `47139d50b5afe140c71d660fb4b7dcbaefe13f2b`; code hash `62bbf4d9cdc3ad3d75d5ac9ad077de151dab083bf6db613a884111efa78d0525`. User waived unavailable grill-me prerequisite. Source data is immutable. Seed 20260910; customer split 60/20/20; minimum 2 weeks, latest 12 state-specific weeks. Primary metric: customer average precision (PR-AUC); operating threshold maximizes balanced accuracy, then F1. All intervals are 95% customer bootstrap (1,000 paired draws). Minimum publishable slice size is 5. Project-span labels are weak labels, not verified EV commissioning dates. Test predictions remain sealed until FINAL FREEZE.

## Data audit

## Validation leaderboard

<!-- LEADERBOARD -->
| Rank | ID | Family | Key change | Customers | PR-AUC | Balanced accuracy | F1 | Brier | Runtime | Status |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | E12_repeated_level | repeat | Repeated power levels across days reject isolated appliance loads | 61 | 0.6485 | 0.8098 | 0.5161 | 0.6070 | 7.5s | validation |
| 2 | E10_sustained_block | block | Repeated sustained residual loads distinguish chargers | 61 | 0.5291 | 0.7927 | 0.5385 | 0.6231 | 81.8s | validation |
| 3 | E20_lgbm_compact | LightGBM compact | Nonlinear interactions improve compact features | 61 | 0.5211 | 0.8002 | 0.5000 | 0.1042 | 8.4s | validation |
| 4 | E22_lgbm_shape | LightGBM shape | Normalized Fourier and time-of-week shape adds transferable evidence | 61 | 0.5046 | 0.7853 | 0.6000 | 0.1070 | 19.0s | validation |
| 5 | E21_lgbm_events | LightGBM events | Charging-event evidence adds value beyond compact statistics | 61 | 0.4694 | 0.8002 | 0.5000 | 0.1064 | 30.1s | validation |
| 6 | E02_logistic | logistic | Compact signal statistics improve on prevalence | 61 | 0.4150 | 0.8291 | 0.5517 | 0.1157 | 5.8s | validation |
| 7 | E24_calibration | calibration | Grouped out-of-fold sigmoid calibration improves probability reliability | 61 | 0.4150 | 0.8291 | 0.5517 | 0.1157 | 6.6s | validation |
| 8 | E11_plateau_edges | plateau | Paired edges and flatness reject heating-like blocks | 61 | 0.3202 | 0.6677 | 0.3590 | 0.3355 | 6.9s | validation |
| 9 | E01_dummy | dummy | Establish customer-balanced prevalence floor | 61 | 0.1475 | 0.5000 | 0.2571 | 0.1258 | 5.2s | validation |
<!-- END LEADERBOARD -->

## Experiment log

### E00 — Data and label audit

All schema, load, duplicate, quality, DST, register-sign and source-count checks passed.

```json
{
  "rows": 27927,
  "customers": 331,
  "load_columns": 672,
  "excluded_rows": 2241,
  "unknown_eligible_rows": 212,
  "target_counts": {
    "0.0": 22424,
    "1.0": 3050
  },
  "date_field_completeness": {
    "datum_unterschrift_earliest": 331,
    "geplanter_baustart_earliest": 184,
    "uebergabe_latest": 234,
    "inbetrieb_datum_latest": 240
  },
  "splits": {
    "development": {
      "all_customers": 199,
      "primary_customers": 183,
      "primary_positive": 26,
      "positive_without_post_week": 15,
      "insufficient_history": 16,
      "supervised_weeks": 16289
    },
    "validation": {
      "all_customers": 66,
      "primary_customers": 61,
      "primary_positive": 9,
      "positive_without_post_week": 5,
      "insufficient_history": 5,
      "supervised_weeks": 4943
    },
    "test": {
      "all_customers": 66,
      "primary_customers": 61,
      "primary_positive": 9,
      "positive_without_post_week": 5,
      "insufficient_history": 5,
      "supervised_weeks": 4242
    }
  },
  "quality_pass": true,
  "units": "kWh / 15 min; kW = 4*kWh",
  "input_summary": {
    "canonical_customer_count": 878,
    "linked_customer_count": 337,
    "unlinked_customer_count": 541,
    "observed_customer_count": 337,
    "eligible_customer_count": 331,
    "attempted_week_count": 33800,
    "eligible_week_count": 27927,
    "monthly_export_count": 42,
    "excluded_source": "temporary March 2023 export",
    "import_obis": "1-1:1.29.0*255",
    "export_obis": "1-1:2.29.0*255",
    "timezone_assumption": "Europe/Zurich",
    "pv_labelled_negative_interval_fraction": 0.3201067636742819,
    "non_pv_labelled_negative_interval_fraction": 0.1072152766593296
  }
}
```

Only 331 of 878 source customers have retained weeks. Eligibility excludes contradictory labels. Validation has few positives; uncertainty and cohort selection limit generalization.


Decision: E00 passed; run prevalence and compact logistic baselines before adding detector complexity.


### E01_dummy — dummy

- Hypothesis: Establish customer-balanced prevalence floor
- Inputs/split: eligible labelled development / validation; fixed 61-customer cohort, 12-week cap.
- Method and parameters: `{"kind": "dummy", "prevalence": 0.14133941093969146, "features": "compact"}`; aggregation `max`, threshold 0.141339.
- Result (validation): AP 0.1475 [0.0656, 0.2295], balanced accuracy 0.5000, F1 0.2571, Brier 0.1258.
- Paired comparison: null.
- Decision: promoted.
- Error slices/observations: [suppressed-small-cell slices](classification_output/E01_dummy/slices.csv), [diagnostics](classification_output/E01_dummy/diagnostics.png).
- Artifacts: [configuration](classification_output/E01_dummy/config.json), [metrics](classification_output/E01_dummy/metrics_validation.json).


Decision: E01_dummy completed; run logistic baseline next.


### E02_logistic — logistic

- Hypothesis: Compact signal statistics improve on prevalence
- Inputs/split: eligible labelled development / validation; fixed 61-customer cohort, 12-week cap.
- Method and parameters: `{"kind": "logistic", "C": 1.0, "features": "compact"}`; aggregation `top3`, threshold 0.167205.
- Result (validation): AP 0.4150 [0.2152, 0.7644], balanced accuracy 0.8291, F1 0.5517, Brier 0.1157.
- Paired comparison: {"delta_pr_auc": 0.26743917512360144, "ci95": [0.10950388591154533, 0.6000103681663307], "draws": 1000}.
- Decision: promoted.
- Error slices/observations: [suppressed-small-cell slices](classification_output/E02_logistic/slices.csv), [diagnostics](classification_output/E02_logistic/diagnostics.png).
- Artifacts: [configuration](classification_output/E02_logistic/config.json), [metrics](classification_output/E02_logistic/metrics_validation.json).


Decision: E02_logistic completed; inspect baseline uncertainty, then test sustained charging events.


### E10_sustained_block — block

- Hypothesis: Repeated sustained residual loads distinguish chargers
- Inputs/split: eligible labelled development / validation; fixed 61-customer cohort, 12-week cap.
- Method and parameters: `{"family": "block", "baseline": "daily_q20", "amplitude": 3.0, "duration": 1.5, "min_events": 3}`; aggregation `top3`, threshold 0.954917.
- Result (validation): AP 0.5291 [0.2114, 0.8239], balanced accuracy 0.7927, F1 0.5385, Brier 0.6231.
- Paired comparison: {"delta_pr_auc": 0.11415029687088507, "ci95": [-0.2243766360345308, 0.37107526042505873], "draws": 1000}.
- Decision: retained earlier simpler champion; AP gain not established beyond paired uncertainty.
- Error slices/observations: [suppressed-small-cell slices](classification_output/E10_sustained_block/slices.csv), [diagnostics](classification_output/E10_sustained_block/diagnostics.png).
- Artifacts: [configuration](classification_output/E10_sustained_block/config.json), [metrics](classification_output/E10_sustained_block/metrics_validation.json).


Decision: E10_sustained_block complete; heuristic champion E10_sustained_block. Continue mandatory families before optional complexity.


### E11_plateau_edges — plateau

- Hypothesis: Paired edges and flatness reject heating-like blocks
- Inputs/split: eligible labelled development / validation; fixed 61-customer cohort, 12-week cap.
- Method and parameters: `{"baseline": "daily_q20", "amplitude": 3.0, "duration": 1.5, "min_events": 3, "family": "plateau", "variation": 0.2, "edge": 0.5, "symmetry": 0.3}`; aggregation `top3`, threshold 0.701642.
- Result (validation): AP 0.3202 [0.1101, 0.5676], balanced accuracy 0.6677, F1 0.3590, Brier 0.3355.
- Paired comparison: {"delta_pr_auc": -0.09482448857448866, "ci95": [-0.46068237171632853, 0.16527003601828325], "draws": 1000}.
- Decision: retained earlier simpler champion; AP gain not established beyond paired uncertainty.
- Error slices/observations: [suppressed-small-cell slices](classification_output/E11_plateau_edges/slices.csv), [diagnostics](classification_output/E11_plateau_edges/diagnostics.png).
- Artifacts: [configuration](classification_output/E11_plateau_edges/config.json), [metrics](classification_output/E11_plateau_edges/metrics_validation.json).


Decision: E11_plateau_edges complete; heuristic champion E10_sustained_block. Test repetition next.


### E12_repeated_level — repeat

- Hypothesis: Repeated power levels across days reject isolated appliance loads
- Inputs/split: eligible labelled development / validation; fixed 61-customer cohort, 12-week cap.
- Method and parameters: `{"baseline": "daily_q20", "amplitude": 3.0, "duration": 1.5, "min_events": 3, "family": "repeat", "tolerance": 0.5, "repeat_days": 3}`; aggregation `top3`, threshold 0.943647.
- Result (validation): AP 0.6485 [0.3199, 0.8977], balanced accuracy 0.8098, F1 0.5161, Brier 0.6070.
- Paired comparison: {"delta_pr_auc": 0.2335637093259043, "ci95": [-0.10962400824388338, 0.4720348788638262], "draws": 1000}.
- Decision: retained earlier simpler champion; AP gain not established beyond paired uncertainty.
- Error slices/observations: [suppressed-small-cell slices](classification_output/E12_repeated_level/slices.csv), [diagnostics](classification_output/E12_repeated_level/diagnostics.png).
- Artifacts: [configuration](classification_output/E12_repeated_level/config.json), [metrics](classification_output/E12_repeated_level/metrics_validation.json).


Decision: E12_repeated_level complete; heuristic champion E10_sustained_block. Continue mandatory families before optional complexity.


Decision: E13 template-only detector and E14 high-pass/wavelet skipped: two successive full heuristic extensions failed to improve the simpler detector beyond paired uncertainty. Template evidence remains available in E21 features.


### E20_lgbm_compact — LightGBM compact

- Hypothesis: Nonlinear interactions improve compact features
- Inputs/split: eligible labelled development / validation; fixed 61-customer cohort, 12-week cap.
- Method and parameters: `{"kind": "lgbm", "features": "compact", "heuristic": {"family": "block", "baseline": "daily_q20", "amplitude": 3.0, "duration": 1.5, "min_events": 3}, "parameters": {"num_leaves": 7, "learning_rate": 0.03, "n_estimators": 350, "min_child_samples": 200, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 1.0, "reg_lambda": 15.0}, "refit_parameters": {"num_leaves": 7, "learning_rate": 0.03, "n_estimators": 95, "min_child_samples": 200, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 1.0, "reg_lambda": 15.0}}`; aggregation `max`, threshold 0.162149.
- Result (validation): AP 0.5211 [0.2507, 0.8335], balanced accuracy 0.8002, F1 0.5000, Brier 0.1042.
- Paired comparison: {"delta_pr_auc": 0.10616129833807164, "ci95": [-0.21520445854417145, 0.3709731023933313], "draws": 1000}.
- Decision: retained earlier simpler champion; AP gain not established beyond paired uncertainty.
- Error slices/observations: [suppressed-small-cell slices](classification_output/E20_lgbm_compact/slices.csv), [diagnostics](classification_output/E20_lgbm_compact/diagnostics.png).
- Artifacts: [configuration](classification_output/E20_lgbm_compact/config.json), [metrics](classification_output/E20_lgbm_compact/metrics_validation.json).


Decision: E20_lgbm_compact finished; test explicit event features next.


### E21_lgbm_events — LightGBM events

- Hypothesis: Charging-event evidence adds value beyond compact statistics
- Inputs/split: eligible labelled development / validation; fixed 61-customer cohort, 12-week cap.
- Method and parameters: `{"kind": "lgbm", "features": "events", "heuristic": {"family": "block", "baseline": "daily_q20", "amplitude": 3.0, "duration": 1.5, "min_events": 3}, "parameters": {"num_leaves": 7, "learning_rate": 0.03, "n_estimators": 350, "min_child_samples": 200, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 1.0, "reg_lambda": 15.0}, "refit_parameters": {"num_leaves": 7, "learning_rate": 0.03, "n_estimators": 88, "min_child_samples": 200, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 1.0, "reg_lambda": 15.0}}`; aggregation `max`, threshold 0.164142.
- Result (validation): AP 0.4694 [0.2262, 0.8145], balanced accuracy 0.8002, F1 0.5000, Brier 0.1064.
- Paired comparison: {"delta_pr_auc": 0.054454031718524365, "ci95": [-0.26868506497239425, 0.35000953849212363], "draws": 1000}.
- Decision: retained earlier simpler champion; AP gain not established beyond paired uncertainty.
- Error slices/observations: [suppressed-small-cell slices](classification_output/E21_lgbm_events/slices.csv), [diagnostics](classification_output/E21_lgbm_events/diagnostics.png).
- Artifacts: [configuration](classification_output/E21_lgbm_events/config.json), [metrics](classification_output/E21_lgbm_events/metrics_validation.json).


Decision: E21_lgbm_events finished; compare normalized temporal shape with feature-family ablation.


### E22_lgbm_shape — LightGBM shape

- Hypothesis: Normalized Fourier and time-of-week shape adds transferable evidence
- Inputs/split: eligible labelled development / validation; fixed 61-customer cohort, 12-week cap.
- Method and parameters: `{"kind": "lgbm", "features": "shape", "heuristic": {"family": "block", "baseline": "daily_q20", "amplitude": 3.0, "duration": 1.5, "min_events": 3}, "parameters": {"num_leaves": 7, "learning_rate": 0.03, "n_estimators": 350, "min_child_samples": 200, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 1.0, "reg_lambda": 15.0}, "refit_parameters": {"num_leaves": 7, "learning_rate": 0.03, "n_estimators": 57, "min_child_samples": 200, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 1.0, "reg_lambda": 15.0}}`; aggregation `max`, threshold 0.223698.
- Result (validation): AP 0.5046 [0.2642, 0.8477], balanced accuracy 0.7853, F1 0.6000, Brier 0.1070.
- Paired comparison: {"delta_pr_auc": 0.08963837088837079, "ci95": [-0.24241353894983553, 0.3801055697923055], "draws": 1000}.
- Decision: retained earlier simpler champion; AP gain not established beyond paired uncertainty.
- Error slices/observations: [suppressed-small-cell slices](classification_output/E22_lgbm_shape/slices.csv), [diagnostics](classification_output/E22_lgbm_shape/diagnostics.png).
- Artifacts: [configuration](classification_output/E22_lgbm_shape/config.json), [metrics](classification_output/E22_lgbm_shape/metrics_validation.json).


Decision: E22_lgbm_shape finished; major model families complete; inspect errors and compare calibration.


### E24_calibration — calibration

- Hypothesis: Grouped out-of-fold sigmoid calibration improves probability reliability
- Inputs/split: eligible labelled development / validation; fixed 61-customer cohort, 12-week cap.
- Method and parameters: `{"kind": "logistic", "C": 1.0, "features": "compact", "base_experiment": "E02_logistic", "calibration": "uncalibrated", "calibration_fit": "3-fold grouped development OOF; final refit uses development+validation OOF"}`; aggregation `top3`, threshold 0.167205.
- Result (validation): AP 0.4150 [0.2152, 0.7644], balanced accuracy 0.8291, F1 0.5517, Brier 0.1157.
- Paired comparison: {"delta_pr_auc": 0.0, "ci95": [0.0, 0.0], "draws": 1000}.
- Decision: retained earlier simpler champion; AP gain not established beyond paired uncertainty.
- Error slices/observations: [suppressed-small-cell slices](classification_output/E24_calibration/slices.csv), [diagnostics](classification_output/E24_calibration/diagnostics.png).
- Artifacts: [configuration](classification_output/E24_calibration/config.json), [metrics](classification_output/E24_calibration/metrics_validation.json).


Decision: E24 selected uncalibrated on E02_logistic; isotonic skipped for insufficient calibration population. E23 alternative tree skipped because LightGBM is available. No additional algorithm search justified.


### E30 — Date-label sensitivity (targeted analysis)

- Hypothesis: If project-boundary label noise dominates, early/late-only bounds or 7/28-day washouts materially change validation performance.
- Method: frozen E02 weekly model and threshold; recompute only temporal targets and fixed-rule customer aggregation.
- Result: all four definitions retained 61 comparison customers and 9 positives, with customer AP 0.4150, balanced accuracy 0.8291, and F1 0.5517. The eligible-but-unknown week count was 2 for the primary and early/late-field definitions, 3 after a 7-day washout, and 6 after a 28-day washout. [Complete metrics](classification_output/date_sensitivity.json).
- Decision: sensitivity analysis only; no feature, model, aggregation, or threshold was changed. Large differences are treated as date-label uncertainty.

## Decision log

- E00 passed; run prevalence and compact logistic baselines before adding detector complexity.
- E01_dummy completed; run logistic baseline next.
- E02_logistic completed; inspect baseline uncertainty, then test sustained charging events.
- E10_sustained_block complete; heuristic champion E10_sustained_block. Continue mandatory families before optional complexity.
- E11_plateau_edges complete; heuristic champion E10_sustained_block. Test repetition next.
- E12_repeated_level complete; heuristic champion E10_sustained_block. Continue mandatory families before optional complexity.
- E13 template-only detector and E14 high-pass/wavelet skipped: two successive full heuristic extensions failed to improve the simpler detector beyond paired uncertainty. Template evidence remains available in E21 features.
- E20_lgbm_compact finished; test explicit event features next.
- E21_lgbm_events finished; compare normalized temporal shape with feature-family ablation.
- E22_lgbm_shape finished; major model families complete; inspect errors and compare calibration.
- E24 selected uncalibrated on E02_logistic; isotonic skipped for insufficient calibration population. E23 alternative tree skipped because LightGBM is available. No additional algorithm search justified.

## Final freeze

FINAL FREEZE — written before test prediction.

```json
{
  "created_utc": "2026-09-10T22:45:02.389390+00:00",
  "selected": "E02_logistic",
  "pipelines": {
    "E02_logistic": {
      "config": {
        "kind": "logistic",
        "C": 1.0,
        "features": "compact",
        "aggregation": "top3",
        "threshold": 0.16720476380141022,
        "study_config": {
          "seed": 20260910,
          "split": [
            0.6,
            0.2,
            0.2
          ],
          "history_cap": 12,
          "minimum_history": 2,
          "primary": "customer_average_precision",
          "threshold_metric": "balanced_accuracy",
          "bootstrap_draws": 1000,
          "privacy_min_cell": 5,
          "minimum_validation_positives": 8,
          "baselines": [
            "daily_q20",
            "rolling_q20"
          ],
          "amplitudes_kw": [
            1.5,
            3.0,
            5.0
          ],
          "durations_hours": [
            1.0,
            1.5
          ],
          "min_events": [
            2,
            3
          ],
          "aggregations": [
            "max",
            "top3",
            "evidence"
          ],
          "plateau_variations": [
            0.2,
            0.4
          ],
          "edge_kw": [
            0.5,
            1.0
          ],
          "edge_symmetry": 0.3,
          "repeat_tolerance_kw": [
            0.5,
            1.0
          ],
          "repeat_days": [
            2,
            3
          ],
          "lgbm_grid": [
            {
              "num_leaves": 7,
              "learning_rate": 0.05,
              "n_estimators": 250,
              "min_child_samples": 100,
              "subsample": 0.8,
              "colsample_bytree": 0.9,
              "reg_alpha": 0.1,
              "reg_lambda": 5.0
            },
            {
              "num_leaves": 15,
              "learning_rate": 0.03,
              "n_estimators": 350,
              "min_child_samples": 150,
              "subsample": 0.8,
              "colsample_bytree": 0.8,
              "reg_alpha": 0.5,
              "reg_lambda": 10.0
            },
            {
              "num_leaves": 7,
              "learning_rate": 0.03,
              "n_estimators": 350,
              "min_child_samples": 200,
              "subsample": 1.0,
              "colsample_bytree": 1.0,
              "reg_alpha": 1.0,
              "reg_lambda": 15.0
            }
          ],
          "early_stopping_rounds": 30,
          "max_experiments": 12,
          "date_bounds": "min/max all four; half-open weeks; eligibility gate first"
        },
        "source_identity": "69a33762f866aeca210389e6866fedd3aa01c65af74cfe72b4225c92f2658fbc",
        "code_hash": "62bbf4d9cdc3ad3d75d5ac9ad077de151dab083bf6db613a884111efa78d0525"
      },
      "feature_manifest": {
        "features": [
          "stat_q5",
          "stat_q25",
          "stat_q50",
          "stat_q75",
          "stat_q95",
          "stat_q99",
          "stat_range",
          "stat_mean",
          "stat_load_factor",
          "stat_ramp50",
          "stat_ramp95",
          "stat_ramp99",
          "stat_night_kwh",
          "stat_day_kwh",
          "stat_weekday_daily_kwh",
          "stat_weekend_daily_kwh",
          "stat_hours_above_1.5",
          "stat_hours_above_3",
          "stat_hours_above_5",
          "stat_hours_above_7",
          "stat_hours_above_11"
        ],
        "hash": "a663ff378d9fa8df1b3970fc0ff2ab25ae861f230c2a0cc1e6f14f81f0d6eb07",
        "prohibited_columns": [
          "battery_storage_label",
          "datum_unterschrift_earliest",
          "eligible_for_supervised_training",
          "ev_label",
          "geplanter_baustart_earliest",
          "gp_nr",
          "heat_pump_boiler_label",
          "heat_pump_label",
          "inbetrieb_datum_latest",
          "kanton",
          "label_date_ambiguous",
          "ort",
          "plz",
          "pv_capacity_kwp",
          "pv_label",
          "split",
          "temporal_target",
          "training_exclusion_reason",
          "uebergabe_latest",
          "week_start",
          "year"
        ],
        "prohibited_check_passed": true
      }
    },
    "E01_dummy": {
      "config": {
        "kind": "dummy",
        "prevalence": 0.14133941093969146,
        "features": "compact",
        "aggregation": "max",
        "threshold": 0.14133941093969146,
        "study_config": {
          "seed": 20260910,
          "split": [
            0.6,
            0.2,
            0.2
          ],
          "history_cap": 12,
          "minimum_history": 2,
          "primary": "customer_average_precision",
          "threshold_metric": "balanced_accuracy",
          "bootstrap_draws": 1000,
          "privacy_min_cell": 5,
          "minimum_validation_positives": 8,
          "baselines": [
            "daily_q20",
            "rolling_q20"
          ],
          "amplitudes_kw": [
            1.5,
            3.0,
            5.0
          ],
          "durations_hours": [
            1.0,
            1.5
          ],
          "min_events": [
            2,
            3
          ],
          "aggregations": [
            "max",
            "top3",
            "evidence"
          ],
          "plateau_variations": [
            0.2,
            0.4
          ],
          "edge_kw": [
            0.5,
            1.0
          ],
          "edge_symmetry": 0.3,
          "repeat_tolerance_kw": [
            0.5,
            1.0
          ],
          "repeat_days": [
            2,
            3
          ],
          "lgbm_grid": [
            {
              "num_leaves": 7,
              "learning_rate": 0.05,
              "n_estimators": 250,
              "min_child_samples": 100,
              "subsample": 0.8,
              "colsample_bytree": 0.9,
              "reg_alpha": 0.1,
              "reg_lambda": 5.0
            },
            {
              "num_leaves": 15,
              "learning_rate": 0.03,
              "n_estimators": 350,
              "min_child_samples": 150,
              "subsample": 0.8,
              "colsample_bytree": 0.8,
              "reg_alpha": 0.5,
              "reg_lambda": 10.0
            },
            {
              "num_leaves": 7,
              "learning_rate": 0.03,
              "n_estimators": 350,
              "min_child_samples": 200,
              "subsample": 1.0,
              "colsample_bytree": 1.0,
              "reg_alpha": 1.0,
              "reg_lambda": 15.0
            }
          ],
          "early_stopping_rounds": 30,
          "max_experiments": 12,
          "date_bounds": "min/max all four; half-open weeks; eligibility gate first"
        },
        "source_identity": "69a33762f866aeca210389e6866fedd3aa01c65af74cfe72b4225c92f2658fbc",
        "code_hash": "62bbf4d9cdc3ad3d75d5ac9ad077de151dab083bf6db613a884111efa78d0525"
      },
      "feature_manifest": {
        "features": [
          "stat_q5",
          "stat_q25",
          "stat_q50",
          "stat_q75",
          "stat_q95",
          "stat_q99",
          "stat_range",
          "stat_mean",
          "stat_load_factor",
          "stat_ramp50",
          "stat_ramp95",
          "stat_ramp99",
          "stat_night_kwh",
          "stat_day_kwh",
          "stat_weekday_daily_kwh",
          "stat_weekend_daily_kwh",
          "stat_hours_above_1.5",
          "stat_hours_above_3",
          "stat_hours_above_5",
          "stat_hours_above_7",
          "stat_hours_above_11"
        ],
        "hash": "a663ff378d9fa8df1b3970fc0ff2ab25ae861f230c2a0cc1e6f14f81f0d6eb07",
        "prohibited_columns": [
          "battery_storage_label",
          "datum_unterschrift_earliest",
          "eligible_for_supervised_training",
          "ev_label",
          "geplanter_baustart_earliest",
          "gp_nr",
          "heat_pump_boiler_label",
          "heat_pump_label",
          "inbetrieb_datum_latest",
          "kanton",
          "label_date_ambiguous",
          "ort",
          "plz",
          "pv_capacity_kwp",
          "pv_label",
          "split",
          "temporal_target",
          "training_exclusion_reason",
          "uebergabe_latest",
          "week_start",
          "year"
        ],
        "prohibited_check_passed": true
      }
    },
    "E10_sustained_block": {
      "config": {
        "family": "block",
        "baseline": "daily_q20",
        "amplitude": 3.0,
        "duration": 1.5,
        "min_events": 3,
        "aggregation": "top3",
        "threshold": 0.9549172899723399,
        "study_config": {
          "seed": 20260910,
          "split": [
            0.6,
            0.2,
            0.2
          ],
          "history_cap": 12,
          "minimum_history": 2,
          "primary": "customer_average_precision",
          "threshold_metric": "balanced_accuracy",
          "bootstrap_draws": 1000,
          "privacy_min_cell": 5,
          "minimum_validation_positives": 8,
          "baselines": [
            "daily_q20",
            "rolling_q20"
          ],
          "amplitudes_kw": [
            1.5,
            3.0,
            5.0
          ],
          "durations_hours": [
            1.0,
            1.5
          ],
          "min_events": [
            2,
            3
          ],
          "aggregations": [
            "max",
            "top3",
            "evidence"
          ],
          "plateau_variations": [
            0.2,
            0.4
          ],
          "edge_kw": [
            0.5,
            1.0
          ],
          "edge_symmetry": 0.3,
          "repeat_tolerance_kw": [
            0.5,
            1.0
          ],
          "repeat_days": [
            2,
            3
          ],
          "lgbm_grid": [
            {
              "num_leaves": 7,
              "learning_rate": 0.05,
              "n_estimators": 250,
              "min_child_samples": 100,
              "subsample": 0.8,
              "colsample_bytree": 0.9,
              "reg_alpha": 0.1,
              "reg_lambda": 5.0
            },
            {
              "num_leaves": 15,
              "learning_rate": 0.03,
              "n_estimators": 350,
              "min_child_samples": 150,
              "subsample": 0.8,
              "colsample_bytree": 0.8,
              "reg_alpha": 0.5,
              "reg_lambda": 10.0
            },
            {
              "num_leaves": 7,
              "learning_rate": 0.03,
              "n_estimators": 350,
              "min_child_samples": 200,
              "subsample": 1.0,
              "colsample_bytree": 1.0,
              "reg_alpha": 1.0,
              "reg_lambda": 15.0
            }
          ],
          "early_stopping_rounds": 30,
          "max_experiments": 12,
          "date_bounds": "min/max all four; half-open weeks; eligibility gate first"
        },
        "source_identity": "69a33762f866aeca210389e6866fedd3aa01c65af74cfe72b4225c92f2658fbc",
        "code_hash": "62bbf4d9cdc3ad3d75d5ac9ad077de151dab083bf6db613a884111efa78d0525"
      },
      "feature_manifest": {
        "features": [
          "event_count",
          "event_days",
          "event_duration",
          "event_energy",
          "event_longest",
          "event_power",
          "event_plateau",
          "event_paired",
          "event_repetition",
          "event_template",
          "event_night",
          "event_negative",
          "event_interruptions"
        ],
        "hash": "5f715fd4e1c3c04f466ee920a0c93af101cf0ae5d937ac99f1c83fe0ee60f636",
        "prohibited_columns": [
          "battery_storage_label",
          "datum_unterschrift_earliest",
          "eligible_for_supervised_training",
          "ev_label",
          "geplanter_baustart_earliest",
          "gp_nr",
          "heat_pump_boiler_label",
          "heat_pump_label",
          "inbetrieb_datum_latest",
          "kanton",
          "label_date_ambiguous",
          "ort",
          "plz",
          "pv_capacity_kwp",
          "pv_label",
          "split",
          "temporal_target",
          "training_exclusion_reason",
          "uebergabe_latest",
          "week_start",
          "year"
        ],
        "prohibited_check_passed": true
      }
    }
  },
  "source_identity": "69a33762f866aeca210389e6866fedd3aa01c65af74cfe72b4225c92f2658fbc",
  "minimum_history": 2,
  "history_cap": 12,
  "stop_reason": "Major heuristic and LightGBM families evaluated; prefer simpler pipelines within paired AP uncertainty.",
  "test_status": "unopened: no test predictions computed",
  "code_hash": "3c70d9b9fb671bb320b21e6d38db12a4d1cf94f45207cd70a1c9ef8396e3f48f",
  "freeze_hash": "ba19a71cae403a43c70de54f9c217642f2bdbfc95c8abffcca19e240ac275c66"
}
```


## Final test

One-time held-out customer evaluation after refitting on development plus validation.

The frozen `E02_logistic` pipeline achieved customer AP 0.2546 (95% customer-bootstrap interval 0.0852--0.5769), balanced accuracy 0.5972, and F1 0.3077 on 61 customers with 9 positives. Its AP improvement over the dummy was 0.1070 (paired 95% interval -0.0073--0.4070), so the held-out improvement is uncertain. The sustained-block heuristic had AP 0.2848; the selected-minus-heuristic difference was -0.0303 (paired interval -0.2413--0.1940). This does not change the frozen selection.

```json
{
  "E02_logistic": {
    "customer": {
      "customers_or_weeks": 61,
      "positives": 9,
      "pr_auc": 0.2545755846070311,
      "balanced_accuracy": 0.5972222222222222,
      "f1": 0.3076923076923077,
      "recall": 0.4444444444444444,
      "precision": 0.23529411764705882,
      "specificity": 0.75,
      "roc_auc": 0.5555555555555556,
      "brier": 0.12375276502097975,
      "confusion_matrix": [
        [
          39,
          13
        ],
        [
          5,
          4
        ]
      ]
    },
    "week": {
      "customers_or_weeks": 4242,
      "positives": 393,
      "pr_auc": 0.22559691421338118,
      "balanced_accuracy": 0.6835670611381166,
      "f1": 0.33307271305707586,
      "recall": 0.5419847328244275,
      "precision": 0.24040632054176073,
      "specificity": 0.8251493894518057,
      "roc_auc": 0.6969901306112357,
      "brier": 0.08053870935065498,
      "confusion_matrix": [
        [
          3176,
          673
        ],
        [
          180,
          213
        ]
      ]
    },
    "ci95": {
      "pr_auc": [
        0.08519393547856276,
        0.5768800220247589
      ],
      "balanced_accuracy": [
        0.4122388357256778,
        0.7803119639794168
      ],
      "f1": [
        0.07685185185185187,
        0.5217391304347826
      ],
      "brier": [
        0.07009498738267429,
        0.18545032294691682
      ]
    },
    "selected_minus_baseline": {
      "delta_pr_auc": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "draws": 1000
    }
  },
  "E01_dummy": {
    "customer": {
      "customers_or_weeks": 61,
      "positives": 9,
      "pr_auc": 0.14754098360655737,
      "balanced_accuracy": 0.5,
      "f1": 0.2571428571428571,
      "recall": 1.0,
      "precision": 0.14754098360655737,
      "specificity": 0.0,
      "roc_auc": 0.5,
      "brier": 0.1257943341403419,
      "confusion_matrix": [
        [
          0,
          52
        ],
        [
          0,
          9
        ]
      ]
    },
    "week": {
      "customers_or_weeks": 4242,
      "positives": 393,
      "pr_auc": 0.09264497878359264,
      "balanced_accuracy": 0.5,
      "f1": 0.16957928802588998,
      "recall": 1.0,
      "precision": 0.09264497878359264,
      "specificity": 0.0,
      "roc_auc": 0.5,
      "brier": 0.08658579328943283,
      "confusion_matrix": [
        [
          0,
          3849
        ],
        [
          0,
          393
        ]
      ]
    },
    "ci95": {
      "pr_auc": [
        0.06557377049180328,
        0.2459016393442623
      ],
      "balanced_accuracy": [
        0.5,
        0.5
      ],
      "f1": [
        0.12307692307692308,
        0.39473684210526316
      ],
      "brier": [
        0.06725064167657693,
        0.19604676509685992
      ]
    },
    "selected_minus_baseline": {
      "delta_pr_auc": 0.10703460100047374,
      "ci95": [
        -0.00728217336044689,
        0.4069748028414034
      ],
      "draws": 1000
    }
  },
  "E10_sustained_block": {
    "customer": {
      "customers_or_weeks": 61,
      "positives": 9,
      "pr_auc": 0.28482734804573884,
      "balanced_accuracy": 0.5512820512820513,
      "f1": 0.25,
      "recall": 0.3333333333333333,
      "precision": 0.2,
      "specificity": 0.7692307692307693,
      "roc_auc": 0.6826923076923077,
      "brier": 0.6227893550920051,
      "confusion_matrix": [
        [
          40,
          12
        ],
        [
          6,
          3
        ]
      ]
    },
    "week": {
      "customers_or_weeks": 4242,
      "positives": 393,
      "pr_auc": 0.15403307390389082,
      "balanced_accuracy": 0.5331621775458679,
      "f1": 0.14878397711015737,
      "recall": 0.13231552162849872,
      "precision": 0.16993464052287582,
      "specificity": 0.9340088334632372,
      "roc_auc": 0.6835918519532188,
      "brier": 0.4137673334812665,
      "confusion_matrix": [
        [
          3595,
          254
        ],
        [
          341,
          52
        ]
      ]
    },
    "ci95": {
      "pr_auc": [
        0.11927424242424244,
        0.5687357055121925
      ],
      "balanced_accuracy": [
        0.3867924528301887,
        0.7530868205868205
      ],
      "f1": [
        0.0,
        0.47619047619047616
      ],
      "brier": [
        0.5284924245535089,
        0.7106875168165702
      ]
    },
    "selected_minus_baseline": {
      "delta_pr_auc": -0.030251763438707724,
      "ci95": [
        -0.24131091700872162,
        0.19400981191336197
      ],
      "draws": 1000
    }
  }
}
```


## Recommendation and limitations

Selected pipeline: `E02_logistic`; deployable local configuration and fitted objects are in [best/](classification_output/best/). Treat this as a small-cohort research baseline, not verified ownership or a production-ready classifier. Customer AP fell from 0.4150 on validation to 0.2546 on test, and the test improvement over the dummy is not established beyond paired uncertainty. A charger can be unused; heating, boilers and other loads can mimic charging; PV and batteries can mask it. Project dates are not confirmed commissioning dates. Only a subset of labelled customers has usable data. Repeated validation searches and the small number of positive customers create substantial selection uncertainty. Use weekly/event evidence for review; insufficient-history customers abstain, and excluded/unknown rows are exploratory only. No pipeline changes were made in response to final-test outcomes.
