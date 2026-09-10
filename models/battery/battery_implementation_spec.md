# Implementation spec — battery detection pipeline

Hand-off document for a coding agent. Self-contained: assume no access to the
conversation that produced it.

**Read first:** [`battery_mvp_plan.md`](battery_mvp_plan.md) (what and why),
[`battery_evidence_base.md`](battery_evidence_base.md) (justification),
[`data_problems.md`](data_problems.md) (every data trap already discovered).
This file says *how to build it*.

---

## 0. Task in one paragraph

Build `models/battery/`, a pipeline that outputs one calibrated probability per
household (`GP-Nr`) that the household operates a behind-the-meter battery. It
reuses the existing PV pipeline's I/O layer unchanged, adds a new streaming
feature pass over the 15-minute smart-meter exports, and trains a
Positive-Unlabeled classifier. Binary classification only — no capacity
estimation, no activity windows, no install-year output.

---

## 1. Environment and ground rules

| | |
|---|---|
| Python | 3.13 |
| Available | `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib` |
| **Not** available | `pyarrow`, `xgboost`, `lightgbm`, `duckdb`, `sktime`, `shap` |
| On-disk interchange | `pickle` (`.pkl`) and CSV — no parquet |
| Data root | `$PV_WORK_ROOT`, default `/home/renku/work`; exports under `store/input_data/` |
| Volume | 42 monthly CSVs, 0.8–3.4 GB each, ~77 GB total |
| Seed | `models.config.SEED` everywhere; no unseeded randomness |

**Hard rules — violating any of these invalidates the result:**

1. **Parse the exports by column position, never by header name.** The
   `April 2023` file has a malformed header. `models/stream.py` already does
   this; use it, do not write a new reader.
2. **Never stream the same file twice.** One pass over the 77 GB, accumulating
   aggregates. Do not collect raw rows.
3. **The universe gate is label-blind** (§4.1). No branch of it may consult a
   label, or the model learns the selection rule instead of the battery.
4. **The 82 `Batterie = -` households never enter training** — not as negatives,
   and not as part of the unlabelled pool either. They are an audit set.
5. **Do not modify anything under `models/` outside `models/battery/`.** The PV
   pipeline must still run unchanged.

---

## 2. What to reuse verbatim

Import these; do not copy or fork them:

| Module | What you get |
|---|---|
| `models.config` | paths, `discover_monthly_exports()`, `QUARTER_HOUR_COLS`, `OBIS_IMPORT`/`OBIS_EXPORT`, season constants, `SEED` |
| `models.stream` | `iter_chunks(path, chunksize, obis)` → `(meta, values)`; `hour_band_sum(values, hours)` |
| `models.io_utils` | `read_gigi(annotated=True)`, `gp_to_mpid()`, `read_zaehler_gp()` |
| `models.build_weather` | `weather_daily.csv`: daily radiation, `is_clear_sky` flag |
| `models/artifacts/meter_flags.csv` | `is_production_meter` per `mp_id` — exclude these meters |
| `models/artifacts/model/oof_predictions.csv` | out-of-fold `pv_probability` per `gp_nr` |

`iter_chunks` yields `meta` (columns `mp_id, obis, date, plz`; `date` is
`DD.MM.YYYY` strings) and `values`, a `float32 (n, 96)` array of quarter-hour
kWh. Column *k* (0-based) is the 15 minutes ending at `(k+1)*15` minutes after
midnight, so **clock hour `h` is slots `[4h, 4h+4)`**.

---

## 3. Module layout to create

```
models/battery/
  __init__.py
  config.py          # battery-specific paths + constants; imports models.config
  solar.py           # sunrise/sunset for canton AG
  build_labels.py    # Step 1
  define_universe.py # Step 2
  build_features.py  # Step 3  <- the one streaming pass
  build_features_agg.py # Step 4
  train.py           # Step 5
  evaluate.py        # Step 6
  score.py           # Step 7
  pipeline.py        # runs 1-7, with --skip-scan
```

Every module gets a `main()` and an `if __name__ == "__main__":` block so it can
be run alone, and prints the verification numbers listed in its section below.
Artifacts go to `models/artifacts/battery/` (create it; git-ignored).

---

## 4. Step-by-step specification

### 4.0 `solar.py`

```python
def sun_hours(date: pd.Timestamp) -> tuple[float, float]:
    """(sunrise, sunset) as decimal local hours for canton AG (47.4N, 8.1E)."""
```
Standard NOAA declination / hour-angle approximation is fine — ±5 minutes is far
below the 15-minute resolution. Vectorise over an array of dates
(`sun_hours_vec(dates) -> (np.ndarray, np.ndarray)`); it is called per meter-day.
Ignore DST vs UTC subtleties beyond a single fixed offset, but **write down in a
docstring which convention the export timestamps use** — verify it once by
checking that median peak export across summer days lands near local solar noon,
and assert it in a test.

### 4.1 `build_labels.py`

Read GIGI via `io_utils.read_gigi(annotated=True)` (columns already renamed:
`gp_nr, pv, battery, ev_charger, heat_pump, date_commissioned,
commissioned_refers_to, plz, kanton`). Group by `gp_nr`, drop blank `gp_nr`.

```
battery_positive = any row has battery.strip().lower() == "x"
audit_negative   = every row has battery.strip() == "-"     # and not positive
otherwise        = unlabelled (blank or "." present) -> neither
```

Battery commissioning date: among the household's rows, take those where
`commissioned_refers_to` contains `"batterie"` (case-insensitive), parse
`date_commissioned` with `format="%d.%m.%Y", errors="coerce"`, take the **min**.

Attach meters with `io_utils.gp_to_mpid()`.

Write `data_addition/battery_labels.csv`, one row per `(gp_nr, mp_id)`:
`gp_nr, mp_id, zpb, anlage, battery_positive, audit_negative,
battery_commissioning_date, has_pv_row, has_meter, n_gigi_rows, plz, kanton`.

**Acceptance — these must print exactly:**

| quantity | expected |
|---|---|
| unique `gp_nr` in GIGI | 878 |
| battery-positive households | 608 |
| … with meter data | 222 |
| … also PV-positive | 582 (212 with meter) |
| … no PV row | 26 (10 with meter) |
| audit-negative households (all rows `-`) | 214 (82 with meter) |
| … and PV-positive | 118 (53 with meter) |
| dated battery commissioning | 469 |
| commissioned 2023-04-01 … 2026-03-01 | 276 (105 with meter) |

If any number differs, stop and report — the label file changed and the rest of
the plan's arithmetic no longer holds.

### 4.2 `define_universe.py`

Runs **after** `build_features.py` has produced per-meter export statistics
(chicken-and-egg: resolve it by having `build_features` emit the export-day
counts for all meters in the same pass, then compute the universe from them).

```
in_universe(gp_nr) ⟺ the household's non-production meters recorded
                     OBIS 2.29 > 0 on >= 10 distinct days across the record
```

Fallback, only for meters with **no 2.29 rows at all** (as opposed to rows that
are present and zero): `pv_probability >= 0.8` from the PV model's
`oof_predictions.csv`. Count and report this subgroup's size; if it is under ~1%
of households, prefer excluding it and say so.

**No label may appear in this function.** Known positives are admitted on
exactly the same terms as everyone else.

**Acceptance:** ≥ 95% of the 222 known battery households must be retained.
Print the retained fraction and write the dropped `gp_nr` list to
`artifacts/battery/universe_dropped.csv`. If retention < 95%, stop and report
rather than loosening the rule silently.

### 4.3 `build_features.py` — the single streaming pass

The expensive step. Budget 1.5–2.5 h for the full run.

**Pairing the two channels.** Import and export for the same meter-day are
separate rows and may fall in different chunks. Do **not** buffer raw 96-vectors
across chunks. Per chunk, reduce each row immediately to one compact record —
the scalars below plus a **packed 96-bit near-zero mask** (two `uint64`
columns, slots 0–63 and 64–95) — append to a per-channel list, then at
**end of file** concatenate each channel into a DataFrame and
`merge(on=["mp_id", "date"], how="outer")`. Rows where one channel is missing
keep `has_both = False` and contribute only their single-channel features.

Threshold: a slot is "near zero" if `value < 0.01` kWh (`EPS`).

**Per meter-day, import channel (OBIS 1.29):**

| column | definition |
|---|---|
| `day_kwh` | `nansum` over 96 slots |
| `night_kwh` | sum over hours 01–04 |
| `evening_zero_share` | share of slots from `ceil(sunset)` to 23:45 with `imp < EPS` |
| `zero_run_after_sunset` | length in hours of the contiguous `imp < EPS` run starting at the first slot after sunset (0 if that slot is not near-zero) |
| `step_slot` | first slot after that run where `imp > 0.5 * night_kwh/12` (per-slot night mean); `NaN` if none |
| `step_magnitude` | `imp[step_slot]` |
| `midday_import_share` | sum over hours 11–14 ÷ `day_kwh` |
| `ramp_std_midday`, `ramp_std_morning` | std of `diff(imp)` over hours 11–15 and 07–10 |
| `night_flat_block_len`, `night_block_power`, `night_block_start_slot` | longest run over hours 22–06 where `imp` is constant within ±10%, its mean power, its start slot |
| `zm0_imp`, `zm1_imp` | packed near-zero mask |

**Per meter-day, export channel (OBIS 2.29):**

| column | definition |
|---|---|
| `exp_day_kwh` | `nansum` |
| `exp_peak_kw` | `4 * max(slot)` (kWh per quarter-hour → kW) |
| `exp_start_slot`, `exp_end_slot` | first/last slot with `exp > EPS` |
| `exp_peak_flatness` | share of slots between start and end within 5% of the daily max |
| `zm0_exp`, `zm1_exp` | packed near-zero mask |

**Joint (needs `has_both`):**

| column | definition |
|---|---|
| `both_zero_share` | popcount(`zm_imp & zm_exp`) ÷ 96 |
| `both_zero_share_daylight` | same, restricted to slots between sunrise and sunset |
| `flat_interval_share` | share of slots in a run of ≥ 4 consecutive slots where `net = imp − exp` is constant within max(0.05 kWh, 10%) |
| `flat_interval_power_mode` | modal `4 * net` over those runs, rounded to 0.25 kW |

Popcount: `np.bitwise_count` if available, else a 256-entry `uint8` lookup table.

**Cost control:** compute `zero_run_after_sunset`, `step_*`, `flat_interval_*`,
`night_flat_block_*` **only for months 3–10**; other months get `NaN` for those
and the cheap scalars for all. Guard with `if month in RUNLEN_MONTHS`.

**Aggregate to `(mp_id, year, month)`:** mean and std of each day-level column,
plus `n_days`, and a clear-sky-only mean of the dip-sensitive ones
(`evening_zero_share`, `both_zero_share`, `flat_interval_share`,
`exp_start_slot`) using `is_clear_sky` from `weather_daily.csv`. Also emit
`exp_days_positive` per `mp_id` (count of days with `exp_day_kwh > 0`) and
`exp_rows_seen` (whether the meter has any 2.29 rows at all) for §4.2.

Outputs:
- `artifacts/battery/features_monthly.pkl`
- `artifacts/battery/meter_export_days.csv` (`mp_id, exp_days_positive, exp_rows_seen`)

**Smoke test before the full run:**
`python -m models.battery.build_features --months 2025-07 2025-01`
Assert: non-empty output, `n_days` between 20 and 31 for the bulk of meters,
`evening_zero_share` in [0, 1], July's mean `exp_day_kwh` ≫ January's, and
`both_zero_share` higher in July than January. **Do not launch the full pass
until these hold.**

### 4.4 `build_features_agg.py`

1. Drop meters in `meter_flags.csv` with `is_production_meter == True`.
2. Drop `(mp_id, year, month)` rows with `n_days < 20`.
3. Aggregate month → `(mp_id, year)`: seasonal means for summer (5–8), winter
   (11,12,1,2) and shoulder, then the **contrast features**
   `<x>_summer − <x>_winter` for `evening_zero_share`, `both_zero_share`,
   `flat_interval_share`, `midday_import_share`. These contrasts matter more
   than the levels — they cancel household size and occupancy.
4. Change-point feature: Pettitt's test on the per-meter daily
   `evening_zero_share` series → `cp_year`, `cp_strength`. Implement Pettitt
   directly (~15 lines, rank-based); do not add a dependency.
5. Roll meters up to `(gp_nr, year)` with a **consumption-weighted mean**
   (weights = `day_kwh_mean` clipped at 0.1), as `models/build_features_agg.py`
   already does. Keep `n_meters`.
6. Attach `pv_probability` (out-of-fold, from the PV model), **binned to three
   levels** `<0.2 / 0.2–0.8 / >0.8` as an integer 0/1/2 — do not pass the
   continuous value. Attach `pv_rate` per PLZ, `plz`, `kanton`.
7. Attach `export_peak_kw_summer_p99` as the PV-size control.
   **Do not add annual export energy as a feature-level control** — a battery
   reduces it, so it is a mediator, not a covariate. It may appear only inside
   family-C signal features.

Output `artifacts/battery/features_gp_year.pkl`, one row per `(gp_nr, year)`.

### 4.5 `train.py`

**Training rows.** Keep `(gp_nr, year)` rows with ≥ 6 months of data at
≥ 20 valid days. Then:

| household | years marked `P` | years in `U` | years dropped |
|---|---|---|---|
| battery-positive, dated | years > install year | years < install year | the install year |
| battery-positive, undated | most recent complete year | earlier years | — |
| audit-negative (82) | — | — | **all** (held out entirely) |
| everything else in the universe | — | all | — |

**Model.** Elkan–Noto case-control PU, mirroring `models/train.py`:

1. `GroupKFold(n_splits=5)` on `gp_nr` → out-of-fold `g(x) = P(labelled | x)`
   from `HistGradientBoostingClassifier(max_depth=4, max_iter=300,
   learning_rate=0.05, l2_regularization=1.0, class_weight="balanced",
   random_state=SEED)`.
2. `c = mean(g_oof[P])`; `p = clip(g_oof / c, 0, 1)`.
3. Logit shift so `mean(p[U])` equals π. **π is a parameter, not a constant:**
   default `0.30`, and the run must repeat for `π ∈ {0.20, 0.30, 0.40}` with all
   three written to the metrics file.

**Baselines to report next to it, same folds:** (a) the single rule
`evening_zero_share_summer > τ` with τ chosen on the training folds;
(b) `LogisticRegression` on the standardised features.

**Ablations** (same protocol, one flag each): import-only features; continuous
vs binned vs absent `pv_probability`; bagging-PU instead of Elkan–Noto.

**MiniRocket branch — optional, do it last, drop it if time-boxed.** ~100 lines
of numpy over three 96-point per-household curves (median clear-sky summer net
day, median shoulder net day, day-to-day std profile) → `RidgeClassifierCV`,
stacked with the GBDT by averaging out-of-fold scores. Ship it only if it beats
the GBDT out of fold; either way record the comparison.

Outputs: `artifacts/battery/model/{model.pkl, meta.json, oof_predictions.csv}`.
`meta.json` carries `features`, `c`, `gamma`, `pi`, `n_positives`,
`n_unlabelled`, `n_households`.

### 4.6 `evaluate.py`

**Headline — audit set A** (212 PV-owning positives vs 53 PV-owning
audit-negatives, neither seen in training):

- **balanced accuracy** = `(sensitivity + specificity) / 2` at the operating
  threshold — this is the number that goes in front of an audience
- sensitivity and specificity reported separately beside it
- raw accuracy, **always printed together with its no-skill baseline of
  `212/265 = 80.0%`**; never emit the accuracy without the baseline in the same
  string
- ROC-AUC and PR-AUC, with **PR-AUC computed as
  `sklearn.metrics.average_precision_score`** — do **not** integrate a PR curve
  with `np.trapezoid`; linear interpolation in PR space is invalid and inflates
  the area. (`models/evaluate.py:43` does exactly this; do not copy it.)
- bootstrap 95% CIs over households, 2000 resamples, seeded

**Audit set B — paired retrofit test** (105 households with an in-window
commissioning): probability in the last pre-install year vs the first full
post-install year. Report paired win rate, median jump, Wilcoxon signed-rank p.

**In-loop PU metrics:** recall on held-out positives, PU-adjusted PR/ROC using
π, the Elkan–Noto `c`, population flag rate.

**Calibration:** reliability curve on audit set A, 10 bins.

**Slices:** by year, by `pv_probability` bin, by `n_meters`, by data span, by
PLZ (top 20 by count).

Outputs `artifacts/battery/metrics.json` and `artifacts/battery/figures/*.png`.

### 4.7 `score.py`

Score every `(gp_nr, year)` in the universe; household probability = the value
in its most recent year with data. Households outside the universe get
`battery_probability = NaN` and a `reason` string — never a fabricated 0.

Recompute the logit shift on this model's own predictions over `U` (the stored
`gamma` was fitted on out-of-fold scores; `models/score.py` shows the pattern).

Output `artifacts/battery/battery_predictions.csv`:
`gp_nr, plz, kanton, in_universe, reason, battery_probability, battery_pred,
pv_probability_bin, n_meters, n_years, top_evidence`.

`top_evidence` is a short plain-language string built from the z-scores of the
interpretable drivers (`evening_zero_share_summer`, `both_zero_share_summer`,
`flat_interval_share_summer`, `step_time_std`), in the style of
`models/score.py::_evidence_strings`.

---

## 5. Order of work

1. `config.py`, `solar.py` (+ the solar-noon assertion)
2. `build_labels.py` → **check the acceptance table before continuing**
3. `build_features.py` on two months → smoke assertions
4. Full streaming pass (long; run it once, in the background, and log per-file timings)
5. `define_universe.py` → **check ≥95% retention before continuing**
6. `build_features_agg.py`
7. `train.py` (GBDT first; MiniRocket only if time remains)
8. `evaluate.py`, `score.py`, `pipeline.py`
9. Write `models/battery/README.md` in the style of `models/README.md`, and a
   `REPORT.md` with results, top features, and limitations

Steps 2 and 5 are gates. If a gate fails, stop and report the discrepancy —
do not adjust the threshold to make it pass.

---

## 6. Definition of done

- `python -m models.battery.pipeline` runs end to end from a clean checkout
- `python -m models.battery.pipeline --skip-scan` reuses `features_monthly.pkl`
- both acceptance gates pass and their numbers appear in the logs
- `battery_predictions.csv` exists, one row per household, no fabricated zeros
- `metrics.json` contains balanced accuracy, sensitivity, specificity, accuracy
  **with its baseline**, ROC-AUC, PR-AUC (average precision), all with CIs, for
  each π in {0.20, 0.30, 0.40}
- the PV pipeline still runs unchanged
- `REPORT.md` states the headline number, the baselines it beats, and the
  limitations — including any gate that only just passed

---

## 7. Things that will go wrong (they already did once)

| Trap | What to do |
|---|---|
| Trusting export headers | Parse by position; `models/stream.py` handles it |
| `2023-03` missing, `2024-10` partial | Expected; multi-month aggregation absorbs it |
| Treating GIGI `-` as "no battery" for PV-style labels | Only the `Batterie/Speicher` column defines this label; `-` in *other* columns means the row is about another asset |
| Admitting known positives to the universe by label | Forbidden — it makes "no feed-in" predict `P` |
| Change-point on year-aggregated values | Failed in the PV model (Spearman 0.05). Run it on the **daily** series |
| Using in-sample `pv_probability` | Use the out-of-fold column |
| `np.trapezoid` on a PR curve | Use `average_precision_score` |
| One household straddling CV folds | `GroupKFold` on `gp_nr`, always |
| Quoting accuracy without its baseline | 80.0% on audit set A; print them together |
