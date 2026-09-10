# data_addition

Derived, ML-ready data for the HackDays load-profile work.

| File | Rows | What it is |
|---|---:|---|
| `HackDays2026 - GIGI - annotated.csv` | 1192 | Source: GIGI list + hand-added `InBetrieb-Datum bezieht sich auf` column (unmodified) |
| `GIGI - annotated - with_loaddata.csv` | 439 | Source restricted to customers that have load data, `MP IDs` / `n_meters` appended |
| `train_property_sequences.csv` | 213 722 | **Raw 15-min series** for the same 342 samples — for sequence models |
| `load_property_daily.csv` | 436 393 | Full cleaned panel, all 338 properties, unwindowed (includes March 2023) |
| `train_property_samples.csv` | 342 | **← TRAIN ON THIS.** Cross-sectional samples + augmented negatives, March 2023 excluded |
| `features_property.csv` | 336 | Earlier feature table (includes March 2023, no augmented rows) — superseded |
| `augmented_load_before_after.csv` | 19 495 | Pre-/post-installation samples (10 properties) |
| `augmentation_manifest.csv` | 20 | Index of the above |
| `meter_month_labelled.csv` | 17 808 | Per-**meter** monthly kWh (import/export, midday, night) for the 413 GIGI-labelled meters |
| `property_aux_features.csv` | 337 | Per-household features that survive meter aggregation (`n_meters`, dedicated-HP / production-only meter flags, per-meter extremes) |
| `train_property_samples_aux.csv` | 342 | `train_property_samples.csv` + the 12 aux columns — same `sample_id` / labels |

Two views of the **same 342 samples**, same `sample_id` / `group_id` / labels:

* `train_property_samples.csv` — 75 aggregated features per sample (tabular ML)
* `train_property_sequences.csv` — the underlying 15-minute series (sequence models)

`load_property_daily.csv` is the unwindowed full panel; use it only if you want
different windows than the ones chosen here.

---

## `train_property_samples.csv` — the training set

342 rows × 75 columns. March 2023 is **excluded** (its source file is an abandoned
`.tmp`, see *Concurrent modification*), costing 4 464 meter-days.

| `sample_type` | rows | labels |
|---|---:|---|
| `cross_sectional` | 335 | the property's current technology set |
| `augmented_before` | 7 | all zero — the same property before it installed anything |

**Windows are anchored to the label.** For the 249 properties with a usable
`InBetrieb-Datum`, features are computed only over days **after** the technology set
was complete, then capped to the most recent 365 days (min 30). Without this, a
property that installed PV in 2025 would have its 2023–24 pre-PV behaviour averaged
into a `has_PV=1` row, diluting the signal.

**Use `group_id` for cross-validation.** An augmented row shares its property with a
cross-sectional row; splitting them across folds leaks. Use `StratifiedGroupKFold`
with `groups=group_id` — as the baselines below do.

Feature columns are identical to `features_property.csv` (see below), plus
`sample_id`, `group_id`, `sample_type`, `window_anchored_to_install`.

### `train_property_sequences.csv` — the same samples as raw series

The feature file collapses each sample's window into **one mean day at hourly
resolution** (`hb00`–`hb23`, `he00`–`he23`); all day-to-day variation is lost. This
file keeps it: one row per **sample · day · direction**, 107 columns —
`sample_id`, `group_id`, `gp_nr`, `sample_type`, `date`, `dow` (0=Mon), `direction`,
the four labels, then `t0015`…`t0000` (96 quarter-hour kWh values, interval-end
labelled, summed across the property's meters).

213 722 rows, 133 MB, all 342 samples. Verified consistent with the feature file:
identical sample set, identical labels, and per-sample day counts that match `n_days`.
Median 364 days per sample (min 38, max 365).

To reshape for a model: group by `sample_id`, and each sample becomes a
`(n_days, 2, 96)` tensor — days × direction × quarter-hours.

### Baseline results

`HistGradientBoostingClassifier(max_iter=200)`, 5-fold grouped CV, 63 numeric features:

| target | pos | ROC-AUC | PR-AUC | base rate |
|---|---:|---:|---:|---:|
| `has_Batterie` | 227 | **0.888** | 0.916 | 0.664 |
| `has_PV` | 281 | 0.806 | 0.939 | 0.822 |
| `has_EV` | 72 | 0.708 | 0.374 | 0.211 |
| `has_Waermepumpe` | 107 | **0.565** | 0.363 | 0.313 |

Battery is the strongest target, which contradicts what the median comparison
suggested — the model reads profile *shape* (the `hb*` hourly columns), which is
exactly what a battery alters, while single-number medians miss it.

**Heat pump does not work and is unlikely to.** 0.565 is near chance, and it does not
improve on the 296 samples that have a full winter *and* summer (0.604), nor does
`winter_summer_ratio` separate the classes even among non-PV properties (3.96 vs 4.64).
A plausible explanation is that many Swiss heat pumps sit on a separate switched
tariff meter that is not in this extract. Do not spend hackathon time tuning this one
before checking whether the heat-pump load is physically present in the series at all.

---

## Preprocessing applied

**1. Schema.** 2023 files originally carried an extra unheadered `Zählpunktbezeichnung`
column (102 fields vs the header's 101, `Datum` in column 4). **The data owner
re-uploaded all twelve 2023 files on 2026-09-10 between 15:59 and 16:27 and fixed
this** — 2023 now matches 2024+ exactly at 101 fields. The extractor auto-detects
either layout, so both are handled. See *Concurrent modification* below.

**2. Missing data is missing, not zero.** Blank cells were the single biggest trap:
7.7 % of meter-days are present as rows but entirely blank, and naive parsing turns
them into `0.000` — a property drawing exactly 0 kWh for a whole day. Rule applied:

* meter-day valid only if **all 96 slots** are present;
* 42 336 fully-empty and 3 310 partial meter-days dropped (8.3 % of 550 048).

This mattered: it cut the before/after set from an apparent 19 properties to a real 10,
because several "before" windows consisted largely of blank days.

**3. Property-level aggregation.** The technology label describes a building, so a
customer's meters are summed per quarter-hour. `n_meters_present` /
`n_meters_expected` / `complete` expose whether every meter reported that day
(99.6 % complete). Meters appearing mid-series would otherwise produce a step change
indistinguishable from a technology switching on.

**4. Time base — no DST correction needed.** Verified empirically: the summer PV peak
sits at slot 12:30 and the generation centre-of-mass at 13:00 in both summer and
winter. Local wall-clock would put the summer peak at 13:30. The series is therefore
**fixed UTC+1 year-round**, every day has exactly 96 real quarter-hours, and the
clock-change days need no special handling. Slot labels are interval **end**.

**5. Known coverage gaps** — real absences, left as gaps rather than imputed:

| Gap | Effect |
|---|---|
| April 2023 | Only the 1st–23rd exist (2 754 property-days vs ~3 700 either side) |
| October 2024 | Truncated extract — 2 626 property-days vs ~9 000 either side |

Filter on `n_days` / `coverage_pct`, or drop these months, if a model is sensitive to
irregular sampling.

---

## `features_property.csv` — 336 properties, 73 columns

Features are computed over each property's **most recent 365 days** of complete data
(≥30 days required), since the label describes current state. `n_days`, `first_day`,
`last_day` record the window actually used.

| Group | Columns |
|---|---|
| Identity | `gp_nr`, `plz`, `n_days`, `first_day`, `last_day` |
| Consumption | `bezug_mean_kwh_d`, `bezug_median_kwh_d`, `bezug_p95_kwh_d`, `bezug_peak_kw`, `frac_qh_above_6kw` |
| Generation | `einsp_mean_kwh_d`, `einsp_max_kwh_d`, `frac_days_feedin` |
| Shape | `night_share` (00–06), `midday_share` (11–15), `evening_share` (17–22) |
| Seasonality | `winter_summer_ratio` (DJF/JJA, blank if <14 days either season), `weekend_weekday_ratio` |
| Hourly profile | `hb00`–`hb23` (draw), `he00`–`he23` (feed-in), mean kWh per hour |
| Labels | `has_PV`, `has_Waermepumpe`, `has_Batterie`, `has_EV`, `has_WP_Boiler` |
| Quality | `observed_feedin`, `label_conflict_PV` |

### Class balance

| classifier | pos | neg | note |
|---|---:|---:|---|
| `has_PV` | 282 | 54 | positive-heavy — accuracy is meaningless here, use PR-AUC |
| `has_Batterie` | 228 | 108 | |
| `has_Waermepumpe` | 107 | 229 | |
| `has_EV` | 72 | 264 | |
| `has_WP_Boiler` | 4 | 332 | **not trainable — drop it** |

### Do the features separate the classes?

| label | feature | pos median | neg median |
|---|---|---:|---:|
| `has_PV` | `einsp_mean_kwh_d` | 19.98 | 0.00 |
| `has_EV` | `frac_qh_above_6kw` | 0.021 | 0.003 |
| `has_Waermepumpe` | `bezug_mean_kwh_d` | 19.56 | 15.52 |
| `has_Batterie` | `midday_share` | 0.076 | 0.098 |

PV is near-separable and EV has a clear 6.6× signal. Heat pump is weak-but-real.
**Battery is barely distinguishable** — expected, since a battery mostly reshapes a
profile that PV already dominates.

⚠️ **`winter_summer_ratio` is confounded by PV**, not a clean heat-pump feature:
heat-pump owners show 6.9 vs 10.6 for non-owners — backwards — because PV owners draw
almost nothing from the grid in summer, inflating their ratio. Use it only alongside
`has_PV`/`einsp_mean_kwh_d`, or restrict to non-PV properties.

### Label noise — the important caveat

GIGI is a **subsidy register**: a blank flag means "not claimed through this
programme", not "absent". PV is the one technology directly visible in the meter, so
it can be measured:

| | properties |
|---|---:|
| PV-negative but feeding in >1 kWh/day (**unrecorded PV**) | 22 of 54 (41 %) |
| PV-positive but no feed-in at all | 21 of 282 |
| `label_conflict_PV = 1` total | 43 |

**Only 32 of the 54 PV negatives are trustworthy.** Expect the same order of
contamination in the negative class of the other three classifiers, where it cannot be
measured directly. This puts a hard ceiling on achievable recall that no model tuning
will lift — treat it as a property of the data, not a bug in your model.

---

## `load_property_daily.csv` — the cleaned panel

One row per **property · day · direction**; `;` separated, 103 columns:
`gp_nr`, `plz`, `date`, `direction` (`bezug` / `einspeisung`), `n_meters_present`,
`n_meters_expected`, `complete`, then `t0015`…`t0000` (96 values, kWh, interval-end
labelled, summed across the property's meters).

Covers 338 properties, 2023-01-01 … 2026-07-31.

---

## `augmented_load_before_after.csv` — 10 properties, 20 samples

Same property observed before and after a technology was installed:
`before` → label `none`, `after` → the technology. Restricted to customers owning
**exactly one** technology, because a single `InBetrieb-Datum` on a multi-technology
row cannot be assumed to apply to all of them at once (the annotation column is a
mechanical restatement of the row's flags — verified, 801 of 801 dated rows — so it
resolves *which row* a date belongs to, not simultaneity).

Extra columns beyond the panel: `sample_id`, `mp_ids`, `period`, `label`,
`install_date`, `days_from_install`, `baseline_clean`.

**8 of 10 have a clean baseline** — 4 PV, 4 Wärmepumpe. The two `Batterie` properties
are flagged `baseline_clean=false`: they already had PV before the battery, so `none`
is the wrong before-label.

These are the **only confirmed negatives in the whole dataset** — the only properties
observed with a technology verifiably not yet present. Too few to train on; use them as
a held-out validation set, where they test the model free of the between-property
confounds that affect everything else.

Caveats: `before` windows are short and seasonally skewed (26–1096 valid days,
`after` 34–1060). Install dates are approximate — detecting real PV onset from feed-in
gives a median offset of −4 days for Mar–Jul commissioning, but only 15 of 37 land
within ±14 days. Use `days_from_install` for a guard band.

---

## Concurrent modification — read this

The source share `/home/renku/work/store/input_data` is **live and was being written
during this work**. On 2026-09-10 all twelve 2023 files were re-uploaded between 15:59
and 16:27 (fixing the extra-column schema). Consequences:

* An extraction running at the time read those files mid-write and silently lost a
  whole month. All 2023 data here was **re-extracted afterwards** and verified
  (12/12 months, uniform 101 fields).
* **`März 2023` is still an abandoned temp file**:
  `2023/2023/März 2023/LG_AIM2Hackerdays_kWh_20260827_084034.csv.tmp.5695`
  — the upload finished at 16:27 but this rename never completed. Its contents are
  complete and valid (01.–31.03., 28 305 meters, 1 754 910 rows, correct schema) and
  were used here, but **`find`/glob patterns matching `*.csv` will silently skip
  March 2023.** Someone should finish that rename.

Re-verify file mtimes before any future extraction.

## Reproducing

Built by filtering the 43 monthly CSVs (77.3 GB) to the 414 relevant metering points in
one full pass — the files are flat text with all ~90 000 meters interleaved and no
index, so every byte must be read. The extracted subset is 300 MB.


## Per-meter auxiliary features (`build_meter_aux` → `build_property_aux`)

Aggregating a household's meters to `gp_nr` is correct — the GIGI label
describes the customer, not the meter — but it hides *which* meter carried the
load. Two cases are worth keeping:

* a **dedicated heat-pump meter** (Wärmepumpentarif): winter-only, near-zero in
  summer. 29 households have one sitting next to a normal-looking meter.
* a **dedicated PV production point**: zero import, pure export. 17 households
  have one; without it their household meter shows no PV signal at all.

Run with:

```bash
cd /home/renku/work
python -m newspaper.models.build_meter_aux    # ~16 min, one filtered 77 GB pass
python -m newspaper.models.build_property_aux  # seconds
```

### Honest status of these features

Measured against the GIGI heat-pump label they show **almost no separation**:

| feature | `has_Waermepumpe=1` | `=0` (unlabelled) |
|---|---:|---:|
| `has_dedicated_hp_meter` | 0.103 | 0.081 |
| `is_winter_dominant` | 0.607 | 0.643 |
| `meter_max_winter_summer_ratio` (median) | 6.72 | 7.87 |

That contrast **cannot establish they are useless**: `has_Waermepumpe = 0` is
*unlabelled*, not a verified negative (a `-` in GIGI means "not part of this
application"), so a near-equal rate is exactly what a PU setting produces when
many zeros are really positives. Treat the table as a floor, not an estimate.

Two known limitations:

* the aux features are **household-static** — computed over each meter's whole
  history — so they are constant across a household's `augmented_before` /
  `current` samples and add nothing to the before/after pairs. Windowing them
  to `first_day`/`last_day` is the obvious next step.
* `is_winter_dominant` fires for 214 of 337 households. On a single-meter
  household "max ratio over meters" is just that household's own seasonality,
  which is winter-heavy for most of this subsidy-applicant population. Prefer
  `has_dedicated_hp_meter` (multi-meter only) or the continuous
  `meter_max_winter_summer_ratio`.

`2024-10` is skipped (known partial month); `2023-03` was still uploading as
`.csv.tmp.5695` and is absent from this build.
