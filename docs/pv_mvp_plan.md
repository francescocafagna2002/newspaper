# MVP Plan — Classify "Household has a PV system" from smart-meter data

Energy Data Hackdays 2026 · AEW "Energy Fingerprints" challenge
Draft: 2026-09-10 · updated after clarification round 1

> **STATUS — built.** The pipeline is implemented in [`../models/pv/`](../models/pv/)
> and runs end to end (`python -m newspaper.models.pv.pipeline`). Final approach,
> results and limitations: [`../models/pv/REPORT.md`](../models/pv/REPORT.md).
> Full list of data problems and how each was handled:
> [`data_problems.md`](data_problems.md). Some sections below (e.g. "weather
> out of scope", "two models") were superseded during the clarification rounds
> — REPORT.md is authoritative.

## 1. Goal & scope

Build the **simplest end-to-end ML classifier** that, given a household's
15-minute smart-meter series, outputs a probability that the household has a
**rooftop PV system**, then apply it to the ~90k unlabeled customers.

**Resolved in clarification (see §3):**
- **Features use the consumption / grid-import channel only** (OBIS 1.29).
  The feed-in register (OBIS 2.29) is **excluded from model features** — the
  task is to detect PV from the net-load shape. The feed-in register is still
  used *offline* for **label enrichment and QC only** (§3a).
- Prediction unit is the **household (`GP-Nr`)**; a household's multiple
  metering points are combined into one decision.
- **Blank `PV` entries in GIGI are not used for training** (blank ≠ no PV).

Explicitly *out of scope* for the MVP (candidate stretch goals in §8):
EV / heat-pump / battery detection, activity-window localisation, per-season
change analysis, deep models, weather enrichment.

Success criterion for the MVP: a reproducible pipeline
(`labels → features → model → scored population`) with an honest
cross-validated score and a short evidence write-up.

## 2. What the data gives us

### 2.1 Join chain (household ↔ meter)

```
monthly export  MP ID
  → mpid_zähler_mapping.csv   (MP ID → Zählpunktbezeichnung)
  → Zähler-GP.csv             (Zählpunktbezeichnung → GPartner, Anlage)
  → HackDays2026 - GIGI.csv   (GPartner == GP-Nr → asset flags)
```

### 2.2 Label source — `HackDays2026 - GIGI.csv`

- 1,192 rows but only **878 unique `GP-Nr`** — the file is keyed by
  *subsidy application / project*, not by household. One household can have
  several rows (e.g. a heat-pump application and a separate PV application).
  (25 rows have a blank `GP-Nr` and are unusable.)
- `PV` column raw values: `x` (731), `X` (24), `-` (311), blank (126).
- Label rule (per §3): positive = household has **≥1 row** with `PV = x/X`;
  negative = **all** rows are explicit `-` with **no** blank; otherwise drop.
- The labelled set is **not a random sample** — every household in it applied
  for *some* subsidised asset. "No PV" here means "installed something else,
  didn't tick PV", i.e. a **weak / unverified negative**, and the class
  balance is heavily skewed to PV-present.

### 2.3 Coverage (measured) — reconciling the "755"

| Step | PV-positive | Negative | Note |
|---|---:|---:|---|
| GIGI rows with the label | 755 rows | 311 rows | `x`+`X` vs `-` |
| → unique households (`GP-Nr`) | **724** | 112 | after the §3 rule |
| → household is present in `Zähler-GP.csv` | **284** | 36 | the join bottleneck |
| → resolves to ≥1 `MP ID` (has meter series) | **284** | 36 | ≈ 355 meter points |
| MP IDs present in one sample month (Jan 2024) | 107 / ~355 (both classes) | meters appear intermittently |

So **755 is the count of PV-marked rows** (724 distinct households). The
number that matters for training is the **~284 PV / ~36 no-PV households that
link to actual smart-meter data** (~320 households, ~355 meters).

**Why the drop from 724 → 284?** Not an ID-format bug — `GP-Nr` and
`Zähler-GP.GPartner` share the same 6-digit numeric space and the true
overlap is 337 across all labels. Canton explains only part: 829 of 878
labelled households are canton AG (AEW territory), the other ~49 (SO/BL/ZH/
LU/BE) almost never join. But **even within AG only 41% (336/829) appear in
the 89,910-row `Zähler-GP` table** — those customers are simply absent from
the meter-data partner list (different customer sample, or meters not yet in
the export). → **Q11**: is a more complete `GP → metering-point` mapping
available? Recovering even half of the missing 440 PV households would more
than double the training set.

→ Three real constraints: **small labelled set (~320 households, badly
imbalanced ~9:1)**, **weak negatives**, and **intermittent meter presence**
per monthly file. Mitigations: aggregate features over many months, group CV
folds by household, and (pending Q10) add feed-in-derived silver labels.

### 2.4 The signal (measured on Jan 2024, a low-sun month)

Each `MP ID` has **two OBIS registers per day**:

- `1-1:1.29.0*255` — **active energy import** (consumption from grid)
- `1-1:2.29.0*255` — **active energy export** (feed-in to grid)

| Group | export > 1 kWh/month | median export | median import |
|---|---|---|---|
| No-PV label | 2 / 15 | 0.0 kWh | 1178 kWh |
| PV label | 50 / 92 | 2.6 kWh | 604 kWh |

Even in mid-winter the export register alone separates the classes well; in
summer it will be close to decisive. **Per the clarification, the model may
not use it** — it is kept only for label enrichment / QC (§3a). The model must
work from the consumption channel: PV homes import noticeably less overall and
show a midday depression that deepens in summer and scatters day-to-day with
cloud cover. That signal is real but weaker, so a low-confidence band is
expected.

### 2.5 File / format notes

- 43 monthly exports, `;`-delimited, 0.8–3.4 GB each → **stream, one file at
  a time**; never load all.
- 101 columns: `MP ID; OBIS-Code; Datum; PLZ; 00:15 … 00:00;` + trailing
  empty column. `Datum` is `DD.MM.YYYY`.
- Month folders are German (`Januar 2024`); **2023 has a doubled path**
  (`2023/2023/`).
- `GIGI` and `Zähler-GP` start with a UTF-8 BOM.
- Env available: Python 3.13, `pandas`, `numpy`, `scikit-learn`, `matplotlib`.
  No xgboost/lightgbm/duckdb — use sklearn GBDT/RF.

## 3. Open questions

### Resolved (clarification round 1)

- **Q1 — feed-in register as a feature?** → **No.** Consumption channel only.
  OBIS 2.29 is used offline for label enrichment / QC (§3a), never as a model
  input.
- **Q2 — prediction unit?** → **Household (`GP-Nr`).** A household *can* have
  several metering points: measured, **46 / 337 labelled households (~14%)
  map to >1 `Zählpunktbezeichnung`** — e.g. a separate low-tariff or
  heat-pump meter, common-area meter, or a dedicated PV-production meter.
  MVP: extract features per `MP ID`, then combine a household's meters
  (concatenate/sum the net-load, or take the PV-max over meter-level
  probabilities) into one `GP-Nr` prediction.
- **Q3 — negatives / blanks?** → **Drop GIGI rows with a blank `PV` cell from
  training** (blank ≠ "no PV"; likewise a blank `PV-Leistung in kWp` proves
  nothing). Positives = household has ≥1 row with `PV = x/X`. Candidate
  negatives = household where **every** row has an explicit `PV = -` **and no
  row is blank**. Even these `-` are only *weak* negatives (subsidy
  applicants for other assets) — enrich and sanity-check via §3a.

### Still open

- **Q4 — primary evaluation metric?** Proposed: **ROC-AUC + PR-AUC**
  headline, calibration curve, F1 at a chosen threshold for reference.
- **Q5 — feature time window?** *(user asked to revisit later)* — full 4
  years vs a representative subset. Working default until decided: **one
  summer + one winter month per year (~8 files)**.
- **Q6 — MeteoSwiss weather enrichment in MVP?** Proposed: no; use calendar
  month + `PLZ` as coarse proxies, add irradiance as a stretch goal.
- **Q7 — is `PV-Leistung in kWp` also a target (regression)?** Proposed:
  classification only for the MVP.
- **Q8 — meter-type flag?** Is there a way (via `Anlage`, OBIS mix, or a
  reference we don't have yet) to tell a household consumption meter from a
  dedicated PV-production meter? Needed so production meters don't leak into
  either features or labels. Proposed interim: exclude any `MP ID` that is
  export-dominant (its 2.29 register ≫ its 1.29 register).
- **Q9 — is `MP ID` stable across monthly exports?** Cross-month feature
  aggregation depends on it. Verify on 2–3 files before relying on it.
- **Q11 — is a more complete `GP-Nr → metering-point` mapping available?**
  Only 284 of 724 PV-labelled households (and 41% even within canton AG)
  currently link to meter data via `Zähler-GP.csv` (§2.3).

### 3a. Label enrichment via the feed-in register (offline only)

The labelled set is tiny (~300 households, ~40 usable negatives). Because the
feed-in register is a near-direct PV indicator, we can build a larger
**silver-label** set *without* putting it in the model:

- **Silver positive:** sustained daytime feed-in (2.29) across a sunny summer
  month, peak around midday.
- **Silver negative:** effectively zero feed-in across a full sunny summer
  (few false negatives — PV with 100% self-consumption is rare for a
  household).

Use silver labels to (a) expand training data for the consumption-only model,
(b) cross-check the GIGI `-` negatives, (c) estimate the population PV base
rate. This keeps the *model* net-load-only while exploiting the strong signal
we're allowed to see during development. **Open question Q10:** is using the
feed-in register for label construction acceptable, or must labels come from
GIGI only?

## 4. MVP pipeline

### Step 0 — Project skeleton (`newspaper/models/pv/`)
`config.py` (paths, month list), `build_labels.py`, `build_features.py`,
`train.py`, `score.py`, plus a `notebooks/` EDA notebook. Deterministic seeds,
everything writes to `store/` or a `models/artifacts/` dir.

### Step 1 — Labelled cohort (`build_labels.py`)
1. Parse GIGI (BOM, `;`), group by `GP-Nr`, apply the §3 label rule
   (positive = any row `PV = x/X`; negative = all rows explicit `-`, no blank;
   otherwise drop).
2. Build `GP-Nr → [MP ID]` via the join chain; keep `PLZ`, `Kanton`,
   `InBetrieb-Datum` (PV commissioning date — useful to exclude "PV installed
   *after* the measurement window" leakage).
3. *(pending Q10)* add silver labels from the feed-in register (§3a),
   tagged with a `label_source` column (`gigi` / `silver`).
4. Output `labels.parquet`: `gp_nr, mp_id, pv_label, label_source, plz,
   kanton, pv_inbetrieb_date`.

### Step 2 — Feature extraction (`build_features.py`, streaming)
For each selected monthly file:
- Read in chunks; keep only rows whose `MP ID` is in the cohort (for training)
  or all rows (for scoring the population).
- Take the **import register (OBIS 1.29)** as the daily 96-vector net-load
  signal. Read the export register (2.29) too, but only to (a) drop
  export-dominant meters (Q8) and (b) feed §3a — **never into features**.
- Accumulate per `(mp_id, month)` running aggregates (see §5), then per `mp_id`
  across months.
Write `features_train.parquet` / `features_all.parquet`.

### Step 3 — Feature table
Join features + labels. Drop meters with < N valid days (e.g. 20). Optionally
drop meters whose only data predates `pv_inbetrieb_date` (leakage / label
wrong for that period).

### Step 4 — Model (`train.py`)
- Split: **`GroupKFold` on `gp_nr`** (5 folds) — never let one household
  straddle train/test. If silver labels are used, keep folds grouped and
  report metrics on GIGI-only labels separately.
- Baselines, in order of complexity:
  1. Single-rule reference: `midday_dip_ratio < τ`.
  2. `LogisticRegression` on ~10 scaled features (interpretable coefficients).
  3. `HistGradientBoostingClassifier` (sklearn) — expected best.
- Handle imbalance with `class_weight='balanced'` and PR-curve threshold
  selection, not resampling.
- Calibrate probabilities (`CalibratedClassifierCV`, isotonic) inside CV.

### Step 5 — Evaluation
- CV ROC-AUC, PR-AUC, confusion matrix at chosen threshold.
- Permutation importance + one SHAP-style partial-dependence plot for the top
  3 features.
- Error review: list the false positives / false negatives with their
  daily-profile plots — check whether "errors" are actually GIGI label noise.
- Report the "net-load-only" model's score separately (Q1).

### Step 6 — Score the population (`score.py`)
Run Step 2 over the full population (all ~90k meters, all selected months),
predict, aggregate MP→GP, write
`pv_predictions.parquet` (`gp_nr, mp_id, pv_probability, pv_pred, top_evidence`).
Sanity checks: predicted PV rate by `Kanton` / `PLZ`, distribution of
probabilities, spot-check 10 high- and 10 low-confidence profiles.

### Step 7 — Deliverable
Short `README` / notebook: method, CV score, top features, 3–4 annotated
example profiles (clear PV, clear no-PV, ambiguous, label-noise case),
limitations, and the open questions above with whatever answers we got.

## 5. Candidate features (consumption / OBIS 1.29 only, per meter, aggregated)

**Midday-depression family (the core PV signal in net load):**
- **midday relative dip**: mean(11:00–14:00) / mean(daily) — PV
  self-consumption pulls this down, strongest on clear days
- summer midday dip minus winter midday dip (PV appears seasonally)
- variance / spread of the midday load across days within a month
  (sunny-vs-cloudy scatter is much wider with PV)
- share of daytime (09:00–16:00) quarter-hours at ~0 import
- count of days with a daytime import minimum below the night base load
  (physically implies local generation)
- "duck curve" shape: evening peak (18:00–21:00) / midday ratio, summer

**Level / seasonality:**
- mean daily import; summer/winter import ratio (PV homes drop more in summer)
- night-time base load 01:00–04:00 (control feature — household size / always-on load)
- correlation of daily midday load with day-of-year (solar-season proxy)

**Context:**
- `PLZ` / `Kanton` (target-encoded within CV), rough latitude via `PLZ`
- number of metering points for the household

*Excluded from features (used only per §3a / Q8):* anything derived from the
feed-in register (OBIS 2.29).

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Only ~320 labelled households (~284 PV / ~36 no-PV) with meter data | Grouped CV, simple models, report CI on the score, don't over-tune; pursue Q10 (silver labels) and Q11 (better mapping) |
| Negatives are unverified subsidy applicants (distribution shift vs 90k population) | Report population predicted-rate sanity checks; caveat loudly; ask for a better negative set (Q3) |
| Meter absent from many monthly files | Aggregate over many months; require min valid days; report coverage |
| Label leakage from PV commissioned *after* the measurement window | Use `InBetrieb-Datum` to filter or window features |
| Consumption-only signal is weak in winter / for high-self-consumption PV | Weight features toward summer months; report that low-confidence band honestly |
| `MP ID` re-use / meaning across exports not confirmed (Q9) | Verify ID stability across 2–3 files before trusting cross-month aggregation |
| Non-consumption meter points (PV production meters) polluting the join | Inspect `Anlage` / OBIS mix per MP; exclude meters that are export-only (Q8) |

## 7. Rough effort (single person, hackathon pace)

| Block | Time |
|---|---|
| EDA + confirm join / MP-ID stability / label rule | 0.5 day |
| `build_labels` + `build_features` (streaming, tested on 2 files) | 0.5 day |
| Feature extraction over the ~8-file window | 0.25 day (compute) |
| Train + evaluate + error review | 0.5 day |
| Score population + sanity checks + write-up | 0.5 day |
| **Total MVP** | **~2.5 days** | -- what do you mean here?

## 8. Stretch goals (after MVP)

- Clear-sky-day detection (from calendar + a simple solar-geometry model, or
  MeteoSwiss) to compute the midday-dip features only on days PV would matter.
- MeteoSwiss irradiance join by `PLZ` → physically-grounded PV features.
- Estimate `kWp` (regression) and PV azimuth/orientation from peak-export timing.
- Extend to EV / heat-pump / battery with the same feature-store scaffold.
- Activity windows: per-day "PV producing" intervals from the export series.
- Multi-year change detection: flag households whose profile shows a PV
  install partway through the record.
