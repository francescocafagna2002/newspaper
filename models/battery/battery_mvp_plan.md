# MVP Plan — Classify "Household has a battery / home storage system" from smart-meter data

Energy Data Hackdays 2026 · AEW "Energy Fingerprints" challenge
Draft: 2026-09-10 · blueprint: [`pv_mvp_plan.md`](pv_mvp_plan.md) + [`../models/`](../models/)

> **STATUS — plan only, not built.** Feature and architecture choices are
> justified against the literature in
> [`battery_evidence_base.md`](battery_evidence_base.md). Every data problem the
> PV model already solved is inherited from
> [`data_problems.md`](data_problems.md) and not re-litigated here; this
> document covers only what is *new* or *different* for batteries.

---

## 0. Decisions taken (clarification round 1)

| # | Decision | Consequence |
|---|---|---|
| **D1** | **Model on net load, `net = OBIS 1.29 − OBIS 2.29`.** The PV model banned the feed-in register because feed-in *is* PV; no register reveals a battery, so both directions are in scope. | Unlocks feature families B (net pinned at zero) and C (feed-in shaping). An import-only ablation is still reported for comparability with the PV model. |
| **D2** | **Positive-Unlabeled learning only.** The 82 households whose GIGI rows all read `Batterie = -` are *not* used as training negatives. | Mirrors the PV model (Elkan–Noto). Those 82 households are instead reserved as a **held-out audit set** — see §2.2 and §6.2. Flagging this explicitly: your answer was about *training* negatives; using them purely to measure is the only way to get one honest ROC number, and it costs nothing. Say the word and they are dropped entirely. |
| **D3** | **Universe = PV-likely households; evaluate the contrast that matters.** 96% of battery households also have PV, so a PU model run over all 77k households would learn "has PV" and stop. Reasoning in §3a-B3. | `P` = known battery households, `U` = *other PV households*. The learned contrast is battery vs PV-without-battery. The gate itself is **measured feed-in, applied blind to labels** — not the PV model's output. See D5 and Step 0. |
| **D5** | **`pv_probability` enters as a down-weighted feature, never as the gate.** It is an uncertain output of another model and must not carry the weight of a hard filter. | Three concrete mechanisms in Step 5 (out-of-fold values, coarsened to 3 bins, dropped if the ablation says it earns nothing). The universe gate moves to a *directly measured* quantity, which is what actually removes the trap. |
| **D4** | **Binary classification only.** No activity windows, no `battery_since_year` deliverable, no capacity estimate. | Deliverable is one calibrated probability per `GP-Nr`. The `(household, year)` modelling unit stays (§4 Step 4) — it is label hygiene for mid-window retrofits, not scope creep — but only the household-level probability ships. Everything else moves to §8. |

---

## 1. Goal & scope

Given a household's 15-minute smart-meter record (2023-01 … 2026-07), output a
calibrated probability that the household operates a **behind-the-meter battery
storage system**, with interpretable evidence for the claim.

**The one thing that makes this task different from PV:** 582 of the 608
battery households in GIGI (96%) also have PV. A battery classifier trained
against the general population will simply re-learn "has PV", score 0.9+ AUC
and be useless. The task is therefore defined as **discriminating battery from
no-battery *within* PV households**, and every headline metric is reported on
that contrast (D3).

Success criterion: a reproducible pipeline
(`labels → features → PU model → scored population`) whose headline numbers are
(a) recall on held-out known positives, (b) the two independent audit results of
§6.2, and (c) an honest statement of the low-confidence band.

Out of scope for the MVP (D4): EV / heat-pump detection, capacity regression,
charge/discharge window extraction, dispatch modelling, deep sequence models.

---

## 2. What the data gives us

### 2.1 Join chain, file formats, streaming, weather, PLZ base rates

**Unchanged from the PV MVP** — reuse `models/config.py`, `models/stream.py`,
`models/io_utils.py`, `models/build_weather.py` verbatim. That means: parse by
column position (the 2023-04 header is malformed), `2023-03` is missing,
`2024-10` is partial, UTF-8 BOM on the reference tables, `MP ID` verified
stable across months, `GP-Nr → MP ID` via
`mpid_zähler_mapping → Zähler-GP → GIGI`, MeteoSwiss AG daily radiation +
clear-sky flag. See [`data_problems.md`](data_problems.md).

### 2.2 Label source — the `Batterie/Speicher` column (measured)

| | rows | households (`GP-Nr`) | of those, with meter data | role under D2 |
|---|---:|---:|---:|---|
| `Batterie/Speicher = x` | 713 | **608** | **222** | **P** (positives) |
| … also PV-positive | — | 582 | 212 | P, inside the universe |
| … battery but no PV row | — | 26 | 10 | P, flagged out-of-universe |
| every row explicitly `-` | 367 | **214** | **82** | **audit set A** — never trained on |
| … and PV-positive → the control group | — | 118 | **53** | audit set A, inside the universe |
| any blank / `.` cell involved → dropped | 111 + 1 | 56 | — | dropped |

1,192 GIGI rows, 878 unique `GP-Nr`, 575 of the 608 battery households in
canton AG. The join loss (608 → 222, 36%) is the same coverage gap the PV
model hit; it is not an ID-format bug.

Everything else in the population is **unlabelled** — a household with no GIGI
row may perfectly well own an unsubsidised battery. That is precisely the PU
setting, and it is why the base-rate anchor of §6.3 does the work that a
negative class would otherwise do.

### 2.3 Commissioning dates — label hygiene and a natural experiment (measured)

The annotated column `InBetrieb-Datum bezieht sich auf` names the asset each
date refers to (`Batterie`, `PV + Batterie`, `Wärmepumpe + PV + Batterie`, …).
**469 households have a dated battery commissioning:**

| year | ≤2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| households | 8 | 18 | 35 | 40 | 68 | 88 | 140 | 55 | 17 |

**276 batteries were commissioned inside the measurement window
(2023-04 … 2026-03); 105 of those households have meter data.**

Two uses, both inside D4's binary scope:

1. **Label hygiene.** A household whose battery went live in 2025 has two years
   of pre-install record. Under whole-record labelling those years pollute the
   positive class. Splitting into `(household, year)` rows and marking only
   post-install years as positive fixes this — and under PU the pre-install
   years simply return to the unlabelled pool, which is exactly correct.
2. **Validation.** Each of the 105 is a *within-household before/after pair* —
   same building, same occupants, same heating, battery switched on partway
   through. It is the strongest validation instrument in this dataset
   (§6.2, audit set B).

It also explains a known failure of the PV model, whose change-point feature
scored Spearman 0.05 against the commissioning date because it differenced
*year-aggregated* midday ratios. Here the equivalent feature runs a proper
change-point test on the *daily* series (§5, family F).

### 2.4 The signal — what a battery does to net load

A home battery does not consume energy; it **moves** it. On a 15-minute net
meter the recognisable effects are, in decreasing order of expected strength
(full physical argument and citations in
[`battery_evidence_base.md`](battery_evidence_base.md)):

1. **Evening zero-import plateau ending in a step.** After sunset the battery
   covers the house, so grid import sits at ≈0 for hours; when it empties, net
   import jumps in one interval from ≈0 to the full base load. PV alone cannot
   do this — a PV-only house starts importing the moment generation falls below
   load. The *step time* drifts with the season and with daily insolation.
2. **Net pinned at zero.** Simultaneous "import ≈ 0 **and** export ≈ 0" for
   long runs — the battery absorbs the whole imbalance. This is the signature
   the published unsupervised BTM-storage detectors key on (flat/constant-power
   intervals within a tolerance band). Needs D1.
3. **Delayed and flat-topped feed-in.** Surplus charges the battery first, so
   export starts later relative to solar noon, peaks lower, and is clipped flat
   while the inverter runs at constant charge power. Needs D1.
4. **Smoothed midday ramps.** Cloud transients that a PV-only meter passes
   straight through are absorbed by the battery → lower 15-min ramp variance at
   midday relative to morning.
5. **Rectangular night import blocks** in households that also charge from the
   grid (tariff arbitrage, winter top-up).
6. **Round-trip loss.** Charged energy exceeds discharged energy by ~5–20%; the
   asymmetry is a falsifiable physical check, not just a correlation.

Effects 1–4 are strong in summer and shoulder season and largely vanish in
deep winter (the battery barely cycles) — so **summer-minus-winter contrasts
are used throughout**, which also cancels household-level confounders
(house size, occupancy, heating).

### 2.5 What could produce the same signature without a battery

Empty/vacation homes (zero evening import, but also zero night base load —
controllable), ripple-controlled boilers on the night tariff (rectangular night
blocks, but at a *fixed clock time* every day, unlike a battery), DSO feed-in
limitation (flat-topped export without storage), and PV homes with a large
midday EV charge (midday net near zero). Each has a discriminating control
feature in §5, family H.

---

## 3. Open questions

### 3a. Resolved from the PV blueprint (no user input needed)

| # | Question | Answer inherited / measured |
|---|---|---|
| B1 | Prediction unit? | **Household (`GP-Nr`)** for the deliverable, `(household, year)` internally (§2.3). Meters rolled up consumption-weighted, production meters excluded via `meter_flags.csv`. |
| B2 | Is `MP ID` stable across months? | Yes — verified during the PV pass. |
| B3 | Why does the universe have to be PV households? *(the "how would a senior ML engineer answer this" question)* | Because under PU with an all-population `U`, the highest-information split available to the model is PV vs no-PV: it separates `P` from ~93% of `U` at almost no cost, and the tree will take it. Restricting `U` to PV households removes the shortcut **structurally** rather than hoping regularisation finds the subtler signal, and it makes the PU prior π meaningful (π = P(battery \| PV) ≈ 0.2–0.4, not ≈0.03). It also matches how the output will be used: nobody needs a model to tell them a house without PV probably has no battery. |
| B4 | Blank cells? | Dropped (56 households), exactly as PV dropped blank `PV` cells. Blank ≠ no battery. |
| B5 | Battery without PV? | 26 households (10 metered). A grid-charged battery with no PV exports nothing, so the label-blind gate of Step 0 will drop most of them — and that is the correct trade, because admitting them by label is the leak described there. They are counted as a **documented blind spot**, listed by `GP-Nr`, and left for the family-E stretch work; the MVP does not claim to detect batteries without PV. |
| B6 | Weather? | Reuse `weather_daily.csv` (AG mean radiation + clear-sky flag). Clear-sky days are where cycling is most regular. |
| B7 | CV scheme? | `GroupKFold` on `gp_nr` — mandatory now, because one household contributes several `(household, year)` rows. |
| B8 | Model family? | `HistGradientBoostingClassifier` inside the Elkan–Noto wrapper (no xgboost/lightgbm in the env), plus a numpy MiniRocket branch on daily profile curves. See §4 Step 5. |
| B9 | Compute? | One more full streaming pass (~77 GB). The PV pass took ~40 min; this one is heavier (per-day run-length work) → budget 1.5–2.5 h. |
| B10 | Silver labels from a register, as PV did with OBIS 2.29? | **Not possible** — no register reveals a battery. Rule-based silver labels would be circular; excluded (the PV model's silver labels inflated its AUC from 0.92 to 0.995). |

### 3b. Still open — for the user

- **Q-B1** — is the promised per-`GP-Nr` pooled-meter dataset available yet?
  It would raise the labelled cohort well above 222 and de-bias the join.
- **Q-B2** — is there any AEW-internal or cantonal register of installed
  storage? Without one, the calibration anchor is the Swissolar
  Batteriemonitor share-of-PV-with-storage, which is national and only weakly
  transferable (§6.3).
- **Q-B3** — confirm the audit-only use of the 82 `Batterie = -` households
  (D2). They are excluded from training either way; the question is whether we
  may *measure* against them.

---

## 4. MVP pipeline

New package `models/battery/`, importing `models.config`, `models.stream`,
`models.io_utils`, `models.build_weather` unchanged.

### Step 0 — `define_universe.py` (new, and it needs settling before anything else)

**The trap.** The PV model's dominant feature is `daytime_zero_summer`, and its
training set is mostly silver labels derived from *sustained summer feed-in*. A
battery suppresses feed-in and fills the daytime zeros — so PV households with
a battery are the ones the PV model is most likely to **under-score**. Gating
on `pv_probability` would therefore filter out exactly the households the
battery model exists to find, and no amount of down-weighting that feature
inside the model can undo it: a household excluded at Step 0 is never scored at
all. **Weight zero beats any small weight.** The gate and the feature are
separate problems and need separate fixes.

**The fix — gate on measurement, not on a model.** D1 already puts the feed-in
register in scope, so we do not need the PV model's *opinion* about whether a
household has PV; for any metered household we can read it off the register:

> `in_universe` ⟺ the household's meters recorded feed-in (OBIS 2.29 > 0) on
> **≥ 10 days** across the record.

A battery reduces export energy; it does not eliminate it. Only a system with
100% self-consumption across four years would register no export at all, and
that is rare enough to be treated as a documented miss rather than designed
around. This gate carries **no propagated model error**, and it is checkable.

**The rule is applied identically to labelled and unlabelled households, with
no escape clause for known positives.** This matters more than it looks. An
earlier draft of this step admitted any GIGI PV- or battery-positive household
regardless of measured feed-in — which would have made "in the universe but no
feed-in" an almost perfect predictor of membership in `P`, since the only
households admitted without feed-in would have been labelled ones. Under PU
that is a straight leak: the model would learn the *selection rule*, not the
battery. Any criterion that admits `P` and `U` on different terms does this, so
the gate has to be blind to labels.

**Acceptance criterion:** the gate must retain **≥95% of the 222 known battery
households**; expect ~100%. Report the retained fraction, and list every known
positive it drops — those are the documented cost of a label-blind gate, and
they are a better thing to report than a leak.

**The one place a `pv_probability` threshold survives:** meters whose export
register is absent altogether (as opposed to present and zero). For those the
measured rule cannot be evaluated, so **`pv_probability ≥ 0.8`** stands in.
Applied label-blindly, like the main rule. Measure this subgroup's size first —
from `feedin_meter_stats.csv` it is a count, not a guess — and if it is small,
prefer excluding it outright over mixing two admission rules.

Measured on the 222 known battery households (`oof_predictions.csv`), the
threshold barely matters, because `pv_probability` is saturated near 0 or 1:

| threshold | 0.2 | 0.5 | **0.8** | 0.9 |
|---|---:|---:|---:|---:|
| known battery households retained | 93.7% | 92.3% | **91.4%** | 90.5% |

Moving 0.2 → 0.8 costs about five households out of 222. Two things follow.
First, the worry that the PV model systematically *under-scores* battery
households does not survive contact with the data — their median
`pv_probability` is 1.000, and the theoretical concern (silver labels require
sustained summer feed-in, which a battery reduces) is offset by the fact that
its top feature is import-side and a battery pushes it the other way. Second,
and more usefully: **no threshold on `pv_probability` meets the ≥95% acceptance
criterion** — not 0.8, not even 0.2. That is the strongest argument for the
measured-feed-in gate being the main rule and this being a narrow fallback.

### Step 1 — `build_battery_labels.py`
Group GIGI by `GP-Nr`; positive = any row `Batterie/Speicher ∈ {x, X}`; drop
anything with a blank or `.`; hold the all-`-` households aside as audit set A.
Parse the battery commissioning date from `InBetrieb-Datum` where
`InBetrieb-Datum bezieht sich auf` mentions *Batterie*. Attach `MP ID`s.
Output `data_addition/battery_labels.csv`:
`gp_nr, mp_id, battery_positive, audit_negative, battery_commissioning_date,
has_pv_label, pv_probability, out_of_universe, n_gigi_rows, plz, kanton`.
Verification print must reproduce §2.2: 608 / 222 / 214 / 82 / 118 / 53 / 469 /
276 / 105.

### Step 2 — `build_battery_features.py` (one streaming pass)
Per `(mp_id, date)` the two OBIS registers are separate rows and may land in
different chunks. Keep a small **carry-over buffer** keyed by `(mp_id, date)`
holding, per channel, a 96-bit near-zero mask (12 bytes) plus ~8 scalars, and
emit a day record only once both channels have arrived (or close the meter-day
out at end of file). Memory ≈ tens of MB per month.

Restrict the read to universe meters (Step 0) — that is ~20% of rows and cuts
the pass time roughly fivefold. Compute the day-level quantities of §5, then
aggregate to `(mp_id, year, month)`. Expensive run-length/change-point work runs
only for March–October; cheap scalars for every month.
Output `artifacts/battery_features_monthly.pkl`.

### Step 3 — `build_battery_features_agg.py`
`(mp_id, month)` → `(mp_id, year)` seasonal features → consumption-weighted
rollup to `(gp_nr, year)`, dropping production meters (`meter_flags.csv`).
Attach `pv_probability`, per-PLZ `pv_rate`, `n_meters`, data-sufficiency counts.
Output `artifacts/battery_features_gp_year.pkl`.

### Step 4 — Training rows: `(household, year)` under PU
- Positive household **with** a commissioning date → years strictly after the
  install year are **P**; the install year and all earlier years return to
  **U** (correct: pre-install, the household genuinely had no battery, but we
  are not asserting it as a negative).
- Positive household **without** a date (139 of 608) → only the most recent
  complete year is **P**; earlier years go to **U**.
- Every other household in the universe → all years **U**.
- Audit set A → excluded from training entirely (D2).
- Require ≥ 6 months of data with ≥ 20 valid days in the year.

### Step 5 — `train_battery.py`
Elkan–Noto (case-control variant), identical machinery to `models/train.py`:
fit `g(x) = P(labelled | x)` under `GroupKFold(5)` on `gp_nr`, estimate the
label frequency `c` on held-out positives, `p = g/c`, then shift so the mean
over `U` matches the π of §6.3.

Base learners, cheapest first, each scored the same way:
1. **Single-rule reference** — `evening_zero_share_summer > τ`.
2. **Logistic regression** on ~15 scaled features (readable coefficients).
3. **`HistGradientBoostingClassifier`** on the full engineered table — expected
   best; justified for small-n heterogeneous tabular data.
4. **MiniRocket branch** (compact numpy implementation, ~100 lines, CPU-only):
   near-deterministic convolutional kernels over three 96-point per-household
   curves — median clear-sky summer net-load day, median shoulder-season
   net-load day, day-to-day standard-deviation profile — into a ridge
   classifier, stacked with (3) by averaging out-of-fold scores. Reported as an
   ablation; dropped if it does not beat (3).

**Down-weighting `pv_probability` (D5).** There is no per-feature weight
parameter in `HistGradientBoostingClassifier`, so "lower weight" has to be
bought with mechanisms that actually bite. Three, applied together:

1. **Out-of-fold values only.** Take `pv_probability` from the PV model's
   `artifacts/model/oof_predictions.csv`, never from a model that saw the
   household in training. Feeding an in-sample prediction of one model into
   another is the classic stacking leak and inflates everything downstream.
2. **Coarsen it to three bins** (`<0.2`, `0.2–0.8`, `>0.8`) instead of a
   continuous value. A boosted tree wins splits with resolution; removing the
   resolution is the most direct way to cap how much of the model's capacity a
   feature can absorb, and it matches the actual information content — the PV
   model's own report says its ranking is reliable and its calibration is not,
   so the third decimal is noise.
3. **Prefer measured PV-size proxies over the model output** wherever both are
   available (§5 family H), and let the boosting choose. If the model still
   leans on `pv_probability`, that is a finding about the proxies, not about the
   battery.

Then settle it empirically rather than by assertion: train with the feature, with
it binned, and without it. **If dropping it does not move the audit-set numbers
(§6.2), drop it** — an uncertain input that earns nothing is pure risk. Report
all three in the ablation table.

One thing the down-weighting must *not* be asked to fix: it does not protect
against the Step-0 trap. That is handled by the measured gate, above.

Robustness: SCAR is the assumption Elkan–Noto rests on, and it is shaky here —
subsidy applicants skew recent and larger. Re-run with bagging-PU as a check
and report whether the ranking moves.

### Step 6 — `evaluate_battery.py`
See §6. Writes `artifacts/battery_metrics.json` and figures.

### Step 7 — `score_battery.py`
Score every `(gp_nr, year)` in the universe; household probability = value in
the most recent year with data. Households outside the universe get
`battery_probability = NaN` with a reason code, not a fabricated zero.
Output `artifacts/battery_predictions.csv`:
`gp_nr, plz, battery_probability, battery_pred, pv_probability, in_universe,
top_evidence, n_meters, n_years`.

### Step 8 — Evidence pack
For a sample of high-, mid- and low-confidence households: a 7-day net-load
plot with the evening zero-plateau and discharge step marked, and the SHAP
top-3 drivers. This is evidence *for the binary claim* — the challenge scores
explainability — not the activity-window deliverable of D4. Doubles as the
error review.

---

## 5. Candidate features

Per meter-day → monthly → per `(household, year)`. `imp` = OBIS 1.29,
`exp` = OBIS 2.29, `net = imp − exp` (D1). ε = 0.01 kWh per quarter-hour.
Sunrise/sunset from a solar-geometry calculation for AG (no library needed).
Sources for every family: [`battery_evidence_base.md`](battery_evidence_base.md).

**A — Evening discharge (expected strongest)**
- `evening_zero_share` — share of quarter-hours between sunset and 23:45 with
  `imp < ε`, by season.
- `zero_run_after_sunset` — median length of the contiguous near-zero run
  starting at sunset (hours).
- `step_time_median`, `step_time_std` — clock time at which import jumps from
  <ε to > half the night base load; a battery makes it *variable* across days.
- `step_magnitude / night_base_load` — a real discharge-end step is a full
  base-load-sized jump, not a gradual ramp.
- `evening_zero_share_summer − evening_zero_share_winter` — the within-household
  seasonal contrast; the single most confounder-resistant feature here.

**B — Net pinned at zero (constant-power / flat-interval detector)**
- `both_zero_share` — share of quarter-hours with `imp < ε` **and** `exp < ε`,
  daytime and evening, by season.
- `flat_interval_share` — share of quarter-hours in a run of ≥ 4 intervals whose
  net is constant within a tolerance band (the published unsupervised
  BTM-storage criterion), plus the modal plateau power.

**C — Feed-in shaping**
- `export_start_offset` — first exporting quarter-hour minus sunrise, summer
  clear-sky days (charging delays first export).
- `export_peak_flatness` — share of the exporting window within 5% of the daily
  export maximum (inverter-clipped plateau).
- `export_share_of_daylight_energy` — export ÷ (export + midday import), a
  self-consumption proxy; storage raises self-consumption by ~13–24 pp.
- `export_end_offset` — export stops earlier when charging resumes late.

**D — Ramp smoothing & regularity**
- `ramp_std_midday / ramp_std_morning`, computed separately on clear-sky and
  cloudy days; batteries flatten the cloudy-day midday ramps.
- `profile_day_to_day_corr` — median correlation between consecutive days'
  evening profiles (battery cycling is highly repeatable).

**E — Grid charging**
- `night_flat_block_share`, `night_block_power_mode`, `night_block_time_std` —
  a ripple-controlled boiler fires at the same clock time every night (low std);
  a battery top-up does not. Also the main handle on the 26 battery-without-PV
  households (B5).

**F — Change-point (per meter, daily series)**
- Pettitt / PELT change-point on the daily `evening_zero_share` series →
  `cp_year`, `cp_strength`, `cp_delta`. As a *feature* only; the
  `battery_since_year` output stays out of scope (D4). Explicitly not the
  year-differenced aggregate that failed in the PV model.

**G — Physical energy balance (explainable evidence)**
- `daily_charge_energy` (missing export vs a clear-sky PV counterfactual) and
  `daily_discharge_energy` (missing evening import vs the household's own
  winter-evening base profile); the ratio should land in the 0.80–0.95 band of
  measured round-trip efficiencies. Out-of-band values are evidence *against* a
  battery — the falsifiable check the challenge asks for.

**H — Controls (against the §2.5 look-alikes)**
- `night_base_load` (occupancy — an empty house is not a battery),
  `day_kwh_mean`, `summer_winter_kwh_ratio`, `n_meters`, `n_months`,
  `pv_rate` per PLZ, `midday_import_share` (large midday EV charging).
- **PV size, measured — and pick the right statistic.** `export_peak_kw`, the
  99th-percentile 15-minute export power on clear summer days, is a good proxy
  for installed kWp *and is largely battery-independent*: on a clear day the
  battery is full by late morning and the array then exports at close to full
  power regardless. `export_annual_kwh` is **not** a valid control — a battery
  cuts it directly, so conditioning on it partially conditions away the very
  effect we are measuring (a mediator, not a covariate). Use the peak as the
  control; the energy belongs in family C as a *signal*.
- `pv_probability` — down-weighted per D5 / Step 5, out-of-fold and binned,
  and dropped outright if the ablation shows it earns nothing.

---

## 6. Evaluation

With no training negatives (D2) there is no ordinary ROC to report from the
training loop. Three layers instead, in increasing order of trustworthiness.

**The number that goes in front of an audience is accuracy** — specifically
balanced accuracy on audit set A, with sensitivity, specificity and the
no-skill baseline beside it (§6.2). Everything else in this section is the
technical backing for it.

### 6.1 PU metrics (in-loop)
Out-of-fold recall on held-out known positives at the operating threshold;
PU-adjusted PR/ROC curves using π (the `evaluate.py` machinery from the PV
model transfers unchanged); the Elkan–Noto `c`; population flag rate. Report
the single-rule baseline of Step 5.1 next to the model — the PV work showed a
GBDT beating a one-feature rule by ~3 AUC points, and that should be stated
plainly rather than dressed up.

### 6.2 The two audit sets (out-of-loop, never trained on)
- **Audit set A — the declined-battery controls.** 53 PV-owning households
  whose GIGI rows all read `Batterie = -`, versus the 212 PV-owning positives.
  This yields **one genuine ROC-AUC / PR-AUC**, the only such number available.
  Bootstrap 95% CIs over households — 53 is a small denominator and a bare
  point estimate would be dishonest.

  **Plus one plain, non-specialist number: balanced accuracy at the operating
  threshold**, i.e. `(sensitivity + specificity) / 2`. It is ordinary accuracy
  corrected for class imbalance, it reduces to accuracy when the classes are
  even, and 50% is chance by construction — so it can be quoted to a judge or a
  planner without a paragraph of caveats. Report sensitivity and specificity
  next to it, since the average alone hides which side the errors fall on.

  Raw accuracy is reported too, but **only ever beside its no-skill baseline**.
  On audit set A that baseline is 212/265 = **80.0%** — the score for labelling
  every household a battery owner. Any accuracy figure quoted without that 80%
  next to it is misleading, and on the full population, where prevalence is
  2–5%, plain accuracy is worthless: "no battery, always" scores 95–98%.
- **Audit set B — the paired retrofit test.** For the 105 metered households
  with an in-window battery commissioning: probability in the last pre-install
  year vs the first full post-install year, same household. Report the paired
  win rate, the median jump, and a Wilcoxon signed-rank p-value. This is immune
  to household-level confounding (size, occupancy, heating, PV size) — the
  closest thing to a controlled experiment in the dataset, and it is validation
  only, so it stays inside D4.

### 6.3 Calibration and population sanity
π = P(battery | PV) is the PU prior *and* the calibration anchor, since the
universe is PV households. Swissolar reports ~42% of newly built single-family
PV systems combined with storage, but the AG installed *stock* is dominated by
pre-2020 systems without storage → plan on **π ∈ [0.20, 0.40]**, sensitivity-test
across it, and state that ranking is trustworthy while the absolute scale is
not. Cross-check: implied population rate π × 0.12 ≈ 2–5%.

### 6.4 Robustness slices
By season, year, PV size (`pv_kwp` where known), meter count, data span, PLZ.
Ablations: net-load vs import-only (D1), engineered-only vs +MiniRocket,
Elkan–Noto vs bagging-PU, measured-feed-in gate vs `pv_probability ≥ τ` for
τ ∈ {0.1, 0.2, 0.5}, and `pv_probability` in continuous / binned / absent form
(D5).

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **The model learns "has PV", not "has battery"** (96% of battery households have PV) | D3: `U` restricted to PV households, so the shortcut is removed structurally rather than by regularisation; `pv_probability` kept as a monitored control |
| **The PV gate drops the battery households it is meant to select** — a battery suppresses the feed-in that the PV model was trained on | Gate on *measured* feed-in rather than `pv_probability` (Step 0), so no model error propagates into an irreversible exclusion; acceptance criterion ≥95% retention of the 222 known battery households, with every dropped household listed |
| **The gate itself leaks** if known positives are admitted on easier terms than the unlabelled pool — the model then learns the selection rule | Step 0 is label-blind by construction; the cost is a documented blind spot (batteries without PV, B5) rather than a hidden shortcut |
| **Error from the upstream PV model propagates into battery predictions** | `pv_probability` used out-of-fold, binned to three levels, benchmarked against measured PV-size proxies, and dropped if the ablation shows it adds nothing (D5) |
| **Conditioning on export energy would cancel the battery effect** (it is a mediator, not a covariate) | PV size is controlled by clear-day peak export power, which a battery barely changes; export *energy* is kept on the signal side (§5 C/H) |
| No training negatives at all (D2) → no honest in-loop ROC | Two out-of-loop audit sets (§6.2); every in-loop number labelled as PU-adjusted, not as accuracy |
| SCAR assumption behind Elkan–Noto is shaky (subsidy applicants skew recent and larger) | Bagging-PU cross-check; report whether the ranking moves; state the assumption in the write-up |
| Small cohort: 222 positives, 53 audit controls inside the PV subgroup | Simple models, grouped repeated CV, bootstrap CIs, no aggressive tuning; `(household, year)` rows add statistical units but **not** independent households — CV grouped by `gp_nr` so this cannot be mistaken for more data |
| Look-alike signatures (empty homes, ripple-controlled boilers, feed-in limitation, midday EV charging) | Dedicated control features (§5 H) and an explicit review of the top false positives |
| Battery installed mid-window dilutes whole-record features | `(household, year)` unit; pre-install years returned to `U`; install year dropped |
| Signature depends on the controller (self-consumption vs tariff arbitrage vs VPP) | Families A–C cover self-consumption operation, family E covers grid charging; VPP/market-driven dispatch documented as a known blind spot |
| Second 77 GB pass costs time | Universe filter cuts it ~5×; expensive features restricted to Mar–Oct; smoke-test on two months first (`--months 2025-07 2025-01`), exactly as the PV pass did |
| Change-point feature failed in the PV model | Root cause understood (year-aggregated differencing); replaced by a daily-series change-point test and validated against 105 known install dates |
| Deep winter carries almost no signal | Features are seasonal contrasts by construction; households with no summer data get a low-confidence flag, not a guess |

---

## 8. Stretch goals (explicitly out of the MVP per D4)

- **Charge/discharge activity windows** per day, and the discharge-end step
  time — answers challenge question 2; the features already compute them.
- **`battery_since_year`** from the family-F change-point, validated against
  the 105 known install dates — answers challenge question 3.
- **Usable capacity (kWh)** from median daily discharge energy; **power rating**
  from the modal flat-interval power.
- **Inverse-optimisation / source-separation disaggregation** to recover an
  actual battery power series instead of features — much stronger evidence,
  much more compute.
- **Joint asset model** — battery, EV and heat pump share the feature store; a
  multi-label head could use "this evening ramp is an EV, not a discharge step".
- **Per-PLZ irradiance** instead of one canton-AG series (carried over from PV).
