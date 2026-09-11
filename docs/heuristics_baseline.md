# Rule-based heuristics — the baseline the ML models must beat

Energy Data Hackdays 2026 · AEW "Energy Fingerprints" · 2026-09-11

Transparent physical rules for **PV**, **EV** and **battery** presence, scored on the 342
labelled samples in `data_addition/train_property_sequences.csv`. Almost nothing here is
fitted: PV and battery use only physical constants (solar noon, the equinoxes, the
symmetric-solar-day null at 0.50). The two exceptions are flagged wherever they appear —
the EV ramp threshold (9 kW, selected out-of-fold) and the duration constants in the
retained EV comparison rules (chosen on this set).

Code: [`models/heuristics.py`](../models/heuristics.py) ·
Run: `python -m newspaper.models.heuristics` (~45 s) ·
Output: `models/artifacts/heuristics/{heuristic_scores,heuristic_metrics,ev_threshold_sweep}.csv`, `checks.json` ·
Confusion matrices for all three rules: *Confusion matrices* below

**Evaluation set:** 342 samples / 335 households. PV base rate **0.822** (281 pos),
EV **0.211** (72 pos), battery **0.664** (227 pos) overall but **0.772** within PV
households, which is the only denominator a battery rule can honestly be scored against
(see *The trap that shapes this whole section*). Windows are label-anchored (see
`data_addition/Readme.md`); median 364 days per sample.

**The bar to beat** — `HistGradientBoostingClassifier(max_iter=200)`, 5-fold
`StratifiedGroupKFold` on 63 numeric features:

| target | ML ROC-AUC | ML PR-AUC |
|---|---:|---:|
| `has_PV` | 0.806 | 0.939 |
| `has_EV` | 0.708 | 0.374 |
| `has_Batterie` | *not yet trained* | — |

No ML battery baseline exists yet, so BAT-A's 0.855 is currently unopposed. When one is
trained it must be scored **within the PV households**, or the comparison is meaningless.

---

## Headline result

| Rule | Target | ROC-AUC | PR-AUC | Precision | Recall | F1 | MCC | vs ML |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **PV-A** meter exported at all | PV | 0.772 | 0.911 | 0.918 | 0.922 | **0.920** | **0.548** | −0.034 AUC |
| **PV-B** midday/night draw ratio | PV | **0.800** | **0.936** | — | — | — | — | −0.006 AUC |
| **EV** max 15-min ramp ≥ 9 kW | EV | 0.686 | 0.319 | 0.395 | 0.681 | **0.500** | **0.342** | −0.022 AUC |
| EV, same ramp at the spec's 5.8 kW | EV | 0.686 | 0.319 | 0.268 | 0.847 | 0.407 | 0.198 | −0.022 AUC |
| *cmp:* all three spec classes | EV | 0.653 | 0.297 | 0.295 | 0.861 | 0.440 | 0.262 | −0.055 AUC |
| *ref:* peak net power ≥ 6 kW (one number) | EV | 0.689 | 0.314 | 0.231 | 0.931 | 0.370 | 0.119 | −0.019 AUC |
| **BAT-A** export after solar noon ≥ 0.50 | Battery | **0.855** | 0.945 | 0.890 | 0.895 | **0.892** | **0.494** | — |
| **BAT-B** evening/night grid draw | Battery | 0.703 | 0.895 | — | — | — | — | — |
| *trap:* `has_PV` used as a battery predictor | Battery | 0.700 | 0.767 | 0.772 | 0.956 | 0.854 | 0.493 | — |

Operating points: PV-A needs none (it is already binary); PV-B is reported
threshold-free; the EV threshold was selected out-of-fold (9.2 kW in ~100/100
`StratifiedGroupKFold` folds) and moves the operating point only — AUC is 0.686 at every
threshold, since it is the same underlying score.

**Battery rules are scored within the PV households only** (n≈259, 77 % battery). 96 % of
batteries sit in a PV house, so `has_PV` alone scores AUC 0.700 on battery — a battery
rule measured on the full set is indistinguishable from a PV detector. Inside the PV
subset `has_PV` is worth 0.500, so BAT-A's 0.855 is genuinely about storage.

**EV precision of 0.40 is a floor, not a measurement** — and roughly three quarters of
≥9 kW ramp days fall in Nov–Mar for labelled EV owners *and* non-owners alike, so the
rule is substantially reading winter heating. See *Precision is 0.40 and that is the real
result*. Quote AUC and recall for EV; do not quote precision without that caveat.

Both PV rules sit at the label-noise ceiling (max attainable AUC ≈ 0.78 against
GIGI as delivered — see *We do not correct the labels*); their differences are inside
the bootstrap CI and should not be used to pick between them.

**Both ML models are essentially matched by one-line rules.** The GBDT adds nothing on
PV, and on EV it sits within noise of "how big is this house's largest power step".
That is the benchmark.

---

## Confusion matrices

One rule per technology, at the operating point named in each section. Rows are the GIGI
label as delivered, columns the rule's call.

**PV — `any(export > 0)`**, all 342 samples, threshold-free by construction.

| | flagged PV | flagged no PV | |
|---|---:|---:|---|
| **has_PV = 1** | **TP 259** | FN 22 | recall 0.922 |
| **has_PV = 0** | FP 23 | **TN 38** | specificity 0.623 |
| | precision 0.918 | | **F1 0.920 · MCC 0.548 · AUC 0.772** |

**EV — `max 15-min ramp ≥ 9 kW`**, all 342 samples.

| | flagged EV | flagged no EV | |
|---|---:|---:|---|
| **has_EV = 1** | **TP 49** | FN 23 | recall 0.681 |
| **has_EV = 0** | FP 75 | **TN 195** | specificity 0.722 |
| | precision 0.395 | | **F1 0.500 · MCC 0.342 · AUC 0.686** |

**Battery — `export after solar noon ≥ 0.50`**, 242 PV households with shoulder-season
export (the 61 non-PV samples and 39 PV households that never export in Mar/Apr/Sep/Oct
are outside the rule's domain, not scored as negatives).

| | flagged battery | flagged none | |
|---|---:|---:|---|
| **has_Batterie = 1** | **TP 170** | FN 20 | recall 0.895 |
| **has_Batterie = 0** | FP 21 | **TN 31** | specificity 0.596 |
| | precision 0.890 | | **F1 0.892 · MCC 0.494 · AUC 0.855** |

### Reading these

The three matrices are **not comparable to each other** — the base rates differ wildly
(PV 82 %, battery 77 % within PV, EV 21 %) and each denominator is a different
population. Three specific cautions:

- **PV's 23 false positives are mostly real PV.** They export a median 17.0 kWh/day
  against 19.6 for labelled PV households, while true negatives export *exactly* 0.0.
  The specificity of 0.623 is a label artifact, not a rule failure — see *Why AUC is 0.77
  when F1 is 0.92*.
- **EV's 75 false positives cannot be explained away.** Roughly three quarters of the
  ≥9 kW ramp days fall in Nov–Mar for labelled EV owners *and* non-owners alike, so a
  large share of them are winter heating rather than unregistered EVs. Precision 0.395 is
  a floor of unknown tightness — see *Precision is 0.40 and that is the real result*.
- **Battery's matrix is conditional on PV** and says nothing about the 10 batteries in
  non-PV households, which the rule cannot see at all.

Every FP column is inflated and every TN column deflated by the Positive–Unlabeled
setting: a `0` in GIGI means "no subsidy claimed", never "verified absent"
(`docs/data_problems.md` ③). Recall and AUC are the trustworthy columns; precision,
specificity and MCC are lower bounds throughout.

---

## PV

### The rules — both free of invented constants

**PV-A (primary).** *The meter exported at all.*

```
has_PV  <=  any quarter-hour in the window with export > 0
```

No threshold, no deadband, no tuning. It is a physical identity: a household cannot
inject what it does not generate. Import and export are separate OBIS registers here
(`1.29` / `2.29`), so "net load went negative" is read directly as `einspeisung > 0`.

> An earlier version carried a 0.2 kW noise deadband. It was invented, and measurement
> shows it does nothing: **0 of 342 samples change classification** when it is removed.
> Deleted. Households without PV export *exactly* zero, always — there is no noise floor
> to guard against.

**PV-B (net-load-only comparison).** *Midday grid draw relative to night grid draw,
over the sunlit half-year.*

```
pv_midday_night_ratio  =  mean import 10:00-15:00  /  mean import 00:15-06:00,
                          over March-October
```

Lower means more PV. The three windows are astronomical rather than tuned: March–October
is equinox to equinox, 10:00–15:00 is solar noon (13:00 in this fixed-UTC+1 series)
± 2.5 h, and 00:15–06:00 is the repo's existing `night_share` base-load window. A 36-way
sweep over plausible alternatives moves the AUC by at most 0.02; the midday window is
worth ± 0.005.

**PV-B is reported threshold-free** — ROC-AUC and PR-AUC need no operating point. There
is no honest cut available: parity (ratio < 1) is *not* a separating point, because
**38 of 61 non-PV households also draw less at midday than at night** — the house is
empty during the day. Any cut would be an invented number, so none is offered.

### Results (vs GIGI as delivered)

| Rule | AUC | PR-AUC | P | R | Spec | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **PV-A** meter exported at all | 0.772 | 0.911 | 0.918 | 0.922 | 0.623 | **0.920** | **0.548** |
| **PV-B** midday/night ratio | **0.800** | **0.936** | — | — | — | — | — |
| *ML baseline* (GBDT, 63 features) | 0.806 | 0.939 | — | — | — | — | — |

PV-A confusion: TP 259 · FP 23 · FN 22 · TN 38.

PV-B is the honest comparison — it never reads the export register, which
`data_problems.md` ④ quarantines, and it lands within 0.006 AUC of the GBDT while
being one ratio of two averages.

### Why AUC is 0.77 when F1 is 0.92

Two effects, neither about the physics.

**① For a binary rule, ROC-AUC *is* balanced accuracy.** A 0/1 output carries no ranking
information, so AUC collapses to `(sensitivity + specificity)/2` = `(0.922 + 0.623)/2`
= **0.772**, exactly the reported value. With 281 positives and only **61 negatives**,
each negative carries 4.6× the weight of each positive: the 23 false positives cost
**0.189 AUC on their own**, the 22 false negatives 0.039. F1 (0.920) and PR-AUC (0.911)
have no such property — that is the whole disagreement between the metrics.

**② The 23 "false positives" have PV.** GIGI is a subsidy register; a blank means "not
claimed through this programme", not "no panels".

| | 23 FP (label: no PV) | 281 labelled PV | 38 true negatives |
|---|---:|---:|---:|
| export, median kWh/day | **17.0** | 19.6 | — |
| export, max kWh/day | 165.9 | — | **0.0** |
| days with export, median | 73 % | — | 0 % |

The distribution is **bimodal with nothing in between**: a household either exports like
a PV owner or exports exactly zero. 22 of 23 export > 1 kWh/day and 10 of 23 export more
than the median labelled PV household.

**The 22 misses are mostly not the rule's fault either.** 17 of 22 show no PV signature
on the independent midday signal either — the PV is not visible at any metering point we
hold. They carry more meters than the hits (1.55 vs 1.20 mean; 27 % multi-meter vs 11 %)
and **none has a linked production-only meter** (vs 4.6 % of hits), the signature of the
join-chain coverage gap: only 337 of 878 labelled households resolve to meter data at all
(`data_problems.md` ②). A further 11 of 22 have no `InBetrieb-Datum`, so their window
could not be anchored to the install.

### We do not correct the labels

The labels are the customer's record; rewriting them to agree with our own detector
would make every downstream metric a measurement of our assumptions. **All numbers here
are against GIGI as delivered.**

The consequence is a hard ceiling, to be read rather than optimised against:

- **23 of 61 negatives cannot be predicted correctly** by any honest detector, because
  they have PV → maximum attainable specificity ≈ 0.62, maximum binary AUC ≈ 0.78.
- **~20 of 281 positives are invisible** on both channels → maximum recall ≈ 0.93.

A rule scoring 0.77–0.80 is at the ceiling. **Do not tune against this metric** —
improvements past ~0.81 mean a model is learning which households GIGI failed to record,
not which have panels. Rank PV models on the 7 verified before/after negatives, or on
PR-AUC and recall over positives.

### Rejected: blending A and B

A vote-count blend of the two channels was built and measured. It is **not** in the
final rule set. Paired bootstrap (2 000 resamples) of every blend against the better
single rule put **every 95 % CI across zero** — `mean` −0.005 [−0.042, +0.032],
`rank-avg` +0.007 [−0.019, +0.033], `max` −0.013 [−0.045, +0.015], `A OR B` −0.025
[−0.072, +0.014]. The single-rule CI is itself [0.740, 0.870]; with 61 negatives nothing
in this range is resolvable. Since the blend also required an invented vote threshold, it
buys an unmeasurable gain at the cost of a tuned constant. Dropped.

### The one non-circular check

The 7 `augmented_before` samples are the only **verified negatives** in the dataset —
the same properties observed before they installed anything.

> **PV-A flags 0 of 7.**

## EV

### The rule

```
ev_score = max over the record of ( P[t] − P[t−1] )      on the import channel
flag EV  ⟸  ev_score ≥ 9 kW
```

The largest step the household's import takes in one quarter-hour. Diffs are taken
*within* a day so no step is manufactured across a gap or a midnight boundary, and the
series is `clip(net, 0)` so a PV afternoon cannot masquerade as a load (finding ② below).

**Why a bare ramp.** The spec's three-class plateau detector — switch-on edge,
pre-step baseline, sustained-duration test, power band — scores **AUC 0.688**. Raw
`max(P)`, one number with no detector at all, scores **0.689**. Paired bootstrap
ΔAUC = **−0.002 [−0.033, +0.029]**. At 15-minute resolution the plateau, the band and the
matched switch-off are not recoverable; the switch-on step is the only part of the
charger signature that survives averaging. So the detector is kept in the module
(`detect_events`) purely to reproduce this comparison, and is not the rule.

**Where 9 kW comes from.** Not the spec — the spec's Large-L2 minimum is 5.8 kW, which is
also reported below as the unfitted variant. 9 kW was selected by sweep, then checked
out-of-fold because selecting a sweep argmax is exactly the failure mode this document
objects to elsewhere. 5 × 20 `StratifiedGroupKFold`, threshold fitted on train folds only:

| | |
|---|---|
| threshold picked on train folds | **9.2 kW in ~100 of 100 folds** (IQR [9.2, 9.2]) |
| out-of-fold MCC, tuned threshold | 0.344 ± 0.091 |
| out-of-fold MCC, fixed 5.8 kW | 0.199 ± 0.089 |
| paired difference, 9.0 vs 5.8 | **+0.145 [+0.008, +0.288]** |

The optimum is stable under resampling and grouping. Note this moves the **operating
point only** — AUC is 0.686 at every threshold, because it is the same score.

### Results

| Rule | AUC | PR-AUC | P | R | F1 | MCC | flag rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| **max 15-min ramp ≥ 9 kW** | 0.686 | 0.319 | 0.395 | 0.681 | **0.500** | **0.342** | 0.363 |
| max 15-min ramp ≥ 5.8 kW *(spec, unfitted)* | 0.686 | 0.319 | 0.268 | 0.847 | 0.407 | 0.198 | 0.667 |
| share of days with a ≥ 9 kW ramp *(robust form)* | 0.688 | 0.298 | — | — | — | — | — |
| *cmp:* L1 (1.4–1.8 kW, ≥6 h) | 0.532 | 0.215 | 0.182 | 0.028 | 0.048 | −0.013 | 0.032 |
| *cmp:* Small L2 (3.6–4.8 kW, ≥1 h) | 0.560 | 0.222 | 0.241 | 0.444 | 0.312 | 0.059 | 0.389 |
| *cmp:* Large L2 (≥5.8 kW, ≥30 min) | 0.688 | 0.325 | 0.331 | 0.639 | 0.436 | 0.244 | 0.406 |
| *cmp:* all three classes | 0.653 | 0.297 | 0.295 | 0.861 | 0.440 | 0.262 | 0.614 |
| *cmp:* Large L2 ≥2 h *(duration tuned on this set)* | 0.727 | 0.331 | 0.420 | 0.472 | 0.444 | 0.286 | 0.237 |
| *ref:* peak power ≥ 6 kW | 0.689 | 0.314 | 0.231 | 0.931 | 0.370 | 0.119 | 0.848 |

`max(ΔP)` also has the cleanest PV hygiene of the candidates: AUC **0.519** against
`has_PV`, against 0.583 for a daily-peak-spread variant and 0.520 for the Large-L2
detector. It is close to the least PV-contaminated EV score available.

Rejected on the "no invented constants" rule: `p95(daily peak) − median(daily peak)`
scores 0.715 but the 95th percentile is a chosen quantile and the gain is inside the CI.

### Precision is 0.40 and that is the real result

At a 21 % base rate, flagging 36 % of households at random would give 0.21. The rule
gives 0.40 — a 1.9× lift, which is a weak detector by any standard. It matters what the
75 false positives are, and the answer is uncomfortable.

**They look more like EV households than the true positives do:**

| | FP (n=75) | TP (n=49) |
|---|---:|---:|
| days carrying a ≥9 kW ramp (median) | 13 | 7 |
| median gap between those days | 5.8 d | 12 d |
| excess energy on those days vs a normal day | 40.7 kWh | 47.0 kWh |

They ramp more often, more regularly, and nearly as hard. Only 3 of 75 have none of
PV / heat pump / battery, and the co-technology rates are flat across TP/FP/TN — battery
ownership is actually *lower* among the FPs (0.60) than among true negatives (0.65), so
battery cycling does not explain them.

That pattern is consistent with label noise, and GIGI supplies the motive: the column is
`Ladestation für Elektrofahrzeuge`, a **charging-station subsidy**. A household that owns
an EV and charges from an ordinary socket, or paid for its own wallbox, is a labelled zero.

**But the seasonality test argues against calling it label noise.** EV charging should be
roughly uniform across the year. It is not:

| | share of ≥9 kW ramp days in Nov–Mar | share of all days |
|---|---:|---:|
| labelled EV = 0 | 0.75 | 0.41 |
| labelled EV = 1 | 0.80 | 0.41 |

Both groups concentrate their big ramps in the heating season, by the same margin
(+0.32 vs +0.35), at the same time of day (median ≈ 17:00). A large share of what this
rule fires on is **winter heating**, in labelled-EV and non-EV households alike. The rule
is not finding EVs; it is finding *large intermittent winter loads*, and EV ownership
correlates with the sort of electrified household that has them.

The obvious fix fails: restricting to May–Sep (no space heating) is significantly
**worse** — AUC 0.618 vs 0.686, ΔAUC −0.068 [−0.122, −0.016]. That test is confounded
(a `max()` over 5 months is mechanically smaller than over 12), so it rules out the fix
without settling the cause.

**Conclusion: the 75 false positives cannot be apportioned between unregistered EVs and
heating loads with this data.** For PV the equivalent question was answerable — exports
are bimodal and true negatives sit at exactly 0.0. Here the FP and TP distributions
overlap completely. Therefore: **quote AUC and recall for EV; treat precision and MCC as
a floor of unknown tightness.** And discount 9 kW accordingly — part of its advantage over
5.8 kW is plausibly that it selects "large electrified house", not "has EV". Out-of-fold
selection confirms the threshold is stable; it cannot confirm what the threshold is
detecting.

### Findings retained from the spec detector

**① The L1 band does not work, and cannot.** AUC 0.532 — chance. Median detection rate is
identical for EV owners and non-owners (1.10 vs 1.11 per 100 days at 6 h). 1.4–1.8 kW is
exactly where ordinary Swiss household loads live: boilers, dishwashers, heat-pump
compressors, dehumidifiers. The combined three-class rule scores *below* Large L2 alone
(0.653 vs 0.688) because L1 and Small L2 add 55 false positives and 4 true ones.

**② Run on the import channel, not raw net load.** Initially the detector ran on net
load. The L1 rule then scored **AUC 0.703 against `has_PV`** and 0.493 against `has_EV` —
it had become a PV detector. On a PV day the afternoon net load climbs from ≈ −3 kW back
toward 0 as the sun sets: a sustained, in-band, positive step lasting hours. Clipping at
0 removes it.

| detector input | AUC vs `has_EV` | AUC vs `has_PV` |
|---|---:|---:|
| raw net load | 0.676 | 0.579 |
| import channel | **0.727** | 0.499 |

The cost is real but small: EV charging under midday PV cover shows as reduced export,
not import, and is invisible to the clipped detector. Splitting by PV ownership,
Large L2 ≥1 h still scores 0.680 on PV households vs 0.630 on non-PV.

**③ Matching the switch-off buys nothing.** Requiring a ≥5.8 kW down-step later on the
same day as the up-step: 0.677, against 0.677 for the up-step alone. Same pattern as the
plateau and band tests — at this resolution the cheapest form of the idea is the right one.

### Sanity check on verified negatives

The ramp rule at 9 kW flags **2 of the 7** `augmented_before` samples — windows recorded
*before* the household's installation, the only verified negatives in the dataset. PV-A
flags 0 of 7. Small numbers, but the EV rule is the one that leaks, and it leaks more
than the spec detector did (1 of 7).

---

## Battery

### The trap that shapes this whole section

Batteries are almost never installed without PV: **96 % of labelled batteries are in a PV
household**, and 77 % of PV households have one.

| | battery = 0 | battery = 1 |
|---|---:|---:|
| PV = 0 | 51 | 10 |
| PV = 1 | 64 | 217 |

The consequence is that **`has_PV`, used directly as a battery predictor, scores AUC
0.700** — and our PV rules score ~0.80 on PV. A battery "detector" that is really a PV
detector will therefore post a respectable-looking number while detecting nothing.

**Every battery rule below is scored within the PV households** — 281 of them, of which
242 have shoulder-season export and enter the BAT-A evaluation (78.5 % battery). Inside that subset `has_PV` is constant and worth exactly AUC
0.500, so any lift is genuinely about storage. Battery rules scored on the full set are
not reported, because they cannot be interpreted.

### The rule

```
bat_score = ( export falling after solar noon ) / ( total export )   over Mar, Apr, Sep, Oct
flag battery  ⟸  bat_score ≥ 0.50
```

**Mechanism.** A PV house *without* storage exports the instant generation exceeds load,
so its export profile inherits the symmetry of the solar day — half before solar noon,
half after. A battery absorbs the morning surplus and only lets export resume once it is
full, which pushes export into the afternoon. The rule measures that asymmetry.

Neither number is fitted. The split point is solar noon (13:00 in this fixed-UTC+1
series, the same one PV-B is centred on) and the decision threshold is **0.50, the
symmetric-solar-day null** — not a cut chosen to separate these labels. It lands between
the two populations rather than inside either:

| | median | IQR |
|---|---:|---:|
| battery = 0 | 0.466 | [0.430, 0.522] |
| battery = 1 | 0.561 | [0.506, 0.616] |

### Results (within PV households)

| Rule | AUC | PR-AUC | P | R | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| **BAT-A** export after noon, shoulder months, ≥ 0.50 | **0.855** | 0.945 | 0.890 | 0.895 | **0.892** | **0.494** |
| BAT-A′ same rule over all of Mar–Oct | 0.803 | 0.920 | 0.891 | 0.780 | 0.832 | 0.410 |
| BAT-B evening/night grid draw *(net-load only)* | 0.703 | 0.895 | — | — | — | — |
| *trap:* `has_PV` as the predictor, whole set | 0.700 | 0.767 | 0.772 | 0.956 | 0.854 | 0.493 |

Bootstrap CIs (2000 resamples, within PV): BAT-A **[0.794, 0.909]**, BAT-A′ [0.731,
0.868], BAT-B [0.670, 0.799]. The threshold is not fitted, so out-of-fold confirmation is
only a check on sampling, and it passes: 5 × 20 `StratifiedGroupKFold`, MCC **0.499 ±
0.125** for BAT-A and 0.408 ± 0.122 for BAT-A′ — matching the in-sample values.

### Why shoulder months, and why that is evidence rather than tuning

The obvious competing explanation for afternoon-weighted export is **panel orientation**:
a west-facing array exports in the afternoon with no battery involved. Azimuth is not in
the data, so it cannot be controlled for directly. But storage and orientation make
*different* seasonal predictions, and this one was stated before it was measured:

> A battery has finite capacity. In high summer it fills early in the morning and export
> resumes sooner, so the asymmetry should **shrink back toward 0.50**. In the shoulder
> months it takes most of the morning to fill, so the asymmetry should be **largest**.
> A fixed panel azimuth cannot produce a seasonal swing of that sign.

| window | battery = 0 | battery = 1 | gap | AUC |
|---|---:|---:|---:|---:|
| Jun–Aug | 0.459 | 0.529 | 0.069 | 0.751 |
| Mar/Apr/Sep/Oct | 0.483 | **0.632** | **0.149** | **0.855** |
| Mar–Oct | 0.466 | 0.562 | 0.096 | 0.803 |

Confirmed. The battery group swings **0.103** across the seasons; the non-battery group
swings 0.024 and stays pinned near the symmetric null, which is what a fixed array does.
That asymmetry-that-varies-with-state-of-charge is a storage signature, and it is the
main reason to believe this rule reads batteries rather than roofs.

Restricting the rule to the shoulder months follows from that mechanism, and the gain is
real — ΔAUC **+0.051 [+0.025, +0.079]**, CI clear of zero. It is nonetheless a window
chosen *after* seeing the seasonal split, so BAT-A′ (all of Mar–Oct, no selection at all)
is reported alongside as the conservative number.

Rejected as invented: the share of export before **10:00** specifically scores 0.854 —
as good as BAT-A — but 10:00 is a window boundary with nothing behind it. The sweep is
flat (0.838–0.854 across plausible ends), so nothing is lost by refusing it.

### BAT-B, the net-load-only fallback

```
share of evening (17:00–23:00) grid draw relative to night (00:15–06:00) draw, Mar–Oct
```

The battery's other half: it discharges into the evening peak, so the evening looks more
like the night. AUC **0.703**, never touching the export channel — the counterpart to
PV-B, and the rule to use where only import is available. Reported threshold-free: there
is no physical null here the way 0.50 is one for BAT-A, and any cut would be invented.

### Limitations specific to battery

- **10 of 227 labelled batteries are in non-PV households** and are invisible to BAT-A by
  construction — no export means no export profile. They are excluded, not scored zero.
- 39 of the 281 PV households never export in the shoulder months and drop out of the
  BAT-A evaluation (281 → 242); for BAT-A′ over Mar–Oct the loss is 22 (281 → 259).
- BAT-A reads the export channel, so like PV-A it is an offline/evaluation rule under the
  `docs/data_problems.md` ④ quarantine. BAT-B is the net-load-only counterpart.
- `has_Batterie = 0` is unlabelled, not verified (block ③), so precision here is a floor
  in the same way every other precision number in this document is.

---

## What this means for the ML models

1. **PV is solved by a rule.** 0.800 AUC / 0.920 F1 from rules with no tuned constants,
   matching the GBDT. Spending model capacity on PV presence is not where the value is;
   PV **onset dating** and **capacity estimation** are open, and the existing
   change-point feature is measurably broken (Spearman 0.05 vs commissioning date,
   `models/REPORT.md`).
2. **EV is not solved by anyone.** The ramp rule (0.686), the GBDT (0.708), the spec
   detector (0.688) and "what is this house's peak power" (0.689) are all within noise of
   each other on 72 positives. Nothing currently reads *charging behaviour* — they all
   read *this house draws a lot of power sometimes, mostly in winter*. An ML model that
   only reaches ~0.70 has learned that trivial feature and should not be reported as an
   EV detector. The concrete test: check whether its positives concentrate in Nov–Mar.
   If they do, it is a heating detector.
3. **The label ceiling is real and asymmetric.** For PV it is measurable (41 % of
   negatives are wrong) and caps binary AUC at ~0.78. For EV it is *not* measurable —
   `has_EV = 0` means "no charger subsidy claimed", and Swiss EV owners frequently charge
   on an ordinary socket with no subsidy. Every EV precision number here is a **lower
   bound**. But unlike PV, this cannot be used to explain the false positives away: the
   seasonality evidence says a large share of them are heating, not unregistered EVs.
4. **Battery is the surprise: 0.855, the strongest rule in this document after PV.**
   It works because storage leaves a *shape* signature, not a magnitude one — export
   displaced past solar noon — and shape survives 15-minute averaging where EV's
   switch-on step does not. The seasonal test (asymmetry largest in the shoulder months,
   smallest in high summer, absent in non-battery houses) is the closest thing to a
   controlled experiment anywhere in this document. Any battery model should be compared
   against 0.855 *within PV households*, never against a whole-set number.
5. **Next rule worth trying:** a genuine session detector — cluster the excess powers
   *within* a household and require ≥ K events within ±10 % of a common level, plus
   near-constant power across the session (EV charging is flat; ovens and boilers
   cycle). A first pass at power-consistency alone did not help (0.672 vs 0.673, Large L2
   ≥1 h on raw net load), but it was applied without the flatness constraint.

## Limitations

- **342 samples, 72 EV positives.** Differences below ~0.05 AUC are not resolvable.
- The EV threshold (9 kW) is empirical, not from the spec; it was selected out-of-fold,
  but out-of-fold selection cannot establish *what* it detects, and the seasonality
  evidence suggests part of it is heating. The duration constants in the retained
  comparison rules are tuned on the evaluation set and are marked as such.
- `augmented_before` samples share a household with a cross-sectional sample; the rules
  are stateless so this cannot leak, but any model compared against these numbers must
  group-split on `group_id`.
- Short windows are noisier: 51 samples have < 180 valid days.
- March 2023 is excluded from the source panel; April 2023 and October 2024 are partial
  (`docs/data_problems.md` ①).
