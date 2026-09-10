# Plan — Detect a Wärmepumpenboiler from 15-minute smart-meter data

Energy Data Hackdays 2026 · AEW "Energy Fingerprints" · drafted 2026-09-10
after a `$grill-me` round. Companion to [`pv_mvp_plan.md`](pv_mvp_plan.md);
data caveats it shares are in [`data_problems.md`](data_problems.md).

> **Status.** Detector logic is implemented and unit-tested
> ([`../models/wpb_core.py`](../models/wpb_core.py),
> [`../tests/test_wpb_core.py`](../tests/test_wpb_core.py)) and the pipeline
> runs end to end on a synthetic export
> ([`../models/wpb.py`](../models/wpb.py)). **Nothing here has touched the
> real 77 GB exports** — they are not present in the environment this was
> written in. Every number below is a design target, not a measurement.

## 1. What we are detecting, and why it is not the heat pump

A *Wärmepumpenboiler* is a heat-pump **water heater**: a small compressor
heating a 200–300 L domestic-hot-water tank. It is not a space-heating heat
pump, and conflating the two is the main way this analysis can go wrong.

| | resistance Elektroboiler | **WPB** | space-heating HP |
|---|---|---|---|
| power | 2–4 kW | **0.3–1.0 kW** | 1–5 kW, modulating |
| duration | ~1–2 h | **3–8 h** | many cycles/day |
| daily energy | 6–8 kWh | **1.5–2.5 kWh** | strongly seasonal |
| seasonality | flat | **flat** (inlet temp only) | tracks degree-days |

The WPB and the Elektroboiler deliver *the same daily hot water*; they differ
by roughly the COP in power and inversely in duration. That is the physical
identity the detector and the validation both rest on.

**Why bother:** ~2 kWh/day of near-perfectly shiftable load, on a tank that is
already a thermal store, in households nobody has on a list. For flexibility
management that is a better target than another PV classifier.

## 2. The blocker: 4 labelled positives

Measured on `HackDays2026 - GIGI - annotated.csv`:

- **10** households carry `Wärmepumpenboiler = x` (631 `-`, 551 blank).
- All 10 resolve in `gigi_augmented.csv`; **4 have meter data**:
  `692590`/mp 60933, `712273`/mp 154700, `738591`/mp 58311, `870640`/mp 124844.
- `712273` is PLZ 4313 (canton SO), outside AEW's territory — it may not
  survive the join chain, so plan for **3–4**.
- Each of the 4 maps to exactly **one** `mp_id`, so the meter→household rollup
  question is moot for the labelled work. (`gigi_augmented.csv` is *not*
  pooled per GP-Nr: 7,142 rows / 7,054 unique `gp_nr`, 56 with >1 meter,
  `mp_id` always scalar. Pooling matters only when scoring the population,
  where presence should be OR-ed across a household's meters — not
  consumption-weighted as the PV model does, because presence is a discrete
  event, not a continuous shape.)
- As with PV, **`-` never means "no WPB"** — it marks a row about a different
  asset. There are **zero confirmed negatives**.

**Consequence: supervised classification is off the table.** This is a
rule-based physical detector, validated without labels.

## 3. Detector

Per meter-month, take **per-slot quantiles across days**, and run the
detector on the **p10 profile** — the "always on" envelope.

Using p10 is the one design decision that does the most work. A load must be
present on ≥90% of days to lift it, so once-a-day timer regularity is enforced
*structurally*, with no separate regularity test — and an EV charged a few
nights a week, or an occasional appliance, cannot produce a plateau.
`tests/test_wpb_core.py::test_p10_envelope_rejects_occasional_load` pins this.

1. **baseline** = quietest rolling hour of the profile (wrapping midnight).
2. **plateau** = contiguous run where `p10 − baseline ≥ 0.25 kW`, circular so a
   23:00 start is one plateau and not two. Search thresholds are deliberately
   permissive; the WPB band is applied afterwards, otherwise resistance
   boilers — needed as contrast class *and* as injection donors — are missed.
3. **classify** by amplitude and duration (§1 table).
4. **score** by requiring the plateau in **both a summer and a winter month**,
   with consistent amplitude and start time.

Step 4 is the confounder killer. Everything else that makes a long, low,
nightly plateau is seasonal: dehumidifier (summer), bathroom floor heating
(winter, tracks degree-days), space HP at part load (winter, multi-cycle).
A WPB is not. Pond/circulation pumps are continuous rather than cyclic;
trickle EV charging is ≥1.4 kW and irregular between days.

## 4. Validation without labels

Three legs, in descending strength:

**(a) Detectability curve — manufactured ground truth.** Take a donor meter
with an unambiguous ≥1.5 kW resistance plateau, and convert its *real* daily
hot-water event into the WPB the same household would have after a retrofit:
**hold daily energy fixed, trade power for duration** (same tank, ~1/COP the
power, ~COP the time). Add that to an unlabelled host and re-run the detector.
Sweep target power against host noise.

Rescaling only the amplitude is wrong and was the first version's bug: it
yields a 0.5 kW plateau lasting 45 minutes, which is not a WPB and which the
classifier correctly rejects. The energy-conserving version is both physically
right and a better story — it is literally the Elektroboiler→WPB swap.

This yields "at X kW on hosts with noise below N, the plateau is recovered P%
of the time". It is a **sensitivity** statement, not accuracy: hosts are
unlabelled and some may already own a WPB.

**(b) PLZ base rate vs. GWR.** Rank-correlate the predicted flag rate per
postcode against the GWR share of buildings whose **hot-water** heat generator
is a heat pump (`GENW1`/`ENW1`, distinct from `GENH1`/`ENH1` for heating).
n ≈ 205 postcodes — a real test, unlike n = 4. Same shape as
`build_baserate.py` does for PV. *Verify the GWR code list before trusting it.*

**(c) The 4 labelled profiles.** Plot and inspect. Anecdote, reported as such.

**Rejected:** slicing the 4 positives into day-windows. It multiplies rows
without adding independent households — pseudo-replication, and any confidence
interval from it is fiction. Slicing *is* legitimate as a temporal hold-out
(detect on 2023–24, confirm on 2025–26), which is a stability metric.

## 5. Staging — every stage is a stopping point

| # | stage | output | why here |
|---|---|---|---|
| 0 | `noise` | noise-floor CSV | **go/no-go** |
| 1 | `profiles` | per-meter-month p10/p50/p90 | one pass, then iterate in seconds |
| 2 | `labelled` | the 4 profiles, all 96 slots | decides the search window |
| 3 | `detect` | `wpb_predictions.csv` + evidence JSON | the deliverable |
| 4 | `inject` | detectability curve | the only quantitative number |
| 5 | `figures` | slide figures | |

**Stage 0 is not optional.** A 0.5 kW WPB adds **0.125 kWh per 15-min slot**.
If the night-load noise floor sits at or above that, amplitude-based detection
cannot work and stages 1–5 are wasted. It is one file and a few minutes.

**Stage 2 before hard-coding a window.** Modern WPBs are PV-self-consumption
controlled and run at *midday*, not at night. GIGI is ~9:1 PV-skewed, so a
night-only detector could miss exactly these 4. Look first, then choose.

## 6. Honest limitations

- **No accuracy number is possible.** 4 positives, 0 negatives. Say so.
- **PV masking is a selection bias, not just a miss**: a PV-controlled WPB
  running at midday is invisible in *net* import. The feed-in channel
  (OBIS 2.29) would resolve it, but the PV model deliberately excludes it as
  the challenge asks for inference from net load — same call here.
- **Single-month profiles smear a jittery start.** A timer-locked boiler is
  fine; a demand-controlled one erodes its own p10 core. Detected duration is
  therefore a *lower* bound (visible in testing: a 4 h plateau read as 2.8 h).
- **A shared 3-phase meter or a second dwelling** breaks the one-tank
  assumption.
- Scope is canton AG; non-AG labels mostly do not join (`data_problems.md` ②).

## 7. If more time

- **Install-date change detection**: CUSUM on the per-meter-day nightly
  minimum across 4 years finds the retrofit date and gives before/after
  evidence. An Elektroboiler→WPB swap — 3 kW short plateau replaced by a
  0.5 kW long one — is the most legible fingerprint in the whole dataset.
- **Ripple control**: households on one circuit switch at the *same* 15-min
  slot. Cross-household synchronisation identifies switched water heaters
  outright; amplitude then splits WPB from resistance. Needs AEW's schedules;
  failing that, cluster on plateau start-time and look for shared sharp edges.
- **Vacation windows** as a confirmation test: when the plateau vanishes with
  the rest of the load, it is hot-water-linked.
