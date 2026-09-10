# Implementation handover — Wärmepumpenboiler detector

> **Completed 2026-09-10.** This document describes the original handover
> state. The implementation, full 43-month run, GWR comparison, report, and
> slide deck are now complete. See [`WPB_REPORT.md`](WPB_REPORT.md) and
> [`README.md`](README.md) for current commands and measured results.

**Audience:** a coding agent with no prior context on this project.
**Read this file top to bottom before writing code.** It is self-contained:
every measured number, parsing quirk and design constraint you need is here.

---

## 0. Situation

A reference implementation already exists as a git patch
(`wpb-detector.patch`, commit `a985d76`). **First action: try to apply it.**

```bash
git checkout -b claude/zealous-carson-5d6w1x     # if not already on it
git am wpb-detector.patch
python tests/test_wpb_core.py                    # must print ALL PASS
```

- **If it applies and the tests pass** → Part 2 is done. Verify against
  Part 3's acceptance criteria, then go to **Part 4** (the real work).
- **If it is unavailable** → build Part 2 from the spec below. The spec is
  complete; do not guess at anything it leaves unstated, ask instead.

The reference implementation has **never been run on the real data** — the
77 GB of exports were absent from the machine it was written on. Everything
in it is verified only against synthetic data. Treat Part 4 as the point
where reality first enters.

---

## 1. Context you need

### What we are detecting

A **Wärmepumpenboiler (WPB)** is a heat-pump *water heater*: a small
compressor that reheats a 200–300 L domestic-hot-water tank. **It is not a
space-heating heat pump.** Conflating the two is the primary failure mode of
this whole analysis.

| | resistance Elektroboiler | **WPB** | space-heating HP |
|---|---|---|---|
| electrical power | 2–4 kW | **0.3–1.0 kW** | 1–5 kW, modulating |
| duration | ~1–2 h | **3–8 h** | many cycles/day |
| daily energy | 6–8 kWh | **1.5–2.5 kWh** | strongly seasonal |
| seasonality | flat | **flat** (inlet temp only) | tracks degree-days |

A WPB and an Elektroboiler deliver *the same daily hot water*. They differ by
roughly the COP (~3) in power, and inversely in duration. **This identity is
load-bearing** — the detector's amplitude split and the validation method
both derive from it.

### The unit conversion everything depends on

The exports hold **kWh per 15-minute slot**. Therefore:

```
kW = kWh_per_slot × 4
a 0.5 kW WPB  ->  0.125 kWh per slot
```

0.125 kWh/slot is close to household noise. **This is the project's central
risk.** See the go/no-go in Part 3.

### Data layout

```
$PV_WORK_ROOT/                          (default /home/renku/work)
├── newspaper/                          this repo
└── store/input_data/
    ├── <year>/<German month> <year>/LG_AIM2Hackerdays_kWh_<ts>.csv
    ├── mpid_zähler_mapping.csv
    ├── Zähler-GP.csv
    └── HackDays2026 - GIGI.csv
```

42–43 monthly exports, Jan 2023 – Jul 2026, 0.8–3.4 GB each, ~77 GB total.

**Join chain:** `MP ID` → `mpid_zähler_mapping` (`Zählpunktbezeichnung`) →
`Zähler-GP` (`GPartner`) → GIGI (`GP-Nr`).

### Parsing quirks — all of these will bite you

`models/config.py` and `models/stream.py` already handle every one. **Use
them; do not write a new reader.**

| quirk | handling |
|---|---|
| `;` delimiter, `DD.MM.YYYY` dates | set explicitly in `config.py` |
| The `April 2023` export's header is **missing the `OBIS-Code` label** — trusting header names silently shifts every column | **parse by column position**, never by name. Row = `MP ID ; OBIS-Code ; Datum ; PLZ ; <96 values>` |
| UTF-8 BOM on `GIGI.csv` and `Zähler-GP.csv` corrupts the first column name | read with `utf-8-sig` |
| Phantom trailing 101st column | `usecols=range(100)` |
| German month folders (`Januar 2024`); 2023 has a doubled `2023/2023/` level | `config.discover_monthly_exports()` |
| Duplicate flat-directory copies of each export | deduplicated by `discover_monthly_exports()` |
| `2023-03` missing; `2024-10` partial (~32k meters vs ~47k) | `config.KNOWN_*_MONTHS` |
| Two rows per meter-day, one per OBIS register | `1-1:1.29.0*255` = grid import (**use this**), `1-1:2.29.0*255` = feed-in (**do not use as a feature** — see Part 5) |
| Meter rollout: ~27k meters (Jan 2023) → ~90k (Jul 2026) | a meter absent from a month is normal; require ≥10 valid days |

### The labelled cohort — measured, not estimated

From `data_addition/HackDays2026 - GIGI - annotated.csv`, column
`Wärmepumpenboiler`: **10 `x`, 631 `-`, 551 blank.**

All 10 resolve in `gigi_augmented.csv`; **4 have meter data**:

| GP-Nr | mp_id | PLZ | note |
|---|---|---|---|
| 692590 | 60933 | 8962 | |
| 712273 | 154700 | 4313 | canton **SO** — outside AEW territory, may not join |
| 738591 | 58311 | 5745 | |
| 870640 | 124844 | 5027 | |

Each maps to **exactly one** `mp_id`, so meter→household rollup is moot for
the labelled work.

> **`-` NEVER means "no WPB".** It marks a GIGI row that is *about a
> different asset*. There are **zero confirmed negatives**. Do not build a
> supervised classifier. Do not compute precision/recall against `-` rows.

`gigi_augmented.csv` is **not** pooled per GP-Nr: 7,142 rows / 7,054 unique
`gp_nr`, 56 with >1 meter, `mp_id` always a scalar. When scoring the
population, roll meters up to a household with **OR / max** (presence is a
discrete event), *not* the consumption-weighted mean the PV model uses for
its continuous shape signal.

---

## 2. Module spec

Three modules. Keep detector logic free of I/O so it stays testable without
the 77 GB.

### 2.1 `models/wpb_core.py` — pure logic, no I/O

Constants:

```python
SLOTS_PER_DAY = 96
SLOTS_PER_HOUR = 4
KW_PER_KWH_SLOT = 4.0
WPB_KW_MIN, WPB_KW_MAX = 0.25, 1.20
RESISTANCE_KW_MIN = 1.50
WPB_MIN_HOURS, WPB_MAX_HOURS = 1.5, 9.0
SEARCH_MIN_HOURS = 0.5      # see TRAP 1
```

**`slot_to_clock(slot: int) -> str`**
0-based slot `k` covers the 15 min *ending* at `(k+1)*15` minutes past
midnight. So slot 0 → `"00:15"`, slot 95 → `"00:00"`.

**`day_quantile_profiles(days, quantiles=(10.,50.,90.)) -> ndarray`**
`(n_days, 96)` kWh → `(len(quantiles), 96)`. Use `np.nanpercentile(axis=0)`
so a partially-delivered day still contributes its present slots. Return all
NaN for empty input.

> **The p10 profile is the core design decision.** Detect on the 10th
> percentile across days, not the mean. A load must be present on ≥90% of
> days to lift it, so once-a-day timer regularity is enforced *structurally* —
> no separate regularity test is needed, and an EV charged a few nights a
> week cannot form a plateau. There is a test that pins exactly this.

**`baseline(profile, window_hours=1.0) -> float`**
The meter's standing load: the minimum of the rolling `window_hours` mean over
the profile. **Must wrap around midnight** — the quiet hour often straddles
00:00. A rolling window rather than a global min stops one anomalous slot
setting the reference too low.

**`night_noise(days, night_slots=None) -> float`**
Default window 00:00–06:00. Per night slot take the median across days, then
the scaled MAD (`1.4826 × median|resid − median(resid)|`) of the residuals.
Units: kWh/slot. **This is the go/no-go statistic** — compare against 0.125.

**`Plateau`** dataclass: `start_slot, end_slot, n_slots, duration_h,
start_clock, end_clock, amplitude_kw, p50_amplitude_kw, energy_kwh,
baseline_kwh_slot, wraps_midnight`, plus `.as_row() -> dict`.

**`find_plateaus(p10, p50, min_kw=WPB_KW_MIN, min_hours=SEARCH_MIN_HOURS, base=None) -> list[Plateau]`**
Contiguous runs where `p10 − baseline ≥ min_kw/4`, **circular** so a 23:00
start is one plateau and not two. Sort longest first.

> **TRAP 1 — search thresholds must be permissive; only `classify` applies
> the WPB band.** Searching with `WPB_MIN_HOURS` makes resistance boilers
> unfindable: they dump the same daily energy in ~1 h, and once start-time
> jitter is eroded by the p10 envelope their always-on core is only 2–3 slots.
> You need them found — they are both the contrast class that makes the
> amplitude split defensible *and* the donors for the injection test.
> This was a real bug in the reference implementation, caught by
> `test_finds_resistance_boiler`.

**`classify(p: Plateau) -> str`**
`"resistance_boiler"` if `amplitude_kw ≥ 1.5`; `"wpb_candidate"` if
amplitude in `[0.25, 1.20]` **and** duration in `[1.5, 9.0]` h; else
`"other"`.

**`score_household(summer: list[Plateau], winter: list[Plateau]) -> (float, dict)`**
- neither month has a `wpb_candidate` → `0.0`
- exactly one → `0.35` ("seasonal load not excluded")
- both → `0.5 + 0.3·amp_ratio + 0.2·max(0, 1 − start_shift_h/3)`, capped at 1.0,
  where `amp_ratio = min/max` amplitude and `start_shift_h` is the circular
  start-slot difference in hours.

Return `(score, evidence_dict)`; the dict carries both plateaus and a
human-readable `reason`.

> **Why the both-seasons requirement does the real work:** every competing
> long, low, nightly load is seasonal — dehumidifier (summer only), bathroom
> floor heating (winter, tracks degree-days), space heat pump at part load
> (winter, multi-cycle). A WPB is not: it tracks *cold-water inlet
> temperature*, which lags air temperature by 1–2 months and varies weakly.
> Pond/circulation pumps are continuous rather than cyclic; trickle EV
> charging is ≥1.4 kW and irregular between days.

### 2.2 `models/wpb.py` — I/O, staging, figures

CLI: `python -m newspaper.models.wpb <stage> [...]`

| stage | args | does |
|---|---|---|
| `profiles` | `--month` (repeatable), `--sample-every N`, `--keep-raw N`, `--labelled-only` | one streaming pass per month → `artifacts/wpb_profiles_<ym>.npz` |
| `noise` | `--month` | go/no-go report → `artifacts/wpb_noise_floor.csv` |
| `labelled` | `--summer --winter` | plots the 4 labelled profiles, all 96 slots |
| `detect` | `--summer --winter [--threshold 0.8]` | → `wpb_predictions.csv` + `wpb_evidence.json` |
| `inject` | `--month [--amplitudes ...] [--n-hosts N]` | → `wpb_detectability.csv` |
| `figures` | `--summer --winter` | → `artifacts/figures/*.png` |

`build_profiles` streams via `stream.iter_chunks(path, obis=C.OBIS_IMPORT)`,
groups day rows per `mp_id`, drops meters with <10 days, stores
`p10/p50/p90 (n_meters, 96)`, `n_days`, `noise`, `plz`. It also keeps raw day
matrices for the first `--keep-raw` meters (default 400) — the injection test
needs raw days, not quantiles.

`--sample-every N` must sub-sample **deterministically by a hash of the meter
id**, so the same meters are selected in every month. A sample that changed
between months would silently break the summer/winter pairing.

Peak RAM for a full-population month is ~1–1.5 GB; use `--sample-every` if
that is too much.

`wpb_predictions.csv` columns: `mp_id, plz, wpb_score,
night_noise_kwh_slot, amplitude_kw, duration_h, start_clock, daily_kwh,
has_winter_month, reason, is_labelled_wpb`.

### 2.3 `models/wpb_slides.py`

Builds `artifacts/wpb_slides.pptx` from whatever artifacts exist, skipping
missing figures so it works at any stage. Requires `python-pptx`.

---

## 3. Acceptance criteria

### Unit tests (`tests/test_wpb_core.py`) — all must pass

Households are synthetic; these pin *logic*, not accuracy.

1. `slot_to_clock(0) == "00:15"`, `slot_to_clock(95) == "00:00"`
2. **finds a 0.5 kW plateau from 23:00 wrapping past midnight**;
   `wraps_midnight` is True, amplitude within 0.4–0.65 kW
3. **finds a 3 kW / 1.5 h resistance boiler** (this is what TRAP 1 breaks)
4. no `wpb_candidate` in a clean household
5. **a 0.5 kW load present on only 50% of days produces no `wpb_candidate`**
   — the p10 design claim
6. year-round score > 0.8; single-season score < 0.5
7. night-noise floor on a quiet synthetic household < 0.0625 kWh/slot

### End-to-end smoke test

Generate a synthetic monthly export in the real on-disk format (including
both OBIS rows and the trailing `;`), point `PV_WORK_ROOT` at it, and run all
six stages. The reference implementation was validated this way; on that
fixture `detect` scored the 4 labelled meters 0.86–0.96 at ~1.6–1.9 kWh/day.

---

## 4. The work that actually remains

Everything above is code. This is where the real data enters. **Run in this
order — stage 0 can cancel the rest.**

### Stage 0 — go/no-go (do this first, it is one file and a few minutes)

```bash
python -m newspaper.models.wpb profiles --month 2025-07 --sample-every 20
python -m newspaper.models.wpb noise    --month 2025-07
```

A 0.5 kW WPB adds 0.125 kWh/slot. Report the fraction of meters whose night
noise floor is below half of that.

- **Most meters below** → proceed.
- **Most meters above** → **stop and report.** Amplitude-based detection
  cannot work on this population. Do not "fix" it by lowering thresholds —
  that manufactures false positives. Either restrict to the quiet subset and
  say so, or fall back to duration-and-regularity features only.

### Stage 1 — look at the 4 labelled households *before* fixing a window

```bash
python -m newspaper.models.wpb labelled --summer 2025-07 --winter 2025-01
```

**Open question the data must answer:** modern WPBs are PV-self-consumption
controlled and run at **midday**, not at night. GIGI is ~9:1 PV-skewed, so a
night-only detector could miss exactly these 4. Look at all 96 slots, then
decide. Report what you see either way — a midday plateau is a finding, not
a failure.

### Stage 2 — population scan

Full 4 years is the goal. Iterate on one summer + one winter month first;
only run the full sweep once thresholds are settled. All tuning happens on
the `.npz` profiles, never by re-reading the exports.

### Stage 3 — the three validation legs

**(a) Detectability curve.** Already implemented (`inject`).

> **TRAP 2 — the injection must conserve energy, not amplitude.** Take a
> donor meter with an unambiguous ≥1.5 kW resistance plateau and convert its
> *real* daily event into the WPB the same household would have after a
> retrofit: **hold daily energy fixed and trade power for duration** (same
> tank, ~1/COP the power, ~COP the time). Rescaling only the amplitude yields
> a 0.5 kW plateau lasting ~45 minutes — which is not a WPB, and which
> `classify` correctly rejects. That was a real bug: recovery was 0% at every
> amplitude until it was fixed, then 95–100% at 0.3–0.5 kW.
>
> Note the curve legitimately *falls* at high amplitude: energy is conserved,
> so more power means shorter duration, and past ~1.2 kW the plateau leaves
> the WPB duration band. The curve measures the **band**, not raw sensitivity.
> Say so rather than "fixing" it.
>
> This is a **sensitivity** statement, not accuracy: hosts are unlabelled and
> some may already own a WPB.

**(b) PLZ base rate vs. GWR — not yet implemented.** Rank-correlate the
predicted flag rate per postcode against the GWR (Gebäude- und
Wohnungsregister) share of buildings whose **hot-water** heat generator is a
heat pump. Fields: `GENW1`/`ENW1` (hot water) — *not* `GENH1`/`ENH1`, which
are space heating. n ≈ 205 postcodes, so this is a real test unlike n = 4.
Mirror the structure of `models/build_baserate.py`, which does the equivalent
for PV from the BFE ElPA register.
**Verify the GWR code list against current BFS documentation before trusting
it — it was not verified when this plan was written.**

**(c) The 4 labelled profiles.** Inspect and include. Anecdote; label it so.

> **Rejected — do not do this.** Slicing the 4 positives into day-windows to
> get "more samples". It multiplies rows without adding independent
> households; any confidence interval from it is fiction. Slicing *is*
> legitimate as a temporal hold-out (fit on 2023–24, confirm on 2025–26),
> which is a stability metric, not a recall measure.

### Stage 4 — deliverables

`wpb_predictions.csv` + `models/WPB_REPORT.md` + `artifacts/wpb_slides.pptx`,
mirroring `models/REPORT.md` for the PV model. The report must state the
limitations in Part 5 explicitly.

### Stage 5 — if time remains

- **Install-date change detection.** CUSUM on the per-meter-day nightly
  minimum across 4 years finds the retrofit date and yields before/after
  evidence. An Elektroboiler→WPB swap — a 3 kW short plateau replaced by a
  0.5 kW long one — is the most legible fingerprint in the entire dataset.
  Requires storing one nightly-minimum value per meter-day (~470 MB at
  float32 for the full population), which `profiles` does not yet do.
- **Ripple control (Rundsteuerung).** Households on one switching circuit
  change state at the *same* 15-min slot. Cross-household synchronisation
  identifies switched water heaters outright; amplitude then splits WPB from
  resistance. Needs AEW's switching schedules — assume unavailable, so
  instead cluster meters on plateau start-time and look for shared sharp
  edges.
- **Vacation windows** as confirmation: when the whole load collapses for
  ≥5 days and the plateau goes with it, the plateau is hot-water-linked.

---

## 5. Constraints, and what must not be claimed

**Hard rules:**

1. **No supervised classifier.** 4 positives, 0 confirmed negatives.
2. **Never treat GIGI `-` as a negative.**
3. **Do not use the feed-in register (OBIS 2.29) as a model feature.** The
   challenge asks for inference from *net load*. The PV model excludes it for
   the same reason and uses it offline only, for label enrichment and QC.
   Same call here.
4. **Do not lower thresholds to raise the flag rate.** The bands are physical
   (§1). If the population disagrees with them, that is a finding to report.
5. Scope is canton **AG**; non-AG labels mostly fail the join.
6. Treat all customer data as anonymised; no per-household identifiers in any
   published output beyond `mp_id`/`GP-Nr`.

**Limitations the report must state:**

- **No accuracy figure is possible.** Say so plainly; it reads as rigour.
- **PV masking is a selection bias, not merely a miss.** A PV-controlled WPB
  running at midday is invisible in net import, and GIGI is ~9:1 PV-skewed.
- **Detected duration is a lower bound.** Start-time jitter erodes the p10
  core — a 4 h plateau read as 2.8 h in testing.
- A shared 3-phase meter or a second dwelling breaks the one-tank assumption.
- Single-month profiles smear a demand-controlled (non-timer) boiler.

## 6. Background reading in this repo

| file | what it gives you |
|---|---|
| `docs/wpb_detector_plan.md` | the reasoning behind every decision above |
| `docs/data_problems.md` | every known data problem and its resolution |
| `docs/pv_mvp_plan.md`, `models/REPORT.md` | the PV model — the pattern to mirror |
| `models/config.py`, `models/stream.py` | paths, constants, the streaming reader |
| `models/build_baserate.py` | the per-PLZ base-rate pattern, for GWR |
