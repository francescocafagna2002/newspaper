# Wärmepumpenboiler detection from smart-meter data

Energy Data Hackdays 2026 · AEW “Energy Fingerprints” · 2026-09-10

This pipeline searches for heat-pump water heaters (WPBs), not space-heating
heat pumps. It processed all 43 available monthly grid-import exports from
January 2023 through July 2026 and produced an interpretable score for each
meter and household.

## Result

The latest January/July pair scored 87,860 meters in canton Aargau. After the
meter-to-household mapping and an OR/max rollup, **838 of 75,094 households
(1.12%)** were flagged at score ≥0.8. The corresponding meter result contains
863 flags (0.98%). Each result includes the amplitude, duration, start time,
daily energy, and the reason for its score.

These are **candidates, not confirmed installations**. No accuracy estimate is
possible because only four labelled-positive households have meter data and
there are no confirmed negatives.

Among the latest 863 flagged meters, the median detected plateau is 0.39 kW,
3.0 hours, and 1.36 kWh/day. The middle 50% spans 0.32–0.53 kW, 2.0–4.75 hours,
and 0.83–2.21 kWh/day. Start times are diverse: 8% start from 00:00–06:00, 30%
from 09:00–17:00, and 58% from 17:00–24:00. The detector therefore searches
the full day rather than imposing a night window.

| January/July pair | meters scored | meters with both months | flags | flag rate |
|---|---:|---:|---:|---:|
| 2023 | 29,226 | 25,545 | 345 | 1.18% |
| 2024 | 44,305 | 34,970 | 473 | 1.07% |
| 2025 | 66,098 | 48,332 | 622 | 0.94% |
| 2026 | 87,860 | 74,948 | 863 | 0.98% |

The growing counts reflect the smart-meter rollout. A flag in one year is not
very stable: 1,727 meters are flagged in at least one annual pair, while only
130 meters observed in at least two pairs are flagged in every observed pair.
This weak temporal stability limits confidence in a single-pair flag and is a
strong reason to retain the annual evidence rather than publish one binary list.

## Method

For each meter-month, the pipeline calculates p10, p50, and p90 consumption for
every quarter-hour across valid days. It detects contiguous runs in the p10
“always-on” envelope at least 0.25 kW above the meter’s quietest rolling hour.
A WPB-shaped plateau has 0.25–1.20 kW amplitude and 1.5–9 hours duration. A
household scores highly only when compatible plateaus appear in both January
and July with similar amplitude and start time.

This rule encodes daily regularity through p10 and uses seasonal persistence to
reject common confounders such as dehumidifiers, floor heating, and part-load
space-heating heat pumps. The thresholds come from device physics; they were not
fit to the four labels. Only OBIS 1.29 grid import is used as a feature.

## Validation

The stage-0 gate passed: in a deterministic 1-in-20 sample of July 2025,
**95.46% of 3,281 meters** had a night-noise floor below 0.0625 kWh per slot,
half the 0.125 kWh/slot step of a 0.5 kW WPB.

The manufactured-ground-truth study extracted daily events from 10 clear
resistance-boiler donors and injected energy-preserving, stretched versions
into 150 unlabelled hosts. On the 137 quiet hosts, recovery was 79.6% at
0.3 kW, 94.2% at 0.5 kW, 84.7% at 0.8 kW, and 35.8% at 1.2 kW. Recovery falls
at 1.2 kW because fixed energy produces a shorter event that often exits the
WPB duration band. This is a sensitivity study on manufactured signals, not an
accuracy measurement. Hosts are unlabelled, and injected electrical energy
has a median of 2.21 kWh/day (mean 2.91; range 1.03–8.25).

The four labelled profiles do not confirm recall. None passes the year-round
score threshold. Their July p10 profiles are nearly zero while January contains
large daytime or evening loads, consistent with PV masking or labels that do not
map to a clean, timer-driven WPB signature. The labels are shown as anecdotal
evidence and were not used to lower thresholds.

As an external ecological check, predicted postcode flag rates were compared
with the Swiss Federal Statistical Office GWR share of existing residential
buildings whose primary hot-water generator is coded as a heat pump
(`GWAERZW1=7610`). The code and German label were verified in the downloaded
GWR code table. Across 72 postcodes with at least 30 meters, Spearman ρ was
−0.12 (p=0.30). This weak, inconclusive association does not validate the
detector. The GWR measure is broader than standalone WPBs because it includes
combined space-heating/domestic-hot-water systems.

## Data handling and reproducibility

The sweep reads all 43 discovered exports, removes duplicate meter-days, rejects
invalid dates and negative/non-finite values, requires at least 10 sufficiently
complete days per meter-month, and handles the malformed April 2023 header by
column position. March 2023 is present in the current export set; October 2024
remains partial. Deterministic CRC32 sampling keeps the same meters across
months. Large profile caches allow detection and reporting to rerun without
rescanning the 77 GB source data.

Run `python -m models.heat_pump_boiler.pipeline --finish-only` from the
`newspaper/` directory to rebuild annual predictions, household results,
validation, and figures from cached profiles. The pure logic and isolated
real-format pipeline fixture are covered by 11 passing tests.

## Limitations

- There are four labelled positives and zero confirmed negatives, so precision,
  recall, ROC-AUC, and an accuracy figure cannot be estimated.
- PV-controlled WPBs may run at midday while PV generation keeps net grid import
  near zero. Because the labelled cohort is strongly PV-skewed, this is selection
  bias as well as a source of missed events.
- A p10 profile erodes start-time jitter, so detected duration is a lower bound.
  Demand-controlled devices may disappear entirely from the always-on envelope.
- Year-to-year flag stability is weak. The score should be used to rank candidates
  for inspection, not as proof of ownership.
- A shared three-phase meter or multiple dwellings behind one meter violates the
  one-tank interpretation.
- The GWR comparison is ecological and covers a broader equipment category than
  standalone heat-pump water heaters.

## Outputs

- `artifacts/wpb_household_predictions.csv` — latest household candidates
- `artifacts/wpb_predictions.csv` and `wpb_evidence.json` — latest meter evidence
- `artifacts/wpb_predictions_annual.csv` — all four annual pairs
- `artifacts/wpb_temporal_stability.csv` — repeated-year evidence
- `artifacts/wpb_detectability.csv` — manufactured-signal sensitivity
- `artifacts/wpb_gwr_correlations.csv` — postcode comparison
- `artifacts/figures/` and `artifacts/wpb_slides.pptx` — presentation outputs
