# Plan extension: LightGBM followed by high-confidence EV evidence

## Goal

Test whether the frozen `E22_lgbm_total` probability can be improved by a
positive-only, interpretable charging-event override. The LightGBM model remains
the base classifier. The second stage may raise a score only when repeated
15-minute load shapes provide unusually strong EV evidence; it may never lower
a model score or turn an existing positive prediction negative.

This is a post-test extension. The original test partition has already been
opened, so its result is a **reused-test diagnostic**, not a second unbiased
performance estimate. Select the extension solely on the existing validation
partition and write its frozen configuration before reading its reused-test
result.

## Research basis

- Hoffmann et al. (CIRED 2019) describe EV charging as extended, kW-level demand
  and warn that ordinary baseline household loads create false positives. This
  motivates residual power, duration, and repetition rather than a peak-only
  override: <https://sintef.brage.unit.no/sintef-xmlui/bitstream/handle/11250/2618624/Hoffmann%2BAutomated%2Bdetection%2Bof%2Belectric%2Bvehicles%2Bin.pdf?sequence=2>.
- Li et al. (2024) infer charging events from contextual start time, duration,
  and power level in smart-meter data. The override therefore retains those
  event fields and requires evidence in multiple weeks:
  <https://doi.org/10.1016/j.epsr.2024.110789>.
- Hangawatta et al. (2025) combine amplitude filtering, a 1.5-hour minimum
  duration, high-pass information, clustering, and template matching for
  low-rate data. This motivates the minimum-duration bound and plateau/edge
  shape checks: <https://doi.org/10.1016/j.segan.2025.101903>.
- Pu and Zhao (2022) jointly consider EV presence, charging rate, and charging
  periods on real 15-minute household data. This supports requiring consistency
  between ownership probability and inspectable event evidence:
  <https://doi.org/10.1109/SmartGridComm52983.2022.9961024>.

Published values define broad candidate ranges only. Every final cutoff is
selected on this dataset's validation properties.

## Data and leakage controls

Use `train_property_samples_auxw_netload.csv` for LightGBM and the already-built
`ev_classification/output/customer_week_table.parquet` for event evidence. Keep
the permanent `_total` group split. For every property, take at most its 12 most
recent eligible weeks without consulting its EV label, model score, or project
dates. A property without detailed weekly history receives no override.

Convert 15-minute kWh to average kW with `kW = 4 × kWh`. Use a household-local
daily 20th-percentile baseline, merge a single 15-minute gap, and test the
predeclared residual amplitudes 1.5, 3, and 5 kW. Test minimum durations 1.5 and
2 hours and the existing block, plateau-with-paired-edges, and repeated-level
families. Plateau candidates require relative MAD ≤0.2, start/end edges ≥0.5
kW, and edge symmetry ≥0.3.

An override candidate is true only when a weekly evidence score reaches 0.85,
0.90, or 0.95 in at least 1, 2, 3, or 4 of the retained weeks. Compare score
floors 0.7 and 0.9. Select by validation balanced accuracy, then PR-AUC,
precision, and fewer flags. The existing LightGBM decision threshold remains
fixed.

## Outputs and acceptance

Write the bounded search, frozen rule, weekly evidence, validation and reused
test predictions, paired bootstrap intervals, metrics, and a short report under
`output/heuristic_extension/`. Report ordinary accuracy and balanced accuracy;
class imbalance makes the latter the decision metric. Accept the extension only
if validation balanced accuracy improves without reducing precision. Regardless
of the reused-test outcome, do not retune after opening it.

## Heuristic extension 2 — steep-edge cascade

Research supports a sharp load jump as one component of an EV signature, rather
than a sufficient signature by itself. Zhang et al. characterize low-rate EV
charging as a high-amplitude square-wave-like segment and use segment boundaries,
duration, amplitude, and filters for air-conditioning interference:
<https://arxiv.org/abs/1404.5020>. EVSense describes the learned local signature
as a load jump followed by a steady waveform and load tail, and warns that
rule-based robustness is limited when other high-power appliances are present:
<https://doi.org/10.1145/3538637.3538840>. Hangawatta et al. likewise prioritize
start/stop detection but retain amplitude, a 1.5-hour duration filter, high-pass
information, clustering, and template matching rather than using slope alone:
<https://doi.org/10.1016/j.segan.2025.101903>.

Test an edge-triggered cascade after the same frozen LightGBM model. A qualifying
event must combine a residual load block, minimum duration, steep positive start
edge, stable plateau, and optionally a paired negative stop edge. Search only the
predeclared validation grid: amplitudes 1.5/3/5 kW; durations 1.5/2 hours; start
edges 1/2/3/4 kW per 15-minute step; relative plateau MAD 0.2/0.4; paired-edge
required or optional; at least 2/3 events across at least 1/2 weeks.

Compare these post-model directions independently:

1. **Positive rescue:** raise a LightGBM-negative score when strong edge evidence
   is present.
2. **Negative veto:** lower a LightGBM-positive score when strong edge evidence is
   absent.
3. **Two-way correction:** apply both operations.

Select the best rule within each direction and the overall direction by validation
balanced accuracy, followed by PR-AUC and precision. Freeze it before scoring the
reused test. Report how many false negatives and false positives each direction
changes. Absence of detected charging is not evidence that a charger is absent, so
the negative-veto hypothesis is expected to be weaker, but it must be tested rather
than assumed.

Promote a direction only if it improves validation balanced accuracy without
reducing precision. If none passes, freeze the best heuristic candidate for an
auditable diagnostic but deploy the no-op direction, leaving LightGBM unchanged.
