# EV-charger classification: customer-week table plan

## Objective

Create a modelling table for every customer (`GP-Nr`) represented in
`HackDays2026 - GIGI.csv`.  A usable observation is one customer in one ISO
calendar week.  It contains the customer's GIGI metadata and 672 quarter-hour
net-load values, one for each point from Monday `00:00` through Sunday `23:45`.

The eventual classification target is whether the customer has an EV charger.
This table is an input dataset for that later classifier; it does not train a
model or claim to identify individual charging sessions.  A later classifier
will score each complete week independently, so arbitrary-length customer
history is supported by splitting it into complete ISO weeks.  Any later
customer-level combination of weekly predictions must retain the individual
weekly scores as evidence; it must not aggregate the 672 load values.

## Inputs and relationship

```text
HackDays2026 - GIGI.csv (GP-Nr and customer/device fields)
  <- Zähler-GP.csv (GPartner, Zählpunktbezeichnung)
  <- mpid_zähler_mapping.csv (Zählpunktbezeichnung, MP ID)
  <- monthly exports (MP ID, OBIS-Code, Datum, 15-minute readings)
```

All inputs are semicolon-delimited.  Read GIGI and Zähler-GP with
`encoding='utf-8-sig'` so their first header is parsed correctly.  Discover
finalized monthly exports rather than assuming uninterrupted coverage.  Do not
use the temporary March 2023 export: report that month as unavailable in the
coverage summary and audit.

One finalized export (April 2023) has a shifted header that omits
`OBIS-Code` while its rows retain that field.  Detect and repair that known
schema variant explicitly, and record the repair in run metadata.

## Population and units

- Start with every non-empty `GP-Nr` in GIGI, including customers that cannot
  be linked to a meter point.
- Keep a separate customer-coverage/audit table with one row per GIGI customer.
  It records whether the customer is unlinked, has no observations, or has one
  or more eligible weeks.
- The modelling table contains only customer-weeks with linked, valid load
  data.  An unlinked customer cannot truthfully be given a 672-value load row.
- Retain `MP ID` in an intermediate long-form table.  At each timestamp, sum
  all linked meter points belonging to the same customer before making the wide
  week row.  Thus the final modelling unit is a customer, not an installation
  (`Anlage`) or individual meter point.
- `Anlage` is provenance only.  Do not use it as a model feature or to split
  load.  Keep mapping ambiguity flags in the audit output.

## Canonical GIGI customer record

GIGI contains repeated `GP-Nr` values.  Preserve all raw rows in a source audit
table, then make exactly one canonical record per `GP-Nr` before joining to
weeks.  Do not silently discard conflicts.

For each raw GIGI heading, retain a normalized canonical value plus an
explicit conflict/count field where needed:

- `PLZ`, `Ort`, `Kanton`: use the most frequent non-empty value.  If two or
  more values tie for the highest frequency, write `null`; retain all raw
  values in the consolidation audit.  Flag a conflict if more than one
  distinct non-empty value occurs.
- Asset flags (`WärmePumpe`, `PV`, `Batterie/Speicher`, EV charger,
  `Wärmepumpenboiler`): retain the raw distinct values for audit and create
  normalized boolean flags.  For the EV target, `ev_label=1` when any value is
  `x` after trimming whitespace and case-folding; otherwise it is `0`.
- `PV-Leistung in kWp`: parse numerically; retain the maximum valid value and
  flag conflicting populated values.
- `Datum Unterschrift`, `geplanter Baustart`, `Übergabe`, `InBetrieb-Datum`:
  parse only exact, valid `dd.mm.yyyy` values; write `null` for other populated
  values and retain their raw text in the audit.  Ignore blanks and retain
  source-value conflicts.  Use the
  earliest non-empty `Datum Unterschrift` and `geplanter Baustart`, and the
  latest non-empty `Übergabe` and `InBetrieb-Datum`.  These values describe the
  observed project span, not a device-specific timeline.  No date may be
  assumed to be the EV-charger commissioning date unless the source semantics
  are confirmed.  Set `label_date_ambiguous=1` by default whenever a charger
  label is present but no confirmed EV-specific commissioning date exists.

The customer-week table carries the original GIGI information in normalized,
unambiguous columns; the raw GIGI source rows remain available for review.

## Meter and register processing

1. Join `GPartner` to canonical `GP-Nr`, then join via
   `Zählpunktbezeichnung` to `MP ID`.
2. Produce a mapping-audit table with `gp_nr`, `mp_id`, meter-point
   designation, linked installation count, mapping status, and source row
   counts.  Do not duplicate readings when a mapping table has duplicate rows.
3. Inspect available OBIS information and document the selected register
   convention.  Derive each interval's net load from the compatible import and
   export registers:

   `net_load_kwh = import_kwh - export_kwh`

   For the initial table, use `1-1:1.29.0*255` as import and
   `1-1:2.29.0*255` as export.  This convention is the one used in
   `heuristics/pv_negative_load_guided.ipynb`, where its resulting negative-load
   behavior plausibly separated PV-labelled and non-PV-labelled customers.
   Record this provenance in the run metadata and retain the paired-register
   sanity check.  If a signed net register is selected in a future revision,
   document its sign convention and never combine it with the import/export
   calculation.
4. Pair registers by `MP ID`, date, and 15-minute interval.  If either required
   value is absent, mark the interval missing; never convert it to zero.
5. Aggregate derived net load by `gp_nr` and timestamp across all of that
   customer's mapped MP IDs.  This is the only aggregation before weekly
   reshaping.

## Week construction and missing data

- Interpret weeks as ISO weeks, Monday `00:00` to Sunday `23:45`, in the source
  time zone.  Exclude daylight-saving-transition weeks until the source
  time-zone/DST convention and a normalization rule are documented.  Do not
  force a 672-column week by interpolation, duplicated intervals, or dropped
  intervals.
- For ordinary 7-day weeks, expect 672 15-minute points.
- A missing value is an interval with no valid derived customer net load after
  the register pairing and customer aggregation.
- Interpolate only an internal consecutive run of at most 4 timestamps (one
  hour), bounded by observed values.  Do not interpolate a leading or trailing
  gap.
- Accept a week only if it has at most 12 missing timestamps before imputation
  (1.8% of 672), every missing run is at most 4 timestamps, and all imputation
  is internal.  Otherwise exclude it from the modelling table.
- Record the observed count, imputed count, longest missing run, and coverage
  ratio for every retained week.  The initial threshold is intentionally strict
  to avoid fabricating EV-like overnight load shapes; reassess only after
  reporting exclusion counts.

## Customer-week output schema

One row represents `(gp_nr, week_start)`.  Store values in kWh per 15-minute
interval.  `week_start` supplies the absolute timestamp; the 672 load headers
are fixed relative positions so every week has identical columns.

```text
gp_nr,
plz,
ort,
kanton,
heat_pump_label,
pv_label,
pv_capacity_kwp,
battery_storage_label,
ev_label,
heat_pump_boiler_label,
datum_unterschrift_earliest,
geplanter_baustart_earliest,
uebergabe_latest,
inbetrieb_datum_latest,
gigi_row_count,
gigi_location_conflict,
gigi_asset_conflict,
ev_label_conflict,
label_date_ambiguous,
eligible_for_supervised_training,
training_exclusion_reason,
week_start,
linked_mp_count,
observed_timestamp_count,
imputed_timestamp_count,
longest_missing_run,
coverage_ratio,
net_load_mon_00_00_kwh,
net_load_mon_00_15_kwh,
...,
net_load_sun_23_30_kwh,
net_load_sun_23_45_kwh
```

Generate all 672 net-load headings programmatically in weekday/time order.  Do
not use a single weekly total: the within-week timing is the classifier signal.

Set `eligible_for_supervised_training=0` and
`training_exclusion_reason='ev_label_conflict'` when the repeated GIGI rows
contain conflicting normalized EV values.  Otherwise set it to `1` and leave
the reason null.  Other asset conflicts and `label_date_ambiguous` remain audit
information and do not affect EV-training eligibility.

## Companion audit outputs

Write lightweight, non-model outputs alongside the table:

1. `customer_coverage`: every canonical GIGI customer, linked meter count,
   observed periods, eligible-week count, and exclusion reason.
2. `meter_mapping_audit`: all included links and their mapping ambiguity flags.
3. `week_quality_audit`: one row per attempted customer-week, with missingness,
   imputation, eligibility, and exclusion reason.
4. `gigi_consolidation_audit`: raw-row count, chosen canonical values, and
   detected conflicts per customer.

## Storage and run metadata

Write the modelling table and all four audit tables as Parquet datasets under
`ev_classification/`, partitioned by `week_start` year where that field exists.
Write a compact CSV run summary alongside them with source-month coverage,
customer and week counts, exclusions, and the selected register convention.
Keep a machine-readable run-metadata record that identifies the discovered
finalized input files, explicitly records the excluded March 2023 temporary
file, and records the register-sign provenance.

## Validation before handoff

1. Confirm the final table has no duplicate `(gp_nr, week_start)` rows and
   exactly 672 ordered net-load columns.
2. Confirm every retained week meets the missing-data policy and has no remaining
   null load cells.
3. Reconcile customer and meter counts with the mapping audit.  Report linked,
   unlinked, observed, and eligible customer totals.
4. Inspect a small, representative sample of customer-week profiles, including
   PV-positive and EV-positive labels, to verify units, signs, and timing.
5. Verify that all target labels originate only from GIGI and that no GIGI fields
   unavailable at real prediction time are subsequently used as model features.
6. Confirm DST-transition weeks are excluded and the March 2023 temporary file
   was not read.
7. Reproduce the paired-register PV sanity check used to support the initial
   import/export convention, and record its result in the run summary.
