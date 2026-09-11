# Plan: PV labels versus negative net load

## Objective

For every customer (`GP-Nr`) labelled as having photovoltaic (PV) generation,
determine whether any linked meter point has at least one negative 15-minute net-load
interval in the monthly exports. The final result has exactly one row per customer and
includes meter-point count, mapping ambiguity, an asset-additions timeline, and
negative-load status. No observed negative load is not proof of no PV generation.

## Fixed definitions

- **Customer grain:** `GP-Nr`. Include every mapped `MP ID`; do not select a primary meter.
- **PV-positive:** trim whitespace and case-fold GIGI's ` PV` field. A customer is
  PV-positive if any of its records equals `x`; blank values and all other values
  (including `-` and `Boiler`) are non-positive. Consolidate duplicate GIGI records
  with this any-positive rule.
- **Negative net load:** when separate import and export registers are appropriate,
  calculate `net_load = import - export` for each MP ID/date/quarter-hour. A value below
  zero is negative load. Use a direct signed-net register only after verifying its
  documented sign convention.
- **Mapping ambiguity:** flag a customer if a linked MP ID maps to multiple customers,
  a metering designation has conflicting duplicate mapping rows, or the MP-to-customer
  relationship is otherwise non-unique. Keep ambiguous meter points in the analysis.

## Approach

1. **Inspect and normalize reference tables.** Read all three semicolon-delimited
   tables using `utf-8-sig`. Record exact headers, row counts, missingness, and duplicate
   keys. Normalize asset flags and preserve source GIGI rows internally for chronology.

2. **Build customer-to-meter-point linkage.** Join GIGI (`GP-Nr`) to `Zähler-GP.csv`
   (`GPartner`) and then to `mpid_zähler_mapping.csv` through `Zählpunktbezeichnung`.
   Retain all resulting meter points, source `Anlage`, postal code, PV flag, and mapping
   provenance. Aggregate to one customer row with `meter_point_count`,
   `mapping_ambiguity`, and `mapping_ambiguity_count`, explicitly retaining unlinked
   customers.

3. **Build the asset-additions timeline.** Normalize PV, heat-pump, battery/storage,
   EV-charger, heat-pump-boiler, and other populated asset fields. Sort each customer's
   GIGI records by parsed `InBetrieb-Datum`, falling back to parsed `Übergabe` only when
   commissioning is absent. Compare each record's assets with the cumulative earlier set
   and record only newly appearing assets. Combine same-date additions into one stable
   event. Store a list of `date`, `assets_added`, `date_source`, and `Anlage` where
   available. Retain undated additions with a null date and `date_source = unknown`.

4. **Validate registers and units.** Inspect `OBIS-Code` values and their source
   documentation/metadata to identify import, export, or verified signed-net registers.
   Confirm units are compatible before pairing import and export. Document selected codes,
   signs, and any conversion; never infer semantics merely from observed values.

5. **Discover and stream monthly exports.** Locate all matching monthly CSVs, report
   their count and calendar coverage, and warn if this differs from the expected 43 files.
   Process one file at a time (or in chunks), reading `MP ID`, `OBIS-Code`, `Datum`, and
   interval columns. Filter to meter points linked to PV-positive customers but retain all
   rows needed to derive net load.

6. **Derive interval evidence.** Parse numeric intervals and pair import/export by
   `MP ID`, date, and interval. Aggregate duplicate pair keys only when documented as
   additive. If either required side is missing, mark the interval invalid; do not replace
   it with zero. For each meter point retain valid-observation count, first negative
   timestamp/value/source file, and negative-interval count.

7. **Aggregate to customer.** A PV-positive customer is `observed_with_negative` if any
   included meter point has a negative derived-net interval. Otherwise use mutually
   exclusive states: `unlinked`, `linked_but_unobserved`, or
   `observed_without_negative`. Include meter-point count, mapping ambiguity, available
   commissioning/handover dates, and the asset-additions timeline. Do not filter readings
   by asset dates in this phase.

8. **Validate and present in the notebook.** Inspect source rows for samples of positive
   and negative cases; check register pairing, magnitudes, and timing relative to asset
   additions. Display aggregate summaries and the customer-level table as needed. Do not
   write a separate diagnostic export; identifiers remain only in transient notebook
   objects required for analysis.

## Expected notebook structure

1. Configuration and discovered data paths
2. Reference-table inspection and customer/PV consolidation
3. Customer-to-meter-point linkage and coverage
4. Customer asset-additions timeline
5. OBIS, unit, and sign validation
6. Streaming scan and derived net-load evidence
7. Customer-level result table and summary
8. Validation checks, limitations, and next analysis questions

## Limitations to state explicitly

- PV may not yield negative net load where simultaneous consumption exceeds generation.
- Some customer records and asset additions lack a reliable date.
- A `GP-Nr` can cover multiple installations/meter points, and mappings can be ambiguous.
- Missing or unpaired observations and incompatible units reduce available evidence.
- Measurements end in July 2026, so asset dates may not fully overlap the data.
