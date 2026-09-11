# Plan: PV labels versus negative net load

## Objective

Determine whether every household/site labelled as having photovoltaic (PV) generation
in `HackDays2026 - GIGI.csv` has at least one negative 15-minute net-load reading in
the available monthly electricity exports. Report the count and identity of labelled
PV sites with and without observed negative load; do not interpret an absence as proof
that a site has no PV generation.

## Approach

1. **Inspect the three reference tables.** Read the semicolon-delimited files with
   `utf-8-sig` encoding and record their column names, row counts, duplicate keys, and
   missing values. Identify the exact PV label column(s) and document the rule used to
   classify a `GP-Nr` as PV-positive, including treatment of blank/unknown values.

2. **Build an auditable linkage table.** Join `HackDays2026 - GIGI.csv` (`GP-Nr`) to
   `Zähler-GP.csv` (`GPartner`) and then to `mpid_zähler_mapping.csv` through
   `Zählpunktbezeichnung`. Preserve one row per resulting `MP ID` and retain the
   originating `GP-Nr`, PV flag, installation (`Anlage`), and postal code. Calculate
   join-coverage diagnostics: PV-labelled GPs with no meter point, meter points linked
   to multiple GPs, and duplicate rows.

3. **Define negative load precisely.** Treat a time value strictly below zero as
   negative net load. Before evaluating it, inspect the measurement units and
   `OBIS-Code` distribution and select the register(s) that represent household net
   energy rather than unrelated cumulative/import/export registers. Record the chosen
   rule in the notebook.

4. **Stream the monthly exports.** Locate all 43 monthly files and process one file
   at a time (and, if necessary, in chunks). Read only `MP ID`, `OBIS-Code`, `Datum`,
   and the 97 quarter-hour columns. Filter immediately to MP IDs linked to PV-positive
   GPs and to the selected load register(s), avoiding a full multi-month in-memory
   dataset.

5. **Detect evidence per meter point.** Coerce quarter-hour columns to numeric using
   the observed decimal convention, then flag an MP ID if any valid interval is below
   zero. Store minimal evidence: first negative timestamp/date and interval, its value,
   source month, and the number of negative intervals. Separately track whether an MP
   ID had any valid observations at all.

6. **Aggregate to household/site.** Roll meter-point results up to `GP-Nr`: a
   PV-labelled household has observed negative load if *any* linked MP ID has a
   negative interval. Keep a separate status for unlinked GPs and GPs whose linked
   meter points were never observed, rather than grouping them with a genuine
   no-negative-load result.

7. **Validate the result.** Check a small sample of positive and negative cases by
   reopening their source rows; inspect negative-value magnitudes and daytime/seasonal
   timing for plausibility. Quantify outcomes at both meter-point and GP levels, and
   verify that row duplication cannot create or suppress a household-level positive.

8. **Present a privacy-conscious summary.** Produce aggregate counts and rates for:
   labelled PV GPs, successfully linked GPs, GPs with observations, and GPs with at
   least one negative interval. Keep individual identifiers only in an in-memory or
   local diagnostic table needed for reproducibility; notebook displays should default
   to aggregates and pseudonymized examples.

## Expected notebook structure

1. Configuration and paths
2. Reference-table inspection and PV-label rule
3. Linkage construction plus coverage diagnostics
4. Monthly streaming scan for negative intervals
5. Household-level aggregation and results
6. Validation samples, limitations, and conclusions

## Limitations to state explicitly

- Net load may not become negative despite PV because concurrent consumption exceeds
  generation, export may be metered separately, or the selected register/sign convention
  differs from the expected one.
- The data ends in July 2026, whereas PV installation/project dates may fall outside
  all or part of the observation window.
- One household can have multiple installations or meter points, and a meter-point
  mapping can be incomplete or ambiguous.
- Missing intervals, invalid values, and seasonal coverage can hide genuine export.
