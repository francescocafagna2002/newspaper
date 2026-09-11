# Battery MVP execution report

Date: 2026-09-11

The GIGI label build reproduces every acceptance count in the plan, including
608 battery-positive households, 222 with sufficient meter history, 214
audit-negative households, 82 with sufficient meter history, 469 dated
commissionings, and 105 metered installations inside the measurement window.

The next mandatory gate fails before model fitting. The complete, existing
all-meter scan shows that only **208/222 (93.7%)** usable known-battery
households have any export energy. “At least 10 positive-export days” is a
strict subset of “any export,” so its retention is bounded above by 93.7%,
below the required 95%.

No threshold was weakened, no label-based exception was introduced, the audit
controls were not used for training, and no battery probabilities were
fabricated. In accordance with §5 and §7 of the implementation specification,
the pipeline was stopped before aggregation, training, evaluation, and
scoring. There is consequently no defensible headline model metric yet.

To unblock the MVP, the plan owner must explicitly choose a revised,
label-blind universe rule or lower the acceptance criterion. A defensible next
experiment would compare measured export plus an import-side PV signature,
applied identically to labelled and unlabelled households; that is a plan
change and was not assumed here.

## Update — 2026-09-11: universe gate rule corrected

The 93.7% figure above was a measurement-rule bug, not a data limitation. The
gate's "measured" test required a *positive* export reading, so it could not
distinguish "register never read" from "register read every day, always
zero." Inspecting the 14 rejected households confirms all 14 have a working
export register with 100+ daily readings each — every one reads exactly
0.0 kWh on every day. That is not missing data: a battery run under a greedy
self-consumption or feed-in-limiting strategy is expected to leave no export
signature at all (`battery_evidence_base.md` §1, §C — "export is delayed,
clipped and truncated," self-consumption maximisation is the dominant
residential control mode). The gate was rejecting a real battery archetype as
if its meter were broken.

The fix (label-blind, no branch consults `battery_positive`): admission now
requires only that the export register was read at least once
(`exp_days > 0` in the prior complete scan), not that it ever read positive.
`define_universe.build_universe()` admits register-present-but-always-zero
households under a new `measured_zero_export` rule, distinct from
`measured_export` (register present, ≥10 positive-export days). The same
correction was applied to `build_features._candidate_meters()`, which fed the
streaming pass the same overly-narrow candidate set.

Corrected retention: **222/222 (100%)**, verified by
`retention_upper_bound()` and covered by
`test_universe_gate_upper_bound_meets_required_retention` in `test_battery.py`
(previously `..._is_below_required_retention`, pinned to the buggy 208/222).

This unblocks the universe gate only. `artifacts/meter_export_days.csv` (the
output of the full Step 3 streaming pass, `build_features.py` run over all 42
monthly exports) still does not exist, and Steps 4–7 (aggregation, training,
evaluation, scoring) are still unimplemented per the original plan. There is
still no trained model, no scored population, and no defensible headline
metric — this update fixes the gate, not the rest of the pipeline.

