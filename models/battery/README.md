# Battery detector

This package implements the first gated stages of
[`battery_mvp_plan.md`](battery_mvp_plan.md): label construction, solar
geometry, the one-pass import/export feature extractor, and the label-blind
PV-universe definition.

## Current status: universe gate passes; full feature pass not yet run

The label gate passes all published checks. The universe gate previously
failed because its "measured" test required *positive* export energy: only
208 of the 222 usable known-battery households ever logged a positive
export reading (93.7%), below the plan's required 95% retention. That test
was wrong, not the data — a working export register that reads exactly zero
on every measured day is a real signature (a battery run under a greedy
self-consumption or feed-in-limiting strategy leaves no export at all, see
[`battery_evidence_base.md`](battery_evidence_base.md) §1, §C), not a missing
meter. All 222 households have a confirmed-working register. `define_universe.py`
now admits register presence (`exp_days > 0`, any value) as `measured_export`
or `measured_zero_export`, retaining 222/222 = 100%.

Reproduce the checks:

```bash
python -m models.battery.build_labels
python -m models.battery.define_universe
```

The second command now passes the retention gate but still exits non-zero:
`build_universe()` needs `artifacts/meter_export_days.csv`, written by the
full streaming pass (Step 3), which has not been run —

```bash
python -m models.battery.build_features   # full run, ~77 GB, all 42 months
```

— after which `define_universe.py` can write the actual universe. Per the
implementation spec, aggregation, model training, evaluation, and population
scoring (Steps 4–7) are still unimplemented; nothing beyond the universe
gate has been unblocked by this fix.

The feature pass is available for development on a subset:

```bash
python -m models.battery.build_features --months 2025-07 2025-01
```

Its outputs are ignored under `models/battery/artifacts/`. Do not treat a
subset scan as a complete universe inventory.

