# Battery detector

This package implements the first gated stages of
[`battery_mvp_plan.md`](battery_mvp_plan.md): label construction, solar
geometry, the one-pass import/export feature extractor, and the label-blind
PV-universe definition.

## Current status: blocked at the mandatory universe gate

The label gate passes all published checks. The universe gate cannot pass on
the current exports: only 208 of the 222 usable known-battery households have
any positive export energy across the complete prior scan (93.7%). Therefore
the stricter proposed rule, positive export on at least 10 days, cannot achieve
the plan's required 95% retention.

Reproduce the checks:

```bash
python -m models.battery.build_labels
python -m models.battery.define_universe
```

The second command intentionally exits non-zero before training. It does not
loosen the threshold or admit known positives by label. Per the implementation
spec, aggregation, model training, evaluation, and population scoring must not
proceed until the gate design or its acceptance criterion is explicitly
revised.

The feature pass is available for development:

```bash
python -m models.battery.build_features --months 2025-07 2025-01
```

Its outputs are ignored under `models/battery/artifacts/`. Do not treat a
subset scan as a complete universe inventory.

