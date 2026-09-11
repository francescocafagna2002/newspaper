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

**Do not "fix" this by relaxing the export rule** — that was tried on
2026-09-11 and is wrong; see the investigation in [`REPORT.md`](REPORT.md).
In short: every meter in the export files has OBIS 2.29 rows (almost all
exact zeros), so admitting "the register was read at all" is true for 100%
of meters, admits the entire 93k population, and destroys the PV-likely
universe that plan decision D3 depends on. The 14 excluded households also
score `pv_probability` 0.012–0.482, so they show no PV signature on the
import side either.

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

Steps 4–7 (`build_features_agg.py`, `train.py`, `evaluate.py`, `score.py`,
`pipeline.py`) were written on 2026-09-11 so the pipeline is ready to run the
moment that decision is made. **They have never been executed against real
data** — the gate stops before Step 3's output exists — so treat them as
unvalidated code, not as a working model.

The feature pass is available for development on a subset:

```bash
python -m models.battery.build_features --months 2025-07 2025-01
```

Its outputs are ignored under `models/battery/artifacts/`. Do not treat a
subset scan as a complete universe inventory.

