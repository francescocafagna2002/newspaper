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

## Update — 2026-09-11: gate re-investigated, conclusion unchanged

An attempt was made to unblock the gate by relaxing its export rule. **It was
wrong and has been reverted.** Recording it here so it is not retried.

The hypothesis was that the gate conflated "export register never read" with
"register read every day, always zero," and that the latter is a real
archetype — a greedy self-consumption or feed-in-limited battery leaves no
export signature (`battery_evidence_base.md` §1, §C). The proposed fix
admitted any household whose register was read at all (`exp_days > 0`)
instead of one that ever read positive (`exp_kwh > 0`). Retention duly went
from 208/222 to 222/222.

Three measurements show that "fix" is vacuous:

1. **Every meter has 2.29 rows.** Of 93,279 meters in the completed scan,
   93,279 have `exp_days > 0` and **zero** have `exp_days == 0`. The export
   register is reported for everyone, almost always as exact zeros. So
   `exp_days > 0` is a constant, not a criterion: it admits the entire
   population, retention is 100% by construction, and the gate stops gating.
   It would also have made the pre-existing `pv_probability` fallback branch
   dead code, since that branch keys on `exp_days == 0`.
2. **That destroys the universe's purpose.** Admitting all ~93k meters is
   precisely the failure mode plan decision D3 exists to prevent: 96% of
   battery households also have PV, so a PU model over the general population
   relearns "has PV" and stops. The universe must stay PV-restricted.
3. **`exp_kwh > 0` is a well-calibrated PV proxy, and the 14 households fail
   it for real.** Positive export covers 11.3% of meters, matching the ~12%
   ElPA PV base rate; `pv_probability` separates exporters from non-exporters
   at ROC-AUC 0.904 (mean 0.831 vs 0.199). The 14 rejected households score
   `pv_probability` 0.012–0.482 — none above 0.5, let alone the 0.8 fallback
   threshold — so they show no PV signature on the **import** side either.
   A self-consuming PV+battery household would still suppress midday import
   and score high. These do not. They are unobservable in this data (most
   likely the generation/storage sits behind a meter outside the household's
   mapped set), not mismeasured.

The original conclusion therefore stands: the gate reflects a genuine data
limitation, and unblocking the MVP still requires the plan owner to choose a
revised label-blind universe rule or a lower acceptance criterion. `exp_kwh >
0` and the `>= MIN_EXPORT_DAYS` rule are restored, and
`test_universe_gate_upper_bound_is_below_required_retention` pins 208/222 with
a comment pointing here.

One thing did change: Steps 4–7 (`build_features_agg.py`, `train.py`,
`evaluate.py`, `score.py`, `pipeline.py`) were written, so the pipeline can
run as soon as that decision is made. `build_features_agg.py` implements
Pettitt's change-point test on the daily series, `train.py` the Elkan–Noto PU
model with GroupKFold on `gp_nr` and the π ∈ {0.20, 0.30, 0.40} sweep,
`evaluate.py` audit sets A and B with bootstrap CIs and
`average_precision_score` for PR-AUC. **None of it has been run against real
data** — Step 3's output does not exist — so it is unvalidated code. There is
still no trained model, no scored population, and no defensible headline
metric.

