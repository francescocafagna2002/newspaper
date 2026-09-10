# Evidence base — why these features and this model architecture (battery detection)

Companion to [`battery_mvp_plan.md`](battery_mvp_plan.md).
Energy Data Hackdays 2026 · AEW "Energy Fingerprints" · 2026-09-10

Every design choice in the battery plan is listed here with the source it rests
on. The point is that a reviewer can check *why* a feature is in the model
without reading the code, and can see where we are extrapolating beyond what
anyone has published.

**How to read the reference marks**

- ✔ — bibliographic details (journal, volume, pages or DOI) verified against a
  publisher or index page while drafting.
- ○ — standard, widely cited reference quoted from working knowledge; the claim
  is solid but re-check the exact volume/pages before external publication.
- ⚠ — not peer-reviewed (patent, industry report). Used only where no
  peer-reviewed equivalent exists, and always labelled as such in the text.

---

## 1. The physical model the features encode

A behind-the-meter battery is not a load. It is an energy *shifter* sitting
between the PV inverter and the meter, and almost every residential unit in
Central Europe runs the same "maximise self-consumption" (greedy) controller:
charge whenever PV exceeds household load, discharge whenever load exceeds PV,
subject to state-of-charge and inverter power limits [R1 ✔, R2 ✔].

Three consequences follow, and they are the whole basis of the feature set:

1. **Self-consumption rises sharply.** A battery sized at 0.5–1 kWh per
   installed kWp raises relative self-consumption by **13–24 percentage
   points** over PV alone [R1 ✔]. On a net meter that appears as less export
   *and* less import, without the household consuming less.
2. **Export is delayed, clipped and truncated.** With greedy charging, surplus
   goes to the battery before the grid, so first export of the day is late and
   the export peak is flattened; operating strategies that additionally cap
   feed-in reshape it further [R2 ✔].
3. **Import stops after sunset until the battery empties, then resumes in a
   step.** Discharge covers the evening load exactly, so measured import sits
   at ≈0 and then jumps to full base load in a single interval when the battery
   hits its lower state-of-charge limit.

Not all of the shifted energy comes back: measured AC-to-AC round-trip
efficiency of residential PV home-storage systems is roughly **0.80–0.95**,
degrading badly at partial load, with non-trivial standby consumption
[R3 ✔]. That asymmetry is what makes the energy-balance feature (§2 G)
a *falsifiable physical check* rather than another correlation.

---

## 2. Feature families and their sources

### A/B — Zero-import plateaus and constant-power ("flat") intervals

The strongest published statement of this signature comes from work on
**unsupervised identification of behind-the-meter battery storage**, which
detects storage by finding *flat intervals — net power constant within a
tolerance band over a minimum continuous duration* — and derives location,
session boundaries and system size from them [R5 ⚠]. Our `flat_interval_share`,
`both_zero_share`, `zero_run_after_sunset` and the step-detection features are
the 15-minute-energy analogue of that criterion.

The same idea underlies the disaggregation literature: BTM battery
charge/discharge is separable from net load precisely because it is piecewise
near-constant and anti-correlated with the PV/load residual [R6 ✔, R7 ✔].
Those papers solve the harder problem (recover the battery power *series*); we
use the same structure only as evidence that the plateau is a real, published
signature and not a hunch.

Note the direction of difficulty, stated explicitly in that literature: a
battery **smooths away** the net-load fluctuations that PV disaggregation
methods rely on [R6 ✔]. What hurts PV disaggregation helps us — it is the
detectable feature (see family D).

### C — Feed-in shaping

*(Requires the feed-in register as a model input — plan decision D1.)*
Grounded in the same operating-strategy work as §1: greedy charging delays and
flattens feed-in, and feed-in-limiting strategies clip it outright [R2 ✔]. The
self-consumption ratio is the standard summary indicator for this effect
[R1 ✔]. The German market reviews document that self-consumption maximisation
is the dominant control mode in the residential segment, which is what lets us
assume one controller family for the whole population [R4 ✔].

### D — Ramp smoothing

Direct corollary of [R6 ✔] (batteries absorb cloud-transient fluctuations).
Ramp statistics are standard practice in load monitoring generally, where edges
and transitions rather than levels carry the appliance identity.

### E — Grid charging vs ripple-controlled loads

Both produce rectangular blocks of import. The discriminator we use — a
ripple-controlled boiler starts at the *same clock time* every night, a battery
top-up does not — rests on timing regularity rather than shape, and is the kind
of confound-control feature the household-characteristics literature found
necessary when inferring appliance ownership from load shape [R8 ✔].

### F — Change-point detection on the daily series

For "when did the battery appear" we use a non-parametric change-point test on
the daily `evening_zero_share` series: Pettitt's non-parametric test for a
single change point [R13 ○]. This is a deliberate correction: the PV model's change-point feature
differenced *year-aggregated* midday ratios and correlated at Spearman 0.05
with the true commissioning date (see [`../models/REPORT.md`](../models/REPORT.md)).
Change-point tests are designed for exactly this and operate on the daily
resolution where the shift is actually visible.

### G — Energy balance and round-trip efficiency

Charge and discharge energies must satisfy `E_discharge ≈ η · E_charge` with
η ≈ 0.80–0.95 [R3 ✔]. Households whose apparent "charge" and "discharge"
volumes fail that band are almost certainly showing something else (an EV, an
absence, a metering artefact). This gives an interpretable accept/reject test
per household, which is what the challenge means by "inspectable evidence".

### H — Feature engineering over raw curves

Using a compact set of engineered, physically named statistics — rather than
feeding raw 96-point days to a network — is what the household-characteristics
literature does, achieving useful accuracy on ~4,200 Irish households with
hand-built consumption features and standard classifiers [R8 ✔]. It is also
what keeps the "explainability" criterion of the challenge satisfiable.

---

## 3. Model architecture

### 3.1 Gradient-boosted trees as the primary model

The training table is small (a few hundred labelled households), tabular,
heterogeneous in scale, and full of informative missingness (a household with
no summer data). A systematic benchmark across 45 tabular datasets found
tree-based ensembles still outperform deep learning in exactly this regime —
medium-sized, heterogeneous, uninformative-feature-heavy tabular data — and
identifies the inductive biases responsible [R9 ○]. Histogram-based boosting
(scikit-learn's `HistGradientBoostingClassifier`) also handles NaNs natively, which matters
because "no summer months on record" is a real and informative state here.

That is also what the PV model used, so the two asset models stay comparable.

### 3.2 The MiniRocket branch

For the *shape* half of the signal (the evening plateau-and-step curve), the
current state of the art in general-purpose time-series classification is the
random convolutional kernel family, of which MiniRocket [R10 ✔] is the
practical representative: it reaches accuracy comparable to far more expensive
deep and meta-ensemble methods at a tiny fraction of the cost, which the most
recent comparative re-evaluation of the field bears out [R11 ○].

MiniRocket is chosen over InceptionTime/HC2 for three reasons that matter here:
it is CPU-only and fast enough to run over 80k households on this hardware; it
needs no GPU or extra dependency (it is ~100 lines of numpy); and with a few
hundred labelled examples a deep model would overfit while a linear head on
fixed random features degrades gracefully. It enters as a *stacked branch* on
three canonical 96-point curves per household, not as a replacement — if it
does not beat the engineered GBDT out of fold, it is dropped and the report says
so.

### 3.3 Positive-Unlabeled learning (the primary setting)

The only candidate negatives are subsidy applicants who did not tick "battery" —
plausible but unverified, and a household that later bought an unsubsidised
battery would silently poison the negative class. Per decision D2 in the plan
they are excluded from training, leaving known positives against an unlabelled
remainder: the Elkan–Noto case-control estimator [R12 ○], the same machinery
the PV model was forced into.

Two things are worth stating precisely, because the whole result rests on them.

**The prior matters more here than it did for PV.** Elkan–Noto recovers
P(y=1 | x) only up to the label frequency `c`, and the population shift that
follows needs a prior π. Because the unlabelled pool is restricted to PV
households (plan D3), π is P(battery | PV) ≈ 0.2–0.4 rather than a population
rate of a few percent — a far better-conditioned problem than the PV model's,
where a 12% anchor had to carry the entire absolute scale [see §4].

**SCAR is the assumption, and it is shaky.** Elkan–Noto assumes labelled
positives are *selected completely at random* among positives. Ours are subsidy
applicants: skewed towards recent, larger and grant-eligible installations. The
PU survey [R14 ○] sets out how SCAR failure biases both `c` and the resulting
ranking; bagging-PU [R15 ○] is the standard alternative when the label-frequency
estimate is unstable, and is run as a cross-check. The unlabelled pool is
contaminated by construction at rate π — that is the definition of the setting,
not a defect, but it is why nothing measured against it is an accuracy.

The consequence for reporting: no number produced inside the PU loop is an
accuracy. Real discrimination is measured only on the two held-out audit sets
(plan §6.2).

### 3.4 Calibration

Two stages, as in the PV model: the Elkan–Noto label frequency `c` recovers the
scale up to a constant, then a logit shift matches the mean over the unlabelled
pool to π (§4). Boosted trees are not calibrated out of the box, and a
probability that is only good for ranking should be labelled as such.

### 3.5 Validation protocol

`GroupKFold` on `gp_nr` because one household contributes several
`(household, year)` rows; nested CV for any hyper-parameter selection, since
selecting on the same folds you report gives an optimistically biased estimate.
On the audit sets, **PR-AUC leads and ROC-AUC is quoted next to it**: the
positive class is a minority even inside the PV subgroup, and ROC curves flatter
imbalanced problems because the false-positive *rate* divides by a large
negative count [R16 ○]. Three caveats keep this from being mistaken for a
methodological advance — it is a reporting convention, not a frontier method:

1. **Compute it as average precision, not as a trapezoid.** Linear
   interpolation between points in PR space is not achievable by any classifier
   and inflates the area [R17 ○]; the step-wise estimator
   (`sklearn.metrics.average_precision_score`) is the correct one. The PV
   pipeline's `evaluate.py` currently integrates its PU-PR curve with
   `np.trapezoid` — worth fixing when that code is reused here.
2. **PR-AUC is prevalence-dependent**, so it cannot be compared across groups
   with different base rates. Within-PV and whole-population numbers are not
   commensurable and must not be shown as if they were.
3. **Neither AUC says anything about calibration.** Both are rank statistics; a
   model can order households perfectly and still be badly wrong about the
   probability. Report a calibration curve, and — because AEW's actual use is
   "inspect the top N households" — **precision@k** at the k values a utility
   would plausibly work through, which is the number that translates directly
   into effort wasted.

**The number that goes in front of an audience is accuracy**, and specifically
**balanced accuracy** at the operating threshold, `(sensitivity +
specificity) / 2`. It is accuracy with the class imbalance divided out, chance
sits at 50% by construction, and it needs no explanation of what an "area under
a curve" is. Plain accuracy may be shown as well, but never without its
no-skill baseline printed beside it — on audit set A that is 80.0% (call
everything a battery), and on the full population 95–98% (call nothing one).

Per-household bootstrap CIs on every headline number — 53 audit controls is a
small denominator.

### 3.6 Explainability

SHAP values [R18 ○] for the global feature ranking and the per-household top-3
drivers, plus the plain-language evidence strings and annotated day plots of the
PV pipeline. TreeSHAP, the exact variant for tree ensembles, is what the
implementation uses.

### 3.7 Feeding one model's output into another

The battery model takes `pv_probability` from the PV model (plan D5). Two
well-established cautions govern how:

**Use out-of-fold predictions, never in-sample ones.** Stacking one model's
output as another's input is standard practice, but the stacking literature is
explicit that the upstream predictions must be produced on data the upstream
model did not train on, or the downstream model learns the upstream model's
memorisation rather than its skill [R19 ○]. The PV pipeline already
writes `artifacts/model/oof_predictions.csv`; that is the column to use.

**A predicted regressor is not a measured one.** Substituting an estimate for
an unobserved variable leaves the downstream model's uncertainty understated —
the long-known generated-regressor problem. Practically, this is the argument for coarsening `pv_probability` to
three bins and for preferring directly measured PV-size proxies wherever they
exist: a quantity we can read off the register carries no estimation error at
all.

**And choose the control variable by its causal role, not its correlation.**
Annual export energy correlates strongly with PV size, which makes it look like
a good control — but a battery *reduces* export energy, so it sits on the causal
path between the thing we are detecting and the thing we measure. Conditioning
on such a mediator removes part of the effect of interest; the taxonomy of good
and bad controls covers exactly this case [R20 ○]. Clear-day *peak* export power
is the safer proxy, since a full battery still lets the array export at close to
full power on a sunny afternoon.

---

## 4. Base rates and priors (Swiss context)

Under plan decision D3 the unlabelled pool is PV households, so the quantity
needed is **π = P(battery | PV)**, not a whole-population rate. No public
register lists home batteries the way the BFE ElPA register lists PV plants, so
the anchor is indirect:

- **Swissolar's Batteriemonitor** reports that ~42% of *newly installed*
  single-family-home PV systems in 2023 were combined with storage (18% for
  multi-family buildings), that behind-the-meter storage capacity in
  Switzerland is growing from ~1.5 to ~2.5 GWh year on year, and that a typical
  15 kWh home system cost ~CHF 8,800 in 2025 [R21 ⚠].
- Those are *new-installation* shares. The installed **stock** of PV systems in
  canton AG is dominated by pre-2020 installations, most without storage, so
  π is materially lower than 42% — plan on **π ∈ [0.20, 0.40]** and
  sensitivity-test across it.
- Cross-check only: combined with the PV model's ElPA-derived AG PV rate of
  ≈11–12% ([`../models/REPORT.md`](../models/REPORT.md)), the implied
  whole-population battery rate is roughly **2–5%**.
- The German market review [R4 ✔] gives an independent order-of-magnitude
  cross-check on adoption and on the dominance of self-consumption operation.

Treat all of these as priors for calibration and sanity checks, never as labels.

---

## 5. Where we are extrapolating beyond the literature

Stated plainly, because the challenge is scored partly on honesty about limits:

- Published BTM-battery work uses **1-minute to 1-second net power**; we have
  **15-minute energy**. Sub-interval charge/discharge switching is invisible to
  us, and short cycles will be smeared. No published accuracy figure at 15-min
  resolution transfers directly to this setting — expect worse.
- Published work mostly does **disaggregation** (recover the battery series)
  and evaluates with RMSE against sub-metered ground truth; we do **binary
  presence classification** against subsidy records. The features are borrowed
  from that work; the reported numbers cannot be compared to it.
- The **seasonal-contrast** framing (summer-minus-winter within household) and
  the **`(household, year)` prediction unit with a paired retrofit test** are
  our own design, not something we can cite. They are defensible from the
  physics of §1 and are validated internally against 105 known install dates —
  which is the right way to defend them.
- Swiss home-battery prevalence rests on an **industry report**, not a
  peer-reviewed source or a register. It is a lower-confidence input, and the
  absolute probability scale inherits that uncertainty.

---

## 6. References

Trimmed to the sources a design decision actually rests on: if one of these were
wrong, something in the plan would have to change. Peer-reviewed work is
preferred; the two non-peer-reviewed entries are kept only because no
peer-reviewed equivalent exists, and are marked.

**Battery and PV physics / operation**

- **[R1] ✔** Luthander, R., Widén, J., Nilsson, D., Palm, J. (2015).
  *Photovoltaic self-consumption in buildings: A review.* Applied Energy 142,
  80–94. https://www.sciencedirect.com/science/article/abs/pii/S0306261914012859
  → self-consumption rises 13–24 pp with storage (family C).
- **[R2] ✔** Weniger, J., Tjaden, T., Quaschning, V. (2014). *Sizing of
  residential PV battery systems.* Energy Procedia 46, 78–87.
  doi:10.1016/j.egypro.2014.01.160
  → greedy charging delays and flattens feed-in (families A, C).
- **[R3] ✔** Munzke, N., Schwarz, B., Büchle, F., Hiller, M. (2021).
  *Evaluation of the efficiency and resulting electrical and economic losses of
  photovoltaic home storage systems.* Journal of Energy Storage 33, 101724.
  → the 0.80–0.95 round-trip band the energy-balance check tests against (family G).
- **[R4] ✔** Figgener, J., Stenzel, P., et al. (2020). *The development of
  stationary battery storage systems in Germany – A market review.* Journal of
  Energy Storage 29, 101153.
  → self-consumption maximisation is the dominant residential control mode, which
  is what lets us assume one controller family for the whole population.

**Behind-the-meter detection**

- **[R5] ⚠** *Method and system for unsupervised identification of
  behind-the-meter battery energy storage systems.* US Patent 12,562,572.
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12562572
  → the flat/constant-power interval criterion (family B). A patent, not
  peer-reviewed; kept because it is the only explicit statement of this rule.
- **[R6] ✔** Chen, X., Ardakanian, O. (2022). *Data efficient energy
  disaggregation with behind-the-meter energy resources.* Sustainable Energy,
  Grids and Networks 32, 100813. doi:10.1016/j.segan.2022.100813
  → a battery smooths the net-load fluctuations PV methods rely on (family D).
- **[R7] ✔** Wang, F., Ge, X., Dong, Z., et al. (2022). *Joint Energy
  Disaggregation of Behind-the-Meter PV and Battery Storage: A Contextually
  Supervised Source Separation Approach.* IEEE Transactions on Industry
  Applications 58(2), 1490–1501. https://ieeexplore.ieee.org/document/9684963/
  → PV and battery are separable from net load at all (families B, G).
- **[R8] ✔** Beckel, C., Sadamori, L., Staake, T., Santini, S. (2014).
  *Revealing household characteristics from smart meter data.* Energy 78,
  397–410. https://vs.inf.ethz.ch/publ/papers/beckel-2014-energy.pdf
  → engineered features plus standard classifiers work for this task shape.

**Modelling**

- **[R9] ○** Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). *Why do
  tree-based models still outperform deep learning on typical tabular data?*
  NeurIPS 2022 Datasets and Benchmarks Track. arXiv:2207.08815
  → why GBDT is the primary model.
- **[R10] ✔** Dempster, A., Schmidt, D.F., Webb, G.I. (2021). *MiniRocket: A Very
  Fast (Almost) Deterministic Transform for Time Series Classification.*
  KDD '21, 248–257. arXiv:2012.08791
  → the shape branch.
- **[R11] ○** Middlehurst, M., Schäfer, P., Bagnall, A. (2024). *Bake off redux:
  a review and experimental evaluation of recent time series classification
  algorithms.* Data Mining and Knowledge Discovery. arXiv:2304.13029
  → why MiniRocket rather than a deep or meta-ensemble alternative.
- **[R12] ○** Elkan, C., Noto, K. (2008). *Learning classifiers from only
  positive and unlabeled data.* KDD '08, 213–220.
  → the PU estimator itself.
- **[R13] ○** Pettitt, A.N. (1979). *A non-parametric approach to the
  change-point problem.* Journal of the Royal Statistical Society: Series C
  28(2), 126–135.
  → the change-point feature (family F).
- **[R14] ○** Bekker, J., Davis, J. (2020). *Learning from positive and unlabeled
  data: a survey.* Machine Learning 109, 719–760.
  → the SCAR assumption and what its failure costs.
- **[R15] ○** Mordelet, F., Vert, J.-P. (2014). *A bagging SVM to learn from
  positive and unlabeled examples.* Pattern Recognition Letters 37, 201–209.
  → the PU robustness cross-check.

**Evaluation, explanation, and using an upstream model's output**

- **[R16] ○** Saito, T., Rehmsmeier, M. (2015). *The precision-recall plot is
  more informative than the ROC plot when evaluating binary classifiers on
  imbalanced datasets.* PLoS ONE 10(3), e0118432.
  → why PR-AUC leads on the audit sets.
- **[R17] ○** Davis, J., Goadrich, M. (2006). *The relationship between
  Precision-Recall and ROC curves.* ICML '06, 233–240.
  → why PR curves must not be interpolated linearly, and how PR and ROC relate.
- **[R18] ○** Lundberg, S.M., Lee, S.-I. (2017). *A unified approach to
  interpreting model predictions.* NeurIPS 30.
  → SHAP; TreeSHAP is the exact variant for this model family.
- **[R19] ○** Wolpert, D.H. (1992). *Stacked generalization.* Neural Networks
  5(2), 241–259.
  → why `pv_probability` must be taken out-of-fold.
- **[R20] ○** Cinelli, C., Forney, A., Pearl, J. (2024). *A crash course in good
  and bad controls.* Sociological Methods & Research 53(3), 1071–1104.
  → why export *energy* is a bad control and peak export power is not.

**Swiss market context**

- **[R21] ⚠** Swissolar, *Batteriemonitor Schweiz* (2025 and 2026 editions).
  https://www.swissolar.ch/de/markt-und-politik/markt-schweiz/batteriebericht
  → the only available anchor for π = P(battery | PV). An industry report, not
  peer-reviewed and not a register; the absolute probability scale inherits that.
