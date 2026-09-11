# LightGBM plus obvious-EV heuristic

## Frozen extension

The selected positive-only override uses `plateau` events at least
3 kW above a daily local baseline, lasting at least
2 hours. Plateau events must have relative MAD
≤0.2, paired edges ≥0.5 kW, and edge symmetry ≥0.3. A property is overridden only
when weekly evidence reaches 0.90 in at least
2 of its latest 12 eligible weeks. Its LightGBM
score is then raised to at least 0.9; all other scores are
unchanged. The rule was selected from the bounded grid on validation only.

## Validation

| Metric | LightGBM | LightGBM + heuristic | Change |
|---|---:|---:|---:|
| Accuracy | 0.7313 | 0.7463 | +0.0149 |
| Balanced accuracy | 0.7513 | 0.7871 | +0.0357 |
| PR-AUC | 0.5884 | 0.6539 | +0.0655 |
| Precision | 0.4231 | 0.4444 | +0.0214 |
| Recall | 0.7857 | 0.8571 | +0.0714 |

Validation confusion matrices changed from `[[38, 15], [3, 11]]` to
`[[38, 15], [2, 12]]`. The extension is **accepted**
under the predeclared validation rule.

Combined-model validation confusion matrix:

| Actual class | Correctly predicted | Incorrectly predicted |
|---|---:|---:|
| Non-EV/unlabelled (53) | **71.7% non-EV** (38) | 28.3% EV (15 false alarms) |
| EV-labelled (14) | **85.7% EV** (12) | 14.3% non-EV (2 missed EVs) |

## Reused-test diagnostic

The test partition had already been opened by the original `_total` run. These numbers
are useful as a stability check, but they are not a fresh unbiased estimate.

| Metric | LightGBM | LightGBM + heuristic | Change |
|---|---:|---:|---:|
| Accuracy | 0.6269 | 0.6119 | -0.0149 |
| Balanced accuracy | 0.6410 | 0.6314 | -0.0096 |
| PR-AUC | 0.3890 | 0.3482 | -0.0407 |
| Precision | 0.3333 | 0.3226 | -0.0108 |
| Recall | 0.6667 | 0.6667 | +0.0000 |

Reused-test confusion matrices changed from `[[32, 20], [5, 10]]` to
`[[31, 21], [5, 10]]`. Because zeros are unlabelled rather than
confirmed negatives, even a strong plateau is evidence for review rather than proof of
EV ownership. A future accuracy claim needs a newly labelled untouched cohort.

Combined-model reused-test confusion matrix:

| Actual class | Correctly predicted | Incorrectly predicted |
|---|---:|---:|
| Non-EV/unlabelled (52) | **59.6% non-EV** (31) | 40.4% EV (21 false alarms) |
| EV-labelled (15) | **66.7% EV** (10) | 33.3% non-EV (5 missed EVs) |

In practical terms, the combined model found about two-thirds of the EV-labelled
test properties, while falsely flagging about two-fifths of the non-EV/unlabelled
properties.

## Recommendation

Keep `E22_lgbm_total` as the deployed classifier. The override found one additional
labelled EV in validation, but on the reused test it changed one unlabelled property
into a false positive and recovered no EV. That is too little and too unstable to
justify changing production predictions. Retain `obvious_ev_override` and its weekly
event rows as inspectable review evidence, then test this frozen rule on a newly
labelled cohort before allowing it to modify the model score.
