# Heuristic extension 2: steep-edge cascade

## Finding

A steep slope makes physical sense as part of an EV charging signature, but no tested
direction improved validation balanced accuracy. The best positive-rescue candidate
was **rejected**, so the deployed direction remains “none” and LightGBM is unchanged.

The diagnostic rule required at least three events across two weeks. Each event began
with a rise of at least 4 kW in one 15-minute step, continued for at least two hours
above a 5 kW local-baseline residual, formed a stable plateau, and had a paired stop
edge.

## Direction test on validation

| Direction | Non-EV correct | EV detected | Accuracy | Balanced accuracy | Precision | Predictions changed |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM, unchanged | **71.7%** (38/53) | **78.6%** (11/14) | 73.1% | **75.1%** | 42.3% | 0 |
| Positive rescue | 56.6% (30/53) | **78.6%** (11/14) | 61.2% | 67.6% | 32.4% | 8 |
| Negative veto | **83.0%** (44/53) | 50.0% (7/14) | **76.1%** | 66.5% | 43.8% | 10 |
| Two-way correction | 58.5% (31/53) | 57.1% (8/14) | 58.2% | 57.8% | 26.7% | 22 |

Positive rescue changed eight LightGBM-negative validation properties, but all eight
were in the non-EV/unlabelled class. It recovered no false negative. Negative veto had
the highest ordinary accuracy because it favored the 53-property majority class, but
it missed half of the EV-labelled properties and reduced balanced accuracy sharply.
This confirms that ordinary accuracy is misleading for choosing the direction.

## Best heuristic candidate diagnostic

| Metric | Validation base | Validation candidate | Change | Reused-test base | Reused-test candidate | Change |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 73.1% | 61.2% | −11.9 pp | 62.7% | 55.2% | −7.5 pp |
| Balanced accuracy | 75.1% | 67.6% | −7.5 pp | 64.1% | 59.3% | −4.8 pp |
| PR-AUC | 0.5884 | 0.5379 | −0.0505 | 0.3890 | 0.3604 | −0.0285 |
| Precision | 42.3% | 32.4% | −9.9 pp | 33.3% | 28.6% | −4.8 pp |
| EV recall | 78.6% | 78.6% | 0.0 pp | 66.7% | 66.7% | 0.0 pp |

On the reused test cohort, the diagnostic positive rescue changed five predictions;
all five were non-EV/unlabelled. It recovered no EV. Non-EV correctness fell from
61.5% (32/52) to 51.9% (27/52), while EV detection stayed at 66.7% (10/15).

The reused-test comparison is a stability diagnostic rather than a fresh unbiased
estimate. No parameter was changed after the extension-2 freeze.

## Recommendation

Do not apply the steep-edge cascade to the classifier. Keep the edge evidence as an
explanation field for manual review. A paired rise, plateau, and fall is a reasonable
charging-event hypothesis, but it is not specific enough to infer EV ownership in
this cohort.
