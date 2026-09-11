# Predicted heat-pump count vs the GWR register

## What is compared

The frozen champion of `run_classification.py` uses features that exist for the
342 labelled property samples only, so it cannot score the population. This
check therefore uses a **population-scoreable surrogate**: the same
`has_Waermepumpe` labels and the same model family, fitted on the PV pipeline's
household feature table (`models/pv/artifacts/features_gp.pkl`), which covers
80,056 households. Its validation accuracy is
**0.7879** with confusion matrix
`[[60, 8], [13, 18]]` on 99 labelled
households - a different feature set from the champion, so do not read this as
the champion's accuracy.

The reference is the BFS GWR building register for canton Aargau, primary
space-heating generator `GWAERZH1` in {7410 "Wärmepumpe für ein Gebäude",
7411 "Wärmepumpe für mehrere Gebäude"}, restricted to existing residential
buildings (`GSTAT=1004`, `GKAT` 1020/1030/1040).

## The counts

| Quantity | Value |
|---|---:|
| Unlabelled households scored | 79,727 |
| Flagged as heat pump (accuracy-optimal threshold 0.519) | 66,982 |
| Model flag rate | 84.0% |
| GWR heat-pump rate (AG existing residential buildings) | 31.3% |
| GWR heat-pump buildings | 49,124 of 157,047 |
| Expected households at the GWR rate | 24,938 |
| **Predicted / expected** | **2.69x** |

The model flags 2.7 times as many households as the
register implies. The threshold is the cause: it was chosen to maximise
accuracy on the GIGI labelled sample, which is a subsidy-applicant population
with a far higher heat-pump share than canton Aargau as a whole, and the fitted
probabilities on the population sit high (median
0.86). Flag counts by threshold:

| Threshold | Flag rate | Households |
|---:|---:|---:|
| 0.3 | 92.9% | 74,089 |
| 0.5 | 84.7% | 67,513 |
| 0.7 | 78.4% | 62,537 |
| 0.8 | 68.6% | 54,677 |
| 0.9 | 27.2% | 21,678 |
| 0.95 | 8.9% | 7,065 |

## Does the ranking agree with the register?

At the accuracy-optimal threshold the postcode agreement is **negative** - when
84% of everyone is flagged, the flag rate mostly measures how urban
a postcode is:

| Min households | Reference | Postcodes | Spearman ρ | p |
|---:|---|---:|---:|---:|
| 1 | gwr_hp_rate | 113 | -0.378 | 3.7e-05 |
| 1 | gwr_hp_rate_known | 113 | -0.377 | 3.9e-05 |
| 30 | gwr_hp_rate | 81 | -0.401 | 0.00021 |
| 30 | gwr_hp_rate_known | 81 | -0.400 | 0.00021 |
| 100 | gwr_hp_rate | 78 | -0.421 | 0.00013 |
| 100 | gwr_hp_rate_known | 78 | -0.420 | 0.00013 |

Forcing the same prevalence as the register (threshold
0.893, flagging exactly 31.3%) flips it
positive - the *ranking* does carry real geographic heat-pump signal:

| Min households | Postcodes | Spearman ρ | p |
|---:|---:|---:|---:|
| 1 | 113 | 0.071 | 0.46 |
| 30 | 81 | 0.313 | 0.0045 |
| 100 | 78 | 0.394 | 0.00035 |

## Reading

Use the score as a ranking calibrated to an external base rate, the way
`models/pv` calibrates to the ElPA register - not as a count at the
accuracy-optimal threshold. Caveats: GWR counts buildings while the model
scores metering customers (a multi-family building is one GWR row but several
customers), the smart-meter rollout is incomplete so the scored population is
not a random sample of the canton, and `has_Waermepumpe = 0` in training is
unlabelled rather than verified heat-pump-absent.
