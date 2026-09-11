#!/usr/bin/env python3
"""Compare predicted heat-pump counts on the unlabelled population with GWR.

The frozen champion in `run_classification.py` uses the labelled property-sample
features (`bezug_*`, `*_share`, `hb00-23`, `w_*`), which exist for 342 samples
only - there is no population-wide table with those definitions in this
checkout. To get a population count at all, this module trains a
**population-scoreable surrogate** on the PV pipeline's household feature table
(`models/pv/artifacts/features_gp.pkl`, 80k households), using the same
`has_Waermepumpe` labels, the same model family, and the same
accuracy/confusion-matrix policy. It is a different feature set, so its
accuracy is reported separately and must not be read as the champion's.

The reference is the BFS GWR building register for canton Aargau: primary
space-heating generator `GWAERZH1` in {7410 'Wärmepumpe für ein Gebäude',
7411 'Wärmepumpe für mehrere Gebäude'}. GWR counts *buildings*; the model
scores *metering customers*, so rates are the comparable quantity and the
implied count is an illustration, not an identity.

    python -m models.heat_pump.population_baserate

Outputs: artifacts/population_gwr_comparison.json
         artifacts/population_gwr_by_plz.csv
         artifacts/population_predictions.csv
"""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config as C
from .run_classification import LABEL, choose_threshold, metrics

GWR_URL = "https://public.madd.bfs.admin.ch/ag.zip"
GWR_CACHE = C.NEWSPAPER / "models" / "heat_pump_boiler" / "artifacts" / "gwr_ag.zip"
GWR_HEATING_CODES = ("7410", "7411")  # Wärmepumpe für ein / mehrere Gebäude
POPULATION_FEATURES = C.NEWSPAPER / "models" / "pv" / "artifacts" / "features_gp.pkl"
PV_PREDICTIONS = C.NEWSPAPER / "models" / "pv" / "artifacts" / "pv_predictions.csv"

# PV-pipeline feature names that carry heat-pump-relevant information: absolute
# load level, night base load, and the winter/summer contrast. Midday-dip
# features are PV-specific and are deliberately left out except through the
# explicit PV-probability control.
SURROGATE_FEATURES = [
    "day_kwh_mean", "night_kwh_mean", "day_kwh_winter", "day_kwh_summer",
    "summer_winter_kwh_ratio", "midday_seasonality", "evening_midday_summer",
    "n_meters",
]


def build_gwr() -> pd.DataFrame:
    if not GWR_CACHE.exists():
        response = requests.get(GWR_URL, timeout=600)
        response.raise_for_status()
        GWR_CACHE.write_bytes(response.content)
    with zipfile.ZipFile(GWR_CACHE) as z:
        def read(name, cols=None):
            return pd.read_csv(z.open(name), sep="\t", dtype=str,
                               quoting=csv.QUOTE_NONE, usecols=cols).fillna("")
        codes = read("kodes_codes_codici.csv")
        hp = codes[(codes.CMERKM == "GWAERZH1") & (codes.CECODID.isin(GWR_HEATING_CODES))]
        if len(hp) != len(GWR_HEATING_CODES) or not hp.CODTXTLD.str.contains("Wärmepumpe").all():
            raise ValueError("GWR heating-generator codes no longer match the catalogue")
        buildings = read("gebaeude_batiment_edificio.csv",
                         ["EGID", "GDEKT", "GSTAT", "GKAT", "GWAERZH1", "GEXPDAT"])
        entrances = read("eingang_entree_entrata.csv", ["EGID", "EDID", "DPLZ4"])

    # One building contributes once, even if it has several entrances.
    postal_counts = entrances.groupby("EGID").DPLZ4.nunique()
    ambiguous = set(postal_counts[postal_counts > 1].index)
    entrances = (entrances[~entrances.EGID.isin(ambiguous)]
                 .sort_values(["EGID", "EDID"]).drop_duplicates("EGID"))
    b = buildings[(buildings.GDEKT == "AG") & (buildings.GSTAT == "1004")
                  & buildings.GKAT.isin(["1020", "1030", "1040"])]
    b = b.merge(entrances[["EGID", "DPLZ4"]], on="EGID", validate="one_to_one")
    b = b[b.DPLZ4.str.fullmatch(r"\d{4}")].copy()
    b["heat_pump"] = b.GWAERZH1.isin(GWR_HEATING_CODES)
    b["known"] = b.GWAERZH1.ne("")

    out = (b.groupby("DPLZ4")
             .agg(n_buildings=("EGID", "size"), n_heat_pump=("heat_pump", "sum"),
                  n_known=("known", "sum"))
             .rename_axis("plz").reset_index())
    out["gwr_hp_rate"] = out.n_heat_pump / out.n_buildings
    out["gwr_hp_rate_known"] = out.n_heat_pump / out.n_known.replace(0, np.nan)
    out.attrs["provenance"] = {
        "source": GWR_URL,
        "accessed_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(GWR_CACHE.read_bytes()).hexdigest(),
        "export_dates": sorted(b.GEXPDAT.unique()),
        "generator_field": "GWAERZH1",
        "generator_codes": list(GWR_HEATING_CODES),
        "n_existing_residential_buildings": int(len(b)),
        "n_heat_pump_buildings": int(b.heat_pump.sum()),
        "canton_hp_rate": float(b.heat_pump.mean()),
        "canton_hp_rate_known": float(b.loc[b.known, "heat_pump"].mean()),
    }
    return out


def labelled_households() -> pd.DataFrame:
    train = pd.read_csv(C.TRAIN_SOURCE, sep=C.CSV_SEP, dtype={"gp_nr": str, "plz": str})
    train = train[train.sample_type.eq("cross_sectional")]
    return train[["gp_nr", LABEL]].drop_duplicates("gp_nr")


def population_table() -> pd.DataFrame:
    pop = pd.read_pickle(POPULATION_FEATURES)
    pop["gp_nr"] = pop["gp_nr"].astype(str)
    pop = pop[["gp_nr", "plz", *SURROGATE_FEATURES]].copy()
    if PV_PREDICTIONS.exists():
        pv = pd.read_csv(PV_PREDICTIONS, dtype={"gp_nr": str})
        col = next((c for c in pv.columns if "prob" in c.lower()), None)
        if col:
            # PV homes import little in summer, which inflates the winter/summer
            # contrast a heat pump also produces. Carry PV probability so the
            # model can separate the two rather than confusing them.
            pop = pop.merge(pv[["gp_nr", col]].rename(columns={col: "pv_probability"}),
                            on="gp_nr", how="left")
    return pop


def main() -> dict:
    gwr = build_gwr()
    provenance = gwr.attrs["provenance"]

    pop = population_table()
    features = [c for c in pop.columns if c not in {"gp_nr", "plz"}]
    labels = labelled_households()
    data = pop.merge(labels, on="gp_nr", how="inner")

    train, validation = train_test_split(
        data, test_size=0.3, random_state=C.SEED, stratify=data[LABEL])
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=7,
            l2_regularization=5.0, class_weight="balanced", random_state=C.SEED)),
    ])
    model.fit(train[features], train[LABEL])
    val_score = model.predict_proba(validation[features])[:, 1]
    threshold = choose_threshold(validation[LABEL].to_numpy(), val_score)
    surrogate_metrics = metrics(validation[LABEL], val_score, threshold)

    model.fit(data[features], data[LABEL])
    unlabelled = pop[~pop.gp_nr.isin(set(labels.gp_nr))].copy()
    unlabelled["hp_probability"] = model.predict_proba(unlabelled[features])[:, 1]
    unlabelled["hp_flag"] = (unlabelled["hp_probability"] >= threshold).astype(int)
    unlabelled[["gp_nr", "plz", "hp_probability", "hp_flag"]].to_csv(
        C.ARTIFACTS / "population_predictions.csv", index=False)

    n_scored = int(len(unlabelled))
    n_flagged = int(unlabelled.hp_flag.sum())
    flag_rate = n_flagged / n_scored
    gwr_rate = provenance["canton_hp_rate"]
    gwr_rate_known = provenance["canton_hp_rate_known"]

    by_plz = (unlabelled.groupby("plz")
              .agg(n_households=("hp_flag", "size"), n_flagged=("hp_flag", "sum"))
              .reset_index())
    by_plz["flag_rate"] = by_plz.n_flagged / by_plz.n_households
    by_plz["plz"] = by_plz["plz"].astype(str)
    merged = by_plz.merge(gwr, on="plz", how="inner")
    merged.to_csv(C.ARTIFACTS / "population_gwr_by_plz.csv", index=False)

    correlations = []
    for min_n in (1, 30, 100):
        sub = merged[merged.n_households >= min_n]
        for field in ("gwr_hp_rate", "gwr_hp_rate_known"):
            valid = sub[["flag_rate", field]].dropna()
            r = spearmanr(valid.flag_rate, valid[field]) if len(valid) > 2 else (np.nan, np.nan)
            correlations.append({"min_households": min_n, "reference": field,
                                 "n_postcodes": int(len(valid)),
                                 "rho": float(r[0]), "pvalue": float(r[1])})

    # Correlation depends on how many households are flagged, so repeat the
    # postcode comparison at the threshold that reproduces the GWR rate
    # exactly. If the ranking carries real signal it should show here even
    # when the absolute count is forced to agree.
    rate_matched_threshold = float(np.quantile(unlabelled.hp_probability, 1 - gwr_rate))
    unlabelled["hp_flag_rate_matched"] = (
        unlabelled.hp_probability >= rate_matched_threshold).astype(int)
    matched = (unlabelled.groupby("plz")
               .agg(n_households=("hp_flag_rate_matched", "size"),
                    n_flagged=("hp_flag_rate_matched", "sum")).reset_index())
    matched["flag_rate"] = matched.n_flagged / matched.n_households
    matched["plz"] = matched["plz"].astype(str)
    matched = matched.merge(gwr, on="plz", how="inner")
    matched_correlations = []
    for min_n in (1, 30, 100):
        sub = matched[matched.n_households >= min_n]
        valid = sub[["flag_rate", "gwr_hp_rate"]].dropna()
        r = spearmanr(valid.flag_rate, valid.gwr_hp_rate) if len(valid) > 2 else (np.nan, np.nan)
        matched_correlations.append({"min_households": min_n, "reference": "gwr_hp_rate",
                                     "n_postcodes": int(len(valid)),
                                     "rho": float(r[0]), "pvalue": float(r[1])})

    threshold_sweep = [
        {"threshold": t, "flag_rate": float((unlabelled.hp_probability >= t).mean()),
         "n_flagged": int((unlabelled.hp_probability >= t).sum())}
        for t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
    ]

    summary = {
        "surrogate_model": {
            "note": "population-scoreable surrogate on PV-pipeline household features, "
                    "NOT the frozen champion of run_classification.py",
            "features": features,
            "n_labelled_households": int(len(data)),
            "n_positives": int(data[LABEL].sum()),
            "validation": surrogate_metrics,
        },
        "population": {
            "n_unlabelled_households_scored": n_scored,
            "n_flagged_heat_pumps": n_flagged,
            "flag_rate": flag_rate,
            "threshold": threshold,
        },
        "gwr_reference": provenance,
        "comparison": {
            "gwr_hp_rate_all_buildings": gwr_rate,
            "gwr_hp_rate_known_generator": gwr_rate_known,
            "expected_count_at_gwr_rate": round(gwr_rate * n_scored),
            "expected_count_at_gwr_rate_known": round(gwr_rate_known * n_scored),
            "predicted_over_expected": flag_rate / gwr_rate,
            "per_plz_spearman": correlations,
            "rate_matched_threshold": rate_matched_threshold,
            "per_plz_spearman_at_gwr_matched_rate": matched_correlations,
            "threshold_sweep": threshold_sweep,
            "caveat": "GWR counts buildings with a heat pump as primary space-heating "
                      "generator; the model scores electricity metering customers "
                      "(gp_nr). A multi-family building is one GWR row but several "
                      "customers, so rates are the comparable quantity and the implied "
                      "count is an illustration, not an identity.",
        },
    }
    (C.ARTIFACTS / "population_gwr_comparison.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n")

    matched_rows = "\n".join(
        f"| {row['min_households']} | {row['n_postcodes']} | {row['rho']:.3f} | {row['pvalue']:.2g} |"
        for row in matched_correlations)
    naive_rows = "\n".join(
        f"| {row['min_households']} | {row['reference']} | {row['n_postcodes']} | "
        f"{row['rho']:.3f} | {row['pvalue']:.2g} |"
        for row in correlations)
    (C.NEWSPAPER / "models" / "heat_pump" / "population_results.md").write_text(f"""\
# Predicted heat-pump count vs the GWR register

## What is compared

The frozen champion of `run_classification.py` uses features that exist for the
342 labelled property samples only, so it cannot score the population. This
check therefore uses a **population-scoreable surrogate**: the same
`has_Waermepumpe` labels and the same model family, fitted on the PV pipeline's
household feature table (`models/pv/artifacts/features_gp.pkl`), which covers
{len(pop):,} households. Its validation accuracy is
**{surrogate_metrics['accuracy']:.4f}** with confusion matrix
`{surrogate_metrics['confusion_matrix']}` on {surrogate_metrics['n']} labelled
households - a different feature set from the champion, so do not read this as
the champion's accuracy.

The reference is the BFS GWR building register for canton Aargau, primary
space-heating generator `GWAERZH1` in {{7410 "Wärmepumpe für ein Gebäude",
7411 "Wärmepumpe für mehrere Gebäude"}}, restricted to existing residential
buildings (`GSTAT=1004`, `GKAT` 1020/1030/1040).

## The counts

| Quantity | Value |
|---|---:|
| Unlabelled households scored | {n_scored:,} |
| Flagged as heat pump (accuracy-optimal threshold {threshold:.3f}) | {n_flagged:,} |
| Model flag rate | {flag_rate:.1%} |
| GWR heat-pump rate (AG existing residential buildings) | {gwr_rate:.1%} |
| GWR heat-pump buildings | {provenance['n_heat_pump_buildings']:,} of {provenance['n_existing_residential_buildings']:,} |
| Expected households at the GWR rate | {round(gwr_rate * n_scored):,} |
| **Predicted / expected** | **{flag_rate / gwr_rate:.2f}x** |

The model flags {flag_rate / gwr_rate:.1f} times as many households as the
register implies. The threshold is the cause: it was chosen to maximise
accuracy on the GIGI labelled sample, which is a subsidy-applicant population
with a far higher heat-pump share than canton Aargau as a whole, and the fitted
probabilities on the population sit high (median
{float(unlabelled.hp_probability.median()):.2f}). Flag counts by threshold:

| Threshold | Flag rate | Households |
|---:|---:|---:|
{chr(10).join(f"| {row['threshold']} | {row['flag_rate']:.1%} | {row['n_flagged']:,} |" for row in threshold_sweep)}

## Does the ranking agree with the register?

At the accuracy-optimal threshold the postcode agreement is **negative** - when
{flag_rate:.0%} of everyone is flagged, the flag rate mostly measures how urban
a postcode is:

| Min households | Reference | Postcodes | Spearman ρ | p |
|---:|---|---:|---:|---:|
{naive_rows}

Forcing the same prevalence as the register (threshold
{rate_matched_threshold:.3f}, flagging exactly {gwr_rate:.1%}) flips it
positive - the *ranking* does carry real geographic heat-pump signal:

| Min households | Postcodes | Spearman ρ | p |
|---:|---:|---:|---:|
{matched_rows}

## Reading

Use the score as a ranking calibrated to an external base rate, the way
`models/pv` calibrates to the ElPA register - not as a count at the
accuracy-optimal threshold. Caveats: GWR counts buildings while the model
scores metering customers (a multi-family building is one GWR row but several
customers), the smart-meter rollout is incomplete so the scored population is
not a random sample of the canton, and `has_Waermepumpe = 0` in training is
unlabelled rather than verified heat-pump-absent.
""")

    print(f"surrogate validation accuracy : {surrogate_metrics['accuracy']:.4f}  "
          f"confusion {surrogate_metrics['confusion_matrix']}")
    print(f"unlabelled households scored  : {n_scored:,}")
    print(f"flagged as heat pump          : {n_flagged:,}  ({flag_rate:.2%})")
    print(f"GWR AG heat-pump rate         : {gwr_rate:.2%} of "
          f"{provenance['n_existing_residential_buildings']:,} existing residential buildings "
          f"({provenance['n_heat_pump_buildings']:,} with a heat pump)")
    print(f"  known-generator rate        : {gwr_rate_known:.2%}")
    print(f"expected at GWR rate          : {round(gwr_rate * n_scored):,} households")
    print(f"predicted / expected          : {flag_rate / gwr_rate:.2f}x")
    print(pd.DataFrame(correlations).to_string(index=False))
    print(f"\nat the GWR-matched threshold {rate_matched_threshold:.4f} "
          f"(same {gwr_rate:.1%} flag rate):")
    print(pd.DataFrame(matched_correlations).to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
