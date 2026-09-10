# Customer-week table builder

Install the required Parquet engine, then run the builder from any directory:

```bash
python -m pip install -r ev_classification/requirements.txt
python ev_classification/build_customer_week_table.py
```

By default, the builder reads `/home/renku/work/store/input_data` and writes
the modelling table, four Parquet audits, `run_summary.csv`, and
`run_metadata.json` to `ev_classification/output/`.

`run_summary.csv` also reports the negative-interval rate for PV-labelled and
non-PV-labelled weeks.  This is the retained sanity check for the provisional
import/export-register convention.

Use `--data-root`, `--output-dir`, `--chunksize`, and `--verbose` to override
those defaults.  It streams each monthly export in chunks and deliberately
excludes the temporary March 2023 file.

Run the compact end-to-end check with:

```bash
python -m unittest ev_classification/test_build_customer_week_table.py -v
```

## EV classification experiments

`run_classification.py` implements the leakage-safe experiment sequence in
`plan_classify_agent.md`.  It reads only the prepared customer-week Parquet
dataset and writes immutable experiment artifacts below
`classification_output/`.

Run each stage from the repository root:

```bash
python ev_classification/run_classification.py audit
python ev_classification/run_classification.py baselines
python ev_classification/run_classification.py heuristics
python ev_classification/run_classification.py models
python ev_classification/run_classification.py calibrate
python ev_classification/run_classification.py sensitivity
python ev_classification/run_classification.py freeze
python ev_classification/run_classification.py final
```

The `final` stage refuses to run twice. It can start only after `freeze` has
written the selected configuration while the held-out test split is sealed.
Use `--smoke` with any stage for a deterministic small-data workflow whose
metrics are kept outside the full validation leaderboard.

The analysis notebook reads the cached artifacts and does not rerun model
selection:

```bash
jupyter execute --inplace --timeout=300 \
  ev_classification/classify_ev.ipynb
```

Run all table-builder and classifier contract tests with:

```bash
python -m pytest -q \
  ev_classification/test_build_customer_week_table.py \
  ev_classification/test_classification.py
```

## Charging-behavior feature study

The post-test extension in `plan_classify_features.md` investigates explicit
evening/night ramps, sustained load, paired edges, high-power distributions,
recurring power levels, multiscale changes, and customer-history recurrence.
It tunes only with folds inside the original development customers and compares
locked candidates on the reused validation cohort. It never reopens the
original final test.

Run or resume the study with:

```bash
python ev_classification/run_feature_study.py
```

Results are recorded in `results_feature_study.md`; immutable configurations,
cross-validation searches, fitted models, features, and validation predictions
are under `feature_study_output/`.
