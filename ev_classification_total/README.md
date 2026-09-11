# EV classification on the total property sample set

This is a self-contained rerun of the adaptive EV-classification study using
`../data_addition/train_property_samples_auxw_netload.csv`, the input marked
**TRAIN ON THIS** in `data_addition/Readme.md`.

The run uses a permanent stratified 60/20/20 split by `group_id`. Augmented and
cross-sectional rows from one property therefore cannot cross splits. Models
never receive IDs, dates, sample metadata, technology labels, or any of the
quarantined feed-in/export features. The seven `augmented_before` rows may help
fit a model, but validation and test metrics use one current cross-sectional row
per property.

Run from the repository root:

```bash
python -m pip install -r ev_classification_total/requirements.txt
python ev_classification_total/run_classification.py
python ev_classification_total/run_heuristic_extension.py
python ev_classification_total/run_heuristic_extension_2.py
python -m pytest -q ev_classification_total/test_classification.py
python -m pytest -q ev_classification_total/test_heuristic_extension.py
python -m pytest -q ev_classification_total/test_heuristic_extension_2.py
```

The runner writes all artifacts under `ev_classification_total/output/` and
refuses to overwrite a completed run. See `results.md` for the executed run,
validation leaderboard, final test result, and limitations.

The post-test extension in `plan_heuristic_extension.md` combines the selected
LightGBM score with high-confidence charging-event evidence from the detailed
weekly Parquet data. Its outputs are under `output/heuristic_extension/`; because
the original test was already opened, the extension labels that comparison as a
reused-test diagnostic.

`run_heuristic_extension_2.py` tests steep start-edge evidence and directly
compares positive-only rescue, negative-only veto, and two-way post-processing.
Its frozen artifacts are written to `output/heuristic_extension_2/`.
