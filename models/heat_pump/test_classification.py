import unittest

import numpy as np

from models.heat_pump.run_classification import (
    LABEL,
    PROHIBITED,
    choose_threshold,
    feature_sets,
    load_data,
    make_split,
    metrics,
)


class ClassificationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_group_split_is_disjoint_and_stratified(self):
        split = make_split(self.data)
        self.assertEqual(split.group_id.nunique(), self.data.group_id.nunique())
        self.assertEqual(set(split.split), {"development", "validation", "test"})
        rates = split.groupby("split")[LABEL].mean()
        self.assertLess(rates.max() - rates.min(), 0.03)

    def test_feature_sets_exclude_ids_dates_and_labels(self):
        for columns in feature_sets(self.data).values():
            self.assertFalse(set(columns) & PROHIBITED)
            self.assertFalse(any(c.startswith("he") for c in columns))
            self.assertFalse(any("einsp" in c for c in columns))

    def test_augmented_rows_cannot_cross_splits(self):
        split = make_split(self.data)
        joined = self.data.merge(split[["group_id", "split"]], on="group_id")
        self.assertEqual(joined.groupby("group_id").split.nunique().max(), 1)

    def test_engineered_features_present(self):
        for column in ("wpb_score", "wpb_flag", "winter_share_over_night_share"):
            self.assertIn(column, self.data.columns)
        self.assertIn("wpb_score", feature_sets(self.data)["compact"])

    def test_confusion_matrix_matches_accuracy(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.9, 0.2, 0.8])
        result = metrics(y, score, 0.5)
        (tn, fp), (fn, tp) = result["confusion_matrix"]
        self.assertEqual([tn, fp, fn, tp], [1, 1, 1, 1])
        self.assertAlmostEqual(result["accuracy"], (tn + tp) / len(y))

    def test_threshold_maximises_accuracy(self):
        y = np.array([0, 0, 0, 1, 1])
        score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        threshold = choose_threshold(y, score)
        self.assertEqual(metrics(y, score, threshold)["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
