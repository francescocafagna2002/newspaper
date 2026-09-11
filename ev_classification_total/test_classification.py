import unittest

import pandas as pd

from ev_classification_total.run_classification import (
    LABEL,
    PROHIBITED,
    feature_sets,
    load_data,
    make_split,
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


if __name__ == "__main__":
    unittest.main()
