import unittest

import numpy as np
import pandas as pd

from ev_classification_total.run_heuristic_extension import positive_override, select_recent_weeks


class HeuristicExtensionContracts(unittest.TestCase):
    def test_positive_override_never_lowers_a_score(self):
        base = np.array([.1, .4, .8])
        result = positive_override(base, [True, False, True], .7)
        np.testing.assert_allclose(result, [.7, .4, .8])
        self.assertTrue(np.all(result >= base))

    def test_history_selection_is_capped_and_eligible(self):
        frame = pd.DataFrame({
            "gp_nr": ["a"] * 14 + ["b"] * 3,
            "week_start": pd.date_range("2024-01-01", periods=17, freq="7D"),
            "eligible_for_supervised_training": [1] * 13 + [0] + [1] * 3,
            "ev_label": [0] * 17,
        })
        selected = select_recent_weeks(frame)
        self.assertEqual(selected.groupby("gp_nr").size().to_dict(), {"a": 12, "b": 3})
        self.assertTrue(selected.eligible_for_supervised_training.eq(1).all())


if __name__ == "__main__":
    unittest.main()
