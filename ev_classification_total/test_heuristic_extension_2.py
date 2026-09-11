import unittest

import numpy as np

from ev_classification_total.run_heuristic_extension_2 import apply_direction


class SteepEdgeDirectionContracts(unittest.TestCase):
    def test_positive_rescue_only_raises_scores(self):
        base = np.array([.2, .5, .8])
        result = apply_direction(base, [True, False, True], "positive_rescue", floor=.7)
        np.testing.assert_allclose(result, [.7, .5, .8])
        self.assertTrue(np.all(result >= base))

    def test_negative_veto_only_lowers_scores(self):
        base = np.array([.2, .5, .8])
        result = apply_direction(base, [True, False, True], "negative_veto", ceiling=.3)
        np.testing.assert_allclose(result, [.2, .3, .8])
        self.assertTrue(np.all(result <= base))

    def test_two_way_applies_both_directions(self):
        result = apply_direction([.2, .8], [True, False], "two_way", floor=.7, ceiling=.3)
        np.testing.assert_allclose(result, [.7, .3])


if __name__ == "__main__":
    unittest.main()
