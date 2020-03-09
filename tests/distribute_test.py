import unittest
import numpy as np
import pandas as pd

from hts import HTSDistributor

class TestForecaster(unittest.TestCase):
    def setUp(self):
        self.hierarchy = {"root":["A","B"], "A":["A1","A2"], "B":["B1","B2","B3"]}
    
    def test_it_builds_hts_distributor(self):
        hts = HTSDistributor(self.hierarchy)

        assert hasattr(hts, "hierarchy"), \
            "'HTSDistributor' instance has missing 'hierarchy' attribute."
        assert hasattr(hts, "tree"), \
            "'HTSDistributor' instance has missing 'tree' attribute."
        assert hasattr(hts, "summing_matrix"), \
            "'HTSDistributor' instance has missing 'summing_matrix' attribute."
        assert hasattr(hts, "tree_nodes"), \
            "'HTSDistributor' instance has missing 'tree_nodes' attribute."
        assert hasattr(hts, "bottom_nodes"), \
            "'HTSDistributor' instance has missing 'bottom_nodes' attribute."
        assert hasattr(hts, "proportions"), \
            "'HTSDistributor' instance has missing 'proportions' attribute."
        
    def test_it_computes_bottom_up(self):
        input_data = pd.DataFrame(
            [[1, 2, 1, 2, 3],
             [2, 3, 2, 3, 4],
             [0, 1, 0, 1, 2]],
            columns = ["A1", "A2", "B1", "B2", "B3"]
        )
        expected = pd.DataFrame(
            [[9, 3, 6, 1, 2, 1, 2, 3],
             [14, 5, 9, 2, 3, 2, 3, 4],
             [4, 1, 3, 0, 1, 0, 1, 2]],
            columns = ["root", "A", "B", "A1", "A2", "B1", "B2", "B3"]
        )

        hts = HTSDistributor(self.hierarchy)
        actual = hts.compute_bottom_up(input_data)

        pd.testing.assert_frame_equal(actual, expected)

    def test_it_computes_top_down_with_ahp(self):
        pass
    
    def test_it_computes_forecast_proportions(self):
        pass
    
    def test_it_computes_optimal_combination(self):
        pass