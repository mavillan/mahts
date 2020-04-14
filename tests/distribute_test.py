import unittest
import numpy as np
import pandas as pd

from mahts import HTSDistributor

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
        data = pd.DataFrame(
            [[2, 1, 2, 1, 2],
             [2, 2, 2, 2, 2]],
            columns = ["A1", "A2", "B1", "B2", "B3"]
        )
        forecast = pd.DataFrame(
            [[10], 
             [20]], 
            columns = ["root"] 
        )
        expected = pd.DataFrame(
            [[10., 3.875, 6.125, 2.25, 1.625, 2.25, 1.625, 2.25],
             [20., 7.75, 12.25, 4.5, 3.25, 4.5, 3.25, 4.5]],
            columns = ["root", "A", "B", "A1", "A2", "B1", "B2", "B3"]
        )

        hts = HTSDistributor(self.hierarchy)
        actual = hts.compute_top_down(data, forecast, kind="ahp")

        pd.testing.assert_frame_equal(actual, expected)

    def test_it_computes_top_down_with_pha(self):
        data = pd.DataFrame(
            [[3, 5, 3, 4, 2],
             [1, 3, 5, 4, 2]],
            columns = ["A1", "A2", "B1", "B2", "B3"]
        )
        forecast = pd.DataFrame(
            [[10], 
             [20]], 
            columns = ["root"] 
        )
        expected = pd.DataFrame(
            [[10., 3.75, 6.25, 1.25, 2.5, 2.5, 2.5, 1.25],
             [20., 7.5, 12.5, 2.5, 5., 5., 5., 2.5]],
            columns = ["root", "A", "B", "A1", "A2", "B1", "B2", "B3"]
        )

        hts = HTSDistributor(self.hierarchy)
        actual = hts.compute_top_down(data, forecast, kind="pha")

        pd.testing.assert_frame_equal(actual, expected)
    
    def test_it_computes_forecast_proportions(self):
        forecast = pd.DataFrame(
            [[10, 4, 6, 2, 2, 1, 2, 3],
             [10, 4, 4, 2, 1, 1, 1, 1],
             [10, 3, 1, 1, 2, 1, 1, 2],
             [10, 1, 3, 2, 1, 1, 1, 1]],
            columns=["root", "A", "B", "A1", "A2", "B1", "B2", "B3"]
        )
        expected = pd.DataFrame(
            [[10., 4., 6., 2., 2., 1., 2., 3.],
             [10., 5., 5., 5*(2/3), 5*(1/3), 5*(1/3), 5*(1/3), 5*(1/3)],
             [10., 7.5, 2.5, 7.5*(1/3), 7.5*(2/3), 2.5*(1/4), 2.5*(1/4), 2.5*(1/2)],
             [10., 2.5, 7.5, 2.5*(2/3), 2.5*(1/3), 7.5*(1/3), 7.5*(1/3), 7.5*(1/3)]],
            columns=["root", "A", "B", "A1", "A2", "B1", "B2", "B3"]
        )

        hts = HTSDistributor(self.hierarchy)
        actual = hts.compute_forecast_proportions(forecast)

        pd.testing.assert_frame_equal(actual, expected)

    def test_it_computes_optimal_combination(self):
        pass