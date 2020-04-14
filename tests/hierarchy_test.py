import unittest
import numpy as np
from anytree import Node, LevelOrderGroupIter
from mahts.hierarchy import build_tree, compute_summing_matrix

class TestForecaster(unittest.TestCase):
    def setUp(self):
        self.hierarchy = {"root":["A","B"], "A":["A1","A2"], "B":["B1","B2","B3"]}

    def test_it_builds_the_tree(self):
        expected_tree = Node("root")
        a = Node("A", parent=expected_tree)
        b = Node("B", parent=expected_tree)
        _ = Node("A1", parent=a)
        _ = Node("A2", parent=a)
        _ = Node("B1", parent=b)
        _ = Node("B2", parent=b)
        _ = Node("B3", parent=b)

        actual_tree = build_tree(self.hierarchy)

        nodes_expected_tree = [node.name 
                               for children in LevelOrderGroupIter(expected_tree) 
                               for node in children]
        nodes_actual_tree = [node.name 
                             for children in LevelOrderGroupIter(actual_tree) 
                             for node in children]
        assert nodes_expected_tree == nodes_actual_tree, \
            "Actual tree is not equal to expected tree."
    
    def test_it_computes_the_summing_matrix(self):
        expected_matrix = np.array([[1, 1, 1, 1, 1],
                                    [1, 1, 0, 0, 0],
                                    [0, 0, 1, 1, 1],
                                    [1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 1]])

        tree = build_tree(self.hierarchy)
        actual_matrix = compute_summing_matrix(tree)[0]

        np.testing.assert_array_equal(actual_matrix, expected_matrix)
