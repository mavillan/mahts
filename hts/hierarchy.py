import numpy as np
from anytree import Node, LevelOrderGroupIter

def build_tree(hierarchy):
    """
    Parameters
    ----------
    hierarchy: dict
        Python dictionary defining the hierarchy
    Returns
    ----------
    anytree.Node 
    """
    assert "root" in hierarchy.keys(), "missing 'root' key in 'hierarchy'"
    tree = Node("root")
    while True:
        if not any(leaf.name in hierarchy.keys() for leaf in tree.leaves):
            break
        all_leaves = [leaf for leaf in tree.leaves]
        for leaf in all_leaves:
            if leaf.name not in hierarchy.keys(): continue
            for child in hierarchy[leaf.name]:
                Node(child, parent=leaf)
    return tree

def get_nodes_per_level(tree, skip_leaves=False):
    if skip_leaves:
        nodes_per_level = [[node.name for node in children if not node.is_leaf]
                           for children in LevelOrderGroupIter(tree)]
    else:
        nodes_per_level = [[node.name for node in children]
                           for children in LevelOrderGroupIter(tree)]
    nodes_per_level = list(filter(lambda x: len(x)>0, nodes_per_level))
    return nodes_per_level

def compute_summing_matrix(tree):
    bottom_nodes = [leaf.name for leaf in tree.leaves]
    tree_nodes = list()
    matrix_rows = list()
    for level in LevelOrderGroupIter(tree):
        for node in level:
            node_leaves = [leaf.name for leaf in node.leaves]
            matrix_rows.append([int(leaf in node_leaves) for leaf in bottom_nodes])
            tree_nodes.append(node.name)     
    summing_matrix = np.asarray(matrix_rows)     
    return summing_matrix, tree_nodes, bottom_nodes