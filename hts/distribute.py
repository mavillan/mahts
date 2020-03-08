import numpy as np
import pandas as pd
from anytree import LevelOrderIter
from hts.hierarchy import build_tree, compute_summing_matrix

class HTSDistributor():
    def __init__(self, hierarchy):
        """
        Parameters
        ----------
        hierarchy: dict 
            Dictionary describing the hierarchical relationship between
            the time series.
        """
        tree = build_tree(hierarchy)
        summing_matrix,tree_nodes,bottom_nodes = compute_summing_matrix(tree)
        self.hierarchy = hierarchy
        self.tree = tree
        self.summing_matrix = summing_matrix
        self.tree_nodes = tree_nodes
        self.bottom_nodes = bottom_nodes
        self.proportions = None

    def compute_bottom_up(self, forecast):
        assert set(forecast.columns) == set(self.bottom_nodes), \
            f"'forecast' dataframe must have only the columns: {self.bottom_nodes}."
        result = np.dot(forecast.loc[:, self.bottom_nodes].values, self.summing_matrix.T)
        return pd.DataFrame(result, columns=self.tree_nodes)

    def compute_top_down(self, data, forecast, kind="ahp"):
        assert set(data.columns) == set(self.bottom_nodes), \
            f"'data' dataframe must have only the columns: {self.bottom_nodes}."
        assert set(forecast.columns) == set(["root"]), \
            "'forecast' dataframe must have only the column: 'root'."
        data = data.copy(deep=True).loc[:, self.bottom_nodes]
        root_timeserie = data.values.sum(axis=1)
        if kind == "ahp":
            proportions = np.mean(data.values / root_timeserie.reshape(-1,1), axis=0)
        elif kind == "pha":
            proportions = data.values.mean(axis=0) / root_timeserie.mean()
        forecast_bottom = pd.DataFrame(forecast["root"].values.reshape(-1,1) * proportions.reshape(1,-1),
                                       columns=self.bottom_nodes)
        self.proportions = proportions
        return self.compute_bottom_up(forecast_bottom)
    
    def compute_forecast_proportions(self, forecast):
        assert set(forecast.columns) == set(self.tree_nodes), \
            f"'forecast' dataframe must have only the columns: {self.tree_nodes}."
        proportions_by_node = pd.DataFrame(np.ones(forecast.shape[0]), columns=["root"])
        for node in LevelOrderIter(self.tree):
            if node.name == "root": continue
            level_nodes = [nd.name for nd in node.parent.children]
            proportions_by_node[node.name] = forecast.loc[:,node.name]/forecast.loc[:,level_nodes].sum(axis=1)
            proportions_by_node.loc[:, node.name] = proportions_by_node[node.name].fillna(0)
            proportions_by_node.loc[:, node.name] *= proportions_by_node[node.parent.name]
        forecast_bottom = proportions_by_node[self.bottom_nodes] * forecast["root"].values.reshape(-1,1)
        self.proportions = proportions_by_node
        return self.compute_bottom_up(forecast_bottom)

    def compute_middle_out(self, data, forecast, middle_level=None):
        pass
    
    def compute_optimal_combination(self, forecast):
        pass
