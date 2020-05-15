import logging
import numpy as np
from scipy import sparse
import pandas as pd
from anytree import LevelOrderIter, RenderTree, AsciiStyle
from mahts.hierarchy import build_tree, compute_summing_matrix

logger = logging.getLogger(__name__)

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
        self.sparse_summing_matrix = sparse.csr_matrix(summing_matrix)
        self.tree_nodes = tree_nodes
        self.bottom_nodes = bottom_nodes
        self.proportions = None
    
    def show_tree(self):
        """
        Renders the hierarchical tree.
        """
        print(RenderTree(self.tree, style=AsciiStyle())) 

    def compute_bottom_up(self, forecast):
        """
        Computes the bottom-up aggregation of the provided data.

        Parameters
        ----------
        forecast: pandas.DataFrame
            Dataframe containing the forecast predictions of the 
            base series in its columns.
        """
        assert set(forecast.columns) == set(self.bottom_nodes), \
            f"'forecast' dataframe must have only the columns: {self.bottom_nodes}."

        result = np.dot(forecast.loc[:, self.bottom_nodes].values, self.summing_matrix.T)
        return pd.DataFrame(result, columns=self.tree_nodes, index=forecast.index)

    def compute_top_down(self, data, forecast, kind="ahp"):
        """
        Compute the top-down reconciliation of the provided forecast
        using the historical data.

        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe containing the historical time series of
            the base series in its columns.
        forecast: pandas.DataFrame
            Dataframe containing the forecast for the total/root
            time serie.
        kind: str
            Kind of proportions used for reconciliation: 1) "ahp"
            average of historical proportions, or 2) "pha" proportions
            of historical average" 
        """
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
        """
        Computes the top-down reconciliation using the forecast
        values as proportions.

        Parameters
        ----------
        forecast: pandas.DataFrame
            Dataframe containing the forecast for the series in all
            levels of the hierarchy in its columns.
        """
        assert set(forecast.columns) == set(self.tree_nodes), \
            f"'forecast' dataframe must have only the columns: {self.tree_nodes}."

        proportions_by_node = pd.DataFrame(np.ones(forecast.shape[0]), columns=["root"], index=forecast.index)
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
    
    def compute_optimal_combination(self, forecast, weights=None, solver_kwargs=dict()):
        """
        Computes optimal-combination reconciliation between all
        levels, using least-squares minimization.

        Parameters
        ----------
        forecast: pandas.DataFrame
            Dataframe containing the forecast for the series in all
            levels of the hierarchy in its columns.
        weights: dict
            Weights used of weighted least squares regression. Must
            contain the time series indentifier in its keys and the 
            corresponding weights as values.
        solver_kwargs: dict
            Extra keyword arguments of scipy.sparse.linalg.lsqr function.
        """
        assert set(forecast.columns) == set(self.tree_nodes), \
            f"'forecast' dataframe must have all (and only) the columns: {self.tree_nodes}."

        if weights is not None:
            assert set(weights.keys()) == set(self.tree_nodes), \
                f"'weights' dict must have all (and only) the keys: {self.tree_nodes}."
            weights_matrix = sparse.diags([weights[node] for node in self.tree_nodes])
            X = weights_matrix.dot(self.sparse_summing_matrix)
        else:
            X = self.sparse_summing_matrix

        adjusted_rows = list()
        for i,row in forecast.iterrows():
            logger.info(f"Reconciling time step {i+1}/{forecast.shape[0]}")
            if weights is not None:
                y = weights_matrix.dot(row[self.tree_nodes].values)
            else:
                y = row[self.tree_nodes].values
            beta = sparse.linalg.lsqr(X, y, **solver_kwargs)[0]
            adjusted_rows.append(beta)
        forecast_bottom = pd.DataFrame(adjusted_rows, columns=self.bottom_nodes)
        return self.compute_bottom_up(forecast_bottom)
