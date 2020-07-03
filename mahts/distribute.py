import numpy as np
from scipy import sparse
from scipy.optimize._lsq.lsq_linear import prepare_bounds
from scipy.optimize._lsq.trf_linear import trf_linear
import pandas as pd
from anytree import LevelOrderIter, RenderTree, AsciiStyle
from mahts.hierarchy import build_tree, compute_summing_matrix
from mahts.utils import format_lstsq_output

DEFAULT_TRF_KWARGS = {"tol":1e-10, "lsq_solver":"lsmr", "lsmr_tol":1e-6, 
                      "max_iter":100, "verbose":0}

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
            if len(node.children) == 0: continue
            children_nodes = [nd.name for nd in node.children]
            children_forecast = forecast.loc[:, children_nodes].values
            children_forecast_agg = forecast.loc[:, children_nodes].values.sum(axis=1).reshape(-1,1)
            children_proportions = children_forecast / (children_forecast_agg + np.finfo(float).eps)
            children_proportions *= proportions_by_node[node.name].values.reshape(-1,1)
            children_proportions_dataframe = pd.DataFrame(children_proportions, 
                                                          columns=children_nodes, 
                                                          index=forecast.index)
            proportions_by_node = pd.concat([proportions_by_node, children_proportions_dataframe], axis=1)
        forecast_bottom = proportions_by_node[self.bottom_nodes] * forecast["root"].values.reshape(-1,1)
        self.proportions = proportions_by_node
        return self.compute_bottom_up(forecast_bottom)

    def compute_middle_out(self, data, forecast, middle_level=None):
        pass
    
    def compute_optimal_combination(self, forecast, weights=None, backend="lsmr", bounds=None, 
                                    solver_kwargs=dict(), trf_kwargs=dict()):
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
        backend: str
            Linear least squares solver: lsqr or lsmr. 
        bounds: list
            List of 2-tuples array_like with the lower and upper bounds 
            for each time step to reconcile. 
        solver_kwargs: dict
            Arguments for the lstsq backend: scipy.sparse.linalg.lsqr or 
            scipy.sparse.linalg.lsmr.
        trf_kwargs: dict
            Arguments for the trf method:
                - tol: terminates if the uniform norm of the gradient, 
                       scaled to account for the presence of the bounds, 
                       is less than tol.
                - lsmr_tol: Tolerance parameters ‘atol’ and ‘btol’ for 
                            scipy.sparse.linalg.lsmr.
                - max_iter: Maximum number of iterations before termination.
                - verbose: 0 = work silently (default); 1 = display a termination 
                           report; 2 = display progress during iterations.
        """
        assert set(forecast.columns) == set(self.tree_nodes), \
            f"'forecast' dataframe must have all (and only) the columns: {self.tree_nodes}."
        
        if not backend in ["lsqr","lsmr"]:
            raise ValueError(f"backend should be 'lsqr' or 'lsmr'.")
        
        if bounds is not None and backend == "lsqr":
            raise ValueError(f"Bounds can be used only with 'lsmr' backend.")

        if weights is not None:
            assert set(weights.keys()) == set(self.tree_nodes), \
                f"'weights' dict must have all (and only) the keys: {self.tree_nodes}."
            weights_matrix = sparse.diags([weights[node] for node in self.tree_nodes])
            X = weights_matrix.dot(self.sparse_summing_matrix)
        else:
            X = self.sparse_summing_matrix

        adjusted_rows = list()
        for i,row in forecast.iterrows():
            print("-"*100)
            print(f" Reconciling time step: {i} ".center(100, "-"))
            print("-"*100)
            if weights is not None:
                y = weights_matrix.dot(row[self.tree_nodes].values)
            else:
                y = row[self.tree_nodes].values
            
            if backend == "lsqr":
                lstsq_output = sparse.linalg.lsqr(X, y, **solver_kwargs)
                beta = lstsq_output[0]
                format_lstsq_output(lstsq_output, backend=backend)
            elif backend == "lsmr":
                lstsq_output = sparse.linalg.lsmr(X, y, **solver_kwargs)
                beta = lstsq_output[0]
                format_lstsq_output(lstsq_output, backend=backend)
                
            if bounds is not None:
                _trf_kwargs = {**DEFAULT_TRF_KWARGS, **trf_kwargs}
                lower_bounds,upper_bounds = prepare_bounds(bounds, X.shape[1])
                trf_output = trf_linear(X, y, beta, lower_bounds, upper_bounds, **_trf_kwargs)
                beta = trf_output["x"]
            adjusted_rows.append(beta)
        forecast_bottom = pd.DataFrame(adjusted_rows, columns=self.bottom_nodes, index=forecast.index)
        return self.compute_bottom_up(forecast_bottom)
