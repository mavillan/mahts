import numpy as np
import pandas as pd

from tree import build_tree, get_nodes_per_level

class HierarchicalTimeSerie:
    def __init__(self, base_series, hierarchy, method, forecast_method):
        """
        Parameters
        ----------
        base_series: pandas.Dataframe 
            Dataframe containing the base series in its columns
        hierarchy: dict 
            Dictionary descriptor of the hierarchy of series
        method: string 
            Hierarchical time series forecasting method
        forecast_method: string
            Forecast method to use (arima, and ets).
        """
        self.base_series = base_series
        self.hierarchy = hierarchy
        self.method = method
        self.forecast_method = forecast_method
        self.tree = build_tree(hierarchy)
        
        # aggregating the series for the rest of the hierarchy
        nodes_per_level = get_nodes_per_level(self.tree, skip_leaves=True)
        hierarchy_series = pd.DataFrame()
        for level in nodes_per_level[::-1]:
            for node in level:
            hierarchy_series[node] = bottom_series.loc[:, hierarchy[node]].sum(axis=1)
        self.hierarchy_series = hierarchy_series
        

    def compute_average_hist_proportions():
        pass

    def computer_hist_average_proportios():
        pass

    def forecast():
        pass
    
