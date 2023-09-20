import pandas as pd
import matplotlib.pyplot as plt
from datatool.cuda.base import dataframe_to_records
from datatool.cuda.cluster import clusterize


class GraphAnalizer:
    def __init__(self, edge_func, features, ts_field='ts'):
        self._edge_func = edge_func
        self._column_dtypes = list(features.items())

    def process(self, df, cluster_init=None):
        items = dataframe_to_records(
            df,
            column_dtypes=self._column_dtypes,
            align=True
        )

        cluster_id_list = clusterize(items, self._edge_func, 
                                     cluster_init=cluster_init)

        cluster_id_series = pd.Series(cluster_id_list)
        cluster_size_series = cluster_id_series.groupby(
            cluster_id_series
        ).cumcount() + 1

        return cluster_id_series, cluster_size_series


def plot_clusters(ts_series, cluster_id_series, cluster_size_series, size=None):
    plt.figure(figsize=(12, 4))

    if size is not None:
        cluster_id_to_plot = cluster_id_series.groupby(
            cluster_id_series
        ).count().sort_values(ascending=False)[:size].index
    else:
        cluster_id_to_plot = cluster_id_series.unique()

    for cluster_id in cluster_id_to_plot:
        where = cluster_id_series == cluster_id
        x = ts_series[where]
        y = cluster_size_series[where]
        plt.plot(x, y)
        plt.fill_between(x, y, alpha=0.2)

    plt.grid(True)
    plt.xlabel('ts')
    plt.ylabel('cluster_size')
    plt.title("Cluster statistics")
    plt.show()
