import numpy as np
import pandas as pd

from movingclusterlib import GraphAnalizer, gates, plot_clusters


class CustomGate(gates.Expr):
    def __init__(self, distance, key_xy):
        expr = f"""l2_dist(
            item1->{key_xy[0]}, item1->{key_xy[1]}, 
            item2->{key_xy[0]}, item2->{key_xy[1]}
        ) < {distance}"""

        preamble = f"""
            __device__ float l2_dist(float x1, float y1, float x2, float y2) {{
                float dx = x1 - x2;
                float dy = y1 - y2;
                return sqrt(dx * dx + dy * dy);
            }}
        """

        super().__init__(expr, preamble=preamble)


if __name__ == "__main__":
    # Set random seed to 0
    np.random.seed(0)

    size = 1000
    window = 86400

    # Generate ts_arr
    dts_arr = np.random.uniform(1, 1000, size).astype(np.uint32)
    ts_arr = 1665058722 + np.cumsum(dts_arr)

    # Define dataset
    df = pd.DataFrame({
        'ts': ts_arr, 
        'a': np.random.randint(0, 100, size),
        'x': np.random.random(size),
        'y': np.random.random(size),
    })
    print(df)

    # Analysis with graph analyzer
    ga = GraphAnalizer(
        edge_func=gates.And(
            gates.TimeWindow(40000),
            gates.Equal("a"),
            CustomGate(0.5, ["x", "y"]),
        ).full_code(),
        features={
            'ts': np.uint32,
            'a': np.int32,
            'x': np.float64,
            'y': np.float64,
        },
    )
    df['cluster_id'], df['cluster_size'] = ga.process(df)

    # Plot clusters
    plot_clusters(df['ts'], df['cluster_id'], df['cluster_size'], size=None)
