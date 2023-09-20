# movingclusterlib

This library implements a tool to analyze cluster evolution during the time for a given Pandas dataframe.

## Installation

```
pip install movingclusterlib
```

## Usage example

```python
import numpy as np
from movingclusterlib import GraphAnalizer, gates, plot_clusters

df = ...

# Cluster analysis
ga = GraphAnalizer(
    edge_func=gates.And(
        gates.TimeWindow(40000),
        gates.Equal("a"),
    ).full_code(),
    features={
        'ts': np.uint32,
        'a': np.int32,
    },
)
df['cluster_id'], df['cluster_size'] = ga.process(df)

# Plot 10 biggest clusters on a diagram
plot_clusters(df['ts'], df['cluster_id'], df['cluster_size'], size=10)
```
