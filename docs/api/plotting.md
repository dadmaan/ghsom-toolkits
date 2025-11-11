# Plotting Functions

The plotting module provides comprehensive visualization functions for GHSOM models, including hierarchy visualizations, heatmaps, and cluster analysis plots.

## Hierarchy Visualization

### visualize_ghsom_hierarchy

```python
visualize_ghsom_hierarchy(node, lookup_table, filename="ghsom_tree.png")
```

Visualize the GHSOM hierarchy as a tree structure using Graphviz. Nodes are colored by their level in the hierarchy.

**Parameters:**

- `node` (object): Root node of the GHSOM tree
- `lookup_table` (dict): Dictionary mapping node IDs (strings) to node objects
- `filename` (str, optional): Output filename (default: "ghsom_tree.png")

**Example:**

```python
from ghsom_toolkits import visualize_ghsom_hierarchy

visualize_ghsom_hierarchy(
    model,
    lookup_table,
    "my_hierarchy.png"
)
```

---

### visualize_node_position

```python
visualize_node_position(
    root_node,
    lookup_table,
    node_id,
    filename="node_position.png",
    plot_descendants=False,
    target_node_color="#5e81ac"
)
```

Highlight a specific node within the hierarchy, showing its ancestors and optionally descendants.

**Parameters:**

- `root_node` (object): Root node of the GHSOM tree
- `lookup_table` (dict): Dictionary mapping node IDs to node objects
- `node_id` (str): ID of the target node to highlight
- `filename` (str, optional): Output filename (default: "node_position.png")
- `plot_descendants` (bool, optional): Include descendants (default: False)
- `target_node_color` (str, optional): Hex color for target node (default: "#5e81ac")

**Example:**

```python
visualize_node_position(
    root_node=model,
    lookup_table=lookup,
    node_id="node_42",
    plot_descendants=True,
    filename="highlighted_node.png"
)
```

---

### plot_hierarchy_treemap

```python
plot_hierarchy_treemap(
    node,
    lookup_table,
    filename=None,
    width=1200,
    height=800
)
```

Create an interactive treemap visualization where rectangle sizes are proportional to cluster sizes.

**Parameters:**

- `node` (object): Root node of the GHSOM tree
- `lookup_table` (dict): Dictionary mapping node IDs to node objects
- `filename` (str, optional): Path to save HTML file (default: None)
- `width` (int, optional): Width in pixels (default: 1200)
- `height` (int, optional): Height in pixels (default: 800)

**Returns:** `plotly.graph_objects.Figure` or `None`

**Requirements:** `pip install ghsom-toolkits[interactive]`

**Example:**

```python
fig = plot_hierarchy_treemap(
    model,
    lookup,
    "treemap.html"
)
```

---

## Heatmap Visualizations

### plot_weight_heatmap

```python
plot_weight_heatmap(
    node,
    neuron_position=None,
    cmap="viridis",
    figsize=(10, 8),
    save_path=None
)
```

Plot weight vectors as a heatmap. Can visualize a single neuron or all neurons in the map.

**Parameters:**

- `node` (object): GHSOM node containing the map
- `neuron_position` (tuple of (int, int), optional): Position (row, col) of specific neuron. If None, plots all neurons.
- `cmap` (str, optional): Matplotlib colormap (default: "viridis")
- `figsize` (tuple, optional): Figure size in inches (default: (10, 8))
- `save_path` (str, optional): Path to save figure (default: None)

**Returns:** `matplotlib.figure.Figure`

**Example:**

```python
from ghsom_toolkits.plotting import plot_weight_heatmap

# Plot specific neuron
fig = plot_weight_heatmap(
    model,
    neuron_position=(0, 0),
    save_path="weights.png"
)

# Plot all neurons
fig = plot_weight_heatmap(model)
plt.show()
```

---

### plot_activation_map

```python
plot_activation_map(
    node,
    data,
    sample_indices=None,
    cmap="YlOrRd",
    figsize=(8, 6),
    save_path=None
)
```

Visualize which neurons activate for given data samples. Shows activation patterns and Best Matching Units (BMU).

**Parameters:**

- `node` (object): GHSOM node containing the trained map
- `data` (numpy.ndarray): Input data samples, shape (n_samples, n_features)
- `sample_indices` (int or list, optional): Sample indices to visualize (default: [0])
- `cmap` (str, optional): Matplotlib colormap (default: "YlOrRd")
- `figsize` (tuple, optional): Figure size (default: (8, 6))
- `save_path` (str, optional): Path to save figure (default: None)

**Returns:** `matplotlib.figure.Figure`

**Example:**

```python
# Visualize multiple samples
fig = plot_activation_map(
    model,
    data,
    sample_indices=[0, 5, 10],
    save_path="activations.png"
)
```

---

### plot_umatrix

```python
plot_umatrix(
    node,
    cmap="gray_r",
    figsize=(8, 6),
    save_path=None
)
```

Plot U-Matrix (Unified Distance Matrix) showing average distances between neurons and their neighbors. Low values indicate cluster centers, high values indicate boundaries.

**Parameters:**

- `node` (object): GHSOM node containing the trained map
- `cmap` (str, optional): Matplotlib colormap (default: "gray_r")
- `figsize` (tuple, optional): Figure size (default: (8, 6))
- `save_path` (str, optional): Path to save figure (default: None)

**Returns:** `matplotlib.figure.Figure`

**Example:**

```python
fig = plot_umatrix(model, save_path="umatrix.png")
plt.show()
```

---

## Cluster Analysis

### plot_cluster_distribution

```python
plot_cluster_distribution(
    node,
    figsize=(10, 6),
    save_path=None
)
```

Plot distribution of cluster sizes across the GHSOM hierarchy using histogram and box plot.

**Parameters:**

- `node` (object): Root GHSOM node
- `figsize` (tuple, optional): Figure size (default: (10, 6))
- `save_path` (str, optional): Path to save figure (default: None)

**Returns:** `matplotlib.figure.Figure`

**Example:**

```python
fig = plot_cluster_distribution(model)
plt.show()
```

---

### plot_cluster_quality

```python
plot_cluster_quality(
    data,
    cluster_labels,
    figsize=(14, 5),
    save_path=None
)
```

Plot cluster quality metrics including Silhouette Score and Davies-Bouldin Index.

**Parameters:**

- `data` (numpy.ndarray): Input data, shape (n_samples, n_features)
- `cluster_labels` (numpy.ndarray): Cluster assignments, shape (n_samples,)
- `figsize` (tuple, optional): Figure size (default: (14, 5))
- `save_path` (str, optional): Path to save figure (default: None)

**Returns:** `matplotlib.figure.Figure`

**Requirements:** `pip install ghsom-toolkits[analysis]`

**Example:**

```python
from ghsom_toolkits.plotting import plot_cluster_quality

fig = plot_cluster_quality(data, cluster_labels)
plt.show()
```

---

### plot_growth_timeline

```python
plot_growth_timeline(
    training_history,
    figsize=(12, 6),
    save_path=None
)
```

Visualize how the GHSOM hierarchy evolved during training, showing number of nodes, depth, and error metrics over time.

**Parameters:**

- `training_history` (list of dict): Training history with keys like 'epoch', 'num_nodes', 'depth', 'qe', 'te'
- `figsize` (tuple, optional): Figure size (default: (12, 6))
- `save_path` (str, optional): Path to save figure (default: None)

**Returns:** `matplotlib.figure.Figure`

**Example:**

```python
history = [
    {'epoch': 0, 'num_nodes': 4, 'depth': 1, 'qe': 0.5},
    {'epoch': 10, 'num_nodes': 8, 'depth': 2, 'qe': 0.3},
]
fig = plot_growth_timeline(history, save_path="timeline.png")
```

---

### plot_ghsom_clusters

```python
plot_ghsom_clusters(
    df,
    cluster_column="GHSOM_cluster"
)
```

Plot GHSOM cluster distribution as a bar chart.

**Parameters:**

- `df` (pandas.DataFrame): DataFrame with cluster assignments
- `cluster_column` (str, optional): Column name containing cluster IDs (default: "GHSOM_cluster")

**Requirements:** `pip install ghsom-toolkits[pandas]`

**Example:**

```python
import pandas as pd
from ghsom_toolkits.plotting import plot_ghsom_clusters

df = pd.DataFrame({"GHSOM_cluster": [0, 0, 1, 1, 2, 2, 2]})
plot_ghsom_clusters(df)
```

---

### plot_ghsom_clusters_pie

```python
plot_ghsom_clusters_pie(
    data,
    cluster_column="GHSOM_cluster"
)
```

Plot GHSOM cluster distribution as a pie chart.

**Parameters:**

- `data` (pandas.DataFrame): DataFrame with cluster assignments
- `cluster_column` (str, optional): Column name (default: "GHSOM_cluster")

**Requirements:** `pip install ghsom-toolkits[pandas]`

---

## Publication-Ready Outputs

All plotting functions support high-resolution output for publications:

```python
# Export at 300 DPI for papers
fig = plot_weight_heatmap(model, save_path="figure1.pdf")
fig = plot_umatrix(model, save_path="figure2.png")

# Or save programmatically
fig.savefig("output.pdf", dpi=300, bbox_inches="tight")
```

## Common Patterns

### Complete Visualization Pipeline

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits import visualize_ghsom_hierarchy
from ghsom_toolkits.plotting import (
    plot_weight_heatmap,
    plot_activation_map,
    plot_umatrix,
    plot_cluster_distribution
)

# Train model
data = np.random.rand(200, 10)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=50)

# Build lookup table
lookup = {"root": model}

# Create all visualizations
visualize_ghsom_hierarchy(model, lookup, "hierarchy.png")
plot_weight_heatmap(model, save_path="weights.png")
plot_activation_map(model, data, [0, 1, 2], save_path="activations.png")
plot_umatrix(model, save_path="umatrix.png")
plot_cluster_distribution(model, save_path="clusters.png")
```
