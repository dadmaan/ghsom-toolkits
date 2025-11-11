# Visualization Gallery

Explore all visualization types available in ghsom-toolkits with code examples and output samples.

## Hierarchy Visualizations

### Full Hierarchy Tree

Visualize the complete GHSOM hierarchy as a tree structure.

```python
from ghsom_toolkits import visualize_ghsom_hierarchy

visualize_ghsom_hierarchy(
    node=model,
    lookup_table=lookup,
    filename="hierarchy.png"
)
```

**Features:**
- Nodes colored by level
- Shows node ID, position, size, and children count
- Top-to-bottom layout
- Publication-ready output

**Example Output:**

![Hierarchy Example](../example_outputs/ghsom_hierarchy_example.png)

---

### Node Position Highlighting

Focus on a specific node and its context.

```python
from ghsom_toolkits import visualize_node_position

visualize_node_position(
    root_node=model,
    lookup_table=lookup,
    node_id="specific_node",
    filename="highlighted_node.png",
    plot_descendants=True,
    target_node_color="#FF5733"
)
```

**Use Cases:**
- Highlight interesting clusters
- Show subtree structure
- Trace hierarchical relationships

---

### Interactive Treemap

Interactive HTML treemap where size = cluster size.

```python
from ghsom_toolkits.plotting import plot_hierarchy_treemap

fig = plot_hierarchy_treemap(
    node=model,
    lookup_table=lookup,
    filename="treemap.html",
    width=1200,
    height=800
)
```

**Features:**
- Zoomable and interactive
- Hover for details
- Color-coded by level
- Proportional sizes

**Requirements:** `pip install ghsom-toolkits[interactive]`

---

## Heatmap Visualizations

### Weight Vectors Heatmap

Visualize neuron weight vectors.

```python
from ghsom_toolkits.plotting import plot_weight_heatmap

# All neurons
fig = plot_weight_heatmap(
    node=model,
    cmap='viridis',
    figsize=(10, 8),
    save_path="weights_all.png"
)

# Single neuron
fig = plot_weight_heatmap(
    node=model,
    neuron_position=(0, 0),
    cmap='viridis',
    save_path="weight_single.png"
)
```

**Example Output:**

![Weight Heatmap](../example_outputs/weight_heatmap_all.png)

**Use Cases:**
- Understand learned patterns
- Compare neuron weights
- Identify feature importance

---

### Activation Maps

Show which neurons activate for specific samples.

```python
from ghsom_toolkits.plotting import plot_activation_map

fig = plot_activation_map(
    node=model,
    data=data,
    sample_indices=[0, 5, 10],
    cmap="YlOrRd",
    save_path="activations.png"
)
```

**Example Output:**

![Activation Maps](../example_outputs/activation_maps.png)

**Features:**
- Multiple samples in one figure
- BMU (Best Matching Unit) marked with star
- Color intensity = activation strength

---

### U-Matrix

Unified Distance Matrix showing cluster boundaries.

```python
from ghsom_toolkits.plotting import plot_umatrix

fig = plot_umatrix(
    node=model,
    cmap="gray_r",
    figsize=(8, 6),
    save_path="umatrix.png"
)
```

**Example Output:**

![U-Matrix](../example_outputs/umatrix.png)

**Interpretation:**
- **Dark areas**: Cluster centers (neurons are similar)
- **Light areas**: Cluster boundaries (neurons are different)

---

## Cluster Analysis

### Cluster Size Distribution

Histogram and box plot of cluster sizes.

```python
from ghsom_toolkits.plotting import plot_cluster_distribution

fig = plot_cluster_distribution(
    node=model,
    figsize=(10, 6),
    save_path="cluster_dist.png"
)
```

**Example Output:**

![Cluster Distribution](../example_outputs/cluster_distribution.png)

**Shows:**
- Histogram of cluster sizes
- Box plot with outliers
- Summary statistics

---

### Cluster Quality Metrics

Silhouette score and Davies-Bouldin index.

```python
from ghsom_toolkits.plotting import plot_cluster_quality

fig = plot_cluster_quality(
    data=data,
    cluster_labels=labels,
    figsize=(14, 5),
    save_path="cluster_quality.png"
)
```

**Example Output:**

![Cluster Quality](../example_outputs/cluster_quality.png)

**Requirements:** `pip install ghsom-toolkits[analysis]`

**Shows:**
- Silhouette plot (left)
- Per-cluster quality scores (right)
- Overall quality metrics

---

### Growth Timeline

Visualize hierarchy evolution during training.

```python
from ghsom_toolkits.plotting import plot_growth_timeline

history = [
    {'epoch': 0, 'num_nodes': 4, 'depth': 1, 'qe': 0.5},
    {'epoch': 10, 'num_nodes': 12, 'depth': 2, 'qe': 0.3},
    {'epoch': 20, 'num_nodes': 24, 'depth': 3, 'qe': 0.2},
]

fig = plot_growth_timeline(
    training_history=history,
    figsize=(12, 6),
    save_path="timeline.png"
)
```

**Example Output:**

![Growth Timeline](../example_outputs/growth_timeline.png)

**Shows:**
- Number of nodes over time
- Hierarchy depth evolution
- Quantization error reduction
- Topological error (if available)

---

## Interactive Tools

### Dashboard

Web-based interactive explorer.

```python
from ghsom_toolkits.interactive import launch_dashboard

launch_dashboard(
    node=model,
    data=data,
    port=8050
)
# Open: http://localhost:8050
```

**Features:**
- Interactive hierarchy visualization
- Node inspector
- Sample path tracer
- Real-time metrics
- Weight exploration

**Requirements:** `pip install ghsom-toolkits[interactive]`

---

## Analysis Visualizations

### Model Comparison (Bar Chart)

Compare multiple models side-by-side.

```python
from ghsom_toolkits.analysis import compare_models, plot_comparison

comparison = compare_models(models, data, names)
fig = plot_comparison(comparison, plot_type='bar', save_path="comparison_bar.png")
```

**Shows:**
- Total nodes
- Max depth
- Leaf nodes (clusters)
- Quantization error

---

### Model Comparison (Radar Chart)

Multi-metric comparison on radar plot.

```python
fig = plot_comparison(comparison, plot_type='radar', save_path="comparison_radar.png")
```

**Best for:**
- Comparing 3-5 models
- Multi-dimensional evaluation
- Identifying trade-offs

---

### Model Comparison (Heatmap)

Matrix view for many models.

```python
fig = plot_comparison(comparison, plot_type='heatmap', save_path="comparison_heatmap.png")
```

**Best for:**
- Grid search results
- Many model configurations
- Identifying patterns

---

## Complete Visualization Pipeline

Generate all visualizations at once:

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits import visualize_ghsom_hierarchy
from ghsom_toolkits.plotting import (
    plot_weight_heatmap,
    plot_activation_map,
    plot_umatrix,
    plot_cluster_distribution
)
import matplotlib.pyplot as plt

# Train model
data = np.random.rand(300, 15)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
result = ghsom.train(epochs_number=100)

# Adapt
model = adapt_model(result, input_dataset_size=len(data))
lookup = build_lookup_table(model)

# Generate all visualizations
print("Generating visualizations...")

visualize_ghsom_hierarchy(model, lookup, "01_hierarchy.png")
plot_weight_heatmap(model, save_path="02_weights.png")
plot_activation_map(model, data, [0, 1, 2], save_path="03_activations.png")
plot_umatrix(model, save_path="04_umatrix.png")
plot_cluster_distribution(model, save_path="05_clusters.png")

plt.close('all')  # Clean up
print("✓ All visualizations saved!")
```

---

## Styling and Customization

### Color Maps

All plotting functions support matplotlib colormaps:

```python
# Sequential
plot_weight_heatmap(model, cmap='viridis')  # Default
plot_weight_heatmap(model, cmap='plasma')
plot_weight_heatmap(model, cmap='cividis')

# Diverging
plot_umatrix(model, cmap='RdBu_r')
plot_umatrix(model, cmap='coolwarm')

# Perceptually uniform
plot_activation_map(model, data, [0], cmap='YlOrRd')
```

### Figure Sizes

Customize dimensions for different outputs:

```python
# Large for presentations
fig = plot_cluster_distribution(model, figsize=(16, 10))

# Compact for papers
fig = plot_cluster_quality(data, labels, figsize=(10, 4))

# Square for posters
fig = plot_umatrix(model, figsize=(8, 8))
```

### Publication Quality

Export high-resolution figures:

```python
fig = plot_weight_heatmap(model)
fig.savefig('figure1.pdf', dpi=300, bbox_inches='tight')

fig = plot_umatrix(model)
fig.savefig('figure2.png', dpi=600, bbox_inches='tight')
```

---

## Next Steps

- [Tutorials](tutorials/01_basic_visualization.md) - Learn step-by-step
- [API Reference](api/plotting.md) - Complete function documentation
- [Examples](https://github.com/dadmaan/ghsom-toolkits/tree/main/examples) - Runnable code

## Tips

1. **Start simple**: Begin with `visualize_ghsom_hierarchy()` to understand structure
2. **Explore interactively**: Use `launch_dashboard()` for initial exploration
3. **Generate reports**: Use `generate_report()` for comprehensive analysis
4. **Compare systematically**: Use `compare_models()` when tuning hyperparameters
5. **Export high-res**: Always save publication figures at 300+ DPI
