# Quick Start

Get started with ghsom-toolkits in 5 minutes!

## Installation

```bash
pip install ghsom-py ghsom-toolkits
```

Don't forget to install [Graphviz](installation.md#installing-graphviz) (system package).

## Your First Visualization

### Step 1: Train a GHSOM Model

```python
import numpy as np
from ghsom import GHSOM

# Generate sample data
np.random.seed(42)
data = np.random.rand(200, 10)

# Train GHSOM
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=50)

print("Training complete!")
```

### Step 2: Adapt and Visualize

```python
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits import visualize_ghsom_hierarchy

# Adapt model for ghsom-toolkits
adapted_model = adapt_model(model, input_dataset_size=len(data))
lookup = build_lookup_table(adapted_model)

# Create visualization
visualize_ghsom_hierarchy(
    adapted_model,
    lookup,
    "my_first_hierarchy.png"
)

print("✓ Visualization saved to my_first_hierarchy.png")
```

That's it! You've created your first GHSOM visualization.

## What's Next?

### Explore More Visualizations

#### Weight Heatmap

```python
from ghsom_toolkits.plotting import plot_weight_heatmap
import matplotlib.pyplot as plt

fig = plot_weight_heatmap(adapted_model)
plt.savefig("weights.png")
plt.show()
```

#### U-Matrix

```python
from ghsom_toolkits.plotting import plot_umatrix

fig = plot_umatrix(adapted_model)
plt.savefig("umatrix.png")
plt.show()
```

#### Cluster Distribution

```python
from ghsom_toolkits.plotting import plot_cluster_distribution

fig = plot_cluster_distribution(adapted_model)
plt.savefig("clusters.png")
plt.show()
```

### Launch Interactive Dashboard

```bash
# First install interactive dependencies
pip install ghsom-toolkits[interactive]
```

```python
from ghsom_toolkits.interactive import launch_dashboard

launch_dashboard(adapted_model, data=data, port=8050)
# Open http://localhost:8050 in your browser
```

### Compare Multiple Models

```python
from ghsom_toolkits.analysis import compare_models

# Train multiple models with different hyperparameters
models = []
names = []

for t1 in [0.3, 0.5, 0.7]:
    ghsom = GHSOM(input_dataset=data, t1=t1, t2=0.05)
    result = ghsom.train(epochs_number=50)
    models.append(adapt_model(result, input_dataset_size=len(data)))
    names.append(f"t1={t1}")

# Compare
comparison = compare_models(models, data, names)
print(comparison)
```

### Generate HTML Report

```bash
# Install interactive dependencies for reports
pip install ghsom-toolkits[interactive]
```

```python
from ghsom_toolkits.analysis import generate_report

generate_report(
    node=adapted_model,
    data=data,
    output_path="ghsom_report.html",
    model_name="My First GHSOM Model"
)

print("✓ Report saved to ghsom_report.html")
```

## Complete Example

Here's a complete workflow in one script:

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits import visualize_ghsom_hierarchy
from ghsom_toolkits.plotting import (
    plot_weight_heatmap,
    plot_umatrix,
    plot_cluster_distribution
)
import matplotlib.pyplot as plt

# 1. Generate data
np.random.seed(42)
data = np.random.rand(300, 15)
print(f"Data shape: {data.shape}")

# 2. Train GHSOM
print("Training GHSOM...")
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=100)

# 3. Adapt for ghsom-toolkits
print("Adapting model...")
adapted = adapt_model(model, input_dataset_size=len(data))
lookup = build_lookup_table(adapted)

# 4. Create all visualizations
print("Generating visualizations...")

# Hierarchy
visualize_ghsom_hierarchy(adapted, lookup, "01_hierarchy.png")

# Heatmaps
fig = plot_weight_heatmap(adapted, save_path="02_weights.png")
plt.close()

fig = plot_umatrix(adapted, save_path="03_umatrix.png")
plt.close()

# Cluster analysis
fig = plot_cluster_distribution(adapted, save_path="04_clusters.png")
plt.close()

print("\n✓ All visualizations saved!")
print("  - 01_hierarchy.png")
print("  - 02_weights.png")
print("  - 03_umatrix.png")
print("  - 04_clusters.png")
```

Save this as `quickstart_demo.py` and run:

```bash
python quickstart_demo.py
```

## Common Patterns

### Pattern 1: Train and Visualize

```python
# Train
from ghsom import GHSOM
model = GHSOM(input_dataset=data, t1=0.5, t2=0.05).train(epochs_number=50)

# Visualize
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits import visualize_ghsom_hierarchy

adapted = adapt_model(model, input_dataset_size=len(data))
lookup = build_lookup_table(adapted)
visualize_ghsom_hierarchy(adapted, lookup, "output.png")
```

### Pattern 2: Compare Configurations

```python
from ghsom_toolkits.analysis import compare_models

configs = [(0.3, 0.02), (0.5, 0.05), (0.7, 0.1)]
models, names = [], []

for t1, t2 in configs:
    m = GHSOM(input_dataset=data, t1=t1, t2=t2).train(epochs_number=50)
    models.append(adapt_model(m, input_dataset_size=len(data)))
    names.append(f"t1={t1}, t2={t2}")

comparison = compare_models(models, data, names)
print(comparison)
```

### Pattern 3: Interactive Exploration

```python
from ghsom_toolkits.interactive import launch_dashboard

# Train model
model = GHSOM(input_dataset=data, t1=0.5, t2=0.05).train(epochs_number=50)
adapted = adapt_model(model, input_dataset_size=len(data))

# Launch dashboard
launch_dashboard(adapted, data=data)
```

## Key Concepts

### Adapters

The adapter layer makes ghsom-py models compatible with ghsom-toolkits:

```python
# ghsom-py model
ghsom_py_model = ghsom.train(epochs_number=50)

# Adapted for ghsom-toolkits
adapted_model = adapt_model(ghsom_py_model, input_dataset_size=len(data))
```

### Lookup Tables

Lookup tables map node IDs to node objects (required for visualizations):

```python
lookup = build_lookup_table(adapted_model)
# Returns: {"root": <node>, "root_0_0": <child_node>, ...}
```

### Visualization Functions

All plotting functions follow similar patterns:

```python
# Basic usage
plot_function(model, save_path="output.png")

# With customization
plot_function(model, cmap="viridis", figsize=(10, 8), save_path="output.png")
```

## Tips for Success

1. **Always adapt models** before using ghsom-toolkits functions
2. **Build lookup tables** once and reuse them
3. **Close figures** with `plt.close()` to free memory
4. **Use virtual environments** to avoid dependency conflicts
5. **Start with basic visualizations** before advanced analysis

## Troubleshooting

**"GraphViz not found"**
- Install Graphviz system package: See [Installation Guide](installation.md#installing-graphviz)

**"No module named 'dash'"**
- Install interactive dependencies: `pip install ghsom-toolkits[interactive]`

**"Cannot adapt model"**
- Ensure you're passing the training result: `model = ghsom.train(...)` then `adapt_model(model)`
- Provide dataset size: `adapt_model(model, input_dataset_size=len(data))`

## Next Steps

### Tutorials

1. [Basic Visualization](tutorials/01_basic_visualization.md) - Hierarchy plots and highlighting
2. [Interactive Dashboard](tutorials/02_interactive_dashboard.md) - Web-based exploration
3. [Model Comparison](tutorials/03_model_comparison.md) - Hyperparameter tuning
4. [Generating Reports](tutorials/04_generating_reports.md) - Comprehensive reports
5. [ghsom-py Integration](tutorials/05_ghsom_py_integration.md) - Advanced patterns

### API Reference

- [Plotting Functions](api/plotting.md)
- [Interactive Tools](api/interactive.md)
- [Analysis Tools](api/analysis.md)
- [Adapters](api/adapters.md)

### Gallery

Browse the [Visualization Gallery](gallery.md) for examples of all plot types.

## Getting Help

- **Documentation**: [https://dadmaan.github.io/ghsom-toolkits/](https://dadmaan.github.io/ghsom-toolkits/)
- **Examples**: [GitHub examples directory](https://github.com/dadmaan/ghsom-toolkits/tree/main/examples)
- **Issues**: [GitHub Issues](https://github.com/dadmaan/ghsom-toolkits/issues)
