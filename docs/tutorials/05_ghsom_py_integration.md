# Tutorial: ghsom-py Integration

Deep dive into using ghsom-toolkits with the ghsom-py library, including advanced patterns and troubleshooting.

## Prerequisites

```bash
pip install ghsom-py ghsom-toolkits
```

## Understanding the Integration

ghsom-py and ghsom-toolkits are designed to work together:

- **ghsom-py**: Core training algorithm, lightweight, minimal dependencies
- **ghsom-toolkits**: Visualization and analysis tools

The adapter layer bridges the two packages seamlessly.

## Basic Integration Pattern

### Step 1: Train with ghsom-py

```python
import numpy as np
from ghsom import GHSOM

# Prepare data
data = np.random.rand(300, 15)

# Train GHSOM
ghsom = GHSOM(
    input_dataset=data,
    t1=0.5,  # Growth threshold
    t2=0.05,  # Expansion threshold
    learning_rate=0.1,
    gaussian_sigma=1.0
)

# Training returns a Neuron with child_map
result = ghsom.train(epochs_number=50, n_workers=-1)

print(f"Training complete!")
print(f"Result type: {type(result)}")
```

### Step 2: Adapt to ghsom-toolkits

```python
from ghsom_toolkits.adapters import adapt_model, build_lookup_table

# Adapt the model
model = adapt_model(result, input_dataset_size=len(data))

# Build lookup table for visualizations
lookup = build_lookup_table(model)

print(f"Adapted model ready with {len(lookup)} nodes")
```

### Step 3: Visualize

```python
from ghsom_toolkits import visualize_ghsom_hierarchy
from ghsom_toolkits.plotting import plot_weight_heatmap, plot_umatrix

# Hierarchy
visualize_ghsom_hierarchy(model, lookup, "hierarchy.png")

# Heatmaps
plot_weight_heatmap(model, save_path="weights.png")
plot_umatrix(model, save_path="umatrix.png")
```

## Understanding ghsom-py Objects

### Neuron Object

The training result is a `Neuron` object representing the root:

```python
# Access Neuron properties
print(f"Map dimensions: {result.map_shape()}")  # e.g., (3, 3)

# Check if neuron has children
if hasattr(result, 'child_map') and result.child_map:
    print("Root neuron has child maps")

# Access the underlying GSOM
if hasattr(result, 'gsom'):
    gsom = result.gsom
    print(f"GSOM shape: {gsom.map_shape()}")
```

### GSOM Object

You can also work directly with GSOM objects:

```python
# Access GSOM from GHSOM instance
gsom = ghsom.gsom

# Adapt GSOM directly
from ghsom_toolkits.adapters import adapt_model

model = adapt_model(gsom, input_dataset_size=len(data))
```

## Advanced Adapter Usage

### Manual Adaptation

For fine-grained control:

```python
from ghsom_toolkits.adapters import GHSOMNodeAdapter

# Manually create adapter
adapted_root = GHSOMNodeAdapter(
    gsom_object=ghsom.gsom,
    position=(0, 0),
    level=0,
    input_dataset_size=len(data)
)

# Access properties
print(f"Rows: {adapted_root.rows}")
print(f"Columns: {adapted_root.columns}")
print(f"Children: {len(adapted_root.children)}")
```

### Custom Lookup Tables

Build lookup tables with custom node IDs:

```python
def custom_build_lookup(node, prefix="N", depth=0):
    """Build lookup with depth-aware IDs."""
    lookup = {}
    lookup[f"{prefix}_L{depth}"] = node

    for i, child in enumerate(node.children):
        child_prefix = f"{prefix}_L{depth}_C{i}"
        child_lookup = custom_build_lookup(child, child_prefix, depth + 1)
        lookup.update(child_lookup)

    return lookup

lookup = custom_build_lookup(model)
print(list(lookup.keys())[:5])  # ['N_L0', 'N_L0_C0_L1', ...]
```

## Working with Training Results

### Access Training Metrics

If ghsom-py provides training history:

```python
# Some versions may provide training history
if hasattr(ghsom, 'training_history'):
    history = ghsom.training_history

    from ghsom_toolkits.plotting import plot_growth_timeline
    plot_growth_timeline(history, save_path='training.png')
```

### Save and Load Models

```python
import pickle

# Save adapted model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load and use
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Rebuild lookup table
lookup = build_lookup_table(loaded_model)
visualize_ghsom_hierarchy(loaded_model, lookup, "loaded_hierarchy.png")
```

## Complete Workflow Examples

### Example 1: Real Dataset Analysis

```python
import numpy as np
import pandas as pd
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits import visualize_ghsom_hierarchy
from ghsom_toolkits.plotting import (
    plot_weight_heatmap,
    plot_cluster_distribution,
    plot_activation_map
)
from ghsom_toolkits.analysis import generate_report

# Load real data (example: Iris dataset)
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
data = StandardScaler().fit_transform(iris.data)

print(f"Data shape: {data.shape}")

# Train GHSOM
ghsom = GHSOM(
    input_dataset=data,
    t1=0.5,
    t2=0.05,
    learning_rate=0.1
)

result = ghsom.train(epochs_number=100, n_workers=-1)

# Adapt
model = adapt_model(result, input_dataset_size=len(data))
lookup = build_lookup_table(model)

# Comprehensive analysis
print("Generating visualizations...")
visualize_ghsom_hierarchy(model, lookup, "iris_hierarchy.png")
plot_weight_heatmap(model, save_path="iris_weights.png")
plot_cluster_distribution(model, save_path="iris_clusters.png")
plot_activation_map(model, data, [0, 50, 100], save_path="iris_activations.png")

# Generate report
generate_report(
    node=model,
    data=data,
    output_path='iris_report.html',
    model_name='Iris Dataset GHSOM'
)

print("✓ Complete analysis saved")
```

### Example 2: Iterative Refinement

```python
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model
from ghsom_toolkits.analysis import compare_models

# Start with baseline
data = np.random.rand(500, 20)

configs = []
models = []
names = []

# Iteratively refine hyperparameters
base_t1, base_t2 = 0.5, 0.05

for i, adjustment in enumerate([0.8, 0.9, 1.0, 1.1, 1.2]):
    t1 = base_t1 * adjustment
    t2 = base_t2 * adjustment

    print(f"Iteration {i+1}: t1={t1:.2f}, t2={t2:.3f}")

    ghsom = GHSOM(input_dataset=data, t1=t1, t2=t2)
    result = ghsom.train(epochs_number=50)

    model = adapt_model(result, input_dataset_size=len(data))
    models.append(model)
    names.append(f"Iter{i+1}: {adjustment:.1f}x")

# Compare iterations
comparison = compare_models(models, data, names)
print("\nIterative Refinement Results:")
print(comparison)

# Find best
best_idx = comparison['Quantization Error'].idxmin()
print(f"\nBest configuration: {names[best_idx]}")
```

### Example 3: Multi-Dataset Comparison

```python
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model
from ghsom_toolkits.analysis import generate_report
import numpy as np

# Different datasets
datasets = {
    'uniform': np.random.rand(200, 10),
    'gaussian': np.random.randn(200, 10),
    'clusters': np.vstack([
        np.random.randn(100, 10) * 0.5 + 2,
        np.random.randn(100, 10) * 0.5 - 2
    ])
}

# Train on each dataset
for name, data in datasets.items():
    print(f"\nProcessing {name} dataset...")

    ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
    result = ghsom.train(epochs_number=50)

    model = adapt_model(result, input_dataset_size=len(data))

    # Generate report
    generate_report(
        node=model,
        data=data,
        output_path=f'report_{name}.html',
        model_name=f'GHSOM on {name} data'
    )

    print(f"  ✓ Report: report_{name}.html")
```

## Performance Optimization

### Parallel Training

ghsom-py supports parallel training:

```python
from ghsom import GHSOM

# Use all CPU cores
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
result = ghsom.train(epochs_number=100, n_workers=-1)  # -1 = all cores

# Or specify number of workers
result = ghsom.train(epochs_number=100, n_workers=4)
```

### Efficient Adaptation

Adapt once, reuse many times:

```python
# Adapt once
model = adapt_model(result, input_dataset_size=len(data))
lookup = build_lookup_table(model)

# Reuse for multiple visualizations
visualize_ghsom_hierarchy(model, lookup, "v1.png")
visualize_ghsom_hierarchy(model, lookup, "v2.pdf")  # Different format

from ghsom_toolkits.plotting import (
    plot_weight_heatmap,
    plot_umatrix,
    plot_cluster_distribution
)

# All use the same adapted model
plot_weight_heatmap(model, save_path="weights.png")
plot_umatrix(model, save_path="umatrix.png")
plot_cluster_distribution(model, save_path="clusters.png")
```

## Troubleshooting

### Issue: "Cannot adapt model - unknown type"

**Cause**: ghsom-py version returns different object type

**Solution**: Try accessing GSOM directly

```python
# Instead of:
model = adapt_model(result)

# Try:
if hasattr(result, 'gsom'):
    model = adapt_model(result.gsom, input_dataset_size=len(data))
else:
    # Or access from GHSOM instance
    model = adapt_model(ghsom.gsom, input_dataset_size=len(data))
```

### Issue: "No children found"

**Cause**: Model may not have hierarchical structure (flat GSOM)

**Solution**: Check hierarchy depth

```python
def has_hierarchy(node):
    """Check if model has hierarchical structure."""
    return len(node.children) > 0

model = adapt_model(result, input_dataset_size=len(data))
if has_hierarchy(model):
    print("Hierarchical structure found")
else:
    print("Flat GSOM (no hierarchy)")
```

### Issue: "Visualization looks wrong"

**Cause**: May not have called build_lookup_table

**Solution**: Always build lookup table

```python
# Required for visualizations
model = adapt_model(result, input_dataset_size=len(data))
lookup = build_lookup_table(model)  # Don't forget this!

visualize_ghsom_hierarchy(model, lookup, "output.png")
```

## Version Compatibility

| ghsom-py | ghsom-toolkits | Status |
|----------|----------------|--------|
| 0.1.x | 0.1.x+ | ✅ Tested |
| 0.2.x | 0.1.x+ | ✅ Expected to work |
| 1.0.x | ? | ⚠️ May need updates |

## Best Practices

1. **Always adapt models** before using ghsom-toolkits functions
2. **Build lookup tables** once and reuse
3. **Save adapted models** for later use
4. **Use parallel training** for large datasets
5. **Validate results** with multiple visualizations

## Next Steps

- [Gallery](../gallery.md) - See all visualization examples
- [API Reference](../api/plotting.md) - Complete API documentation
- [GitHub Issues](https://github.com/dadmaan/ghsom-toolkits/issues) - Report integration issues
