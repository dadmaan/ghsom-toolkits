# Adapters

The adapters module provides compatibility layers for using different GHSOM implementations with ghsom-toolkits. Currently supports **ghsom-py**.

## Overview

The adapter system allows ghsom-toolkits to work seamlessly with models from different GHSOM libraries by wrapping them in a unified interface. This eliminates the need for manual data structure conversion.

### Key Features

- ✅ **Zero-copy wrapping** - No data duplication
- ✅ **Auto-detection** - Automatically identifies input type
- ✅ **Lazy loading** - Efficient memory usage
- ✅ **Full compatibility** - Works with all ghsom-toolkits features
- ✅ **<5% overhead** - Minimal performance impact

---

## ghsom-py Integration

### adapt_model

```python
adapt_model(model, input_dataset_size=None)
```

Automatically adapt a ghsom-py model to the ghsom-toolkits interface. Supports multiple input types with smart auto-detection.

**Parameters:**

- `model` (object): Model from ghsom-py training. Can be:
  - `Neuron` object with `child_map` (typical output from `ghsom.train()`)
  - `GSOM` object
  - Already compatible `GHSOMNode` (pass-through)
- `input_dataset_size` (int, optional): Total dataset size. Auto-detected if not provided.

**Returns:** Adapted node compatible with all ghsom-toolkits functions

**Example:**

```python
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits import visualize_ghsom_hierarchy
import numpy as np

# 1. Train with ghsom-py
data = np.random.rand(200, 10)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
result = ghsom.train(epochs_number=50)  # Returns Neuron with child_map

# 2. Adapt to ghsom-toolkits interface
model = adapt_model(result, input_dataset_size=len(data))

# 3. Build lookup table
lookup = build_lookup_table(model)

# 4. Use with any ghsom-toolkits function
visualize_ghsom_hierarchy(model, lookup, "hierarchy.png")
```

**Supported Input Types:**

1. **Neuron with child_map** (most common):
   ```python
   # ghsom.train() returns a Neuron with child_map attribute
   result = ghsom.train(epochs_number=50)
   model = adapt_model(result)
   ```

2. **GSOM object directly**:
   ```python
   # Access GSOM directly from GHSOM instance
   gsom = ghsom.gsom  # Root GSOM
   model = adapt_model(gsom)
   ```

3. **Already compatible GHSOMNode**:
   ```python
   # If already a GHSOMNode, returns as-is
   model = adapt_model(existing_node)
   ```

---

### GHSOMNodeAdapter

```python
GHSOMNodeAdapter(gsom_object, position=(0, 0), level=0, input_dataset_size=None)
```

Low-level adapter class that wraps ghsom-py GSOM objects. Usually you should use `adapt_model()` instead of this directly.

**Properties:**

- `rows` (int): Number of rows in the map
- `columns` (int): Number of columns in the map
- `map` (list of lists): 2D grid of neurons
- `children` (list): List of child nodes
- `level` (int): Depth in hierarchy (0 = root)
- `position` (tuple): Position as (row, col) in parent map
- `input_dataset_size` (int): Number of samples in this node's dataset

**Example (Advanced Usage):**

```python
from ghsom_toolkits.adapters import GHSOMNodeAdapter

# Manually wrap a GSOM object
gsom = ghsom.gsom
adapted = GHSOMNodeAdapter(
    gsom_object=gsom,
    position=(0, 0),
    level=0,
    input_dataset_size=len(data)
)

print(f"Map shape: {adapted.rows}x{adapted.columns}")
print(f"Children: {len(adapted.children)}")
```

---

### build_lookup_table

```python
build_lookup_table(node, prefix="root")
```

Build a lookup table mapping node IDs to node objects. Required for most visualization functions.

**Parameters:**

- `node` (object): Root node of the GHSOM tree
- `prefix` (str, optional): Prefix for node IDs (default: "root")

**Returns:** Dictionary mapping node ID strings to node objects

**Example:**

```python
from ghsom_toolkits.adapters import adapt_model, build_lookup_table

# After adapting model
model = adapt_model(ghsom_result)

# Build lookup table
lookup = build_lookup_table(model, prefix="my_model")

# Use with visualizations
visualize_ghsom_hierarchy(model, lookup, "output.png")
```

**Lookup Table Structure:**

```python
{
    "root": <root_node>,
    "root_0_0": <child at position (0,0)>,
    "root_0_1": <child at position (0,1)>,
    "root_0_0_1_2": <grandchild at position (1,2) under root_0_0>,
    ...
}
```

---

## Complete Workflow Examples

### Basic ghsom-py Integration

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits.plotting import (
    plot_weight_heatmap,
    plot_activation_map,
    plot_umatrix
)
from ghsom_toolkits import visualize_ghsom_hierarchy

# 1. Train with ghsom-py
data = np.random.rand(500, 20)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
result = ghsom.train(epochs_number=100)

# 2. Adapt for ghsom-toolkits
model = adapt_model(result, input_dataset_size=len(data))
lookup = build_lookup_table(model)

# 3. Visualize everything
visualize_ghsom_hierarchy(model, lookup, "hierarchy.png")
plot_weight_heatmap(model, save_path="weights.png")
plot_activation_map(model, data, [0, 1, 2], save_path="activations.png")
plot_umatrix(model, save_path="umatrix.png")
```

### With Interactive Dashboard

```python
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits.interactive import launch_dashboard
import numpy as np

# Train model
data = np.random.rand(300, 15)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
result = ghsom.train(epochs_number=50)

# Adapt and launch dashboard
model = adapt_model(result, input_dataset_size=len(data))
lookup = build_lookup_table(model)

launch_dashboard(model, data=data, lookup_table=lookup, port=8050)
```

### With Model Comparison

```python
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model
from ghsom_toolkits.analysis import compare_models, plot_comparison
import numpy as np

data = np.random.rand(200, 10)

# Train multiple ghsom-py models
configs = [
    {'t1': 0.7, 't2': 0.1, 'name': 'Loose'},
    {'t1': 0.5, 't2': 0.05, 'name': 'Medium'},
    {'t1': 0.3, 't2': 0.02, 'name': 'Tight'}
]

adapted_models = []
names = []

for config in configs:
    ghsom = GHSOM(input_dataset=data, t1=config['t1'], t2=config['t2'])
    result = ghsom.train(epochs_number=50)

    # Adapt each model
    model = adapt_model(result, input_dataset_size=len(data))
    adapted_models.append(model)
    names.append(config['name'])

# Compare adapted models
comparison = compare_models(adapted_models, data, model_names=names)
print(comparison)

plot_comparison(comparison, plot_type='bar', save_path='comparison.png')
```

### With Report Generation

```python
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model
from ghsom_toolkits.analysis import generate_report
import numpy as np

# Train
data = np.random.rand(400, 25)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
result = ghsom.train(epochs_number=100)

# Adapt
model = adapt_model(result, input_dataset_size=len(data))

# Generate comprehensive report
generate_report(
    node=model,
    data=data,
    output_path='ghsom_analysis.html',
    model_name='My GHSOM Model'
)
```

---

## Performance Characteristics

### Memory Overhead

- **Wrapper objects only**: ~100 bytes per node
- **No data duplication**: Original weights/maps are referenced
- **Lazy children**: Built only when accessed

### Computational Overhead

| Operation | Overhead | Notes |
|-----------|----------|-------|
| Adapt model | < 1ms | One-time cost |
| Build lookup table | ~2ms per 100 nodes | One-time cost |
| Property access | ~5% | Negligible in practice |
| Child iteration | < 1% | Cached after first access |

### Benchmark Results

```python
# 1000 node hierarchy, typical operations:
adapt_model(result)              # 1.2 ms
build_lookup_table(model)        # 18 ms
visualize_ghsom_hierarchy(...)   # 142 ms (same as without adapter)
```

---

## Compatibility Matrix

| ghsom-py Version | Adapter Support | Notes |
|------------------|-----------------|-------|
| 0.1.x | ✅ Full | Tested and verified |
| 0.2.x | ✅ Full | Expected compatible |
| 1.0.x | ⚠️ Unknown | May require updates |

---

## Troubleshooting

### "AttributeError: 'Neuron' object has no attribute 'child_map'"

**Solution:** Your ghsom-py version may return a different object type. Try:

```python
# Access GSOM directly
gsom = ghsom.gsom  # or ghsom.root
model = adapt_model(gsom)
```

### "Cannot detect input_dataset_size"

**Solution:** Provide it explicitly:

```python
model = adapt_model(result, input_dataset_size=len(data))
```

### Visualization functions fail with adapted models

**Solution:** Ensure you build the lookup table:

```python
model = adapt_model(result)
lookup = build_lookup_table(model)  # Don't forget this!
visualize_ghsom_hierarchy(model, lookup, "output.png")
```

---

## Future Adapters

Planned support for additional GHSOM implementations:

- **GHSOM-Java**: Adapter for original Java implementation
- **PyMVPA GHSOM**: Adapter for PyMVPA's GHSOM
- **Custom formats**: Generic adapter for pickled/saved models

If you need an adapter for a specific GHSOM implementation, please [open an issue](https://github.com/dadmaan/ghsom-toolkits/issues).
