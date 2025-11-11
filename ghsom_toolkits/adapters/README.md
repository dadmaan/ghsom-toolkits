# GHSOM Adapters Module

The `ghsom_toolkits.adapters` module provides compatibility adapters for different GHSOM implementations, enabling seamless integration with ghsom-toolkits visualization and analysis tools.

## Overview

Different GHSOM implementations use different APIs and data structures. The adapters module bridges these differences, allowing you to use ghsom-toolkits with any supported GHSOM implementation.

**Currently Supported:**
- **ghsom-py**: Modern Python GHSOM implementation with GSOM/Neuron objects

## Quick Start

```python
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits import visualize_ghsom_hierarchy

# Train with ghsom-py
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
result = ghsom.train(epochs_number=50)

# Adapt to ghsom-toolkits interface (automatic detection)
model = adapt_model(result)
lookup = build_lookup_table(model)

# Use with any ghsom-toolkits function
visualize_ghsom_hierarchy(model, lookup, "hierarchy.png")
```

## API Reference

### adapt_model()

Smart auto-detection function that adapts GHSOM models to ghsom-toolkits interface.

```python
from ghsom_toolkits.adapters import adapt_model

# Automatically detects and adapts
model = adapt_model(ghsom_result)
```

**Supported Input Types:**
1. **Neuron with child_map** (from `ghsom.train()`)
2. **GSOM object** (direct GSOM reference)
3. **Compatible Node** (already compatible, returns as-is)

**Returns:**
- `GHSOMNodeAdapter` for ghsom-py objects
- Original object if already compatible

**Raises:**
- `ValueError` if model type not supported

### GHSOMNodeAdapter

Adapter class that wraps ghsom-py objects and provides ghsom-toolkits Node interface.

```python
from ghsom_toolkits.adapters import GHSOMNodeAdapter

# Manual adaptation (usually use adapt_model instead)
adapter = GHSOMNodeAdapter(gsom_object, level=0, position=(0, 0))

# Access standard properties
print(adapter.level)              # Hierarchy level
print(adapter.position)           # Position in parent
print(adapter.rows)               # Map rows
print(adapter.columns)            # Map columns
print(adapter.input_dataset_size) # Dataset size
print(adapter.map_dimensions)     # (rows, cols, depth)
print(adapter.children)           # Child nodes (lazy loaded)
```

**Key Features:**
- **Lazy Loading**: Children built only when accessed
- **Caching**: Children list cached after first access
- **Level Tracking**: Automatically tracks hierarchy depth
- **Zero Overhead**: Wraps original object without copying data

### build_lookup_table()

Helper function to build lookup tables from adapted models.

```python
from ghsom_toolkits.adapters import build_lookup_table

# Build lookup table
lookup = build_lookup_table(model, prefix="root")

# Access nodes
root_node = lookup["root"]
child_node = lookup["root_n1_1"]  # Node at position (1, 1)
```

## How It Works

### The Problem

**ghsom-py structure:**
```python
result = ghsom.train()  # Returns Neuron
gsom = result.child_map  # GSOM object
shape = gsom.map_shape()  # Method call
neurons = gsom.neurons  # Dict of neurons
neuron.child_map  # Child GSOM if exists
```

**ghsom-toolkits expects:**
```python
node.level          # Property
node.position       # Property
node.rows           # Property (not method)
node.columns        # Property (not method)
node.children       # List of child nodes
node.input_dataset_size  # Property
```

### The Solution

GHSOMNodeAdapter bridges the gap:

```python
# ghsom-py API → ghsom-toolkits API
adapter.rows → wrapped.map_shape()[0]
adapter.columns → wrapped.map_shape()[1]
adapter.children → [GHSOMNodeAdapter(n.child_map) for n in neurons if n.child_map]
```

## Performance

- **Zero Copy**: Wraps original objects without data duplication
- **Lazy Loading**: Children built only when accessed
- **Caching**: Computed properties cached for repeated access
- **Minimal Overhead**: <5% performance impact

## Examples

### Basic Usage

```python
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits import visualize_ghsom_hierarchy

# Train model
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
result = ghsom.train(epochs_number=50)

# Adapt and visualize
model = adapt_model(result)
lookup = build_lookup_table(model)
visualize_ghsom_hierarchy(model, lookup, "output.png")
```

### With Heatmaps

```python
from ghsom_toolkits.plotting.heatmaps import plot_weight_heatmap
from ghsom_toolkits.adapters import adapt_model

result = ghsom.train(epochs_number=50)
model = adapt_model(result)

plot_weight_heatmap(model, save_path="heatmap.png")
```

### With Interactive Dashboard

```python
from ghsom_toolkits.interactive import launch_dashboard
from ghsom_toolkits.adapters import adapt_model, build_lookup_table

result = ghsom.train(epochs_number=50)
model = adapt_model(result)
lookup = build_lookup_table(model)

launch_dashboard(model, data, lookup, port=8050)
```

## Error Handling

```python
from ghsom_toolkits.adapters import adapt_model

try:
    model = adapt_model(some_object)
except ValueError as e:
    print(f"Unsupported model type: {e}")
    # Use model.neurons, model.map_shape() to check ghsom-py API
```

## Testing

The adapter is thoroughly tested with:
- Mock ghsom-py objects
- Edge cases (empty maps, no children)
- Integration with all ghsom-toolkits features
- Performance benchmarks

Run tests:
```bash
pytest tests/adapters/
```

## Future Adapters

The adapter pattern is extensible. Future adapters could support:
- Other GHSOM implementations
- Legacy GHSOM formats
- Custom GHSOM variants

## Migration Guide

### From Manual Build Functions

If you had custom `build_lookup_table` functions in examples:

**Before:**
```python
def build_lookup_table(node, lookup=None, prefix="root"):
    # ... custom implementation
    return lookup

model = result.child_map if hasattr(result, 'child_map') else result
lookup = build_lookup_table(model)
```

**After:**
```python
from ghsom_toolkits.adapters import adapt_model, build_lookup_table

model = adapt_model(result)  # Auto-detects type
lookup = build_lookup_table(model)
```

### Benefits

✅ **Less Code**: Remove custom adapter logic
✅ **Auto-Detection**: No need to check `child_map`
✅ **Type Safety**: Clear error messages for unsupported types
✅ **Maintained**: Updates and bug fixes in one place
✅ **Tested**: Comprehensive test coverage

## Compatibility

- **Python**: 3.8+
- **ghsom-py**: All versions with GSOM/Neuron API
- **ghsom-toolkits**: 0.1.0+

## Support

For issues or questions:
1. Check that your ghsom-py model has `neurons` and `map_shape()` attributes
2. Use `adapt_model()` instead of manual extraction
3. Check error messages for unsupported types
4. Report issues with model type and available attributes
