# Interactive Tools

The interactive module provides web-based dashboards and exploration utilities for GHSOM models using Plotly and Dash.

!!! note "Installation Required"
    Interactive features require additional dependencies:
    ```bash
    pip install ghsom-toolkits[interactive]
    ```

## Dashboard

### launch_dashboard

```python
launch_dashboard(
    node,
    data=None,
    lookup_table=None,
    port=8050,
    debug=False
)
```

Launch an interactive web-based dashboard for exploring GHSOM models. The dashboard provides:

- Interactive hierarchy visualization
- Node statistics and information
- Sample path tracing
- Neuron weight exploration
- Real-time metrics

**Parameters:**

- `node` (object): Root GHSOM node
- `data` (numpy.ndarray, optional): Original training data for enhanced features
- `lookup_table` (dict, optional): Lookup table mapping node IDs to nodes. If None, builds automatically.
- `port` (int, optional): Port number for the web server (default: 8050)
- `debug` (bool, optional): Enable debug mode (default: False)

**Example:**

```python
from ghsom import GHSOM
from ghsom_toolkits.interactive import launch_dashboard
import numpy as np

# Train model
data = np.random.rand(200, 10)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=50)

# Launch dashboard
launch_dashboard(model, data=data, port=8050)
# Open browser to http://localhost:8050
```

**Dashboard Features:**

- **Hierarchy View**: Interactive tree visualization with zoom/pan
- **Node Inspector**: Click nodes to see detailed statistics
- **Sample Tracer**: Track how samples flow through the hierarchy
- **Weight Heatmaps**: Visualize neuron weights interactively
- **Performance Metrics**: Real-time cluster quality metrics

---

## Explorer Utilities

### explore_neuron

```python
explore_neuron(
    node,
    neuron_position,
    data=None
)
```

Extract detailed information about a specific neuron in the GHSOM map.

**Parameters:**

- `node` (object): GHSOM node containing the map
- `neuron_position` (tuple of (int, int)): Neuron position (row, col)
- `data` (numpy.ndarray, optional): Input data for activation analysis

**Returns:** Dictionary with keys:
- `position`: (row, col) tuple
- `weights`: Weight vector
- `has_child`: Boolean indicating if neuron has a child map
- `child_node`: Child node object if exists
- `activations`: List of activation strengths for data samples (if data provided)

**Example:**

```python
from ghsom_toolkits.interactive import explore_neuron

# Explore specific neuron
info = explore_neuron(model, neuron_position=(0, 0), data=data)
print(f"Position: {info['position']}")
print(f"Weights: {info['weights']}")
print(f"Has child: {info['has_child']}")
if data is not None:
    print(f"Mean activation: {np.mean(info['activations'])}")
```

---

### trace_sample_path

```python
trace_sample_path(
    root_node,
    sample,
    lookup_table=None
)
```

Trace the path a sample takes through the GHSOM hierarchy from root to leaf.

**Parameters:**

- `root_node` (object): Root node of the GHSOM tree
- `sample` (numpy.ndarray): Input sample, shape (n_features,)
- `lookup_table` (dict, optional): Lookup table for node IDs

**Returns:** List of dictionaries, each containing:
- `node_id`: Node identifier
- `node`: Node object
- `level`: Depth in hierarchy
- `bmu_position`: Best Matching Unit position (row, col)
- `distance`: Distance from sample to BMU

**Example:**

```python
from ghsom_toolkits.interactive import trace_sample_path

# Trace how a sample traverses the hierarchy
sample = data[0]
path = trace_sample_path(model, sample)

for step in path:
    print(f"Level {step['level']}: BMU at {step['bmu_position']}, "
          f"distance={step['distance']:.3f}")
```

---

### get_subtree

```python
get_subtree(
    node,
    lookup_table,
    root_id="root"
)
```

Extract a subtree from a specific node for focused exploration.

**Parameters:**

- `node` (object): Root node of the subtree to extract
- `lookup_table` (dict): Full lookup table
- `root_id` (str, optional): ID for the subtree root (default: "root")

**Returns:** Dictionary with keys:
- `root_node`: The subtree root node
- `lookup_table`: New lookup table for the subtree
- `num_nodes`: Total nodes in subtree
- `depth`: Maximum depth of subtree

**Example:**

```python
from ghsom_toolkits.interactive import get_subtree

# Extract subtree for focused analysis
subtree_info = get_subtree(
    node=child_node,
    lookup_table=full_lookup,
    root_id="subtree_root"
)

print(f"Subtree has {subtree_info['num_nodes']} nodes")
print(f"Maximum depth: {subtree_info['depth']}")

# Can visualize just the subtree
visualize_ghsom_hierarchy(
    subtree_info['root_node'],
    subtree_info['lookup_table'],
    "subtree.png"
)
```

---

## Dashboard Usage Patterns

### Basic Dashboard

```python
# Minimal setup
launch_dashboard(model, data=data)
```

### Custom Port

```python
# Run on different port (useful for multiple models)
launch_dashboard(model, data=data, port=8051)
```

### Multiple Models Comparison

```python
# Launch separate dashboards for comparison
import threading

def launch_model1():
    launch_dashboard(model1, data=data1, port=8050)

def launch_model2():
    launch_dashboard(model2, data=data2, port=8051)

thread1 = threading.Thread(target=launch_model1)
thread2 = threading.Thread(target=launch_model2)

thread1.start()
thread2.start()

# Open both: http://localhost:8050 and http://localhost:8051
```

### Sample Path Analysis

```python
# Trace multiple samples
for i in range(5):
    sample = data[i]
    path = trace_sample_path(model, sample)

    print(f"\nSample {i} path:")
    for step in path:
        print(f"  Level {step['level']}: "
              f"BMU=({step['bmu_position'][0]},{step['bmu_position'][1]}), "
              f"dist={step['distance']:.4f}")
```

### Neuron Exploration

```python
# Systematically explore all neurons in root map
for row in range(model.rows):
    for col in range(model.columns):
        info = explore_neuron(model, (row, col))
        if info['has_child']:
            print(f"Neuron ({row},{col}) has child with "
                  f"{info['child_node'].rows}x{info['child_node'].columns} map")
```

---

## Interactive Workflow Example

Complete workflow combining dashboard and utilities:

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.interactive import (
    launch_dashboard,
    trace_sample_path,
    explore_neuron
)

# 1. Train model
data = np.random.rand(500, 20)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=100)

# 2. Explore interesting samples programmatically
interesting_samples = [0, 10, 50]
for idx in interesting_samples:
    path = trace_sample_path(model, data[idx])
    print(f"\nSample {idx} path depth: {len(path)}")

# 3. Inspect neurons along the path
sample_path = trace_sample_path(model, data[0])
for step in sample_path:
    bmu_pos = step['bmu_position']
    neuron_info = explore_neuron(step['node'], bmu_pos)
    print(f"Level {step['level']}, BMU {bmu_pos}: "
          f"has_child={neuron_info['has_child']}")

# 4. Launch interactive dashboard for visual exploration
print("\nLaunching interactive dashboard...")
launch_dashboard(model, data=data, port=8050)
```

---

## Browser Compatibility

The dashboard is tested and works with:
- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

For best performance, use a modern browser with JavaScript enabled.
