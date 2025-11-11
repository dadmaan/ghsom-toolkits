# Tutorial: Interactive Dashboard

Learn how to use the interactive web-based dashboard to explore GHSOM models in real-time.

## Prerequisites

```bash
pip install ghsom-toolkits[interactive]
```

This installs Dash and Plotly for interactive visualizations.

## Step 1: Prepare Your Model

Start with a trained GHSOM model:

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model

# Train model
np.random.seed(42)
data = np.random.rand(300, 15)

ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=100)

# Adapt for ghsom-toolkits
adapted_model = adapt_model(model, input_dataset_size=len(data))
```

## Step 2: Launch the Dashboard

Launch the interactive dashboard:

```python
from ghsom_toolkits.interactive import launch_dashboard

# Launch on default port (8050)
launch_dashboard(
    node=adapted_model,
    data=data,
    port=8050
)
```

**Output:**
```
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'ghsom_toolkits.interactive.dashboard'
 * Debug mode: off
```

Open your browser to `http://localhost:8050` to view the dashboard!

## Dashboard Features

### 1. Hierarchy Overview

The main view shows the complete hierarchy as an interactive tree:

- **Zoom**: Scroll to zoom in/out
- **Pan**: Click and drag to move around
- **Hover**: See node details on hover

### 2. Node Inspector

Click any node to see detailed information:

- Map dimensions (rows × columns)
- Number of children
- Dataset size
- Level in hierarchy
- Position in parent map

### 3. Sample Tracer

Enter a sample index to trace its path through the hierarchy:

```python
# The dashboard will show:
# - Path from root to leaf
# - BMU positions at each level
# - Distances at each step
```

### 4. Weight Visualization

View weight vectors for any node interactively.

## Step 3: Programmatic Exploration

While the dashboard runs, you can explore programmatically:

```python
from ghsom_toolkits.interactive import trace_sample_path, explore_neuron

# Trace a specific sample
sample_idx = 0
path = trace_sample_path(adapted_model, data[sample_idx])

print(f"\nSample {sample_idx} path:")
for step in path:
    print(f"  Level {step['level']}: "
          f"BMU at {step['bmu_position']}, "
          f"distance={step['distance']:.4f}")

# Explore a specific neuron
neuron_info = explore_neuron(
    node=adapted_model,
    neuron_position=(0, 0),
    data=data
)

print(f"\nNeuron (0,0) info:")
print(f"  Has child: {neuron_info['has_child']}")
print(f"  Weight vector shape: {neuron_info['weights'].shape}")
if data is not None:
    print(f"  Mean activation: {np.mean(neuron_info['activations']):.3f}")
```

## Step 4: Multiple Dashboards

Compare models by launching multiple dashboards:

```python
import threading

def launch_model_dashboard(model, data, port, name):
    """Launch dashboard in a thread."""
    print(f"Launching {name} on port {port}")
    launch_dashboard(model, data, port=port)

# Train multiple models
models_data = []

for t1 in [0.3, 0.5, 0.7]:
    ghsom = GHSOM(input_dataset=data, t1=t1, t2=0.05)
    m = ghsom.train(epochs_number=50)
    adapted = adapt_model(m, input_dataset_size=len(data))
    models_data.append((adapted, f"t1={t1}"))

# Launch dashboards in separate threads
threads = []
for i, (model, name) in enumerate(models_data):
    port = 8050 + i
    thread = threading.Thread(
        target=launch_model_dashboard,
        args=(model, data, port, name)
    )
    thread.daemon = True
    thread.start()
    threads.append(thread)
    print(f"→ {name}: http://localhost:{port}")

# Keep main thread alive
try:
    while True:
        pass
except KeyboardInterrupt:
    print("\nShutting down dashboards...")
```

## Step 5: Advanced Sample Tracing

Trace multiple samples and analyze patterns:

```python
from ghsom_toolkits.interactive import trace_sample_path
import pandas as pd

# Trace multiple samples
traces = []
for idx in range(10):
    path = trace_sample_path(adapted_model, data[idx])
    traces.append({
        'sample_idx': idx,
        'path_length': len(path),
        'final_distance': path[-1]['distance'] if path else 0,
        'final_bmu': path[-1]['bmu_position'] if path else None
    })

# Convert to DataFrame for analysis
df = pd.DataFrame(traces)
print("\nSample Trace Analysis:")
print(df)

print(f"\nAverage path length: {df['path_length'].mean():.1f}")
print(f"Average final distance: {df['final_distance'].mean():.4f}")
```

## Step 6: Custom Dashboard Integration

Create a custom script that combines dashboard and analysis:

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model
from ghsom_toolkits.interactive import launch_dashboard, trace_sample_path
from ghsom_toolkits.plotting import plot_cluster_distribution
import matplotlib.pyplot as plt

# 1. Train model
data = np.random.rand(500, 20)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=100)
adapted = adapt_model(model, input_dataset_size=len(data))

# 2. Generate static visualizations first
print("Generating static visualizations...")
plot_cluster_distribution(adapted, save_path='cluster_dist.png')
plt.close()

# 3. Analyze interesting samples
print("\nAnalyzing sample paths...")
interesting_samples = []
for i in range(len(data)):
    path = trace_sample_path(adapted, data[i])
    if len(path) > 3:  # Deep paths
        interesting_samples.append(i)

print(f"Found {len(interesting_samples)} samples with deep paths")
print(f"Indices: {interesting_samples[:10]}")  # Show first 10

# 4. Launch interactive dashboard
print("\nLaunching dashboard...")
print("Try exploring samples:", interesting_samples[:5])
launch_dashboard(adapted, data=data, port=8050)
```

## Dashboard Customization

### Custom Port

Useful if port 8050 is already in use:

```python
launch_dashboard(adapted_model, data=data, port=8888)
# Access at: http://localhost:8888
```

### Debug Mode

Enable debug mode for development:

```python
launch_dashboard(adapted_model, data=data, debug=True)
# Shows detailed error messages and auto-reloads on changes
```

## Use Cases

### 1. Model Inspection

Use the dashboard to:
- Verify hierarchy structure
- Check cluster sizes
- Identify overfit/underfit regions

### 2. Sample Analysis

- Find which samples are hard to cluster (high distances)
- Identify outliers (unique paths)
- Understand cluster membership

### 3. Hyperparameter Tuning

- Compare models side-by-side in multiple browser tabs
- Quickly assess hierarchy depth and structure
- Evaluate cluster quality visually

### 4. Presentation and Demos

- Interactive exploration during presentations
- Allow stakeholders to explore the model
- Real-time Q&A about model behavior

## Complete Example: End-to-End Workflow

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits.interactive import (
    launch_dashboard,
    trace_sample_path,
    explore_neuron
)
from ghsom_toolkits import visualize_ghsom_hierarchy

# 1. Generate synthetic data with clusters
np.random.seed(42)
cluster_centers = [
    np.array([0.2, 0.2]),
    np.array([0.8, 0.8]),
    np.array([0.2, 0.8]),
]
data = []
for center in cluster_centers:
    samples = np.random.randn(100, 2) * 0.1 + center
    data.append(samples)
data = np.vstack(data)

# 2. Train GHSOM
print("Training GHSOM...")
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=100)

# 3. Adapt and visualize
adapted = adapt_model(model, input_dataset_size=len(data))
lookup = build_lookup_table(adapted)

print("Generating hierarchy visualization...")
visualize_ghsom_hierarchy(adapted, lookup, "hierarchy.png")

# 4. Pre-analysis
print("\n=== Pre-Dashboard Analysis ===")
print(f"Total nodes: {len(lookup)}")

# Find deepest paths
depths = []
for i in range(min(50, len(data))):
    path = trace_sample_path(adapted, data[i])
    depths.append(len(path))

print(f"Path depth range: {min(depths)} - {max(depths)}")
print(f"Average depth: {np.mean(depths):.1f}")

# Explore root neurons
print("\nRoot map neurons with children:")
for r in range(adapted.rows):
    for c in range(adapted.columns):
        info = explore_neuron(adapted, (r, c))
        if info['has_child']:
            print(f"  ({r},{c}): has child map")

# 5. Launch dashboard
print("\n=== Launching Interactive Dashboard ===")
print("Try exploring:")
print(f"  - Deepest path sample indices: {[i for i, d in enumerate(depths) if d == max(depths)][:3]}")
print(f"  - Root neurons with children")
print("\nPress Ctrl+C to stop the dashboard")

launch_dashboard(adapted, data=data, port=8050)
```

## Next Steps

- [Tutorial 3: Model Comparison](03_model_comparison.md) - Compare multiple models
- [Tutorial 4: Generating Reports](04_generating_reports.md) - Create HTML reports
- [API Reference: Interactive Tools](../api/interactive.md) - Full API documentation

## Troubleshooting

**Dashboard doesn't load**
- Check if port is already in use: `lsof -i :8050` (macOS/Linux)
- Try a different port: `launch_dashboard(model, data, port=8888)`

**"ModuleNotFoundError: No module named 'dash'"**
- Install interactive dependencies: `pip install ghsom-toolkits[interactive]`

**Browser shows "ERR_CONNECTION_REFUSED"**
- Dashboard may have crashed - check terminal for error messages
- Ensure firewall isn't blocking the port
