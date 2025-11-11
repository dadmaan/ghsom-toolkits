# Analysis Tools

The analysis module provides tools for comparing multiple GHSOM models and generating comprehensive reports.

## Model Comparison

### compare_models

```python
compare_models(
    models,
    data,
    model_names=None,
    compute_qe=True
)
```

Compare multiple GHSOM models trained with different hyperparameters.

**Parameters:**

- `models` (list of objects): List of GHSOM model root nodes to compare
- `data` (numpy.ndarray): Input data used for training, shape (n_samples, n_features)
- `model_names` (list of str, optional): Names for each model (default: ["Model 1", "Model 2", ...])
- `compute_qe` (bool, optional): Whether to compute quantization error (default: True)

**Returns:** `pandas.DataFrame` with columns:
- `Model`: Model name
- `Total Nodes`: Total number of nodes in hierarchy
- `Max Depth`: Maximum depth of hierarchy
- `Leaf Nodes`: Number of leaf nodes (clusters)
- `Avg Cluster Size`: Average size of clusters
- `Quantization Error`: Average distance from samples to BMUs (if `compute_qe=True`)

**Requirements:** `pip install ghsom-toolkits[analysis]`

**Example:**

```python
from ghsom import GHSOM
from ghsom_toolkits.analysis import compare_models
import numpy as np

# Train multiple models with different parameters
data = np.random.rand(200, 10)

ghsom_loose = GHSOM(input_dataset=data, t1=0.7, t2=0.1)
model_loose = ghsom_loose.train(epochs_number=50)

ghsom_medium = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model_medium = ghsom_medium.train(epochs_number=50)

ghsom_tight = GHSOM(input_dataset=data, t1=0.3, t2=0.02)
model_tight = ghsom_tight.train(epochs_number=50)

# Compare models
comparison = compare_models(
    models=[model_loose, model_medium, model_tight],
    data=data,
    model_names=['Loose (0.7/0.1)', 'Medium (0.5/0.05)', 'Tight (0.3/0.02)']
)

print(comparison)
```

**Output:**
```
                 Model  Total Nodes  Max Depth  Leaf Nodes  Avg Cluster Size  Quantization Error
0   Loose (0.7/0.1)            15          2           8              25.0              0.245
1  Medium (0.5/0.05)           28          3          12              16.7              0.187
2   Tight (0.3/0.02)           45          4          18              11.1              0.142
```

---

### plot_comparison

```python
plot_comparison(
    comparison_df,
    plot_type='bar',
    figsize=(12, 6),
    save_path=None
)
```

Visualize model comparison results as bar chart, radar chart, or heatmap.

**Parameters:**

- `comparison_df` (pandas.DataFrame): Output from `compare_models()`
- `plot_type` (str, optional): Type of plot - 'bar', 'radar', or 'heatmap' (default: 'bar')
- `figsize` (tuple, optional): Figure size in inches (default: (12, 6))
- `save_path` (str, optional): Path to save the figure (default: None)

**Returns:** `matplotlib.figure.Figure`

**Example:**

```python
from ghsom_toolkits.analysis import compare_models, plot_comparison

# Compare models
comparison = compare_models(models, data, model_names)

# Bar chart comparison
fig = plot_comparison(comparison, plot_type='bar')
plt.show()

# Radar chart for multi-metric view
fig = plot_comparison(comparison, plot_type='radar', save_path='comparison_radar.png')

# Heatmap for many models
fig = plot_comparison(comparison, plot_type='heatmap', save_path='comparison_heatmap.png')
```

---

## Report Generation

### generate_report

```python
generate_report(
    node,
    data,
    output_path='ghsom_report.html',
    model_name='GHSOM Model',
    include_visualizations=True
)
```

Generate a comprehensive HTML report with model statistics, visualizations, and analysis.

**Parameters:**

- `node` (object): Root GHSOM node
- `data` (numpy.ndarray): Input data, shape (n_samples, n_features)
- `output_path` (str, optional): Path for output HTML file (default: 'ghsom_report.html')
- `model_name` (str, optional): Name for the model (default: 'GHSOM Model')
- `include_visualizations` (bool, optional): Include embedded plots (default: True)

**Requirements:** `pip install ghsom-toolkits[interactive]`

**Report Sections:**

1. **Model Overview**
   - Total nodes, depth, clusters
   - Hyperparameters (if available)
   - Training statistics

2. **Hierarchy Visualization**
   - Interactive hierarchy tree
   - Cluster size distribution

3. **Quality Metrics** (if sklearn available)
   - Quantization error
   - Silhouette score
   - Davies-Bouldin index

4. **Cluster Analysis**
   - Cluster size statistics
   - Distribution plots
   - Quality metrics per cluster

5. **Weight Analysis**
   - Weight vector visualizations
   - U-Matrix
   - Activation patterns

**Example:**

```python
from ghsom import GHSOM
from ghsom_toolkits.analysis import generate_report
import numpy as np

# Train model
data = np.random.rand(500, 20)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=100)

# Generate comprehensive report
generate_report(
    node=model,
    data=data,
    output_path='my_ghsom_analysis.html',
    model_name='Experiment #42',
    include_visualizations=True
)

print("Report generated: my_ghsom_analysis.html")
# Open in browser to view interactive report
```

---

## Analysis Workflows

### Hyperparameter Tuning

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.analysis import compare_models, plot_comparison

# Grid search over hyperparameters
t1_values = [0.3, 0.5, 0.7]
t2_values = [0.02, 0.05, 0.1]

models = []
names = []

for t1 in t1_values:
    for t2 in t2_values:
        ghsom = GHSOM(input_dataset=data, t1=t1, t2=t2)
        model = ghsom.train(epochs_number=50)
        models.append(model)
        names.append(f't1={t1}, t2={t2}')

# Compare all models
comparison = compare_models(models, data, model_names=names)

# Find best model based on criteria
best_by_qe = comparison.loc[comparison['Quantization Error'].idxmin()]
print(f"Best model by QE: {best_by_qe['Model']}")

best_by_clusters = comparison.loc[
    (comparison['Leaf Nodes'] - 10).abs().idxmin()  # Target ~10 clusters
]
print(f"Best model for ~10 clusters: {best_by_clusters['Model']}")

# Visualize comparison
plot_comparison(comparison, plot_type='bar', save_path='hyperparam_comparison.png')
```

### Complete Analysis Pipeline

```python
from ghsom import GHSOM
from ghsom_toolkits.analysis import compare_models, plot_comparison, generate_report
from ghsom_toolkits.plotting import (
    plot_cluster_distribution,
    plot_cluster_quality,
    plot_weight_heatmap
)
import numpy as np

# 1. Train multiple models
data = np.random.rand(500, 20)

models = []
for t1 in [0.3, 0.5, 0.7]:
    ghsom = GHSOM(input_dataset=data, t1=t1, t2=0.05)
    models.append(ghsom.train(epochs_number=100))

# 2. Compare models
comparison = compare_models(
    models,
    data,
    model_names=['Tight', 'Medium', 'Loose']
)
print(comparison)

# 3. Visualize comparison
plot_comparison(comparison, plot_type='radar', save_path='comparison.png')

# 4. Select best model (e.g., by QE)
best_idx = comparison['Quantization Error'].idxmin()
best_model = models[best_idx]
best_name = comparison.loc[best_idx, 'Model']

print(f"\nBest model: {best_name}")

# 5. Generate detailed report for best model
generate_report(
    node=best_model,
    data=data,
    output_path=f'report_{best_name}.html',
    model_name=f'Best Model - {best_name}'
)

# 6. Additional analysis
plot_cluster_distribution(best_model, save_path='cluster_dist.png')

# Get cluster assignments (you need to implement this based on your needs)
# cluster_labels = assign_clusters(best_model, data)
# plot_cluster_quality(data, cluster_labels, save_path='cluster_quality.png')
```

### Model Versioning and Tracking

```python
import json
from datetime import datetime
from ghsom_toolkits.analysis import compare_models
from ghsom.io import save_model  # Hypothetical save function

# Train and track model
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=100)

# Get model statistics
comparison = compare_models([model], data, model_names=['current'])
stats = comparison.iloc[0].to_dict()

# Save metadata
metadata = {
    'timestamp': datetime.now().isoformat(),
    'hyperparameters': {'t1': 0.5, 't2': 0.05, 'epochs': 100},
    'statistics': stats,
    'data_shape': data.shape
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Save model
save_model(model, 'model_checkpoint.pkl')

# Generate report
generate_report(
    model,
    data,
    output_path=f'report_{datetime.now():%Y%m%d_%H%M%S}.html'
)
```

---

## Metrics Interpretation

### Quantization Error (QE)
- **Lower is better**
- Measures average distance from samples to their BMUs
- Range: 0 to ∞
- Typical good values: < 0.2 (normalized data)

### Number of Clusters (Leaf Nodes)
- **Domain dependent**
- Too few → underfitting
- Too many → overfitting
- Balance with interpretability

### Maximum Depth
- **Trade-off metric**
- Deeper → more hierarchical structure
- Shallower → simpler, flatter clustering
- Typical range: 2-5 levels

### Silhouette Score
- **Range: -1 to 1, higher is better**
- > 0.5: Good clustering
- 0.2-0.5: Moderate clustering
- < 0.2: Poor clustering

### Davies-Bouldin Index
- **Lower is better**
- Measures cluster separation
- Good values: < 1.0
- Higher values indicate overlapping clusters
