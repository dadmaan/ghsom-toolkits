# Tutorial: Model Comparison

Learn how to compare multiple GHSOM models trained with different hyperparameters to find the optimal configuration.

## Prerequisites

```bash
pip install ghsom-toolkits[analysis]
```

## Overview

When working with GHSOM, you often need to experiment with different hyperparameters:
- **t1**: Growth threshold (controls when to expand the hierarchy)
- **t2**: Expansion threshold (controls when to grow individual maps)
- **learning_rate**: Training speed
- **gaussian_sigma**: Neighborhood function width

This tutorial shows how to systematically compare models.

## Step 1: Train Multiple Models

First, train models with different hyperparameter combinations:

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model

# Generate sample data
np.random.seed(42)
data = np.random.rand(300, 15)

# Define hyperparameter configurations
configs = [
    {'t1': 0.7, 't2': 0.1, 'name': 'Loose'},
    {'t1': 0.5, 't2': 0.05, 'name': 'Medium'},
    {'t1': 0.3, 't2': 0.02, 'name': 'Tight'},
]

# Train all models
models = []
names = []

print("Training models...")
for config in configs:
    print(f"  Training {config['name']}...")
    ghsom = GHSOM(
        input_dataset=data,
        t1=config['t1'],
        t2=config['t2']
    )
    result = ghsom.train(epochs_number=50)

    # Adapt for ghsom-toolkits
    adapted = adapt_model(result, input_dataset_size=len(data))
    models.append(adapted)
    names.append(config['name'])

print("✓ All models trained")
```

## Step 2: Compare Models

Use the `compare_models()` function to generate a comparison table:

```python
from ghsom_toolkits.analysis import compare_models

# Compare all models
comparison = compare_models(
    models=models,
    data=data,
    model_names=names,
    compute_qe=True  # Compute quantization error
)

print("\n=== Model Comparison ===")
print(comparison)
```

**Example Output:**
```
      Model  Total Nodes  Max Depth  Leaf Nodes  Avg Cluster Size  Quantization Error
0    Loose           12          2           6              50.0              0.245
1   Medium           24          3          10              30.0              0.187
2    Tight           38          4          15              20.0              0.142
```

## Step 3: Visualize Comparison

Create visualizations to better understand the differences:

```python
from ghsom_toolkits.analysis import plot_comparison
import matplotlib.pyplot as plt

# Bar chart comparison
fig = plot_comparison(
    comparison,
    plot_type='bar',
    figsize=(14, 6),
    save_path='comparison_bar.png'
)
plt.show()

# Radar chart for multi-metric view
fig = plot_comparison(
    comparison,
    plot_type='radar',
    figsize=(8, 8),
    save_path='comparison_radar.png'
)
plt.show()

# Heatmap (useful for many models)
fig = plot_comparison(
    comparison,
    plot_type='heatmap',
    figsize=(10, 6),
    save_path='comparison_heatmap.png'
)
plt.show()
```

## Step 4: Analyze Results

Interpret the comparison to select the best model:

```python
# Find model with lowest quantization error
best_qe_idx = comparison['Quantization Error'].idxmin()
best_qe_model = comparison.loc[best_qe_idx]
print(f"\nBest by Quantization Error:")
print(f"  Model: {best_qe_model['Model']}")
print(f"  QE: {best_qe_model['Quantization Error']:.3f}")

# Find model with target number of clusters
target_clusters = 12
comparison['Cluster Diff'] = (comparison['Leaf Nodes'] - target_clusters).abs()
best_cluster_idx = comparison['Cluster Diff'].idxmin()
best_cluster_model = comparison.loc[best_cluster_idx]
print(f"\nClosest to {target_clusters} clusters:")
print(f"  Model: {best_cluster_model['Model']}")
print(f"  Clusters: {int(best_cluster_model['Leaf Nodes'])}")

# Trade-off: good QE with reasonable depth
comparison['Score'] = (
    -comparison['Quantization Error'] * 2 +  # Lower QE is better
    -comparison['Max Depth'] * 0.5           # Prefer shallower
)
best_overall_idx = comparison['Score'].idxmax()
best_overall = comparison.loc[best_overall_idx]
print(f"\nBest overall (balanced):")
print(f"  Model: {best_overall['Model']}")
print(f"  QE: {best_overall['Quantization Error']:.3f}")
print(f"  Depth: {int(best_overall['Max Depth'])}")
```

## Step 5: Grid Search

Perform systematic grid search over hyperparameters:

```python
import itertools
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model
from ghsom_toolkits.analysis import compare_models

# Define parameter grid
t1_values = [0.3, 0.5, 0.7]
t2_values = [0.02, 0.05, 0.1]

# Generate all combinations
param_combinations = list(itertools.product(t1_values, t2_values))

print(f"Testing {len(param_combinations)} configurations...")

models = []
names = []

for t1, t2 in param_combinations:
    print(f"  Training t1={t1}, t2={t2}...")

    ghsom = GHSOM(input_dataset=data, t1=t1, t2=t2)
    result = ghsom.train(epochs_number=50)

    adapted = adapt_model(result, input_dataset_size=len(data))
    models.append(adapted)
    names.append(f't1={t1}, t2={t2}')

# Compare all configurations
comparison = compare_models(models, data, names, compute_qe=True)

# Sort by quantization error
comparison_sorted = comparison.sort_values('Quantization Error')
print("\n=== Top 5 Configurations ===")
print(comparison_sorted.head())

# Save results
comparison_sorted.to_csv('grid_search_results.csv', index=False)
print("\n✓ Results saved to grid_search_results.csv")
```

## Step 6: Visualize Individual Models

After identifying the best models, visualize them:

```python
from ghsom_toolkits import visualize_ghsom_hierarchy
from ghsom_toolkits.adapters import build_lookup_table
from ghsom_toolkits.plotting import plot_cluster_distribution

# Get best model
best_idx = comparison['Quantization Error'].idxmin()
best_model = models[best_idx]
best_name = names[best_idx]

print(f"\nVisualizing best model: {best_name}")

# Build lookup table
lookup = build_lookup_table(best_model)

# Create visualizations
visualize_ghsom_hierarchy(
    best_model,
    lookup,
    f"best_model_{best_name.replace(', ', '_')}_hierarchy.png"
)

plot_cluster_distribution(
    best_model,
    save_path=f"best_model_{best_name.replace(', ', '_')}_clusters.png"
)

print("✓ Visualizations saved")
```

## Step 7: Generate Reports for Top Models

Create detailed reports for the top-performing models:

```python
from ghsom_toolkits.analysis import generate_report

# Get top 3 models
top_3_indices = comparison.nsmallest(3, 'Quantization Error').index

for idx in top_3_indices:
    model = models[idx]
    name = names[idx]

    print(f"Generating report for {name}...")

    generate_report(
        node=model,
        data=data,
        output_path=f"report_{name.replace(', ', '_')}.html",
        model_name=name
    )

print("✓ Reports generated")
```

## Advanced: Cross-Validation

Compare models using cross-validation:

```python
import numpy as np
from sklearn.model_selection import KFold
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model
from ghsom_toolkits.analysis import compare_models

def cross_validate_ghsom(data, t1, t2, n_folds=5):
    """Cross-validate a GHSOM configuration."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    qe_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        train_data = data[train_idx]
        val_data = data[val_idx]

        # Train on fold
        ghsom = GHSOM(input_dataset=train_data, t1=t1, t2=t2)
        result = ghsom.train(epochs_number=30)
        model = adapt_model(result, input_dataset_size=len(train_data))

        # Compute QE on validation set
        # (simplified - you'd implement proper QE calculation)
        from ghsom_toolkits.analysis.comparisons import _compute_quantization_error
        qe = _compute_quantization_error(model, val_data)
        qe_scores.append(qe)

    return {
        'mean_qe': np.mean(qe_scores),
        'std_qe': np.std(qe_scores),
        'scores': qe_scores
    }

# Test configurations with cross-validation
configs = [
    (0.5, 0.05),
    (0.6, 0.05),
    (0.5, 0.03),
]

results = []
for t1, t2 in configs:
    print(f"Cross-validating t1={t1}, t2={t2}...")
    cv_result = cross_validate_ghsom(data, t1, t2)
    results.append({
        't1': t1,
        't2': t2,
        'mean_qe': cv_result['mean_qe'],
        'std_qe': cv_result['std_qe']
    })

# Display results
import pandas as pd
cv_df = pd.DataFrame(results)
cv_df = cv_df.sort_values('mean_qe')
print("\n=== Cross-Validation Results ===")
print(cv_df)
```

## Complete Comparison Workflow

Here's a complete script combining all steps:

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits.analysis import compare_models, plot_comparison, generate_report
from ghsom_toolkits import visualize_ghsom_hierarchy
import matplotlib.pyplot as plt

# 1. Generate data
np.random.seed(42)
data = np.random.rand(400, 20)

# 2. Define configurations
configs = [
    {'t1': 0.7, 't2': 0.1, 'name': 'Loose (Many large clusters)'},
    {'t1': 0.5, 't2': 0.05, 'name': 'Medium (Balanced)'},
    {'t1': 0.3, 't2': 0.02, 'name': 'Tight (Many small clusters)'},
    {'t1': 0.5, 't2': 0.1, 'name': 'Wide maps, shallow'},
    {'t1': 0.3, 't2': 0.05, 'name': 'Narrow maps, deep'},
]

# 3. Train all models
print("Training models...")
models, names = [], []
for config in configs:
    print(f"  {config['name']}...")
    ghsom = GHSOM(input_dataset=data, t1=config['t1'], t2=config['t2'])
    result = ghsom.train(epochs_number=50)
    models.append(adapt_model(result, input_dataset_size=len(data)))
    names.append(config['name'])

# 4. Compare
print("\nComparing models...")
comparison = compare_models(models, data, names)
print(comparison)

# 5. Visualize comparison
print("\nGenerating comparison plots...")
plot_comparison(comparison, 'bar', save_path='comparison_bar.png')
plot_comparison(comparison, 'radar', save_path='comparison_radar.png')
plt.close('all')

# 6. Select best
best_idx = comparison['Quantization Error'].idxmin()
best_model = models[best_idx]
best_name = names[best_idx]
print(f"\n✓ Best model: {best_name}")

# 7. Visualize best model
print("Visualizing best model...")
lookup = build_lookup_table(best_model)
visualize_ghsom_hierarchy(best_model, lookup, 'best_hierarchy.png')

# 8. Generate report
print("Generating report...")
generate_report(best_model, data, 'best_model_report.html', best_name)

print("\n✓ Comparison complete!")
print(f"  - Best model: {best_name}")
print(f"  - QE: {comparison.loc[best_idx, 'Quantization Error']:.3f}")
print(f"  - Report: best_model_report.html")
```

## Tips for Effective Comparison

### 1. Choose Metrics Based on Goals

- **Quantization Error**: Data representation quality
- **Number of Clusters**: Interpretability
- **Depth**: Hierarchical structure complexity
- **Cluster Size**: Granularity

### 2. Consider Domain Requirements

```python
# Example: Target specific cluster count
target = 15
comparison['Cluster Match'] = (comparison['Leaf Nodes'] - target).abs()
best = comparison.loc[comparison['Cluster Match'].idxmin()]
```

### 3. Balance Multiple Objectives

```python
# Multi-objective scoring
comparison['Score'] = (
    -2 * comparison['Quantization Error'] +  # Quality (weight=2)
    -1 * comparison['Max Depth'] / 10 +       # Simplicity (weight=0.1)
    -1 * abs(comparison['Leaf Nodes'] - 12)   # Target 12 clusters
)
```

## Next Steps

- [Tutorial 4: Generating Reports](04_generating_reports.md) - Create comprehensive HTML reports
- [Tutorial 5: ghsom-py Integration](05_ghsom_py_integration.md) - Deep dive into ghsom-py usage
- [API Reference: Analysis Tools](../api/analysis.md) - Full API documentation
