# Tutorial: Generating Reports

Learn how to create comprehensive HTML reports for GHSOM models that you can share with colleagues or include in presentations.

## Prerequisites

```bash
pip install ghsom-toolkits[interactive]
```

The report generator uses Jinja2 templates and Plotly for interactive visualizations.

## Step 1: Basic Report Generation

The simplest way to generate a report:

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model
from ghsom_toolkits.analysis import generate_report

# Train model
np.random.seed(42)
data = np.random.rand(500, 20)

ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=100)

# Adapt model
adapted_model = adapt_model(model, input_dataset_size=len(data))

# Generate report
generate_report(
    node=adapted_model,
    data=data,
    output_path='ghsom_report.html',
    model_name='My GHSOM Model'
)

print("✓ Report generated: ghsom_report.html")
print("  Open it in your browser!")
```

## Step 2: Understanding Report Sections

The generated report includes:

### 1. Model Overview
- Total nodes in hierarchy
- Maximum depth
- Number of clusters (leaf nodes)
- Average cluster size
- Root map dimensions

### 2. Hierarchy Visualization
- Interactive tree view (if Plotly available)
- Zoomable and pannable
- Click nodes for details

### 3. Cluster Analysis
- Cluster size distribution (histogram + box plot)
- Statistics (mean, median, min, max)
- Identify imbalanced clusters

### 4. Quality Metrics (if scikit-learn available)
- Quantization Error
- Silhouette Score
- Davies-Bouldin Index
- Per-cluster quality scores

### 5. Weight Visualizations
- Weight heatmaps
- U-Matrix
- Activation patterns (for sample data)

## Step 3: Customizing Reports

### Custom Model Name and Metadata

```python
generate_report(
    node=adapted_model,
    data=data,
    output_path='experiment_42_report.html',
    model_name='Experiment 42: t1=0.5, t2=0.05, epochs=100',
    include_visualizations=True
)
```

### Without Embedded Visualizations

For faster generation and smaller file sizes:

```python
generate_report(
    node=adapted_model,
    data=data,
    output_path='lightweight_report.html',
    include_visualizations=False  # Skip embedded plots
)
```

## Step 4: Batch Report Generation

Generate reports for multiple models:

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model
from ghsom_toolkits.analysis import generate_report
from datetime import datetime

# Common data
data = np.random.rand(500, 20)

# Multiple configurations
configs = [
    {'t1': 0.7, 't2': 0.1, 'name': 'Loose'},
    {'t1': 0.5, 't2': 0.05, 'name': 'Medium'},
    {'t1': 0.3, 't2': 0.02, 'name': 'Tight'},
]

# Generate reports for all
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

for config in configs:
    print(f"Processing {config['name']}...")

    # Train
    ghsom = GHSOM(input_dataset=data, t1=config['t1'], t2=config['t2'])
    result = ghsom.train(epochs_number=50)
    model = adapt_model(result, input_dataset_size=len(data))

    # Generate report
    report_name = f"report_{config['name']}_{timestamp}.html"
    generate_report(
        node=model,
        data=data,
        output_path=report_name,
        model_name=f"{config['name']} (t1={config['t1']}, t2={config['t2']})"
    )
    print(f"  ✓ {report_name}")

print("\n✓ All reports generated")
```

## Step 5: Adding Custom Analysis

Combine reports with additional analysis:

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits.analysis import generate_report, compare_models
from ghsom_toolkits.plotting import plot_cluster_distribution
import matplotlib.pyplot as plt

# Train model
data = np.random.rand(500, 20)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
result = ghsom.train(epochs_number=100)
model = adapt_model(result, input_dataset_size=len(data))

# Generate additional visualizations
print("Creating additional visualizations...")
plot_cluster_distribution(model, save_path='cluster_analysis.png')
plt.close()

# Generate main report
print("Generating main report...")
generate_report(
    node=model,
    data=data,
    output_path='comprehensive_report.html',
    model_name='Main Model'
)

print("✓ Reports complete")
print("  Main report: comprehensive_report.html")
print("  Additional: cluster_analysis.png")
```

## Step 6: Comparison Report

Create a comparison page for multiple models:

```python
from ghsom_toolkits.analysis import compare_models, plot_comparison
import pandas as pd

# Train multiple models (from previous example)
models = []
names = []
for config in configs:
    ghsom = GHSOM(input_dataset=data, t1=config['t1'], t2=config['t2'])
    result = ghsom.train(epochs_number=50)
    models.append(adapt_model(result, input_dataset_size=len(data)))
    names.append(config['name'])

# Generate comparison
comparison = compare_models(models, data, names)

# Save comparison as HTML table
html_table = comparison.to_html(index=False, classes='table table-striped')

# Create custom comparison page
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
    <style>
        body {{ padding: 20px; }}
        h1 {{ margin-bottom: 30px; }}
        .table {{ margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>GHSOM Model Comparison</h1>
        <p>Comparison of {len(models)} models trained with different hyperparameters.</p>
        {html_table}
        <h2 style="margin-top: 40px;">Individual Reports</h2>
        <ul>
"""

for name in names:
    html_content += f'            <li><a href="report_{name}_{timestamp}.html">{name}</a></li>\n'

html_content += """
        </ul>
    </div>
</body>
</html>
"""

# Save comparison page
with open('comparison_index.html', 'w') as f:
    f.write(html_content)

print("✓ Comparison index created: comparison_index.html")
```

## Step 7: Automated Report Pipeline

Create a complete automated pipeline:

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits.analysis import generate_report, compare_models
from ghsom_toolkits import visualize_ghsom_hierarchy
from datetime import datetime
import json
import os

class GHSOMReportPipeline:
    """Automated report generation pipeline."""

    def __init__(self, output_dir='reports'):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)

    def run_experiment(self, data, config, name):
        """Run single experiment and generate report."""
        print(f"Running experiment: {name}...")

        # Train
        ghsom = GHSOM(input_dataset=data, **config)
        result = ghsom.train(epochs_number=config.get('epochs', 50))
        model = adapt_model(result, input_dataset_size=len(data))

        # Generate visualizations
        lookup = build_lookup_table(model)
        hierarchy_path = f"{self.output_dir}/{name}_hierarchy.png"
        visualize_ghsom_hierarchy(model, lookup, hierarchy_path)

        # Generate report
        report_path = f"{self.output_dir}/{name}_report_{self.timestamp}.html"
        generate_report(
            node=model,
            data=data,
            output_path=report_path,
            model_name=name
        )

        # Save metadata
        metadata = {
            'name': name,
            'timestamp': self.timestamp,
            'config': config,
            'data_shape': data.shape,
            'report_path': report_path
        }

        metadata_path = f"{self.output_dir}/{name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return model, metadata

    def run_comparison(self, data, experiments):
        """Run multiple experiments and generate comparison."""
        models = []
        names = []
        all_metadata = []

        for exp_config, exp_name in experiments:
            model, metadata = self.run_experiment(data, exp_config, exp_name)
            models.append(model)
            names.append(exp_name)
            all_metadata.append(metadata)

        # Generate comparison
        comparison = compare_models(models, data, names)

        # Save comparison
        comparison_path = f"{self.output_dir}/comparison_{self.timestamp}.csv"
        comparison.to_csv(comparison_path, index=False)

        print(f"\n✓ Pipeline complete")
        print(f"  Reports in: {self.output_dir}/")
        print(f"  Comparison: {comparison_path}")

        return comparison, all_metadata

# Usage
pipeline = GHSOMReportPipeline(output_dir='ghsom_reports')

data = np.random.rand(500, 20)

experiments = [
    ({'t1': 0.7, 't2': 0.1, 'epochs': 50}, 'Loose'),
    ({'t1': 0.5, 't2': 0.05, 'epochs': 50}, 'Medium'),
    ({'t1': 0.3, 't2': 0.02, 'epochs': 50}, 'Tight'),
]

comparison, metadata = pipeline.run_comparison(data, experiments)
print(comparison)
```

## Report Best Practices

### 1. Descriptive Names

Use informative model names:

```python
generate_report(
    node=model,
    data=data,
    model_name=f"GHSOM: t1={t1}, t2={t2}, epochs={epochs}, data={len(data)}x{data.shape[1]}"
)
```

### 2. Include Context

Add information about the experiment:

```python
# Save experiment context separately
context = {
    'date': datetime.now().isoformat(),
    'data_source': 'synthetic_random',
    'preprocessing': 'normalized',
    'notes': 'Testing optimal t1/t2 ratio'
}

with open('experiment_context.json', 'w') as f:
    json.dump(context, f, indent=2)
```

### 3. Version Control

Track model versions:

```python
import hashlib

# Compute data hash for reproducibility
data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:8]

generate_report(
    node=model,
    data=data,
    output_path=f'report_v1.2_{data_hash}.html',
    model_name=f'Model v1.2 (data: {data_hash})'
)
```

## Complete Example: Research Report

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model
from ghsom_toolkits.analysis import generate_report, compare_models
from datetime import datetime
import json

# Experiment metadata
experiment = {
    'name': 'GHSOM Hyperparameter Study',
    'date': datetime.now().isoformat(),
    'researcher': 'Data Science Team',
    'objective': 'Find optimal t1/t2 for music clustering'
}

# Generate data (replace with real data)
np.random.seed(42)
data = np.random.rand(1000, 50)

# Train models
configs = [
    ({'t1': 0.5, 't2': 0.05}, 'baseline'),
    ({'t1': 0.6, 't2': 0.05}, 'higher_t1'),
    ({'t1': 0.5, 't2': 0.03}, 'lower_t2'),
]

models, names = [], []
for config, name in configs:
    print(f"Training {name}...")
    ghsom = GHSOM(input_dataset=data, **config)
    result = ghsom.train(epochs_number=100)
    model = adapt_model(result, input_dataset_size=len(data))
    models.append(model)
    names.append(name)

    # Generate individual report
    generate_report(
        node=model,
        data=data,
        output_path=f'report_{name}.html',
        model_name=f"{experiment['name']} - {name}"
    )

# Generate comparison
comparison = compare_models(models, data, names)
comparison.to_csv('results_comparison.csv')

# Save experiment summary
summary = {
    **experiment,
    'n_models': len(models),
    'data_shape': data.shape,
    'results': comparison.to_dict('records')
}

with open('experiment_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✓ Research report package complete")
print("  Individual reports: report_*.html")
print("  Comparison: results_comparison.csv")
print("  Summary: experiment_summary.json")
```

## Next Steps

- [Tutorial 5: ghsom-py Integration](05_ghsom_py_integration.md) - Advanced integration patterns
- [Gallery](../gallery.md) - See example visualizations
- [API Reference: Analysis](../api/analysis.md) - Full API documentation

## Troubleshooting

**"Jinja2 not found"**
- Install: `pip install ghsom-toolkits[interactive]`

**Report is very large (>10 MB)**
- Use `include_visualizations=False`
- Reduce data size for visualization samples

**Charts not showing in report**
- Ensure Plotly is installed
- Check browser JavaScript console for errors
