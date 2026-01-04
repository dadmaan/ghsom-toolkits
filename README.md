# GHSOM Toolkits

[![PyPI version](https://img.shields.io/pypi/v/ghsom-toolkits.svg)](https://pypi.org/project/ghsom-toolkits/)
[![Python versions](https://img.shields.io/pypi/pyversions/ghsom-toolkits.svg)](https://pypi.org/project/ghsom-toolkits/)
[![License](https://img.shields.io/github/license/dadmaan/ghsom-toolkits.svg)](https://github.com/dadmaan/ghsom-toolkits/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://dadmaan.github.io/ghsom-toolkits/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/dadmaan/ghsom-toolkits)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Visualization and analysis tools for **Growing Hierarchical Self-Organizing Maps (GHSOM)**.

A companion package to [ghsom-py](https://github.com/dadmaan/ghsom-py) providing rich visualization capabilities for exploring and analyzing GHSOM models.

---

**📚 [Documentation](https://dadmaan.github.io/ghsom-toolkits/)** |
**🚀 [Quick Start](https://dadmaan.github.io/ghsom-toolkits/quickstart/)** |
**🎨 [Gallery](https://dadmaan.github.io/ghsom-toolkits/gallery/)** |
**💬 [Discussions](https://github.com/dadmaan/ghsom-toolkits/discussions)**

---

## Features

### Visualization
- 🌳 **Hierarchical Visualization**: Tree-based visualization with Graphviz and interactive treemaps
- 🔥 **Heatmaps**: Weight vectors, activation maps, and U-Matrix visualizations
- 📊 **Cluster Analysis**: Distribution, quality metrics (Silhouette, Davies-Bouldin), and timeline plots
- 🎯 **Node Highlighting**: Focus on specific nodes with ancestor/descendant context

### Interactive Tools
- 🖥️ **Dash Dashboard**: Interactive web-based model exploration
- 🔍 **Explorer Utilities**: Sample tracing, neuron exploration, and subtree extraction
- 📈 **Real-time Metrics**: Monitor model performance during exploration

### Analysis
- ⚖️ **Model Comparison**: Compare multiple models with different hyperparameters
- 📋 **HTML Reports**: Generate comprehensive model analysis reports
- 📊 **Multiple Plot Types**: Bar charts, radar charts, heatmaps for comparisons

### Quality
- 🎨 **Publication Ready**: Export high-quality figures (300 DPI) for papers
- 🔌 **Simple API**: Easy integration with ghsom-py models
- ✅ **Well Tested**: >70% test coverage with comprehensive test suite

## Installation

```bash
pip install ghsom-toolkits
```

### With Optional Dependencies

```bash
# For pandas DataFrame support (cluster plotting, comparisons)
pip install ghsom-toolkits[pandas]

# For interactive dashboards and reports
pip install ghsom-toolkits[interactive]

# For advanced analysis (quality metrics, comparisons)
pip install ghsom-toolkits[analysis]

# Install everything
pip install ghsom-toolkits[all]
```

### Requirements

- **ghsom-py** >= 0.1.0 (core GHSOM package)
- **matplotlib** >= 3.5.0
- **pydot** >= 1.4.2
- **Graphviz** (system dependency for pydot)

**Installing Graphviz:**
```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
# Download from https://graphviz.org/download/
```

## Screenshots

<table>
<tr>
<td width="50%">
<img src="example_outputs/ghsom_hierarchy_example.png" alt="Hierarchy Visualization"/>
<p align="center"><em>Hierarchy Tree Visualization</em></p>
</td>
<td width="50%">
<img src="example_outputs/weight_heatmap_all.png" alt="Weight Heatmap"/>
<p align="center"><em>Weight Vectors Heatmap</em></p>
</td>
</tr>
<tr>
<td width="50%">
<img src="example_outputs/umatrix.png" alt="U-Matrix"/>
<p align="center"><em>U-Matrix Visualization</em></p>
</td>
<td width="50%">
<img src="example_outputs/cluster_distribution.png" alt="Cluster Distribution"/>
<p align="center"><em>Cluster Size Distribution</em></p>
</td>
</tr>
</table>

## Quick Start

### 1. Install

```bash
pip install ghsom-py ghsom-toolkits
```

### 2. Train and Visualize

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits import visualize_ghsom_hierarchy

# Train GHSOM
data = np.random.rand(200, 10)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
result = ghsom.train(epochs_number=50)

# Adapt and visualize
model = adapt_model(result, input_dataset_size=len(data))
lookup = build_lookup_table(model)
visualize_ghsom_hierarchy(model, lookup, "hierarchy.png")
```

**That's it!** See the [Quick Start Guide](https://dadmaan.github.io/ghsom-toolkits/quickstart/) for more examples.

## More Examples

### Heatmaps

```python
from ghsom_toolkits.plotting import plot_weight_heatmap, plot_umatrix

plot_weight_heatmap(model, save_path="weights.png")
plot_umatrix(model, save_path="umatrix.png")
```

### Interactive Dashboard

```python
from ghsom_toolkits.interactive import launch_dashboard

launch_dashboard(model, data=data, port=8050)
# Open http://localhost:8050
```

### Model Comparison

```python
from ghsom_toolkits.analysis import compare_models

comparison = compare_models([model1, model2, model3], data,
                           model_names=['Loose', 'Medium', 'Tight'])
print(comparison)
```

### Generate Report

```python
from ghsom_toolkits.analysis import generate_report

generate_report(model, data, "ghsom_report.html", "My Model")
```

## Documentation

Full documentation is available at [https://dadmaan.github.io/ghsom-toolkits/](https://dadmaan.github.io/ghsom-toolkits/)

## Requirements

- Python >= 3.8
- ghsom-py >= 0.1.0
- NumPy >= 1.20.0
- matplotlib >= 3.5.0
- pydot >= 1.4.2
- Graphviz (system package)

## Development

```bash
# Clone repository
git clone https://github.com/dadmaan/ghsom-toolkits.git
cd ghsom-toolkits

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black ghsom_toolkits/ tests/

# Lint
ruff check ghsom_toolkits/ tests/

# Type check
mypy ghsom_toolkits/
```

## Phase 6 Features (NEW!)

### Heatmap Visualizations
```python
from ghsom_toolkits.plotting.heatmaps import (
    plot_weight_heatmap, plot_activation_map, plot_umatrix
)

# Weight vectors heatmap
plot_weight_heatmap(model, save_path="weights.png")

# Activation patterns for specific samples
plot_activation_map(model, data, sample_indices=[0, 1, 2])

# U-Matrix showing neuron distances
plot_umatrix(model, save_path="umatrix.png")
```

### Interactive Dashboard
```python
from ghsom_toolkits.interactive import launch_dashboard

# Launch web-based interactive explorer
launch_dashboard(model, data=data, port=8050)
# Open browser to http://localhost:8050
```

### Model Comparison
```python
from ghsom_toolkits.analysis import compare_models, plot_comparison

# Compare multiple models
models = [model1, model2, model3]
comparison = compare_models(models, data, model_names=['Loose', 'Medium', 'Tight'])
print(comparison)

# Visualize comparison
plot_comparison(comparison, plot_type='radar')
```

### HTML Reports
```python
from ghsom_toolkits.analysis import generate_report

# Generate comprehensive HTML report
generate_report(model, data, output_path="ghsom_report.html")
```

### Advanced Cluster Analysis
```python
from ghsom_toolkits.plotting.clusters import (
    plot_cluster_distribution, plot_cluster_quality, plot_growth_timeline
)

# Cluster size distribution
plot_cluster_distribution(model)

# Quality metrics (Silhouette, Davies-Bouldin)
plot_cluster_quality(data, cluster_labels)

# Training evolution
plot_growth_timeline(training_history)
```

## Roadmap

- ✅ Hierarchical tree visualization
- ✅ Cluster distribution plots
- ✅ Node position highlighting
- ✅ Interactive dashboards with Plotly/Dash
- ✅ Heatmap visualizations (weights, activation, U-Matrix)
- ✅ Model comparison and analysis
- ✅ HTML report generation
- ⬜ 3D hierarchy visualization
- ⬜ Network graph layouts
- ⬜ Jupyter notebook integration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GHSOM Toolkits in your research, please cite:

```bibtex
@software{ghsom_toolkits,
  title={GHSOM Toolkits: Visualization Tools for Growing Hierarchical Self-Organizing Maps},
  author={GHSOM-Py Contributors},
  year={2024},
  url={https://github.com/dadmaan/ghsom-toolkits}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Projects

- [ghsom-py](https://github.com/dadmaan/ghsom-py) - Core GHSOM implementation
- [aria](https://github.com/dadmaan/aria) - Multi-agent RL framework for user-centric music generation

## Acknowledgments

This package was created as part of the GHSOM-Py project, extracted from a music generation reinforcement learning system.
