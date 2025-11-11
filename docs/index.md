# GHSOM Toolkits

**Visualization and analysis tools for Growing Hierarchical Self-Organizing Maps (GHSOM)**

A companion package to [ghsom-py](https://github.com/dadmaan/ghsom-py) providing rich visualization capabilities for exploring and analyzing GHSOM models.

---

## Features

### 🌳 Hierarchical Visualization
- Tree-based visualization with Graphviz
- Interactive treemaps with Plotly
- Node highlighting and subtree extraction

### 🔥 Heatmaps
- Weight vector visualizations
- Activation maps showing BMUs
- U-Matrix for cluster boundaries

### 📊 Cluster Analysis
- Distribution plots (histogram + box plot)
- Quality metrics (Silhouette, Davies-Bouldin)
- Growth timeline visualization

### 🖥️ Interactive Tools
- Web-based Dash dashboard
- Sample path tracing
- Real-time model exploration

### ⚖️ Model Comparison
- Compare multiple models
- Bar, radar, and heatmap visualizations
- Hyperparameter optimization support

### 📋 Reports
- Comprehensive HTML reports
- Embedded interactive visualizations
- Publication-ready output

---

## Quick Start

### Installation

```bash
# Basic installation
pip install ghsom-toolkits

# With interactive features
pip install ghsom-toolkits[interactive]

# With analysis tools
pip install ghsom-toolkits[analysis]

# Everything
pip install ghsom-toolkits[all]
```

### System Requirements

**Graphviz** is required for hierarchy visualizations:

```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
# Download from https://graphviz.org/download/
```

### Basic Usage

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits import visualize_ghsom_hierarchy

# 1. Train GHSOM with ghsom-py
data = np.random.rand(200, 10)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=50)

# 2. Adapt for ghsom-toolkits
adapted_model = adapt_model(model, input_dataset_size=len(data))
lookup = build_lookup_table(adapted_model)

# 3. Visualize
visualize_ghsom_hierarchy(adapted_model, lookup, "hierarchy.png")
```

---

## Key Capabilities

### Hierarchy Visualization

```python
from ghsom_toolkits import visualize_ghsom_hierarchy

visualize_ghsom_hierarchy(model, lookup, "output.png")
```

### Heatmap Analysis

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
from ghsom_toolkits.analysis import compare_models, plot_comparison

comparison = compare_models([model1, model2, model3], data,
                           model_names=['Loose', 'Medium', 'Tight'])
plot_comparison(comparison, plot_type='radar')
```

### HTML Reports

```python
from ghsom_toolkits.analysis import generate_report

generate_report(model, data, output_path="report.html",
               model_name="My GHSOM Model")
```

---

## Documentation

### Getting Started
- **[Installation Guide](installation.md)** - Setup and requirements
- **[Quick Start](quickstart.md)** - Your first visualization in 5 minutes

### Tutorials
1. **[Basic Visualization](tutorials/01_basic_visualization.md)** - Hierarchy plots and node highlighting
2. **[Interactive Dashboard](tutorials/02_interactive_dashboard.md)** - Web-based exploration
3. **[Model Comparison](tutorials/03_model_comparison.md)** - Hyperparameter tuning
4. **[Generating Reports](tutorials/04_generating_reports.md)** - Comprehensive HTML reports
5. **[ghsom-py Integration](tutorials/05_ghsom_py_integration.md)** - Advanced integration patterns

### API Reference
- **[Plotting Functions](api/plotting.md)** - Hierarchy, heatmaps, clusters
- **[Interactive Tools](api/interactive.md)** - Dashboard and exploration
- **[Analysis Tools](api/analysis.md)** - Comparison and reports
- **[Adapters](api/adapters.md)** - ghsom-py integration

### Resources
- **[Gallery](gallery.md)** - All visualization types with examples
- **[Contributing](contributing.md)** - How to contribute

---

## Why GHSOM Toolkits?

### 🎯 Designed for ghsom-py

Seamless integration with the ghsom-py library through smart adapters:

```python
# Train with ghsom-py
result = ghsom.train(epochs_number=50)

# Automatically adapt and visualize
model = adapt_model(result)
visualize_ghsom_hierarchy(model, build_lookup_table(model), "output.png")
```

### 🚀 Minimal Overhead

- **Zero-copy adaptation** - No data duplication
- **Lazy loading** - Build structures only when needed
- **<5% overhead** - Negligible performance impact

### 📈 Publication Ready

- **High-resolution output** - 300+ DPI for papers
- **Multiple formats** - PNG, PDF, SVG
- **Customizable styling** - Full control over appearance

### 🔧 Extensible

- **Modular design** - Use only what you need
- **Custom visualizations** - Easy to extend
- **Multiple backends** - Matplotlib, Plotly, Graphviz

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ghsom-toolkits                          │
│                                                             │
│  ┌────────────┐      ┌────────────┐      ┌─────────────┐  │
│  │   Core     │      │  Adapters  │      │ Plotting/   │  │
│  │  Module    │◄─────│   Module   │◄─────│ Analysis    │  │
│  │            │      │            │      │             │  │
│  │ GHSOMNode  │      │  Adapter   │      │ visualize_  │  │
│  │  parsing   │      │  adapt_    │      │ hierarchy() │  │
│  │            │      │   model()  │      │             │  │
│  └────────────┘      └────────────┘      └─────────────┘  │
│       ▲                    ▲                                │
└───────┼────────────────────┼────────────────────────────────┘
        │                    │
        │                    │ wraps
┌───────┴─────────┐   ┌─────┴──────┐
│  Main Project   │   │  ghsom-py  │
│                 │   │            │
│ Your Code       │   │ GSOM       │
│                 │   │ Neuron     │
└─────────────────┘   └────────────┘
```

---

## Package Structure

- **`ghsom_toolkits.plotting`** - Visualization functions (hierarchy, heatmaps, clusters)
- **`ghsom_toolkits.interactive`** - Dashboard and exploration tools
- **`ghsom_toolkits.analysis`** - Model comparison and reporting
- **`ghsom_toolkits.adapters`** - Compatibility with ghsom-py
- **`ghsom_toolkits.core`** - Internal data structures
- **`ghsom_toolkits.export`** - Export utilities

---

## Requirements

- **Python** >= 3.8
- **ghsom-py** >= 0.1.0 (core GHSOM implementation)
- **NumPy** >= 1.20.0
- **Matplotlib** >= 3.5.0
- **pydot** >= 1.4.2
- **Graphviz** (system package)

### Optional Dependencies

- **pandas** >= 1.3.0 (for cluster plotting)
- **plotly** >= 5.0.0 (for interactive visualizations)
- **dash** >= 2.0.0 (for dashboard)
- **scikit-learn** >= 1.0.0 (for quality metrics)
- **scipy** >= 1.7.0 (for advanced analysis)

---

## Examples

Browse the [examples directory](https://github.com/dadmaan/ghsom-toolkits/tree/main/examples) for runnable scripts:

- `basic_visualization.py` - Hierarchy and node highlighting
- `heatmap_visualization.py` - Weight heatmaps and U-Matrix
- `cluster_analysis.py` - Cluster distribution and quality
- `interactive_dashboard.py` - Launch web dashboard
- `model_comparison.py` - Compare multiple models
- `generate_report.py` - Create HTML reports

---

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

And the original GHSOM paper:

```bibtex
@article{rauber2002growing,
  title={The growing hierarchical self-organizing map: exploratory analysis of high-dimensional data},
  author={Rauber, Andreas and Merkl, Dieter and Dittenbach, Michael},
  journal={IEEE Transactions on Neural Networks},
  volume={13},
  number={6},
  pages={1331--1341},
  year={2002},
  publisher={IEEE}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/dadmaan/ghsom-toolkits/blob/main/LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please see our [Contributing Guide](contributing.md) for details.

---

## Related Projects

- **[ghsom-py](https://github.com/dadmaan/ghsom-py)** - Core GHSOM implementation
- **[Original GHSOM](https://www.ifs.tuwien.ac.at/~andi/ghsom/)** - Original Java implementation

---

## Support

- **Documentation**: [https://dadmaan.github.io/ghsom-toolkits/](https://dadmaan.github.io/ghsom-toolkits/)
- **Issues**: [GitHub Issues](https://github.com/dadmaan/ghsom-toolkits/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dadmaan/ghsom-toolkits/discussions)

---

**Get started now**: [Installation Guide](installation.md) → [Quick Start](quickstart.md) → [Tutorials](tutorials/01_basic_visualization.md)
