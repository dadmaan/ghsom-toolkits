# Installation Guide

## Quick Installation

```bash
pip install ghsom-toolkits
```

This installs the basic package with core visualization capabilities.

## System Requirements

- **Python** 3.8 or later
- **Operating System**: Linux, macOS, or Windows
- **Graphviz** (system package) - Required for hierarchy visualizations

## Installing Graphviz

Graphviz must be installed as a system package (not just the Python package):

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install graphviz
```

### macOS

```bash
brew install graphviz
```

### Windows

1. Download installer from [graphviz.org/download](https://graphviz.org/download/)
2. Run the installer
3. Add Graphviz to your PATH:
   - Add `C:\Program Files\Graphviz\bin` to your System PATH
   - Or use the installer option to automatically add to PATH

### Verify Installation

```bash
dot -V
# Should output: dot - graphviz version X.X.X
```

## Optional Dependencies

Install additional features as needed:

### Interactive Features

For web-based dashboards and interactive visualizations:

```bash
pip install ghsom-toolkits[interactive]
```

Includes:
- `plotly` - Interactive plots
- `dash` - Web dashboard
- `jinja2` - HTML report templates

### Analysis Tools

For model comparison and quality metrics:

```bash
pip install ghsom-toolkits[analysis]
```

Includes:
- `scikit-learn` - Quality metrics (Silhouette, Davies-Bouldin)
- `scipy` - Advanced analysis
- `pandas` - Data manipulation

### Pandas Support

For cluster plotting with DataFrames:

```bash
pip install ghsom-toolkits[pandas]
```

### Complete Installation

Install everything:

```bash
pip install ghsom-toolkits[all]
```

Equivalent to:
```bash
pip install ghsom-toolkits[pandas,interactive,analysis]
```

## Development Installation

For contributing to ghsom-toolkits:

```bash
# Clone repository
git clone https://github.com/dadmaan/ghsom-toolkits.git
cd ghsom-toolkits

# Install in editable mode with development dependencies
pip install -e .[dev]
```

Development dependencies include:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reports
- `black` - Code formatting
- `ruff` - Linting
- `mypy` - Type checking

## Installing ghsom-py

ghsom-toolkits is designed to work with [ghsom-py](https://github.com/dadmaan/ghsom-py):

```bash
pip install ghsom-py
```

Or install from source:

```bash
git clone https://github.com/dadmaan/ghsom-py.git
cd ghsom-py
pip install -e .
```

## Verifying Installation

Test your installation:

```python
import ghsom_toolkits
print(ghsom_toolkits.__version__)

# Check available modules
from ghsom_toolkits import visualize_ghsom_hierarchy
from ghsom_toolkits.plotting import plot_weight_heatmap
from ghsom_toolkits.adapters import adapt_model

print("✓ Installation successful!")
```

With interactive features:

```python
try:
    from ghsom_toolkits.interactive import launch_dashboard
    print("✓ Interactive features available")
except ImportError:
    print("✗ Interactive features not installed")
    print("  Install with: pip install ghsom-toolkits[interactive]")
```

With analysis tools:

```python
try:
    from ghsom_toolkits.analysis import compare_models
    from sklearn.metrics import silhouette_score
    print("✓ Analysis features available")
except ImportError:
    print("✗ Analysis features not installed")
    print("  Install with: pip install ghsom-toolkits[analysis]")
```

## Troubleshooting

### "GraphViz not found" or "GraphViz executables not found"

**Problem**: Graphviz system package not installed or not in PATH

**Solution**:
1. Install Graphviz system package (see above)
2. Verify with `dot -V`
3. On Windows, ensure Graphviz bin directory is in PATH
4. Restart your terminal/IDE after installation

### "ModuleNotFoundError: No module named 'plotly'"

**Problem**: Interactive dependencies not installed

**Solution**:
```bash
pip install ghsom-toolkits[interactive]
```

### "ModuleNotFoundError: No module named 'sklearn'"

**Problem**: Analysis dependencies not installed

**Solution**:
```bash
pip install ghsom-toolkits[analysis]
```

### Permission errors on Linux/macOS

**Problem**: Installing system package requires sudo

**Solution**:
```bash
# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install ghsom-toolkits[all]
```

### Import errors with ghsom-py

**Problem**: ghsom-py not installed

**Solution**:
```bash
pip install ghsom-py
```

## Virtual Environment (Recommended)

Always use a virtual environment:

### Using venv

```bash
# Create virtual environment
python -m venv ghsom_env

# Activate
source ghsom_env/bin/activate  # Linux/macOS
ghsom_env\Scripts\activate  # Windows

# Install packages
pip install ghsom-py ghsom-toolkits[all]
```

### Using conda

```bash
# Create environment
conda create -n ghsom python=3.10

# Activate
conda activate ghsom

# Install system dependencies
conda install graphviz

# Install Python packages
pip install ghsom-py ghsom-toolkits[all]
```

## Upgrading

Upgrade to the latest version:

```bash
pip install --upgrade ghsom-toolkits
```

Upgrade with all features:

```bash
pip install --upgrade ghsom-toolkits[all]
```

## Uninstalling

```bash
pip uninstall ghsom-toolkits
```

## Platform-Specific Notes

### Linux

- Most distributions include Python 3.8+
- Use system package manager for Graphviz
- Virtual environment recommended

### macOS

- Python 3 available via Homebrew: `brew install python`
- Install Graphviz via Homebrew: `brew install graphviz`
- Virtual environment recommended

### Windows

- Download Python from [python.org](https://www.python.org/)
- Install Graphviz from [graphviz.org](https://graphviz.org/)
- Add both Python and Graphviz to PATH
- Use PowerShell or Command Prompt
- Virtual environment strongly recommended

## Next Steps

After installation:

1. **[Quick Start Guide](quickstart.md)** - Create your first visualization
2. **[Basic Visualization Tutorial](tutorials/01_basic_visualization.md)** - Learn the basics
3. **[Gallery](gallery.md)** - Explore examples

## Getting Help

If you encounter issues:

1. Check [Troubleshooting](#troubleshooting) section above
2. Search [GitHub Issues](https://github.com/dadmaan/ghsom-toolkits/issues)
3. Open a new issue with:
   - Python version (`python --version`)
   - Operating system
   - Installation command used
   - Full error message
