"""Report generation utilities for GHSOM models."""

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from ghsom_toolkits.interactive.explorer import get_node_statistics
from ghsom_toolkits.plotting.hierarchy import visualize_ghsom_hierarchy
from ghsom_toolkits.plotting.clusters import plot_cluster_distribution

logger = logging.getLogger(__name__)


# HTML template for reports
REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GHSOM Model Report - {{ timestamp }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 { margin: 0; font-size: 2.5em; }
        h2 {
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        h3 { color: #555; margin-top: 25px; }
        .metadata {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }
        .figure-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .figure-container img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #eee;
        }
        tr:hover { background-color: #f8f9fa; }
        .footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        .highlight {
            background-color: #fff3cd;
            padding: 2px 6px;
            border-radius: 3px;
        }
        .section-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 30px 0 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section-header h3 {
            margin: 0;
            color: white;
            font-size: 1.3em;
        }
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🗺️ GHSOM Model Report</h1>
        <p>Generated on {{ timestamp }}</p>
    </div>

    <div class="metadata">
        <h2>📋 Model Information</h2>
        {% if model_params %}
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            {% for key, value in model_params.items() %}
            <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>

    <h2>📊 Key Metrics</h2>
    <div class="metrics-grid">
        {% for metric_name, metric_value in metrics.items() %}
        <div class="metric-card">
            <div class="metric-label">{{ metric_name }}</div>
            <div class="metric-value">{{ metric_value }}</div>
        </div>
        {% endfor %}
    </div>

    {% if hierarchy_plots or heatmap_plots or cluster_plots or training_plots or other_plots %}
    <h2>📈 Visualizations</h2>

    {% if hierarchy_plots %}
    <div class="section-header">
        <h3>🌲 Hierarchy & Structure</h3>
    </div>
    {% for plot_name, plot_data in hierarchy_plots.items() %}
    <div class="figure-container">
        <h3>{{ plot_name }}</h3>
        <img src="data:image/png;base64,{{ plot_data }}" alt="{{ plot_name }}">
    </div>
    {% endfor %}
    {% endif %}

    {% if heatmap_plots %}
    <div class="section-header">
        <h3>🎨 Weight & Neuron Analysis</h3>
    </div>
    <div class="plot-grid">
    {% for plot_name, plot_data in heatmap_plots.items() %}
        <div class="figure-container">
            <h3>{{ plot_name }}</h3>
            <img src="data:image/png;base64,{{ plot_data }}" alt="{{ plot_name }}">
        </div>
    {% endfor %}
    </div>
    {% endif %}

    {% if cluster_plots %}
    <div class="section-header">
        <h3>📊 Cluster Analysis</h3>
    </div>
    <div class="plot-grid">
    {% for plot_name, plot_data in cluster_plots.items() %}
        <div class="figure-container">
            <h3>{{ plot_name }}</h3>
            <img src="data:image/png;base64,{{ plot_data }}" alt="{{ plot_name }}">
        </div>
    {% endfor %}
    </div>
    {% endif %}

    {% if training_plots %}
    <div class="section-header">
        <h3>📈 Training Dynamics</h3>
    </div>
    {% for plot_name, plot_data in training_plots.items() %}
    <div class="figure-container">
        <h3>{{ plot_name }}</h3>
        <img src="data:image/png;base64,{{ plot_data }}" alt="{{ plot_name }}">
    </div>
    {% endfor %}
    {% endif %}

    {% if other_plots %}
    <div class="section-header">
        <h3>📋 Additional Visualizations</h3>
    </div>
    {% for plot_name, plot_data in other_plots.items() %}
    <div class="figure-container">
        <h3>{{ plot_name }}</h3>
        <img src="data:image/png;base64,{{ plot_data }}" alt="{{ plot_name }}">
    </div>
    {% endfor %}
    {% endif %}
    {% endif %}

    {% if additional_info %}
    <h2>ℹ️ Additional Information</h2>
    <div class="metadata">
        {{ additional_info }}
    </div>
    {% endif %}

    <div class="footer">
        <p>Generated with <a href="https://github.com/dadmaan/ghsom-toolkits" target="_blank">ghsom-toolkits</a></p>
        <p>🤖 Generated with <a href="https://claude.com/claude-code" target="_blank">Claude Code</a></p>
    </div>
</body>
</html>
"""


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str


def _image_file_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string.

    Parameters
    ----------
    image_path : str
        Path to image file (PNG, JPG, etc.)

    Returns
    -------
    str
        Base64 encoded image data
    """
    with open(image_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    return img_data


def generate_report(
    model: Any,
    data: np.ndarray,
    output_path: str = "ghsom_report.html",
    lookup_table: Optional[Dict[str, Any]] = None,
    include_plots: bool = True,
    custom_info: Optional[str] = None,
    external_plots: Optional[Dict[str, str]] = None,
) -> str:
    """
    Generate a comprehensive HTML report for a GHSOM model.

    The report includes:
    - Model parameters and configuration
    - Key metrics (number of nodes, clusters, depth, etc.)
    - Visualizations (hierarchy, cluster distribution, etc.)
    - Additional custom information

    Parameters
    ----------
    model : object
        Trained GHSOM model
    data : numpy.ndarray
        Training data, shape (n_samples, n_features)
    output_path : str, optional
        Path to save the HTML report (default: "ghsom_report.html")
    lookup_table : dict, optional
        Lookup table for node IDs. If None, will be generated.
    include_plots : bool, optional
        Whether to include visualizations (default: True)
    custom_info : str, optional
        Additional custom information to include in the report (HTML formatted)
    external_plots : dict, optional
        Dictionary mapping plot names to image file paths.
        Plots are automatically categorized based on keywords in their names:
        - "hierarchy", "structure" → Hierarchy & Structure
        - "weight", "heatmap", "umatrix", "activation" → Weight & Neuron Analysis
        - "cluster", "distribution", "quality" → Cluster Analysis
        - "growth", "timeline", "training" → Training Dynamics
        Example: {"Weight Heatmap": "path/to/heatmap.png", ...}

    Returns
    -------
    str
        Path to the generated report file

    Raises
    ------
    ImportError
        If jinja2 is not installed

    Examples
    --------
    >>> from ghsom import GHSOM
    >>> from ghsom_toolkits.analysis import generate_report
    >>> ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
    >>> model = ghsom.train(epochs_number=50)
    >>>
    >>> # Basic report
    >>> report_path = generate_report(model, data, "my_report.html")
    >>>
    >>> # Report with external plots
    >>> external_plots = {
    >>>     "Weight Heatmap": "outputs/weight_heatmap.png",
    >>>     "Cluster Quality": "outputs/cluster_quality.png",
    >>> }
    >>> report_path = generate_report(model, data, "report.html",
    >>>                                external_plots=external_plots)
    """
    if not JINJA2_AVAILABLE:
        raise ImportError(
            "jinja2 is required for report generation. "
            "Install with: pip install ghsom-toolkits[interactive]"
        )

    logger.info("Generating GHSOM report...")

    # Build lookup table if not provided
    if lookup_table is None:
        lookup_table = {}

        def build_lookup(n: Any, prefix: str) -> None:
            lookup_table[prefix] = n
            for i, child in enumerate(n.children):
                build_lookup(child, f"{prefix}_c{i}")

        build_lookup(model, "root")

    # Collect model parameters
    model_params = {}
    if hasattr(model, 't1'):
        model_params['Growth Threshold (t1)'] = model.t1
    if hasattr(model, 't2'):
        model_params['Stopping Criterion (t2)'] = model.t2
    if hasattr(model, 'learning_rate'):
        model_params['Learning Rate'] = model.learning_rate
    if hasattr(model, 'gaussian_sigma'):
        model_params['Gaussian Sigma'] = model.gaussian_sigma
    if hasattr(model, 'decay'):
        model_params['Decay'] = model.decay

    model_params['Dataset Shape'] = f"{data.shape[0]} samples × {data.shape[1]} features"

    # Compute metrics
    stats = get_node_statistics(model)
    num_total_nodes = len(lookup_table)
    num_clusters = 0

    def count_leaves(n: Any) -> int:
        if not n.children:
            return 1
        return sum(count_leaves(child) for child in n.children)

    num_clusters = count_leaves(model)

    def get_depth(n: Any, d: int = 0) -> int:
        if not n.children:
            return d
        return max(get_depth(child, d + 1) for child in n.children)

    max_depth = get_depth(model)

    metrics = {
        'Total Nodes': num_total_nodes,
        'Leaf Clusters': num_clusters,
        'Max Depth': max_depth,
        'Root Map Size': f"{stats['map_shape'][0]}×{stats['map_shape'][1]}",
        'Expanded Neurons': stats['num_expanded_neurons'],
        'Expansion Rate': f"{stats['expansion_rate']:.1%}",
    }

    # Helper function to categorize plots
    def categorize_plot(plot_name: str) -> str:
        """Categorize plot based on name keywords."""
        name_lower = plot_name.lower()
        if any(kw in name_lower for kw in ['hierarchy', 'structure', 'tree']):
            return 'hierarchy'
        elif any(kw in name_lower for kw in ['weight', 'heatmap', 'umatrix', 'u-matrix', 'activation']):
            return 'heatmap'
        elif any(kw in name_lower for kw in ['cluster', 'distribution', 'quality']):
            return 'cluster'
        elif any(kw in name_lower for kw in ['growth', 'timeline', 'training', 'epoch']):
            return 'training'
        else:
            return 'other'

    # Initialize plot categories
    hierarchy_plots = {}
    heatmap_plots = {}
    cluster_plots = {}
    training_plots = {}
    other_plots = {}

    # Generate plots
    if include_plots:
        logger.info("Generating visualizations...")

        # Cluster distribution plot
        try:
            fig = plot_cluster_distribution(model, figsize=(10, 5))
            cluster_plots['Cluster Size Distribution'] = _fig_to_base64(fig)
        except Exception as e:
            logger.warning(f"Failed to generate cluster distribution: {e}")

        # Hierarchy visualization (if not too large)
        if num_total_nodes < 50:  # Only for smaller hierarchies
            try:
                # Create temporary file for graphviz output
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name

                visualize_ghsom_hierarchy(model, lookup_table, tmp_path)

                # Read and encode the image
                with open(tmp_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                hierarchy_plots['Hierarchy Structure'] = img_data

                # Clean up
                Path(tmp_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to generate hierarchy visualization: {e}")

    # Load external plots from files
    if external_plots:
        logger.info(f"Loading {len(external_plots)} external plots...")
        for plot_name, plot_path in external_plots.items():
            try:
                img_data = _image_file_to_base64(plot_path)
                category = categorize_plot(plot_name)

                if category == 'hierarchy':
                    hierarchy_plots[plot_name] = img_data
                elif category == 'heatmap':
                    heatmap_plots[plot_name] = img_data
                elif category == 'cluster':
                    cluster_plots[plot_name] = img_data
                elif category == 'training':
                    training_plots[plot_name] = img_data
                else:
                    other_plots[plot_name] = img_data

                logger.info(f"  ✓ Loaded {plot_name} ({category})")
            except FileNotFoundError:
                logger.warning(f"  ✗ Plot file not found: {plot_path}")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load {plot_name}: {e}")

    # Prepare template data
    template_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_params': model_params,
        'metrics': metrics,
        'hierarchy_plots': hierarchy_plots,
        'heatmap_plots': heatmap_plots,
        'cluster_plots': cluster_plots,
        'training_plots': training_plots,
        'other_plots': other_plots,
        'additional_info': custom_info,
    }

    # Render template
    template = Template(REPORT_TEMPLATE)
    html_content = template.render(**template_data)

    # Write to file
    output_path = Path(output_path)
    output_path.write_text(html_content, encoding='utf-8')

    logger.info(f"Report successfully generated: {output_path}")
    return str(output_path)
