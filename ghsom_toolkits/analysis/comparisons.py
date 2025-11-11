"""Model comparison utilities for GHSOM."""

import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Import pandas only when needed
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


def _count_all_nodes(node: Any) -> int:
    """Recursively count all nodes in the hierarchy."""
    count = 1  # Count this node
    for child in node.children:
        count += _count_all_nodes(child)
    return count


def _get_max_depth(node: Any, current_depth: int = 0) -> int:
    """Recursively find maximum depth of hierarchy."""
    if not node.children:
        return current_depth

    max_child_depth = current_depth
    for child in node.children:
        child_depth = _get_max_depth(child, current_depth + 1)
        max_child_depth = max(max_child_depth, child_depth)

    return max_child_depth


def _count_leaf_nodes(node: Any) -> int:
    """Count leaf nodes (clusters) in the hierarchy."""
    if not node.children:
        return 1

    count = 0
    for child in node.children:
        count += _count_leaf_nodes(child)
    return count


def _compute_quantization_error(node: Any, data: np.ndarray) -> float:
    """
    Compute average quantization error for the model.

    Quantization error is the average distance between each sample
    and its best matching unit.
    """
    total_error = 0.0
    total_samples = 0

    def compute_node_qe(n: Any, node_data: np.ndarray) -> None:
        nonlocal total_error, total_samples

        if not n.children:
            # Leaf node - compute QE
            for sample in node_data:
                # Find BMU
                min_dist = float("inf")
                for r in range(n.rows):
                    for c in range(n.columns):
                        dist = np.linalg.norm(sample - n.map[r][c].weights)
                        min_dist = min(min_dist, dist)

                total_error += min_dist
                total_samples += 1
        else:
            # Internal node - distribute data to children
            for sample in node_data:
                # Find BMU
                min_dist = float("inf")
                bmu_pos = (0, 0)
                for r in range(n.rows):
                    for c in range(n.columns):
                        dist = np.linalg.norm(sample - n.map[r][c].weights)
                        if dist < min_dist:
                            min_dist = dist
                            bmu_pos = (r, c)

                # If BMU has child, recurse
                bmu_neuron = n.map[bmu_pos[0]][bmu_pos[1]]
                if bmu_neuron.children:
                    compute_node_qe(bmu_neuron.children[0], np.array([sample]))

    compute_node_qe(node, data)

    return total_error / total_samples if total_samples > 0 else 0.0


def compare_models(
    models: List[Any],
    data: np.ndarray,
    model_names: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """
    Compare multiple GHSOM models.

    Parameters
    ----------
    models : list of objects
        List of trained GHSOM models to compare
    data : numpy.ndarray
        Input data used for training, shape (n_samples, n_features)
    model_names : list of str, optional
        Names for each model. If None, uses "Model 1", "Model 2", etc.
    metrics : list of str, optional
        Metrics to compute. If None, computes all available metrics.
        Available: 'num_nodes', 'num_clusters', 'depth', 'qe', 'avg_cluster_size'

    Returns
    -------
    pandas.DataFrame
        Comparison table with models as rows and metrics as columns

    Raises
    ------
    ImportError
        If pandas is not installed

    Examples
    --------
    >>> from ghsom import GHSOM
    >>> model1 = GHSOM(input_dataset=data, t1=0.5, t2=0.05).train(epochs_number=50)
    >>> model2 = GHSOM(input_dataset=data, t1=0.3, t2=0.03).train(epochs_number=50)
    >>> comparison = compare_models([model1, model2], data, ['Loose', 'Tight'])
    >>> print(comparison)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for model comparison. "
            "Install with: pip install ghsom-toolkits[pandas]"
        )

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]

    if len(model_names) != len(models):
        raise ValueError(f"Number of model names ({len(model_names)}) must match number of models ({len(models)})")

    # Default metrics
    if metrics is None:
        metrics = ['num_nodes', 'num_clusters', 'depth', 'qe', 'avg_cluster_size']

    # Compute metrics for each model
    results = []

    for model, name in zip(models, model_names):
        model_metrics = {'model_name': name}

        if 'num_nodes' in metrics:
            model_metrics['num_nodes'] = _count_all_nodes(model)

        if 'num_clusters' in metrics:
            model_metrics['num_clusters'] = _count_leaf_nodes(model)

        if 'depth' in metrics:
            model_metrics['depth'] = _get_max_depth(model)

        if 'qe' in metrics:
            model_metrics['qe'] = _compute_quantization_error(model, data)

        if 'avg_cluster_size' in metrics:
            num_clusters = _count_leaf_nodes(model)
            model_metrics['avg_cluster_size'] = len(data) / num_clusters if num_clusters > 0 else 0

        # Add model parameters if available
        if hasattr(model, 't1'):
            model_metrics['t1'] = model.t1
        if hasattr(model, 't2'):
            model_metrics['t2'] = model.t2

        results.append(model_metrics)

    df = pd.DataFrame(results)
    df = df.set_index('model_name')

    logger.info(f"Compared {len(models)} models across {len(metrics)} metrics")

    return df


def plot_comparison(
    comparison_df: "pd.DataFrame",
    metrics: Optional[List[str]] = None,
    plot_type: str = "bar",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize model comparison results.

    Parameters
    ----------
    comparison_df : pandas.DataFrame
        DataFrame from compare_models()
    metrics : list of str, optional
        Metrics to plot. If None, plots all metrics in the DataFrame.
    plot_type : str, optional
        Type of plot: 'bar', 'radar', or 'heatmap' (default: 'bar')
    figsize : tuple, optional
        Figure size in inches (default: (12, 6))
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure

    Examples
    --------
    >>> comparison = compare_models([model1, model2], data)
    >>> fig = plot_comparison(comparison, plot_type='bar')
    >>> plt.show()
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for plotting comparisons. "
            "Install with: pip install ghsom-toolkits[pandas]"
        )

    if metrics is None:
        # Use all numeric columns
        metrics = comparison_df.select_dtypes(include=[np.number]).columns.tolist()

    if not metrics:
        raise ValueError("No metrics to plot")

    # Filter to requested metrics
    plot_df = comparison_df[metrics]

    if plot_type == "bar":
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]
            plot_df[metric].plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('')
            ax.set_ylabel('Value')
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        fig.suptitle("Model Comparison", fontsize=14, y=1.02)
        plt.tight_layout()

    elif plot_type == "radar":
        # Normalize metrics for radar plot
        normalized = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min())
        normalized = normalized.fillna(0)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for idx, model_name in enumerate(normalized.index):
            values = normalized.loc[model_name].tolist()
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title("Model Comparison (Normalized)", y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

    elif plot_type == "heatmap":
        import seaborn as sns

        fig, ax = plt.subplots(figsize=figsize)

        # Normalize for better visualization
        normalized = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min())
        normalized = normalized.fillna(0)

        sns.heatmap(
            normalized.T,
            annot=plot_df.T,
            fmt='.3f',
            cmap='YlGnBu',
            ax=ax,
            cbar_kws={'label': 'Normalized Value'},
        )

        ax.set_title("Model Comparison Heatmap")
        ax.set_xlabel("Model")
        ax.set_ylabel("Metric")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'bar', 'radar', or 'heatmap'")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")

    return fig
