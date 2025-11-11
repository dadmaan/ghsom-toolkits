"""Cluster visualization functions for GHSOM."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# Import scipy/sklearn only when needed (optional dependencies)
try:
    from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


def plot_ghsom_clusters(
    df: "pd.DataFrame",
    cluster_column: str = "GHSOM_cluster",
) -> None:
    """
    Plot GHSOM cluster distribution as a bar plot.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the cluster assignments
    cluster_column : str, optional
        Name of the column containing cluster IDs (default: "GHSOM_cluster")

    Returns
    -------
    None
        Displays the plot using matplotlib.pyplot.show()

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"GHSOM_cluster": [0, 0, 1, 1, 2, 2, 2]})
    >>> plot_ghsom_clusters(df)
    """
    # Count the frequency of each cluster
    cluster_counts = df[cluster_column].value_counts()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    cluster_counts.plot(kind="bar", color="black", fontsize=8)
    plt.title("Frequency of GHSOM Clusters")
    plt.xlabel("Cluster ID")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show()


def plot_ghsom_clusters_pie(
    data: "pd.DataFrame",
    cluster_column: str = "GHSOM_cluster",
) -> None:
    """
    Plot GHSOM cluster distribution as a pie chart.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the cluster assignments
    cluster_column : str, optional
        Name of the column containing cluster IDs (default: "GHSOM_cluster")

    Returns
    -------
    None
        Displays the plot using matplotlib.pyplot.show()

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"GHSOM_cluster": [0, 0, 1, 1, 2, 2, 2]})
    >>> plot_ghsom_clusters_pie(df)
    """
    # Count the frequency of each cluster
    cluster_counts = data[cluster_column].value_counts()

    # Create a pie plot
    plt.figure(figsize=(8, 8))
    plt.pie(
        cluster_counts,
        labels=cluster_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=plt.cm.tab20.colors,
    )
    plt.title("Proportion of GHSOM Clusters")
    plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


def plot_ghsom_clusters_genre_stacked(df: "pd.DataFrame") -> None:
    """
    Plot GHSOM clusters with genre distribution as a stacked bar chart.

    This function creates a stacked bar plot showing the distribution of genres
    within each GHSOM cluster, sorted by total count.

    Note: This function is specific to datasets with a 'genre' column and serves
    as an example of domain-specific visualization.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'GHSOM_cluster' and 'genre' columns

    Returns
    -------
    None
        Displays the plot using matplotlib.pyplot.show()

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "GHSOM_cluster": [0, 0, 1, 1, 2, 2],
    ...     "genre": ["rock", "jazz", "rock", "rock", "jazz", "classical"]
    ... })
    >>> plot_ghsom_clusters_genre_stacked(df)
    """
    # Aggregate the data to count the number of occurrences of each genre within each GHSOM cluster
    cluster_genre_counts = (
        df.groupby(["GHSOM_cluster", "genre"]).size().unstack("genre").fillna(0)
    )

    # Calculate the total count for each 'GHSOM_cluster' and sort by this total count
    cluster_genre_counts["total_count"] = cluster_genre_counts.sum(axis=1)
    cluster_genre_counts_sorted = cluster_genre_counts.sort_values(
        "total_count", ascending=False
    )

    # Drop the 'total_count' column as it's no longer needed for plotting
    cluster_genre_counts_sorted = cluster_genre_counts_sorted.drop(
        columns="total_count"
    )

    # Plot the counts as a stacked bar plot
    ax = cluster_genre_counts_sorted.plot(
        kind="bar", stacked=True, figsize=(10, 6), fontsize=8
    )

    # Set plot title and labels
    ax.set_title("GHSOM Cluster Distribution by Genre")
    ax.set_xlabel("GHSOM Cluster")
    ax.set_ylabel("Count")

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=90, ha="right")

    # Display the legend
    ax.legend(title="Genre", loc="upper right")

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_cluster_distribution(
    node: Any,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot distribution of cluster sizes across the GHSOM hierarchy.

    Parameters
    ----------
    node : object
        Root GHSOM node
    figsize : tuple, optional
        Figure size in inches (default: (10, 6))
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure

    Examples
    --------
    >>> from ghsom import GHSOM
    >>> ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
    >>> model = ghsom.train(epochs_number=50)
    >>> fig = plot_cluster_distribution(model)
    >>> plt.show()
    """
    # Collect cluster sizes from all leaf nodes
    def collect_leaf_sizes(n: Any, sizes: List[int]) -> None:
        if not n.children:  # Leaf node
            sizes.append(n.input_dataset_size)
        else:
            for child in n.children:
                collect_leaf_sizes(child, sizes)

    cluster_sizes = []
    collect_leaf_sizes(node, cluster_sizes)

    # Create histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1.hist(cluster_sizes, bins=min(30, len(cluster_sizes)), color="steelblue", edgecolor="black")
    ax1.set_xlabel("Cluster Size (Number of Samples)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Cluster Sizes")
    ax1.grid(axis="y", alpha=0.3)

    # Box plot
    ax2.boxplot(cluster_sizes, vert=True)
    ax2.set_ylabel("Cluster Size (Number of Samples)")
    ax2.set_title("Cluster Size Statistics")
    ax2.grid(axis="y", alpha=0.3)

    # Add statistics text
    stats_text = (
        f"Total Clusters: {len(cluster_sizes)}\n"
        f"Mean: {np.mean(cluster_sizes):.1f}\n"
        f"Median: {np.median(cluster_sizes):.1f}\n"
        f"Min: {np.min(cluster_sizes)}\n"
        f"Max: {np.max(cluster_sizes)}"
    )
    fig.text(0.02, 0.98, stats_text, transform=fig.transFigure,
             fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Cluster distribution saved to {save_path}")

    return fig


def plot_cluster_quality(
    data: np.ndarray,
    cluster_labels: np.ndarray,
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot cluster quality metrics (Silhouette Score and Davies-Bouldin Index).

    Parameters
    ----------
    data : numpy.ndarray
        Input data, shape (n_samples, n_features)
    cluster_labels : numpy.ndarray
        Cluster assignment for each sample, shape (n_samples,)
    figsize : tuple, optional
        Figure size in inches (default: (14, 5))
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure

    Raises
    ------
    ImportError
        If scikit-learn is not installed

    Examples
    --------
    >>> # After clustering with GHSOM
    >>> cluster_labels = np.array([0, 0, 1, 1, 2, 2])
    >>> fig = plot_cluster_quality(data, cluster_labels)
    >>> plt.show()
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for quality metrics. "
            "Install with: pip install ghsom-toolkits[analysis]"
        )

    # Calculate overall metrics
    silhouette_avg = silhouette_score(data, cluster_labels)
    davies_bouldin = davies_bouldin_score(data, cluster_labels)
    silhouette_values = silhouette_samples(data, cluster_labels)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Silhouette plot
    unique_labels = np.unique(cluster_labels)
    y_lower = 10

    for i, label in enumerate(unique_labels):
        cluster_silhouette_values = silhouette_values[cluster_labels == label]
        cluster_silhouette_values.sort()

        size_cluster = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster

        color = plt.cm.nipy_spectral(float(i) / len(unique_labels))
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax1.text(-0.05, y_lower + 0.5 * size_cluster, str(label))
        y_lower = y_upper + 10

    ax1.set_title("Silhouette Plot for Clusters")
    ax1.set_xlabel("Silhouette Coefficient")
    ax1.set_ylabel("Cluster")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--", label=f"Average: {silhouette_avg:.3f}")
    ax1.set_ylim([0, len(data) + (len(unique_labels) + 1) * 10])
    ax1.legend()

    # Per-cluster silhouette scores
    cluster_scores = []
    for label in unique_labels:
        score = np.mean(silhouette_values[cluster_labels == label])
        cluster_scores.append(score)

    ax2.bar(unique_labels, cluster_scores, color="steelblue", edgecolor="black")
    ax2.axhline(y=silhouette_avg, color="red", linestyle="--", label=f"Average: {silhouette_avg:.3f}")
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Mean Silhouette Score")
    ax2.set_title(f"Cluster Quality Metrics\nDavies-Bouldin: {davies_bouldin:.3f} (lower is better)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Cluster quality plot saved to {save_path}")

    return fig


def plot_growth_timeline(
    training_history: List[Dict[str, Any]],
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot how the GHSOM hierarchy evolved during training.

    Parameters
    ----------
    training_history : list of dict
        Training history with keys like 'epoch', 'num_nodes', 'depth', 'qe', 'te'
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
    >>> history = [
    ...     {'epoch': 0, 'num_nodes': 4, 'depth': 1, 'qe': 0.5},
    ...     {'epoch': 10, 'num_nodes': 8, 'depth': 2, 'qe': 0.3},
    ... ]
    >>> fig = plot_growth_timeline(history)
    >>> plt.show()
    """
    if not training_history:
        raise ValueError("Training history is empty")

    epochs = [entry.get("epoch", i) for i, entry in enumerate(training_history)]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Number of nodes over time
    if "num_nodes" in training_history[0]:
        num_nodes = [entry["num_nodes"] for entry in training_history]
        axes[0, 0].plot(epochs, num_nodes, marker="o", color="steelblue")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Number of Nodes")
        axes[0, 0].set_title("Hierarchy Growth")
        axes[0, 0].grid(alpha=0.3)

    # Plot 2: Depth over time
    if "depth" in training_history[0]:
        depth = [entry["depth"] for entry in training_history]
        axes[0, 1].plot(epochs, depth, marker="s", color="forestgreen")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Max Depth")
        axes[0, 1].set_title("Hierarchy Depth")
        axes[0, 1].grid(alpha=0.3)

    # Plot 3: Quantization Error over time
    if "qe" in training_history[0]:
        qe = [entry["qe"] for entry in training_history]
        axes[1, 0].plot(epochs, qe, marker="^", color="darkorange")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Quantization Error")
        axes[1, 0].set_title("Quantization Error")
        axes[1, 0].grid(alpha=0.3)

    # Plot 4: Topological Error over time (if available)
    if "te" in training_history[0]:
        te = [entry["te"] for entry in training_history]
        axes[1, 1].plot(epochs, te, marker="d", color="crimson")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Topological Error")
        axes[1, 1].set_title("Topological Error")
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "No topological error data",
                       ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].axis("off")

    fig.suptitle("GHSOM Training Timeline", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Growth timeline saved to {save_path}")

    return fig
