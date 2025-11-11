"""Heatmap visualization functions for GHSOM."""

import logging
from typing import Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_weight_heatmap(
    node: Any,
    neuron_position: Optional[Tuple[int, int]] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot weight vectors as a heatmap.

    If neuron_position is specified, plots weights for that specific neuron.
    Otherwise, plots all weight vectors in the map as a grid.

    Parameters
    ----------
    node : object
        GHSOM node containing the map with weight vectors
    neuron_position : tuple of (int, int), optional
        Position (row, col) of specific neuron to visualize. If None, plots all neurons.
    cmap : str, optional
        Matplotlib colormap name (default: "viridis")
    figsize : tuple of (int, int), optional
        Figure size in inches (default: (10, 8))
    save_path : str, optional
        Path to save the figure. If None, figure is not saved automatically.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object

    Examples
    --------
    >>> from ghsom import GHSOM
    >>> ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
    >>> model = ghsom.train(epochs_number=50)
    >>> fig = plot_weight_heatmap(model, neuron_position=(0, 0))
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    if neuron_position is not None:
        # Plot single neuron's weight vector
        row, col = neuron_position
        if not (0 <= row < node.rows and 0 <= col < node.columns):
            raise ValueError(
                f"Invalid neuron position {neuron_position}. "
                f"Map shape is ({node.rows}, {node.columns})"
            )

        weights = node.map[row][col].weights
        im = ax.imshow(weights.reshape(1, -1), cmap=cmap, aspect="auto")

        ax.set_title(f"Weight Vector for Neuron at Position {neuron_position}")
        ax.set_xlabel("Weight Dimension")
        ax.set_ylabel("Neuron")
        ax.set_yticks([0])
        ax.set_yticklabels([f"({row},{col})"])

        plt.colorbar(im, ax=ax, label="Weight Value")

    else:
        # Plot all weight vectors as a grid
        rows, cols = node.rows, node.columns
        weight_dim = node.map[0][0].weights.shape[0]

        # Collect all weights into a matrix
        weights_matrix = np.zeros((rows * cols, weight_dim))
        neuron_labels = []

        idx = 0
        for r in range(rows):
            for c in range(cols):
                weights_matrix[idx] = node.map[r][c].weights
                neuron_labels.append(f"({r},{c})")
                idx += 1

        # Create heatmap
        sns.heatmap(
            weights_matrix,
            cmap=cmap,
            ax=ax,
            cbar_kws={"label": "Weight Value"},
            yticklabels=neuron_labels,
            xticklabels=range(weight_dim),
        )

        ax.set_title(f"Weight Vectors Heatmap - Map Shape: ({rows}, {cols})")
        ax.set_xlabel("Weight Dimension")
        ax.set_ylabel("Neuron Position")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Weight heatmap saved to {save_path}")

    return fig


def plot_activation_map(
    node: Any,
    data: np.ndarray,
    sample_indices: Optional[Union[int, list]] = None,
    cmap: str = "YlOrRd",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot activation patterns showing which neurons activate for given data samples.

    The activation is measured as the inverse Euclidean distance between the sample
    and each neuron's weight vector (closer = higher activation).

    Parameters
    ----------
    node : object
        GHSOM node containing the trained map
    data : numpy.ndarray
        Input data samples, shape (n_samples, n_features)
    sample_indices : int or list of int, optional
        Index or indices of samples to visualize. If None, uses first sample.
    cmap : str, optional
        Matplotlib colormap name (default: "YlOrRd")
    figsize : tuple of (int, int), optional
        Figure size in inches (default: (8, 6))
    save_path : str, optional
        Path to save the figure. If None, figure is not saved automatically.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object

    Examples
    --------
    >>> fig = plot_activation_map(model, data, sample_indices=[0, 5, 10])
    >>> plt.show()
    """
    if sample_indices is None:
        sample_indices = [0]
    elif isinstance(sample_indices, int):
        sample_indices = [sample_indices]

    n_samples = len(sample_indices)
    rows, cols = node.rows, node.columns

    # Create subplots for multiple samples
    n_plot_cols = min(3, n_samples)
    n_plot_rows = (n_samples + n_plot_cols - 1) // n_plot_cols

    fig, axes = plt.subplots(n_plot_rows, n_plot_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for plot_idx, sample_idx in enumerate(sample_indices):
        if sample_idx >= len(data):
            raise ValueError(f"Sample index {sample_idx} out of range (data has {len(data)} samples)")

        sample = data[sample_idx]

        # Compute activation map (inverse of distances)
        activation = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                distance = np.linalg.norm(sample - node.map[r][c].weights)
                # Convert distance to activation (smaller distance = higher activation)
                activation[r, c] = 1.0 / (1.0 + distance)

        # Plot activation map
        ax = axes[plot_idx]
        im = ax.imshow(activation, cmap=cmap, aspect="auto")
        ax.set_title(f"Sample {sample_idx}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        # Add colorbar for each subplot
        plt.colorbar(im, ax=ax, label="Activation")

        # Mark the best matching unit (BMU)
        bmu_pos = np.unravel_index(np.argmax(activation), activation.shape)
        ax.plot(bmu_pos[1], bmu_pos[0], "r*", markersize=15, label="BMU")
        ax.legend()

    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Activation Maps", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Activation map saved to {save_path}")

    return fig


def plot_umatrix(
    node: Any,
    cmap: str = "gray_r",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot U-Matrix (Unified Distance Matrix) visualization.

    The U-Matrix shows the average distance between each neuron and its immediate
    neighbors, revealing cluster boundaries (high values) and cluster centers (low values).

    Parameters
    ----------
    node : object
        GHSOM node containing the trained map
    cmap : str, optional
        Matplotlib colormap name (default: "gray_r" - darker = closer neurons)
    figsize : tuple of (int, int), optional
        Figure size in inches (default: (8, 6))
    save_path : str, optional
        Path to save the figure. If None, figure is not saved automatically.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object

    Examples
    --------
    >>> fig = plot_umatrix(model)
    >>> plt.show()
    """
    rows, cols = node.rows, node.columns
    umatrix = np.zeros((rows, cols))

    # Compute average distance to neighbors for each neuron
    for r in range(rows):
        for c in range(cols):
            distances = []
            current_weights = node.map[r][c].weights

            # Check all 4-connected neighbors (up, down, left, right)
            neighbors = [
                (r - 1, c),  # up
                (r + 1, c),  # down
                (r, c - 1),  # left
                (r, c + 1),  # right
            ]

            for nr, nc in neighbors:
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor_weights = node.map[nr][nc].weights
                    distance = np.linalg.norm(current_weights - neighbor_weights)
                    distances.append(distance)

            # Average distance to neighbors
            umatrix[r, c] = np.mean(distances) if distances else 0

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(umatrix, cmap=cmap, aspect="auto")

    ax.set_title("U-Matrix (Unified Distance Matrix)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Add grid for better visualization
    ax.set_xticks(np.arange(cols) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label="Average Distance to Neighbors")
    cbar.ax.text(
        0.5,
        -0.05,
        "Low = Cluster Center\nHigh = Boundary",
        transform=cbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"U-Matrix saved to {save_path}")

    return fig
