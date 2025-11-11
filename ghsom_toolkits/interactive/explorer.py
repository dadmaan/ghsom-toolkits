"""Interactive exploration utilities for GHSOM models."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def explore_neuron(
    node: Any,
    neuron_position: Tuple[int, int],
    lookup_table: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get detailed information about a specific neuron.

    Parameters
    ----------
    node : object
        GHSOM node containing the map
    neuron_position : tuple of (int, int)
        Position (row, col) of the neuron to explore
    lookup_table : dict, optional
        Lookup table for node IDs (if needed)

    Returns
    -------
    dict
        Dictionary containing neuron metadata:
        - position: (row, col)
        - weights: weight vector
        - has_child: whether this neuron has expanded into a child map
        - child_node: reference to child node if exists
        - num_samples: number of samples mapped to this neuron
        - level: hierarchy level

    Examples
    --------
    >>> from ghsom import GHSOM
    >>> ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
    >>> model = ghsom.train(epochs_number=50)
    >>> info = explore_neuron(model, (0, 0))
    >>> print(info['weights'])
    """
    row, col = neuron_position

    if not (0 <= row < node.rows and 0 <= col < node.columns):
        raise ValueError(
            f"Invalid neuron position {neuron_position}. "
            f"Map shape is ({node.rows}, {node.columns})"
        )

    neuron = node.map[row][col]

    # Collect neuron information
    info = {
        "position": neuron_position,
        "weights": neuron.weights.copy(),
        "weight_shape": neuron.weights.shape,
        "has_child": len(neuron.children) > 0,
        "num_children": len(neuron.children),
        "level": node.level,
        "parent_map_shape": (node.rows, node.columns),
    }

    # Add child information if exists
    if neuron.children:
        child_node = neuron.children[0]
        info["child_node"] = child_node
        info["child_map_shape"] = (child_node.rows, child_node.columns)
        info["child_dataset_size"] = child_node.input_dataset_size
    else:
        info["child_node"] = None
        info["child_map_shape"] = None
        info["child_dataset_size"] = None

    # Get dataset size if available
    if hasattr(neuron, "input_dataset_size"):
        info["num_samples"] = neuron.input_dataset_size
    elif hasattr(node, "input_dataset_size"):
        # Approximate based on parent node
        total_neurons = node.rows * node.columns
        info["num_samples_approx"] = node.input_dataset_size // total_neurons
    else:
        info["num_samples"] = None

    return info


def get_subtree(
    node: Any,
    lookup_table: Dict[str, Any],
    node_id: str,
    max_depth: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract a subtree starting from a specific node.

    Parameters
    ----------
    node : object
        Root node of the hierarchy
    lookup_table : dict
        Lookup table mapping node IDs to node objects
    node_id : str
        ID of the node to use as subtree root
    max_depth : int, optional
        Maximum depth to extract (relative to subtree root). If None, extract entire subtree.

    Returns
    -------
    dict
        Subtree information with keys:
        - root_node: the subtree root node object
        - node_id: the node ID
        - subtree_lookup: lookup table for the subtree
        - num_nodes: number of nodes in subtree
        - max_depth: depth of subtree

    Examples
    --------
    >>> subtree = get_subtree(model, lookup, "root_c0", max_depth=2)
    >>> print(f"Subtree has {subtree['num_nodes']} nodes")
    """
    if node_id not in lookup_table:
        raise ValueError(f"Node ID '{node_id}' not found in lookup table")

    root_node = lookup_table[node_id]

    # Build subtree lookup table
    subtree_lookup = {}
    node_count = [0]
    max_depth_found = [0]

    def traverse(n: Any, prefix: str, current_depth: int) -> None:
        if max_depth is not None and current_depth > max_depth:
            return

        subtree_lookup[prefix] = n
        node_count[0] += 1
        max_depth_found[0] = max(max_depth_found[0], current_depth)

        for i, child in enumerate(n.children):
            child_id = f"{prefix}_c{i}"
            traverse(child, child_id, current_depth + 1)

    traverse(root_node, node_id, 0)

    return {
        "root_node": root_node,
        "node_id": node_id,
        "subtree_lookup": subtree_lookup,
        "num_nodes": node_count[0],
        "max_depth": max_depth_found[0],
    }


def trace_sample_path(
    root_node: Any,
    sample: np.ndarray,
    max_depth: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Trace the path of a sample through the GHSOM hierarchy.

    Finds the Best Matching Unit (BMU) at each level of the hierarchy,
    showing how the sample is progressively refined into finer clusters.

    Parameters
    ----------
    root_node : object
        Root node of the GHSOM hierarchy
    sample : numpy.ndarray
        Input sample to trace, shape (n_features,)
    max_depth : int, optional
        Maximum depth to trace. If None, traces to leaf node.

    Returns
    -------
    list of dict
        Path through hierarchy, where each dict contains:
        - level: hierarchy level
        - node: the node object at this level
        - bmu_position: (row, col) of best matching unit
        - bmu_distance: distance to BMU
        - map_shape: (rows, cols) of map at this level
        - has_child: whether the BMU has a child map

    Examples
    --------
    >>> path = trace_sample_path(model, data[0])
    >>> for step in path:
    ...     print(f"Level {step['level']}: BMU at {step['bmu_position']}")
    """
    if sample.ndim != 1:
        raise ValueError(f"Sample must be 1D array, got shape {sample.shape}")

    path = []
    current_node = root_node
    current_level = 0

    while current_node is not None:
        if max_depth is not None and current_level > max_depth:
            break

        # Find BMU in current map
        min_distance = float("inf")
        bmu_pos = (0, 0)

        for r in range(current_node.rows):
            for c in range(current_node.columns):
                neuron = current_node.map[r][c]
                distance = np.linalg.norm(sample - neuron.weights)

                if distance < min_distance:
                    min_distance = distance
                    bmu_pos = (r, c)

        # Get BMU neuron
        bmu_neuron = current_node.map[bmu_pos[0]][bmu_pos[1]]
        has_child = len(bmu_neuron.children) > 0

        # Record this step
        path.append(
            {
                "level": current_level,
                "node": current_node,
                "bmu_position": bmu_pos,
                "bmu_distance": min_distance,
                "map_shape": (current_node.rows, current_node.columns),
                "has_child": has_child,
            }
        )

        # Move to child map if exists
        if has_child:
            current_node = bmu_neuron.children[0]
            current_level += 1
        else:
            # Reached leaf node
            break

    return path


def get_node_statistics(node: Any) -> Dict[str, Any]:
    """
    Compute statistics for a GHSOM node.

    Parameters
    ----------
    node : object
        GHSOM node to analyze

    Returns
    -------
    dict
        Statistics including:
        - map_shape: (rows, cols)
        - num_neurons: total neurons in this map
        - num_leaf_neurons: neurons without children
        - num_expanded_neurons: neurons with children
        - dataset_size: number of samples
        - level: hierarchy level

    Examples
    --------
    >>> stats = get_node_statistics(model)
    >>> print(f"Map has {stats['num_neurons']} neurons")
    """
    rows, cols = node.rows, node.columns
    total_neurons = rows * cols

    # Count expanded vs leaf neurons
    expanded_count = 0
    for r in range(rows):
        for c in range(cols):
            if len(node.map[r][c].children) > 0:
                expanded_count += 1

    leaf_count = total_neurons - expanded_count

    stats = {
        "map_shape": (rows, cols),
        "num_neurons": total_neurons,
        "num_leaf_neurons": leaf_count,
        "num_expanded_neurons": expanded_count,
        "expansion_rate": expanded_count / total_neurons if total_neurons > 0 else 0,
        "dataset_size": node.input_dataset_size,
        "level": node.level,
    }

    return stats


def find_similar_neurons(
    node: Any,
    reference_position: Tuple[int, int],
    top_k: int = 5,
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Find neurons most similar to a reference neuron based on weight vectors.

    Parameters
    ----------
    node : object
        GHSOM node containing the map
    reference_position : tuple of (int, int)
        Position of reference neuron
    top_k : int, optional
        Number of similar neurons to return (default: 5)

    Returns
    -------
    list of tuple
        List of (position, distance) tuples, sorted by distance (most similar first)

    Examples
    --------
    >>> similar = find_similar_neurons(model, (0, 0), top_k=3)
    >>> for pos, dist in similar:
    ...     print(f"Neuron at {pos}: distance = {dist:.4f}")
    """
    ref_row, ref_col = reference_position

    if not (0 <= ref_row < node.rows and 0 <= ref_col < node.columns):
        raise ValueError(
            f"Invalid reference position {reference_position}. "
            f"Map shape is ({node.rows}, {node.columns})"
        )

    ref_weights = node.map[ref_row][ref_col].weights

    # Compute distances to all other neurons
    distances = []
    for r in range(node.rows):
        for c in range(node.columns):
            if (r, c) == reference_position:
                continue  # Skip reference neuron itself

            neuron_weights = node.map[r][c].weights
            distance = np.linalg.norm(ref_weights - neuron_weights)
            distances.append(((r, c), distance))

    # Sort by distance and return top_k
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]
