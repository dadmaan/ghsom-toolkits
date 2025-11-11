"""Hierarchical visualization functions for GHSOM."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pydot

from ghsom_toolkits.export.graphviz import (
    add_edge_to_graph,
    add_node_to_graph,
    get_node_id,
    get_node_label,
)
from ghsom_toolkits.utils.colors import get_level_colors

logger = logging.getLogger(__name__)

# Import plotly only when needed (optional dependency)
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def _add_descendants_to_graph(
    graph: pydot.Dot,
    node: Any,
    lookup_table: Dict[str, Any],
    color: str = "lightblue",
    node_size: float = 0.25,
) -> None:
    """
    Recursively add node and descendants to graph.

    Parameters
    ----------
    graph : pydot.Dot
        The graph to add nodes to
    node : object
        GHSOM node to add with its descendants
    lookup_table : dict
        Dictionary mapping node IDs to node objects
    color : str, optional
        Color for the nodes (default: "lightblue")
    node_size : float, optional
        Size of the nodes (default: 0.25)
    """
    node_id = next(key for key, value in lookup_table.items() if value is node)
    node_label = get_node_label(node, node_id)
    add_node_to_graph(graph, node_id, node_label, color, node_size)

    for child in node.children:
        child_id = next(key for key, value in lookup_table.items() if value is child)
        add_edge_to_graph(graph, node_id, child_id)
        _add_descendants_to_graph(graph, child, lookup_table, color, node_size)


def visualize_ghsom_hierarchy(
    node: Any,
    lookup_table: Dict[str, Any],
    filename: str = "ghsom_tree.png",
) -> None:
    """
    Visualize the GHSOM hierarchy using pydot and Graphviz in tree format.

    Creates a hierarchical tree visualization where nodes are colored by their
    level in the hierarchy. Each node displays its ID, position, dataset size,
    and number of children.

    Parameters
    ----------
    node : object
        The root node of the GHSOM tree. Must have attributes: level, position,
        input_dataset_size, children (list of child nodes)
    lookup_table : dict
        Dictionary mapping unique node IDs (strings) to node objects
    filename : str, optional
        The filename where the output image will be saved (default: "ghsom_tree.png")

    Returns
    -------
    None
        The function saves the visualization to the specified filename

    Examples
    --------
    >>> from ghsom import GHSOM
    >>> ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
    >>> model = ghsom.train(epochs_number=50)
    >>> lookup = {f"node_{i}": node for i, node in enumerate(all_nodes)}
    >>> visualize_ghsom_hierarchy(model, lookup, "my_hierarchy.png")
    """
    level_colors = get_level_colors()

    def add_nodes_edges_graphviz(
        node: Any,
        graph: pydot.Dot,
        lookup_table: Dict[str, Any],
        parent_name: Optional[str] = None,
    ) -> None:
        """Recursive helper function to add nodes and edges to the graph."""
        node_id = get_node_id(lookup_table, node)
        node_color = level_colors.get(
            node.level, "grey"
        )  # Default to grey if level color not defined
        node_label = get_node_label(node, node_id)
        add_node_to_graph(graph, node_id, node_label, node_color)

        if parent_name:
            add_edge_to_graph(graph, parent_name, node_id)

        for child in node.children:
            add_nodes_edges_graphviz(child, graph, lookup_table, node_id)

    graph = pydot.Dot(graph_type="digraph", rankdir="TB")
    add_nodes_edges_graphviz(node, graph, lookup_table)

    graph.write_png(filename)
    logger.info(f"Graph saved to {filename}")


def visualize_node_position(
    root_node: Any,
    lookup_table: Dict[str, Any],
    node_id: str,
    filename: str = "node_position.png",
    plot_descendants: bool = False,
    target_node_color: str = "#5e81ac",
) -> None:
    """
    Visualize the relative position of a node within the GHSOM hierarchy.

    The visualization includes the target node's ancestors and optionally its descendants.
    The target node is highlighted with a distinct color.

    Parameters
    ----------
    root_node : object
        The root node of the GHSOM tree
    lookup_table : dict
        Dictionary mapping node IDs (strings) to node objects
    node_id : str
        The unique ID of the target node to highlight
    filename : str, optional
        The filename where the output image will be saved (default: "node_position.png")
    plot_descendants : bool, optional
        If True, all descendants of the target node will be included in the visualization.
        If False, only the target node and its ancestors will be visualized (default: False)
    target_node_color : str, optional
        The color used to highlight the target node in hex format (default: "#5e81ac")

    Returns
    -------
    None
        The function saves the visualization to the specified filename

    Raises
    ------
    ValueError
        If the specified node_id is not found in the lookup table

    Examples
    --------
    >>> visualize_node_position(
    ...     root_node=model,
    ...     lookup_table=lookup,
    ...     node_id="node_42",
    ...     filename="node_42_position.png",
    ...     plot_descendants=True
    ... )
    """
    target_node = lookup_table.get(node_id)
    if not target_node:
        raise ValueError(f"Node with ID {node_id} not found in the lookup table.")

    graph = pydot.Dot(graph_type="digraph", rankdir="TB", strict=True)
    target_node_label = get_node_label(target_node, node_id)
    node_size = 0.5  # Size of the nodes

    # Add the target node to the graph
    add_node_to_graph(graph, node_id, target_node_label, target_node_color, node_size)

    if plot_descendants:
        # Add all descendants of the target node to the graph
        _add_descendants_to_graph(
            graph, target_node, lookup_table, color="lightblue", node_size=node_size
        )

    # Traverse upwards to add ancestor nodes and edges to the graph
    current_node = target_node
    current_id = node_id
    while current_node is not root_node:
        parent_node = None
        for key, node in lookup_table.items():
            if current_node in node.children:
                parent_node = node
                parent_id = key
                break
        if parent_node is None:
            break

        parent_label = get_node_label(parent_node, parent_id)
        add_node_to_graph(graph, parent_id, parent_label, node_size=node_size)
        add_edge_to_graph(graph, parent_id, current_id)

        current_node = parent_node
        current_id = parent_id

    # Add child nodes and edges to the graph
    for child in target_node.children:
        child_id = next(key for key, value in lookup_table.items() if value is child)
        child_label = get_node_label(child, child_id)
        add_node_to_graph(graph, child_id, child_label, node_size=node_size)
        add_edge_to_graph(graph, node_id, child_id)

    graph.write_png(filename)
    logger.info(f"Graph saved to {filename}")


def _collect_hierarchy_data(
    node: Any,
    lookup_table: Dict[str, Any],
    parent_id: str = "",
) -> Tuple[List[str], List[str], List[int], List[str]]:
    """
    Collect hierarchy data for treemap visualization.

    Parameters
    ----------
    node : object
        Current GHSOM node
    lookup_table : dict
        Lookup table mapping node IDs to nodes
    parent_id : str
        ID of parent node

    Returns
    -------
    tuple
        (labels, parents, values, ids) for plotly treemap
    """
    node_id = get_node_id(lookup_table, node)
    labels = [f"{node_id}<br>Size: {node.input_dataset_size}"]
    parents = [parent_id]
    values = [node.input_dataset_size]
    ids = [node_id]

    for child in node.children:
        child_labels, child_parents, child_values, child_ids = _collect_hierarchy_data(
            child, lookup_table, node_id
        )
        labels.extend(child_labels)
        parents.extend(child_parents)
        values.extend(child_values)
        ids.extend(child_ids)

    return labels, parents, values, ids


def plot_hierarchy_treemap(
    node: Any,
    lookup_table: Dict[str, Any],
    filename: Optional[str] = None,
    width: int = 1200,
    height: int = 800,
) -> Optional[go.Figure]:
    """
    Create an interactive treemap visualization of the GHSOM hierarchy.

    The treemap shows the hierarchical structure where the size of each rectangle
    is proportional to the number of data points in that cluster.

    Parameters
    ----------
    node : object
        The root node of the GHSOM tree
    lookup_table : dict
        Dictionary mapping node IDs to node objects
    filename : str, optional
        Path to save the interactive HTML file. If None, returns figure without saving.
    width : int, optional
        Width of the plot in pixels (default: 1200)
    height : int, optional
        Height of the plot in pixels (default: 800)

    Returns
    -------
    plotly.graph_objects.Figure or None
        The treemap figure if plotly is available, None otherwise

    Raises
    ------
    ImportError
        If plotly is not installed

    Examples
    --------
    >>> from ghsom import GHSOM
    >>> ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
    >>> model = ghsom.train(epochs_number=50)
    >>> lookup = {"root": model}
    >>> fig = plot_hierarchy_treemap(model, lookup, "treemap.html")
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for treemap visualization. "
            "Install with: pip install ghsom-toolkits[interactive]"
        )

    # Collect hierarchy data
    labels, parents, values, ids = _collect_hierarchy_data(node, lookup_table)

    # Create treemap
    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            ids=ids,
            branchvalues="total",
            marker=dict(
                colorscale="Viridis",
                cmid=sum(values) / len(values),
                line=dict(width=2),
            ),
            textposition="middle center",
            hovertemplate="<b>%{label}</b><br>Samples: %{value}<extra></extra>",
        )
    )

    fig.update_layout(
        title="GHSOM Hierarchy Treemap",
        width=width,
        height=height,
        margin=dict(t=50, l=25, r=25, b=25),
    )

    if filename:
        fig.write_html(filename)
        logger.info(f"Treemap saved to {filename}")

    return fig
