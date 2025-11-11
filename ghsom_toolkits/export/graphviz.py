"""Graphviz/pydot helper functions for GHSOM visualization."""

from typing import Any, Dict

import pydot


def get_node_id(lookup_table: Dict[str, Any], node: Any) -> str:
    """
    Get unique ID for GHSOM node from lookup table.

    Parameters
    ----------
    lookup_table : dict
        Dictionary mapping node IDs to node objects
    node : object
        GHSOM node object to find

    Returns
    -------
    str
        The unique ID of the node

    Raises
    ------
    StopIteration
        If node is not found in lookup table
    """
    return next(key for key, value in lookup_table.items() if value is node)


def add_node_to_graph(
    graph: pydot.Dot,
    node_id: str,
    label: str,
    color: str = "#FFFFFF",
    node_size: float = 0.25,
) -> None:
    """
    Add a node to a pydot graph.

    Parameters
    ----------
    graph : pydot.Dot
        The graph to add the node to
    node_id : str
        Unique identifier for the node
    label : str
        Label text to display on the node
    color : str, optional
        Fill color for the node in hex format (default: "#FFFFFF")
    node_size : float, optional
        Size of the node in inches (default: 0.25)
    """
    graph.add_node(
        pydot.Node(
            node_id,
            label=label,
            style="filled",
            fillcolor=color,
            width=str(node_size),
            height=str(node_size),
        )
    )


def add_edge_to_graph(graph: pydot.Dot, parent_id: str, child_id: str) -> None:
    """
    Add an edge between two nodes in a graph.

    Parameters
    ----------
    graph : pydot.Dot
        The graph to add the edge to
    parent_id : str
        ID of the parent node
    child_id : str
        ID of the child node
    """
    graph.add_edge(pydot.Edge(parent_id, child_id))


def get_node_label(node: Any, node_id: str) -> str:
    """
    Create a formatted label for a GHSOM node.

    Parameters
    ----------
    node : object
        GHSOM node object with position, input_dataset_size, and children attributes
    node_id : str
        Unique identifier for the node

    Returns
    -------
    str
        Formatted label string with node information
    """
    return (
        f"ID: {node_id}\n"
        f"Pos: {node.position}\n"
        f"Size: {node.input_dataset_size}\n"
        f"Children: {len(node.children)}"
    )
