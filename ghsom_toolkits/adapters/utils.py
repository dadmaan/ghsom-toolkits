"""
Utility functions for GHSOM adapters.

This module provides helper functions for working with adapted GHSOM models.
"""

from typing import Dict, Union


def build_lookup_table(
    root_node,
    lookup: Dict[str, object] = None,
    prefix: str = "root"
) -> Dict[str, object]:
    """
    Build a lookup table mapping node IDs to node objects.

    This is a convenience function that works with both adapted ghsom-py models
    and native GHSOMNode objects. It creates a flat lookup dictionary for
    easy node access in visualization and analysis functions.

    Parameters
    ----------
    root_node : object
        The root node to start from (adapted or native)
    lookup : dict, optional
        Existing lookup table to extend
    prefix : str, optional
        Prefix for node IDs (default: "root")

    Returns
    -------
    dict
        Lookup table mapping node IDs to node objects

    Examples
    --------
    >>> from ghsom import GHSOM
    >>> from ghsom_toolkits.adapters import adapt_model, build_lookup_table
    >>>
    >>> ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
    >>> result = ghsom.train(epochs_number=50)
    >>> model = adapt_model(result)
    >>> lookup = build_lookup_table(model)
    >>> "root" in lookup
    True
    """
    if lookup is None:
        lookup = {}

    lookup[prefix] = root_node

    # Check if node has children (works for both adapted and native nodes)
    if hasattr(root_node, 'children'):
        children = root_node.children
        if children:
            for i, child in enumerate(children):
                # Create unique ID for child based on its position if available
                if hasattr(child, 'position'):
                    pos = child.position
                    child_id = f"{prefix}_n{pos[0]}_{pos[1]}"
                else:
                    child_id = f"{prefix}_c{i}"

                build_lookup_table(child, lookup, child_id)

    return lookup


def count_nodes(root_node) -> int:
    """
    Count total number of nodes in hierarchy.

    Parameters
    ----------
    root_node : object
        The root node to start from

    Returns
    -------
    int
        Total number of nodes in the hierarchy
    """
    count = 1  # Count root

    if hasattr(root_node, 'children'):
        for child in root_node.children:
            count += count_nodes(child)

    return count


def get_max_level(root_node) -> int:
    """
    Get maximum level in hierarchy.

    Parameters
    ----------
    root_node : object
        The root node to start from

    Returns
    -------
    int
        Maximum level in the hierarchy
    """
    max_level = root_node.level if hasattr(root_node, 'level') else 0

    if hasattr(root_node, 'children'):
        for child in root_node.children:
            child_max = get_max_level(child)
            max_level = max(max_level, child_max)

    return max_level


def print_hierarchy(root_node, indent: int = 0) -> None:
    """
    Print hierarchy structure for debugging.

    Parameters
    ----------
    root_node : object
        The root node to start from
    indent : int
        Current indentation level
    """
    print("  " * indent + str(root_node))

    if hasattr(root_node, 'children'):
        for child in root_node.children:
            print_hierarchy(child, indent + 1)
