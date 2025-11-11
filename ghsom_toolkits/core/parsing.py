"""
GHSOM hierarchy parsing and lookup utilities.

This module provides optimized parsing functions for GHSOM hierarchy strings
and utilities for creating and managing node lookup tables.
"""

import logging
import re
from typing import Dict, Optional, Union

from ghsom_toolkits.core.node import GHSOMNode

logger = logging.getLogger(__name__)


def parse_ghsom_hierarchy(ghsom_hierarchy: str) -> GHSOMNode:
    """
    Parse GHSOM hierarchy string into efficient node tree.

    Parses a text representation of a GHSOM hierarchy and constructs a tree
    of GHSOMNode objects with proper parent-child relationships.

    Parameters
    ----------
    ghsom_hierarchy : str
        Text representation of GHSOM hierarchy with format:
        "position (x, y) -- map dimensions (r, c, d) -- input dataset N element(s) -- level L"

    Returns
    -------
    GHSOMNode
        Root node of the parsed hierarchy tree

    Raises
    ------
    ValueError
        If the hierarchy string is malformed or empty

    Examples
    --------
    >>> hierarchy_str = '''position (0, 0) -- map dimensions (3, 3, 2) -- input dataset 100 element(s) -- level 0
    ... position (1, 1) -- map dimensions (2, 2, 2) -- input dataset 25 element(s) -- level 1'''
    >>> root = parse_ghsom_hierarchy(hierarchy_str)
    >>> root.level
    0
    >>> len(root.children)
    1
    """
    if not ghsom_hierarchy or not ghsom_hierarchy.strip():
        raise ValueError("Empty hierarchy string provided")

    root = None
    node_stack = []

    for line_num, line in enumerate(ghsom_hierarchy.strip().split("\n"), 1):
        try:
            parts = line.strip().split("--")
            if len(parts) != 4:
                raise ValueError(f"Line {line_num}: Expected 4 parts separated by '--', got {len(parts)}")

            # Parse position: (x, y)
            pos_pattern = r"\((-?\d+),\s*(-?\d+)\)"
            position_str = re.findall(pos_pattern, parts[0])
            if not position_str:
                raise ValueError(f"Line {line_num}: Could not parse position from '{parts[0]}'")
            position_int = (int(position_str[0][0]), int(position_str[0][1]))

            # Parse dimensions: (rows, cols, depth)
            dim_pattern = r"\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)"
            dimension_str = re.findall(dim_pattern, parts[1])
            if not dimension_str:
                raise ValueError(f"Line {line_num}: Could not parse dimensions from '{parts[1]}'")
            dimension_int = (int(dimension_str[0][0]), int(dimension_str[0][1]), int(dimension_str[0][2]))

            # Parse input dataset size
            dataset_match = re.search(r"(\d+)\s+element", parts[2])
            if not dataset_match:
                raise ValueError(f"Line {line_num}: Could not parse dataset size from '{parts[2]}'")
            input_dataset_size = int(dataset_match.group(1))

            # Parse level
            level_match = re.search(r"level\s+(\d+)", parts[3])
            if not level_match:
                raise ValueError(f"Line {line_num}: Could not parse level from '{parts[3]}'")
            level = int(level_match.group(1))

            # Create new node
            new_node = GHSOMNode(
                position=position_int,
                map_dimensions=dimension_int,
                input_dataset_size=input_dataset_size,
                level=level
            )

            # Check if this is the root node
            if level == 0:
                if root is not None:
                    logger.warning(f"Line {line_num}: Multiple root nodes found, using first one")
                else:
                    root = new_node
                    node_stack = [new_node]
            else:
                # Pop nodes from the stack until we find the parent level
                while node_stack and node_stack[-1].level >= level:
                    node_stack.pop()

                if not node_stack:
                    raise ValueError(f"Line {line_num}: No parent found for level {level} node")

                # The current top of the stack is the parent
                parent_node = node_stack[-1]
                parent_node.add_child(new_node)
                node_stack.append(new_node)

        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing line {line_num}: {str(e)}")

    if root is None:
        raise ValueError("No root node found in hierarchy string")

    return root


def create_lookup_table(
    root_node: GHSOMNode,
    short_id: bool = False
) -> Dict[Union[str, int], GHSOMNode]:
    """
    Create lookup table for the GHSOM hierarchy with unique IDs.

    Assigns a unique ID to each node in the hierarchy and creates a dictionary
    mapping IDs to nodes for fast lookup operations.

    Parameters
    ----------
    root_node : GHSOMNode
        The root node of the GHSOM tree
    short_id : bool, optional
        If True, creates simple integer IDs for each cluster.
        Otherwise, encodes position, map dimensions and level into ID.
        Default is False.

    Returns
    -------
    Dict[Union[str, int], GHSOMNode]
        Lookup table where keys are unique IDs and values are nodes.
        Root node always has key "root".

    Examples
    --------
    >>> root = GHSOMNode((0, 0), (3, 3, 2), 100, 0)
    >>> lookup = create_lookup_table(root, short_id=True)
    >>> "root" in lookup
    True
    """
    lookup_table = {}
    next_id = 1

    def traverse_and_add_to_table(
        node: GHSOMNode,
        lookup_table: Dict[Union[str, int], GHSOMNode],
        next_id: int,
        short_id: bool
    ) -> int:
        """
        Recursively traverse tree and add each node to lookup table.

        Parameters
        ----------
        node : GHSOMNode
            Current node being visited
        lookup_table : dict
            Lookup table being populated
        next_id : int
            Next available ID number
        short_id : bool
            Whether to use short integer IDs

        Returns
        -------
        int
            Next available ID after processing this node and its children
        """
        if node is root_node:
            node_id = "root"
            lookup_table[node_id] = node
        else:
            if short_id:
                node_id = next_id
            else:
                # Create identifier using ID, level, position, and dimensions
                node_id = int(
                    f"{next_id}{node.level}{node.position[0]}{node.position[1]}"
                    f"{node.map_dimensions[0]}{node.map_dimensions[1]}{node.map_dimensions[2]}"
                )

            lookup_table[node_id] = node
            next_id += 1

        # Recursively add child nodes
        for child in node.children:
            next_id = traverse_and_add_to_table(child, lookup_table, next_id, short_id)

        return next_id

    traverse_and_add_to_table(root_node, lookup_table, next_id, short_id)

    return lookup_table


def encode_ghsom_node(
    lookup_table: Dict[Union[str, int], GHSOMNode],
    input_node: Union[str, GHSOMNode]
) -> Optional[Union[str, int]]:
    """
    Encode GHSOM node by finding its key in the lookup table.

    Parameters
    ----------
    lookup_table : Dict[Union[str, int], GHSOMNode]
        Lookup table containing node IDs and nodes
    input_node : Union[str, GHSOMNode]
        Node or node string representation to encode

    Returns
    -------
    Optional[Union[str, int]]
        Key corresponding to the node in lookup table, or None if not found

    Examples
    --------
    >>> root = GHSOMNode((0, 0), (3, 3, 2), 100, 0)
    >>> lookup = create_lookup_table(root)
    >>> encode_ghsom_node(lookup, root)
    'root'
    """
    if not isinstance(input_node, str):
        input_node_str = str(input_node)
    else:
        input_node_str = input_node

    for key, node in lookup_table.items():
        if str(node) == input_node_str:
            return key

    return None


def decode_ghsom_node(
    lookup_table: Dict[Union[str, int], GHSOMNode],
    node_id: Union[str, int]
) -> Optional[GHSOMNode]:
    """
    Retrieve GHSOM node from lookup table by ID.

    Parameters
    ----------
    lookup_table : Dict[Union[str, int], GHSOMNode]
        Lookup table containing node IDs and nodes
    node_id : Union[str, int]
        Unique identifier of the node to retrieve

    Returns
    -------
    Optional[GHSOMNode]
        Node associated with the given ID, or None if not found

    Examples
    --------
    >>> root = GHSOMNode((0, 0), (3, 3, 2), 100, 0)
    >>> lookup = create_lookup_table(root)
    >>> node = decode_ghsom_node(lookup, "root")
    >>> node.level
    0
    """
    return lookup_table.get(node_id)


def get_ghsom_node_statistics(
    lookup_table: Dict[Union[str, int], GHSOMNode],
    node_id: Union[str, int]
) -> Optional[dict]:
    """
    Report statistics for a specific GHSOM node.

    Parameters
    ----------
    lookup_table : Dict[Union[str, int], GHSOMNode]
        Lookup table containing node IDs and nodes
    node_id : Union[str, int]
        Unique ID of the target node

    Returns
    -------
    Optional[dict]
        Dictionary containing node statistics, or None if node not found.
        Statistics include:
        - node_id: The node's ID
        - position: Node position
        - map_dimensions: Map dimensions
        - input_dataset_size: Dataset size
        - level: Hierarchy level
        - number_of_children: Number of child nodes
        - children: List of child info (if children exist)
        - total_input_dataset_size_of_children: Sum of children's dataset sizes
    """
    node = lookup_table.get(node_id)

    if not node:
        return None

    statistics = {
        "node_id": node_id,
        "position": node.position,
        "map_dimensions": node.map_dimensions,
        "input_dataset_size": node.input_dataset_size,
        "level": node.level,
        "number_of_children": len(node.children),
    }

    if node.children:
        statistics["children"] = [
            {
                "node_id": encode_ghsom_node(lookup_table, child),
                "position": child.position,
            }
            for child in node.children
        ]
        statistics["total_input_dataset_size_of_children"] = sum(
            child.input_dataset_size for child in node.children
        )

    return statistics


def get_node_relative_path_by_id(
    lookup_table: Dict[Union[str, int], GHSOMNode],
    node: Union[int, str, GHSOMNode],
    print_output: bool = True
) -> Optional[list]:
    """
    Retrieve path in GHSOM hierarchy for a specific node.

    Parameters
    ----------
    lookup_table : Dict[Union[str, int], GHSOMNode]
        Lookup table containing node IDs and nodes
    node : Union[int, str, GHSOMNode]
        Node ID or node object to find path for
    print_output : bool, optional
        If True, logs the path with indentation. Default is True.

    Returns
    -------
    Optional[list]
        List of string representations from root to target node,
        or None if node not found
    """
    # Get the target node
    if isinstance(node, (int, str)):
        target_node = lookup_table.get(node)
    else:
        node_id = encode_ghsom_node(lookup_table, node)
        target_node = lookup_table.get(node_id) if node_id else None

    if not target_node:
        return None

    # Construct path from target to root
    path = []
    current_node = target_node

    while current_node is not None:
        if print_output:
            indentation = "    " * current_node.level
            node_representation = (
                f"{indentation}position {current_node.position} -- "
                f"map dimensions {current_node.map_dimensions} -- "
                f"input dataset {current_node.input_dataset_size} element(s) -- "
                f"level {current_node.level}"
            )
        else:
            node_representation = str(current_node)

        path.insert(0, node_representation)

        # Find parent by looking for node that has current as child
        parent_node = None
        for lookup_node in lookup_table.values():
            if current_node in lookup_node.children:
                parent_node = lookup_node
                break
        current_node = parent_node

    if print_output:
        path_str = "\n".join(path)
        logger.info(path_str)

    return path


def create_clusters_dict(root_node: GHSOMNode) -> dict:
    """
    Create dictionary of clusters from GHSOM tree.

    Parameters
    ----------
    root_node : GHSOMNode
        Root node of the GHSOM tree

    Returns
    -------
    dict
        Dictionary where keys are cluster identifiers and values are
        cluster attributes including position, dimensions, level, parent,
        and children IDs
    """
    clusters_dict = {}

    def traverse_tree(node: GHSOMNode, clusters_dict: dict, parent_id: Optional[str] = None):
        """Recursively traverse tree and populate clusters dictionary."""
        cluster_id = f"level_{node.level}_pos_{node.position}"

        clusters_dict[cluster_id] = {
            "position": node.position,
            "map_dimensions": node.map_dimensions,
            "input_dataset_size": node.input_dataset_size,
            "level": node.level,
            "parent_id": parent_id,
            "children_ids": [],
        }

        # Recursively add children
        for child in node.children:
            child_id = f"level_{child.level}_pos_{child.position}"
            clusters_dict[cluster_id]["children_ids"].append(child_id)
            traverse_tree(child, clusters_dict, parent_id=cluster_id)

    traverse_tree(root_node, clusters_dict)

    return clusters_dict


def get_clusters_by_level(root_node: GHSOMNode, target_level: int) -> list:
    """
    Retrieve all clusters at a specific level.

    Parameters
    ----------
    root_node : GHSOMNode
        Root node of the GHSOM tree
    target_level : int
        Level of clusters to retrieve

    Returns
    -------
    list
        List of GHSOMNode objects at the specified level
    """
    return root_node.find_by_level(target_level)


def get_ghsom_statistics(root_node: GHSOMNode) -> dict:
    """
    Report statistics from GHSOM tree.

    Parameters
    ----------
    root_node : GHSOMNode
        Root node of the GHSOM tree

    Returns
    -------
    dict
        Dictionary containing:
        - total_nodes: Total number of nodes
        - levels: Dictionary mapping level to node count
        - max_children: Maximum number of children for any node
        - max_input_dataset_size: Maximum dataset size for any node
    """
    statistics = {
        "total_nodes": 0,
        "levels": {},
        "max_children": 0,
        "max_input_dataset_size": 0,
    }

    def traverse_tree(node: GHSOMNode, statistics: dict):
        """Recursively traverse and collect statistics."""
        statistics["total_nodes"] += 1

        level = node.level
        if level not in statistics["levels"]:
            statistics["levels"][level] = 0
        statistics["levels"][level] += 1

        num_children = len(node.children)
        statistics["max_children"] = max(statistics["max_children"], num_children)

        statistics["max_input_dataset_size"] = max(
            statistics["max_input_dataset_size"], node.input_dataset_size
        )

        for child in node.children:
            traverse_tree(child, statistics)

    traverse_tree(root_node, statistics)

    return statistics
