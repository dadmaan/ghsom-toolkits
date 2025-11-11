"""
Enhanced GHSOM Node implementation with lazy loading and tree operations.

This module provides an efficient implementation of GHSOM hierarchy nodes with
caching, lazy evaluation, and comprehensive tree traversal utilities.
"""

from typing import List, Optional, Tuple


class GHSOMNode:
    """
    Efficient GHSOM node with lazy loading, caching, and tree operations.

    Represents a node in the GHSOM hierarchy tree with optimized performance
    through lazy evaluation and property caching.

    Attributes
    ----------
    position : Tuple[int, int]
        Position of the node in its parent map (x, y coordinates)
    map_dimensions : Tuple[int, int, int]
        Dimensions of the map at this node (rows, cols, depth)
    input_dataset_size : int
        Number of input data points mapped to this node
    level : int
        Depth level in the hierarchy (0 for root)
    children : List[GHSOMNode]
        Child nodes in the hierarchy

    Examples
    --------
    >>> node = GHSOMNode(position=(0, 0), map_dimensions=(3, 3, 2),
    ...                  input_dataset_size=100, level=0)
    >>> child = GHSOMNode(position=(1, 1), map_dimensions=(2, 2, 2),
    ...                   input_dataset_size=25, level=1)
    >>> node.add_child(child)
    >>> node.rows
    3
    >>> len(node.get_leaves())
    1
    """

    def __init__(
        self,
        position: Tuple[int, int],
        map_dimensions: Tuple[int, int, int],
        input_dataset_size: int,
        level: int
    ):
        """
        Initialize a GHSOM node.

        Parameters
        ----------
        position : Tuple[int, int]
            Position of the node in its parent map
        map_dimensions : Tuple[int, int, int]
            Dimensions (rows, cols, depth) of the map
        input_dataset_size : int
            Number of input data points
        level : int
            Depth level in the hierarchy
        """
        self.position = position
        self.map_dimensions = map_dimensions
        self.input_dataset_size = input_dataset_size
        self.level = level
        self._children: List[GHSOMNode] = []

        # Cache for expensive operations
        self._descendants_cache: Optional[List[GHSOMNode]] = None
        self._leaves_cache: Optional[List[GHSOMNode]] = None

    @property
    def children(self) -> List["GHSOMNode"]:
        """Get list of child nodes."""
        return self._children

    @property
    def rows(self) -> int:
        """Get number of rows in the map (cached property)."""
        return self.map_dimensions[0]

    @property
    def columns(self) -> int:
        """Get number of columns in the map (cached property)."""
        return self.map_dimensions[1]

    @property
    def depth(self) -> int:
        """Get depth dimension of the map (cached property)."""
        return self.map_dimensions[2]

    def add_child(self, child_node: "GHSOMNode") -> None:
        """
        Add a child node to this node.

        Invalidates cached descendants and leaves when a child is added.

        Parameters
        ----------
        child_node : GHSOMNode
            The child node to add
        """
        self._children.append(child_node)
        # Invalidate caches
        self._descendants_cache = None
        self._leaves_cache = None

    def get_descendants(self) -> List["GHSOMNode"]:
        """
        Get all descendant nodes (children, grandchildren, etc.).

        Uses caching for performance on repeated calls.

        Returns
        -------
        List[GHSOMNode]
            All descendant nodes in depth-first order
        """
        if self._descendants_cache is not None:
            return self._descendants_cache

        descendants = []
        for child in self._children:
            descendants.append(child)
            descendants.extend(child.get_descendants())

        self._descendants_cache = descendants
        return descendants

    def get_leaves(self) -> List["GHSOMNode"]:
        """
        Get all leaf nodes (nodes with no children) in subtree.

        Uses caching for performance on repeated calls.

        Returns
        -------
        List[GHSOMNode]
            All leaf nodes under this node
        """
        if self._leaves_cache is not None:
            return self._leaves_cache

        if not self._children:
            # This node is a leaf
            leaves = [self]
        else:
            # Collect leaves from all children
            leaves = []
            for child in self._children:
                leaves.extend(child.get_leaves())

        self._leaves_cache = leaves
        return leaves

    def find_by_level(self, target_level: int) -> List["GHSOMNode"]:
        """
        Find all nodes at a specific level in the hierarchy.

        Parameters
        ----------
        target_level : int
            The level to search for

        Returns
        -------
        List[GHSOMNode]
            All nodes at the specified level
        """
        nodes_at_level = []

        if self.level == target_level:
            nodes_at_level.append(self)

        # Recursively search children
        for child in self._children:
            nodes_at_level.extend(child.find_by_level(target_level))

        return nodes_at_level

    def get_subtree_stats(self) -> dict:
        """
        Calculate statistics for this node's subtree.

        Returns
        -------
        dict
            Dictionary containing:
            - total_nodes: Total number of nodes in subtree
            - max_depth: Maximum depth of subtree
            - total_leaves: Number of leaf nodes
            - total_dataset_size: Sum of all input_dataset_size values
        """
        descendants = self.get_descendants()
        leaves = self.get_leaves()

        max_depth = self.level
        if descendants:
            max_depth = max(node.level for node in descendants)

        total_dataset_size = self.input_dataset_size
        for node in descendants:
            total_dataset_size += node.input_dataset_size

        return {
            "total_nodes": 1 + len(descendants),
            "max_depth": max_depth - self.level,
            "total_leaves": len(leaves),
            "total_dataset_size": total_dataset_size,
        }

    def __repr__(self) -> str:
        """String representation of the node."""
        return (
            f"position {self.position} -- map dimensions {self.map_dimensions} -- "
            f"input dataset {self.input_dataset_size} element(s) -- level {self.level}"
        )

    def __str__(self) -> str:
        """String representation of the node."""
        return self.__repr__()
