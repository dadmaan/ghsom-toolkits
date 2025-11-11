"""
Core GHSOM data structures and parsing utilities.

This module provides the fundamental data structures and parsing utilities
for working with GHSOM hierarchies. It includes optimized node representations
and comprehensive parsing functions.

Key Components
--------------
- GHSOMNode: Enhanced node class with lazy loading and tree operations
- Parsing functions: Efficient hierarchy parsing and lookup tables
- Statistics utilities: Comprehensive tree analysis functions

Examples
--------
>>> from ghsom_toolkits.core import GHSOMNode, parse_ghsom_hierarchy, create_lookup_table
>>>
>>> # Parse a hierarchy string
>>> hierarchy = '''position (0, 0) -- map dimensions (3, 3, 2) -- input dataset 100 element(s) -- level 0
... position (1, 1) -- map dimensions (2, 2, 2) -- input dataset 25 element(s) -- level 1'''
>>> root = parse_ghsom_hierarchy(hierarchy)
>>>
>>> # Create lookup table
>>> lookup = create_lookup_table(root, short_id=True)
>>>
>>> # Access node properties
>>> root.rows
3
>>> len(root.children)
1
"""

from ghsom_toolkits.core.node import GHSOMNode
from ghsom_toolkits.core.parsing import (
    create_clusters_dict,
    create_lookup_table,
    decode_ghsom_node,
    encode_ghsom_node,
    get_clusters_by_level,
    get_ghsom_node_statistics,
    get_ghsom_statistics,
    get_node_relative_path_by_id,
    parse_ghsom_hierarchy,
)

__all__ = [
    # Core node class
    "GHSOMNode",
    # Parsing functions
    "parse_ghsom_hierarchy",
    "create_lookup_table",
    "encode_ghsom_node",
    "decode_ghsom_node",
    # Statistics and analysis
    "get_ghsom_node_statistics",
    "get_ghsom_statistics",
    "get_node_relative_path_by_id",
    "get_clusters_by_level",
    "create_clusters_dict",
]
