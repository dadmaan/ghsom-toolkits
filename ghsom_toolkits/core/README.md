## # GHSOM Core Module

The `ghsom_toolkits.core` module provides fundamental data structures and parsing utilities for working with GHSOM (Growing Hierarchical Self-Organizing Map) hierarchies.

## Overview

This module serves as the foundation for all ghsom-toolkits functionality, providing:

- **Enhanced GHSOMNode class**: Efficient node representation with lazy loading and caching
- **Parsing functions**: Convert GHSOM hierarchy strings into node trees
- **Lookup tables**: Fast node access and encoding/decoding
- **Statistics utilities**: Comprehensive hierarchy analysis

## Key Components

### GHSOMNode

An enhanced node class with optimized performance:

```python
from ghsom_toolkits.core import GHSOMNode

# Create a node
node = GHSOMNode(
    position=(0, 0),
    map_dimensions=(3, 3, 10),
    input_dataset_size=100,
    level=0
)

# Access properties
print(node.rows)      # 3
print(node.columns)   # 3
print(node.depth)     # 10

# Tree operations
children = node.children
descendants = node.get_descendants()
leaves = node.get_leaves()
level_1_nodes = node.find_by_level(1)
stats = node.get_subtree_stats()
```

### Parsing Functions

Convert GHSOM hierarchy strings into node trees:

```python
from ghsom_toolkits.core import parse_ghsom_hierarchy, create_lookup_table

hierarchy_string = """
position (0, 0) -- map dimensions (3, 3, 10) -- input dataset 100 element(s) -- level 0
position (1, 1) -- map dimensions (2, 2, 10) -- input dataset 25 element(s) -- level 1
"""

# Parse into node tree
root = parse_ghsom_hierarchy(hierarchy_string)

# Create lookup table for fast access
lookup = create_lookup_table(root, short_id=True)
```

### Utility Functions

```python
from ghsom_toolkits.core import (
    encode_ghsom_node,
    decode_ghsom_node,
    get_ghsom_statistics,
    get_ghsom_node_statistics,
    get_clusters_by_level,
    create_clusters_dict,
)

# Encode/decode nodes
node_id = encode_ghsom_node(lookup, node)
node = decode_ghsom_node(lookup, node_id)

# Get statistics
hierarchy_stats = get_ghsom_statistics(root)
node_stats = get_ghsom_node_statistics(lookup, "root")

# Get clusters at specific level
level_1_clusters = get_clusters_by_level(root, 1)

# Create clusters dictionary
clusters = create_clusters_dict(root)
```

## Performance Features

- **Lazy Loading**: Children lists built only when accessed
- **Caching**: Descendants and leaves cached after first computation
- **Efficient Traversal**: Optimized tree traversal algorithms

## API Reference

### GHSOMNode

**Attributes:**
- `position`: Tuple[int, int] - Node position in parent map
- `map_dimensions`: Tuple[int, int, int] - Map dimensions (rows, cols, depth)
- `input_dataset_size`: int - Number of input data points
- `level`: int - Hierarchy depth level
- `children`: List[GHSOMNode] - Child nodes

**Properties:**
- `rows`: int - Number of rows in map
- `columns`: int - Number of columns in map
- `depth`: int - Depth dimension of map

**Methods:**
- `add_child(child_node)`: Add child node
- `get_descendants()`: Get all descendant nodes
- `get_leaves()`: Get all leaf nodes
- `find_by_level(level)`: Find nodes at specific level
- `get_subtree_stats()`: Calculate subtree statistics

### Functions

See module docstrings for detailed parameter descriptions and return types.

## Integration

This module is used by:
- **Main Project**: Imported by `src/ghsom_manager.py` for GHSOM hierarchy management
- **Visualization**: Used by all plotting functions
- **Analysis**: Used by analysis and reporting tools
- **Adapters**: Provides target interface for compatibility adapters

## Migration from Main Project

If you were using GHSOMNode from `src.models.ghsom.ghsom_parser`, update your imports:

```python
# Old
from src.models.ghsom.ghsom_parser import GHSOMNode, parse_ghsom_hierarchy

# New
from ghsom_toolkits.core import GHSOMNode, parse_ghsom_hierarchy
```

All functionality remains the same with additional performance improvements.
