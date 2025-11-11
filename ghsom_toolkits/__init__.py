"""
GHSOM Toolkits - Visualization and Analysis Tools for GHSOM
============================================================

A companion package to ghsom-py providing visualization and analysis
capabilities for Growing Hierarchical Self-Organizing Maps.

Main Features
-------------
- Core GHSOM data structures and parsing utilities
- Hierarchical tree and treemap visualizations
- Heatmap visualizations (weight vectors, activation maps, U-Matrix)
- Cluster distribution and quality metrics plots
- Interactive Dash dashboard for model exploration
- Model comparison and analysis tools
- Comprehensive HTML report generation
- Export utilities for publication-ready figures
- Compatibility adapters for different GHSOM implementations

Quick Example
-------------
>>> from ghsom import GHSOM
>>> from ghsom_toolkits import visualize_ghsom_hierarchy
>>> from ghsom_toolkits.core import parse_ghsom_hierarchy, create_lookup_table
>>>
>>> # Train a GHSOM model
>>> ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
>>> model = ghsom.train(epochs_number=50)
>>>
>>> # Create lookup table for visualization
>>> lookup = {"root": model}
>>> visualize_ghsom_hierarchy(model, lookup, "hierarchy.png")
"""

# Core data structures and parsing
from ghsom_toolkits.core import (
    GHSOMNode,
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

# Core plotting functions
from ghsom_toolkits.plotting.clusters import (
    plot_cluster_distribution,
    plot_cluster_quality,
    plot_ghsom_clusters,
    plot_ghsom_clusters_genre_stacked,
    plot_ghsom_clusters_pie,
    plot_growth_timeline,
)
from ghsom_toolkits.plotting.hierarchy import (
    plot_hierarchy_treemap,
    visualize_ghsom_hierarchy,
    visualize_node_position,
)
from ghsom_toolkits.plotting.heatmaps import (
    plot_activation_map,
    plot_umatrix,
    plot_weight_heatmap,
)

# Analysis tools
from ghsom_toolkits.analysis import (
    compare_models,
    generate_report,
    plot_comparison,
)

__version__ = "0.1.0"

__all__ = [
    # Core data structures
    "GHSOMNode",
    "parse_ghsom_hierarchy",
    "create_lookup_table",
    "encode_ghsom_node",
    "decode_ghsom_node",
    "get_ghsom_node_statistics",
    "get_ghsom_statistics",
    "get_node_relative_path_by_id",
    "get_clusters_by_level",
    "create_clusters_dict",
    # Hierarchy visualization
    "visualize_ghsom_hierarchy",
    "visualize_node_position",
    "plot_hierarchy_treemap",
    # Heatmap visualization
    "plot_weight_heatmap",
    "plot_activation_map",
    "plot_umatrix",
    # Cluster plotting
    "plot_ghsom_clusters",
    "plot_ghsom_clusters_pie",
    "plot_ghsom_clusters_genre_stacked",
    "plot_cluster_distribution",
    "plot_cluster_quality",
    "plot_growth_timeline",
    # Analysis tools
    "compare_models",
    "plot_comparison",
    "generate_report",
]
