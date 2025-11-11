"""Test basic imports of ghsom-toolkits package."""

import pytest


def test_main_package_import():
    """Test that main package can be imported."""
    import ghsom_toolkits

    assert ghsom_toolkits is not None
    assert hasattr(ghsom_toolkits, "__version__")


def test_hierarchy_imports():
    """Test that hierarchy visualization functions can be imported."""
    from ghsom_toolkits import visualize_ghsom_hierarchy, visualize_node_position

    assert visualize_ghsom_hierarchy is not None
    assert visualize_node_position is not None


def test_cluster_imports():
    """Test that cluster plotting functions can be imported."""
    from ghsom_toolkits import (
        plot_ghsom_clusters,
        plot_ghsom_clusters_genre_stacked,
        plot_ghsom_clusters_pie,
    )

    assert plot_ghsom_clusters is not None
    assert plot_ghsom_clusters_pie is not None
    assert plot_ghsom_clusters_genre_stacked is not None


def test_export_utils():
    """Test that export utilities can be imported."""
    from ghsom_toolkits.export import (
        add_edge_to_graph,
        add_node_to_graph,
        get_node_id,
        get_node_label,
    )

    assert get_node_id is not None
    assert add_node_to_graph is not None
    assert add_edge_to_graph is not None
    assert get_node_label is not None


def test_color_utils():
    """Test that color utilities can be imported."""
    from ghsom_toolkits.utils import get_level_colors

    assert get_level_colors is not None
    colors = get_level_colors()
    assert isinstance(colors, dict)
    assert len(colors) > 0
    assert 0 in colors
