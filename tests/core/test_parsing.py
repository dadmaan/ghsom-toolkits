"""
Tests for ghsom_toolkits.core.parsing module.
"""

import pytest
from ghsom_toolkits.core.parsing import (
    parse_ghsom_hierarchy,
    create_lookup_table,
    encode_ghsom_node,
    decode_ghsom_node,
    get_ghsom_node_statistics,
    get_ghsom_statistics,
    create_clusters_dict,
    get_clusters_by_level,
)
from ghsom_toolkits.core.node import GHSOMNode


SAMPLE_HIERARCHY = """position (0, 0) -- map dimensions (3, 3, 10) -- input dataset 100 element(s) -- level 0
position (1, 1) -- map dimensions (2, 2, 10) -- input dataset 25 element(s) -- level 1
position (2, 2) -- map dimensions (2, 2, 10) -- input dataset 30 element(s) -- level 1"""


class TestParseGHSOMHierarchy:
    """Test parse_ghsom_hierarchy function."""

    def test_parse_single_node(self):
        """Test parsing hierarchy with single node."""
        hierarchy_str = "position (0, 0) -- map dimensions (3, 3, 10) -- input dataset 100 element(s) -- level 0"
        root = parse_ghsom_hierarchy(hierarchy_str)

        assert root.position == (0, 0)
        assert root.map_dimensions == (3, 3, 10)
        assert root.input_dataset_size == 100
        assert root.level == 0
        assert len(root.children) == 0

    def test_parse_with_children(self):
        """Test parsing hierarchy with children."""
        root = parse_ghsom_hierarchy(SAMPLE_HIERARCHY)

        assert root.level == 0
        assert len(root.children) == 2
        assert all(child.level == 1 for child in root.children)

    def test_parse_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Empty hierarchy string"):
            parse_ghsom_hierarchy("")

    def test_parse_negative_coordinates(self):
        """Test parsing with negative coordinates."""
        hierarchy_str = "position (-1, -2) -- map dimensions (3, 3, 10) -- input dataset 100 element(s) -- level 0"
        root = parse_ghsom_hierarchy(hierarchy_str)

        assert root.position == (-1, -2)


class TestCreateLookupTable:
    """Test create_lookup_table function."""

    def test_lookup_table_single_node(self):
        """Test lookup table with single node."""
        node = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        lookup = create_lookup_table(node)

        assert "root" in lookup
        assert lookup["root"] == node

    def test_lookup_table_with_children(self):
        """Test lookup table with children."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        root.add_child(child)

        lookup = create_lookup_table(root, short_id=True)

        assert "root" in lookup
        assert 1 in lookup
        assert len(lookup) == 2

    def test_lookup_table_short_id(self):
        """Test lookup table with short_id=True."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        root.add_child(child)

        lookup = create_lookup_table(root, short_id=True)

        assert 1 in lookup
        assert isinstance(list(lookup.keys())[1], int)

    def test_lookup_table_long_id(self):
        """Test lookup table with short_id=False."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        root.add_child(child)

        lookup = create_lookup_table(root, short_id=False)

        # First non-root key should be an integer with encoded information
        keys = [k for k in lookup.keys() if k != "root"]
        assert len(keys) > 0
        assert isinstance(keys[0], int)


class TestEncodeDecodeNode:
    """Test encode_ghsom_node and decode_ghsom_node functions."""

    def test_encode_node(self):
        """Test encoding a node."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        lookup = create_lookup_table(root)

        node_id = encode_ghsom_node(lookup, root)
        assert node_id == "root"

    def test_encode_node_string_input(self):
        """Test encoding with string input."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        lookup = create_lookup_table(root)

        node_str = str(root)
        node_id = encode_ghsom_node(lookup, node_str)
        assert node_id == "root"

    def test_decode_node(self):
        """Test decoding a node."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        lookup = create_lookup_table(root)

        node = decode_ghsom_node(lookup, "root")
        assert node == root

    def test_decode_nonexistent_node(self):
        """Test decoding nonexistent node returns None."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        lookup = create_lookup_table(root)

        node = decode_ghsom_node(lookup, "nonexistent")
        assert node is None


class TestGetGHSOMNodeStatistics:
    """Test get_ghsom_node_statistics function."""

    def test_node_statistics_root(self):
        """Test statistics for root node."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        lookup = create_lookup_table(root)

        stats = get_ghsom_node_statistics(lookup, "root")

        assert stats["node_id"] == "root"
        assert stats["position"] == (0, 0)
        assert stats["level"] == 0
        assert stats["number_of_children"] == 0

    def test_node_statistics_with_children(self):
        """Test statistics for node with children."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        root.add_child(child)

        lookup = create_lookup_table(root)
        stats = get_ghsom_node_statistics(lookup, "root")

        assert stats["number_of_children"] == 1
        assert "children" in stats
        assert "total_input_dataset_size_of_children" in stats

    def test_node_statistics_nonexistent(self):
        """Test statistics for nonexistent node."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        lookup = create_lookup_table(root)

        stats = get_ghsom_node_statistics(lookup, "nonexistent")
        assert stats is None


class TestGetGHSOMStatistics:
    """Test get_ghsom_statistics function."""

    def test_statistics_single_node(self):
        """Test statistics for single node."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        stats = get_ghsom_statistics(root)

        assert stats["total_nodes"] == 1
        assert stats["levels"] == {0: 1}
        assert stats["max_children"] == 0
        assert stats["max_input_dataset_size"] == 100

    def test_statistics_with_hierarchy(self):
        """Test statistics for hierarchy."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child1 = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        child2 = GHSOMNode((2, 2), (2, 2, 10), 30, 1)
        root.add_child(child1)
        root.add_child(child2)

        stats = get_ghsom_statistics(root)

        assert stats["total_nodes"] == 3
        assert stats["levels"] == {0: 1, 1: 2}
        assert stats["max_children"] == 2
        assert stats["max_input_dataset_size"] == 100


class TestCreateClustersDict:
    """Test create_clusters_dict function."""

    def test_clusters_dict_single_node(self):
        """Test clusters dict for single node."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        clusters = create_clusters_dict(root)

        assert len(clusters) == 1
        cluster_id = "level_0_pos_(0, 0)"
        assert cluster_id in clusters

    def test_clusters_dict_with_children(self):
        """Test clusters dict with children."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        root.add_child(child)

        clusters = create_clusters_dict(root)

        assert len(clusters) == 2
        root_id = "level_0_pos_(0, 0)"
        child_id = "level_1_pos_(1, 1)"
        assert child_id in clusters[root_id]["children_ids"]


class TestGetClustersByLevel:
    """Test get_clusters_by_level function."""

    def test_get_clusters_by_level(self):
        """Test getting clusters at specific level."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child1 = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        child2 = GHSOMNode((2, 2), (2, 2, 10), 30, 1)
        root.add_child(child1)
        root.add_child(child2)

        level_0 = get_clusters_by_level(root, 0)
        assert len(level_0) == 1
        assert root in level_0

        level_1 = get_clusters_by_level(root, 1)
        assert len(level_1) == 2

    def test_get_clusters_nonexistent_level(self):
        """Test getting clusters at nonexistent level."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)

        level_5 = get_clusters_by_level(root, 5)
        assert len(level_5) == 0
