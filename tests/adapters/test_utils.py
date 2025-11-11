"""
Tests for ghsom_toolkits.adapters.utils module.
"""

import pytest
from unittest.mock import Mock
from ghsom_toolkits.adapters.utils import (
    build_lookup_table,
    count_nodes,
    get_max_level,
    print_hierarchy,
)
from ghsom_toolkits.core.node import GHSOMNode


class TestBuildLookupTable:
    """Test build_lookup_table function."""

    def test_single_node(self):
        """Test lookup table for single node."""
        node = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        lookup = build_lookup_table(node)

        assert "root" in lookup
        assert lookup["root"] == node

    def test_with_children(self):
        """Test lookup table with children."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child1 = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        child2 = GHSOMNode((2, 2), (2, 2, 10), 30, 1)
        root.add_child(child1)
        root.add_child(child2)

        lookup = build_lookup_table(root)

        assert len(lookup) == 3
        assert "root" in lookup
        assert "root_n1_1" in lookup
        assert "root_n2_2" in lookup

    def test_custom_prefix(self):
        """Test lookup table with custom prefix."""
        node = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        lookup = build_lookup_table(node, prefix="custom")

        assert "custom" in lookup


class TestCountNodes:
    """Test count_nodes function."""

    def test_single_node(self):
        """Test counting single node."""
        node = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        count = count_nodes(node)

        assert count == 1

    def test_with_children(self):
        """Test counting nodes with children."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child1 = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        child2 = GHSOMNode((2, 2), (2, 2, 10), 30, 1)
        root.add_child(child1)
        root.add_child(child2)

        count = count_nodes(root)

        assert count == 3


class TestGetMaxLevel:
    """Test get_max_level function."""

    def test_single_node(self):
        """Test max level for single node."""
        node = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        max_level = get_max_level(node)

        assert max_level == 0

    def test_with_hierarchy(self):
        """Test max level with hierarchy."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        grandchild = GHSOMNode((0, 0), (2, 2, 10), 10, 2)
        root.add_child(child)
        child.add_child(grandchild)

        max_level = get_max_level(root)

        assert max_level == 2


class TestPrintHierarchy:
    """Test print_hierarchy function."""

    def test_print_single_node(self, capsys):
        """Test printing single node."""
        node = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        print_hierarchy(node)

        captured = capsys.readouterr()
        assert 'position (0, 0)' in captured.out

    def test_print_with_children(self, capsys):
        """Test printing hierarchy with children."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        root.add_child(child)

        print_hierarchy(root)

        captured = capsys.readouterr()
        # Child should be indented
        assert '  position (1, 1)' in captured.out
