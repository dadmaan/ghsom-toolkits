"""
Tests for ghsom_toolkits.core.node module.
"""

import pytest
from ghsom_toolkits.core.node import GHSOMNode


class TestGHSOMNodeInit:
    """Test GHSOMNode initialization."""

    def test_init_basic(self):
        """Test basic node initialization."""
        node = GHSOMNode(
            position=(0, 0),
            map_dimensions=(3, 3, 10),
            input_dataset_size=100,
            level=0
        )
        assert node.position == (0, 0)
        assert node.map_dimensions == (3, 3, 10)
        assert node.input_dataset_size == 100
        assert node.level == 0
        assert len(node.children) == 0

    def test_rows_property(self):
        """Test rows property returns correct value."""
        node = GHSOMNode((0, 0), (5, 3, 10), 100, 0)
        assert node.rows == 5

    def test_columns_property(self):
        """Test columns property returns correct value."""
        node = GHSOMNode((0, 0), (5, 7, 10), 100, 0)
        assert node.columns == 7

    def test_depth_property(self):
        """Test depth property returns correct value."""
        node = GHSOMNode((0, 0), (5, 3, 15), 100, 0)
        assert node.depth == 15


class TestGHSOMNodeChildren:
    """Test GHSOMNode children management."""

    def test_add_child(self):
        """Test adding a child node."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child = GHSOMNode((1, 1), (2, 2, 10), 25, 1)

        root.add_child(child)

        assert len(root.children) == 1
        assert root.children[0] == child

    def test_add_multiple_children(self):
        """Test adding multiple children."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child1 = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        child2 = GHSOMNode((2, 2), (2, 2, 10), 30, 1)

        root.add_child(child1)
        root.add_child(child2)

        assert len(root.children) == 2

    def test_add_child_invalidates_cache(self):
        """Test that adding child invalidates caches."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)

        # Access descendants to populate cache
        _ = root.get_descendants()
        assert root._descendants_cache is not None

        # Add child should invalidate cache
        child = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        root.add_child(child)

        assert root._descendants_cache is None


class TestGHSOMNodeTraversal:
    """Test GHSOMNode tree traversal methods."""

    def test_get_descendants_empty(self):
        """Test get_descendants with no children."""
        node = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        descendants = node.get_descendants()
        assert len(descendants) == 0

    def test_get_descendants_one_level(self):
        """Test get_descendants with one level of children."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child1 = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        child2 = GHSOMNode((2, 2), (2, 2, 10), 30, 1)

        root.add_child(child1)
        root.add_child(child2)

        descendants = root.get_descendants()
        assert len(descendants) == 2
        assert child1 in descendants
        assert child2 in descendants

    def test_get_descendants_multiple_levels(self):
        """Test get_descendants with multiple levels."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child = GHSOMNode((1, 1), (2, 2, 10), 50, 1)
        grandchild = GHSOMNode((0, 0), (2, 2, 10), 20, 2)

        root.add_child(child)
        child.add_child(grandchild)

        descendants = root.get_descendants()
        assert len(descendants) == 2
        assert child in descendants
        assert grandchild in descendants

    def test_get_leaves_no_children(self):
        """Test get_leaves when node is a leaf."""
        node = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        leaves = node.get_leaves()

        assert len(leaves) == 1
        assert leaves[0] == node

    def test_get_leaves_with_children(self):
        """Test get_leaves with child nodes."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child1 = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        child2 = GHSOMNode((2, 2), (2, 2, 10), 30, 1)

        root.add_child(child1)
        root.add_child(child2)

        leaves = root.get_leaves()
        assert len(leaves) == 2
        assert child1 in leaves
        assert child2 in leaves
        assert root not in leaves

    def test_find_by_level(self):
        """Test find_by_level finds nodes at specific level."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child1 = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        child2 = GHSOMNode((2, 2), (2, 2, 10), 30, 1)
        grandchild = GHSOMNode((0, 0), (2, 2, 10), 10, 2)

        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)

        level_0 = root.find_by_level(0)
        assert len(level_0) == 1
        assert root in level_0

        level_1 = root.find_by_level(1)
        assert len(level_1) == 2
        assert child1 in level_1
        assert child2 in level_1

        level_2 = root.find_by_level(2)
        assert len(level_2) == 1
        assert grandchild in level_2


class TestGHSOMNodeStatistics:
    """Test GHSOMNode statistics methods."""

    def test_get_subtree_stats_single_node(self):
        """Test subtree stats for single node."""
        node = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        stats = node.get_subtree_stats()

        assert stats['total_nodes'] == 1
        assert stats['max_depth'] == 0
        assert stats['total_leaves'] == 1
        assert stats['total_dataset_size'] == 100

    def test_get_subtree_stats_with_children(self):
        """Test subtree stats with children."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child1 = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        child2 = GHSOMNode((2, 2), (2, 2, 10), 30, 1)
        grandchild = GHSOMNode((0, 0), (2, 2, 10), 10, 2)

        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)

        stats = root.get_subtree_stats()

        assert stats['total_nodes'] == 4
        assert stats['max_depth'] == 2
        assert stats['total_leaves'] == 2
        assert stats['total_dataset_size'] == 165


class TestGHSOMNodeString:
    """Test GHSOMNode string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        node = GHSOMNode((1, 2), (3, 4, 5), 100, 2)
        repr_str = repr(node)

        assert 'position (1, 2)' in repr_str
        assert 'map dimensions (3, 4, 5)' in repr_str
        assert 'input dataset 100 element(s)' in repr_str
        assert 'level 2' in repr_str

    def test_str(self):
        """Test __str__ method."""
        node = GHSOMNode((1, 2), (3, 4, 5), 100, 2)
        str_str = str(node)

        assert 'position (1, 2)' in str_str
        assert 'map dimensions (3, 4, 5)' in str_str
        assert 'input dataset 100 element(s)' in str_str
        assert 'level 2' in str_str


class TestGHSOMNodeCaching:
    """Test GHSOMNode caching behavior."""

    def test_descendants_cache_works(self):
        """Test that descendants are cached after first call."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        root.add_child(child)

        # First call populates cache
        descendants1 = root.get_descendants()
        assert root._descendants_cache is not None

        # Second call uses cache
        descendants2 = root.get_descendants()
        assert descendants1 is descendants2

    def test_leaves_cache_works(self):
        """Test that leaves are cached after first call."""
        root = GHSOMNode((0, 0), (3, 3, 10), 100, 0)
        child = GHSOMNode((1, 1), (2, 2, 10), 25, 1)
        root.add_child(child)

        # First call populates cache
        leaves1 = root.get_leaves()
        assert root._leaves_cache is not None

        # Second call uses cache
        leaves2 = root.get_leaves()
        assert leaves1 is leaves2
