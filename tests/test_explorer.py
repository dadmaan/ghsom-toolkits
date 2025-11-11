"""Tests for interactive explorer utilities."""

import pytest
import numpy as np
from unittest.mock import Mock


class TestExploreNeuron:
    """Test neuron exploration function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_node = Mock()
        self.mock_node.rows = 2
        self.mock_node.columns = 2
        self.mock_node.level = 0
        self.mock_node.input_dataset_size = 100
        self.mock_node.map = []

        for r in range(2):
            row = []
            for c in range(2):
                neuron = Mock()
                neuron.weights = np.random.rand(5)
                neuron.children = []
                row.append(neuron)
            self.mock_node.map.append(row)

    def test_explore_neuron_basic(self):
        """Test basic neuron exploration."""
        from ghsom_toolkits.interactive.explorer import explore_neuron

        info = explore_neuron(self.mock_node, (0, 0))

        assert info['position'] == (0, 0)
        assert 'weights' in info
        assert info['has_child'] is False
        assert info['level'] == 0

    def test_explore_neuron_invalid_position(self):
        """Test that invalid position raises ValueError."""
        from ghsom_toolkits.interactive.explorer import explore_neuron

        with pytest.raises(ValueError, match="Invalid neuron position"):
            explore_neuron(self.mock_node, (10, 10))

    def test_explore_neuron_with_child(self):
        """Test exploring neuron with child map."""
        from ghsom_toolkits.interactive.explorer import explore_neuron

        # Add child to neuron
        child_node = Mock()
        child_node.rows = 2
        child_node.columns = 2
        child_node.input_dataset_size = 25

        self.mock_node.map[0][0].children = [child_node]

        info = explore_neuron(self.mock_node, (0, 0))

        assert info['has_child'] is True
        assert info['num_children'] == 1
        assert info['child_node'] is not None


class TestGetSubtree:
    """Test subtree extraction function."""

    def setup_method(self):
        """Set up test hierarchy."""
        self.root = Mock()
        self.root.rows = 2
        self.root.columns = 2
        self.root.children = []

        self.child1 = Mock()
        self.child1.children = []

        self.child2 = Mock()
        self.child2.children = []

        self.root.children = [self.child1, self.child2]

        self.lookup = {
            "root": self.root,
            "root_c0": self.child1,
            "root_c1": self.child2,
        }

    def test_get_subtree_basic(self):
        """Test basic subtree extraction."""
        from ghsom_toolkits.interactive.explorer import get_subtree

        subtree = get_subtree(self.root, self.lookup, "root")

        assert subtree['node_id'] == "root"
        assert subtree['num_nodes'] == 3  # root + 2 children
        assert len(subtree['subtree_lookup']) == 3

    def test_get_subtree_invalid_id(self):
        """Test that invalid node ID raises ValueError."""
        from ghsom_toolkits.interactive.explorer import get_subtree

        with pytest.raises(ValueError, match="not found"):
            get_subtree(self.root, self.lookup, "nonexistent")

    def test_get_subtree_with_max_depth(self):
        """Test subtree extraction with depth limit."""
        from ghsom_toolkits.interactive.explorer import get_subtree

        subtree = get_subtree(self.root, self.lookup, "root", max_depth=0)

        assert subtree['num_nodes'] == 1  # Only root


class TestTraceSamplePath:
    """Test sample path tracing function."""

    def setup_method(self):
        """Set up test hierarchy."""
        self.root = Mock()
        self.root.rows = 2
        self.root.columns = 2
        self.root.map = []

        for r in range(2):
            row = []
            for c in range(2):
                neuron = Mock()
                neuron.weights = np.array([r, c, 0, 0, 0])
                neuron.children = []
                row.append(neuron)
            self.root.map.append(row)

    def test_trace_sample_path_basic(self):
        """Test basic sample path tracing."""
        from ghsom_toolkits.interactive.explorer import trace_sample_path

        sample = np.array([0, 0, 0, 0, 0])
        path = trace_sample_path(self.root, sample)

        assert len(path) == 1  # Only root level (no children)
        assert path[0]['level'] == 0
        assert 'bmu_position' in path[0]
        assert 'bmu_distance' in path[0]

    def test_trace_sample_path_invalid_shape(self):
        """Test that 2D sample raises ValueError."""
        from ghsom_toolkits.interactive.explorer import trace_sample_path

        sample = np.array([[0, 0, 0, 0, 0]])  # 2D

        with pytest.raises(ValueError, match="must be 1D"):
            trace_sample_path(self.root, sample)

    def test_trace_sample_path_with_max_depth(self):
        """Test path tracing with depth limit."""
        from ghsom_toolkits.interactive.explorer import trace_sample_path

        # Add child to root
        child = Mock()
        child.rows = 2
        child.columns = 2
        child.map = [[Mock() for _ in range(2)] for _ in range(2)]

        for r in range(2):
            for c in range(2):
                child.map[r][c].weights = np.random.rand(5)
                child.map[r][c].children = []

        self.root.map[0][0].children = [child]

        sample = np.array([0, 0, 0, 0, 0])
        path = trace_sample_path(self.root, sample, max_depth=0)

        assert len(path) == 1  # Limited to depth 0


class TestNodeStatistics:
    """Test node statistics function."""

    def setup_method(self):
        """Set up test node."""
        self.node = Mock()
        self.node.rows = 3
        self.node.columns = 3
        self.node.input_dataset_size = 100
        self.node.level = 1
        self.node.map = []

        # Create map with some expanded neurons
        for r in range(3):
            row = []
            for c in range(3):
                neuron = Mock()
                # Make some neurons have children
                if r == 0 and c == 0:
                    neuron.children = [Mock()]
                else:
                    neuron.children = []
                row.append(neuron)
            self.node.map.append(row)

    def test_get_node_statistics(self):
        """Test node statistics computation."""
        from ghsom_toolkits.interactive.explorer import get_node_statistics

        stats = get_node_statistics(self.node)

        assert stats['map_shape'] == (3, 3)
        assert stats['num_neurons'] == 9
        assert stats['num_expanded_neurons'] == 1
        assert stats['num_leaf_neurons'] == 8
        assert stats['dataset_size'] == 100
        assert stats['level'] == 1
        assert 'expansion_rate' in stats


class TestFindSimilarNeurons:
    """Test similar neuron finding function."""

    def setup_method(self):
        """Set up test node."""
        self.node = Mock()
        self.node.rows = 3
        self.node.columns = 3
        self.node.map = []

        # Create map with known weights
        for r in range(3):
            row = []
            for c in range(3):
                neuron = Mock()
                neuron.weights = np.array([r, c, 0, 0, 0], dtype=float)
                row.append(neuron)
            self.node.map.append(row)

    def test_find_similar_neurons(self):
        """Test finding similar neurons."""
        from ghsom_toolkits.interactive.explorer import find_similar_neurons

        similar = find_similar_neurons(self.node, (0, 0), top_k=3)

        assert len(similar) == 3
        assert all(isinstance(item, tuple) for item in similar)
        assert all(len(item) == 2 for item in similar)  # (position, distance)

    def test_find_similar_neurons_invalid_position(self):
        """Test that invalid reference position raises ValueError."""
        from ghsom_toolkits.interactive.explorer import find_similar_neurons

        with pytest.raises(ValueError, match="Invalid reference position"):
            find_similar_neurons(self.node, (10, 10))
