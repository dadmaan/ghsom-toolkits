"""Tests for extended cluster plotting functions."""

import pytest
import numpy as np
from unittest.mock import Mock
import matplotlib.pyplot as plt


class TestClusterDistribution:
    """Test cluster distribution plotting."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock hierarchy
        self.root_node = Mock()
        self.root_node.rows = 2
        self.root_node.columns = 2
        self.root_node.input_dataset_size = 100
        self.root_node.map = [[Mock() for _ in range(2)] for _ in range(2)]

        # Create leaf nodes (clusters)
        self.leaf1 = Mock()
        self.leaf1.input_dataset_size = 30
        self.leaf1.children = []

        self.leaf2 = Mock()
        self.leaf2.input_dataset_size = 40
        self.leaf2.children = []

        self.leaf3 = Mock()
        self.leaf3.input_dataset_size = 30
        self.leaf3.children = []

        # Root has children
        self.root_node.children = [self.leaf1, self.leaf2, self.leaf3]

    def test_plot_cluster_distribution(self):
        """Test cluster distribution plot."""
        from ghsom_toolkits.plotting.clusters import plot_cluster_distribution

        fig = plot_cluster_distribution(self.root_node)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cluster_distribution_with_save(self, tmp_path):
        """Test saving cluster distribution plot."""
        from ghsom_toolkits.plotting.clusters import plot_cluster_distribution

        save_path = tmp_path / "cluster_dist.png"
        fig = plot_cluster_distribution(self.root_node, save_path=str(save_path))

        assert save_path.exists()
        plt.close(fig)


class TestClusterQuality:
    """Test cluster quality plotting."""

    def test_plot_cluster_quality_with_sklearn(self):
        """Test cluster quality plot when sklearn is available."""
        pytest.importorskip("sklearn")

        from ghsom_toolkits.plotting.clusters import plot_cluster_quality

        # Create simple clustered data
        np.random.seed(42)
        data = np.vstack([
            np.random.randn(30, 5) + [0, 0, 0, 0, 0],
            np.random.randn(30, 5) + [5, 5, 5, 5, 5],
            np.random.randn(30, 5) + [-5, -5, -5, -5, -5],
        ])
        labels = np.array([0] * 30 + [1] * 30 + [2] * 30)

        fig = plot_cluster_quality(data, labels)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cluster_quality_without_sklearn(self, monkeypatch):
        """Test that missing sklearn raises ImportError."""
        # Mock sklearn availability
        import ghsom_toolkits.plotting.clusters as clusters_module
        monkeypatch.setattr(clusters_module, "SKLEARN_AVAILABLE", False)

        from ghsom_toolkits.plotting.clusters import plot_cluster_quality

        data = np.random.rand(50, 5)
        labels = np.random.randint(0, 3, 50)

        with pytest.raises(ImportError, match="scikit-learn is required"):
            plot_cluster_quality(data, labels)


class TestGrowthTimeline:
    """Test growth timeline plotting."""

    def test_plot_growth_timeline(self):
        """Test growth timeline plot."""
        from ghsom_toolkits.plotting.clusters import plot_growth_timeline

        history = [
            {'epoch': 0, 'num_nodes': 4, 'depth': 1, 'qe': 0.5, 'te': 0.1},
            {'epoch': 10, 'num_nodes': 8, 'depth': 2, 'qe': 0.4, 'te': 0.08},
            {'epoch': 20, 'num_nodes': 12, 'depth': 2, 'qe': 0.3, 'te': 0.06},
            {'epoch': 30, 'num_nodes': 15, 'depth': 3, 'qe': 0.25, 'te': 0.05},
        ]

        fig = plot_growth_timeline(history)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_growth_timeline_empty(self):
        """Test that empty history raises ValueError."""
        from ghsom_toolkits.plotting.clusters import plot_growth_timeline

        with pytest.raises(ValueError, match="empty"):
            plot_growth_timeline([])

    def test_plot_growth_timeline_without_te(self):
        """Test growth timeline without topological error."""
        from ghsom_toolkits.plotting.clusters import plot_growth_timeline

        history = [
            {'epoch': 0, 'num_nodes': 4, 'depth': 1, 'qe': 0.5},
            {'epoch': 10, 'num_nodes': 8, 'depth': 2, 'qe': 0.4},
        ]

        fig = plot_growth_timeline(history)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
