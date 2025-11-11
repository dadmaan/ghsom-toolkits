"""Tests for heatmap visualization functions."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import matplotlib.pyplot as plt


class TestHeatmapPlotting:
    """Test heatmap plotting functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock node with map
        self.mock_node = Mock()
        self.mock_node.rows = 3
        self.mock_node.columns = 3
        self.mock_node.map = []

        for r in range(3):
            row = []
            for c in range(3):
                neuron = Mock()
                neuron.weights = np.random.rand(10)
                neuron.children = []
                row.append(neuron)
            self.mock_node.map.append(row)

    def test_plot_weight_heatmap_single_neuron(self):
        """Test weight heatmap for a single neuron."""
        from ghsom_toolkits.plotting.heatmaps import plot_weight_heatmap

        fig = plot_weight_heatmap(self.mock_node, neuron_position=(0, 0))

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_weight_heatmap_all_neurons(self):
        """Test weight heatmap for all neurons."""
        from ghsom_toolkits.plotting.heatmaps import plot_weight_heatmap

        fig = plot_weight_heatmap(self.mock_node)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_weight_heatmap_invalid_position(self):
        """Test that invalid neuron position raises ValueError."""
        from ghsom_toolkits.plotting.heatmaps import plot_weight_heatmap

        with pytest.raises(ValueError, match="Invalid neuron position"):
            plot_weight_heatmap(self.mock_node, neuron_position=(10, 10))

    def test_plot_activation_map(self):
        """Test activation map visualization."""
        from ghsom_toolkits.plotting.heatmaps import plot_activation_map

        data = np.random.rand(5, 10)
        fig = plot_activation_map(self.mock_node, data, sample_indices=0)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_activation_map_multiple_samples(self):
        """Test activation map with multiple samples."""
        from ghsom_toolkits.plotting.heatmaps import plot_activation_map

        data = np.random.rand(10, 10)
        fig = plot_activation_map(self.mock_node, data, sample_indices=[0, 1, 2])

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_activation_map_invalid_index(self):
        """Test that invalid sample index raises ValueError."""
        from ghsom_toolkits.plotting.heatmaps import plot_activation_map

        data = np.random.rand(5, 10)

        with pytest.raises(ValueError, match="out of range"):
            plot_activation_map(self.mock_node, data, sample_indices=100)

    def test_plot_umatrix(self):
        """Test U-Matrix visualization."""
        from ghsom_toolkits.plotting.heatmaps import plot_umatrix

        fig = plot_umatrix(self.mock_node)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_weight_heatmap_with_save(self, tmp_path):
        """Test saving weight heatmap to file."""
        from ghsom_toolkits.plotting.heatmaps import plot_weight_heatmap

        save_path = tmp_path / "test_heatmap.png"
        fig = plot_weight_heatmap(self.mock_node, save_path=str(save_path))

        assert save_path.exists()
        plt.close(fig)
