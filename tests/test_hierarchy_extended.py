"""Tests for extended hierarchy plotting functions."""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestHierarchyTreemap:
    """Test treemap visualization."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock hierarchy
        self.root_node = Mock()
        self.root_node.rows = 2
        self.root_node.columns = 2
        self.root_node.input_dataset_size = 100
        self.root_node.level = 0
        self.root_node.children = []

        self.lookup_table = {"root": self.root_node}

    def test_plot_hierarchy_treemap_with_plotly(self):
        """Test treemap creation when plotly is available."""
        pytest.importorskip("plotly")

        from ghsom_toolkits.plotting.hierarchy import plot_hierarchy_treemap

        fig = plot_hierarchy_treemap(self.root_node, self.lookup_table)

        assert fig is not None

    def test_plot_hierarchy_treemap_without_plotly(self, monkeypatch):
        """Test that missing plotly raises ImportError."""
        import ghsom_toolkits.plotting.hierarchy as hierarchy_module
        monkeypatch.setattr(hierarchy_module, "PLOTLY_AVAILABLE", False)

        from ghsom_toolkits.plotting.hierarchy import plot_hierarchy_treemap

        with pytest.raises(ImportError, match="plotly is required"):
            plot_hierarchy_treemap(self.root_node, self.lookup_table)

    @pytest.mark.skipif(
        not pytest.importorskip("plotly", reason="plotly not installed"),
        reason="plotly not available"
    )
    def test_plot_hierarchy_treemap_with_save(self, tmp_path):
        """Test saving treemap to HTML file."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not available")

        from ghsom_toolkits.plotting.hierarchy import plot_hierarchy_treemap

        save_path = tmp_path / "treemap.html"
        fig = plot_hierarchy_treemap(
            self.root_node,
            self.lookup_table,
            filename=str(save_path)
        )

        assert save_path.exists()

    def test_plot_hierarchy_treemap_with_children(self):
        """Test treemap with hierarchy containing children."""
        try:
            import plotly
        except ImportError:
            pytest.skip("plotly not available")

        from ghsom_toolkits.plotting.hierarchy import plot_hierarchy_treemap

        # Add children to root
        child1 = Mock()
        child1.rows = 2
        child1.columns = 2
        child1.input_dataset_size = 50
        child1.children = []

        child2 = Mock()
        child2.rows = 2
        child2.columns = 2
        child2.input_dataset_size = 50
        child2.children = []

        self.root_node.children = [child1, child2]
        self.lookup_table["root_c0"] = child1
        self.lookup_table["root_c1"] = child2

        fig = plot_hierarchy_treemap(self.root_node, self.lookup_table)

        assert fig is not None
