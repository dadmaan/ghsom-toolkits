"""Basic tests for plotting functions (no actual plotting)."""

import pytest
from unittest.mock import Mock, MagicMock, patch


class TestHierarchyPlotting:
    """Test hierarchy plotting functions."""

    def test_visualize_ghsom_hierarchy_basic(self):
        """Test basic hierarchy visualization with mocked graph."""
        from ghsom_toolkits.plotting.hierarchy import visualize_ghsom_hierarchy

        # Create mock node
        mock_node = Mock()
        mock_node.level = 0
        mock_node.position = (0, 0)
        mock_node.input_dataset_size = 100
        mock_node.children = []

        lookup_table = {"root": mock_node}

        # Mock pydot to avoid actual graph creation
        with patch("ghsom_toolkits.plotting.hierarchy.pydot") as mock_pydot:
            mock_graph = MagicMock()
            mock_pydot.Dot.return_value = mock_graph

            visualize_ghsom_hierarchy(
                node=mock_node, lookup_table=lookup_table, filename="test.png"
            )

            # Verify graph was created and saved
            mock_pydot.Dot.assert_called_once()
            mock_graph.write_png.assert_called_once_with("test.png")

    def test_visualize_node_position_basic(self):
        """Test node position visualization with mocked graph."""
        from ghsom_toolkits.plotting.hierarchy import visualize_node_position

        # Create mock nodes
        root_node = Mock()
        root_node.position = (0, 0)
        root_node.input_dataset_size = 100
        root_node.children = []

        target_node = Mock()
        target_node.position = (1, 1)
        target_node.input_dataset_size = 50
        target_node.children = []

        lookup_table = {"root": root_node, "target": target_node}

        # Mock pydot to avoid actual graph creation
        with patch("ghsom_toolkits.plotting.hierarchy.pydot") as mock_pydot:
            mock_graph = MagicMock()
            mock_pydot.Dot.return_value = mock_graph

            visualize_node_position(
                root_node=root_node,
                lookup_table=lookup_table,
                node_id="target",
                filename="test_position.png",
            )

            # Verify graph was created and saved
            mock_pydot.Dot.assert_called_once()
            mock_graph.write_png.assert_called_once_with("test_position.png")

    def test_visualize_node_position_invalid_id(self):
        """Test that invalid node_id raises ValueError."""
        from ghsom_toolkits.plotting.hierarchy import visualize_node_position

        root_node = Mock()
        lookup_table = {"root": root_node}

        with pytest.raises(ValueError, match="not found in the lookup table"):
            visualize_node_position(
                root_node=root_node,
                lookup_table=lookup_table,
                node_id="nonexistent",
            )


class TestGraphvizUtils:
    """Test graphviz utility functions."""

    def test_get_node_id(self):
        """Test node ID retrieval from lookup table."""
        from ghsom_toolkits.export.graphviz import get_node_id

        mock_node = Mock()
        lookup_table = {"node_1": mock_node}

        result = get_node_id(lookup_table, mock_node)
        assert result == "node_1"

    def test_get_node_id_not_found(self):
        """Test that get_node_id raises StopIteration when node not found."""
        from ghsom_toolkits.export.graphviz import get_node_id

        mock_node = Mock()
        lookup_table = {}

        with pytest.raises(StopIteration):
            get_node_id(lookup_table, mock_node)

    def test_get_node_label(self):
        """Test node label generation."""
        from ghsom_toolkits.export.graphviz import get_node_label

        mock_node = Mock()
        mock_node.position = (2, 3)
        mock_node.input_dataset_size = 42
        mock_node.children = [Mock(), Mock()]

        label = get_node_label(mock_node, "test_id")

        assert "test_id" in label
        assert "(2, 3)" in label
        assert "42" in label
        assert "2" in label  # Number of children

    def test_add_node_to_graph(self):
        """Test adding node to pydot graph."""
        from ghsom_toolkits.export.graphviz import add_node_to_graph

        mock_graph = Mock()

        add_node_to_graph(
            graph=mock_graph,
            node_id="test_node",
            label="Test Label",
            color="#FF0000",
            node_size=0.5,
        )

        # Verify add_node was called
        mock_graph.add_node.assert_called_once()

    def test_add_edge_to_graph(self):
        """Test adding edge to pydot graph."""
        from ghsom_toolkits.export.graphviz import add_edge_to_graph

        mock_graph = Mock()

        add_edge_to_graph(graph=mock_graph, parent_id="parent", child_id="child")

        # Verify add_edge was called
        mock_graph.add_edge.assert_called_once()


class TestColorUtils:
    """Test color utility functions."""

    def test_get_level_colors(self):
        """Test level color scheme retrieval."""
        from ghsom_toolkits.utils.colors import get_level_colors

        colors = get_level_colors()

        assert isinstance(colors, dict)
        assert len(colors) == 10  # 10 levels defined
        assert all(isinstance(k, int) for k in colors.keys())
        assert all(isinstance(v, str) for v in colors.values())
        assert all(v.startswith("#") for v in colors.values())
