"""
Tests for ghsom_toolkits.adapters.ghsom_py_adapter module.
"""

import pytest
from unittest.mock import Mock, MagicMock
from ghsom_toolkits.adapters.ghsom_py_adapter import GHSOMNodeAdapter, adapt_model
from ghsom_toolkits.core.node import GHSOMNode


class MockGSOM:
    """Mock GSOM object for testing."""

    def __init__(self, rows=3, cols=3, neurons_with_children=None):
        self.neurons = {}
        for i in range(rows):
            for j in range(cols):
                neuron = Mock()
                neuron.child_map = None
                neuron.input_dataset = list(range(10))  # Mock dataset
                neuron.weight_vector = Mock(return_value=[0.1] * 10)
                self.neurons[(i, j)] = neuron

        # Add child maps if specified
        if neurons_with_children:
            for pos in neurons_with_children:
                if pos in self.neurons:
                    child_gsom = MockGSOM(2, 2)
                    self.neurons[pos].child_map = child_gsom

    def map_shape(self):
        """Return map shape."""
        if not self.neurons:
            return (0, 0)
        max_row = max(pos[0] for pos in self.neurons.keys())
        max_col = max(pos[1] for pos in self.neurons.keys())
        return (max_row + 1, max_col + 1)


class MockNeuron:
    """Mock Neuron object for testing."""

    def __init__(self, child_map=None):
        self.child_map = child_map


class TestGHSOMNodeAdapterInit:
    """Test GHSOMNodeAdapter initialization."""

    def test_init_with_gsom(self):
        """Test initialization with GSOM object."""
        gsom = MockGSOM(3, 3)
        adapter = GHSOMNodeAdapter(gsom, level=0, position=(0, 0))

        assert adapter.level == 0
        assert adapter.position == (0, 0)
        assert adapter._wrapped == gsom

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        gsom = MockGSOM(3, 3)
        adapter = GHSOMNodeAdapter(gsom)

        assert adapter.level == 0
        assert adapter.position == (0, 0)


class TestGHSOMNodeAdapterProperties:
    """Test GHSOMNodeAdapter property access."""

    def test_level_property(self):
        """Test level property."""
        gsom = MockGSOM(3, 3)
        adapter = GHSOMNodeAdapter(gsom, level=2)

        assert adapter.level == 2

    def test_position_property(self):
        """Test position property."""
        gsom = MockGSOM(3, 3)
        adapter = GHSOMNodeAdapter(gsom, position=(1, 2))

        assert adapter.position == (1, 2)

    def test_rows_property(self):
        """Test rows property."""
        gsom = MockGSOM(5, 3)
        adapter = GHSOMNodeAdapter(gsom)

        assert adapter.rows == 5

    def test_columns_property(self):
        """Test columns property."""
        gsom = MockGSOM(5, 7)
        adapter = GHSOMNodeAdapter(gsom)

        assert adapter.columns == 7

    def test_rows_fallback(self):
        """Test rows property fallback when map_shape fails."""
        gsom = Mock()
        gsom.map_shape = Mock(side_effect=AttributeError)
        gsom.neurons = {(0, 0): Mock(), (2, 1): Mock()}
        adapter = GHSOMNodeAdapter(gsom)

        assert adapter.rows == 3  # max row + 1

    def test_columns_fallback(self):
        """Test columns property fallback when map_shape fails."""
        gsom = Mock()
        gsom.map_shape = Mock(side_effect=AttributeError)
        gsom.neurons = {(0, 0): Mock(), (1, 3): Mock()}
        adapter = GHSOMNodeAdapter(gsom)

        assert adapter.columns == 4  # max col + 1

    def test_input_dataset_size(self):
        """Test input_dataset_size property."""
        gsom = MockGSOM(3, 3)
        adapter = GHSOMNodeAdapter(gsom)

        # 3x3 = 9 neurons, each with dataset of 10
        assert adapter.input_dataset_size == 90

    def test_map_dimensions(self):
        """Test map_dimensions property."""
        gsom = MockGSOM(4, 5)
        adapter = GHSOMNodeAdapter(gsom)

        dims = adapter.map_dimensions
        assert dims[0] == 4  # rows
        assert dims[1] == 5  # columns
        assert dims[2] == 10  # weight vector length


class TestGHSOMNodeAdapterChildren:
    """Test GHSOMNodeAdapter children handling."""

    def test_children_empty(self):
        """Test children property with no children."""
        gsom = MockGSOM(3, 3)
        adapter = GHSOMNodeAdapter(gsom)

        children = adapter.children
        assert len(children) == 0

    def test_children_with_child_maps(self):
        """Test children property with child maps."""
        gsom = MockGSOM(3, 3, neurons_with_children=[(1, 1), (2, 2)])
        adapter = GHSOMNodeAdapter(gsom)

        children = adapter.children
        assert len(children) == 2
        assert all(isinstance(child, GHSOMNodeAdapter) for child in children)

    def test_children_lazy_loading(self):
        """Test that children are lazy loaded."""
        gsom = MockGSOM(3, 3, neurons_with_children=[(1, 1)])
        adapter = GHSOMNodeAdapter(gsom)

        assert adapter._children_cache is None

        # Access children triggers loading
        children = adapter.children
        assert adapter._children_cache is not None
        assert len(children) == 1

        # Second access uses cache
        children2 = adapter.children
        assert children is children2

    def test_children_level_increments(self):
        """Test that children have incremented level."""
        gsom = MockGSOM(3, 3, neurons_with_children=[(1, 1)])
        adapter = GHSOMNodeAdapter(gsom, level=1)

        children = adapter.children
        assert children[0].level == 2


class TestGHSOMNodeAdapterString:
    """Test GHSOMNodeAdapter string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        gsom = MockGSOM(3, 4)
        adapter = GHSOMNodeAdapter(gsom, level=1, position=(2, 3))

        repr_str = repr(adapter)
        assert 'position (2, 3)' in repr_str
        assert 'map dimensions' in repr_str
        assert 'input dataset' in repr_str
        assert 'level 1' in repr_str

    def test_str(self):
        """Test __str__ method."""
        gsom = MockGSOM(3, 4)
        adapter = GHSOMNodeAdapter(gsom, level=1, position=(2, 3))

        str_str = str(adapter)
        assert 'position (2, 3)' in str_str
        assert 'level 1' in str_str


class TestAdaptModel:
    """Test adapt_model function."""

    def test_adapt_neuron_with_child_map(self):
        """Test adapting Neuron object with child_map."""
        gsom = MockGSOM(3, 3)
        neuron = MockNeuron(child_map=gsom)

        adapted = adapt_model(neuron)

        assert isinstance(adapted, GHSOMNodeAdapter)
        assert adapted.level == 0
        assert adapted.position == (0, 0)

    def test_adapt_gsom_directly(self):
        """Test adapting GSOM object directly."""
        gsom = MockGSOM(3, 3)

        adapted = adapt_model(gsom)

        assert isinstance(adapted, GHSOMNodeAdapter)
        assert adapted.level == 0

    def test_adapt_compatible_node(self):
        """Test adapting already compatible node."""
        node = GHSOMNode((0, 0), (3, 3, 10), 100, 0)

        adapted = adapt_model(node)

        # Should return the same object
        assert adapted is node

    def test_adapt_unsupported_type_raises(self):
        """Test that unsupported type raises ValueError."""
        unsupported = {"some": "dict"}

        with pytest.raises(ValueError, match="Unsupported model type"):
            adapt_model(unsupported)

    def test_adapt_none_child_map(self):
        """Test adapting Neuron with None child_map falls through."""
        neuron = MockNeuron(child_map=None)

        with pytest.raises(ValueError, match="Unsupported model type"):
            adapt_model(neuron)


class TestGHSOMNodeAdapterIntegration:
    """Integration tests for GHSOMNodeAdapter."""

    def test_adapter_hierarchy(self):
        """Test adapter creates proper hierarchy."""
        # Create GSOM with hierarchy
        root_gsom = MockGSOM(3, 3, neurons_with_children=[(1, 1)])
        adapter = GHSOMNodeAdapter(root_gsom)

        # Check hierarchy
        assert adapter.level == 0
        assert len(adapter.children) == 1
        assert adapter.children[0].level == 1

    def test_adapter_matches_node_interface(self):
        """Test that adapter matches GHSOMNode interface."""
        gsom = MockGSOM(3, 3)
        adapter = GHSOMNodeAdapter(gsom)
        node = GHSOMNode((0, 0), (3, 3, 10), 90, 0)

        # Check that adapter has same attributes as node
        assert hasattr(adapter, 'level')
        assert hasattr(adapter, 'position')
        assert hasattr(adapter, 'children')
        assert hasattr(adapter, 'rows')
        assert hasattr(adapter, 'columns')
        assert hasattr(adapter, 'input_dataset_size')
        assert hasattr(adapter, 'map_dimensions')
