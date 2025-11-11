"""
Adapter for ghsom-py GSOM and Neuron objects.

This module provides adapters to make ghsom-py's GSOM and Neuron objects
compatible with ghsom-toolkits' expected Node interface.
"""

from typing import List, Optional, Tuple, Union
import numpy as np


class NeuronWrapper:
    """
    Wrapper for ghsom-py Neuron to provide weights property.

    This wrapper adds a 'weights' property that returns the weight vector,
    making neurons compatible with heatmap visualization functions.
    """

    def __init__(self, neuron):
        """Initialize wrapper with original neuron."""
        self._neuron = neuron

    @property
    def weights(self) -> np.ndarray:
        """Get weight vector as numpy array."""
        if hasattr(self._neuron, 'weight_vector'):
            return self._neuron.weight_vector()
        return np.array([])

    @property
    def children(self) -> List:
        """
        Get children list.

        Maps ghsom-py's child_map to a list for compatibility.
        Returns a list containing the adapted child_map if it exists.
        """
        if hasattr(self._neuron, 'child_map') and self._neuron.child_map is not None:
            # Import here to avoid circular dependency
            # Return adapted child map
            from ghsom_toolkits.adapters.ghsom_py_adapter import GHSOMNodeAdapter
            child_level = getattr(self, 'level', 0) + 1
            child_pos = getattr(self._neuron, 'position', (0, 0))
            return [GHSOMNodeAdapter(self._neuron.child_map, level=child_level, position=child_pos)]
        return []

    def __getattr__(self, name):
        """Delegate all other attributes to wrapped neuron."""
        return getattr(self._neuron, name)


class GHSOMNodeAdapter:
    """
    Adapts ghsom-py GSOM/Neuron objects to ghsom-toolkits Node interface.

    This adapter wraps ghsom-py's GSOM or Neuron objects and provides the
    properties expected by ghsom-toolkits visualization and analysis functions.

    Parameters
    ----------
    gsom_or_neuron : object
        A GSOM or Neuron object from ghsom-py
    level : int, optional
        Depth level in the hierarchy (default: 0)
    position : Tuple[int, int], optional
        Position in parent map (default: (0, 0))

    Attributes
    ----------
    level : int
        Depth level in the hierarchy
    position : Tuple[int, int]
        Position in parent map
    children : List[GHSOMNodeAdapter]
        Child nodes (lazy loaded)
    rows : int
        Number of rows in the map
    columns : int
        Number of columns in the map
    input_dataset_size : int
        Number of input data points
    map_dimensions : Tuple[int, int, int]
        Map dimensions (rows, cols, depth)

    Examples
    --------
    >>> from ghsom import GHSOM
    >>> from ghsom_toolkits.adapters import GHSOMNodeAdapter
    >>>
    >>> ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
    >>> result = ghsom.train(epochs_number=50)
    >>> model = result.child_map  # Get GSOM
    >>> adapted = GHSOMNodeAdapter(model, level=0, position=(0, 0))
    >>> adapted.rows
    5
    """

    def __init__(
        self,
        gsom_or_neuron,
        level: int = 0,
        position: Tuple[int, int] = (0, 0)
    ):
        """
        Initialize adapter for ghsom-py object.

        Parameters
        ----------
        gsom_or_neuron : object
            GSOM or Neuron object from ghsom-py
        level : int
            Hierarchy level
        position : Tuple[int, int]
            Position in parent map
        """
        self._wrapped = gsom_or_neuron
        self._level = level
        self._position = position
        self._children_cache: Optional[List["GHSOMNodeAdapter"]] = None

    @property
    def level(self) -> int:
        """Get hierarchy level."""
        return self._level

    @property
    def position(self) -> Tuple[int, int]:
        """Get position in parent map."""
        return self._position

    @property
    def children(self) -> List["GHSOMNodeAdapter"]:
        """
        Get list of child nodes (lazy loaded).

        Builds the children list by traversing the GSOM's neurons dict
        and finding neurons with child_map attributes.

        Returns
        -------
        List[GHSOMNodeAdapter]
            List of adapted child nodes
        """
        if self._children_cache is None:
            self._children_cache = self._build_children_list()
        return self._children_cache

    @property
    def rows(self) -> int:
        """Get number of rows in the map."""
        try:
            shape = self._wrapped.map_shape()
            return shape[0]
        except (AttributeError, IndexError):
            # Fallback: try to infer from neurons dict
            if hasattr(self._wrapped, 'neurons') and self._wrapped.neurons:
                max_row = max(pos[0] for pos in self._wrapped.neurons.keys())
                return max_row + 1
            return 0

    @property
    def columns(self) -> int:
        """Get number of columns in the map."""
        try:
            shape = self._wrapped.map_shape()
            return shape[1]
        except (AttributeError, IndexError):
            # Fallback: try to infer from neurons dict
            if hasattr(self._wrapped, 'neurons') and self._wrapped.neurons:
                max_col = max(pos[1] for pos in self._wrapped.neurons.keys())
                return max_col + 1
            return 0

    @property
    def input_dataset_size(self) -> int:
        """
        Get number of input data points.

        Calculates the total number of input data points by summing
        the dataset sizes of all neurons in the map.

        Returns
        -------
        int
            Total number of input data points
        """
        if hasattr(self._wrapped, 'neurons') and isinstance(self._wrapped.neurons, dict):
            total_size = 0
            for neuron in self._wrapped.neurons.values():
                if hasattr(neuron, 'input_dataset') and neuron.input_dataset is not None:
                    try:
                        total_size += len(neuron.input_dataset)
                    except TypeError:
                        # input_dataset might not be a sized collection
                        total_size += 1
            return total_size
        return 0

    @property
    def map_dimensions(self) -> Tuple[int, int, int]:
        """
        Get map dimensions tuple.

        Returns
        -------
        Tuple[int, int, int]
            Map dimensions (rows, cols, depth)
        """
        # For ghsom-py, depth is typically the weight vector dimension
        depth = 0
        if hasattr(self._wrapped, 'neurons') and self._wrapped.neurons:
            first_neuron = next(iter(self._wrapped.neurons.values()))
            if hasattr(first_neuron, 'weight_vector'):
                try:
                    weight_vec = first_neuron.weight_vector()
                    if hasattr(weight_vec, '__len__'):
                        depth = len(weight_vec)
                except (AttributeError, TypeError):
                    depth = 0

        return (self.rows, self.columns, depth)

    @property
    def map(self):
        """
        Get 2D array-like access to neurons.

        Returns
        -------
        List[List[NeuronWrapper]]
            2D list of wrapped neurons accessible via map[row][col]
        """
        if not hasattr(self._wrapped, 'neurons'):
            return []

        # Build 2D array from neurons dict with wrappers
        rows, cols = self.rows, self.columns
        neuron_map = [[None for _ in range(cols)] for _ in range(rows)]

        for pos, neuron in self._wrapped.neurons.items():
            row, col = pos
            neuron_map[row][col] = NeuronWrapper(neuron)

        return neuron_map

    def _build_children_list(self) -> List["GHSOMNodeAdapter"]:
        """
        Build list of child nodes from neurons dict.

        Traverses the GSOM's neurons dictionary and creates adapted
        child nodes for neurons that have child_map attributes.

        Returns
        -------
        List[GHSOMNodeAdapter]
            List of adapted child nodes
        """
        children = []

        if hasattr(self._wrapped, 'neurons') and isinstance(self._wrapped.neurons, dict):
            for pos, neuron in self._wrapped.neurons.items():
                if hasattr(neuron, 'child_map') and neuron.child_map is not None:
                    child_adapter = GHSOMNodeAdapter(
                        neuron.child_map,
                        level=self._level + 1,
                        position=pos
                    )
                    children.append(child_adapter)

        return children

    def __repr__(self) -> str:
        """String representation matching GHSOMNode format."""
        return (
            f"position {self.position} -- map dimensions {self.map_dimensions} -- "
            f"input dataset {self.input_dataset_size} element(s) -- level {self.level}"
        )

    def __str__(self) -> str:
        """String representation matching GHSOMNode format."""
        return self.__repr__()


def adapt_model(model) -> Union[GHSOMNodeAdapter, object]:
    """
    Auto-detect and adapt ghsom-py model to ghsom-toolkits interface.

    This function intelligently detects the type of GHSOM model and returns
    an appropriate adapter or passes through compatible models.

    Parameters
    ----------
    model : object
        A GHSOM model object (Neuron, GSOM, or GHSOMNode)

    Returns
    -------
    Union[GHSOMNodeAdapter, object]
        Adapted model compatible with ghsom-toolkits interface

    Raises
    ------
    ValueError
        If the model type is not supported

    Examples
    --------
    >>> from ghsom import GHSOM
    >>> from ghsom_toolkits.adapters import adapt_model
    >>>
    >>> ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
    >>> result = ghsom.train(epochs_number=50)
    >>> model = adapt_model(result)  # Auto-detects and adapts
    >>> model.level
    0
    """
    # Check if already compatible (has required interface)
    if (hasattr(model, 'level') and
        hasattr(model, 'children') and
        hasattr(model, 'rows') and
        hasattr(model, 'columns') and
        hasattr(model, 'position') and
        hasattr(model, 'input_dataset_size')):
        # Already compatible (e.g., GHSOMNode from main project)
        return model

    # Check if it's a Neuron object with child_map (from ghsom.train())
    if hasattr(model, 'child_map') and model.child_map is not None:
        # Root Neuron from ghsom-py, extract the GSOM
        return GHSOMNodeAdapter(model.child_map, level=0, position=(0, 0))

    # Check if it's a GSOM object directly
    if hasattr(model, 'neurons') and hasattr(model, 'map_shape'):
        # GSOM object from ghsom-py
        return GHSOMNodeAdapter(model, level=0, position=(0, 0))

    # Unsupported type
    raise ValueError(
        f"Unsupported model type: {type(model).__name__}. "
        f"Expected Neuron (with child_map), GSOM, or compatible Node object. "
        f"Available attributes: {dir(model)[:10]}..."
    )
