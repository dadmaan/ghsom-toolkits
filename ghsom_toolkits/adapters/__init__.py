"""
Compatibility adapters for different GHSOM implementations.

This module provides adapters to make different GHSOM implementations compatible
with ghsom-toolkits' expected Node interface. Currently supports ghsom-py.

Key Components
--------------
- GHSOMNodeAdapter: Adapts ghsom-py GSOM/Neuron objects to Node interface
- adapt_model: Smart auto-detection and adaptation function
- build_lookup_table: Helper to build lookup tables from adapted models

Examples
--------
>>> from ghsom import GHSOM
>>> from ghsom_toolkits.adapters import adapt_model, build_lookup_table
>>> from ghsom_toolkits import visualize_ghsom_hierarchy
>>>
>>> # Train with ghsom-py
>>> ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
>>> result = ghsom.train(epochs_number=50)
>>>
>>> # Adapt to ghsom-toolkits interface
>>> model = adapt_model(result)
>>> lookup = build_lookup_table(model)
>>>
>>> # Use with ghsom-toolkits visualizations
>>> visualize_ghsom_hierarchy(model, lookup, "hierarchy.png")
"""

from ghsom_toolkits.adapters.ghsom_py_adapter import (
    GHSOMNodeAdapter,
    adapt_model,
)
from ghsom_toolkits.adapters.utils import build_lookup_table

__all__ = [
    "GHSOMNodeAdapter",
    "adapt_model",
    "build_lookup_table",
]
