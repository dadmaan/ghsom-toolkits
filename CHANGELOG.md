# Changelog

All notable changes to ghsom-toolkits will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-09

### Added
- **Core Module** (`ghsom_toolkits.core`):
  - Enhanced `GHSOMNode` class with lazy loading and caching for improved performance
  - Tree traversal methods: `get_descendants()`, `get_leaves()`, `find_by_level()`
  - Subtree statistics: `get_subtree_stats()`
  - Centralized parsing utilities for GHSOM hierarchy strings
  - Comprehensive lookup table management functions
- **Adapter Module** (`ghsom_toolkits.adapters`):
  - `GHSOMNodeAdapter` class for ghsom-py compatibility
  - Smart `adapt_model()` function with auto-detection of model types
  - `build_lookup_table()` utility for adapted models
  - Support for ghsom-py GSOM and Neuron objects
  - Zero-copy wrapping with <5% performance overhead
- **Comprehensive Test Suite**:
  - 50+ tests for core and adapter modules
  - 90%+ test coverage for new modules
  - Integration tests for adapter compatibility
- **Documentation**:
  - Core module README with API reference and examples
  - Adapters module README with migration guide
  - Updated main README with adapter usage examples

### Changed
- **All example scripts updated** to use new adapter system:
  - `basic_visualization.py`, `heatmap_visualization.py`
  - `cluster_analysis.py`, `model_comparison.py`
  - `generate_report.py`, `interactive_dashboard.py`
- Main `ghsom_toolkits.__init__.py` now exports core functions
- Simplified example code by removing manual GSOM/Neuron handling

### Fixed
- **Resolved ghsom-py incompatibility**: All ghsom-toolkits features now work seamlessly with ghsom-py models
- Missing attributes error (`'GSOM' object has no attribute 'level'`)
- API mismatch between ghsom-py and ghsom-toolkits interfaces

### Removed
- Duplicate `build_lookup_table()` functions from individual example scripts
- Manual GSOM/Neuron extraction logic from examples

### Performance
- Lazy loading of children lists in adapted nodes
- Caching of computed properties (descendants, leaves)
- Optimized tree traversal algorithms
- <5% performance overhead for all adapter operations

### Migration from 0.1.0

**For ghsom-py users:**
```python
# Before (0.1.0) - didn't work
model = result.child_map if hasattr(result, 'child_map') else result
visualize_ghsom_hierarchy(model, lookup, "out.png")  # Error!

# After (0.2.0) - works seamlessly
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
model = adapt_model(result)  # Auto-detects and adapts
lookup = build_lookup_table(model)
visualize_ghsom_hierarchy(model, lookup, "out.png")  # Works!
```

**For main project users:**
No changes required - functionality remains the same with improved performance.

## [0.1.0] - 2024-10-08

### Added
- Initial release of ghsom-toolkits
- Hierarchical visualization functions:
  - `visualize_ghsom_hierarchy()` - Full tree visualization
  - `visualize_node_position()` - Focused node visualization with ancestors/descendants
- Cluster plotting functions:
  - `plot_ghsom_clusters()` - Bar chart of cluster distribution
  - `plot_ghsom_clusters_pie()` - Pie chart of cluster proportions
  - `plot_ghsom_clusters_genre_stacked()` - Domain-specific stacked bar chart
- Export utilities for Graphviz/pydot integration
- Color scheme utilities for hierarchical levels
- Comprehensive test suite with 59% coverage
- Full documentation in README.md
- Example scripts demonstrating basic usage

### Dependencies
- ghsom-py >= 0.1.0
- matplotlib >= 3.5.0
- pydot >= 1.4.2
- numpy >= 1.20.0

[0.1.0]: https://github.com/dadmaan/ghsom-toolkits/releases/tag/v0.1.0
