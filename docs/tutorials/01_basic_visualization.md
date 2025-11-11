# Tutorial: Basic Visualization

This tutorial covers the fundamentals of visualizing GHSOM hierarchies using ghsom-toolkits.

## Prerequisites

```bash
pip install ghsom-py ghsom-toolkits
# On Ubuntu/Debian: sudo apt-get install graphviz
# On macOS: brew install graphviz
```

## Step 1: Train a GHSOM Model

First, let's create some sample data and train a GHSOM model:

```python
import numpy as np
from ghsom import GHSOM

# Generate sample data (200 samples, 10 features)
np.random.seed(42)
data = np.random.rand(200, 10)

# Initialize and train GHSOM
ghsom = GHSOM(
    input_dataset=data,
    t1=0.5,  # Growth threshold
    t2=0.05,  # Expansion threshold
    learning_rate=0.1,
    gaussian_sigma=1.0
)

# Train for 50 epochs
model = ghsom.train(epochs_number=50)
print("Training complete!")
```

## Step 2: Adapt the Model

Since we're using ghsom-py, we need to adapt the model to ghsom-toolkits interface:

```python
from ghsom_toolkits.adapters import adapt_model, build_lookup_table

# Adapt the model
adapted_model = adapt_model(model, input_dataset_size=len(data))

# Build lookup table (maps node IDs to node objects)
lookup_table = build_lookup_table(adapted_model)

print(f"Model has {len(lookup_table)} nodes")
```

## Step 3: Visualize the Full Hierarchy

Now let's create our first visualization - the complete hierarchy tree:

```python
from ghsom_toolkits import visualize_ghsom_hierarchy

# Create hierarchy visualization
visualize_ghsom_hierarchy(
    node=adapted_model,
    lookup_table=lookup_table,
    filename="ghsom_hierarchy.png"
)

print("Hierarchy saved to ghsom_hierarchy.png")
```

**What you'll see:**
- Nodes colored by level (root in one color, children in another)
- Each node shows: ID, position, dataset size, and number of children
- Hierarchical tree structure from top (root) to bottom (leaves)

## Step 4: Highlight Specific Nodes

Sometimes you want to focus on a particular node and its context:

```python
from ghsom_toolkits import visualize_node_position

# Find an interesting node (e.g., one with children)
target_node_id = None
for node_id, node in lookup_table.items():
    if len(node.children) > 0 and node_id != "root":
        target_node_id = node_id
        break

if target_node_id:
    # Visualize this node with its ancestors
    visualize_node_position(
        root_node=adapted_model,
        lookup_table=lookup_table,
        node_id=target_node_id,
        filename="node_highlighted.png",
        plot_descendants=False  # Only show ancestors
    )
    print(f"Highlighted node {target_node_id}")

    # Now with descendants
    visualize_node_position(
        root_node=adapted_model,
        lookup_table=lookup_table,
        node_id=target_node_id,
        filename="node_with_descendants.png",
        plot_descendants=True  # Show full subtree
    )
    print("Visualization with descendants saved")
```

## Step 5: Interactive Treemap (Optional)

If you have the interactive dependencies installed, create an interactive treemap:

```python
try:
    from ghsom_toolkits.plotting import plot_hierarchy_treemap

    # Create interactive treemap
    fig = plot_hierarchy_treemap(
        node=adapted_model,
        lookup_table=lookup_table,
        filename="treemap.html",
        width=1200,
        height=800
    )
    print("Interactive treemap saved to treemap.html")
    print("Open it in your browser!")

except ImportError:
    print("Skipping treemap - install with: pip install ghsom-toolkits[interactive]")
```

## Step 6: Exploring the Hierarchy

Let's print some statistics about our hierarchy:

```python
def count_nodes(node):
    """Count total nodes in hierarchy."""
    count = 1
    for child in node.children:
        count += count_nodes(child)
    return count

def max_depth(node, current=0):
    """Find maximum depth."""
    if not node.children:
        return current
    return max(max_depth(child, current + 1) for child in node.children)

def count_leaves(node):
    """Count leaf nodes (clusters)."""
    if not node.children:
        return 1
    return sum(count_leaves(child) for child in node.children)

# Print statistics
print("\n=== Hierarchy Statistics ===")
print(f"Total nodes: {count_nodes(adapted_model)}")
print(f"Maximum depth: {max_depth(adapted_model)}")
print(f"Number of clusters (leaves): {count_leaves(adapted_model)}")
print(f"Root map shape: {adapted_model.rows}x{adapted_model.columns}")
```

## Complete Example Script

Here's everything together:

```python
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits import visualize_ghsom_hierarchy, visualize_node_position

# 1. Generate and train
np.random.seed(42)
data = np.random.rand(200, 10)

ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=50)

# 2. Adapt
adapted_model = adapt_model(model, input_dataset_size=len(data))
lookup_table = build_lookup_table(adapted_model)

# 3. Visualize full hierarchy
visualize_ghsom_hierarchy(
    adapted_model,
    lookup_table,
    "hierarchy.png"
)

# 4. Highlight a node
node_ids = [nid for nid in lookup_table.keys() if nid != "root"]
if node_ids:
    visualize_node_position(
        adapted_model,
        lookup_table,
        node_ids[0],
        "highlighted.png",
        plot_descendants=True
    )

print("✓ Visualizations complete!")
```

## Tips and Best Practices

### Choosing Node Colors

Customize the highlight color for better visibility:

```python
visualize_node_position(
    root_node=adapted_model,
    lookup_table=lookup_table,
    node_id="some_node",
    target_node_color="#FF5733",  # Custom orange color
    filename="custom_color.png"
)
```

### Working with Large Hierarchies

For large hierarchies (>100 nodes), consider:

1. **Visualize subtrees** instead of the full tree
2. **Use interactive treemap** for better navigation
3. **Export to PDF** for vector graphics (scales better)

```python
# Save as PDF for better quality
visualize_ghsom_hierarchy(
    adapted_model,
    lookup_table,
    "hierarchy.pdf"  # PDF instead of PNG
)
```

### Building Custom Lookup Tables

You can customize node IDs in the lookup table:

```python
def custom_lookup(node, prefix="N", lookup=None):
    """Build lookup with custom IDs."""
    if lookup is None:
        lookup = {}

    lookup[prefix] = node

    for i, child in enumerate(node.children):
        child_id = f"{prefix}_{i}"
        custom_lookup(child, child_id, lookup)

    return lookup

lookup = custom_lookup(adapted_model)
visualize_ghsom_hierarchy(adapted_model, lookup, "custom_ids.png")
```

## Next Steps

- [Tutorial 2: Interactive Dashboard](02_interactive_dashboard.md) - Explore models interactively
- [Tutorial 3: Model Comparison](03_model_comparison.md) - Compare different hyperparameters
- [Gallery](../gallery.md) - See all visualization types

## Common Issues

**"GraphViz not found"**
- Install GraphViz system package (not just Python package)
- Add GraphViz to your PATH

**"Lookup table is empty"**
- Make sure to call `build_lookup_table()` after adapting

**"Node not found in lookup table"**
- Verify the node_id exists: `print(list(lookup_table.keys()))`
