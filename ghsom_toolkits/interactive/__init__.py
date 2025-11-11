"""Interactive visualization tools for GHSOM."""

# Check for optional dependencies
try:
    import dash
    import plotly
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False

if INTERACTIVE_AVAILABLE:
    from ghsom_toolkits.interactive.dashboard import launch_dashboard
    from ghsom_toolkits.interactive.explorer import (
        explore_neuron,
        get_subtree,
        trace_sample_path,
    )

    __all__ = [
        "launch_dashboard",
        "explore_neuron",
        "get_subtree",
        "trace_sample_path",
    ]
else:
    __all__ = []
