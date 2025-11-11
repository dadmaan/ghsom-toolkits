"""Interactive Dash dashboard for GHSOM exploration."""

import logging
from typing import Any, Dict, Optional

import numpy as np

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import plotly.graph_objects as go
    import plotly.express as px
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from ghsom_toolkits.interactive.explorer import (
    explore_neuron,
    get_node_statistics,
    trace_sample_path,
)

logger = logging.getLogger(__name__)


def _check_dash_available() -> None:
    """Check if Dash is available and raise error if not."""
    if not DASH_AVAILABLE:
        raise ImportError(
            "dash and plotly are required for interactive dashboard. "
            "Install with: pip install ghsom-toolkits[interactive]"
        )


def _create_hierarchy_figure(node: Any, lookup_table: Dict[str, Any]) -> go.Figure:
    """
    Create a plotly figure showing the hierarchy structure.

    Parameters
    ----------
    node : object
        Root GHSOM node
    lookup_table : dict
        Lookup table mapping node IDs to nodes

    Returns
    -------
    plotly.graph_objects.Figure
        Hierarchy visualization
    """
    # Collect node positions and edges
    nodes_x = []
    nodes_y = []
    node_texts = []
    node_ids = []

    edges_x = []
    edges_y = []

    # Layout parameters
    level_height = 1.0
    node_spacing = 1.0

    # Track node positions at each level
    level_counters = {}

    def traverse(n: Any, node_id: str, level: int, parent_x: Optional[float] = None) -> float:
        # Get position for this node
        if level not in level_counters:
            level_counters[level] = 0

        x_pos = level_counters[level] * node_spacing
        level_counters[level] += 1

        y_pos = -level * level_height

        nodes_x.append(x_pos)
        nodes_y.append(y_pos)
        node_texts.append(f"{node_id}<br>Size: {n.input_dataset_size}<br>Shape: {n.rows}x{n.columns}")
        node_ids.append(node_id)

        # Add edge from parent if exists
        if parent_x is not None:
            edges_x.extend([parent_x, x_pos, None])
            edges_y.extend([-level * level_height + level_height, y_pos, None])

        # Traverse children
        for i, child in enumerate(n.children):
            child_id = f"{node_id}_c{i}"
            traverse(child, child_id, level + 1, x_pos)

        return x_pos

    traverse(node, "root", 0)

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(
        go.Scatter(
            x=edges_x,
            y=edges_y,
            mode="lines",
            line=dict(color="gray", width=1),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Add nodes
    fig.add_trace(
        go.Scatter(
            x=nodes_x,
            y=nodes_y,
            mode="markers+text",
            marker=dict(size=20, color="steelblue", line=dict(color="black", width=1)),
            text=node_ids,
            textposition="top center",
            hovertext=node_texts,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="GHSOM Hierarchy",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        height=600,
    )

    return fig


def _create_map_figure(node: Any) -> go.Figure:
    """
    Create a heatmap visualization of a GHSOM map.

    Parameters
    ----------
    node : object
        GHSOM node to visualize

    Returns
    -------
    plotly.graph_objects.Figure
        Map visualization
    """
    rows, cols = node.rows, node.columns

    # Create a simple representation (e.g., number of children)
    z_data = np.zeros((rows, cols))
    hover_text = []

    for r in range(rows):
        row_text = []
        for c in range(cols):
            neuron = node.map[r][c]
            z_data[r, c] = len(neuron.children)
            row_text.append(
                f"Position: ({r}, {c})<br>"
                f"Children: {len(neuron.children)}<br>"
                f"Expanded: {len(neuron.children) > 0}"
            )
        hover_text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            text=hover_text,
            hoverinfo="text",
            colorscale="Blues",
            showscale=True,
        )
    )

    fig.update_layout(
        title=f"Map Structure ({rows}x{cols})",
        xaxis_title="Column",
        yaxis_title="Row",
        height=400,
    )

    return fig


def launch_dashboard(
    node: Any,
    lookup_table: Optional[Dict[str, Any]] = None,
    data: Optional[np.ndarray] = None,
    port: int = 8050,
    debug: bool = False,
) -> None:
    """
    Launch an interactive Dash dashboard for GHSOM exploration.

    The dashboard provides:
    - Interactive hierarchy visualization
    - Metrics panel showing model statistics
    - Node detail view with map visualization
    - Sample tracing (if data provided)

    Parameters
    ----------
    node : object
        Root GHSOM node
    lookup_table : dict, optional
        Lookup table mapping node IDs to nodes. If None, will be generated.
    data : numpy.ndarray, optional
        Original training data for sample tracing. Shape (n_samples, n_features).
    port : int, optional
        Port to run the dashboard on (default: 8050)
    debug : bool, optional
        Enable debug mode (default: False)

    Examples
    --------
    >>> from ghsom import GHSOM
    >>> from ghsom_toolkits.interactive import launch_dashboard
    >>> ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
    >>> model = ghsom.train(epochs_number=50)
    >>> launch_dashboard(model, data=data, port=8050)
    """
    _check_dash_available()

    # Build lookup table if not provided
    if lookup_table is None:
        lookup_table = {}

        def build_lookup(n: Any, prefix: str) -> None:
            lookup_table[prefix] = n
            for i, child in enumerate(n.children):
                build_lookup(child, f"{prefix}_c{i}")

        build_lookup(node, "root")

    # Get model statistics
    stats = get_node_statistics(node)

    # Count total nodes in hierarchy
    total_nodes = len(lookup_table)

    # Create Dash app
    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    # Define layout
    app.layout = html.Div(
        [
            html.H1("GHSOM Explorer Dashboard", style={"textAlign": "center"}),
            html.Hr(),
            # Metrics row
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Model Statistics"),
                            html.P(f"Total Nodes: {total_nodes}"),
                            html.P(f"Root Map Shape: {stats['map_shape'][0]}x{stats['map_shape'][1]}"),
                            html.P(f"Dataset Size: {stats['dataset_size']}"),
                            html.P(f"Leaf Neurons: {stats['num_leaf_neurons']}"),
                            html.P(f"Expanded Neurons: {stats['num_expanded_neurons']}"),
                            html.P(
                                f"Expansion Rate: {stats['expansion_rate']:.2%}"
                            ),
                        ],
                        style={
                            "padding": "20px",
                            "backgroundColor": "#f0f0f0",
                            "borderRadius": "5px",
                            "marginBottom": "20px",
                        },
                    ),
                ]
            ),
            # Hierarchy visualization
            html.Div(
                [
                    html.H3("Hierarchy Structure"),
                    dcc.Graph(
                        id="hierarchy-graph",
                        figure=_create_hierarchy_figure(node, lookup_table),
                    ),
                ]
            ),
            # Map visualization
            html.Div(
                [
                    html.H3("Root Map Visualization"),
                    dcc.Graph(id="map-graph", figure=_create_map_figure(node)),
                ]
            ),
            # Sample tracing (if data provided)
            html.Div(
                [
                    html.H3("Sample Path Tracing"),
                    html.P("Enter sample index to trace through hierarchy:"),
                    dcc.Input(
                        id="sample-index",
                        type="number",
                        placeholder="Sample index",
                        min=0,
                        max=len(data) - 1 if data is not None else 0,
                        value=0,
                        disabled=data is None,
                    ),
                    html.Button("Trace", id="trace-button", n_clicks=0, disabled=data is None),
                    html.Div(id="trace-output"),
                ]
                if data is not None
                else html.Div(html.P("No data provided for sample tracing")),
                style={"marginTop": "20px"},
            ),
        ],
        style={"padding": "20px", "maxWidth": "1400px", "margin": "0 auto"},
    )

    # Callback for sample tracing
    if data is not None:

        @app.callback(
            Output("trace-output", "children"),
            Input("trace-button", "n_clicks"),
            Input("sample-index", "value"),
        )
        def trace_sample(n_clicks: int, sample_idx: Optional[int]) -> html.Div:
            if n_clicks == 0 or sample_idx is None:
                return html.Div()

            if sample_idx < 0 or sample_idx >= len(data):
                return html.Div(
                    f"Invalid sample index. Must be between 0 and {len(data) - 1}",
                    style={"color": "red"},
                )

            path = trace_sample_path(node, data[sample_idx])

            # Format path output
            path_items = []
            for step in path:
                path_items.append(
                    html.Div(
                        [
                            html.P(
                                f"Level {step['level']}: "
                                f"BMU at {step['bmu_position']} "
                                f"(distance: {step['bmu_distance']:.4f})",
                                style={"marginBottom": "5px"},
                            )
                        ]
                    )
                )

            return html.Div(
                [html.H4(f"Path for Sample {sample_idx}:")] + path_items,
                style={
                    "marginTop": "10px",
                    "padding": "10px",
                    "backgroundColor": "#e8f4f8",
                    "borderRadius": "5px",
                },
            )

    # Run server
    logger.info(f"Starting GHSOM dashboard on http://localhost:{port}")
    print(f"\n🚀 GHSOM Dashboard starting...")
    print(f"📊 Open your browser and navigate to: http://localhost:{port}")
    print(f"📈 Model has {total_nodes} nodes with {stats['dataset_size']} samples")
    print(f"\nPress Ctrl+C to stop the server\n")

    app.run_server(debug=debug, port=port, host="127.0.0.1")
