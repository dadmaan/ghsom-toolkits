"""Color schemes and utilities for GHSOM visualizations."""

from typing import Dict


def get_level_colors() -> Dict[int, str]:
    """
    Get color scheme for different hierarchy levels.

    Returns a dictionary mapping hierarchy levels to hex color codes
    with a Nordic/blue color palette.

    Returns
    -------
    dict
        Mapping of level numbers (0-9) to hex color strings
    """
    return {
        0: "#5e81ac",  # Slate blue
        1: "#88c0d0",  # Polar blue
        2: "#81a1c1",  # Nordic blue
        3: "#5e81ac",  # Glacier blue
        4: "#8fbcbb",  # Iceberg green
        5: "#88c0d0",  # Frost blue
        6: "#81a1c1",  # Arctic blue
        7: "#5e81ac",  # Fjord blue
        8: "#8fbcbb",  # Frozen green
        9: "#88c0d0",  # Polar night blue
    }
