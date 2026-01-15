"""Utility functions for situated negotiations."""

from __future__ import annotations

from collections import defaultdict

from .common import EDGE_COLORS, EDGE_TYPES

__all__ = ["safe_min", "deflistdict", "show_edge_colors"]


def safe_min(a, b):
    """Returns min(a, b) assuming None is less than anything."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def deflistdict():
    """Deflistdict."""
    return defaultdict(list)


def show_edge_colors(show: bool = True):
    """Plots the edge colors used with their meaning

    Args:
        show: If True, displays the figure. If False, returns the figure without displaying.

    Returns:
        The plotly figure if show=False, None otherwise.
    """
    import plotly.graph_objects as go

    colors = {}
    for t in EDGE_TYPES:
        colors[t] = EDGE_COLORS[t]

    sorted_names = list(colors.keys())

    n = len(sorted_names)
    ncols = 2
    nrows = n // ncols + 1

    # Create figure with plotly
    fig = go.Figure()

    # Figure dimensions
    width = 800
    height = 500
    h = height / (nrows + 1)
    w = width / ncols

    for i, name in enumerate(sorted_names):
        col = i % ncols
        row = i // ncols
        y = height - (row * h) - h

        xi_line = w * (col + 0.05)
        xf_line = w * (col + 0.25)
        xi_text = w * (col + 0.3)

        # Add text annotation
        fig.add_annotation(
            x=xi_text,
            y=y,
            text=name,
            font=dict(size=h * 0.3),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
        )

        # Add horizontal line
        fig.add_shape(
            type="line",
            x0=xi_line,
            y0=y + h * 0.1,
            x1=xf_line,
            y1=y + h * 0.1,
            line=dict(color=colors[name], width=h * 0.6),
        )

    fig.update_layout(
        width=width,
        height=height,
        xaxis=dict(range=[0, width], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, height], showgrid=False, zeroline=False, visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="white",
    )

    if show:
        fig.show()
        return None

    return fig
