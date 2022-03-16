from __future__ import annotations

from collections import defaultdict
from pathlib import Path

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
    return defaultdict(list)


def show_edge_colors():
    """Plots the edge colors used with their meaning"""
    import matplotlib.pyplot as plt

    colors = {}
    for t in EDGE_TYPES:
        colors[t] = EDGE_COLORS[t]

    # Sort colors by hue, saturation, value and name.
    # sorted_colors = colors.values()
    sorted_names = colors.keys()

    n = len(sorted_names)
    ncols = 2
    nrows = n // ncols + 1

    fig, ax = plt.subplots(figsize=(8, 5))

    # Get height and width
    X, Y = fig.get_dpi() * fig.get_size_inches()
    h = Y / (nrows + 1)
    w = X / ncols

    for i, name in enumerate(sorted_names):
        col = i % ncols
        row = i // ncols
        y = Y - (row * h) - h

        xi_line = w * (col + 0.05)
        xf_line = w * (col + 0.25)
        xi_text = w * (col + 0.3)

        ax.text(
            xi_text,
            y,
            name,
            fontsize=(h * 0.3),
            horizontalalignment="left",
            verticalalignment="center",
        )

        ax.hlines(
            y + h * 0.1, xi_line, xf_line, color=colors[name], linewidth=(h * 0.6)
        )

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    plt.show()
