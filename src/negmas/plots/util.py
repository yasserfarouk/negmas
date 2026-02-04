"""Plotting utilities for visualizing negotiation runs and utility functions."""

from __future__ import annotations

import math
import pathlib
import uuid
from typing import TYPE_CHECKING, Callable, Protocol, TypeVar, Generic

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from negmas.common import MechanismState, NegotiatorMechanismInterface, TraceElement
from negmas.gb import ResponseType
from negmas.helpers.misc import make_callable
from negmas.helpers.strings import humanize_time
from negmas.negotiators import Negotiator
from negmas.outcomes.base_issue import Issue
from negmas.outcomes.common import Outcome, os_or_none
from negmas.outcomes.outcome_space import make_os
from negmas.outcomes.protocols import OutcomeSpace
from negmas.preferences import BaseUtilityFunction
from negmas.preferences.crisp_ufun import UtilityFunction
from negmas.preferences.ops import (
    kalai_points,
    ks_points,
    max_relative_welfare_points,
    max_welfare_points,
    nash_points,
    pareto_frontier,
)

DEFAULT_IMAGE_FORMAT = "webp"
SUPPORTED_IMAGE_FORMATS = {"webp", "png", "jpg", "jpeg", "svg", "pdf"}

if TYPE_CHECKING:
    pass

__all__ = [
    "plot_offer_utilities",
    "plot_2dutils",
    "plot_mechanism_run",
    "opacity_colorizer",
    "default_colorizer",
    "Colorizer",
    "plot_offline_run",
]


Colorizer = Callable[[TraceElement], float]


def _is_subplot_figure(fig: go.Figure) -> bool:
    """Check if a figure was created with make_subplots."""
    try:
        grid_ref = fig._grid_ref
        return grid_ref is not None
    except AttributeError:
        return False


def _add_trace_safe(
    fig: go.Figure,
    trace: go.Scatter | go.Bar,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add a trace to a figure, handling both subplot and non-subplot figures."""
    if _is_subplot_figure(fig) and row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)


def _add_annotation_safe(
    fig: go.Figure, row: int | None = None, col: int | None = None, **kwargs
) -> None:
    """Add an annotation to a figure, handling both subplot and non-subplot figures."""
    if _is_subplot_figure(fig) and row is not None and col is not None:
        fig.add_annotation(row=row, col=col, **kwargs)
    else:
        fig.add_annotation(**kwargs)


def _add_shape_safe(
    fig: go.Figure, row: int | None = None, col: int | None = None, **kwargs
) -> None:
    """Add a shape to a figure, handling both subplot and non-subplot figures."""
    if _is_subplot_figure(fig) and row is not None and col is not None:
        fig.add_shape(row=row, col=col, **kwargs)
    else:
        fig.add_shape(**kwargs)


def _update_xaxes_safe(
    fig: go.Figure, row: int | None = None, col: int | None = None, **kwargs
) -> None:
    """Update x-axes, handling both subplot and non-subplot figures."""
    if _is_subplot_figure(fig) and row is not None and col is not None:
        fig.update_xaxes(row=row, col=col, **kwargs)
    else:
        fig.update_xaxes(**kwargs)


def _update_yaxes_safe(
    fig: go.Figure, row: int | None = None, col: int | None = None, **kwargs
) -> None:
    """Update y-axes, handling both subplot and non-subplot figures."""
    if _is_subplot_figure(fig) and row is not None and col is not None:
        fig.update_yaxes(row=row, col=col, **kwargs)
    else:
        fig.update_yaxes(**kwargs)


def opacity_colorizer(t: TraceElement, alpha: float = 1.0):
    """Returns a constant opacity value, ignoring the trace element.

    Args:
        t: The trace element (unused).
        alpha: The constant opacity value to return (0.0 to 1.0).
    """
    _ = t
    return alpha


def default_colorizer(t: TraceElement):
    """Computes opacity based on the fraction of negotiators accepting the offer.

    Args:
        t: The trace element containing offer and response information.
    """
    if not t.responses:
        return 1.0
    return (
        0.9
        * len([_ for _ in t.responses.values() if _ == ResponseType.ACCEPT_OFFER])
        / len(t.responses)
        + 0.1
    )


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert hex color to rgba string.

    Args:
        hex_color: Hex color string (e.g., '#FF0000' or 'FF0000').
        alpha: Alpha value (0.0 to 1.0).

    Returns:
        RGBA string for plotly.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def plotly_to_mpl_color(color):
    """Convert Plotly color format to matplotlib color format.

    Args:
        color: Color in various formats (hex, rgb/rgba string, named color, tuple).

    Returns:
        Color in matplotlib-compatible format (hex string, named color, or RGB tuple).
    """
    if isinstance(color, str):
        if color.startswith("#"):
            # Hex color - matplotlib supports directly
            return color
        elif color.startswith("rgb"):
            # Parse rgb(...) or rgba(...)
            import re

            match = re.match(
                r"rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)", color
            )
            if match:
                r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
                # Convert to 0-1 range for matplotlib
                return (r / 255.0, g / 255.0, b / 255.0)
            return "black"  # fallback
        else:
            # Named color - matplotlib should support directly
            return color
    elif isinstance(color, (tuple, list)):
        # Already a tuple - check if 0-1 range or 0-255 range
        if len(color) >= 3:
            if all(0 <= c <= 1 for c in color[:3]):
                return tuple(color[:3])  # Already in matplotlib format
            else:
                # Convert from 0-255 to 0-1
                return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    return "black"  # fallback


def color_to_rgba(color, alpha: float = 1.0) -> str:
    """Convert various color formats to rgba string.

    Args:
        color: Color in various formats (hex, rgb tuple, rgba tuple, named color).
        alpha: Alpha value (0.0 to 1.0).

    Returns:
        RGBA string for plotly.
    """
    if isinstance(color, str):
        if color.startswith("#"):
            return hex_to_rgba(color, alpha)
        elif color.startswith("rgb"):
            # Already in rgb/rgba format
            if color.startswith("rgba"):
                return color
            # Convert rgb to rgba
            values = color[4:-1].split(",")
            return f"rgba({values[0]},{values[1]},{values[2]},{alpha})"
        else:
            # Named color - return as is with opacity
            return color
    elif isinstance(color, (tuple, list)):
        if len(color) == 3:
            # RGB tuple (0-1 range or 0-255 range)
            r, g, b = color
            if all(0 <= c <= 1 for c in color):
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
            return f"rgba({r},{g},{b},{alpha})"
        elif len(color) == 4:
            # RGBA tuple
            r, g, b, a = color
            if all(0 <= c <= 1 for c in color):
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
            return f"rgba({r},{g},{b},{a})"
    return f"rgba(0,0,0,{alpha})"


ALL_MARKERS = [
    "square",
    "circle",
    "triangle-down",
    "triangle-up",
    "triangle-left",
    "triangle-right",
    "pentagon",
    "hexagon",
    "star",
    "diamond",
    "cross",
    "x",
    "hourglass",
    "bowtie",
]
PROPOSALS_ALPHA = 0.7
AGREEMENT_ALPHA = 0.9
PARETO_ALPHA = 0.4
NASH_ALPHA = 0.6
KALAI_ALPHA = 0.6
KS_ALPHA = 0.6
RESERVED_ALPHA = 0.08
WELFARE_ALPHA = 0.6

# Marker symbols for special points
KALAI_MARKER = "triangle-down"
KS_MARKER = "triangle-up"
WELFARE_MARKER = "triangle-right"
NASH_MARKER = "triangle-left"

KALAI_COLOR = "green"
KS_COLOR = "cyan"
WELFARE_COLOR = "blue"
NASH_COLOR = "brown"

AGREEMENT_SCALE = 20
NASH_SCALE = 12
KALAI_SCALE = 12
KS_SCALE = 12
WELFARE_SCALE = 12
OUTCOMES_SCALE = 6
PARETO_SCALE = 8

TNegotiator = TypeVar("TNegotiator", bound=Negotiator)
TNMI = TypeVar("TNMI", bound=NegotiatorMechanismInterface, covariant=True)


class PlottableMechanism(Protocol, Generic[TNMI, TNegotiator]):
    @property
    def outcome_space(self) -> OutcomeSpace:
        """The space of all possible outcomes in this mechanism."""
        ...

    @property
    def negotiators(self) -> list[TNegotiator]:
        """The list of negotiators participating in this mechanism."""
        ...

    @property
    def negotiator_ids(self) -> list[str]:
        """The unique identifiers of all negotiators in this mechanism."""
        ...

    @property
    def negotiator_names(self) -> list[str]:
        """The human-readable names of all negotiators in this mechanism."""
        ...

    @property
    def nmi(self) -> TNMI:
        """The negotiator-mechanism interface for accessing mechanism state."""
        ...

    @property
    def agreement(self) -> Outcome | None:
        """The final agreement reached, or None if no agreement was reached."""
        ...

    @property
    def state(self) -> MechanismState:
        """The current state of the mechanism including timing and error info."""
        ...

    @property
    def full_trace(self) -> list[TraceElement]:
        """Complete history of all offers and responses during negotiation."""
        ...

    def discrete_outcomes(self) -> list[Outcome]:
        """Returns all discrete outcomes in the outcome space."""
        ...

    def negotiator_index(self, source: str) -> int | None:
        """Returns the index of a negotiator given their ID or name.

        Args:
            source: The negotiator ID or name to look up.

        Returns:
            int | None: The index of the negotiator, or None if not found.
        """
        ...


DEFAULT_COLORMAP = "Jet"

# Plotly color scales
PLOTLY_COLORMAPS = {
    "jet": "Jet",
    "viridis": "Viridis",
    "plasma": "Plasma",
    "inferno": "Inferno",
    "magma": "Magma",
    "cividis": "Cividis",
    "turbo": "Turbo",
}


def get_cmap_colors(n: int, name: str = DEFAULT_COLORMAP) -> list[str]:
    """Returns a list of n distinct colors from a colormap.

    Args:
        n: Number of colors to generate.
        name: Colormap name.

    Returns:
        List of color strings.
    """
    import plotly.express as px

    # Map common matplotlib colormap names to plotly equivalents
    name_lower = name.lower()
    if name_lower in PLOTLY_COLORMAPS:
        name = PLOTLY_COLORMAPS[name_lower]

    # Get colors from plotly's color scales
    try:
        colors = px.colors.sample_colorscale(
            name, [i / (n - 1) if n > 1 else 0.5 for i in range(n)]
        )
    except Exception:
        # Fallback to qualitative colors
        colors = px.colors.qualitative.Plotly[:n]
        if len(colors) < n:
            # Cycle through if not enough colors
            colors = [colors[i % len(colors)] for i in range(n)]
    return colors


def make_colors_and_markers(colors, markers, n: int, colormap=DEFAULT_COLORMAP):
    """Generates color and marker lists for plotting multiple series.

    Args:
        colors: Predefined color list, or None to auto-generate from colormap.
        markers: Predefined marker list, or None to auto-generate.
        n: Number of distinct colors/markers needed.
        colormap: Name of colormap to use when auto-generating colors.
    """
    if not colors:
        colors = get_cmap_colors(n, colormap)
    if not markers:
        markers = [ALL_MARKERS[i % len(ALL_MARKERS)] for i in range(n)]
    return colors, markers


def plot_offer_utilities(
    trace: list[TraceElement],
    negotiator: str,
    plotting_ufuns: list[BaseUtilityFunction | None],
    plotting_negotiators: list[str],
    ignore_none_offers: bool = True,
    name_map: dict[str, str] | Callable[[str], str] | None = None,
    colors: list | None = None,
    markers: list | None = None,
    colormap: str = DEFAULT_COLORMAP,
    fig: go.Figure | None = None,
    row: int = 1,
    col: int = 1,
    sharey: bool = False,
    xdim: str = "relative_time",
    ylimits: tuple[float, float] | None = None,
    show_legend: bool = True,
    show_x_label: bool = True,
    ignore_markers_limit: int = 200,
    show_reserved: bool = True,
    colorizer: Colorizer | None = None,
    first_color_index: int = 0,
    mark_offers_view: bool = True,
):
    """Plots utility values of offers over time for a specific negotiator.

    Args:
        trace: List of trace elements recording all offers made during negotiation.
        negotiator: ID of the negotiator whose offers to plot on x-axis.
        plotting_ufuns: Utility functions to evaluate offers against.
        plotting_negotiators: IDs of negotiators corresponding to each utility function.
        ignore_none_offers: Whether to skip None offers in the plot.
        name_map: Mapping from negotiator IDs to display names.
        colors: List of colors for each negotiator's line.
        markers: List of marker symbols for each negotiator.
        colormap: Colormap name for auto-generating colors.
        fig: Existing Plotly figure to add traces to, or None to create new.
        row: Row index in subplot grid (1-indexed).
        col: Column index in subplot grid (1-indexed).
        sharey: Whether this subplot shares y-axis with others.
        xdim: X-axis dimension ('relative_time', 'step', or 'time').
        ylimits: Optional (min, max) limits for y-axis.
        show_legend: Whether to display legend entries.
        show_x_label: Whether to show x-axis label.
        ignore_markers_limit: Hide markers if trace has more elements than this.
        show_reserved: Whether to show reserved value lines.
        colorizer: Function to compute opacity for each trace element.
        first_color_index: Index offset for color assignment.
        mark_offers_view: Whether to mark special states (errors, agreement, timeout).
    """
    if colorizer is None:
        colorizer = default_colorizer

    map_ = make_callable(name_map)
    if fig is None:
        fig = go.Figure()

    colors, markers = make_colors_and_markers(
        colors, markers, len(plotting_negotiators), colormap
    )
    if first_color_index:
        colors = (
            colors[:first_color_index]
            + colors[first_color_index + 1 :]
            + [colors[first_color_index]]
        )
        markers = (
            markers[:first_color_index]
            + markers[first_color_index + 1 :]
            + [markers[first_color_index]]
        )

    if xdim.startswith("step") or xdim.startswith("round"):
        trace_info = [(_, _.step) for _ in trace if _.negotiator == negotiator]
    elif xdim.startswith("time") or xdim.startswith("real"):
        trace_info = [(_, _.time) for _ in trace if _.negotiator == negotiator]
    else:
        trace_info = [(_, _.relative_time) for _ in trace if _.negotiator == negotiator]
    x = [_[-1] for _ in trace_info]
    simple_offers_view = (
        len(plotting_negotiators) == 1 and plotting_negotiators[0] == negotiator
    )

    for i, (u, neg) in enumerate(zip(plotting_ufuns, plotting_negotiators)):
        if u is None:
            continue
        name = map_(neg)
        alphas = [colorizer(_[0]) for _ in trace_info]
        y = [u(_[0].offer) for _ in trace_info]
        reserved = None
        if show_reserved:
            r = u.reserved_value
            if r is not None and math.isfinite(r):
                reserved = [r] * len(y)
        if not ignore_none_offers:
            xx, aa = x, alphas
        else:
            good_indices = [j for j, _ in enumerate(y) if _ is not None]
            xx = [x[_] for _ in good_indices]
            aa = [alphas[_] for _ in good_indices]
            y = [y[_] for _ in good_indices]
        thecolor = colors[i % len(colors)]

        # Add line trace
        line_style = "solid" if neg == negotiator else "dot"
        line_width = 1 if neg == negotiator else 0.5
        _add_trace_safe(
            fig,
            go.Scatter(
                x=xx,
                y=y,
                mode="lines+markers"
                if len(trace_info) < ignore_markers_limit and neg == negotiator
                else "lines",
                name="Utility" if simple_offers_view else name,
                line=dict(color=thecolor, width=line_width, dash=line_style),
                marker=dict(
                    symbol=markers[i % len(markers)],
                    size=6,
                    color=[color_to_rgba(thecolor, a) for a in aa],
                ),
                legendgroup=f"group_{i}",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )

        if mark_offers_view and neg == negotiator:
            mark_map = dict(
                errors=dict(marker="x", size=15, color="red"),
                agreement=dict(marker="star", size=15, color="black"),
                timedout=dict(marker="circle", size=15, color="black"),
            )
            for state, plotinfo in mark_map.items():
                elements = [
                    (_[1], u(_[0].offer)) for _ in trace_info if _[0].state == state
                ]
                if not elements:
                    continue
                elements = list(set(elements))
                _add_trace_safe(
                    fig,
                    go.Scatter(
                        x=[_[0] for _ in elements],
                        y=[float(_[1]) if _[1] is not None else 0 for _ in elements],
                        mode="markers",
                        marker=dict(
                            symbol=plotinfo["marker"],
                            size=plotinfo["size"],
                            color=plotinfo.get("color", thecolor),
                        ),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        if reserved:
            _add_trace_safe(
                fig,
                go.Scatter(
                    x=xx,
                    y=reserved,
                    mode="lines",
                    line=dict(
                        color=thecolor,
                        width=0.5,
                        dash="solid" if neg == negotiator else "dot",
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    # Update axes
    ylabel = (
        f"{map_(negotiator)} ({plotting_negotiators.index(negotiator)})"
        if negotiator in plotting_negotiators and not simple_offers_view
        else "utility"
    )
    _update_yaxes_safe(fig, title_text=ylabel, row=row, col=col)
    if ylimits is not None:
        _update_yaxes_safe(fig, range=ylimits, row=row, col=col)
    if show_x_label:
        _update_xaxes_safe(fig, title_text=xdim, row=row, col=col)

    return fig


def plot_2dutils(
    trace: list[TraceElement],
    plotting_ufuns: list[UtilityFunction] | tuple[UtilityFunction, ...],
    plotting_negotiators: list[str] | tuple[str, ...],
    offering_negotiators: list[str] | tuple[str, ...] | None = None,
    agreement: Outcome | None = None,
    outcome_space: OutcomeSpace | None = None,
    issues: list[Issue] | tuple[Issue, ...] | None = None,
    outcomes: list[Outcome] | tuple[Outcome, ...] | None = None,
    with_lines: bool = True,
    show_annotations: bool = True,
    show_agreement: bool = False,
    show_pareto_distance: bool = True,
    show_nash_distance: bool = True,
    show_kalai_distance: bool = True,
    show_ks_distance: bool = True,
    show_max_welfare_distance: bool = True,
    mark_pareto_points: bool = True,
    mark_all_outcomes: bool = True,
    mark_nash_points: bool = True,
    mark_kalai_points: bool = True,
    mark_ks_points: bool = True,
    mark_max_welfare_points: bool = True,
    show_max_relative_welfare_distance: bool = True,
    show_reserved: bool = True,
    show_total_time: bool = True,
    show_relative_time: bool = True,
    show_n_steps: bool = True,
    end_reason: str | None = None,
    extra_annotation: str | None = None,
    name_map: dict[str, str] | Callable[[str], str] | None = None,
    colors: list | None = None,
    markers: list[str] | None = None,
    colormap: str = DEFAULT_COLORMAP,
    fig: go.Figure | None = None,
    row: int = 1,
    col: int = 1,
    colorizer: Colorizer | None = None,
    fast: bool = False,
    backend: str = "plotly",
):
    """Plots negotiation trace in 2D utility space for two negotiators.

    Args:
        trace: List of trace elements recording all offers made during negotiation.
        plotting_ufuns: The two utility functions for the x and y axes.
        plotting_negotiators: IDs of the two negotiators being plotted.
        offering_negotiators: IDs of all negotiators who made offers to display.
        agreement: The final agreement outcome, if any.
        outcome_space: The space of possible outcomes.
        issues: List of negotiation issues (alternative to outcome_space).
        outcomes: Explicit list of outcomes (alternative to outcome_space).
        with_lines: Whether to connect consecutive offers with lines.
        show_annotations: Whether to show text labels on special points.
        show_agreement: Whether to include agreement details in annotation.
        show_pareto_distance: Whether to show distance from agreement to Pareto frontier.
        show_nash_distance: Whether to show distance from agreement to Nash point.
        show_kalai_distance: Whether to show distance from agreement to Kalai point.
        show_ks_distance: Whether to show distance from agreement to Kalai-Smorodinsky point.
        show_max_welfare_distance: Whether to show distance to max welfare point.
        mark_pareto_points: Whether to mark Pareto optimal outcomes.
        mark_all_outcomes: Whether to mark all possible outcomes.
        mark_nash_points: Whether to mark Nash bargaining solution.
        mark_kalai_points: Whether to mark Kalai proportional solution.
        mark_ks_points: Whether to mark Kalai-Smorodinsky solution.
        mark_max_welfare_points: Whether to mark maximum welfare points.
        show_max_relative_welfare_distance: Whether to show distance to max relative welfare.
        show_reserved: Whether to shade regions below reserved values.
        show_total_time: Whether to show total negotiation time in annotation.
        show_relative_time: Whether to show relative time (0-1) in annotation.
        show_n_steps: Whether to show number of steps in annotation.
        end_reason: Text describing why negotiation ended.
        extra_annotation: Additional text to include in annotation.
        name_map: Mapping from negotiator IDs to display names.
        colors: List of colors for each negotiator's markers.
        markers: List of marker symbols for each negotiator.
        colormap: Colormap name for auto-generating colors.
        fig: Existing figure to add traces to, or None to create new (type depends on backend).
        row: Row index in subplot grid (1-indexed) - only used for plotly backend.
        col: Column index in subplot grid (1-indexed) - only used for plotly backend.
        colorizer: Function to compute opacity for each trace element.
        fast: Whether to skip expensive calculations (Pareto, Nash, etc.).
        backend: Plotting backend to use. Either "matplotlib" or "plotly". Default is "plotly".

    Returns:
        A matplotlib Figure object if backend="matplotlib", or a plotly Figure object if backend="plotly".

    Raises:
        ValueError: If backend is not "matplotlib" or "plotly".
    """
    # Route to appropriate backend
    if backend == "matplotlib":
        return _plot_2dutils_matplotlib(
            trace=trace,
            plotting_ufuns=plotting_ufuns,
            plotting_negotiators=plotting_negotiators,
            offering_negotiators=offering_negotiators,
            agreement=agreement,
            outcome_space=outcome_space,
            issues=issues,
            outcomes=outcomes,
            with_lines=with_lines,
            show_annotations=show_annotations,
            show_agreement=show_agreement,
            show_pareto_distance=show_pareto_distance,
            show_nash_distance=show_nash_distance,
            show_kalai_distance=show_kalai_distance,
            show_ks_distance=show_ks_distance,
            show_max_welfare_distance=show_max_welfare_distance,
            mark_pareto_points=mark_pareto_points,
            mark_all_outcomes=mark_all_outcomes,
            mark_nash_points=mark_nash_points,
            mark_kalai_points=mark_kalai_points,
            mark_ks_points=mark_ks_points,
            mark_max_welfare_points=mark_max_welfare_points,
            show_max_relative_welfare_distance=show_max_relative_welfare_distance,
            show_reserved=show_reserved,
            show_total_time=show_total_time,
            show_relative_time=show_relative_time,
            show_n_steps=show_n_steps,
            end_reason=end_reason,
            extra_annotation=extra_annotation,
            name_map=name_map,
            colors=colors,
            markers=markers,
            colormap=colormap,
            fig=fig,
            colorizer=colorizer,
            fast=fast,
        )
    elif backend == "plotly":
        return _plot_2dutils_plotly(
            trace=trace,
            plotting_ufuns=plotting_ufuns,
            plotting_negotiators=plotting_negotiators,
            offering_negotiators=offering_negotiators,
            agreement=agreement,
            outcome_space=outcome_space,
            issues=issues,
            outcomes=outcomes,
            with_lines=with_lines,
            show_annotations=show_annotations,
            show_agreement=show_agreement,
            show_pareto_distance=show_pareto_distance,
            show_nash_distance=show_nash_distance,
            show_kalai_distance=show_kalai_distance,
            show_ks_distance=show_ks_distance,
            show_max_welfare_distance=show_max_welfare_distance,
            mark_pareto_points=mark_pareto_points,
            mark_all_outcomes=mark_all_outcomes,
            mark_nash_points=mark_nash_points,
            mark_kalai_points=mark_kalai_points,
            mark_ks_points=mark_ks_points,
            mark_max_welfare_points=mark_max_welfare_points,
            show_max_relative_welfare_distance=show_max_relative_welfare_distance,
            show_reserved=show_reserved,
            show_total_time=show_total_time,
            show_relative_time=show_relative_time,
            show_n_steps=show_n_steps,
            end_reason=end_reason,
            extra_annotation=extra_annotation,
            name_map=name_map,
            colors=colors,
            markers=markers,
            colormap=colormap,
            fig=fig,
            row=row,
            col=col,
            colorizer=colorizer,
            fast=fast,
        )
    else:
        raise ValueError(
            f"Invalid backend '{backend}'. Must be 'matplotlib' or 'plotly'."
        )


def _plot_2dutils_matplotlib(
    trace: list[TraceElement],
    plotting_ufuns: list[UtilityFunction] | tuple[UtilityFunction, ...],
    plotting_negotiators: list[str] | tuple[str, ...],
    offering_negotiators: list[str] | tuple[str, ...] | None = None,
    agreement: Outcome | None = None,
    outcome_space: OutcomeSpace | None = None,
    issues: list[Issue] | tuple[Issue, ...] | None = None,
    outcomes: list[Outcome] | tuple[Outcome, ...] | None = None,
    with_lines: bool = True,
    show_annotations: bool = True,
    show_agreement: bool = False,
    show_pareto_distance: bool = True,
    show_nash_distance: bool = True,
    show_kalai_distance: bool = True,
    show_ks_distance: bool = True,
    show_max_welfare_distance: bool = True,
    mark_pareto_points: bool = True,
    mark_all_outcomes: bool = True,
    mark_nash_points: bool = True,
    mark_kalai_points: bool = True,
    mark_ks_points: bool = True,
    mark_max_welfare_points: bool = True,
    show_max_relative_welfare_distance: bool = True,
    show_reserved: bool = True,
    show_total_time: bool = True,
    show_relative_time: bool = True,
    show_n_steps: bool = True,
    end_reason: str | None = None,
    extra_annotation: str | None = None,
    name_map: dict[str, str] | Callable[[str], str] | None = None,
    colors: list | None = None,
    markers: list[str] | None = None,
    colormap: str = DEFAULT_COLORMAP,
    fig=None,
    colorizer: Colorizer | None = None,
    fast: bool = False,
):
    """Matplotlib implementation of 2D utility space plotting."""
    import matplotlib.pyplot as plt
    from matplotlib import patches

    if not colorizer:
        colorizer = default_colorizer

    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        ax = fig.gca() if hasattr(fig, "gca") else fig.axes[0] if fig.axes else None
        if ax is None:
            ax = fig.add_subplot(111)

    map_ = make_callable(name_map)
    if not outcomes:
        outcome_space = os_or_none(outcome_space, issues, outcomes)
        if outcome_space:
            outcomes = list(outcome_space.enumerate_or_sample(10, 1000))
    if not outcomes:
        outcomes = list({_.offer for _ in trace})
    if not outcome_space:
        outcome_space = make_os(issues=issues, outcomes=outcomes)
    if not offering_negotiators:
        offering_negotiators = list({_.negotiator for _ in trace})

    utils = [tuple(f(o) for f in plotting_ufuns) for o in outcomes]
    colors_list, markers_list = make_colors_and_markers(
        colors, markers, len(offering_negotiators), colormap
    )

    # Convert plotly colors to matplotlib format
    colors_list = [plotly_to_mpl_color(c) for c in colors_list]

    # Convert plotly markers to matplotlib markers
    # This mapping ensures matplotlib plots look identical to plotly plots
    marker_map = {
        "circle": "o",
        "square": "s",
        "diamond": "D",
        "cross": "+",
        "x": "x",
        "triangle-up": "^",
        "triangle-down": "v",
        "triangle-left": "<",
        "triangle-right": ">",
        "star": "*",
        "pentagon": "p",
        "hexagon": "h",
        "hourglass": "d",  # Using thin_diamond as approximation
        "bowtie": "d",  # Using thin_diamond as approximation
    }

    agreement_utility = tuple(u(agreement) for u in plotting_ufuns)
    unknown_agreement_utility = None in agreement_utility
    if unknown_agreement_utility:
        show_pareto_distance = show_nash_distance = False

    # Plot all outcomes
    if mark_all_outcomes:
        ax.plot(
            [_[0] for _ in utils],
            [_[1] for _ in utils],
            "o",
            color="gray",
            markersize=OUTCOMES_SCALE,
            alpha=0.3,
            label="Outcomes",
        )

    agent_names = [map_(_) for _ in plotting_negotiators]
    if fast:
        frontier, frontier_outcome = [], []
        nash_pts = []
        kalai_pts = []
        ks_pts = []
        mwelfare_pts = []
        mrwelfare_pts = []
    else:
        frontier, frontier_outcome = pareto_frontier(
            ufuns=plotting_ufuns,
            issues=outcome_space.issues,  # type: ignore
            outcomes=outcomes if not issues else None,  # type: ignore
            sort_by_welfare=True,
        )
        frontier_indices = [
            i
            for i, _ in enumerate(frontier)
            if _[0] is not None
            and _[0] > float("-inf")
            and _[1] is not None
            and _[1] > float("-inf")
        ]
        frontier = [frontier[i] for i in frontier_indices]
        frontier_outcome = [frontier_outcome[i] for i in frontier_indices]

        nash_pts = nash_points(plotting_ufuns, frontier, outcome_space=outcome_space)
        kalai_pts = kalai_points(plotting_ufuns, frontier, outcome_space=outcome_space)
        ks_pts = ks_points(plotting_ufuns, frontier, outcome_space=outcome_space)
        mwelfare_pts = max_welfare_points(
            plotting_ufuns, frontier, outcome_space=outcome_space
        )
        mrwelfare_pts = max_relative_welfare_points(
            plotting_ufuns, frontier, outcome_space=outcome_space
        )

    if not nash_pts:
        show_nash_distance = False
    if not kalai_pts:
        show_kalai_distance = False
    if not ks_pts:
        show_ks_distance = False
    if not mwelfare_pts:
        show_max_welfare_distance = False
    if not mrwelfare_pts:
        show_max_relative_welfare_distance = False

    pareto_distance = float("inf")
    nash_distance, kalai_distance = float("inf"), float("inf")
    ks_distance = float("inf")
    max_welfare_distance, max_relative_welfare_distance = float("inf"), float("inf")

    # Plot Pareto frontier
    if mark_pareto_points and frontier:
        f1, f2 = [_[0] for _ in frontier], [_[1] for _ in frontier]
        ax.plot(
            f1,
            f2,
            "o",
            color=(
                238 / 255,
                232 / 255,
                170 / 255,
            ),  # rgb(238,232,170) - matches plotly
            markersize=PARETO_SCALE,
            alpha=PARETO_ALPHA,
            label="Pareto",
        )

    cu = agreement_utility
    if not unknown_agreement_utility and not fast:
        for nash, _ in nash_pts:
            nash_distance = min(
                nash_distance, ((nash[0] - cu[0]) ** 2 + (nash[1] - cu[1]) ** 2) ** 0.5
            )
        for kalai, _ in kalai_pts:
            kalai_distance = min(
                kalai_distance,
                ((kalai[0] - cu[0]) ** 2 + (kalai[1] - cu[1]) ** 2) ** 0.5,
            )
        for ks, _ in ks_pts:
            ks_distance = min(
                ks_distance, ((ks[0] - cu[0]) ** 2 + (ks[1] - cu[1]) ** 2) ** 0.5
            )
        for pt, _ in mwelfare_pts:
            max_welfare_distance = min(
                max_welfare_distance,
                ((pt[0] - cu[0]) ** 2 + (pt[1] - cu[1]) ** 2) ** 0.5,
            )
        for pt, _ in mrwelfare_pts:
            max_relative_welfare_distance = min(
                max_relative_welfare_distance,
                ((pt[0] - cu[0]) ** 2 + (pt[1] - cu[1]) ** 2) ** 0.5,
            )
        for pu in frontier:
            dist = ((pu[0] - cu[0]) ** 2 + (pu[1] - cu[1]) ** 2) ** 0.5
            if dist < pareto_distance:
                pareto_distance = dist

    if trace:
        n_steps = trace[-1].step + 1
        relative_time = trace[-1].relative_time
        total_time = trace[-1].time
    else:
        n_steps = relative_time = total_time = 0

    # Build annotation text
    txt_lines = []
    if show_agreement:
        txt_lines.append(f"Agreement:{agreement}")
    if not fast and show_pareto_distance and agreement is not None:
        txt_lines.append(f"Pareto-distance={pareto_distance:5.2f}")
    if not fast and show_nash_distance and agreement is not None:
        txt_lines.append(f"Nash-distance={nash_distance:5.2f}")
    if not fast and show_kalai_distance and agreement is not None:
        txt_lines.append(f"Kalai-distance={kalai_distance:5.2f}")
    if not fast and show_ks_distance and agreement is not None:
        txt_lines.append(f"KS-distance={ks_distance:5.2f}")
    if not fast and show_max_welfare_distance and agreement is not None:
        txt_lines.append(f"MaxWelfare-distance={max_welfare_distance:5.2f}")
    if not fast and show_max_relative_welfare_distance and agreement is not None:
        txt_lines.append(
            f"MaxRelativeWelfare-distance={max_relative_welfare_distance:5.2f}"
        )
    if show_relative_time and relative_time:
        txt_lines.append(f"Relative Time={relative_time:5.2f}")
    if show_total_time:
        txt_lines.append(f"Total Time={humanize_time(total_time, show_ms=True)}")
    if show_n_steps:
        txt_lines.append(f"N. Steps={n_steps}")
    if end_reason:
        txt_lines.append(f"{end_reason}")
    if extra_annotation:
        txt_lines.append(f"{extra_annotation}")

    if txt_lines and show_annotations:
        ax.text(
            0.02,
            0.02,
            "\n".join(txt_lines),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Draw reserved value regions
    if show_reserved:
        ranges = [
            plotting_ufuns[_].minmax(outcome_space=outcome_space)
            for _ in range(len(plotting_ufuns))
        ]
        for i, (mn, mx) in enumerate(ranges):
            if any(_ is None or not math.isfinite(_) for _ in (mn, mx)):
                x_vals = []
                for a, neg in enumerate(offering_negotiators):
                    negtrace = [_ for _ in trace if _.negotiator == neg]
                    x_vals += [plotting_ufuns[i](_.offer) for _ in negtrace]
                if x_vals:
                    ranges[i] = (min(x_vals), max(x_vals))
                else:
                    ranges[i] = (0, 1)

        for i, (mn, mx) in enumerate(ranges):
            r = plotting_ufuns[i].reserved_value
            if r is None or not math.isfinite(r):
                r = mn
            if i == 0:
                x0, x1 = r, mx
                y0, y1 = ranges[1 - i][0], ranges[1 - i][1]
            else:
                x0, x1 = ranges[1 - i][0], ranges[1 - i][1]
                y0, y1 = r, mx

            # Convert hex/rgb to matplotlib color
            color = colors_list[i % len(colors_list)]
            rect = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=0,
                facecolor=color,
                alpha=RESERVED_ALPHA,
                zorder=0,
            )
            ax.add_patch(rect)

    # Plot offers from each negotiator
    for a, neg in enumerate(offering_negotiators):
        negtrace = [_ for _ in trace if _.negotiator == neg]
        x = [plotting_ufuns[0](_.offer) for _ in negtrace]
        y = [plotting_ufuns[1](_.offer) for _ in negtrace]
        alphas = [colorizer(_) for _ in negtrace]

        color = colors_list[a % len(colors_list)]
        marker = marker_map.get(
            markers_list[a % len(markers_list)], markers_list[a % len(markers_list)]
        )

        # Plot markers with varying alpha (use plot for speed)
        for xi, yi, alpha in zip(x, y, alphas):
            ax.plot(
                [xi],
                [yi],
                marker=marker,
                color=color,
                markersize=8,
                alpha=PROPOSALS_ALPHA * alpha,
                linestyle="",
                zorder=2,
            )

        # Plot lines
        if with_lines and len(x) > 1:
            ax.plot(x, y, color=color, linestyle=":", linewidth=1, alpha=0.5, zorder=1)

        # Add to legend (just once per negotiator)
        ax.plot(
            [],
            [],
            marker=marker,
            color=color,
            markersize=8,
            linestyle="",
            label=f"{map_(neg)}",
        )

    # Plot special points
    if not fast:
        if mwelfare_pts and mark_max_welfare_points:
            ax.plot(
                [mwelfare[0] for mwelfare, _ in mwelfare_pts],
                [mwelfare[1] for mwelfare, _ in mwelfare_pts],
                marker=marker_map.get(WELFARE_MARKER, ">"),
                color=WELFARE_COLOR,
                markersize=WELFARE_SCALE,
                alpha=WELFARE_ALPHA,
                linestyle="",
                label="Max Welfare Points",
                zorder=3,
            )
            if show_annotations:
                for mwelfare, _ in mwelfare_pts:
                    ax.annotate(
                        "Max Welfare",
                        (mwelfare[0], mwelfare[1]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )

        if kalai_pts and mark_kalai_points:
            ax.plot(
                [kalai[0] for kalai, _ in kalai_pts],
                [kalai[1] for kalai, _ in kalai_pts],
                marker=marker_map.get(KALAI_MARKER, "v"),
                color=KALAI_COLOR,
                markersize=KALAI_SCALE,
                alpha=KALAI_ALPHA,
                linestyle="",
                label="Kalai Point",
                zorder=3,
            )
            if show_annotations:
                for kalai, _ in kalai_pts:
                    ax.annotate(
                        "Kalai",
                        (kalai[0], kalai[1]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )

        if ks_pts and mark_ks_points:
            ax.plot(
                [ks[0] for ks, _ in ks_pts],
                [ks[1] for ks, _ in ks_pts],
                marker=marker_map.get(KS_MARKER, "^"),
                color=KS_COLOR,
                markersize=KS_SCALE,
                alpha=KS_ALPHA,
                linestyle="",
                label="KS Point",
                zorder=3,
            )
            if show_annotations:
                for ks, _ in ks_pts:
                    ax.annotate(
                        "KS",
                        (ks[0], ks[1]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )

        if nash_pts and mark_nash_points:
            ax.plot(
                [nash[0] for nash, _ in nash_pts],
                [nash[1] for nash, _ in nash_pts],
                marker=marker_map.get(NASH_MARKER, "<"),
                color=NASH_COLOR,
                markersize=NASH_SCALE,
                alpha=NASH_ALPHA,
                linestyle="",
                label="Nash Point",
                zorder=3,
            )
            if show_annotations:
                for nash, _ in nash_pts:
                    ax.annotate(
                        "Nash",
                        (nash[0], nash[1]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )

    # Plot agreement
    if agreement is not None:
        ax.plot(
            [plotting_ufuns[0](agreement)],
            [plotting_ufuns[1](agreement)],
            marker="*",  # star marker - matches plotly
            color="black",
            markersize=AGREEMENT_SCALE,
            alpha=AGREEMENT_ALPHA,
            linestyle="",
            label="Agreement",
            zorder=4,
        )
        if show_annotations:
            ax.annotate(
                "Agreement",
                (plotting_ufuns[0](agreement), plotting_ufuns[1](agreement)),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    # Set labels and legend
    ax.set_xlabel(f"{agent_names[0]}(0) utility")
    ax.set_ylabel(f"{agent_names[1]}(1) utility")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig


def _plot_2dutils_plotly(
    trace: list[TraceElement],
    plotting_ufuns: list[UtilityFunction] | tuple[UtilityFunction, ...],
    plotting_negotiators: list[str] | tuple[str, ...],
    offering_negotiators: list[str] | tuple[str, ...] | None = None,
    agreement: Outcome | None = None,
    outcome_space: OutcomeSpace | None = None,
    issues: list[Issue] | tuple[Issue, ...] | None = None,
    outcomes: list[Outcome] | tuple[Outcome, ...] | None = None,
    with_lines: bool = True,
    show_annotations: bool = True,
    show_agreement: bool = False,
    show_pareto_distance: bool = True,
    show_nash_distance: bool = True,
    show_kalai_distance: bool = True,
    show_ks_distance: bool = True,
    show_max_welfare_distance: bool = True,
    mark_pareto_points: bool = True,
    mark_all_outcomes: bool = True,
    mark_nash_points: bool = True,
    mark_kalai_points: bool = True,
    mark_ks_points: bool = True,
    mark_max_welfare_points: bool = True,
    show_max_relative_welfare_distance: bool = True,
    show_reserved: bool = True,
    show_total_time: bool = True,
    show_relative_time: bool = True,
    show_n_steps: bool = True,
    end_reason: str | None = None,
    extra_annotation: str | None = None,
    name_map: dict[str, str] | Callable[[str], str] | None = None,
    colors: list | None = None,
    markers: list[str] | None = None,
    colormap: str = DEFAULT_COLORMAP,
    fig: go.Figure | None = None,
    row: int = 1,
    col: int = 1,
    colorizer: Colorizer | None = None,
    fast: bool = False,
):
    """Plotly implementation of 2D utility space plotting."""
    if not colorizer:
        colorizer = default_colorizer

    if fig is None:
        fig = go.Figure()

    map_ = make_callable(name_map)
    if not outcomes:
        outcome_space = os_or_none(outcome_space, issues, outcomes)
        if outcome_space:
            outcomes = list(outcome_space.enumerate_or_sample(10, 1000))
    if not outcomes:
        outcomes = list({_.offer for _ in trace})
    if not outcome_space:
        outcome_space = make_os(issues=issues, outcomes=outcomes)
    if not offering_negotiators:
        offering_negotiators = list({_.negotiator for _ in trace})

    utils = [tuple(f(o) for f in plotting_ufuns) for o in outcomes]
    colors, markers = make_colors_and_markers(
        colors, markers, len(offering_negotiators), colormap
    )

    agreement_utility = tuple(u(agreement) for u in plotting_ufuns)
    unknown_agreement_utility = None in agreement_utility
    if unknown_agreement_utility:
        show_pareto_distance = show_nash_distance = False

    if mark_all_outcomes:
        _add_trace_safe(
            fig,
            go.Scatter(
                x=[_[0] for _ in utils],
                y=[_[1] for _ in utils],
                mode="markers",
                marker=dict(color="gray", symbol="circle", size=OUTCOMES_SCALE),
                name="Outcomes",
                showlegend=True,
            ),
            row=row,
            col=col,
        )

    agent_names = [map_(_) for _ in plotting_negotiators]
    if fast:
        frontier, frontier_outcome = [], []
        frontier_indices = []
        nash_pts = []
        kalai_pts = []
        ks_pts = []
        mwelfare_pts = []
        mrwelfare_pts = []
    else:
        frontier, frontier_outcome = pareto_frontier(
            ufuns=plotting_ufuns,
            issues=outcome_space.issues,  # type: ignore
            outcomes=outcomes if not issues else None,  # type: ignore
            sort_by_welfare=True,
        )
        frontier_indices = [
            i
            for i, _ in enumerate(frontier)
            if _[0] is not None
            and _[0] > float("-inf")
            and _[1] is not None
            and _[1] > float("-inf")
        ]
        frontier = [frontier[i] for i in frontier_indices]
        frontier_outcome = [frontier_outcome[i] for i in frontier_indices]

        nash_pts = nash_points(plotting_ufuns, frontier, outcome_space=outcome_space)
        kalai_pts = kalai_points(plotting_ufuns, frontier, outcome_space=outcome_space)
        ks_pts = ks_points(plotting_ufuns, frontier, outcome_space=outcome_space)
        mwelfare_pts = max_welfare_points(
            plotting_ufuns, frontier, outcome_space=outcome_space
        )
        mrwelfare_pts = max_relative_welfare_points(
            plotting_ufuns, frontier, outcome_space=outcome_space
        )

    if not nash_pts:
        show_nash_distance = False
    if not kalai_pts:
        show_kalai_distance = False
    if not ks_pts:
        show_ks_distance = False
    if not mwelfare_pts:
        show_max_welfare_distance = False
    if not mrwelfare_pts:
        show_max_relative_welfare_distance = False

    pareto_distance = float("inf")
    nash_distance, kalai_distance = float("inf"), float("inf")
    ks_distance = float("inf")
    max_welfare_distance, max_relative_welfare_distance = float("inf"), float("inf")

    f1, f2 = [_[0] for _ in frontier], [_[1] for _ in frontier]
    if mark_pareto_points:
        _add_trace_safe(
            fig,
            go.Scatter(
                x=f1,
                y=f2,
                mode="markers",
                marker=dict(
                    color="rgb(238,232,170)",
                    symbol="circle",
                    size=PARETO_SCALE,
                    opacity=PARETO_ALPHA,
                ),
                name="Pareto",
                showlegend=True,
            ),
            row=row,
            col=col,
        )

    cu = agreement_utility
    if not unknown_agreement_utility and not fast:
        for nash, _ in nash_pts:
            nash_distance = min(
                nash_distance,
                math.sqrt((nash[0] - cu[0]) ** 2 + (nash[1] - cu[1]) ** 2),
            )
        for kalai, _ in kalai_pts:
            kalai_distance = min(
                kalai_distance,
                math.sqrt((kalai[0] - cu[0]) ** 2 + (kalai[1] - cu[1]) ** 2),
            )
        for ks, _ in ks_pts:
            ks_distance = min(
                ks_distance, math.sqrt((ks[0] - cu[0]) ** 2 + (ks[1] - cu[1]) ** 2)
            )
        for pt, _ in mwelfare_pts:
            max_welfare_distance = min(
                max_welfare_distance,
                math.sqrt((pt[0] - cu[0]) ** 2 + (pt[1] - cu[1]) ** 2),
            )
        for pt, _ in mrwelfare_pts:
            max_relative_welfare_distance = min(
                max_relative_welfare_distance,
                math.sqrt((pt[0] - cu[0]) ** 2 + (pt[1] - cu[1]) ** 2),
            )
        for pu in frontier:
            dist = math.sqrt((pu[0] - cu[0]) ** 2 + (pu[1] - cu[1]) ** 2)
            if dist < pareto_distance:
                pareto_distance = dist

    if trace:
        n_steps = trace[-1].step + 1
        relative_time = trace[-1].relative_time
        total_time = trace[-1].time
    else:
        n_steps = relative_time = total_time = 0

    # Build annotation text
    txt_lines = []
    if show_agreement:
        txt_lines.append(f"Agreement:{agreement}")
    if not fast and show_pareto_distance and agreement is not None:
        txt_lines.append(f"Pareto-distance={pareto_distance:5.2f}")
    if not fast and show_nash_distance and agreement is not None:
        txt_lines.append(f"Nash-distance={nash_distance:5.2f}")
    if not fast and show_kalai_distance and agreement is not None:
        txt_lines.append(f"Kalai-distance={kalai_distance:5.2f}")
    if not fast and show_ks_distance and agreement is not None:
        txt_lines.append(f"KS-distance={ks_distance:5.2f}")
    if not fast and show_max_welfare_distance and agreement is not None:
        txt_lines.append(f"MaxWelfare-distance={max_welfare_distance:5.2f}")
    if not fast and show_max_relative_welfare_distance and agreement is not None:
        txt_lines.append(
            f"MaxRelativeWelfare-distance={max_relative_welfare_distance:5.2f}"
        )
    if show_relative_time and relative_time:
        txt_lines.append(f"Relative Time={relative_time:5.2f}")
    if show_total_time:
        txt_lines.append(f"Total Time={humanize_time(total_time, show_ms=True)}")
    if show_n_steps:
        txt_lines.append(f"N. Steps={n_steps}")
    if end_reason:
        txt_lines.append(f"{end_reason}")
    if extra_annotation:
        txt_lines.append(f"{extra_annotation}")

    if txt_lines:
        _add_annotation_safe(
            fig,
            x=0.05,
            y=0.05,
            xref="x domain",
            yref="y domain",
            text="<br>".join(txt_lines),
            showarrow=False,
            font=dict(size=10, color="black"),
            align="left",
            xanchor="left",
            yanchor="bottom",
            row=row,
            col=col,
        )

    # Draw reserved value regions
    if show_reserved:
        ranges = [
            plotting_ufuns[_].minmax(outcome_space=outcome_space)
            for _ in range(len(plotting_ufuns))
        ]
        for i, (mn, mx) in enumerate(ranges):
            if any(_ is None or not math.isfinite(_) for _ in (mn, mx)):
                x_vals = []
                for a, neg in enumerate(offering_negotiators):
                    negtrace = [_ for _ in trace if _.negotiator == neg]
                    x_vals += [plotting_ufuns[i](_.offer) for _ in negtrace]
                if x_vals:
                    ranges[i] = (min(x_vals), max(x_vals))
                else:
                    ranges[i] = (0, 1)

        for i, (mn, mx) in enumerate(ranges):
            r = plotting_ufuns[i].reserved_value
            if r is None or not math.isfinite(r):
                r = mn
            if i == 0:
                x0, x1 = r, mx
                y0, y1 = ranges[1 - i][0], ranges[1 - i][1]
            else:
                x0, x1 = ranges[1 - i][0], ranges[1 - i][1]
                y0, y1 = r, mx

            _add_shape_safe(
                fig,
                type="rect",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                fillcolor=color_to_rgba(colors[i % len(colors)], RESERVED_ALPHA),
                line=dict(width=0),
                layer="below",
                row=row,
                col=col,
            )

    # Plot offers from each negotiator
    for a, neg in enumerate(offering_negotiators):
        negtrace = [_ for _ in trace if _.negotiator == neg]
        x = [plotting_ufuns[0](_.offer) for _ in negtrace]
        y = [plotting_ufuns[1](_.offer) for _ in negtrace]
        alphas = [colorizer(_) for _ in negtrace]

        marker_colors = [
            color_to_rgba(colors[a % len(colors)], PROPOSALS_ALPHA * alpha)
            for alpha in alphas
        ]

        mode = "lines+markers" if with_lines else "markers"
        _add_trace_safe(
            fig,
            go.Scatter(
                x=x,
                y=y,
                mode=mode,
                name=f"{map_(neg)}",
                marker=dict(
                    symbol=markers[a % len(markers)], size=8, color=marker_colors
                ),
                line=dict(color=colors[a % len(colors)], width=1, dash="dot")
                if with_lines
                else None,
                showlegend=True,
            ),
            row=row,
            col=col,
        )

    # Plot special points
    if not fast:
        if mwelfare_pts and mark_max_welfare_points:
            _add_trace_safe(
                fig,
                go.Scatter(
                    x=[mwelfare[0] for mwelfare, _ in mwelfare_pts],
                    y=[mwelfare[1] for mwelfare, _ in mwelfare_pts],
                    mode="markers+text" if show_annotations else "markers",
                    marker=dict(
                        color=WELFARE_COLOR,
                        symbol=WELFARE_MARKER,
                        size=WELFARE_SCALE,
                        opacity=WELFARE_ALPHA,
                    ),
                    text=["Max Welfare Point"] * len(mwelfare_pts)
                    if show_annotations
                    else None,
                    textposition="top right",
                    name="Max Welfare Points",
                    showlegend=True,
                ),
                row=row,
                col=col,
            )

        if kalai_pts and mark_kalai_points:
            _add_trace_safe(
                fig,
                go.Scatter(
                    x=[kalai[0] for kalai, _ in kalai_pts],
                    y=[kalai[1] for kalai, _ in kalai_pts],
                    mode="markers+text" if show_annotations else "markers",
                    marker=dict(
                        color=KALAI_COLOR,
                        symbol=KALAI_MARKER,
                        size=KALAI_SCALE,
                        opacity=KALAI_ALPHA,
                    ),
                    text=["Kalai Point"] * len(kalai_pts) if show_annotations else None,
                    textposition="top right",
                    name="Kalai Point",
                    showlegend=True,
                ),
                row=row,
                col=col,
            )

        if ks_pts and mark_ks_points:
            _add_trace_safe(
                fig,
                go.Scatter(
                    x=[ks[0] for ks, _ in ks_pts],
                    y=[ks[1] for ks, _ in ks_pts],
                    mode="markers+text" if show_annotations else "markers",
                    marker=dict(
                        color=KS_COLOR,
                        symbol=KS_MARKER,
                        size=KS_SCALE,
                        opacity=KS_ALPHA,
                    ),
                    text=["KS Point"] * len(ks_pts) if show_annotations else None,
                    textposition="top right",
                    name="KS Point",
                    showlegend=True,
                ),
                row=row,
                col=col,
            )

        if nash_pts and mark_nash_points:
            _add_trace_safe(
                fig,
                go.Scatter(
                    x=[nash[0] for nash, _ in nash_pts],
                    y=[nash[1] for nash, _ in nash_pts],
                    mode="markers+text" if show_annotations else "markers",
                    marker=dict(
                        color=NASH_COLOR,
                        symbol=NASH_MARKER,
                        size=NASH_SCALE,
                        opacity=NASH_ALPHA,
                    ),
                    text=["Nash Point"] * len(nash_pts) if show_annotations else None,
                    textposition="top right",
                    name="Nash Point",
                    showlegend=True,
                ),
                row=row,
                col=col,
            )

    # Plot agreement
    if agreement is not None:
        _add_trace_safe(
            fig,
            go.Scatter(
                x=[plotting_ufuns[0](agreement)],
                y=[plotting_ufuns[1](agreement)],
                mode="markers+text" if show_annotations else "markers",
                marker=dict(
                    color="black",
                    symbol="star",
                    size=AGREEMENT_SCALE,
                    opacity=AGREEMENT_ALPHA,
                ),
                text=["Agreement"] if show_annotations else None,
                textposition="top right",
                name="Agreement",
                showlegend=True,
            ),
            row=row,
            col=col,
        )

    # Update axes
    _update_xaxes_safe(fig, title_text=agent_names[0] + "(0) utility", row=row, col=col)
    _update_yaxes_safe(fig, title_text=agent_names[1] + "(1) utility", row=row, col=col)

    return fig


def plot_offline_run(
    trace: list[TraceElement],
    ids: list[str],
    ufuns: list[BaseUtilityFunction] | tuple[BaseUtilityFunction, ...],
    agreement: Outcome | None,
    timedout: bool,
    broken: bool,
    has_error: bool,
    errstr: str = "",
    names: list[str] | None = None,
    *,
    negotiators: tuple[int, int] | tuple[str, str] | None = (0, 1),
    save_fig: bool = False,
    path: str | None = None,
    fig_name: str | None = None,
    image_format: str = DEFAULT_IMAGE_FORMAT,
    ignore_none_offers: bool = True,
    with_lines: bool = True,
    show_agreement: bool = False,
    show_pareto_distance: bool = True,
    show_nash_distance: bool = True,
    show_kalai_distance: bool = True,
    show_ks_distance: bool = True,
    show_max_welfare_distance: bool = True,
    show_max_relative_welfare_distance: bool = False,
    show_end_reason: bool = True,
    show_annotations: bool = False,
    show_reserved: bool = True,
    show_total_time: bool = True,
    show_relative_time: bool = True,
    show_n_steps: bool = True,
    colors: list | None = None,
    markers: list[str] | None = None,
    colormap: str = DEFAULT_COLORMAP,
    ylimits: tuple[float, float] | None = None,
    common_legend: bool = True,
    extra_annotation: str = "",
    xdim: str = "relative_time",
    colorizer: Colorizer | None = None,
    only2d: bool = False,
    no2d: bool = False,
    fast: bool = False,
    simple_offers_view: bool = False,
    mark_offers_view: bool = True,
    mark_pareto_points: bool = True,
    mark_all_outcomes: bool = True,
    mark_nash_points: bool = True,
    mark_kalai_points: bool = True,
    mark_ks_points: bool = True,
    mark_max_welfare_points: bool = True,
    show: bool = True,
):
    """Plots a negotiation run from saved trace data (without a live mechanism).

    Args:
        trace: List of trace elements recording all offers made during negotiation.
        ids: List of negotiator IDs in the order they participated.
        ufuns: List of utility functions corresponding to each negotiator.
        agreement: The final agreement outcome, or None if no agreement.
        timedout: Whether the negotiation ended due to timeout.
        broken: Whether the negotiation was broken/terminated early.
        has_error: Whether an error occurred during negotiation.
        errstr: Error message string if has_error is True.
        names: Display names for negotiators (defaults to IDs if None).
        save_fig: If True, save the figure to disk.
        path: Directory path where to save the figure (if save_fig is True).
        fig_name: Filename for the saved figure. If provided with an extension (e.g., "plot.png"),
                 that extension is used. If provided without an extension or None, the image_format
                 parameter determines the extension.
        image_format: Image format to use when auto-generating filenames (default: webp).
                     Supported formats: webp, png, jpg, jpeg, svg, pdf. Only used if fig_name
                     is None or doesn't have an extension.
    """
    if names is None:
        names = [_ for _ in ids]

    assert not (no2d and only2d), "Cannot specify no2d and only2d together"

    if negotiators is None:
        negotiators = (0, 1)
    if len(negotiators) != 2:
        raise ValueError(
            "Cannot plot the 2D plot for the mechanism run without knowing two plotting negotiators"
        )
    plotting_negotiators = []
    plotting_ufuns = []
    for n in negotiators:
        if isinstance(n, int):
            i = n
        else:
            i = ids.index(n)
            if i is None:
                raise ValueError(f"Cannot find negotiator with ID {n}")
        plotting_negotiators.append(ids[i])
        plotting_ufuns.append(ufuns[i])

    name_map = dict(zip(ids, names))
    colors, markers = make_colors_and_markers(colors, markers, len(ids), colormap)

    # Create figure with subplots
    if only2d:
        fig = go.Figure()
    elif no2d:
        fig = make_subplots(rows=len(ids), cols=1, shared_xaxes=True)
    else:
        # Create subplot layout: 2D plot on left, offer plots on right
        fig = make_subplots(
            rows=len(ids),
            cols=2,
            column_widths=[0.5, 0.5],
            specs=[
                [{"rowspan": len(ids)}, {}] if i == 0 else [None, {}]
                for i in range(len(ids))
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.05,
        )

    # Plot offer utilities
    if not only2d:
        all_ufuns = ufuns
        for a, neg in enumerate(ids):
            plot_offer_utilities(
                trace=trace,
                negotiator=neg,
                plotting_ufuns=[all_ufuns[a]]
                if simple_offers_view
                else list(all_ufuns),
                plotting_negotiators=[neg] if simple_offers_view else ids,
                fig=fig,
                row=a + 1,
                col=2 if not no2d else 1,
                name_map=name_map,
                colors=colors,
                markers=markers,
                ignore_none_offers=ignore_none_offers,
                ylimits=ylimits,
                show_legend=(not common_legend or a == 0) and not simple_offers_view,
                show_x_label=a == len(ids) - 1,
                show_reserved=show_reserved,
                xdim=xdim,
                colorizer=colorizer,
                first_color_index=a if simple_offers_view else 0,
                mark_offers_view=mark_offers_view,
            )

    # Plot 2D utilities
    if not no2d:
        if not show_end_reason:
            reason = None
        else:
            if timedout:
                reason = "Negotiation Timedout"
            elif agreement is not None:
                reason = "Negotiation Success"
            elif has_error:
                reason = f"Negotiation ERROR: {errstr}"
            elif agreement is not None:
                reason = "Agreement Reached"
            elif broken:
                reason = "Negotiation Ended"
            elif agreement is None:
                reason = "No Agreement"
            else:
                reason = "Unknown state!!"

        assert len(ufuns) and ufuns[0].outcome_space
        plot_2dutils(
            trace=trace,
            plotting_ufuns=plotting_ufuns,
            plotting_negotiators=plotting_negotiators,
            offering_negotiators=ids,
            outcome_space=ufuns[0].outcome_space,
            outcomes=list(ufuns[0].outcome_space.enumerate_or_sample(levels=10)),
            fig=fig,
            row=1,
            col=1,
            name_map=name_map,
            with_lines=with_lines,
            show_agreement=show_agreement,
            show_pareto_distance=show_pareto_distance,
            show_nash_distance=show_nash_distance,
            show_kalai_distance=show_kalai_distance,
            show_ks_distance=show_ks_distance,
            show_max_welfare_distance=show_max_welfare_distance,
            show_max_relative_welfare_distance=show_max_relative_welfare_distance,
            show_annotations=show_annotations,
            show_reserved=show_reserved,
            colors=colors,
            markers=markers,
            agreement=agreement,
            end_reason=reason,
            extra_annotation=extra_annotation,
            colorizer=colorizer,
            show_total_time=show_total_time,
            show_relative_time=show_relative_time,
            show_n_steps=show_n_steps,
            fast=fast,
            mark_pareto_points=mark_pareto_points,
            mark_all_outcomes=mark_all_outcomes,
            mark_nash_points=mark_nash_points,
            mark_kalai_points=mark_kalai_points,
            mark_ks_points=mark_ks_points,
            mark_max_welfare_points=mark_max_welfare_points,
        )

    # Update layout - use autosize for responsive sizing in viewers
    fig.update_layout(
        autosize=True,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    if save_fig:
        if fig_name is None:
            fig_name = str(uuid.uuid4()) + f".{image_format}"
        elif not pathlib.Path(fig_name).suffix:
            # User provided name without extension, add image_format
            fig_name = f"{fig_name}.{image_format}"
        # else: User provided name with extension, use as-is

        if path is None:
            path_ = pathlib.Path().absolute()
        else:
            path_ = pathlib.Path(path)
        path_.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(path_ / fig_name))

    if show:
        fig.show()
        return None

    return fig


def plot_mechanism_run(
    mechanism: PlottableMechanism,
    *,
    negotiators: tuple[int, int] | tuple[str, str] | None = (0, 1),
    save_fig: bool = False,
    path: str | None = None,
    fig_name: str | None = None,
    image_format: str = DEFAULT_IMAGE_FORMAT,
    ignore_none_offers: bool = True,
    with_lines: bool = True,
    show_agreement: bool = False,
    show_pareto_distance: bool = True,
    show_nash_distance: bool = True,
    show_kalai_distance: bool = True,
    show_ks_distance: bool = True,
    show_max_welfare_distance: bool = True,
    show_max_relative_welfare_distance: bool = False,
    show_end_reason: bool = True,
    show_annotations: bool = False,
    show_reserved: bool = True,
    show_total_time: bool = True,
    show_relative_time: bool = True,
    show_n_steps: bool = True,
    colors: list | None = None,
    markers: list[str] | None = None,
    colormap: str = DEFAULT_COLORMAP,
    ylimits: tuple[float, float] | None = None,
    common_legend: bool = True,
    extra_annotation: str = "",
    xdim: str = "relative_time",
    colorizer: Colorizer | None = None,
    only2d: bool = False,
    no2d: bool = False,
    fast: bool = False,
    simple_offers_view: bool = False,
    mark_offers_view: bool = True,
    mark_pareto_points: bool = True,
    mark_all_outcomes: bool = True,
    mark_nash_points: bool = True,
    mark_kalai_points: bool = True,
    mark_ks_points: bool = True,
    mark_max_welfare_points: bool = True,
    show: bool = True,
):
    """Plots a complete visualization of a negotiation mechanism run.

    Args:
        mechanism: The mechanism object containing negotiation state and history.
    """
    assert not (no2d and only2d), "Cannot specify no2d and only2d together"

    if negotiators is None:
        negotiators = (0, 1)
    if len(negotiators) != 2:
        raise ValueError(
            "Cannot plot the 2D plot for the mechanism run without knowing two plotting negotiators"
        )
    plotting_negotiators = []
    plotting_ufuns = []
    for n in negotiators:
        if isinstance(n, int):
            i = n
        else:
            i = mechanism.negotiator_index(n)
            if i is None:
                raise ValueError(f"Cannot find negotiator with ID {n}")
        plotting_negotiators.append(mechanism.negotiators[i].id)
        plotting_ufuns.append(mechanism.negotiators[i].ufun)

    name_map = dict(zip(mechanism.negotiator_ids, mechanism.negotiator_names))
    colors, markers = make_colors_and_markers(
        colors, markers, len(mechanism.negotiators), colormap
    )

    # Create figure with subplots
    n_negotiators = mechanism.nmi.n_negotiators
    if only2d:
        fig = go.Figure()
    elif no2d:
        fig = make_subplots(rows=n_negotiators, cols=1, shared_xaxes=True)
    else:
        # Create subplot layout: 2D plot on left, offer plots on right
        fig = make_subplots(
            rows=n_negotiators,
            cols=2,
            column_widths=[0.5, 0.5],
            specs=[
                [{"rowspan": n_negotiators}, {}] if i == 0 else [None, {}]
                for i in range(n_negotiators)
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.05,
        )

    # Plot offer utilities
    if not only2d:
        all_ufuns = [_.ufun for _ in mechanism.negotiators]
        for a, neg in enumerate(mechanism.negotiator_ids):
            # Only show legend in offer utilities if 2D plot is not shown (to avoid duplicate legends)
            # When both 2D and offer utilities are shown, the 2D plot provides the legend
            show_offer_legend = (
                no2d  # Only show legend when 2D plot is hidden
                and (
                    not common_legend or a == 0
                )  # Show for first negotiator when common_legend
                and not simple_offers_view
            )
            plot_offer_utilities(
                trace=mechanism.full_trace,
                negotiator=neg,
                plotting_ufuns=[all_ufuns[a]] if simple_offers_view else all_ufuns,
                plotting_negotiators=[neg]
                if simple_offers_view
                else mechanism.negotiator_ids,
                fig=fig,
                row=a + 1,
                col=2 if not no2d else 1,
                name_map=name_map,
                colors=colors,
                markers=markers,
                ignore_none_offers=ignore_none_offers,
                ylimits=ylimits,
                show_legend=show_offer_legend,
                show_x_label=a == len(mechanism.negotiator_ids) - 1,
                show_reserved=show_reserved,
                xdim=xdim,
                colorizer=colorizer,
                first_color_index=a if simple_offers_view else 0,
                mark_offers_view=mark_offers_view,
            )

    # Plot 2D utilities
    if not no2d:
        agreement = mechanism.agreement
        state = mechanism.state
        if not state.erred_negotiator:
            erredneg = errdetails = ""
        else:
            try:
                ids = mechanism.negotiator_ids
                names = mechanism.negotiator_names
                erredneg = names[ids.index(state.erred_negotiator)]
                errdetails = state.error_details.split("\n")[-1][-30:]
            except Exception:
                erredneg = errdetails = ""

        if not show_end_reason:
            reason = None
        else:
            if state.timedout:
                reason = "Negotiation Timedout"
            elif agreement is not None:
                reason = "Negotiation Success"
            elif state.has_error:
                reason = f"ERROR by {erredneg}: {errdetails}"
            elif agreement is not None:
                reason = "Agreement Reached"
            elif state.broken:
                reason = "Negotiation Ended"
            elif agreement is None:
                reason = "No Agreement"
            else:
                reason = "Unknown state!!"

        plot_2dutils(
            trace=mechanism.full_trace,
            plotting_ufuns=plotting_ufuns,
            plotting_negotiators=plotting_negotiators,
            offering_negotiators=mechanism.negotiator_ids,
            outcome_space=mechanism.outcome_space,
            outcomes=mechanism.discrete_outcomes(),
            fig=fig,
            row=1,
            col=1,
            name_map=name_map,
            with_lines=with_lines,
            show_agreement=show_agreement,
            show_pareto_distance=show_pareto_distance,
            show_nash_distance=show_nash_distance,
            show_kalai_distance=show_kalai_distance,
            show_ks_distance=show_ks_distance,
            show_max_welfare_distance=show_max_welfare_distance,
            show_max_relative_welfare_distance=show_max_relative_welfare_distance,
            show_annotations=show_annotations,
            show_reserved=show_reserved,
            colors=colors,
            markers=markers,
            agreement=mechanism.agreement,
            end_reason=reason,
            extra_annotation=extra_annotation,
            colorizer=colorizer,
            show_total_time=show_total_time,
            show_relative_time=show_relative_time,
            show_n_steps=show_n_steps,
            fast=fast,
            mark_pareto_points=mark_pareto_points,
            mark_all_outcomes=mark_all_outcomes,
            mark_nash_points=mark_nash_points,
            mark_kalai_points=mark_kalai_points,
            mark_ks_points=mark_ks_points,
            mark_max_welfare_points=mark_max_welfare_points,
        )

    # Update layout - use autosize for responsive sizing in viewers
    fig.update_layout(
        autosize=True,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    if save_fig:
        if fig_name is None:
            fig_name = str(uuid.uuid4()) + f".{image_format}"
        elif not pathlib.Path(fig_name).suffix:
            # User provided name without extension, add image_format
            fig_name = f"{fig_name}.{image_format}"
        # else: User provided name with extension, use as-is

        if path is None:
            path_ = pathlib.Path().absolute()
        else:
            path_ = pathlib.Path(path)
        path_.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(path_ / fig_name))

    if show:
        fig.show()
        return None
    return fig
