"""Module for util functionality."""

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
    """Opacity colorizer.

    Args:
        t: T.
        alpha: Alpha.
    """
    _ = t
    return alpha


def default_colorizer(t: TraceElement):
    """Default colorizer.

    Args:
        t: T.
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
        """Outcome space.

        Returns:
            OutcomeSpace: The result.
        """
        ...

    @property
    def negotiators(self) -> list[TNegotiator]:
        """Negotiators.

        Returns:
            list[TNegotiator]: The result.
        """
        ...

    @property
    def negotiator_ids(self) -> list[str]:
        """Negotiator ids.

        Returns:
            list[str]: The result.
        """
        ...

    @property
    def negotiator_names(self) -> list[str]:
        """Negotiator names.

        Returns:
            list[str]: The result.
        """
        ...

    @property
    def nmi(self) -> TNMI:
        """Nmi.

        Returns:
            TNMI: The result.
        """
        ...

    @property
    def agreement(self) -> Outcome | None:
        """Agreement.

        Returns:
            Outcome | None: The result.
        """
        ...

    @property
    def state(self) -> MechanismState:
        """State.

        Returns:
            MechanismState: The result.
        """
        ...

    @property
    def full_trace(self) -> list[TraceElement]:
        """Full trace.

        Returns:
            list[TraceElement]: The result.
        """
        ...

    def discrete_outcomes(self) -> list[Outcome]:
        """Discrete outcomes.

        Returns:
            list[Outcome]: The result.
        """
        ...

    def negotiator_index(self, source: str) -> int | None:
        """Negotiator index.

        Args:
            source: Source identifier.

        Returns:
            int | None: The result.
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
    """Make colors and markers.

    Args:
        colors: Colors.
        markers: Markers.
        n: Number of items.
        colormap: Colormap.
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
    """Plot offer utilities using plotly.

    Args:
        trace: Trace.
        negotiator: Negotiator.
        plotting_ufuns: Plotting ufuns.
        plotting_negotiators: Plotting negotiators.
        ignore_none_offers: Ignore none offers.
        name_map: Name map.
        colors: Colors.
        markers: Markers.
        colormap: Colormap.
        fig: Plotly figure (if None, creates new one).
        row: Row in subplot grid.
        col: Column in subplot grid.
        sharey: Share y axis.
        xdim: X dimension.
        ylimits: Y limits.
        show_legend: Show legend.
        show_x_label: Show x label.
        ignore_markers_limit: Ignore markers limit.
        show_reserved: Show reserved.
        colorizer: Colorizer.
        first_color_index: First color index.
        mark_offers_view: Mark offers view.
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
):
    """Plot 2D utilities using plotly.

    Args:
        trace: Trace.
        plotting_ufuns: Plotting ufuns.
        plotting_negotiators: Plotting negotiators.
        offering_negotiators: Offering negotiators.
        agreement: Agreement.
        outcome_space: Outcome space.
        issues: Issues.
        outcomes: Outcomes.
        with_lines: With lines.
        show_annotations: Show annotations.
        show_agreement: Show agreement.
        show_pareto_distance: Show pareto distance.
        show_nash_distance: Show nash distance.
        show_kalai_distance: Show kalai distance.
        show_ks_distance: Show ks distance.
        show_max_welfare_distance: Show max welfare distance.
        mark_pareto_points: Mark pareto points.
        mark_all_outcomes: Mark all outcomes.
        mark_nash_points: Mark nash points.
        mark_kalai_points: Mark kalai points.
        mark_ks_points: Mark ks points.
        mark_max_welfare_points: Mark max welfare points.
        show_max_relative_welfare_distance: Show max relative welfare distance.
        show_reserved: Show reserved.
        show_total_time: Show total time.
        show_relative_time: Show relative time.
        show_n_steps: Show n steps.
        end_reason: End reason.
        extra_annotation: Extra annotation.
        name_map: Name map.
        colors: Colors.
        markers: Markers.
        colormap: Colormap.
        fig: Plotly figure.
        row: Row in subplot.
        col: Column in subplot.
        colorizer: Colorizer.
        fast: Fast mode.
    """
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
    """Plot offline run using plotly.

    Args:
        trace: Trace.
        ids: Ids.
        ufuns: Ufuns.
        agreement: Agreement.
        timedout: Timedout.
        broken: Broken.
        has_error: Has error.
        errstr: Error string.
        names: Names.
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

    # Update layout
    fig.update_layout(
        width=1280 if not only2d and not no2d else None,
        height=480 if not only2d and not no2d else None,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    if save_fig:
        if fig_name is None:
            fig_name = str(uuid.uuid4()) + ".png"
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
    """Plot mechanism run using plotly.

    Args:
        mechanism: Mechanism.
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

    # Update layout
    fig.update_layout(
        width=1280 if not only2d and not no2d else None,
        height=480 if not only2d and not no2d else None,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    if save_fig:
        if fig_name is None:
            fig_name = str(uuid.uuid4()) + ".png"
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
