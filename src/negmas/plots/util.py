"""Module for util functionality."""

from __future__ import annotations

import math
import pathlib
import uuid
from typing import TYPE_CHECKING, Callable, Protocol, TypeVar, Generic

from matplotlib.markers import CARETDOWN, CARETLEFT, CARETRIGHT, CARETUP

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
    from matplotlib.axes import Axes

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


def scatter_with_transparency(x, y, color, alpha_arr, ax, **kwarg):
    """Scatter with transparency.

    Args:
        x: X.
        y: Y.
        color: Color.
        alpha_arr: Alpha arr.
        ax: Ax.
        **kwarg: Additional keyword arguments.
    """
    from matplotlib.colors import to_rgb  # , to_rgba

    r, g, b = to_rgb(color)
    # r, g, b, _ = to_rgba(color)
    color = [(r, g, b, alpha) for alpha in alpha_arr]
    ax.scatter(x, y, c=color, **kwarg)


def make_transparent(color, alpha):
    """Make transparent.

    Args:
        color: Color.
        alpha: Alpha.
    """
    if alpha is None:
        return color
    from matplotlib.colors import to_rgb  # , to_rgba

    r, g, b = to_rgb(color)
    # r, g, b, _ = to_rgba(color)
    return (r, g, b, alpha)


def plot_with_trancparency(
    x,
    y,
    alpha,
    color,
    marker,
    ax,
    label,
    with_lines=False,
    alpha_global=None,
    linewidth: float | int = 1,
    linestyle="solid",
):
    """Plot with trancparency.

    Args:
        x: X.
        y: Y.
        alpha: Alpha.
        color: Color.
        marker: Marker.
        ax: Ax.
        label: Label.
        with_lines: With lines.
        alpha_global: Alpha global.
        linewidth: Linewidth.
        linestyle: Linestyle.
    """
    if alpha_global is not None:
        alpha = [alpha_global * _ for _ in alpha]
    scatter_with_transparency(
        x, y, label=label, color=color, alpha_arr=alpha, ax=ax, marker=marker
    )
    if with_lines:
        ax.plot(
            x,
            y,
            color=make_transparent(color, alpha_global),
            marker=None,
            linewidth=linewidth,
            linestyle=linestyle,
        )


ALL_MARKERS = ["s", "o", "v", "^", "<", ">", "p", "P", "h", "H", "1", "2", "3", "4"]
PROPOSALS_ALPHA = 0.7
AGREEMENT_ALPHA = 0.9
PARETO_ALPHA = 0.4
NASH_ALPHA = 0.4
KALAI_ALPHA = 0.4
KS_ALPHA = 0.4
KALAI_MARKER = CARETDOWN
KS_MARKER = CARETUP
WELFARE_MARKER = CARETRIGHT
NASH_MARKER = CARETLEFT
KALAI_COLOR = "green"
KS_COLOR = "cyan"
WELFARE_COLOR = "blue"
NASH_COLOR = "brown"
RESERVED_ALPHA = 0.08
WELFARE_ALPHA = 0.6
NASH_ALPHA = 0.6
KALAI_ALPHA = 0.6
KS_ALPHA = 0.6
AGREEMENT_SCALE = 7
NASH_SCALE = 3
KALAI_SCALE = 3
KS_SCALE = 3
WELFARE_SCALE = 3
OUTCOMES_SCALE = 0.5
PARETO_SCALE = 1.0

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


DEFAULT_COLORMAP = "jet"


def get_cmap(n, name=DEFAULT_COLORMAP):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    import matplotlib.pyplot as plt

    return plt.cm.get_cmap(name, n)


def make_colors_and_markers(colors, markers, n: int, colormap=DEFAULT_COLORMAP):
    """Make colors and markers.

    Args:
        colors: Colors.
        markers: Markers.
        n: Number of items.
        colormap: Colormap.
    """
    if not colors:
        cmap = get_cmap(n, colormap)
        colors = [cmap(i) for i in range(n)]
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
    ax: Axes | None = None,  # type: ignore
    sharey=False,
    xdim: str = "relative_time",
    ylimits: tuple[float, float] | None = None,
    show_legend=True,
    show_x_label=True,
    ignore_markers_limit=200,
    show_reserved=True,
    colorizer: Colorizer | None = None,
    first_color_index: int = 0,
    mark_offers_view: bool = True,
):
    """Plot offer utilities.

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
        ax: Ax.
        sharey: Sharey.
        xdim: Xdim.
        ylimits: Ylimits.
        show_legend: Show legend.
        show_x_label: Show x label.
        ignore_markers_limit: Ignore markers limit.
        show_reserved: Show reserved.
        colorizer: Colorizer.
        first_color_index: First color index.
        mark_offers_view: Mark offers view.
    """
    import matplotlib.pyplot as plt

    if colorizer is None:
        colorizer = default_colorizer

    map_ = make_callable(name_map)
    if ax is None:
        _, ax = plt.subplots()  # type: ignore
    ax: Axes
    one_y = True
    axes = [ax] * len(plotting_negotiators)
    if not sharey and len(plotting_negotiators) == 2:
        axes = [ax, ax.twinx()]
        one_y = False

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
    # reorder to plot the negotiator last if it was in the plotting-negotiators
    # if negotiator in plotting_negotiators:
    #     indx = plotting_negotiators.index(negotiator)
    #     plotting_negotiators = (
    #         plotting_negotiators[:indx]
    #         + plotting_negotiators[indx + 1 :]
    #         + [plotting_negotiators[indx]]
    #     )
    #     axes = axes[:indx] + axes[indx + 1 :] + [axes[indx]]
    #     plotting_ufuns = (
    #         plotting_ufuns[:indx] + plotting_ufuns[indx + 1 :] + [plotting_ufuns[indx]]
    #     )
    #     markers = markers[:indx] + markers[indx + 1 :] + [markers[indx]]
    #     colors = colors[:indx] + colors[indx + 1 :] + [colors[indx]]
    for i, (u, neg, a) in enumerate(zip(plotting_ufuns, plotting_negotiators, axes)):
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
            good_indices = [i for i, _ in enumerate(y) if _ is not None]
            xx = [x[_] for _ in good_indices]
            aa = [alphas[_] for _ in good_indices]
            y = [y[_] for _ in good_indices]
        thecolor = colors[i % len(colors)]
        # a.plot(
        #     xx,
        #     y,
        #     label=f"{name} ({i})",
        #     color=thecolor,
        #     linestyle="solid" if neg == negotiator else "dotted",
        #     linewidth=2 if neg == negotiator else 1,
        #     marker=None,
        # )
        plot_with_trancparency(
            x=xx,
            y=y,
            alpha=aa,
            color=thecolor,
            marker=markers[i % len(markers)]
            if len(trace_info) < ignore_markers_limit and neg == negotiator
            else "",
            label="Utility" if simple_offers_view else name,
            ax=a,
            with_lines=True,
            linestyle="solid" if neg == negotiator else ":",
            linewidth=1 if neg == negotiator else 0.5,
        )
        if mark_offers_view and neg == negotiator:
            mark_map = dict(
                errors=dict(marker="x", s=110, color="red"),
                agreement=dict(marker="*", s=110, color="black"),
                timedout=dict(marker="o", s=110, color="black"),
            )
            for state, plotinfo in mark_map.items():
                elements = [
                    (_[1], u(_[0].offer)) for _ in trace_info if _[0].state == state
                ]
                if not elements:
                    continue
                elements = list(set(elements))
                a.scatter(
                    [_[0] for _ in elements],
                    [float(_[1]) for _ in elements],
                    color=plotinfo.get("color", thecolor),
                    marker=plotinfo["marker"],  # type: ignore
                    s=plotinfo["s"],
                )
        if reserved:
            a.plot(
                xx,
                reserved,
                # label=f"{name} ({i})",
                color=colors[i % len(colors)],
                linestyle="solid" if neg == negotiator else "dotted",
                linewidth=0.5,
            )
        if ylimits is not None:
            a.set_ylim(ylimits)
        a.set_ylabel(f"{name} ({i})" if not one_y or simple_offers_view else "utility")
        if show_legend and len(plotting_negotiators) == 2:
            a.legend(
                loc=f"upper {'left' if not i else 'right'}",
                bbox_to_anchor=(0.0, 1.4, 1.0, 0.12),
            )

    axes[0].set_title(f"{map_(negotiator)} Offers")
    if show_legend and len(plotting_negotiators) != 2:
        ax.legend(
            bbox_to_anchor=(0.0, 1.22, 1.0, 0.102),
            loc="upper center",
            ncol=2,
            mode="expand",
            borderaxespad=0.0,
            labelcolor=colors,
            # labelcolor="linecolor",
            # draggablebool=True,
        )
    if show_x_label:
        ax.set_xlabel(xdim)

    plt.tight_layout(w_pad=1.08, h_pad=1.12)


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
    show_total_time=True,
    show_relative_time=True,
    show_n_steps=True,
    end_reason: str | None = None,
    extra_annotation: str | None = None,
    name_map: dict[str, str] | Callable[[str], str] | None = None,
    colors: list | None = None,
    markers: list[str] | None = None,
    colormap: str = DEFAULT_COLORMAP,
    ax: Axes | None = None,  # type: ignore
    colorizer: Colorizer | None = None,
    fast: bool = False,
):
    """Plot 2dutils.

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
        ax: Ax.
        colorizer: Colorizer.
        fast: Fast.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if not colorizer:
        colorizer = default_colorizer

    if ax is None:
        _, ax = plt.subplots()  # type: ignore
    ax: Axes
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
    yrange = max(_[1] for _ in utils) - min(_[1] for _ in utils)
    colors, markers = make_colors_and_markers(
        colors, markers, len(offering_negotiators), colormap
    )

    agreement_utility = tuple(u(agreement) for u in plotting_ufuns)
    unknown_agreement_utility = None in agreement_utility
    if unknown_agreement_utility:
        show_pareto_distance = show_nash_distance = False
    default_marker_size = plt.rcParams.get("lines.markersize", 20) ** 2
    if mark_all_outcomes:
        ax.scatter(
            [_[0] for _ in utils],
            [_[1] for _ in utils],
            color="gray",
            marker=".",
            s=int(default_marker_size * OUTCOMES_SCALE),
        )
    agent_names = [map_(_) for _ in plotting_negotiators]
    if fast:
        frontier, frontier_outcome = [], []
        frontier_indices = []
        frontier_outcome = [frontier_outcome[i] for i in frontier_indices]
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
        ax.scatter(
            f1,
            f2,
            c=[(238 / 255.0, 232 / 255.0, 170 / 255.0)] * len(f1),
            marker="o",
            s=int(default_marker_size * PARETO_SCALE),
            alpha=PARETO_ALPHA,
            label="Pareto",
        )
    ax.set_xlabel(agent_names[0] + "(0) utility")
    ax.set_ylabel(agent_names[1] + "(1) utility")
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
    txt = ""
    if show_agreement:
        txt += f"Agreement:{agreement}\n"
    if not fast and show_pareto_distance and agreement is not None:
        txt += f"Pareto-distance={pareto_distance:5.2}\n"
    if not fast and show_nash_distance and agreement is not None:
        txt += f"Nash-distance={nash_distance:5.2}\n"
    if not fast and show_kalai_distance and agreement is not None:
        txt += f"Kalai-distance={kalai_distance:5.2}\n"
    if not fast and show_ks_distance and agreement is not None:
        txt += f"KS-distance={ks_distance:5.2}\n"
    if not fast and show_max_welfare_distance and agreement is not None:
        txt += f"MaxWelfare-distance={max_welfare_distance:5.2}\n"
    if not fast and show_max_relative_welfare_distance and agreement is not None:
        txt += f"MaxRelativeWelfare-distance={max_relative_welfare_distance:5.2}\n"
    if show_relative_time and relative_time:
        txt += f"Relative Time={relative_time:5.2}\n"
    if show_total_time:
        txt += f"Total Time={humanize_time(total_time, show_ms=True)}\n"
    if show_n_steps:
        txt += f"N. Steps={n_steps}\n"
    if end_reason:
        txt += f"{end_reason}\n"
    if extra_annotation:
        txt += f"{extra_annotation}\n"

    ax.text(
        0.05,
        0.05,
        txt,
        verticalalignment="bottom",
        transform=ax.transAxes,
        weight="bold",
    )

    if show_reserved:
        ranges = [
            plotting_ufuns[_].minmax(outcome_space=outcome_space)
            for _ in range(len(plotting_ufuns))
        ]
        for i, (mn, mx) in enumerate(ranges):
            if any(_ is None or not math.isfinite(_) for _ in (mn, mx)):
                x = []
                for a, neg in enumerate(offering_negotiators):
                    negtrace = [_ for _ in trace if _.negotiator == neg]
                    x += [plotting_ufuns[i](_.offer) for _ in negtrace]
                ranges[i] = (min(x), max(x))
        for i, (mn, mx) in enumerate(ranges):
            r = plotting_ufuns[i].reserved_value
            if r is None or not math.isfinite(r):
                r = mn
            if i == 0:
                pt = (r, ranges[1 - i][0])
                width = mx - r
                height = ranges[1 - i][1] - ranges[1 - i][0]
            else:
                pt = (ranges[1 - i][0], r)
                height = mx - r
                width = ranges[1 - i][1] - ranges[1 - i][0]

            ax.add_patch(
                mpatches.Rectangle(
                    pt,
                    width=width,
                    height=height,
                    fill=True,
                    color=colors[i % len(colors)],
                    alpha=RESERVED_ALPHA,
                    linewidth=0,
                )
            )
    for a, neg in enumerate(offering_negotiators):
        negtrace = [_ for _ in trace if _.negotiator == neg]
        x = [plotting_ufuns[0](_.offer) for _ in negtrace]  # type: ignore
        y = [plotting_ufuns[1](_.offer) for _ in negtrace]  # type: ignore
        alphas = [colorizer(_) for _ in negtrace]
        # (ax.scatter if not with_lines else ax.plot)(
        plot_with_trancparency(
            x=x,
            y=y,
            alpha=alphas,
            color=colors[a % len(colors)],
            alpha_global=PROPOSALS_ALPHA,
            label=f"{map_(neg)}",
            ax=ax,
            marker=markers[a % len(markers)],
            with_lines=with_lines,
            linestyle=":",
        )
    # if not fast and frontier:
    #     welfare, mx = [frontier[0]], sum(frontier[0])
    #     for u in frontier[1:]:
    #         if sum(u) < mx - 1e-12:
    #             break
    #         welfare.append(u)
    #     ax.scatter(
    #         [_[0] for _ in welfare],
    #         [_[1] for _ in welfare],
    #         color="magenta",
    #         label=f"Max. Welfare",
    #         marker="s",
    #         alpha=WELFARE_ALPHA,
    #         s=int(default_marker_size * WELFARE_SCALE),
    #     )
    #     if show_annotations:
    #         for f in welfare:
    #             ax.annotate(
    #                 "Max. Welfare",
    #                 xy=f,  # type: ignore (theta, radius)
    #                 xytext=(
    #                     f[0] + 0.02,
    #                     f[1] + 0.02 * yrange,
    #                 ),
    #                 horizontalalignment="left",
    #                 verticalalignment="bottom",
    #             )
    #
    if not fast:
        if mwelfare_pts and mark_max_welfare_points:
            ax.scatter(
                [mwelfare[0] for mwelfare, _ in mwelfare_pts],
                [mwelfare[1] for mwelfare, _ in mwelfare_pts],
                color=WELFARE_COLOR,
                label="Max Welfare Points",
                marker=WELFARE_MARKER,
                alpha=NASH_ALPHA,
                s=int(default_marker_size * WELFARE_SCALE),
            )
            if show_annotations:
                for mwelfare, _ in mwelfare_pts:
                    ax.annotate(
                        "Max Welfare Point",
                        xy=mwelfare,  # type: ignore (theta, radius)
                        xytext=(
                            mwelfare[0] + 0.02,
                            mwelfare[1] - 0.02 * yrange,
                        ),  # fraction, fraction
                        horizontalalignment="left",
                        verticalalignment="bottom",
                    )
        if kalai_pts and mark_kalai_points:
            ax.scatter(
                [kalai[0] for kalai, _ in kalai_pts],
                [kalai[1] for kalai, _ in kalai_pts],
                color=KALAI_COLOR,
                label="Kalai Point",
                marker=KALAI_MARKER,
                alpha=KALAI_ALPHA,
                s=int(default_marker_size * KALAI_SCALE),
            )
            if show_annotations:
                for kalai, _ in kalai_pts:
                    ax.annotate(
                        "Kalai Point",
                        xy=kalai,  # type: ignore (theta, radius)
                        xytext=(
                            kalai[0] + 0.02,
                            kalai[1] - 0.02 * yrange,
                        ),  # fraction, fraction
                        horizontalalignment="left",
                        verticalalignment="bottom",
                    )

        if ks_pts and mark_ks_points:
            ax.scatter(
                [ks[0] for ks, _ in ks_pts],
                [ks[1] for ks, _ in ks_pts],
                color=KS_COLOR,
                label="KS Point",
                marker=KS_MARKER,
                alpha=KS_ALPHA,
                s=int(default_marker_size * KS_SCALE),
            )
            if show_annotations:
                for ks, _ in ks_pts:
                    ax.annotate(
                        "KS Point",
                        xy=ks,  # type: ignore (theta, radius)
                        xytext=(
                            ks[0] + 0.02,
                            ks[1] - 0.02 * yrange,
                        ),  # fraction, fraction
                        horizontalalignment="left",
                        verticalalignment="bottom",
                    )
        if nash_pts and mark_nash_points:
            ax.scatter(
                [nash[0] for nash, _ in nash_pts],
                [nash[1] for nash, _ in nash_pts],
                color=NASH_COLOR,
                label="Nash Point",
                marker=NASH_MARKER,
                alpha=NASH_ALPHA,
                s=int(default_marker_size * NASH_SCALE),
            )
            if show_annotations:
                for nash, _ in nash_pts:
                    ax.annotate(
                        "Nash Point",
                        xy=nash,  # type: ignore (theta, radius)
                        xytext=(
                            nash[0] + 0.02,
                            nash[1] - 0.02 * yrange,
                        ),  # fraction, fraction
                        horizontalalignment="left",
                        verticalalignment="bottom",
                    )

    if agreement is not None:
        ax.scatter(
            [plotting_ufuns[0](agreement)],
            [plotting_ufuns[1](agreement)],  # type: ignore
            color="black",
            marker="*",
            s=int(default_marker_size * AGREEMENT_SCALE),
            alpha=AGREEMENT_ALPHA,
            label="Agreement",
        )
        if show_annotations:
            ax.annotate(
                "Agreement",
                xy=agreement_utility,  # type: ignore
                xytext=(
                    agreement_utility[0] + 0.02,
                    agreement_utility[1] + 0.02,
                ),  # fraction, fraction
                horizontalalignment="left",
                verticalalignment="bottom",
            )

    if not fast:
        ax.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncol=2,
            mode="expand",
            borderaxespad=0.0,
        )


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
    show_total_time=True,
    show_relative_time=True,
    show_n_steps=True,
    colors: list | None = None,
    markers: list[str] | None = None,
    colormap: str = DEFAULT_COLORMAP,
    ylimits: tuple[float, float] | None = None,
    common_legend=True,
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
):
    """Plot offline run.

    Args:
        trace: Trace.
        ids: Ids.
        ufuns: Ufuns.
        agreement: Agreement.
        timedout: Timedout.
        broken: Broken.
        has_error: Has error.
        errstr: Errstr.
        names: Names.
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

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
    fig = plt.figure(figsize=(12.8, 4.8) if not only2d and not no2d else None)
    extra_col = int(not no2d)
    if only2d:
        axu = fig.subplots()
    else:
        gs = gridspec.GridSpec(len(ids), 1 + extra_col)
        axs = []
        colors, markers = make_colors_and_markers(colors, markers, len(ids), colormap)

        all_ufuns = ufuns
        for a, neg in enumerate(ids):
            if a == 0:
                axs.append(fig.add_subplot(gs[a, extra_col]))
            else:
                axs.append(fig.add_subplot(gs[a, extra_col], sharex=axs[0]))
            plot_offer_utilities(
                trace=trace,
                negotiator=neg,
                plotting_ufuns=[all_ufuns[a]] if simple_offers_view else all_ufuns,  # type: ignore
                plotting_negotiators=[neg] if simple_offers_view else ids,
                ax=axs[-1],
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
        if not no2d:
            axu = fig.add_subplot(gs[:, 0])
    if not no2d:
        # agreement = state.agreement
        # state = state
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
            ax=axu,  # type: ignore
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
    if save_fig:
        if fig_name is None:
            fig_name = str(uuid.uuid4()) + ".png"
        if path is None:
            path_ = pathlib.Path().absolute()
        else:
            path_ = pathlib.Path(path)
        path_.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            str(path_ / fig_name),
            bbox_inches="tight",
            transparent=False,
            pad_inches=0.05,
        )  # type: ignore
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
    show_total_time=True,
    show_relative_time=True,
    show_n_steps=True,
    colors: list | None = None,
    markers: list[str] | None = None,
    colormap: str = DEFAULT_COLORMAP,
    ylimits: tuple[float, float] | None = None,
    common_legend=True,
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
):
    """Plot mechanism run.

    Args:
        mechanism: Mechanism.
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

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
    fig = plt.figure(figsize=(12.8, 4.8) if not only2d and not no2d else None)
    extra_col = int(not no2d)
    if only2d:
        axu = fig.subplots()
    else:
        gs = gridspec.GridSpec(mechanism.nmi.n_negotiators, 1 + extra_col)
        axs = []
        colors, markers = make_colors_and_markers(
            colors, markers, len(mechanism.negotiators), colormap
        )

        all_ufuns = [_.ufun for _ in mechanism.negotiators]
        for a, neg in enumerate(mechanism.negotiator_ids):
            if a == 0:
                axs.append(fig.add_subplot(gs[a, extra_col]))
            else:
                axs.append(fig.add_subplot(gs[a, extra_col], sharex=axs[0]))
            plot_offer_utilities(
                trace=mechanism.full_trace,
                negotiator=neg,
                plotting_ufuns=[all_ufuns[a]] if simple_offers_view else all_ufuns,
                plotting_negotiators=[neg]
                if simple_offers_view
                else mechanism.negotiator_ids,
                ax=axs[-1],
                name_map=name_map,
                colors=colors,
                markers=markers,
                ignore_none_offers=ignore_none_offers,
                ylimits=ylimits,
                show_legend=(not common_legend or a == 0) and not simple_offers_view,
                show_x_label=a == len(mechanism.negotiator_ids) - 1,
                show_reserved=show_reserved,
                xdim=xdim,
                colorizer=colorizer,
                first_color_index=a if simple_offers_view else 0,
                mark_offers_view=mark_offers_view,
            )
        if not no2d:
            axu = fig.add_subplot(gs[:, 0])
    if not no2d:
        agreement = mechanism.agreement
        state = mechanism.state
        if not state.erred_negotiator:
            erredneg = errdetails = ""
        else:
            state = mechanism.state
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
            ax=axu,  # type: ignore
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
    if save_fig:
        if fig_name is None:
            fig_name = str(uuid.uuid4()) + ".png"
        if path is None:
            path_ = pathlib.Path().absolute()
        else:
            path_ = pathlib.Path(path)
        path_.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            str(path_ / fig_name),
            bbox_inches="tight",
            transparent=False,
            pad_inches=0.05,
        )  # type: ignore
    return fig
