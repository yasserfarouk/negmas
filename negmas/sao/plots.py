from __future__ import annotations

import math
import pathlib
import uuid
from typing import TYPE_CHECKING, Callable

from negmas.helpers.misc import make_callable
from negmas.outcomes.base_issue import Issue
from negmas.outcomes.common import Outcome, os_or_none
from negmas.outcomes.outcome_space import make_os
from negmas.outcomes.protocols import OutcomeSpace
from negmas.preferences.crisp_ufun import UtilityFunction
from negmas.preferences.ops import nash_point, pareto_frontier
from negmas.sao.mechanism import TraceElement

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["plot_offer_utilities", "plot_mechanism_run", "plot_2dutils"]

ALL_MARKERS = ["s", "o", "v", "^", "<", ">", "p", "P", "h", "H", "1", "2", "3", "4"]
PROPOSALS_ALPHA = 0.9
AGREEMENT_ALPHA = 0.9
NASH_ALPHA = 0.6
RESERVED_ALPHA = 0.08
WELFARE_ALPHA = 0.6
AGREEMENT_SCALE = 10
NASH_SCALE = 4
WELFARE_SCALE = 2
OUTCOMES_SCALE = 0.5
PARETO_SCALE = 1.5


def get_cmap(n, name="jet"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    import matplotlib.pyplot as plt

    return plt.cm.get_cmap(name, n)


def make_colors_and_markers(colors, markers, n: int, colormap="jet"):
    if not colors:
        cmap = get_cmap(n, colormap)
        colors = [cmap(i) for i in range(n)]
    if not markers:
        markers = [ALL_MARKERS[i % len(ALL_MARKERS)] for i in range(n)]
    return colors, markers


def plot_offer_utilities(
    trace: list[TraceElement],
    negotiator: str,
    plotting_ufuns: list[UtilityFunction],
    plotting_negotiators: list[str],
    ignore_none_offers: bool = True,
    name_map: dict[str, str] | Callable[[str], str] | None = None,
    colors: list | None = None,
    markers: list | None = None,
    colormap: str = "jet",
    ax: Axes | None = None,  # type: ignore
    sharey=False,
    xdim: str = "relative_time",
    ylimits: tuple[float, float] | None = None,
    show_legend=True,
    show_x_label=True,
    ignore_markers_limit=50,
    show_reserved=True,
):
    import matplotlib.pyplot as plt

    map_ = make_callable(name_map)
    if ax is None:
        _, ax = plt.subplots()
    ax: Axes
    one_y = True
    axes = [ax] * len(plotting_negotiators)
    if not sharey and len(plotting_negotiators) == 2:
        axes = [ax, ax.twinx()]
        one_y = False

    colors, markers = make_colors_and_markers(
        colors, markers, len(plotting_negotiators), colormap
    )

    if xdim.startswith("step") or xdim.startswith("round"):
        trace_info = [(_.offer, _.step) for _ in trace if _.negotiator == negotiator]
    elif xdim.startswith("time") or xdim.startswith("real"):
        trace_info = [(_.offer, _.time) for _ in trace if _.negotiator == negotiator]
    else:
        trace_info = [
            (_.offer, _.relative_time) for _ in trace if _.negotiator == negotiator
        ]
    x = [_[-1] for _ in trace_info]
    for i, (u, neg, a) in enumerate(zip(plotting_ufuns, plotting_negotiators, axes)):
        name = map_(neg)
        y = [u(_[0]) for _ in trace_info]
        reserved = None
        if show_reserved:
            r = u.reserved_value
            if r is not None and math.isfinite(r):
                reserved = [r] * len(y)
        if not ignore_none_offers:
            xx = x
        else:
            good_indices = [i for i, _ in enumerate(y) if _ is not None]
            xx = [x[_] for _ in good_indices]
            y = [y[_] for _ in good_indices]
        a.plot(
            xx,
            y,
            label=f"{name} ({i})",
            color=colors[i % len(colors)],
            linestyle="solid" if neg == negotiator else "dotted",
            linewidth=2 if neg == negotiator else 1,
            marker=markers[i % len(markers)]
            if len(trace_info) < ignore_markers_limit
            else None,
        )
        if reserved:
            a.plot(
                xx,
                reserved,
                # label=f"{name} ({i})",
                color=colors[i % len(colors)],
                linestyle="solid" if neg == negotiator else "dotted",
                linewidth=0.5,
                marker=None,
            )
        if ylimits is not None:
            a.set_ylim(ylimits)
        a.set_ylabel(f"{name} ({i}) utility" if not one_y else "utility")
        if show_legend and len(plotting_negotiators) == 2:
            a.legend(
                loc=f"upper {'left' if not i else 'right'}", bbox_to_anchor=(i, 1.2)
            )
    if show_legend and len(plotting_negotiators) != 2:
        ax.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncol=2,
            mode="expand",
            borderaxespad=0.0,
        )
    if show_x_label:
        ax.set_xlabel(xdim)


def plot_2dutils(
    trace: list[TraceElement],
    plotting_ufuns: list[UtilityFunction],
    plotting_negotiators: list[str],
    offering_negotiators: list[str] | None = None,
    agreement: Outcome | None = None,
    outcome_space: OutcomeSpace | None = None,
    issues: list[Issue] | None = None,
    outcomes: list[Outcome] | None = None,
    with_lines: bool = True,
    show_annotations: bool = True,
    show_agreement: bool = False,
    show_pareto_distance: bool = True,
    show_nash_distance: bool = True,
    show_reserved: bool = True,
    end_reason: str | None = None,
    last_negotiator: str | None = None,
    name_map: dict[str, str] | Callable[[str], str] | None = None,
    colors: list | None = None,
    markers: list[str] | None = None,
    colormap: str = "jet",
    ax: Axes | None = None,  # type: ignore
):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
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
    max(_[0] for _ in utils) - min(_[0] for _ in utils)
    yrange = max(_[1] for _ in utils) - min(_[1] for _ in utils)
    frontier, frontier_outcome = pareto_frontier(
        ufuns=plotting_ufuns,
        issues=issues,
        outcomes=outcomes if not issues else None,
        sort_by_welfare=True,
    )
    nash, _ = nash_point(plotting_ufuns, frontier, outcome_space=outcome_space)
    if not nash:
        show_nash_distance = False
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

    colors, markers = make_colors_and_markers(
        colors, markers, len(offering_negotiators), colormap
    )

    agreement_utility = tuple(u(agreement) for u in plotting_ufuns)
    unknown_agreement_utility = None in agreement_utility
    if unknown_agreement_utility:
        show_pareto_distance = show_nash_distance = False
    default_marker_size = plt.rcParams.get("lines.markersize", 20) ** 2
    ax.scatter(
        [_[0] for _ in utils],
        [_[1] for _ in utils],
        color="gray",
        marker=".",
        s=int(default_marker_size * OUTCOMES_SCALE),
    )
    agent_names = [map_(_) for _ in plotting_negotiators]
    pareto_distance = float("inf")
    nash_distance = float("inf")
    f1, f2 = [_[0] for _ in frontier], [_[1] for _ in frontier]
    ax.scatter(
        f1, f2, color="red", marker="x", s=int(default_marker_size * PARETO_SCALE)
    )
    ax.set_xlabel(agent_names[0] + f"(0) utility")
    ax.set_ylabel(agent_names[1] + f"(1) utility")
    cu = agreement_utility
    if not unknown_agreement_utility:
        if nash:
            nash_distance = math.sqrt((nash[0] - cu[0]) ** 2 + (nash[1] - cu[1]) ** 2)
        for pu in frontier:
            dist = math.sqrt((pu[0] - cu[0]) ** 2 + (pu[1] - cu[1]) ** 2)
            if dist < pareto_distance:
                pareto_distance = dist
    txt = ""
    if show_agreement:
        txt += f"Agreement:{agreement}\n"
    if show_pareto_distance and agreement is not None:
        txt += f"Pareto-distance={pareto_distance:5.2}\n"
    if show_nash_distance and agreement is not None:
        txt += f"Nash-distance={nash_distance:5.2}\n"
    if end_reason:
        txt += f"{end_reason}\n"
    if last_negotiator:
        txt += f"Last: {last_negotiator}\n"

    ax.text(
        0.05,
        0.05,
        txt,
        verticalalignment="bottom",
        transform=ax.transAxes,
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
        x = [plotting_ufuns[0](_.offer) for _ in negtrace]
        y = [plotting_ufuns[1](_.offer) for _ in negtrace]
        (ax.scatter if not with_lines else ax.plot)(
            x,
            y,
            color=colors[a % len(colors)],
            alpha=PROPOSALS_ALPHA,
            label=f"{map_(neg)}",
            marker=markers[a % len(markers)],
        )
    if frontier:
        ax.scatter(
            [frontier[0][0]],
            [frontier[0][1]],
            color="magenta",
            label=f"Max. Welfare",
            marker="s",
            alpha=WELFARE_ALPHA,
            s=int(default_marker_size * WELFARE_SCALE),
        )
        if show_annotations:
            ax.annotate(
                "Max. Welfare",
                xy=frontier[0],  # theta, radius
                xytext=(
                    frontier[0][0] + 0.02,
                    frontier[0][1] + 0.02 * yrange,
                ),  # fraction, fraction
                horizontalalignment="left",
                verticalalignment="bottom",
            )
    if nash:
        ax.scatter(
            [nash[0]],
            [nash[1]],
            color="cyan",
            label=f"Nash Point",
            marker="x",
            alpha=NASH_ALPHA,
            s=int(default_marker_size * NASH_SCALE),
        )
        if show_annotations:
            ax.annotate(
                "Nash Point",
                xy=nash,  # theta, radius
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
            [plotting_ufuns[1](agreement)],
            color="black",
            marker="*",
            s=int(default_marker_size * AGREEMENT_SCALE),
            alpha=AGREEMENT_ALPHA,
            label="Agreement",
        )
        if show_annotations:
            ax.annotate(
                "Agreement",
                xy=nash,  # theta, radius
                xytext=(
                    agreement_utility[0] + 0.02,
                    agreement_utility[1] + 0.02,
                ),  # fraction, fraction
                horizontalalignment="left",
                verticalalignment="bottom",
            )
    ax.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
    )


def plot_mechanism_run(
    mechanism,
    negotiators: tuple[int, int] | tuple[str, str] | None = (0, 1),
    save_fig: bool = False,
    path: str = None,
    fig_name: str = None,
    ignore_none_offers: bool = True,
    with_lines: bool = True,
    show_agreement: bool = False,
    show_pareto_distance: bool = True,
    show_nash_distance: bool = True,
    show_end_reason: bool = True,
    show_last_negotiator: bool = True,
    show_annotations: bool = False,
    show_reserved: bool = True,
    colors: list | None = None,
    markers: list[str] | None = None,
    colormap: str = "jet",
    ylimits: tuple[float, float] | None = None,
    common_legend=True,
):
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    if negotiators is None:
        negotiators = (0, 1)
    if len(negotiators) != 2:
        raise ValueError(
            f"Cannot plot the 2D plot for the mechanism run without knowing two plotting negotiators"
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

    fig = plt.figure()
    gs = gridspec.GridSpec(mechanism.nmi.n_negotiators, 2)
    axs = []
    colors, markers = make_colors_and_markers(
        colors, markers, len(mechanism.negotiators), colormap
    )

    name_map = dict(zip(mechanism.negotiator_ids, mechanism.negotiator_names))
    all_ufuns = [_.ufun for _ in mechanism.negotiators]
    for a, neg in enumerate(mechanism.negotiator_ids):
        if a == 0:
            axs.append(fig.add_subplot(gs[a, 1]))
        else:
            axs.append(fig.add_subplot(gs[a, 1], sharex=axs[0]))
        plot_offer_utilities(
            trace=mechanism.full_trace,
            negotiator=neg,
            plotting_ufuns=all_ufuns,
            plotting_negotiators=mechanism.negotiator_ids,
            ax=axs[-1],
            name_map=name_map,
            colors=colors,
            markers=markers,
            ignore_none_offers=ignore_none_offers,
            ylimits=ylimits,
            show_legend=not common_legend or a == 0,
            show_x_label=a == len(mechanism.negotiator_ids) - 1,
            show_reserved=show_reserved,
        )
    axu = fig.add_subplot(gs[:, 0])
    agreement = mechanism.agreement
    state = mechanism.state
    if not show_last_negotiator:
        last_negotiator = None
    else:
        last_negotiator = state.last_negotiator
    if not show_end_reason:
        reason = None
    else:
        if state.timedout:
            reason = "Negotiation Timedout"
        elif agreement is not None:
            reason = "Negotiation Success"
        elif state.has_error:
            reason = "Negotiation ERROR"
        elif agreement is not None:
            reason = "Agreemend Reached"
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
        ax=axu,
        name_map=name_map,
        with_lines=with_lines,
        show_agreement=show_agreement,
        show_pareto_distance=show_pareto_distance,
        show_nash_distance=show_nash_distance,
        show_annotations=show_annotations,
        show_reserved=show_reserved,
        colors=colors,
        markers=markers,
        agreement=mechanism.state.agreement,
        end_reason=reason,
        last_negotiator=last_negotiator,
    )
    if save_fig:
        if fig_name is None:
            fig_name = str(uuid.uuid4()) + ".png"
        if path is None:
            path_ = pathlib.Path().absolute()
        else:
            path_ = pathlib.Path(path)
        path_.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path_ / fig_name), bbox_inches="tight", transparent=False)
    return fig
