"""Optional plotting helpers for the tournament analysis core functions.

Every function here takes the result of a core computation (a
:class:`~negmas.tournaments.analysis.payoff.PayoffTable`, a
:class:`~negmas.tournaments.analysis.dynamics.ReplicatorTrajectory`, or a
ranking ``DataFrame``) and an optional matplotlib ``Axes`` to draw on, and
returns the ``Axes`` used. Plotting is entirely optional: every core
computation in this package can be used without ever calling into this
module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .dynamics import ReplicatorTrajectory
from .equilibria import pure_symmetric_nash_equilibria
from .payoff import PayoffTable

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

__all__ = [
    "plot_payoff_heatmap",
    "plot_replicator_dynamics",
    "plot_deviation_graph",
    "plot_ranking",
    "plot_scores_bar",
]


def _get_ax(ax: Axes | None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_payoff_heatmap(
    payoff: PayoffTable, ax: Axes | None = None, annotate: bool = True
) -> Axes:
    """Plots the payoff matrix as a heatmap (rows: strategy, columns: opponent)."""
    ax = _get_ax(ax)
    im = ax.imshow(payoff.matrix, cmap="viridis")
    ax.set_xticks(range(payoff.n))
    ax.set_xticklabels(payoff.strategies, rotation=45, ha="right")
    ax.set_yticks(range(payoff.n))
    ax.set_yticklabels(payoff.strategies)
    ax.set_xlabel("opponent")
    ax.set_ylabel("strategy")
    ax.set_title(f"Payoff matrix ({payoff.metric}, {payoff.stat})")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if annotate:
        for i in range(payoff.n):
            for j in range(payoff.n):
                v = payoff.matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white")
    return ax


def plot_replicator_dynamics(
    trajectory: ReplicatorTrajectory,
    ax: Axes | None = None,
    population: str = "x",
    eps_label: float = 0.05,
) -> Axes:
    """Plots strategy-fraction trajectories over time for one population of a
    replicator dynamics run.

    Only strategies whose final fraction exceeds ``eps_label`` are labeled in
    the legend (others are still drawn, just unlabeled) to keep the legend
    readable when many strategies vanish early.
    """
    ax = _get_ax(ax)
    if population == "x":
        data, names = trajectory.x, trajectory.strategies_x
    elif population == "y":
        if trajectory.y is None or trajectory.strategies_y is None:
            raise ValueError("trajectory has no population 'y' (symmetric case)")
        data, names = trajectory.y, trajectory.strategies_y
    else:
        raise ValueError("population must be 'x' or 'y'")

    for i, name in enumerate(names):
        label = name if data[-1, i] > eps_label else None
        ax.plot(trajectory.times, data[:, i], label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("population fraction")
    ax.set_title("Replicator dynamics")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return ax


def plot_deviation_graph(payoff: PayoffTable, ax: Axes | None = None) -> Axes:
    """Plots the "deviation graph" of the payoff matrix (EGTA survey, section
    3.5, specialized to the pure-symmetric-strategy setting): a directed edge
    from strategy ``j`` to its best response ``br(j) = argmax_i M[i, j]``,
    labeled with the payoff gain of deviating. Nodes that are their own best
    response (pure symmetric Nash equilibria, see
    :func:`negmas.tournaments.analysis.equilibria.pure_symmetric_nash_equilibria`)
    are drawn with a self-loop and highlighted.
    """
    import networkx as nx

    ax = _get_ax(ax)
    nash = set(pure_symmetric_nash_equilibria(payoff))
    g = nx.DiGraph()
    g.add_nodes_from(payoff.strategies)
    for j, name_j in enumerate(payoff.strategies):
        col = payoff.matrix[:, j]
        if np.isnan(col).all():
            continue
        i = int(np.nanargmax(col))
        gain = col[i] - col[j] if not np.isnan(col[j]) else float("nan")
        g.add_edge(name_j, payoff.strategies[i], gain=gain)

    pos = nx.spring_layout(g, seed=0)
    node_colors = ["tab:orange" if s in nash else "tab:blue" for s in g.nodes]
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors)
    nx.draw_networkx_labels(g, pos, ax=ax)
    nx.draw_networkx_edges(g, pos, ax=ax, connectionstyle="arc3,rad=0.1", arrowsize=15)
    edge_labels = {
        (u, v): f"{d['gain']:.2f}" for u, v, d in g.edges(data=True) if u != v
    }
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, ax=ax, font_size=8)
    ax.set_title("Deviation graph (orange = pure symmetric Nash equilibrium)")
    ax.set_axis_off()
    return ax


def plot_ranking(ranking: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plots a ranking DataFrame (as returned by
    :func:`negmas.tournaments.analysis.ranking.sequential_elimination_ranking`,
    :func:`negmas.tournaments.analysis.ranking.tournament_evaluation_ranking`
    or :func:`negmas.tournaments.analysis.equilibria.nash_averaging_ranking`)
    as a horizontal bar chart ordered best (top) to worst (bottom).
    """
    ax = _get_ax(ax)
    ordered = ranking.sort_values("rank", ascending=False)
    value_col = next(
        (c for c in ("score", "regret", "deviation_payoff") if c in ordered.columns),
        None,
    )
    y = np.arange(len(ordered))
    if value_col is not None:
        ax.barh(y, ordered[value_col])
        ax.set_xlabel(value_col)
    else:
        ax.barh(y, ordered["rank"].max() + 1 - ordered["rank"])
        ax.set_xlabel("rank (higher bar = better)")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["strategy"])
    ax.set_title("Ranking")
    return ax


def plot_scores_bar(
    scores: pd.DataFrame,
    name_col: str = "strategy",
    value_col: str = "score",
    ax: Axes | None = None,
    title: str = "Scores",
) -> Axes:
    """Generic horizontal bar chart of a name/value score table, ordered best
    (top) to worst (bottom) by ``value_col``. Used to plot the raw tournament
    result (e.g. ``SimpleTournamentResults.final_scores`` or
    ``TournamentResults.total_scores``) alongside the post-tournament
    analysis plots.
    """
    ax = _get_ax(ax)
    ordered = scores.sort_values(value_col, ascending=True)
    y = np.arange(len(ordered))
    ax.barh(y, ordered[value_col])
    ax.set_yticks(y)
    ax.set_yticklabels(ordered[name_col])
    ax.set_xlabel(value_col)
    ax.set_title(title)
    return ax
