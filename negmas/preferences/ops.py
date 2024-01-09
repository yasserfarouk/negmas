from __future__ import annotations

import itertools
import math
from functools import reduce
from math import sqrt
from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence, TypeVar, overload

import numpy as np
from attrs import define
from numpy.typing import NDArray
from scipy import spatial
from scipy.stats import rankdata

from negmas import warnings
from negmas.helpers.numba_checks import jit  # type: ignore
from negmas.outcomes import Issue, Outcome, discretize_and_enumerate_issues
from negmas.outcomes.common import os_or_none
from negmas.outcomes.issue_ops import enumerate_issues
from negmas.outcomes.protocols import OutcomeSpace
from negmas.preferences.crisp.mapping import MappingUtilityFunction
from negmas.warnings import warn_if_slow

if TYPE_CHECKING:
    from negmas.preferences.prob_ufun import ProbUtilityFunction

    from .base_ufun import BaseUtilityFunction
    from .crisp_ufun import UtilityFunction
    from .discounted import DiscountedUtilityFunction

    UFunType = TypeVar("UFunType", UtilityFunction, ProbUtilityFunction)

__all__ = [
    "pareto_frontier",
    "pareto_frontier_of",
    "pareto_frontier_bf",
    "pareto_frontier_active",
    "nash_points",
    "kalai_points",
    "max_welfare_points",
    "max_relative_welfare_points",
    "make_discounted_ufun",
    "scale_max",
    "normalize",
    "sample_outcome_with_utility",
    "extreme_outcomes",
    "minmax",
    "conflict_level",
    "opposition_level",
    "winwin_level",
    "get_ranks",
    "distance_to",
    "distance_between",
    "calc_outcome_distances",
    "calc_scenario_stats",
    "ScenarioStats",
    "OutcomeDistances",
    "OutcomeOptimality",
    "sort_by_utility",
    "calc_reserved_value",
    "dominating_points",
]


@overload
def sort_by_utility(
    ufun: BaseUtilityFunction,
    outcomes: Iterable[Outcome] | None = None,
    *,
    max_cardinality: int | float = float("inf"),
    best_first: bool = True,
    rational_only: bool = False,
    return_sorted_outcomes: Literal[True] = True,
) -> tuple[NDArray[np.floating[Any]], list[Outcome]]:
    ...


@overload
def sort_by_utility(
    ufun: BaseUtilityFunction,
    outcomes: Iterable[Outcome] | None = None,
    *,
    max_cardinality: int | float = float("inf"),
    best_first: bool = True,
    rational_only: bool = False,
    return_sorted_outcomes: Literal[False],
) -> NDArray[np.floating[Any]]:
    ...


def sort_by_utility(
    ufun: BaseUtilityFunction,
    outcomes: Iterable[Outcome] | None = None,
    *,
    max_cardinality: int | float = float("inf"),
    best_first: bool = True,
    rational_only: bool = False,
    return_sorted_outcomes: bool = True,
) -> tuple[NDArray[np.floating[Any]], list[Outcome]] | NDArray[np.floating[Any]]:
    """
    Returns an ordered list of utility values and outcomes for the given ufun

    Remarks:
        - If outcomes is not given, the outcome-space of the ufun is used

    Returns:
        A tuple of two lists (first two are numpy arrays):
        - utility values ordered from best to worst or worst to best based on `best_first`
        - outcomes corresponding to the sorted utility values

    """
    if outcomes is None:
        if ufun.outcome_space is None:
            raise ValueError(
                f"Cannot find outcomes of the given ufun. Pass them explicitly"
            )
        outcomes = ufun.outcome_space.enumerate_or_sample(
            max_cardinality=max_cardinality
        )
    outcomes = list(outcomes)
    c = -1.0 if best_first else 1.0
    r = ufun.reserved_value
    if r is None:
        rational_only = False
    if rational_only:
        ou = [(outcome, u) for outcome in outcomes if (u := ufun(outcome)) >= r]
        outcomes = [_[0] for _ in ou]
        utils = c * np.asarray([_[1] for _ in ou], dtype=float)
    else:
        utils = c * np.asarray(
            [float(ufun(outcome)) for outcome in outcomes], dtype=float
        )
    indices = np.argsort(utils)
    utils = c * utils
    if not return_sorted_outcomes:
        return utils[indices]
    return utils[indices], [outcomes[_] for _ in indices]


@define
class ScenarioStats:
    opposition: float
    utility_ranges: list[tuple[float, float]]
    pareto_utils: tuple[tuple[float, ...], ...]
    pareto_outcomes: list[Outcome]
    nash_utils: list[tuple[float, ...]]
    nash_outcomes: list[Outcome]
    kalai_utils: list[tuple[float, ...]]
    kalai_outcomes: list[Outcome]
    modified_kalai_utils: list[tuple[float, ...]]
    modified_kalai_outcomes: list[Outcome]
    max_welfare_utils: list[tuple[float, ...]]
    max_welfare_outcomes: list[Outcome]
    max_relative_welfare_utils: list[tuple[float, ...]]
    max_relative_welfare_outcomes: list[Outcome]
    # TODO: Add more stats here. See the negobench overleaf for examples.
    # Add advantage_range_ratio (ratio of rational ufun ranges), utility_range_ratio.
    # Add distances between major points like kalai, nash and max-welfare
    # Add max and min advantage-range and utility-range
    # Add Spread = area of the convexhull of all points divided by are of the rectangle constructed by ranges
    # Add Optimality = fraction of outcomes on the Pareto-frontier

    @classmethod
    def from_ufuns(
        cls,
        ufuns: list[UtilityFunction] | tuple[UtilityFunction, ...],
        outcomes: Sequence[Outcome] | None = None,
        eps=1e-12,
    ) -> ScenarioStats:
        return calc_scenario_stats(ufuns, outcomes, eps)

    def restrict(
        self, ufuns: tuple[UtilityFunction], reserved_values: tuple[float, ...]
    ) -> ScenarioStats:
        ranges = self.utility_ranges
        pareto_indices = [
            i
            for i, _ in enumerate(self.pareto_utils)
            if all(_[j] >= r for j, r in enumerate(reserved_values))
        ]
        warn_if_slow(
            len(pareto_indices), "Restricting a large Pareto Frontier", lambda x: x * 10
        )
        pareto_utils = tuple(self.pareto_utils[_] for _ in pareto_indices)
        pareto_outcomes = [self.pareto_outcomes[_] for _ in pareto_indices]
        nash = nash_points(ufuns, ranges=ranges, frontier=pareto_utils)
        nash_utils, nash_indices = [_[0] for _ in nash], [_[1] for _ in nash]
        nash_outcomes = [pareto_outcomes[_] for _ in nash_indices]
        kalai = kalai_points(
            ufuns, ranges=ranges, frontier=pareto_utils, subtract_reserved_value=True
        )
        kalai_utils, kalai_indices = [_[0] for _ in kalai], [_[1] for _ in kalai]
        kalai_outcomes = [pareto_outcomes[_] for _ in kalai_indices]
        modified_kalai = kalai_points(
            ufuns, ranges=ranges, frontier=pareto_utils, subtract_reserved_value=False
        )
        modified_kalai_utils, modified_kalai_indices = [_[0] for _ in modified_kalai], [
            _[1] for _ in modified_kalai
        ]
        modified_kalai_outcomes = [pareto_outcomes[_] for _ in modified_kalai_indices]
        welfare = max_welfare_points(ufuns, ranges=ranges, frontier=pareto_utils)
        welfare_utils, welfare_indices = [_[0] for _ in welfare], [
            _[1] for _ in welfare
        ]
        welfare_outcomes = [pareto_outcomes[_] for _ in welfare_indices]
        relative_welfare = max_relative_welfare_points(
            ufuns, ranges=ranges, frontier=pareto_utils
        )
        relative_welfare_utils, relative_welfare_indices = [
            _[0] for _ in relative_welfare
        ], [_[1] for _ in relative_welfare]
        relative_welfare_outcomes = [
            pareto_outcomes[_] for _ in relative_welfare_indices
        ]
        return ScenarioStats(
            opposition=self.opposition,
            utility_ranges=self.utility_ranges,
            pareto_utils=pareto_utils,
            pareto_outcomes=pareto_outcomes,
            nash_utils=nash_utils,
            nash_outcomes=nash_outcomes,
            kalai_utils=kalai_utils,
            kalai_outcomes=kalai_outcomes,
            modified_kalai_utils=modified_kalai_utils,
            modified_kalai_outcomes=modified_kalai_outcomes,
            max_welfare_utils=welfare_utils,
            max_welfare_outcomes=welfare_outcomes,
            max_relative_welfare_utils=relative_welfare_utils,
            max_relative_welfare_outcomes=relative_welfare_outcomes,
        )


@define
class OutcomeOptimality:
    pareto_optimality: float
    nash_optimality: float
    kalai_optimality: float
    modified_kalai_optimality: float
    max_welfare_optimality: float
    # max_relative_welfare_optimality: float


@define
class OutcomeDistances:
    pareto_dist: float
    nash_dist: float
    kalai_dist: float
    modified_kalai_dist: float
    max_welfare: float
    # max_relative_welfare: float


def calc_reserved_value(
    ufun: UtilityFunction,
    fraction: float = float("inf"),
    nmin: int = 0,
    nmax: int = float("inf"),  # type: ignore
    max_cardinality: int | float = float("inf"),
    finite: bool = True,
    tight: bool = True,
) -> float:
    """
    Calculates a reserved value that keeps the given fraction of outcomes
    (saturated between nmin and nmax).

    Remarks:
        - If `finite`, the returned reserved value will be guaranteed to be finite as long as ufun always returns finite values.
        - max_cardinality is used to sample outcomes for continuous outcome spaces
        - If tight is given then the reserved values will be as near as possible to the range of the ufun
    """
    os = ufun.outcome_space
    if os is None:
        raise ValueError(
            f"Cannot calc reserved values if the outcome space is not given and the same in all ufuns"
        )
    utils, _ = sort_by_utility(ufun, max_cardinality=max_cardinality, best_first=True)
    noutcomes = len(utils)
    warn_if_slow(
        noutcomes, "Calculating Reserved Value for the given fraction is too slow"
    )
    n = min(noutcomes, min(nmax, max(nmin, int(math.ceil(fraction * noutcomes)))))
    if n <= 0:
        r = utils[0] + 1e-9 if finite else float("inf")
    elif n < noutcomes:
        r = 0.5 * (utils[n - 1] + utils[n])
    else:
        r = utils[-1] - 1e-9 if finite else float("-inf")
    if tight:
        mn, mx = ufun.minmax(above_reserve=False)
        r = max(mn - 1e-9, min(mx + 1e-9, r))
    return r


def make_discounted_ufun(
    ufun: UFunType,
    cost_per_round: float | None = None,
    power_per_round: float | None = None,
    discount_per_round: float | None = None,
    cost_per_relative_time: float | None = None,
    power_per_relative_time: float | None = None,
    discount_per_relative_time: float | None = None,
    cost_per_real_time: float | None = None,
    power_per_real_time: float | None = None,
    discount_per_real_time: float | None = None,
    dynamic_reservation: bool = True,
) -> DiscountedUtilityFunction | UFunType:
    from negmas.preferences.discounted import ExpDiscountedUFun, LinDiscountedUFun

    if cost_per_round is not None and cost_per_round > 0.0:
        ufun = LinDiscountedUFun(  # type: ignore (discounted ufuns return values of the same tyupe as their base ufun)
            ufun=ufun,
            cost=cost_per_round,
            factor="step",
            power=power_per_round,
            dynamic_reservation=dynamic_reservation,
            name=ufun.name,
            outcome_space=ufun.outcome_space,
        )
    if cost_per_relative_time is not None and cost_per_relative_time > 0.0:
        ufun = LinDiscountedUFun(  # type: ignore (discounted ufuns return values of the same tyupe as their base ufun)
            ufun=ufun,
            cost=cost_per_relative_time,
            factor="relative_time",
            power=power_per_relative_time,
            dynamic_reservation=dynamic_reservation,
            name=ufun.name,
            outcome_space=ufun.outcome_space,
        )
    if cost_per_real_time is not None and cost_per_real_time > 0.0:
        ufun = LinDiscountedUFun(  # type: ignore (discounted ufuns return values of the same tyupe as their base ufun)
            ufun=ufun,
            cost=cost_per_real_time,
            factor="real_time",
            power=power_per_real_time,
            dynamic_reservation=dynamic_reservation,
            name=ufun.name,
            outcome_space=ufun.outcome_space,
        )
    if discount_per_round is not None and discount_per_round > 0.0:
        ufun = ExpDiscountedUFun(  # type: ignore (discounted ufuns return values of the same tyupe as their base ufun)
            ufun=ufun,
            discount=discount_per_round,
            factor="step",
            dynamic_reservation=dynamic_reservation,
            name=ufun.name,
            outcome_space=ufun.outcome_space,
        )
    if discount_per_relative_time is not None and discount_per_relative_time > 0.0:
        ufun = ExpDiscountedUFun(  # type: ignore (discounted ufuns return values of the same tyupe as their base ufun)
            ufun=ufun,
            discount=discount_per_relative_time,
            factor="relative_time",
            dynamic_reservation=dynamic_reservation,
            name=ufun.name,
            outcome_space=ufun.outcome_space,
        )
    if discount_per_real_time is not None and discount_per_real_time > 0.0:
        ufun = ExpDiscountedUFun(  # type: ignore (discounted ufuns return values of the same tyupe as their base ufun)
            ufun=ufun,
            discount=discount_per_real_time,
            factor="real_time",
            dynamic_reservation=dynamic_reservation,
            name=ufun.name,
            outcome_space=ufun.outcome_space,
        )
    return ufun


def pareto_frontier_bf(
    points: np.ndarray | Iterable[Iterable[float]],
    eps=-1e-12,
    sort_by_welfare=True,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 1:
        return points

    warn_if_slow(
        len(points),
        f"Pareto's Quadratic Operation is too Slow",
        lambda x: x * x,
    )
    return _pareto_frontier_bf(points, eps, sort_by_welfare)


def pareto_frontier_chatgpt(
    points: np.ndarray | list[tuple[float, ...]],
    eps=-1e-12,
    sort_by_welfare=True,
    presort=True,
):
    """
    Finds the pareto-frontier of a set of points.

    Args:
        points: list of points each is a tuple of utility values for one outcome
        eps: A (usually negative) small number to treat as zero during calculations
        sort_by_welfare: If True, the results are sorted descindingly by total welfare
        presort: Apply the heuristic of pre-sorting all points by sum of utility

    Returns:
        indices of Pareto optimal outcomes
    """

    points = np.asarray(points)
    n_points = points.shape[0]
    sorted_indices = []
    if presort:
        sort_mask = points.sum(1).argsort()[::-1]
        sorted_indices = np.arange(n_points, dtype=np.int32)[sort_mask]
        points = points[sort_mask]
    warn_if_slow(
        n_points,
        f"Pareto's Operation is too Slow",
        lambda x: x * 10,
    )
    points = -points
    indices = np.arange(n_points, dtype=np.int32)
    points = np.asarray(points)
    hull = spatial.ConvexHull(points)
    pareto_points = []
    pareto_indices = []

    for eq in hull.equations:
        if eq[-1] > eps:
            continue
        pareto_points.extend(points[np.dot(points, eq[:-1]) + eq[-1] <= eps])
        pareto_indices.extend(indices[np.dot(points, eq[:-1]) + eq[-1] <= eps])

    indices = np.asarray(pareto_indices, dtype=np.int32)
    if sort_by_welfare:
        frontier = points[indices]
        welfare = np.sum(frontier, axis=1)
        assert frontier.shape[0] == welfare.size
        welfare_sort_order = welfare.argsort()[::-1]
        indices = indices[welfare_sort_order]
    if presort:
        indices = sorted_indices[indices]
    return indices


def pareto_frontier_convex_hull(
    points: np.ndarray | list[tuple[float, ...]],
    eps=-1e-12,
    sort_by_welfare=True,
    presort=True,
) -> np.ndarray:
    """
    Finds the pareto-frontier of a set of points.

    Args:
        points: list of points each is a tuple of utility values for one outcome
        eps: A (usually negative) small number to treat as zero during calculations
        sort_by_welfare: If True, the results are sorted descindingly by total welfare
        presort: Apply the heuristic of pre-sorting all points by sum of utility

    Returns:
        indices of Pareto optimal outcomes
    """

    points = np.asarray(points)
    n_points = points.shape[0]
    sorted_indices = []
    if presort:
        sort_mask = points.sum(1).argsort()[::-1]
        sorted_indices = np.arange(n_points, dtype=np.int32)[sort_mask]
        points = points[sort_mask]
    warn_if_slow(
        n_points,
        f"Pareto's Operation is too Slow",
        lambda x: x * 10,
    )
    n_negotiators = points.shape[1]
    if n_points < 1 or n_negotiators < 1:
        return np.empty(0, dtype=np.int64)

    def get_pareto_undominated_by(pts1, indices, pts2=None) -> NDArray[np.integer[Any]]:
        """
        Return all points in pts1 that are not Pareto dominated by any points in pts2
        """
        if pts2 is None:
            pts2 = pts1

        def filter_(pts, point, indices=indices):
            """
            Get all points in pts that are not Pareto dominated by the point pt
            """
            weakly_worse = (pts >= point).all(axis=-1)
            strictly_worse = (pts > point - eps).any(axis=-1)
            return indices[~(weakly_worse & strictly_worse)]

        return reduce(filter_, pts2, pts1)

    def get_pareto_frontier(points, indices):
        """
        Iteratively filter points based on the convex hull heuristic
        """
        pareto_groups = []

        # loop while there are points remaining
        while points.shape[0]:
            # brute force if there are few points:
            if points.shape[0] < 10:
                pareto_groups.append(get_pareto_undominated_by(points, indices))
                break

            # compute vertices of the convex hull
            hull_vertices = spatial.ConvexHull(points).vertices

            # get corresponding points
            hull_pts = points[hull_vertices]
            hull_indices = indices[hull_vertices]

            # get points in pts that are not convex hull vertices
            nonhull_mask = np.ones(points.shape[0], dtype=bool)
            nonhull_mask[hull_vertices] = False
            points = points[nonhull_mask]
            indices = indices[nonhull_mask]

            # get points in the convex hull that are on the Pareto frontier
            pareto_indices = get_pareto_undominated_by(hull_pts, hull_indices)
            pareto_groups.append(points[pareto_indices])
            pareto = points[pareto_indices]

            # filter remaining points to keep those not dominated by
            # Pareto points of the convex hull
            points = get_pareto_undominated_by(points, indices, pareto)

        return np.vstack(pareto_groups)

    indices = np.arange(n_points, dtype=np.int32)
    points = get_pareto_frontier(points, indices)

    if sort_by_welfare:
        frontier = points[indices]
        welfare = np.sum(frontier, axis=1)
        assert frontier.shape[0] == welfare.size
        welfare_sort_order = welfare.argsort()[::-1]
        indices = indices[welfare_sort_order]
    if presort:
        indices = sorted_indices[indices]
    return indices


# Faster than is_pareto_efficient_simple, but less readable.
def pareto_frontier_numpy(
    points: np.ndarray | list[tuple[float, ...]],
    eps=-1e-12,
    sort_by_welfare=True,
    presort=True,
) -> np.ndarray:
    """
    Finds the pareto-frontier of a set of points.

    Args:
        points: list of points each is a tuple of utility values for one outcome
        eps: A (usually negative) small number to treat as zero during calculations
        sort_by_welfare: If True, the results are sorted descindingly by total welfare
        presort: Apply the heuristic of pre-sorting all points by sum of utility

    Returns:
        indices of Pareto optimal outcomes
    """
    points = np.asarray(points)
    n_points = points.shape[0]
    sorted_indices = []
    if presort:
        sorted_indices = points.sum(1).argsort()[::-1]
        points = points[sorted_indices]
    warn_if_slow(
        n_points,
        f"Pareto's Operation is too Slow",
        lambda x: x * 10,
    )
    n_negotiators = points.shape[1]
    if n_points < 1 or n_negotiators < 1:
        return np.empty(0, dtype=np.int64)
    indices = np.ones(n_points, dtype=bool)
    for i, c in enumerate(points):
        if not indices[i]:
            continue
        # Keep any point with a higher utility
        indices[indices] = np.any(points[indices] > c, axis=1)
        indices[i] = True  # And keep self
    indices = np.nonzero(indices)[0]

    # indices = np.arange(n_points)
    # next_point_index = 0  # Next index in the indices array to search for
    # while next_point_index < n_points:
    #     nondominated_point_mask = np.any(points > points[next_point_index], axis=1)
    #     nondominated_point_mask[next_point_index] = True
    #     indices = indices[nondominated_point_mask]  # Remove dominated points
    #     points = points[nondominated_point_mask]
    #     next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    if sort_by_welfare:
        frontier = points[indices]
        welfare = np.sum(frontier, axis=1)
        assert frontier.shape[0] == welfare.size
        welfare_sort_order = welfare.argsort()[::-1]
        indices = indices[welfare_sort_order]
    if presort:
        indices = sorted_indices[indices]
    return indices


def pareto_frontier_numpy_faster(
    points: np.ndarray | list[tuple[float, ...]],
    eps=-1e-12,
    sort_by_welfare=True,
    presort=True,
) -> np.ndarray:
    """
    Finds the pareto-frontier of a set of points.

    Args:
        points: list of points each is a tuple of utility values for one outcome
        eps: A (usually negative) small number to treat as zero during calculations
        sort_by_welfare: If True, the results are sorted descindingly by total welfare
        presort: Apply the heuristic of pre-sorting all points by sum of utility

    Returns:
        indices of Pareto optimal outcomes
    """
    points = np.asarray(points)
    n_points = points.shape[0]
    sorted_indices = []
    if presort:
        sort_mask = points.sum(1).argsort()[::-1]
        sorted_indices = np.arange(n_points, dtype=np.int32)[sort_mask]
        points = points[sort_mask]
    warn_if_slow(
        n_points,
        f"Pareto's Operation is too Slow",
        lambda x: x * 10,
    )
    n_negotiators = points.shape[1]
    if n_points < 1 or n_negotiators < 1:
        return np.empty(0, dtype=np.int64)

    indices = np.arange(n_points)
    next_point_index = 0  # Next index in the indices array to search for
    while next_point_index < n_points:
        nondominated_point_mask = np.any(
            points > points[next_point_index] - eps, axis=1
        )
        nondominated_point_mask[next_point_index] = True
        indices = indices[nondominated_point_mask]  # Remove dominated points
        points = points[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    if sort_by_welfare:
        frontier = points[indices]
        welfare = np.sum(frontier, axis=1)
        assert frontier.shape[0] == welfare.size
        welfare_sort_order = welfare.argsort()[::-1]
        indices = indices[welfare_sort_order]
    if presort:
        indices = sorted_indices[indices]
    return indices


def pareto_frontier_of(
    points: np.ndarray | Iterable[Iterable[float]],
    eps=-1e-12,
    sort_by_welfare=True,
) -> np.ndarray:
    """Finds the pareto-frontier of a set of utils (i.e. utility values). Uses
    a fast algorithm.

    Args:
        points: list of utils
        eps: A (usually negative) small number to treat as zero during calculations
        sort_by_welfare: If True, the results are sorted descindingly by total welfare

    Returns:
    """
    utils = np.asarray(points)
    n = len(utils)
    warn_if_slow(n, f"Pareto's Linear Operation is too Slow", lambda x: x * 5)
    # for j in range(utils.shape[1]):
    #     order = utils[:, 0].argsort()[::-1]
    #     utils = utils[order]
    #     indices = order

    frontier = []
    found = []
    for p in range(0, n):
        if p in found:
            continue
        current = utils[p, :]
        to_remove = []
        for i, f in enumerate(frontier):
            current_better, current_worse = current > f, current < f
            if (current == f).all():
                frontier.append(current)
                found.append(p)
            if not current_better.any() and current_worse.any():
                # current is dominated, break
                break
            if current_better.any():
                if not current_worse.any():
                    # current dominates f, append it, remove f and scan for anything else dominated by current
                    for j, g in enumerate(frontier[i + 1 :]):
                        if (current == g).all():
                            to_remove.append(i)
                            break
                        if (current > g).any() and not (current < g).any():
                            to_remove.append(j)
                    else:
                        frontier[i] = current
                        found[i] = p
                    for i in sorted(to_remove, reverse=True):
                        frontier = frontier[:i] + frontier[i + 1 :]
                        found = found[:i] + found[i + 1 :]
                else:
                    # neither current nor f dominate each other, append current only if it is not
                    # dominated by anything in frontier
                    for j, g in enumerate(frontier[i + 1 :]):
                        if (current == g).all() or (
                            (g > current).any() and not (current > g).any()
                        ):
                            break
                    else:
                        if p not in found:
                            frontier.append(current)
                            found.append(p)
        else:
            if p not in found:
                frontier.append(current)
                found.append(p)

    if sort_by_welfare:
        frontier = np.asarray(frontier)
        welfare = np.sum(frontier, axis=1)
        welfare_sort_order = welfare.argsort()[::-1]
        found = [found[_] for _ in welfare_sort_order]
        # welfare = [_.sum() for _ in frontier]
        # indx = sorted(range(len(welfare)), key=lambda x: welfare[x], reverse=True)
        # found = [found[_] for _ in indx]
    return np.asarray([_ for _ in found])


@jit(nopython=True)
def _pareto_frontier_bf(
    points: np.ndarray,
    eps=-1e-12,
    sort_by_welfare=True,
) -> np.ndarray:
    """
    Finds the pareto-frontier of a set of points using brute-force. This is
    extremely slow but is guaranteed to be correct.

    Args:
        points: list of points each is a tuple of utility values for one outcome
        eps: A (usually negative) small number to treat as zero during calculations
        sort_by_welfare: If True, the results are sorted descindingly by total welfare

    Returns:
        indices of Pareto optimal outcomes
    """

    frontier, indices = [], []
    if len(points) < 1:
        return np.empty(0, dtype=np.int64)
    m = points.shape[1]
    if m < 1:
        return np.empty(0, dtype=np.int64)
    # points = points[points[:, 0].argsort()[::-1]]
    for i, current in enumerate(points):
        for j, test in enumerate(points):
            if j == i:
                continue
            has_better = has_worse = False
            for k in range(m):
                a, b = current[k], test[k]
                if a > b:
                    has_better = True
                    continue
                if a < b - eps:
                    has_worse = True

            if not has_better and has_worse:
                # current is dominated, break
                break
        else:
            indices.append(i)
            frontier.append(current)

    indices = np.asarray(indices, dtype=np.int64)
    # frontier = np.vstack(tuple(frontier))
    if sort_by_welfare:
        n = len(frontier)
        # welfare = frontier.sum(axis=1)
        welfare = np.zeros(n, dtype=np.float32)
        for i, f in enumerate(frontier):
            welfare[i] = f.sum()
        welfare_sort_order = welfare.argsort()[::-1]
        indices = indices[welfare_sort_order]
        # welfare = [(np.sum(_[0]), i) for i, _ in enumerate(frontier)]
        # indx = sorted(welfare, reverse=True)
        # indices = [frontier[_] for _ in indx]
    return indices


def kalai_points(
    ufuns: Sequence[UtilityFunction],
    frontier: Sequence[tuple[float, ...]],
    ranges: Sequence[tuple[float, ...]] | None = None,
    outcome_space: OutcomeSpace | None = None,
    issues: Sequence[Issue] | None = None,
    outcomes: Sequence[Outcome] | None = None,
    eps: float = 1e-12,
    subtract_reserved_value: bool = True,
) -> tuple[tuple[tuple[float, ...], int], ...]:
    """
    Calculates the all Kalai bargaining solutions on the Pareto frontier of a negotiation which is the most Egaliterian solution
    ref:  Kalai, Ehud (1977). "Proportional solutions to bargaining situations: Intertemporal utility comparisons" (PDF). Econometrica. 45 (7): 1623–1630. doi:10.2307/1913954. JSTOR 1913954.

    Args:
        ufuns: A list of ufuns to use
        frontier: a list of tuples each giving the utility values at some outcome on the frontier (usually found by `pareto_frontier`) to search within
        outcome_space: The outcome-space to consider
        issues: The issues on which the ufun is defined (outcomes may be passed instead)
        outcomes: The outcomes on which the ufun is defined (outcomes may be passed instead)

    Returns:

        A tuple of three values (all will be None if reserved values are unknown)

        - A tuple of utility values at the nash point
        - The index of the given frontier corresponding to the nash point

    Remarks:

        - The function searches within the given frontier only.

    """
    if not frontier:
        return tuple()
    # calculate the minimum and maximum value for each ufun
    if not ranges:
        ranges = [
            _.minmax(outcome_space, above_reserve=False)
            if outcome_space
            else _.minmax(issues=issues, above_reserve=False)
            if issues
            else _.minmax(outcomes=outcomes, above_reserve=False)
            for _ in ufuns
        ]
    # find reserved values
    rs = tuple(_.reserved_value for _ in ufuns)
    rs = tuple(float(_) if _ is not None else float("-inf") for _ in rs)
    # find all reserved values
    ranges = list(ranges)
    for i, (r, rng) in enumerate(zip(rs, ranges)):
        if any(_ is None or not math.isfinite(_) for _ in rng):
            raise ValueError(f"Cannot find the range for ufun {i}: {rng}")
        if r is None or r < rng[0]:
            continue
        ranges[i] = (r, rng[1])
    # if all ranges are very tiny, return everything as optimal
    if any([(_[1] - _[0]) <= eps for _ in ranges]):
        return tuple(zip(frontier, range(len(frontier))))
    # find difference between reserved value and maximum
    vals, results = [], []
    optim_val, optim_indx = float("-inf"), None
    for indx, outcome in enumerate(frontier):
        if any(u < r for u, r in zip(outcome, rs)):
            vals.append(float("-inf"))
            continue
        if subtract_reserved_value:
            val = min(float(u) - r for u, (r, _) in zip(outcome, ranges))
        else:
            val = min(float(u) for u in outcome)
        vals.append(val)
        if val > optim_val:
            optim_val = val
            optim_indx = indx
    if optim_indx is None:
        return tuple()

    for indx, (val, outcome) in enumerate(zip(vals, frontier)):
        if val >= optim_val - eps:
            results.append((outcome, indx))
    return tuple(results)


def nash_points(
    ufuns: Sequence[UtilityFunction] | None,
    frontier: Sequence[tuple[float, ...]],
    ranges: Sequence[tuple[float, ...]] | None = None,
    outcome_space: OutcomeSpace | None = None,
    issues: Sequence[Issue] | None = None,
    outcomes: Sequence[Outcome] | None = None,
    eps=1e-12,
) -> tuple[tuple[tuple[float, ...], int], ...]:
    """Calculates all the Nash Bargaining Solutions on the Pareto frontier of a
    negotiation.

    Args:
        ufuns: A list of ufuns to use. If not given, the reserved value is assumed to be zero for all ufuns
        frontier: a list of tuples each giving the utility values at some outcome on the frontier (usually found by `pareto_frontier`) to search within
        ranges: The ranges of the utility functions involved.
        outcome_space: The outcome-space to consider
        issues: The issues on which the ufun is defined (outcomes may be passed instead)
        outcomes: The outcomes on which the ufun is defined (outcomes may be passed instead)

    Returns:

        A list of tuples (empty if cannot be calculated) each consists of:
        - A tuple of utility values at the Nash point
        - The index of the given frontier corresponding to the Nash point

    Remarks:

        - The function searches within the given frontier only.
    """
    if not frontier or len(frontier) == 0:
        return tuple()
    # calculate the minimum and maximum value for each ufun
    if not ranges:
        assert ufuns is not None
        ranges = [
            _.minmax(outcome_space, above_reserve=False)
            if outcome_space
            else _.minmax(issues=issues, above_reserve=False)
            if issues
            else _.minmax(outcomes=outcomes, above_reserve=False)
            for _ in ufuns
        ]
    n_ufuns = len(ufuns) if ufuns else len(frontier[0])
    # find reserved values
    rs = (
        tuple(_.reserved_value for _ in ufuns)
        if ufuns
        else tuple(0 for _ in range(n_ufuns))
    )
    rs = tuple(float(_) if _ is not None else float("-inf") for _ in rs)
    # find all reserved values
    ranges = list(ranges)
    for i, (r, rng) in enumerate(zip(rs, ranges)):
        if any(_ is None or not math.isfinite(_) for _ in rng):
            raise ValueError(f"Cannot find the range for ufun {i}: {rng}")
        if r is None or r < rng[0]:
            continue
        ranges[i] = (r, rng[1])
    # if all ranges are very tiny, return everything as optimal
    if any([(_[1] - _[0]) <= eps for _ in ranges]):
        return tuple(zip(frontier, range(len(frontier))))
    vals, results = [], []
    optim_val, optim_indx = float("-inf"), None
    for indx, outcome in enumerate(frontier):
        val = 1.0
        for u, (r, _) in zip(outcome, ranges):
            val *= float(u) - r
        vals.append(val)
        if val > optim_val:
            optim_val = val
            optim_indx = indx
    if optim_indx is None:
        return tuple()

    for indx, (val, outcome) in enumerate(zip(vals, frontier)):
        if val >= optim_val - eps:
            results.append((outcome, indx))
    return tuple(results)


def max_welfare_points(
    ufuns: Sequence[UtilityFunction],
    frontier: Sequence[tuple[float, ...]],
    ranges: Sequence[tuple[float, ...]] | None = None,
    outcome_space: OutcomeSpace | None = None,
    issues: Sequence[Issue] | None = None,
    outcomes: Sequence[Outcome] | None = None,
    eps=1e-12,
) -> tuple[tuple[tuple[float, ...], int], ...]:
    """Calculates all the points with maximum relative welfare (i.e. sum of
    improvements above reserved value) on the Pareto frontier of a negotiation.

    Args:
        ufuns: A list of ufuns to use
        frontier: a list of tuples each giving the utility values at some outcome on the frontier (usually found by `pareto_frontier`) to search within
        ranges: the minimum and maximum value for each ufun. If not given, outcome_space, issues, or outcomes must be given to calculate it
        outcome_space: The outcome-space to consider
        issues: The issues on which the ufun is defined (outcomes may be passed instead)
        outcomes: The outcomes on which the ufun is defined (outcomes may be passed instead)

    Returns:

        A list of tuples (empty if cannot be calculated) each consists of:
        - A tuple of utility values at the Nash point
        - The index of the given frontier corresponding to the Nash point

    Remarks:

        - The function searches within the given frontier only.
    """
    if not frontier:
        return tuple()
    # calculate the minimum and maximum value for each ufun
    if not ranges:
        ranges = [
            _.minmax(outcome_space, above_reserve=False)
            if outcome_space
            else _.minmax(issues=issues, above_reserve=False)
            if issues
            else _.minmax(outcomes=outcomes, above_reserve=False)
            for _ in ufuns
        ]
    # find all reserved values
    for i, rng in enumerate(ranges):
        if any(_ is None or not math.isfinite(_) for _ in rng):
            raise ValueError(f"Cannot find the range for ufun {i}: {rng}")
        # if r is None or r < rng[0]:
        #     continue
        # ranges[i] = (r, rng[1])

    # if all ranges are very tiny, return everything as optimal
    if any([(_[1] - _[0]) <= eps for _ in ranges]):
        return tuple(zip(frontier, range(len(frontier))))
    vals, results = [], []
    optim_val, optim_indx = float("-inf"), None
    for indx, outcome in enumerate(frontier):
        val = sum(float(u) for u in outcome)
        vals.append(val)
        if val > optim_val:
            optim_val = val
            optim_indx = indx
    if optim_indx is None:
        return tuple()

    for indx, (val, outcome) in enumerate(zip(vals, frontier)):
        if val >= optim_val - eps:
            results.append((outcome, indx))
    return tuple(results)


def dist(x: tuple[float, ...], y: tuple[float, ...]) -> float:
    if x is None or y is None:
        return float("nan")
    return sqrt(sum((a - b) ** 2 for a, b in zip(x, y, strict=True)))


def distance_between(w: tuple[float, ...], n: tuple[float, ...]) -> float:
    return dist(w, n)


def distance_to(w: tuple[float, ...], p: Iterable[tuple[float, ...]]) -> float:
    dists = list(distance_between(w, x) for x in p)
    if not dists:
        return float("nan")
    return min(dists)


def max_distance_to(w: tuple[float, ...], p: Iterable[tuple[float, ...]]) -> float:
    dists = list(distance_between(w, x) for x in p)
    if not dists:
        return float("nan")
    return max(dists)


def is_rational(ufuns: Iterable[UtilityFunction], outcome: Outcome) -> bool:
    for u in ufuns:
        if u(outcome) < u.reserved_value:
            return False
    return True


def is_irrational(utils: Iterable[UtilityFunction], outcome: Outcome) -> bool:
    return not is_rational(utils, outcome)


def calc_outcome_distances(
    utils: tuple[float, ...],
    stats: ScenarioStats,
) -> OutcomeDistances:
    pdist = distance_to(utils, stats.pareto_utils)
    ndist = distance_to(utils, stats.nash_utils)
    kdist = distance_to(utils, stats.kalai_utils)
    mkdist = distance_to(utils, stats.modified_kalai_utils)
    welfare = sum(utils)
    return OutcomeDistances(pdist, ndist, kdist, mkdist, welfare)


def estimate_max_dist(ufuns: Sequence[UtilityFunction]) -> float:
    ranges = [_.minmax(ufuns[0].outcome_space, above_reserve=False) for _ in ufuns]
    diffs = [float(b) - float(a) for a, b in ranges]
    assert all(_ >= 0 for _ in diffs)
    return sqrt(sum(d**2 for d in diffs))


def estimate_max_dist_using_outcomes(
    ufuns: Sequence[UtilityFunction],
    outcome_utils: Sequence[tuple[float, ...]],
) -> float:
    if not outcome_utils:
        return estimate_max_dist(ufuns)
    ranges = [_.minmax(ufuns[0].outcome_space) for _ in ufuns]
    mins = [float(a) for a, _ in ranges]
    maxs = [float(a) for a, _ in ranges]
    # distances between outcomes and minimum ranges
    dists = [(_ - mn for _, mn in zip(p, mins, strict=True)) for p in outcome_utils]
    # distances between outcomes and maximum ranges
    dists += [(_ - mx for _, mx in zip(p, maxs, strict=True)) for p in outcome_utils]
    # distances between outcomes themselves
    dists += [
        (a - b for a, b in zip(p, p2, strict=True))
        for p, p2 in itertools.product(outcome_utils, outcome_utils)
        if p != p2
    ]
    return sqrt(max(sum(_**2 for _ in d) for d in dists))


def calc_outcome_optimality(
    dists: OutcomeDistances,
    stats: ScenarioStats,
    max_dist: float,
) -> OutcomeOptimality:
    optim = dict()
    for name, lst, diff in (
        ("pareto_optimality", stats.pareto_utils, dists.pareto_dist),
        ("nash_optimality", stats.nash_utils, dists.nash_dist),
        ("kalai_optimality", stats.kalai_utils, dists.kalai_dist),
        (
            "modified_kalai_optimality",
            stats.modified_kalai_utils,
            dists.modified_kalai_dist,
        ),
    ):
        if not lst:
            optim[name] = float("nan")
            continue
        if abs(max_dist) < 1e-12:
            optim[name] = float("nan")
            continue
        optim[name] = max(0, 1 - diff / max_dist)
        # assert 0 <= optim[name] <= 1, f"{name=}, {optim[name]=}"
    for name, lst, diff in (
        ("max_welfare_optimality", stats.pareto_utils, dists.max_welfare),
    ):
        if not lst:
            optim[name] = float("nan")
            continue
        mx = max(sum(_) for _ in lst)
        if abs(mx) < 1e-12:
            optim[name] = float("nan")
            continue
        optim[name] = max(0, diff / mx)
        assert 0 <= optim[name], f"{name=}, {optim[name]=}"
    return OutcomeOptimality(**optim)


def calc_scenario_stats(
    ufuns: tuple[UtilityFunction, ...] | list[UtilityFunction],
    outcomes: Sequence[Outcome] | None = None,
    eps=1e-12,
) -> ScenarioStats:
    if not ufuns:
        raise ValueError("Must pass the ufuns")
    ufuns = list(ufuns)
    os = ufuns[0].outcome_space
    ranges = [_.minmax(os, above_reserve=False) for _ in ufuns]
    if os is None:
        raise ValueError(
            f"Cannot find stats if the outcome space is not given and the same in all ufuns"
        )
    for i, u in enumerate(ufuns):
        if u.outcome_space is None or u.outcome_space != os:
            raise ValueError(
                f"Ufun {i} has a different outcome space than the first ufun:\n\tos[0]: {os}\n\tos[{i}]={u.outcome_space}"
            )
    if outcomes is None:
        outcomes = list(os.enumerate_or_sample(max_cardinality=float("inf")))  # type: ignore
    else:
        for o in outcomes:
            if not os.is_valid(o):
                raise ValueError(f"Outcome {o} is invalid for outcome space {os}")
    pareto_utils, pareto_indices = pareto_frontier(
        ufuns, outcomes, sort_by_welfare=True, eps=eps
    )
    pareto_outcomes = [outcomes[_] for _ in pareto_indices]
    nash = nash_points(ufuns, ranges=ranges, frontier=pareto_utils)
    nash_utils, nash_indices = [_[0] for _ in nash], [_[1] for _ in nash]
    nash_outcomes = [pareto_outcomes[_] for _ in nash_indices]
    kalai = kalai_points(
        ufuns, ranges=ranges, frontier=pareto_utils, subtract_reserved_value=True
    )
    kalai_utils, kalai_indices = [_[0] for _ in kalai], [_[1] for _ in kalai]
    kalai_outcomes = [pareto_outcomes[_] for _ in kalai_indices]
    modified_kalai = kalai_points(
        ufuns, ranges=ranges, frontier=pareto_utils, subtract_reserved_value=False
    )
    modified_kalai_utils, modified_kalai_indices = [_[0] for _ in modified_kalai], [
        _[1] for _ in modified_kalai
    ]
    modified_kalai_outcomes = [pareto_outcomes[_] for _ in modified_kalai_indices]
    welfare = max_welfare_points(ufuns, ranges=ranges, frontier=pareto_utils)
    welfare_utils, welfare_indices = [_[0] for _ in welfare], [_[1] for _ in welfare]
    welfare_outcomes = [pareto_outcomes[_] for _ in welfare_indices]
    relative_welfare = max_relative_welfare_points(
        ufuns, ranges=ranges, frontier=pareto_utils
    )
    relative_welfare_utils, relative_welfare_indices = [
        _[0] for _ in relative_welfare
    ], [_[1] for _ in relative_welfare]
    relative_welfare_outcomes = [pareto_outcomes[_] for _ in relative_welfare_indices]
    minmax = [u.minmax() for u in ufuns]
    opposition = opposition_level(
        ufuns,
        max_utils=tuple(_[1] for _ in minmax),  # type: ignore
        outcomes=outcomes,
    )
    return ScenarioStats(
        opposition=opposition,
        utility_ranges=ranges,
        pareto_utils=pareto_utils,
        pareto_outcomes=pareto_outcomes,
        nash_utils=nash_utils,
        nash_outcomes=nash_outcomes,
        kalai_utils=kalai_utils,
        kalai_outcomes=kalai_outcomes,
        modified_kalai_utils=modified_kalai_utils,
        modified_kalai_outcomes=modified_kalai_outcomes,
        max_welfare_utils=welfare_utils,
        max_welfare_outcomes=welfare_outcomes,
        max_relative_welfare_utils=relative_welfare_utils,
        max_relative_welfare_outcomes=relative_welfare_outcomes,
    )


def max_relative_welfare_points(
    ufuns: Sequence[UtilityFunction],
    frontier: Sequence[tuple[float, ...]],
    ranges: Sequence[tuple[float, ...]] | None = None,
    outcome_space: OutcomeSpace | None = None,
    issues: Sequence[Issue] | None = None,
    outcomes: Sequence[Outcome] | None = None,
    eps=1e-12,
) -> tuple[tuple[tuple[float, ...], int], ...]:
    """Calculates all the points with maximum relative welfare (i.e. sum of
    improvements above reserved value) on the Pareto frontier of a negotiation.

    Args:
        ufuns: A list of ufuns to use
        frontier: a list of tuples each giving the utility values at some outcome on the frontier (usually found by `pareto_frontier`) to search within
        outcome_space: The outcome-space to consider
        issues: The issues on which the ufun is defined (outcomes may be passed instead)
        outcomes: The outcomes on which the ufun is defined (outcomes may be passed instead)

    Returns:

        A list of tuples (empty if cannot be calculated) each consists of:
        - A tuple of utility values at the Nash point
        - The index of the given frontier corresponding to the Nash point

    Remarks:

        - The function searches within the given frontier only.
    """
    if not frontier:
        return tuple()
    # calculate the minimum and maximum value for each ufun
    if not ranges:
        ranges = [
            _.minmax(outcome_space, above_reserve=False)
            if outcome_space
            else _.minmax(issues=issues, above_reserve=False)
            if issues
            else _.minmax(outcomes=outcomes, above_reserve=False)
            for _ in ufuns
        ]
    # find all reserved values
    for i, rng in enumerate(ranges):
        if any(_ is None or not math.isfinite(_) for _ in rng):
            raise ValueError(f"Cannot find the range for ufun {i}: {rng}")
        # if r is None or r < rng[0]:
        #     continue
        # ranges[i] = (r, rng[1])

    # if all ranges are very tiny, return everything as optimal
    if any([(_[1] - _[0]) <= eps for _ in ranges]):
        return tuple(zip(frontier, range(len(frontier))))
    # find reserved values
    rs = tuple(_.reserved_value for _ in ufuns)
    rs = tuple(float(_) if _ is not None else float("-inf") for _ in rs)
    # find difference between reserved value and maximum
    diffs = [a[1] - max(b, a[0]) for a, b in zip(ranges, rs)]
    # find the optimal utilities
    vals, results = [], []
    optim_val, optim_indx = float("-inf"), None
    for indx, outcome in enumerate(frontier):
        if any(u < r for u, r in zip(outcome, rs)):
            vals.append(float("-inf"))
            continue
        val = 0.0
        for u, (r, _), d in zip(outcome, ranges, diffs):
            val += (float(u) - r) / (d if d else 1.0)
        vals.append(val)
        if val > optim_val:
            optim_val = val
            optim_indx = indx
    if optim_indx is None:
        return tuple()

    for indx, (val, outcome) in enumerate(zip(vals, frontier)):
        if val >= optim_val - eps:
            results.append((outcome, indx))
    return tuple(results)


def dominating_points(
    utils: NDArray[np.floating[Any]] | tuple[float, ...],
    points: NDArray[np.floating[Any]] | tuple[tuple[float, ...]],
) -> NDArray[np.integer[Any]]:
    """
    Tests whether the given point in utility space is dominated by any in the given points (eps is the tolerance).
    """
    points = np.asarray(points)
    utils = np.asarray(utils)
    dominating = np.any(points > utils, axis=1)
    dominating[dominating] = np.all(points[dominating] >= utils, axis=1)
    return np.nonzero(dominating)[0]


def pareto_frontier(
    ufuns: Sequence[BaseUtilityFunction],
    outcomes: Sequence[Outcome] | None = None,
    issues: Sequence[Issue] | None = None,
    n_discretization: int | None = None,
    max_cardinality: int | float = float("inf"),
    sort_by_welfare=True,
    eps: float = 1e-12,
) -> tuple[tuple[tuple[float, ...], ...], tuple[int, ...]]:
    """Finds all pareto-optimal outcomes in the list.

    Args:

        ufuns: The utility functions
        outcomes: the outcomes to be checked. If None then all possible outcomes from the issues will be checked
        issues: The set of issues (only used when outcomes is None)
        n_discretization: The number of items to discretize each real-dimension into
        sort_by_welfare: If True, the results are sorted descendingly by total welfare
        rational_only: If true, only rational outcomes can be members of the Pareto frontier.
        eps: resolution

    Returns:
        Two lists of the same length. First list gives the utilities at Pareto frontier points and second list gives their indices
    """

    ufuns = tuple(ufuns)
    if issues:
        issues = tuple(issues)
    if outcomes:
        outcomes = tuple(outcomes)

    # calculate all candidate outcomes
    if outcomes is None:
        if issues is None:
            try:
                issues = ufuns[0].outcome_space.issues  # type: ignore
            except:
                return ((), ())
        outcomes = discretize_and_enumerate_issues(
            issues, n_discretization=n_discretization, max_cardinality=max_cardinality  # type: ignore
        )
        # outcomes = itertools.product(
        #     *[issue.value_generator(n=n_discretization) for issue in issues]
        # )
    points = np.asarray(
        [[ufun(outcome) for ufun in ufuns] for outcome in outcomes], dtype=float
    )
    warn_if_slow(len(points), "Too many outcomes in the OS (Pareto Calculation)")
    reservs = np.asarray(
        [_.reserved_value if _ is not None else float("-inf") for _ in ufuns],
        dtype=float,
    )
    rational_indices = np.all(points >= reservs, axis=1)
    rational_indices = np.nonzero(rational_indices)[0]
    points = points[rational_indices]
    # rational_indices = [
    #     i for i, _ in enumerate(points) if all(a >= b for a, b in zip(_, reservs))
    # ]
    # points = [points[_] for _ in rational_indices]
    pareto_indices = pareto_frontier_active(
        points, sort_by_welfare=sort_by_welfare, eps=eps
    )
    return tuple(map(tuple, points[pareto_indices])), tuple(
        rational_indices[pareto_indices]
    )

    # return [points[_] for _ in indices], [rational_indices[_] for _ in indices]


def scale_max(
    ufun: UFunType,
    to: float = 1.0,
    outcome_space: OutcomeSpace | None = None,
    issues: Sequence[Issue] | None = None,
    outcomes: Sequence[Outcome] | None = None,
) -> UFunType:
    """Normalizes a utility function to the given range.

    Args:
        ufun: The utility function to normalize
        outcomes: A collection of outcomes to normalize for
        rng: range to normalize to. Default is [0, 1]
        epsilon: A small number specifying the resolution

    Returns:
        UtilityFunction: A utility function that is guaranteed to be normalized for the set of given outcomes
    """
    return ufun.scale_max_for(
        to, issues=issues, outcome_space=outcome_space, outcomes=outcomes
    )


def normalize(
    ufun: BaseUtilityFunction,
    to: tuple[float, float] = (0.0, 1.0),
    outcome_space: OutcomeSpace | None = None,
    issues: Sequence[Issue] | None = None,
    outcomes: Sequence[Outcome] | None = None,
) -> BaseUtilityFunction:
    """Normalizes a utility function to the given range.

    Args:
        ufun: The utility function to normalize
        to: range to normalize to. Default is [0, 1]
        outcomes: A collection of outcomes to normalize for

    Returns:
        UtilityFunction: A utility function that is guaranteed to be normalized for the set of given outcomes
    """
    outcome_space = os_or_none(outcome_space, issues, outcomes)
    if outcome_space is None:
        return ufun.normalize(to)
    return ufun.normalize_for(to, outcome_space=outcome_space)


def sample_outcome_with_utility(
    ufun: BaseUtilityFunction,
    rng: tuple[float, float],
    outcome_space: OutcomeSpace | None = None,
    issues: Sequence[Issue] | None = None,
    outcomes: Sequence[Outcome] | None = None,
    n_trials: int = 100,
) -> Outcome | None:
    """Gets one outcome within the given utility range or None on failure.

    Args:
        ufun: The utility function
        rng: The utility range
        outcome_space: The outcome-space within which to sample
        issues: The issues the utility function is defined on
        outcomes: The outcomes to sample from
        n_trials: The maximum number of trials

    Returns:

        - Either issues, or outcomes should be given but not both
    """
    return ufun.sample_outcome_with_utility(
        rng, outcome_space, issues, outcomes, n_trials
    )


def extreme_outcomes(
    ufun: UtilityFunction,
    outcome_space: OutcomeSpace | None = None,
    issues: Sequence[Issue] | None = None,
    outcomes: Sequence[Outcome] | None = None,
    max_cardinality=1000,
) -> tuple[Outcome, Outcome]:
    """Finds the best and worst outcomes.

    Args:
        ufun: The utility function
        outcome_space: An outcome-space to consider
        issues: list of issues (optional)
        outcomes: A collection of outcomes (optional)
        max_cardinality: the maximum number of outcomes to try sampling (if sampling is used and outcomes are not
                        given)
    Returns:
        Outcomes with minumum utility, maximum utility
    """
    return ufun.extreme_outcomes(
        outcome_space=outcome_space,
        issues=issues,
        outcomes=outcomes,
        max_cardinality=max_cardinality,
    )


def minmax(
    ufun: UtilityFunction,
    outcome_space: OutcomeSpace | None = None,
    issues: Sequence[Issue] | tuple[Issue, ...] | None = None,
    outcomes: Sequence[Outcome] | tuple[Outcome, ...] | None = None,
    max_cardinality=1000,
) -> tuple[float, float]:
    """Finds the range of the given utility function for the given outcomes.

    Args:
        ufun: The utility function
        outcome_space: An outcome-space to consider
        issues: list of issues (optional)
        outcomes: A collection of outcomes (optional)
        max_cardinality: the maximum number of outcomes to try sampling (if sampling is used and outcomes are not
                        given)
    Returns:
        Minumum utility, maximum utility
    """
    return ufun.minmax(
        outcome_space=outcome_space,
        issues=issues,
        outcomes=outcomes,
        max_cardinality=max_cardinality,
    )


def opposition_level(
    ufuns: Sequence[UtilityFunction],
    max_utils: float | tuple[float, float] = 1.0,  # type: ignore
    outcomes: int | Sequence[Outcome] | None = None,
    issues: Sequence[Issue] | None = None,
    max_tests: int = 10000,
) -> float:
    """Finds the opposition level of the two ufuns defined as the minimum
    distance to outcome (1, 1)

    Args:
        ufuns: A list of utility functions to use.
        max_utils: A list of maximum utility value for each ufun (or a single number if they are equal).
        outcomes: A list of outcomes (should be the complete issue space) or an integer giving the number
                 of outcomes. In the later case, ufuns should expect a tuple of a single integer.
        issues: The issues (only used if outcomes is None).
        max_tests: The maximum number of outcomes to use. Only used if issues is given and has more
                   outcomes than this value.


    Examples:


        - Opposition level of the same ufun repeated is always 0
        >>> from negmas.preferences.crisp.mapping import MappingUtilityFunction
        >>> from negmas.preferences.ops import opposition_level
        >>> u1, u2 = lambda x: x[0], lambda x: x[0]
        >>> opposition_level([u1, u2], outcomes=10, max_utils=9)
        0.0

        - Opposition level of two ufuns that are zero-sum
        >>> u1, u2 = MappingUtilityFunction(lambda x: x[0]), MappingUtilityFunction(lambda x: 9 - x[0])
        >>> opposition_level([u1, u2], outcomes=10, max_utils=9)
        0.7114582486036499
    """
    if outcomes is None:
        if issues is None:
            raise ValueError("You must either give outcomes or issues")
        outcomes = list(enumerate_issues(tuple(issues), max_cardinality=max_tests))
    if isinstance(outcomes, int):
        outcomes = [(_,) for _ in range(outcomes)]
    if not isinstance(max_utils, Iterable):
        max_utils: Iterable[float] = [max_utils] * len(ufuns)
    if len(ufuns) != len(max_utils):
        raise ValueError(
            f"Cannot use {len(ufuns)} ufuns with only {len(max_utils)} max. utility values"
        )

    nearest_val = float("inf")

    def is_irrational(outcome, ufun: BaseUtilityFunction):
        try:
            return ufun.is_worse(outcome, None)
        except:
            try:
                return ufun(outcome) < ufun(None)
            except:
                return False

    for outcome in outcomes:
        if any(is_irrational(outcome, u) for u in ufuns):
            continue
        v = sum(
            (1.0 - float(u(outcome)) / max_util) ** 2
            if max_util
            else (1.0 - float(u(outcome))) ** 2
            for max_util, u in zip(max_utils, ufuns)
        )
        if v == float("inf"):
            warnings.warn(
                f"u is infinity: {outcome}, {[_(outcome) for _ in ufuns]}, max_utils",
                warnings.NegmasNumericWarning,
            )
        if v < nearest_val:
            nearest_val = v
    return sqrt(nearest_val)


def conflict_level(
    u1: UtilityFunction,
    u2: UtilityFunction,
    outcomes: int | Sequence[Outcome],
    max_tests: int = 10000,
) -> float:
    """Finds the conflict level in these two ufuns.

    Args:
        u1: first utility function
        u2: second utility function

    Examples:
        - A nonlinear strictly zero sum case
        >>> from negmas.preferences.crisp.mapping import MappingUtilityFunction
        >>> from negmas.preferences import conflict_level
        >>> outcomes = [(_,) for _ in range(10)]
        >>> u1 = MappingUtilityFunction(dict(zip(outcomes,
        ... np.random.random(len(outcomes)))))
        >>> u2 = MappingUtilityFunction(dict(zip(outcomes,
        ... 1.0 - np.array(list(u1.mapping.values())))))
        >>> print(conflict_level(u1=u1, u2=u2, outcomes=outcomes))
        1.0

        - The same ufun
        >>> print(conflict_level(u1=u1, u2=u1, outcomes=outcomes))
        0.0

        - A linear strictly zero sum case
        >>> outcomes = [(i,) for i in range(10)]
        >>> u1 = MappingUtilityFunction(dict(zip(outcomes,
        ... np.linspace(0.0, 1.0, len(outcomes), endpoint=True))))
        >>> u2 = MappingUtilityFunction(dict(zip(outcomes,
        ... np.linspace(1.0, 0.0, len(outcomes), endpoint=True))))
        >>> print(conflict_level(u1=u1, u2=u2, outcomes=outcomes))
        1.0
    """
    if isinstance(outcomes, int):
        outcomes = [(_,) for _ in range(outcomes)]
    else:
        outcomes = list(outcomes)
    n_outcomes = len(outcomes)
    if n_outcomes == 0:
        raise ValueError(f"Cannot calculate conflit level with no outcomes")
    points = np.array([[u1(o), u2(o)] for o in outcomes])
    order = np.random.permutation(np.array(range(n_outcomes)))
    p1, p2 = points[order, 0], points[order, 1]
    signs = []
    trial = 0
    for i in range(n_outcomes - 1):
        for j in range(i + 1, n_outcomes):
            if trial >= max_tests:
                break
            trial += 1
            o11, o12 = p1[i], p1[j]
            o21, o22 = p2[i], p2[j]
            if o12 == o11 and o21 == o22:
                continue
            signs.append(int((o12 > o11 and o21 > o22) or (o12 < o11 and o21 < o22)))
    signs = np.asarray(signs)
    # TODO: confirm this is correct
    if len(signs) == 0:
        return 1.0
    return signs.mean()


def winwin_level(
    u1: UtilityFunction,
    u2: UtilityFunction,
    outcomes: int | Sequence[Outcome],
    max_tests: int = 10000,
) -> float:
    """Finds the win-win level in these two ufuns.

    Args:
        u1: first utility function
        u2: second utility function

    Examples:
        - A nonlinear same ufun case
        >>> from negmas.preferences.crisp.mapping import MappingUtilityFunction
        >>> outcomes = [(_,) for _ in range(10)]
        >>> u1 = MappingUtilityFunction(dict(zip(outcomes,
        ... np.linspace(1.0, 0.0, len(outcomes), endpoint=True))))

        - A linear strictly zero sum case
        >>> outcomes = [(_,) for _ in range(10)]
        >>> u1 = MappingUtilityFunction(dict(zip(outcomes,
        ... np.linspace(0.0, 1.0, len(outcomes), endpoint=True))))
        >>> u2 = MappingUtilityFunction(dict(zip(outcomes,
        ... np.linspace(1.0, 0.0, len(outcomes), endpoint=True))))
    """
    if isinstance(outcomes, int):
        outcomes = [(_,) for _ in range(outcomes)]
    else:
        outcomes = list(outcomes)
    n_outcomes = len(outcomes)
    points = np.array([[u1(o), u2(o)] for o in outcomes])
    order = np.random.permutation(np.array(range(n_outcomes)))
    p1, p2 = points[order, 0], points[order, 1]
    signed_diffs = []
    for trial, (i, j) in enumerate(zip(range(n_outcomes - 1), range(1, n_outcomes))):
        if trial >= max_tests:
            break
        o11, o12 = p1[i], p1[j]
        o21, o22 = p2[i], p2[j]
        if o11 == o12:
            if o21 == o22:
                continue
            else:
                win = abs(o22 - o21)
        elif o11 < o12:
            if o21 == o22:
                win = o12 - o11
            else:
                win = (o12 - o11) + (o22 - o21)
        else:
            if o21 == o22:
                win = o11 - o12
            else:
                win = (o11 - o12) + (o22 - o21)
        signed_diffs.append(win)
    signed_diffs = np.asarray(signed_diffs)
    if len(signed_diffs) == 0:
        raise ValueError("Could not calculate any signs")
    return signed_diffs.mean()


def get_ranks_bf(
    ufun: UtilityFunction, outcomes: Sequence[Outcome | None]
) -> list[float]:
    assert ufun.outcome_space is not None
    assert ufun.outcome_space.is_discrete()
    alloutcomes = list(ufun.outcome_space.enumerate_or_sample())
    n = len(alloutcomes)
    warn_if_slow(n, "Calculating Rank UFun is too Slow")
    # vals: list[tuple[float, Outcome | None]]
    u, o = sort_by_utility(
        ufun, alloutcomes, best_first=True, return_sorted_outcomes=True
    )
    ordered: list[tuple[float, Outcome | None]]
    ordered = list(zip(u, o))
    # vals = [(ufun(_), _) for _ in alloutcomes]
    # ordered = sorted(vals, reverse=True)
    # insert null into its place
    r = ufun.reserved_value
    loc = n
    if r is None:
        r = float("-inf")
    else:
        for k, (u, o) in enumerate(ordered):
            if u < r:
                loc = k
                break
    if loc == n:
        ordered.append((r, None))
    else:
        ordered.insert(loc, (r, None))

    ordered_lists = list(zip(range(n, -1, -1), ordered, strict=True))
    # mark outcomes with equal utils with the same rank
    for i, (second, first) in enumerate(
        zip(ordered_lists[1:], ordered_lists[:-1], strict=True)
    ):
        if abs(first[1][0] - second[1][0]) < 1e-10:
            ordered_lists[i + 1] = (first[0], second[1])
    results = []
    for outcome in outcomes:
        for v in ordered_lists:
            k, _ = v
            u, o = _
            if o == outcome:
                results.append(k / n)
                break
        else:
            raise ValueError(f"Could not find {outcome}")
    return results


def get_ranks(
    ufun: UtilityFunction, outcomes: Sequence[Outcome | None], normalize=False
) -> list[float] | NDArray[np.floating[Any]]:
    assert ufun.outcome_space is not None
    assert ufun.outcome_space.is_discrete()
    alloutcomes = (
        list(ufun.outcome_space.enumerate_or_sample())
        if not outcomes
        else list(outcomes)
    )
    n = len(alloutcomes)
    warn_if_slow(n, "Calculating Rank UFun is too Slow", lambda x: x * math.log(x))
    r = ufun.reserved_value
    changed = False
    if r is None:
        changed, ufun.reserved_value = True, float("-inf")
    vals = np.asarray([ufun(_) for _ in alloutcomes + [None]])
    if changed:
        ufun.reserved_value = None  # type: ignore
    ranks = rankdata(vals, method="dense") - 1.0
    if normalize:
        ranks = ranks / np.max(ranks)
    return ranks
    # # insert null into its place
    # # loc = n
    # # r = ufun.reserved_value
    # # loc = n
    # # if r is None:
    # #     r = float("-inf")
    # # else:
    # #     for k, (u, o) in enumerate(ordered):
    # #         if u < r:
    # #             loc = k
    # #             break
    # # if loc == n:
    # #     ordered.append((r, None))
    # # else:
    # #     ordered.insert(loc, (r, None))
    #
    # ordered = list(zip(range(n, -1, -1), ordered, strict=True))
    # # mark outcomes with equal utils with the same rank
    # for i, (second, first) in enumerate(zip(ordered[1:], ordered[:-1], strict=True)):
    #     if abs(first[1][0] - second[1][0]) < 1e-10:
    #         ordered[i + 1] = (first[0], second[1])
    # results = []
    # for outcome in outcomes:
    #     for v in ordered:
    #         k, _ = v
    #         u, o = _
    #         if o == outcome:
    #             results.append(k / n)
    #             break
    #     else:
    #         raise ValueError(f"Could not find {outcome}")
    # return results


def make_rank_ufun(ufun: UtilityFunction, normalize: bool = False) -> UtilityFunction:
    """
    Generates a ufun with the same ordering as the given one but with all differences
    between reserved values being equal

    Args:
        ufun: input ufun.
        normalize: if True, the resulting ranks will be normalized between 0 and 1.
    """
    assert ufun.outcome_space is not None
    assert ufun.outcome_space.is_discrete()
    alloutcomes = list(ufun.outcome_space.enumerate_or_sample()) + [None]
    ranks = get_ranks(ufun, alloutcomes, normalize=normalize)
    reserved = ranks[-1]
    return MappingUtilityFunction(
        mapping=dict(zip(alloutcomes[:-1], ranks[:-1])),
        outcome_space=ufun.outcome_space,
        reserved_value=reserved,
    )


# pareto_frontier_active = pareto_frontier_bf if NUMBA_OK else pareto_frontier_of
pareto_frontier_active = pareto_frontier_numpy
# pareto_frontier_of = pareto_frontier_numpy
