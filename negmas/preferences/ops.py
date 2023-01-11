from __future__ import annotations

import itertools
import math
from math import sqrt
from typing import TYPE_CHECKING, Iterable, Sequence, TypeVar

import numpy as np
from attr import define

from negmas import warnings
from negmas.helpers.numba_checks import jit  # type: ignore
from negmas.outcomes import Issue, Outcome, discretize_and_enumerate_issues
from negmas.outcomes.common import os_or_none
from negmas.outcomes.issue_ops import enumerate_issues
from negmas.outcomes.protocols import OutcomeSpace
from negmas.preferences.crisp.mapping import MappingUtilityFunction

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
]


@define
class ScenarioStats:
    opposition: float
    utility_ranges: list[tuple[float, float]]
    pareto_utils: list[tuple[float, ...]]
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

    @classmethod
    def from_ufuns(
        cls,
        ufuns: Sequence[UtilityFunction],
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
        pareto_utils = [self.pareto_utils[_] for _ in pareto_indices]
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
    """Calculates a reserved value that keeps the given fraction of outcomes
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
    outcomes = list(os.enumerate_or_sample(max_cardinality=max_cardinality))
    noutcomes = len(outcomes)
    utils = [ufun(o) for o in outcomes]
    uo = sorted(list(zip(utils, outcomes, strict=True)), reverse=True)
    utils = [_[0] for _ in uo]
    n = min(noutcomes, min(nmax, max(nmin, int(math.ceil(fraction * noutcomes)))))
    if n <= 0:
        r = utils[0] + 1e-9 if finite else float("inf")
    elif n < noutcomes:
        r = 0.5 * (uo[n - 1][0] + uo[n][0])
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
    return _pareto_frontier_bf(points, eps, sort_by_welfare)


@jit(nopython=True)
def _pareto_frontier_bf(
    points: np.ndarray | Iterable[Iterable[float]],
    eps=-1e-12,
    sort_by_welfare=True,
) -> np.ndarray:
    """Finds the pareto-frontier of a set of points using brute-force. This is
    extremely slow but is guaranteed to be correct.

    Args:
        points: list of points
        eps: A (usually negative) small number to treat as zero during calculations
        sort_by_welfare: If True, the results are sorted descindingly by total welfare

    Returns:
        indices of pareto optimal outcomes
    """

    frontier, indices = [], []
    for i, current in enumerate(points):
        for j, test in enumerate(points):
            if j == i:
                continue
            has_better = has_worse = False
            for a, b in zip(current, test, strict=True):
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

    indices = np.asarray(indices)
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
        welfare = [_.sum() for _ in frontier]
        indx = sorted(range(len(welfare)), key=lambda x: welfare[x], reverse=True)
        found = [found[_] for _ in indx]
    return np.asarray([_ for _ in found])


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
    Calculates the all Kalai bargaining solutions on the pareto frontier of a negotiation which is the most Egaliterian solution
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
    ufuns: Sequence[UtilityFunction],
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
        # (
        #     "max_relative_welfare_optimality",
        #     stats.pareto_utils,
        #     dists.max_relative_welfare,
        # ),
    ):
        if not lst:
            optim[name] = float("nan")
            continue
        if abs(max_dist) < 1e-12:
            optim[name] = float("nan")
            continue
        mx = max(sum(_) for _ in lst)
        optim[name] = max(0, 1 - (mx - diff) / mx)
        assert 0 <= optim[name], f"{name=}, {optim[name]=}"
    return OutcomeOptimality(**optim)


def calc_scenario_stats(
    ufuns: Sequence[UtilityFunction],
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
        ufuns, outcomes, sort_by_welfare=True
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
        max_utils=tuple(_[1] for _ in minmax),
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


def pareto_frontier(
    ufuns: Sequence[UtilityFunction],
    outcomes: Sequence[Outcome] | None = None,
    issues: Sequence[Issue] | None = None,
    n_discretization: int | None = None,
    max_cardinality: int | float = float("inf"),
    sort_by_welfare=True,
    eps: float = 1e-12,
) -> tuple[list[tuple[float, ...]], list[int]]:
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
        Two lists of the same length. First list gives the utilities at pareto frontier points and second list gives their indices
    """

    ufuns = tuple(ufuns)
    if issues:
        issues = tuple(issues)
    if outcomes:
        outcomes = tuple(outcomes)

    # calculate all candidate outcomes
    if outcomes is None:
        if issues is None:
            return [], []
        outcomes = discretize_and_enumerate_issues(
            issues, n_discretization=n_discretization, max_cardinality=max_cardinality
        )
        # outcomes = itertools.product(
        #     *[issue.value_generator(n=n_discretization) for issue in issues]
        # )
    points = [tuple(ufun(outcome) for ufun in ufuns) for outcome in outcomes]
    reservs = tuple(_.reserved_value if _ is not None else float("-inf") for _ in ufuns)
    rational_indices = [
        i for i, _ in enumerate(points) if all(a >= b for a, b in zip(_, reservs))
    ]
    points = [points[_] for _ in rational_indices]
    indices = list(
        pareto_frontier_active(points, sort_by_welfare=sort_by_welfare, eps=eps)
    )
    return [points[_] for _ in indices], [rational_indices[_] for _ in indices]


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
    for outcome in outcomes:
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
    signs = np.array(signs)
    # todo: confirm this is correct
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
    signed_diffs = np.array(signed_diffs)
    if len(signed_diffs) == 0:
        raise ValueError("Could not calculate any signs")
    return signed_diffs.mean()


def get_ranks(ufun: UtilityFunction, outcomes: Sequence[Outcome | None]) -> list[float]:
    assert ufun.outcome_space is not None
    assert ufun.outcome_space.is_discrete()
    alloutcomes = list(ufun.outcome_space.enumerate_or_sample())
    n = len(alloutcomes)
    vals: list[tuple[float, Outcome | None]]
    vals = [(ufun(_), _) for _ in alloutcomes]
    ordered = sorted(vals, reverse=True)
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

    ordered = list(zip(range(n, -1, -1), ordered, strict=True))
    # mark outcomes with equal utils with the same rank
    for i, (second, first) in enumerate(zip(ordered[1:], ordered[:-1], strict=True)):
        if abs(first[1][0] - second[1][0]) < 1e-10:
            ordered[i + 1] = (first[0], second[1])
    results = []
    for outcome in outcomes:
        for v in ordered:
            k, _ = v
            u, o = _
            if o == outcome:
                results.append(k / n)
                break
        else:
            raise ValueError(f"Could not find {outcome}")
    return results


def make_rank_ufun(ufun: UtilityFunction) -> UtilityFunction:
    assert ufun.outcome_space is not None
    assert ufun.outcome_space.is_discrete()
    alloutcomes = list(ufun.outcome_space.enumerate_or_sample()) + [None]
    ranks = get_ranks(ufun, alloutcomes)
    reserved = ranks[-1]
    return MappingUtilityFunction(
        mapping=dict(zip(alloutcomes[:-1], ranks[:-1])),
        outcome_space=ufun.outcome_space,
        reserved_value=reserved,
    )


# pareto_frontier_active = pareto_frontier_bf if NUMBA_OK else pareto_frontier_of
pareto_frontier_active = pareto_frontier_bf
pareto_frontier_of = pareto_frontier_bf
