from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, TypeVar

import numpy as np
from numpy.ma.core import sqrt

from negmas import warnings
from negmas.outcomes import Issue, Outcome, discretize_and_enumerate_issues
from negmas.outcomes.common import os_or_none
from negmas.outcomes.issue_ops import enumerate_issues
from negmas.outcomes.protocols import OutcomeSpace

from .base_ufun import BaseUtilityFunction

if TYPE_CHECKING:

    from negmas.preferences.prob_ufun import ProbUtilityFunction

    from .base_ufun import BaseUtilityFunction
    from .crisp_ufun import UtilityFunction
    from .discounted import DiscountedUtilityFunction

    UFunType = TypeVar("UFunType", UtilityFunction, ProbUtilityFunction)

__all__ = [
    "pareto_frontier",
    "nash_point",
    "make_discounted_ufun",
    "scale_max",
    "normalize",
    "sample_outcome_with_utility",
    "extreme_outcomes",
    "minmax",
    "conflict_level",
    "opposition_level",
    "winwin_level",
]


def make_discounted_ufun(
    ufun: UFunType,
    cost_per_round: float = None,
    power_per_round: float = None,
    discount_per_round: float = None,
    cost_per_relative_time: float = None,
    power_per_relative_time: float = None,
    discount_per_relative_time: float = None,
    cost_per_real_time: float = None,
    power_per_real_time: float = None,
    discount_per_real_time: float = None,
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


def _pareto_frontier(
    points, eps=-1e-18, sort_by_welfare=False
) -> tuple[list[tuple[float]], list[int]]:
    """Finds the pareto-frontier of a set of points

    Args:
        points: list of points
        eps: A (usually negative) small number to treat as zero during calculations
        sort_by_welfare: If True, the results are sorted descindingly by total welfare

    Returns:

    """
    points = np.asarray(points)
    n = len(points)
    indices = np.array(range(n))
    for j in range(points.shape[1]):
        order = points[:, 0].argsort()[-1::-1]
        points = points[order]
        indices = indices[order]

    frontier = [(indices[0], points[0, :])]
    for p in range(1, n):
        current = points[p, :]
        for i, (_, f) in enumerate(frontier):
            current_better, current_worse = current > f, current < f
            if np.all(current == f):
                break
            if not np.any(current_better) and np.any(current_worse):
                # current is dominated, break
                break
            if np.any(current_better):
                if not np.any(current_worse):
                    # current dominates f, append it, remove f and scan for anything else dominated by current
                    for j, (_, g) in enumerate(frontier[i + 1 :]):
                        if np.all(current == g):
                            frontier = frontier[:i] + frontier[i + 1 :]
                            break
                        if np.any(current > g) and not np.any(current < g):
                            frontier = frontier[:j] + frontier[j + 1 :]
                    else:
                        frontier[i] = (indices[p], current)
                else:
                    # neither current nor f dominate each other, append current only if it is not
                    # dominated by anything in frontier
                    for j, (_, g) in enumerate(frontier[i + 1 :]):
                        if np.all(current == g) or (
                            np.any(g > current) and not np.any(current > g)
                        ):
                            break
                    else:
                        frontier.append((indices[p], current))
    if sort_by_welfare:
        welfare = [np.sum(_[1]) for _ in frontier]
        indx = sorted(range(len(welfare)), key=lambda x: welfare[x], reverse=True)
        frontier = [frontier[_] for _ in indx]
    return [tuple(_[1]) for _ in frontier], [_[0] for _ in frontier]


def nash_point(
    ufuns: Iterable[UtilityFunction],
    frontier: Iterable[tuple[float]],
    outcome_space: OutcomeSpace | None = None,
    issues: tuple[Issue] | None = None,
    outcomes: tuple[Outcome] | None = None,
) -> tuple[tuple[float, ...] | None, int | None]:
    """
    Calculates the nash point on the pareto frontier of a negotiation

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
    nash_val = float("-inf")
    u_vals = None
    nash_indx = None
    ranges = [_.minmax(outcome_space, issues, outcomes) for _ in ufuns]
    for i, (rng, ufun) in enumerate(zip(ranges, ufuns)):
        if any(_ is None or not math.isfinite(_) for _ in rng):
            return None, None
        r = ufun.reserved_value
        if r is None or r < rng[0]:
            continue
        ranges[i] = (r, rng[1])
    if any([_[1] <= 1.0e-9 for _ in ranges]):
        return None, None
    diffs = [float(b) - float(r) for r, b in ranges]
    for indx, us in enumerate(frontier):
        val = 0.0
        for u, (r, _), d in zip(us, ranges, diffs):
            val *= (float(u) - float(r)) / d
        if val > nash_val:
            nash_val = val
            u_vals = us
            nash_indx = indx
    return u_vals, nash_indx


def pareto_frontier(
    ufuns: Iterable[UtilityFunction],
    outcomes: Iterable[Outcome] = None,
    issues: Iterable[Issue] = None,
    n_discretization: int | None = 10,
    sort_by_welfare=False,
) -> tuple[list[tuple[float, ...]], list[int]]:
    """Finds all pareto-optimal outcomes in the list

    Args:

        ufuns: The utility functions
        outcomes: the outcomes to be checked. If None then all possible outcomes from the issues will be checked
        issues: The set of issues (only used when outcomes is None)
        n_discretization: The number of items to discretize each real-dimension into
        sort_by_welfare: If True, the resutls are sorted descendingly by total welfare

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
        outcomes = discretize_and_enumerate_issues(issues, n_discretization)
        # outcomes = itertools.product(
        #     *[issue.value_generator(n=n_discretization) for issue in issues]
        # )
    points = [[ufun(outcome) for ufun in ufuns] for outcome in outcomes]
    return _pareto_frontier(points, sort_by_welfare=sort_by_welfare)


def scale_max(
    ufun: UFunType,
    to: float = 1.0,
    outcome_space: OutcomeSpace | None = None,
    issues: list[Issue] | None = None,
    outcomes: list[Outcome] | None = None,
) -> UFunType:
    """Normalizes a utility function to the given range

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
    issues: list[Issue] | None = None,
    outcomes: list[Outcome] | None = None,
) -> BaseUtilityFunction:
    """Normalizes a utility function to the given range

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
    issues: list[Issue] | None = None,
    outcomes: list[Outcome] | None = None,
    n_trials: int = 100,
) -> Outcome | None:
    """
    Gets one outcome within the given utility range or None on failure

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
    issues: list[Issue] | None = None,
    outcomes: list[Outcome] | None = None,
    max_cardinality=1000,
) -> tuple[Outcome, Outcome]:
    """
    Finds the best and worst outcomes

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
    issues: list[Issue] | None = None,
    outcomes: list[Outcome] | None = None,
    max_cardinality=1000,
) -> tuple[float, float]:
    """Finds the range of the given utility function for the given outcomes

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
    ufuns: list[UtilityFunction],
    max_utils: float | tuple[float, float] = 1.0,  # type: ignore
    outcomes: int | list[Outcome] = None,
    issues: list[Issue] = None,
    max_tests: int = 10000,
) -> float:
    """
    Finds the opposition level of the two ufuns defined as the minimum distance to outcome (1, 1)

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
    outcomes: int | list[Outcome],
    max_tests: int = 10000,
) -> float:
    """
    Finds the conflict level in these two ufuns

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
    outcomes: int | list[Outcome],
    max_tests: int = 10000,
) -> float:
    """
    Finds the win-win level in these two ufuns

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
