from typing import (
    Collection,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import numpy as np

from negmas.outcomes import (
    Issue,
    Outcome,
)

from .base import UtilityFunction, UtilityValue
from negmas.common import AgentMechanismInterface

__all__ = [
    "pareto_frontier",
    "make_discounted_ufun",
    "normalize",
    "outcome_with_utility",
    "utility_range",
]


def make_discounted_ufun(
    ufun: UtilityFunction,
    ami: AgentMechanismInterface,
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
):
    from negmas.utilities.discounted import LinDiscountedUFun, ExpDiscountedUFun

    if cost_per_round is not None and cost_per_round > 0.0:
        ufun = LinDiscountedUFun(
            ufun=ufun,
            ami=ami,
            cost=cost_per_round,
            factor="step",
            power=power_per_round,
            dynamic_reservation=dynamic_reservation,
        )
    if cost_per_relative_time is not None and cost_per_relative_time > 0.0:
        ufun = LinDiscountedUFun(
            ufun=ufun,
            ami=ami,
            cost=cost_per_relative_time,
            factor="relative_time",
            power=power_per_relative_time,
            dynamic_reservation=dynamic_reservation,
        )
    if cost_per_real_time is not None and cost_per_real_time > 0.0:
        ufun = LinDiscountedUFun(
            ufun=ufun,
            ami=ami,
            cost=cost_per_real_time,
            factor="real_time",
            power=power_per_real_time,
            dynamic_reservation=dynamic_reservation,
        )
    if discount_per_round is not None and discount_per_round > 0.0:
        ufun = ExpDiscountedUFun(
            ufun=ufun,
            ami=ami,
            discount=discount_per_round,
            factor="step",
            dynamic_reservation=dynamic_reservation,
        )
    if discount_per_relative_time is not None and discount_per_relative_time > 0.0:
        ufun = ExpDiscountedUFun(
            ufun=ufun,
            ami=ami,
            discount=discount_per_relative_time,
            factor="relative_time",
            dynamic_reservation=dynamic_reservation,
        )
    if discount_per_real_time is not None and discount_per_real_time > 0.0:
        ufun = ExpDiscountedUFun(
            ufun=ufun,
            ami=ami,
            discount=discount_per_real_time,
            factor="real_time",
            dynamic_reservation=dynamic_reservation,
        )
    return ufun


def _pareto_frontier(
    points, eps=-1e-18, sort_by_welfare=False
) -> Tuple[List[Tuple[float]], List[int]]:
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


def pareto_frontier(
    ufuns: Iterable[UtilityFunction],
    outcomes: Iterable[Outcome] = None,
    issues: Iterable[Issue] = None,
    n_discretization: Optional[int] = 10,
    sort_by_welfare=False,
) -> Tuple[List[Tuple[float]], List[int]]:
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

    ufuns = list(ufuns)
    if issues:
        issues = list(issues)
    if outcomes:
        outcomes = list(outcomes)

    # calculate all candidate outcomes
    if outcomes is None:
        if issues is None:
            return [], []
        outcomes = Issue.discretize_and_enumerate(issues, n_discretization)
        # outcomes = itertools.product(
        #     *[issue.alli(n=n_discretization) for issue in issues]
        # )
    points = [[ufun(outcome) for ufun in ufuns] for outcome in outcomes]
    return _pareto_frontier(points, sort_by_welfare=sort_by_welfare)


def normalize(
    ufun: UtilityFunction,
    outcomes: Collection[Outcome],
    rng: Tuple[float, float] = (0.0, 1.0),
    epsilon: float = 1e-6,
    infeasible_cutoff: Optional[float] = None,
    max_only: bool = False,
) -> UtilityFunction:
    """Normalizes a utility function to the range [0, 1]

    Args:
        ufun: The utility function to normalize
        outcomes: A collection of outcomes to normalize for
        rng: range to normalize to. Default is [0, 1]
        epsilon: A small number specifying the resolution
        infeasible_cutoff: A value under which any utility is considered infeasible and is not used in normalization
        max_only: If true, normalization is done by dividing by the max otherwise the range will be used.

    Returns:
        UtilityFunction: A utility function that is guaranteed to be normalized for the set of given outcomes

    """
    from negmas.utilities.complex import (
        ComplexNonlinearUtilityFunction,
        ComplexWeightedUtilityFunction,
    )
    from negmas.utilities.static import ConstUFun

    u = [ufun(o) for o in outcomes]
    u = [float(_) for _ in u if _ is not None]
    if infeasible_cutoff is not None:
        u = [_ for _ in u if _ > infeasible_cutoff]
    if len(u) == 0:
        return ufun
    mx, mn = max(u), (rng[0] if max_only else min(u))
    if abs(mx - 1.0) < epsilon and abs(mn) < epsilon:
        return ufun
    if mx == mn:
        if -epsilon <= mn <= 1 + epsilon:
            return ufun
        else:
            r = float(ufun.reserved_value) / mn if mn != 0.0 else 0.0
            if infeasible_cutoff is not None:
                return ComplexNonlinearUtilityFunction(
                    ufuns=[ufun],
                    combination_function=lambda x: infeasible_cutoff
                    if x[0] is None
                    else x[0]
                    if x[0] < infeasible_cutoff
                    else 0.5 * x[0] / mn,
                )
            else:
                return ComplexWeightedUtilityFunction(
                    ufuns=[ufun],
                    weights=[0.5 / mn],
                    name=ufun.name + "-normalized",
                    reserved_value=r,
                    ami=ufun.ami,
                )
    scale = (rng[1] - rng[0]) / (mx - mn) if not max_only else (rng[1] / mx)
    r = scale * (ufun.reserved_value - mn)
    if abs(mn - rng[0] / scale) < epsilon:
        return ufun
    if infeasible_cutoff is not None:
        return ComplexNonlinearUtilityFunction(
            ufuns=[ufun],
            combination_function=lambda x: infeasible_cutoff
            if x[0] is None
            else x[0]
            if x[0] < infeasible_cutoff
            else scale * (x[0] - mn) + rng[0],
        )
    else:
        return ComplexWeightedUtilityFunction(
            ufuns=[ufun, ConstUFun(-mn + rng[0] / scale)],
            weights=[scale, scale],
            name=ufun.name + "-normalized",
            reserved_value=r,
            ami=ufun.ami,
        )


def outcome_with_utility(
    ufun: UtilityFunction,
    rng: Tuple[float, float],
    issues: List[Issue] = None,
    outcomes: List[Outcome] = None,
    n_trials: int = 100,
) -> Optional["Outcome"]:
    """
    Gets one outcome within the given utility range or None on failure

    Args:
        ufun: The utility function
        rng: The utility range
        issues: The issues the utility function is defined on
        outcomes: The outcomes to sample from
        n_trials: The maximum number of trials

    Returns:

        - Either issues, or outcomes should be given but not both

    """
    return ufun.outcome_with_utility(rng, issues, outcomes, n_trials)


def utility_range(
    ufun: UtilityFunction,
    issues: List[Issue] = None,
    outcomes: Collection[Outcome] = None,
    infeasible_cutoff: Optional[float] = None,
    return_outcomes=False,
    max_n_outcomes=1000,
    ami: Optional["AgentMechanismInterface"] = None,
) -> Union[
    Tuple[UtilityValue, UtilityValue],
    Tuple[UtilityValue, UtilityValue, Outcome, Outcome],
]:
    """Finds the range of the given utility function for the given outcomes

    Args:
        ufun: The utility function
        issues: List of issues (optional)
        outcomes: A collection of outcomes (optional)
        infeasible_cutoff: A value under which any utility is considered infeasible and is not used in calculation
        return_outcomes: If true, returns an outcome with the min and another with the max utility
        max_n_outcomes: the maximum number of outcomes to try sampling (if sampling is used and outcomes are not
                        given)
        ami: Optional AMI to use (if not given the internal AMI can be used)
    Returns:
        Minumum utility, maximum utility (and if return_outcomes, an outcome at each)

    """
    return ufun.utility_range(
        issues, outcomes, infeasible_cutoff, return_outcomes, max_n_outcomes, ami
    )
