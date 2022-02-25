"""
Functions for handling outcome spaces
"""
from __future__ import annotations

import math
import numbers
from typing import TYPE_CHECKING, Any, Iterable, Sequence, overload

import numpy as np

from negmas.generics import ienumerate, iget, ikeys
from negmas.helpers.numeric import isint, isreal
from negmas.outcomes.cardinal_issue import CardinalIssue
from negmas.outcomes.range_issue import RangeIssue

from .base_issue import Issue

if TYPE_CHECKING:
    from .common import Outcome, OutcomeRange
    from .outcome_space import DistanceFun, OutcomeSpace

__all__ = [
    "dict2outcome",
    "outcome2dict",
    "outcome_in_range",
    "outcome_is_complete",
    "outcome_types_are_ok",
    "outcome_is_valid",
    "generalized_minkowski_distance",
    "min_dist",
]


def _is_single(x):
    """Checks whether a value is a single value which is defined as either a string or not an Iterable."""

    return isinstance(x, str) or isinstance(x, numbers.Number)


@overload
def outcome2dict(outcome: None, issues: Sequence[str | Issue]) -> None:
    ...


@overload
def outcome2dict(outcome: Outcome, issues: Sequence[str | Issue]) -> dict[str, Any]:
    ...


def outcome2dict(
    outcome: Outcome | None, issues: Sequence[str | Issue]
) -> dict[str, Any] | None:
    """
    Converts the outcome to a dict no matter what was its type.


    Args:

        outcome: The outcome to be converted (as a tuple)
        issues: The issues/issue names used as dictionary keys in the output


    Remarks:
        - If called with a dict that is already converted, it will just return it.
        - None is converted to None

    Examples:

        >>> from negmas import make_issue
        >>> issues = [make_issue(10, "price"), make_issue(5, "quantity")]
        >>> outcome2dict((3, 4), issues=issues)
        {'price': 3, 'quantity': 4}

        You can also use issue names without creating Issue objects
        >>> issues = ["price", "quantity"]
        >>> outcome2dict((3, 4), issues=issues)
        {'price': 3, 'quantity': 4}

        Trying to convert an already converted outcome does nothing
        >>> issues = ["price", "quantity"]
        >>> outcome2dict(outcome2dict((3, 4), issues=issues), issues=issues)
        {'price': 3, 'quantity': 4}
    """

    if outcome is None:
        return None

    # if len(outcome) != len(issues):
    #     raise ValueError(
    #         f"Cannot convert {len(outcome)} valued outcome to a dict with {len(issues)} issues"
    #     )

    if isinstance(outcome, np.ndarray):
        outcome = tuple(outcome.tolist())

    if isinstance(outcome, dict):
        names = {_.name if isinstance(_, Issue) else _ for _ in issues}
        for k in outcome.keys():
            if k not in names:
                raise ValueError(
                    f"{k} not in the issue names ({names}). An invalid dict is given!!"
                )
        return outcome

    return dict(zip([_ if isinstance(_, str) else _.name for _ in issues], outcome))


def dict2outcome(d: dict[str, Any] | None, issues: list[str | Issue]) -> Outcome | None:
    """
    Converts the outcome to a tuple no matter what was its type

    Args:
        d: the dictionary to be converted
        issues: A list of issues or issue names (as strings) to order the tuple

    Remarks:
        - If called with a tuple outcome, it will issue a warning

    """

    if d is None:
        return None

    if isinstance(d, tuple):
        return d

    return tuple(d[_ if isinstance(_, str) else _.name] for _ in issues)


def generalized_minkowski_distance(
    a: Outcome,
    b: Outcome,
    outcome_space: OutcomeSpace | None,
    *,
    weights: Sequence[float] | None = None,
    dist_power: float = 2,
) -> float:
    r"""
    Calculates the difference between two outcomes given an outcome-space (optionally with issue weights). This is defined as the distance.

    Args:

        outcome_space: The outcome space used for comparison (If None an apporximate implementation is provided)
        a: first outcome
        b: second outcome
        weights: Issue weights
        dist_power: The exponent used when calculating the distance

    Remarks:

        - Implements the following distance measure:

          .. math::

             d(a, b) = \left( \sum_{i=1}^{N} w_i {\left| a_i - b_i \right|}^p \right)^{frac{1}{p}}

          where $a, b$ are the outocmes, $x_i$ is value for issue $i$ of outcoem $x$, $w_i$ is the weight of issue $i$ and $p$ is the `dist_power` passsed. Categorical issue differences is defined as $1$ if the values are not
          equal and $0$ otherwise.
        - Becomes the Euclidean distance if all issues are numeric and no weights are given
        - You can control the power:

            - Setting it to 1 is the city-block distance
            - Setting it to 0 is the maximum issue difference
    """
    from negmas.outcomes import CartesianOutcomeSpace

    if not weights:
        weights = [1] * len(a)
    if dist_power <= 0 or dist_power == float("inf"):
        if not isinstance(outcome_space, CartesianOutcomeSpace):
            return max(
                (w * abs(x - y))
                if (isint(x) or isreal(x)) and (isint(y) or isreal(y))
                else (w * int(x == y))
                for w, x, y in zip(weights, a, b)
            )
        d = float("-inf")
        for issue, w, x, y in zip(outcome_space.issues, weights, a, b):
            if isinstance(issue, CardinalIssue):
                c = w * abs(x - y)
            else:
                c = w * int(x == y)
            if c > d:
                d = c
        return d
    if not isinstance(outcome_space, CartesianOutcomeSpace):
        return math.pow(
            sum(
                (w * math.pow(abs(x - y), dist_power))
                if (isint(x) or isreal(x)) and (isint(y) or isreal(y))
                else (w * int(x == y))
                for w, x, y in zip(weights, a, b)
            ),
            1.0 / dist_power,
        )
    d = 0.0
    for issue, w, x, y in zip(outcome_space.issues, weights, a, b):
        if isinstance(issue, CardinalIssue):
            d += w * math.pow(abs(x - y), dist_power)
            continue
        d += w * int(x == y)
    return math.pow(d, 1.0 / dist_power)


def min_dist(
    test_outcome: Outcome,
    outcomes: Sequence[Outcome],
    outcome_space: OutcomeSpace | None,
    distance_fun: DistanceFun = generalized_minkowski_distance,
    **kwargs,
) -> float:
    """
    Minimum distance between an outcome and a set of outcomes in an outcome-spaceself.

    Args:
        test_outcome: The outcome tested
        outcomes: A sequence of outcomes to compare to
        outcome_space: The outcomespace used for comparison
        distance_fun: The distance function
        kwargs: Paramters to pass to the distance function


    See Also:

        `generalized_euclidean_distance`
    """
    if not outcomes:
        return 1.0
    return min(distance_fun(test_outcome, _, outcome_space, **kwargs) for _ in outcomes)


def outcome_is_valid(outcome: Outcome, issues: Iterable[Issue]) -> bool:
    """
    Test validity of an outcome given a set of issues.

    Examples:

        >>> from negmas.outcomes import make_issue
        >>> issues = [make_issue((0.5, 2.0), 'price'), make_issue(['2018.10.'+ str(_) for _ in range(1, 4)], 'date')\
                , make_issue(20, 'count')]
        >>> for _ in issues: print(_)
        price: (0.5, 2.0)
        date: ['2018.10.1', '2018.10.2', '2018.10.3']
        count: (0, 19)
        >>> print([outcome_is_valid({'price':3.0}, issues), outcome_is_valid({'date': '2018.10.4'}, issues)\
            , outcome_is_valid({'count': 21}, issues)])
        [False, False, False]
        >>> valid_incomplete = {'price': 1.9}
        >>> print(outcome_is_valid(valid_incomplete, issues))
        True
        >>> print(outcome_is_complete(valid_incomplete, issues))
        False
        >>> valid_incomplete.update({'date': '2018.10.2', 'count': 5})
        >>> print(outcome_is_complete(valid_incomplete, issues))
        True

    Args:
        outcome: outcome tested which can contain values for a partial set of issue values
        issues: issues

    Returns:
        Union[bool, Tuple[bool, str]]: If return_problem is True then a second return value contains a string with
                                      reason of failure
    """
    outcome_dict = outcome2dict(outcome, [_.name for _ in issues])

    for issue in issues:
        for key in outcome_dict.keys():
            if str(issue.name) == str(key):
                break

        else:
            continue

        value = iget(outcome_dict, key)

        if isinstance(issue, RangeIssue) and (
            isinstance(value, str) or not issue.min_value <= value <= issue.max_value
        ):
            return False

        if isinstance(issue._values, list) and value not in issue._values:
            return False

    return True


def outcome_types_are_ok(outcome: Outcome, issues: Iterable[Issue]) -> bool:
    """
    Checks that the types of all issue values in the outcome are correct
    """
    if not issues or not outcome:
        return True
    for v, i in zip(outcome, issues):
        if i.value_type is None:
            continue
        if not isinstance(v, i.value_type):
            return False
    return True


def cast_value_types(outcome: Outcome, issues: Iterable[Issue]) -> Outcome:
    """
    Casts the types of values in the outcomes to the value-type of each issue (if given)
    """
    if not issues or not outcome:
        return outcome
    new_outcome = list(outcome)
    for indx, (v, i) in enumerate(zip(outcome, issues)):
        if i.value_type is None:
            continue
        new_outcome[indx] = i.value_type(v)  # type: ignore I know that value_type is callable
    return tuple(new_outcome)


def outcome_is_complete(outcome: Outcome, issues: Sequence[Issue]) -> bool:
    """
    Tests that the outcome is valid and complete.

    Examples:

        >>> from negmas.outcomes import make_issue
        >>> issues = [make_issue((0.5, 2.0), 'price'), make_issue(['2018.10.'+ str(_) for _ in range(1, 4)], 'date')\
                , make_issue(20, 'count')]
        >>> for _ in issues: print(_)
        price: (0.5, 2.0)
        date: ['2018.10.1', '2018.10.2', '2018.10.3']
        count: (0, 19)
        >>> print([outcome_is_complete({'price':3.0}, issues), outcome_is_complete({'date': '2018.10.4'}, issues)\
            , outcome_is_complete({'count': 21}, issues)])
        [False, False, False]
        >>> valid_incomplete = {'price': 1.9}
        >>> print(outcome_is_complete(valid_incomplete, issues))
        False
        >>> valid_incomplete.update({'date': '2018.10.2', 'count': 5})
        >>> print(outcome_is_complete(valid_incomplete, issues))
        True
        >>> invalid = {'price': 2000, 'date': '2018.10.2', 'count': 5}
        >>> print(outcome_is_complete(invalid, issues))
        False
        >>> invalid = {'unknown': 2000, 'date': '2018.10.2', 'count': 5}
        >>> print(outcome_is_complete(invalid, issues))
        False

     Args:
        outcome: outcome tested which much contain valid values all issues if it is to be considered complete.
        issues: issues

    Returns:
        Union[bool, Tuple[bool, str]]: If return_problem is True then a second return value contains a string with
                                      reason of failure

    """
    try:
        outcome2dict(outcome, issues)
    except ValueError:
        return False

    if len(outcome) != len(issues):
        return False

    valid = outcome_is_valid(outcome, issues)

    if not valid:
        return False

    outcome_keys = [str(k) for k in ikeys(outcome)]

    for issue in issues:
        if str(issue.name) not in outcome_keys:
            return False

    return True


def outcome_in_range(
    outcome: Outcome,
    outcome_range: OutcomeRange,
    *,
    strict=False,
    fail_incomplete=False,
) -> bool:
    """
    Tests that the outcome is contained within the given range of outcomes.

    An outcome range defines a value or a range of values for each issue.

    Args:

        outcome: "Outcome" being tested
        outcome_range: "Outcome" range being tested against
        strict: Whether to enforce that all issues in the outcome must be mentioned in the outcome_range
        fail_incomplete: If True then outcomes that do not sepcify a value for all keys in the outcome_range
        will be considered not falling within it. If False then these outcomes will be considered falling
        within the range given that the values for the issues mentioned in the outcome satisfy the range
        constraints.

    Examples:

        >>> outcome_range = {'price': (0.0, 2.0), 'distance': [0.3, 0.4], 'type': ['a', 'b'], 'area': 3}
        >>> outcome_range_2 = {'price': [(0.0, 1.0), (1.5, 2.0)], 'area': [(3, 4), (7, 9)]}
        >>> outcome_in_range({'price':3.0}, outcome_range)
        False
        >>> outcome_in_range({'date': '2018.10.4'}, outcome_range)
        True
        >>> outcome_in_range({'date': '2018.10.4'}, outcome_range, strict=True)
        False
        >>> outcome_in_range({'area': 3}, outcome_range, fail_incomplete=True)
        False
        >>> outcome_in_range({'area': 3}, outcome_range)
        True
        >>> outcome_in_range({'type': 'c'}, outcome_range)
        False
        >>> outcome_in_range({'type': 'a'}, outcome_range)
        True
        >>> outcome_in_range({'date': '2018.10.4'}, outcome_range_2)
        True
        >>> outcome_in_range({'area': 3.1}, outcome_range_2)
        True
        >>> outcome_in_range({'area': 3}, outcome_range_2)
        False
        >>> outcome_in_range({'area': 5}, outcome_range_2)
        False
        >>> outcome_in_range({'price': 0.4}, outcome_range_2)
        True
        >>> outcome_in_range({'price': 0.4}, outcome_range_2, fail_incomplete=True)
        False
        >>> outcome_in_range({'price': 1.2}, outcome_range_2)
        False
        >>> outcome_in_range({'price': 0.4, 'area': 3.9}, outcome_range_2)
        True
        >>> outcome_in_range({'price': 0.4, 'area': 10}, outcome_range_2)
        False
        >>> outcome_in_range({'price': 1.2, 'area': 10}, outcome_range_2)
        False
        >>> outcome_in_range({'price': 1.2, 'area': 4}, outcome_range_2)
        False
        >>> outcome_in_range({'type': 'a'}, outcome_range_2)
        True
        >>> outcome_in_range({'type': 'a'}, outcome_range_2, strict=True)
        False
        >>> outcome_range = {'price': 10}
        >>> outcome_in_range({'price': 10}, outcome_range)
        True
        >>> outcome_in_range({'price': 11}, outcome_range)
        False

    Returns:

        bool: Success or failure

    Remarks:
        Outcome ranges specify regions in an outcome space. They can have any of the following conditions:

        - A key/issue not mentioned in the outcome range does not add any constraints meaning that **All**
          values are acceptable except if strict == True. If strict == True then *NO* value will be accepted for issues
          not in the outcome_range.
        - A key/issue with the value None in the outcome range means **All** values on this issue are acceptable.
          This is the same as having this key/issue removed from the outcome space
        - A key/issue withe the value [] (empty list) accepts *NO* outcomes
        - A key/issue with  a single value means that it is the only one acceptable
        - A key/issue with a single 2-items tuple (min, max) means that any value within that range is acceptable.
        - A key/issue with a list of values means an output is acceptable if it falls within the condition specified
          by any of the values in the list (list == union). Each such value can be a single value, a 2-items
          tuple or another list. Notice that lists of lists can always be combined into a single list of values

    """

    if (
        fail_incomplete
        and len(set(ikeys(outcome_range)).difference(ikeys(outcome))) > 0
    ):
        return False

    for key, value in ienumerate(outcome):
        if key not in ikeys(outcome_range):
            if strict:
                return False

            continue

        values = iget(outcome_range, key, None)

        if values is None:
            return False

        if _is_single(values) and value != values:
            return False

        if isinstance(values, tuple) and not values[0] < value < values[1]:
            return False

        if isinstance(values, list):
            for constraint in values:
                if _is_single(constraint):
                    if value == constraint:
                        break

                elif isinstance(constraint, list):
                    if value in constraint:
                        break

                elif isinstance(constraint, tuple):
                    if constraint[0] < value < constraint[1]:
                        break

            else:
                return False

            continue

    return True
