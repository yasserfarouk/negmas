"""
Functions for handling outcome spaces
"""
from __future__ import annotations

import copy
import itertools
import numbers
import random
import warnings
from typing import TYPE_CHECKING, Any, Collection, List, Optional, Union

import numpy as np

from negmas.generics import ienumerate, iget, ikeys
from negmas.outcomes.base_issue import RangeIssue

from .base_issue import Issue

if TYPE_CHECKING:
    from .common import Outcome, OutcomeRange, PartialOutcome

__all__ = [
    "dict2outcome",
    "outcome2dict",
    "dict2partial_outcome",
    "partial_outcome2dict",
    "outcome_in_range",
    "outcome_is_complete",
    "outcome_types_are_ok",
    "outcome_is_valid",
]


def _is_single(x):
    """Checks whether a value is a single value which is defined as either a string or not an Iterable."""

    return isinstance(x, str) or isinstance(x, numbers.Number)


def outcome_is_valid(outcome: "Outcome", issues: Collection["Issue"]) -> bool:
    """Test validity of an outcome given a set of issues.

    Examples:

        >>> from negmas.outcomes import Issue
        >>> issues = [Issue((0.5, 2.0), 'price'), Issue(['2018.10.'+ str(_) for _ in range(1, 4)], 'date')\
                , Issue(20, 'count')]
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
    outcome = outcome2dict(outcome, [_.name for _ in issues])

    for issue in issues:
        for key in outcome.keys():
            if str(issue.name) == str(key):
                break

        else:
            continue

        value = iget(outcome, key)

        if isinstance(issue, RangeIssue) and (
            isinstance(value, str) or not issue.min_value <= value <= issue.max_value
        ):
            return False

        if isinstance(issue._values, list) and value not in issue._values:
            return False

    return True


def outcome_types_are_ok(outcome: "Outcome", issues: List["Issue"]) -> bool:
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


def cast_value_types(outcome: "Outcome", issues: List["Issue"]) -> "Outcome":
    """
    Casts the types of values in the outcomes to the value-type of each issue (if given)
    """
    if not issues or not outcome:
        return outcome
    new_outcome = list(outcome)
    for indx, (v, i) in enumerate(zip(outcome, issues)):
        if i.value_type is None:
            continue
        new_outcome[indx] = i.value_type(v)
    return tuple(new_outcome)


def outcome_is_complete(outcome: "Outcome", issues: Collection["Issue"]) -> bool:
    """Tests that the outcome is valid and complete.

    Examples:

        >>> from negmas.outcomes import Issue
        >>> issues = [Issue((0.5, 2.0), 'price'), Issue(['2018.10.'+ str(_) for _ in range(1, 4)], 'date')\
                , Issue(20, 'count')]
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
        outcome = outcome2dict(outcome, issues)
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
    outcome: "Outcome",
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


def outcome2dict(outcome: "Outcome", issues: list[str | "Issue"]) -> dict[str, Any]:
    """
    Converts the outcome to a dict no matter what was its type.


    Args:

        outcome: The outcome to be converted (as a tuple)
        issues: The issues/issue names used as dictionary keys in the output


    Remarks:
        - If called with a dict that is already converted, it will just return it.
        - None is converted to None

    Examples:

        >>> from negmas import Issue
        >>> issues = [Issue(10, "price"), Issue(5, "quantity")]
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
        names = set(_.name if isinstance(_, Issue) else _ for _ in issues)
        for k in outcome.keys():
            if k not in names:
                raise ValueError(
                    f"{k} not in the issue names ({names}). An invalid dict is given!!"
                )
        return outcome

    return dict(zip([_ if isinstance(_, str) else _.name for _ in issues], outcome))


def partial_outcome2dict(
    outcome: "PartialOutcome", issues: list[str | "Issue"]
) -> dict[str, Any]:
    """Converts a partial outcome to a dict no matter what was its type"""

    if outcome is None:
        return None

    if isinstance(outcome, dict):
        return outcome

    if isinstance(outcome, np.ndarray):
        outcome = tuple(outcome.tolist())

    return dict(
        zip(
            [
                _ if isinstance(_, str) else _.name
                for _ in [issues[j] for j in outcome.keys()]
            ],
            outcome.values(),
        )
    )


def dict2partial_outcome(
    d: dict[str, Any] | None, issues: Optional[List[Union[str, "Issue"]]]
) -> PartialOutcome:
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

    vals = tuple(d.get(_ if isinstance(_, str) else _.name, None) for _ in issues)
    outcome = dict()
    for i, v in enumerate(vals):
        if v is None:
            continue
        outcome[i] = v
    return outcome


def dict2outcome(
    d: dict[str, Any] | None, issues: list[str | "Issue"]
) -> Outcome | None:
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
