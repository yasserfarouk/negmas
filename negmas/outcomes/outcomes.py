"""
Functions for handling outcome spaces
"""

import copy
import itertools
import numbers
import random
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np

from negmas.generics import ienumerate, iget, ikeys

from .common import Outcome, OutcomeRange, OutcomeType

if TYPE_CHECKING:
    from negmas import Mechanism

    from .issues import Issue

__all__ = [
    "outcome_for",
    "outcome_as",
    "outcome_as_tuple",
    "outcome_as_dict",
    "outcome_in_range",
    "outcome_range_is_complete",
    "outcome_range_is_valid",
    "outcome_is_complete",
    "cast_outcome",
    "outcome_types_are_ok",
    "outcome_is_valid",
    "sample_outcomes",
]


def sample_outcomes(
    issues: Iterable["Issue"],
    n_outcomes: Optional[int] = None,
    keep_issue_names=None,
    astype=dict,
    min_per_dim=5,
    expansion_policy=None,
) -> Optional[List[Optional["Outcome"]]]:
    """Discretizes the issue space and returns either a predefined number of outcomes or uniform samples

    Args:
        issues: The issues describing the issue space to be discretized
        n_outcomes: If None then exactly `min_per_dim` bins will be used for every continuous dimension and all outcomes
        will be returned
        keep_issue_names: DEPRICATED. Use `astype` instead
        min_per_dim: Max levels of discretization per dimension
        expansion_policy: None or 'repeat' or 'null' or 'no'. If repeat, then some of the outcomes will be repeated
        if None or 'no' then no expansion will happen if the total number of outcomes is less than
        n_outcomes: If 'null' then expansion will be with None values
        astype: The type used for returning outcomes. Can be tuple, dict or any `OutcomeType`

    Returns:
        List of outcomes

    Examples:

        enumberate the whole space

        >>> from negmas.outcomes import Issue
        >>> issues = [Issue(values=(0.0, 1.0), name='Price'), Issue(values=['a', 'b'], name='Name')]
        >>> sample_outcomes(issues=issues)
        [{'Price': 0.0, 'Name': 'a'}, {'Price': 0.0, 'Name': 'b'}, {'Price': 0.25, 'Name': 'a'}, {'Price': 0.25, 'Name': 'b'}, {'Price': 0.5, 'Name': 'a'}, {'Price': 0.5, 'Name': 'b'}, {'Price': 0.75, 'Name': 'a'}, {'Price': 0.75, 'Name': 'b'}, {'Price': 1.0, 'Name': 'a'}, {'Price': 1.0, 'Name': 'b'}]

        enumerate with sampling for very large space (we have 10 outcomes in the discretized space)

        >>> from negmas.outcomes import Issue
        >>> issues = [Issue(values=(0, 1), name='Price', value_type=float), Issue(values=['a', 'b'], name='Name')]
        >>> issues[0].is_continuous()
        True
        >>> sampled=sample_outcomes(issues=issues, n_outcomes=5)
        >>> len(sampled)
        5
        >>> len(set(tuple(_.values()) for _ in sampled))
        5

        >>> from negmas.outcomes import Issue
        >>> issues = [Issue(values=(0, 1), name='Price'), Issue(values=['a', 'b'], name='Name')]
        >>> issues[0].is_continuous()
        False
        >>> sampled=sample_outcomes(issues=issues, n_outcomes=5)
        >>> len(sampled)
        4
        >>> len(set(tuple(_.values()) for _ in sampled))
        4

    """
    from negmas.outcomes import Issue, enumerate_outcomes

    if keep_issue_names is not None:
        warnings.warn(
            "keep_issue_names is depricated. Use outcome_type instead.\n"
            "keep_issue_names=True <--> outcome_type=dict\n"
            "keep_issue_names=False <--> outcome_type=tuple\n",
            DeprecationWarning,
        )
        astype = dict if keep_issue_names else tuple
    issues = [copy.deepcopy(_) for _ in issues]
    continuous = []
    uncountable = []
    indx = []
    uindx = []
    discrete = []
    n_disc = 0

    for i, issue in enumerate(issues):
        if issue.is_continuous():
            continuous.append(issue)
            indx.append(i)
        elif issue.is_uncountable():
            uncountable.append(issue)
            uindx.append(i)
        else:
            discrete.append(issue)
            n_disc += issue.cardinality

    if len(continuous) > 0:
        if n_outcomes is not None:
            n_per_issue = max(min_per_dim, (n_outcomes - n_disc) / len(continuous))
        else:
            n_per_issue = min_per_dim

        for i, issue in enumerate(continuous):
            issues[indx[i]] = Issue(
                name=issue.name,
                values=list(
                    np.linspace(
                        issue.min_value, issue.max_value, num=n_per_issue, endpoint=True
                    ).tolist()
                ),
            )

    if len(uncountable) > 0:
        if n_outcomes is not None:
            n_per_issue = max(min_per_dim, (n_outcomes - n_disc) / len(uncountable))
        else:
            n_per_issue = min_per_dim

        for i, issue in enumerate(uncountable):
            issues[uindx[i]] = Issue(
                name=issue.name, values=[issue.values() for _ in range(n_per_issue)]
            )

    cardinality = 1

    for issue in issues:
        cardinality *= issue.cardinality

    if cardinality == n_outcomes or n_outcomes is None:
        return list(enumerate_outcomes(issues, astype=astype))

    if cardinality < n_outcomes:
        outcomes = list(enumerate_outcomes(issues, astype=astype))

        if expansion_policy == "no" or expansion_policy is None:
            return outcomes
        elif expansion_policy == "null":
            return outcomes + [None] * (n_outcomes - cardinality)
        elif expansion_policy == "repeat":
            n_reps = n_outcomes // cardinality
            n_rem = n_outcomes % cardinality

            if n_reps > 1:
                for _ in n_reps:
                    outcomes += outcomes

            if n_rem > 0:
                outcomes += outcomes[:n_rem]

            return outcomes
    n_per_issue = 1 + int(n_outcomes ** (1 / len(issues)))
    vals = []
    n_found = 1
    for issue in issues:
        if n_per_issue < 2:
            if random.random() < 0.5:
                vals.append((issue.min_value,))
            else:
                vals.append((issue.max_value,))
            continue
        vals.append(issue.alli(n=n_per_issue))
        n_found *= n_per_issue
        if n_found > n_outcomes:
            n_per_issue = 1
    outcomes = itertools.product(*vals)
    if issubclass(astype, tuple):
        return list(outcomes)[:n_outcomes]
    names = [i.name for i in issues]
    outcomes = list(dict(zip(names, o)) for o in outcomes)[:n_outcomes]
    if issubclass(astype, dict):
        return outcomes
    return [astype(*o) for o in outcomes]


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
    outcome = outcome_as_dict(outcome, [_.name for _ in issues])

    for issue in issues:
        for key in ikeys(outcome):
            if str(issue.name) == str(key):
                break

        else:
            continue

        value = iget(outcome, key)

        if issue._is_range and (
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
    if not isinstance(outcome, tuple):
        outcome = outcome_as_tuple(outcome, [_.name for _ in issues])
    for v, i in zip(outcome, issues):
        if i.value_type is None:
            continue
        if not isinstance(v, i.value_type):
            return False
    return True


def cast_outcome(outcome: "Outocme", issues: List["Issue"]) -> bool:
    """
    Casts the types of values in the outcomes to the value-type of each issue (if given)
    """
    if not issues or not outcome:
        return outcome
    is_dict = isinstance(outcome, dict)
    if is_dict:
        keys = outcome.keys()
        outcome = outcome_as_tuple(outcome, [_.name for _ in issues])
    new_outcome = list(outcome)
    for indx, (v, i) in enumerate(zip(outcome, issues)):
        if i.value_type is None:
            continue
        new_outcome[indx] = i.value_type(v)
    return dict(zip(keys, new_outcome)) if is_dict else tuple(new_outcome)


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
    outcome = outcome_as_dict(outcome, [_.name for _ in issues])

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


def outcome_range_is_valid(
    outcome_range: OutcomeRange, issues: Optional[Collection["Issue"]] = None
) -> Union[bool, Tuple[bool, str]]:
    """Tests whether the outcome range is valid for the set of issues.

    Args:
        outcome_range:
        issues:

    Example:
        >>> try:
        ...     outcome_range_is_valid({'price': (0, 10)})
        ... except NotImplementedError:
        ...     print('Not implemented')
        Not implemented

    Returns:

    """
    # TODO implement this function
    raise NotImplementedError()


def outcome_range_is_complete(
    outcome_range: OutcomeRange, issues: Optional[Collection["Issue"]] = None
) -> Union[bool, Tuple[bool, str]]:
    """Tests whether the outcome range is valid and complete for the set of issues

    Args:
        outcome_range:
        issues:

    Example:
        >>> try:
        ...     outcome_range_is_complete({'price': (0, 10)})
        ... except NotImplementedError:
        ...     print('Not implemented')
        Not implemented

    Returns:

    """
    # TODO implement this function
    raise NotImplementedError()


#################################
# Outcome space implementation  #
#################################
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


def outcome_as_dict(outcome: "Outcome", issues: List[Union[str, "Issue"]] = None):
    """Converts the outcome to a dict no matter what was its type"""

    if outcome is None:
        return None

    if isinstance(outcome, np.ndarray):
        outcome = tuple(outcome.tolist())

    if isinstance(outcome, dict):
        return outcome

    if isinstance(outcome, OutcomeType):
        return outcome.asdict()

    if issues is not None:
        return dict(zip([_ if isinstance(_, str) else _.name for _ in issues], outcome))
    warnings.warn(f"Outcome {outcome} is converted to a dict without issue names!!")
    return dict(zip((str(_) for _ in range(len(outcome))), outcome))


def outcome_as_tuple(outcome: "Outcome", issues: Optional[List[Union[str, "Issue"]]]):
    """
    Converts the outcome to a tuple no matter what was its type

    Args:
        outcome: the outcome to be converted
        issues: A list of issues or issue names (as strings) to order the tuple

    Remarks:
        - If called with a tuple outcome, it will issue a warning

    """

    if outcome is None:
        return None

    if isinstance(outcome, tuple):
        if issues is not None and len(outcome) > 1:
            warnings.warn(
                f"Converting an outcome {outcome} which is already a a tuple to a tuple with issue names specified. There is no guarantee that the order in the input is the same as issue_names"
            )
        return outcome

    if isinstance(outcome, dict):
        if issues is None:
            return tuple(outcome.values())
        return tuple(outcome[_ if isinstance(_, str) else _.name] for _ in issues)

    if isinstance(outcome, OutcomeType):
        if issues is not None:
            return outcome_as_tuple(outcome.asdict(), issues)
        return outcome.astuple()

    if isinstance(outcome, np.ndarray):
        if issues is not None:
            warnings.warn(
                f"Converting an outcome {outcome} to a tuple with issue names is not supported for np.ndarray"
            )
        return tuple(outcome.tolist())

    if isinstance(outcome, Iterable):
        if issues is not None:
            warnings.warn(
                f"Converting an outcome {outcome} to a tuple with issue names is not supported for general iterables"
            )
        return tuple(outcome)

    raise ValueError(f"Unknown type for outcome {type(outcome)}")


def outcome_as(
    outcome: "Outcome", astype: Type, issues: Optional[List[Union[str, "Issue"]]] = None
):
    """Converts the outcome to tuple, dict or any `OutcomeType`.

    Args:
         outcome: The outcome to adjust type
         astype: The type to return. None returns the outcome as it is
         issues: Only needed if `astype` is not tuple. You can also pass issues instead of their names

    """
    if astype is None:
        return outcome
    if issubclass(astype, tuple):
        if isinstance(outcome, tuple):
            return outcome
        return outcome_as_tuple(outcome, issues)
    if issubclass(astype, dict):
        if isinstance(outcome, dict):
            return outcome
        return outcome_as_dict(outcome, issues)
    return astype(**outcome_as_dict(outcome, issues))


def outcome_for(outcome: "Outcome", ami: "Mechanism") -> Optional["Outcome"]:
    """Converts the outcome the type specified by the mechanism

    Args:
         outcome: The outcome to adjust type
         ami: The Agent Mechanism Interface

    """
    astype = ami.outcome_type
    issue_names = None if issubclass(astype, tuple) else [_.name for _ in ami.issues]
    return outcome_as(outcome, astype, issue_names)
