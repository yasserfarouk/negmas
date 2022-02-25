"""
Common datastructures used in the outcomes module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from negmas.outcomes.issue_ops import issues_from_outcomes

__all__ = [
    "Outcome",
    "PartialOutcomeDict",
    "PartialOutcomeTuple",
    "OutcomeRange",
    "check_one_and_only",
    "check_one_at_most",
    "ensure_os",
    "os_or_none",
    "DEFAULT_LEVELS",
]

if TYPE_CHECKING:
    from negmas.outcomes import OutcomeSpace

Outcome = tuple
"""An outcome is a tuple of issue values."""

PartialOutcomeDict = Mapping[int, Any]
"""A partial outcome is a simple mapping between issue INDEX and its value. Both a `tuple` and a `dict[int, Any]` satisfy this definition."""

PartialOutcomeTuple = tuple
"""A partial outcome is a partial outcome by mapping missing issues to None"""

OutcomeRange = Mapping[int, Any]
"""An outcome range is a a mapping between issue INDEX and either a value, a list of values or a Tuple with two values"""

DEFAULT_LEVELS = 10


def check_one_at_most(outcome_space, issues, outcomes) -> None:
    """Ensures that at most one of the three inputs is given (i.e. not None)"""
    if outcomes and issues:
        raise ValueError("You cannot pass `issues` and `outcomes`")
    if outcomes and outcome_space:
        raise ValueError("You cannot pass `outcome_space` and `outcomes`")
    if issues and outcome_space:
        raise ValueError("You cannot pass `issues` and `outcome_space`")


def check_one_and_only(outcome_space, issues, outcomes) -> None:
    """Ensures that one and only one of the three inputs is given (i.e. not None)"""
    if not outcomes and not issues and not outcome_space:
        raise ValueError("You must pass `outcome_spae`, `issues` or `outcomes`")
    if outcomes and issues:
        raise ValueError("You cannot pass `issues` and `outcomes`")
    if outcomes and outcome_space:
        raise ValueError("You cannot pass `outcome_space` and `outcomes`")
    if issues and outcome_space:
        raise ValueError("You cannot pass `issues` and `outcomes`")


def os_or_none(outcome_space, issues, outcomes) -> OutcomeSpace | None:
    """
    Returns an outcome space from either an outcome-space, a list of issues, a list of outcomes, or the number of outcomes

    Remakrs:
        - Precedence is in the order of paramters `outcome_space` > `issues` > `outcomes`.
        - If nothing is given, it will just return None
        - Will not copy the outcome-space if it is given
        - A `CartesianOutcomeSpace` will be created if an outcome-space is not given
        - If outcomes is given or all issues are discrete, a `DiscreteCartesianOutcomeSpace` will be created
    """
    from negmas.outcomes.outcome_space import make_os

    if outcome_space:
        return outcome_space
    if issues:
        return make_os(issues)
    if not outcomes:
        return None
    return make_os(issues_from_outcomes(outcomes))


def ensure_os(outcome_space, issues, outcomes) -> OutcomeSpace:
    """
    Returns an outcome space from either an outcome-space, a list of issues, a list of outcomes, or the number of outcomes

    Remakrs:
        - Precedence is in the order of paramters `outcome_space` > `issues` > `outcomes`.
        - If neither is given, a `ValueError` exception will be raised.
        - Will not copy the outcome-space if it is given
        - A `CartesianOutcomeSpace` will be created if an outcome-space is not given
        - If outcomes is given or all issues are discrete, a `DiscreteCartesianOutcomeSpace` will be created
    """
    from negmas.outcomes.outcome_space import make_os

    if outcome_space:
        return outcome_space
    if issues:
        return make_os(issues)
    if not outcomes:
        raise ValueError(
            "No way to create an outcome-space: outcome_space, issses, and outcomes are all None or empty"
        )
    return make_os(issues_from_outcomes(outcomes))
