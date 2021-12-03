from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Iterable

from negmas.helpers import unique_name
from negmas.outcomes.outcome_ops import outcome_is_valid

from ..protocols import XmlSerializable
from .base_issue import DiscreteIssue, Issue
from .categorical_issue import CategoricalIssue
from .common import Outcome
from .contiguous_issue import ContiguousIssue
from .issue_ops import (
    enumerate_discrete_issues,
    issues_from_outcomes,
    issues_from_xml_str,
    issues_to_xml_str,
    sample_issues,
)
from .protocols import DiscreteOutcomeSpace, OutcomeSpace
from .range_issue import RangeIssue

__all__ = ["CartesianOutcomeSpace", "DiscreteCartesianOutcomeSpace"]


def make_os(issues: tuple[Issue, ...], name: str = None) -> CartesianOutcomeSpace:
    if all(_.is_discrete() for _ in issues):
        return DiscreteCartesianOutcomeSpace(issues, name=name)
    return CartesianOutcomeSpace(issues, name=name)


class CartesianOutcomeSpace(OutcomeSpace, XmlSerializable):
    def __init__(self, issues: tuple[Issue, ...], name: str | None = None):
        if name is None:
            name = unique_name("os", add_time=False, sep=".")
        self.name = name
        self.issues = issues

    @property
    def issue_names(self) -> list[str]:
        """Returns an ordered list of issue names"""
        return [_.name for _ in self.issues]

    @property
    def cardinality(self) -> int | float:
        """The space cardinality = the number of outcomes"""
        return reduce(mul, [_.cardinality for _ in self.issues], initial=1)

    def is_compact(self) -> bool:
        """Checks whether all issues are complete ranges"""
        return all(isinstance(_, RangeIssue) for _ in self.issues)

    def is_discrete(self) -> bool:
        """Checks whether all issues are discrete"""
        return all(isinstance(_, DiscreteIssue) for _ in self.issues)

    def is_numeric(self) -> bool:
        """Checks whether all issues are numeric"""
        return all(_.is_numeric() for _ in self.issues)

    def is_integer(self) -> bool:
        """Checks whether all issues are integer"""
        return all(_.is_integer() for _ in self.issues)

    def is_float(self) -> bool:
        """Checks whether all issues are real"""
        return all(_.is_float() for _ in self.issues)

    def to_single_issue(
        self,
        numeric=False,
        stringify=False,
        levels: int = 5,
        max_cardinality: int | float = float("inf"),
    ) -> "DiscreteCartesianOutcomeSpace":
        """
        Creates a new outcome space that is a single-issue version of this one discretizing it as needed

        Args:
            numeric: If given, the output issue will be a `ContiguousIssue` otberwise it will be a `CategoricalIssue`
            stringify:  If given, the output issue will have string values. Checked only if `numeric` is `False`
            levels: Number of levels to discretize any continuous issue

        Remarks:
            - Will discretize inifinte outcome spaces
        """
        dos = self.to_discrete(levels, max_cardinality)
        return dos.to_single_issue(numeric, stringify)

    def __hash__(self) -> int:
        return hash((tuple(self.issues), self.name))

    def to_discrete(
        self, levels: int = 10, max_cardinality: int | float = float("inf")
    ) -> "CartesianOutcomeSpace":
        """
        Discretizes the outcome space by sampling `levels` values for each continuous issue.

        The result of the discretization is stable in the sense that repeated calls will return the same output.
        """
        if max_cardinality != float("inf"):
            c = reduce(
                mul,
                [_.cardinality if _.is_discrete() else levels for _ in self.issues],
                initial=1,
            )
            if c > max_cardinality:
                raise ValueError(
                    f"Cannot convert OutcomeSpace to a discrete OutcomeSpace with at most {max_cardinality} (at least {c} outcomes are required)"
                )
        issues: tuple[DiscreteIssue, ...] = tuple(
            issue.to_discrete(
                levels if issue.is_continuous() else None,
                compact=False,
                grid=True,
                endpoints=True,
            )
            for issue in self.issues
        )
        return DiscreteCartesianOutcomeSpace(issues=issues, name=self.name)

    @classmethod
    def from_xml_str(
        cls, xml_str: str, safe_parsing=True, name=None
    ) -> "CartesianOutcomeSpace":
        issues, _ = issues_from_xml_str(
            xml_str,
            force_single_issue=False,
            force_numeric=False,
            keep_value_names=True,
            keep_issue_names=True,
            safe_parsing=safe_parsing,
            n_discretization=None,
        )
        if not issues:
            raise ValueError(f"Failed to read an issue space from an xml string")
        if all(isinstance(_, DiscreteIssue) for _ in issues):
            return DiscreteCartesianOutcomeSpace(issues, name=name)  # type: ignore
        return cls(issues, name=name)

    @staticmethod
    def from_outcomes(
        outcomes: list[Outcome],
        numeric_as_ranges: bool = False,
        issue_names: list[str] | None = None,
        name: str = None,
    ) -> "DiscreteCartesianOutcomeSpace":
        return DiscreteCartesianOutcomeSpace(
            issues_from_outcomes(outcomes, numeric_as_ranges, issue_names), name=name
        )

    def sample(
        self,
        n_outcomes: int,
        with_replacement: bool = True,
        fail_if_not_enough=True,
    ) -> Iterable[Outcome]:
        return sample_issues(
            self.issues, n_outcomes, with_replacement, fail_if_not_enough
        )

    def to_xml_str(self, enumerate_integer: bool = False) -> str:
        return issues_to_xml_str(self.issues, enumerate_integer)

    def is_valid(self, outcome: Outcome) -> bool:
        return outcome_is_valid(outcome, self.issues)


class DiscreteCartesianOutcomeSpace(DiscreteOutcomeSpace, CartesianOutcomeSpace):
    # issues: list[DiscreteIssue]

    def __init__(self, issues: tuple[Issue, ...], name: str | None = None):
        self.issues = tuple(
            _.to_discrete(n=None if _.is_discrete() else int(_.cardinality))
            for _ in issues
        )
        self.name = name

    def is_discrete(self) -> bool:
        """Checks whether all issues are discrete"""
        return True

    def to_single_issue(
        self, numeric=False, stringify=False
    ) -> "CartesianOutcomeSpace":
        """
        Creates a new outcome space that is a single-issue version of this one

        Args:
            numeric: If given, the output issue will be a `ContiguousIssue` otherwise it will be a `CategoricalIssue`
            stringify:  If given, the output issue will have string values. Checked only if `numeric` is `False`

        Remarks:
            - maps the agenda and ufuns to work correctly together
            - Only works if the outcome space is finite
        """
        outcomes = list(self.enumerate())
        values = (
            range(len(outcomes))
            if numeric
            else [str(_) for _ in outcomes]
            if stringify
            else outcomes
        )
        issue = (
            ContiguousIssue(len(outcomes), name="-".join(self.issue_names))
            if numeric
            else CategoricalIssue(values, name="-".join(self.issue_names))
        )
        return CartesianOutcomeSpace(
            issues=(issue,),
            name=self.name,
        )

    def __hash__(self) -> int:
        return hash((self.issues, self.name))

    def enumerate(self) -> Iterable[Outcome]:
        return enumerate_discrete_issues(self.issues)

    def limit_cardinality(
        self,
        max_cardinality: int | float = float("inf"),
        levels: int | float = float("inf"),
    ) -> "DiscreteOutcomeSpace":
        """
        Limits the cardinality of the outcome space to the given maximum (or the number of levels for each issue to `levels`)

        Args:
            max_cardinality: The maximum number of outcomes in the resulting space
            levels: The maximum number of levels for each issue/subissue
        """
        if self.cardinality <= max_cardinality or all(
            _.cardinality < levels for _ in self.issues
        ):
            return self
        new_levels = [_.cardinality for _ in self.issues]  # type: ignore will be corrected the next line
        new_levels = [int(_) if _ < levels else int(levels) for _ in new_levels]
        new_cardinality = reduce(mul, new_levels, 1)

        def _reduce_total_cardinality(new_levels, max_cardinality, new_cardinality):
            sort = reversed(sorted([(_, i) for i, _ in enumerate(new_levels)]))
            sorted_levels = [_[0] for _ in sort]
            indices = [_[1] for _ in sort]
            needed = new_cardinality - max_cardinality
            current = 0
            n = len(sorted_levels)
            while needed > 0 and current < n:
                nxt = n - 1
                v = sorted_levels[current]
                if v == 1:
                    continue
                for i in range(current + 1, n - 1):
                    if v == sorted_levels[i]:
                        continue
                    nxt = i
                    break
                diff = v - sorted_levels[nxt]
                if not diff:
                    diff = 1
                new_levels[indices[current]] -= 1
                max_cardinality = (max_cardinality // v) * (v - 1)
                sort = reversed(sorted([(_, i) for i, _ in enumerate(new_levels)]))
                sorted_levels = [_[0] for _ in sort]
                current = 0
                needed = new_cardinality - max_cardinality
            return new_levels

        if new_cardinality > max_cardinality:
            new_levels: list[int] = _reduce_total_cardinality(
                new_levels, max_cardinality, new_cardinality
            )
        issues: list[Issue] = []
        for j, i, issue in zip(
            new_levels, (_.cardinality for _ in self.issues), self.issues
        ):
            issues.append(issue if j >= i else issue.to_discrete(j, compact=True))
        return DiscreteCartesianOutcomeSpace(
            tuple(issues), name=self.name + f"-{max_cardinality}"
        )
