from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import numpy as np

from negmas.helpers import PATH, unique_name
from negmas.outcomes.ordinal_issue import OrdinalIssue

from .base_issue import DiscreteIssue, Issue, RangeIssue
from .categorical_issue import CategoricalIssue
from .common import Outcome
from .contiguous_issue import ContiguousIssue
from .issue_ops import (
    combine_issues,
    discretize_and_enumerate_issues,
    enumerate_discrete_issues,
    enumerate_issues,
    generate_issues,
    issues_from_genius,
    issues_from_outcomes,
    issues_from_xml_str,
    issues_to_genius,
    issues_to_xml_str,
    num_outcomes,
    sample_issues,
    sample_outcomes,
)

__all__ = ["OutcomeSpace"]


@dataclass
class OutcomeSpace:
    issues: list[Issue]
    name: str | None = None
    _cached_outcomes: list["Outcome"] | None = field(init=False, default=None)

    def __post_init__(self):
        if self.name is None:
            self.name = unique_name("os", add_time=False, sep=".")

    @property
    def issue_names(self) -> list[str]:
        """Returns an ordered list of issue names"""
        return [_.name for _ in self.issues]

    @property
    def cardinality(self) -> int | float:
        """The space cardinality = the number of outcomes"""
        n = self.num_outcomes()
        if n is None:
            return float("inf")
        return n

    @property
    def is_finite(self) -> bool:
        """Checks whether the space is finite"""
        n = self.num_outcomes()
        return isinstance(n, int)

    @property
    def is_compact(self) -> bool:
        """Checks whether all issues are complete ranges"""
        return all(isinstance(_, RangeIssue) for _ in self.issues)

    @property
    def is_discrete(self) -> bool:
        """Checks whether all issues are discrete"""
        return all(isinstance(_, DiscreteIssue) for _ in self.issues)

    @property
    def is_numeric(self) -> bool:
        """Checks whether all issues are numeric"""
        return all(_.is_numeric() for _ in self.issues)

    @property
    def is_integer(self) -> bool:
        """Checks whether all issues are integer"""
        return all(_.is_integer() for _ in self.issues)

    @property
    def is_float(self) -> bool:
        """Checks whether all issues are real"""
        return all(_.is_float() for _ in self.issues)

    def to_single_issue(self, numeric=False, stringify=False) -> "OutcomeSpace":
        """
        Creates a new outcome space that is a single-issue version of this one

        Args:
            numeric: If given, the output issue will be a `ContiguousIssue` otherwise it will be a `CategoricalIssue`
            stringify:  If given, the output issue will have string values. Checked only if `numeric` is `False`

        Remarks:
            - maps the agenda and ufuns to work correctly together
            - Only works if the outcome space is finite
        """
        if not self.is_finite:
            raise ValueError(
                f"Cannot convert an infinite outcome space to a single issue"
            )
        outcomes = self.enumerate_discrete()
        values = (
            range(len(outcomes))
            if numeric
            else [str(_) for _ in outcomes]
            if stringify
            else outcomes
        )
        return OutcomeSpace(
            issues=[
                ContiguousIssue(len(outcomes), name="-".join(self.issue_names))
                if numeric
                else CategoricalIssue(values, name="-".join(self.issue_names))
            ],
            name=self.name,
        )

    def __hash__(self) -> int:
        return hash((self.issues, self.name))

    def discrete_outcomes(
        self, levels: int = 10, max_n_outcomes: int | None = None
    ) -> list[Outcome]:
        """
        Returns a **stable** set of discrete outcomes covering the outcome-space

        Args:
            levels: Number of discretization levels per outcome
            max_n_outcomes: Maximum allowed number of outcomes to return (None for no limit)

        Returns:
            A list of at most `max_n_outcomes` valid outcomes

        Remarks:
            - Uses LRU caching to return the same outcomes for the same inputs.
        """
        if self._cached_outcomes is None:
            self._cached_outcomes = self.discretize_and_enumerate(
                n_discretization=levels, max_n_outcomes=max_n_outcomes
            )
        return self._cached_outcomes

    def discretize(self, levels: int = 10, inplace=True) -> "OutcomeSpace":
        """
        Discretizes the outcome space by sampling `levels` values for each continuous issue
        """
        issues = [
            OrdinalIssue(
                values=issue.alli(levels) if issue.is_uncountable() else issue.all,
                name=issue.name,
            )
            if issue.is_continuous()
            else issue
            for issue in self.issues
        ]
        if inplace:
            self.issues = issues
            return self
        return OutcomeSpace(issues=issues, name=self.name)

    @staticmethod
    def from_genius(
        file_name: PATH,
        force_single_issue=False,
        force_numeric=False,
        keep_value_names=True,
        keep_issue_names=True,
        safe_parsing=True,
        n_discretization: int | None = None,
        max_n_outcomes: int = 1_000_000,
        name: str = None,
    ) -> "OutcomeSpace":
        issues, _ = issues_from_genius(
            file_name,
            force_single_issue,
            force_numeric,
            keep_value_names,
            keep_issue_names,
            safe_parsing,
            n_discretization,
            max_n_outcomes,
        )
        return OutcomeSpace(issues, name=name)

    @staticmethod
    def from_xml_str(
        xml_str: str,
        force_single_issue=False,
        force_numeric=False,
        keep_value_names=True,
        keep_issue_names=True,
        safe_parsing=True,
        n_discretization: int | None = None,
        max_n_outcomes: int = 1_000_000,
        name=None,
    ) -> "OutcomeSpace":
        issues, _ = issues_from_xml_str(
            xml_str,
            force_single_issue,
            force_numeric,
            keep_value_names,
            keep_issue_names,
            safe_parsing,
            n_discretization,
            max_n_outcomes,
        )
        return OutcomeSpace(issues, name=name)

    @staticmethod
    def from_outcomes(
        outcomes: list[Outcome], numeric_as_ranges: bool = False, name: str = None
    ) -> "OutcomeSpace":
        return Outcomespace(
            issues_from_outcomes(outcomes, numeric_as_ranges), name=name
        )

    @staticmethod
    def generate(
        params: list[int | list[str] | tuple[int, int] | tuple[float, float]],
        counts: list[int] | None = None,
        names: list[str] | None = None,
        name: str = None,
    ) -> "OutcomeSpace":
        return OutcomeSpace(generate_issues(params, counts, names), name=name)

    def num_outcomes(self) -> int | None:
        return num_outcomes(self.issues)

    def enumerate(
        self,
        max_n_outcomes: int = None,
    ) -> list[Outcome]:
        return enumerate_issues(self.issues, max_n_outcomes)

    def enumerate_discrete(self) -> list[Outcome]:
        return enumerate_discrete_issues(self.issues)

    def discretize_and_enumerate(
        self,
        n_discretization: int = 10,
        max_n_outcomes: int = None,
    ) -> list[Outcome]:
        return discretize_and_enumerate_issues(
            self.issues, n_discretization, max_n_outcomes
        )

    def sample_outcomes(
        self,
        n_outcomes: int | None = None,
        keep_issue_names=None,
        min_per_dim=5,
        expansion_policy=None,
    ) -> list[Outcome] | None:
        return sample_outcomes(
            self.issues,
            n_outcomes,
            keep_issue_names,
            min_per_dim,
            expansion_policy,
        )

    def sample(
        self,
        n_outcomes: int,
        with_replacement: bool = True,
        fail_if_not_enough=True,
    ) -> list[Outcome]:
        return sample_issues(
            self.issues, n_outcomes, with_replacement, fail_if_not_enough
        )

    def combine(
        self,
        issue_name="combined",
        keep_issue_names=True,
        keep_value_names=True,
        issue_sep="_",
        value_sep="-",
    ) -> Issue | None:
        return combine_issues(
            self.issues,
            issue_name,
            keep_issue_names,
            keep_value_names,
            issue_sep,
            value_sep,
        )

    def to_xml_str(self, enumerate_integer: bool = False) -> str:
        return issues_to_xml_str(self.issues, enumerate_integer)

    def to_genius(self, file_name: PATH, enumerate_integer: bool = False) -> None:
        return issues_to_genius(self.issues, file_name, enumerate_integer)
