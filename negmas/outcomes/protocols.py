from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Collection,
    Container,
    Iterable,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .base_issue import DiscreteIssue, Issue
    from .common import Outcome
    from .outcome_space import CartesianOutcomeSpace

__all__ = [
    "OutcomeSpace",
    "DiscreteOutcomeSpace",
    "IndependentIssuesOS",
    "IndependentDiscreteIssuesOS",
]


@runtime_checkable
class OutcomeSpace(Container, Protocol):
    """
    The base protocol for all outcome spaces.
    """

    def is_valid(self, outcome: Outcome) -> bool:
        """Checks if the given outcome is valid for that outcome space"""
        ...

    def are_types_ok(self, outcome: Outcome) -> bool:
        """Checks if the type of each value in the outcome is correct for the given issue"""
        ...

    def ensure_correct_types(self, outcome: Outcome) -> Outcome:
        """Returns an outcome that is guaratneed to have correct types or raises an exception"""
        ...

    @property
    def cardinality(self) -> int | float:
        """The space cardinality = the number of outcomes"""
        ...

    def is_numeric(self) -> bool:
        """Checks whether all values in all outcomes are numeric"""
        ...

    def is_integer(self) -> bool:
        """Checks whether all values in all outcomes are integers"""
        ...

    def is_float(self) -> bool:
        """Checks whether all values in all outcomes are real"""
        ...

    def to_discrete(
        self, levels: int | float = 5, max_cardinality: int | float = float("inf")
    ) -> DiscreteOutcomeSpace:
        """
        Returns a **stable** finite outcome space. If the outcome-space is already finite. It shoud return itself.

        Args:
            levels: The levels of discretization of any continuous dimension (or subdimension)
            max_cardintlity: The maximum cardinality allowed for the resulting outcomespace (if the original OS was infinite).
                             This limitation is **NOT** applied for outcome spaces that are alredy discretized. See `limit_cardinality()`
                             for a method to limit the cardinality of an already discrete space

        If called again, it should return the same discrete outcome space every time.
        """
        ...

    def sample(
        self,
        n_outcomes: int,
        with_replacement: bool = False,
        fail_if_not_enough=False,
    ) -> Iterable[Outcome]:
        """Samples up to n_outcomes with or without replacement"""
        ...

    def to_largest_discrete(
        self, levels: int, max_cardinality: int | float = float("inf"), **kwargs
    ) -> DiscreteOutcomeSpace:
        ...

    def cardinality_if_discretized(
        self,
        levels: int,
        max_cardinality: int | float = float("inf"),
    ) -> int:
        """
        Returns the cardinality if discretized the given way.
        """
        ...

    def enumerate_or_sample(
        self,
        levels: int | float = float("inf"),
        max_cardinality: int | float = float("inf"),
    ) -> Iterable[Outcome]:
        """Enumerates all outcomes if possible (i.e. discrete space) or returns `max_cardinality` different outcomes otherwise"""
        ...

    def is_discrete(self) -> bool:
        """Checks whether there are no continua components of the space"""
        ...

    def is_finite(self) -> bool:
        """Checks whether the space is finite"""
        return self.is_discrete()

    def __contains__(self, item: Outcome | OutcomeSpace | Issue) -> bool:
        ...


@runtime_checkable
class DiscreteOutcomeSpace(OutcomeSpace, Collection, Protocol):
    """
    The base protocol for all outcome spaces with a finite number of items.

    This type of outcome-space acts as a standard python `Collection` which
    means that its length can be found using `len()` and it can be iterated over
    to return outcomes.
    """

    @property
    def cardinality(self) -> int:
        """The space cardinality = the number of outcomes"""
        ...

    def enumerate(self) -> Iterable[Outcome]:
        """
        Enumerates the outcome space returning all its outcomes (or up to max_cardinality for infinite ones)
        """
        ...

    def limit_cardinality(
        self,
        max_cardinality: int | float = float("inf"),
        levels: int | float = float("inf"),
    ) -> DiscreteOutcomeSpace:
        """
        Limits the cardinality of the outcome space to the given maximum (or the number of levels for each issue to `levels`)

        Args:
            max_cardinality: The maximum number of outcomes in the resulting space
            levels: The maximum levels allowed per issue (if issues are defined for this outcome space)
        """
        ...

    def to_single_issue(
        self, numeric: bool = False, stringify: bool = True
    ) -> CartesianOutcomeSpace:
        ...

    def sample(
        self,
        n_outcomes: int,
        with_replacement: bool = False,
        fail_if_not_enough=True,
    ) -> Iterable[Outcome]:
        ...

    def is_discrete(self) -> bool:
        """Checks whether there are no continua components of the space"""
        return True

    def to_discrete(self, *args, **kwargs) -> DiscreteOutcomeSpace:
        return self

    def __iter__(self):
        return self.enumerate().__iter__()

    def __len__(self) -> int:
        return self.cardinality


@runtime_checkable
class IndependentIssuesOS(Protocol):
    """
    An Outcome-Space that is constructed from a tuple of `Issue` objects.
    """

    issues: tuple[Issue, ...]


@runtime_checkable
class IndependentDiscreteIssuesOS(Protocol):
    """
    An Outcome-Space that is constructed from a tuple of `DiscreteIssue` objects.
    """

    issues: tuple[DiscreteIssue, ...]
