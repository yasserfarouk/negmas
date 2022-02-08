from __future__ import annotations

import random
from abc import abstractmethod
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

    @abstractmethod
    def __contains__(self, item: Outcome | OutcomeSpace | Issue):
        ...

    @abstractmethod
    def is_valid(self, outcome: Outcome) -> bool:
        """Checks if the given outcome is valid for that outcome space"""

    @abstractmethod
    def are_types_ok(self, outcome: Outcome) -> bool:
        """Checks if the type of each value in the outcome is correct for the given issue"""

    @abstractmethod
    def ensure_correct_types(self, outcome: Outcome) -> Outcome:
        """Returns an outcome that is guaratneed to have correct types or raises an exception"""

    @property
    @abstractmethod
    def cardinality(self) -> int | float:
        """The space cardinality = the number of outcomes"""

    @abstractmethod
    def is_numeric(self) -> bool:
        """Checks whether all values in all outcomes are numeric"""

    @abstractmethod
    def is_integer(self) -> bool:
        """Checks whether all values in all outcomes are integers"""

    @abstractmethod
    def is_float(self) -> bool:
        """Checks whether all values in all outcomes are real"""

    @abstractmethod
    def to_discrete(
        self,
        levels: int | float = 5,
        max_cardinality: int | float = float("inf"),
        **kwargs,
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

    def to_largest_discrete(
        self, levels: int, max_cardinality: int | float = float("inf"), **kwargs
    ) -> DiscreteOutcomeSpace:
        for l in range(levels, 0, -1):
            if self.cardinality_if_discretized(levels) < max_cardinality:
                break
        else:
            raise ValueError(
                f"Cannot discretize with levels <= {levels} keeping the cardinality under {max_cardinality} Outocme space cardinality is {self.cardinality}\nOutcome space: {self}"
            )
        return self.to_discrete(l, max_cardinality, **kwargs)

    def cardinality_if_discretized(
        self,
        levels: int,
        max_cardinality: int | float = float("inf"),
        **kwargs,
    ) -> int:
        """
        Returns the cardinality if discretized the given way.
        """
        dos = self.to_discrete(levels, max_cardinality, **kwargs)
        return dos.cardinality

    @abstractmethod
    def sample(
        self,
        n_outcomes: int,
        with_replacement: bool = False,
        fail_if_not_enough=False,
    ) -> Iterable[Outcome]:
        """Samples up to n_outcomes with or without replacement"""

    def enumerate_or_sample(
        self,
        levels: int | float = float("inf"),
        max_cardinality: int | float = float("inf"),
    ) -> Iterable[Outcome]:
        """Enumerates all outcomes if possible (i.e. discrete space) or returns `max_cardinality` different outcomes otherwise"""
        if (
            levels == float("inf")
            and max_cardinality == float("inf")
            and not self.is_discrete()
        ):
            raise ValueError(
                "Cannot enumerate-or-sample an outcome space with infinite outcomes without specifying `levels` and/or `max_cardinality`"
            )
        from negmas.outcomes.outcome_space import DiscreteCartesianOutcomeSpace

        if isinstance(self, DiscreteCartesianOutcomeSpace):
            return self.enumerate()  # type: ignore We know the outcome space is correct
        if max_cardinality == float("inf"):
            return self.to_discrete(
                levels=levels, max_cardinality=max_cardinality
            ).enumerate()
        return self.sample(
            int(max_cardinality), with_replacement=False, fail_if_not_enough=False
        )

    def is_finite(self) -> bool:
        """Checks whether the space is finite"""
        return self.is_discrete()

    def is_discrete(self) -> bool:
        """Checks whether there are no continua components of the space"""
        return isinstance(self, DiscreteOutcomeSpace)

    def __hash__(self):
        """All outcome spaces must be hashable"""
        return hash(vars(self))


@runtime_checkable
class DiscreteOutcomeSpace(OutcomeSpace, Collection, Protocol):
    """
    The base protocol for all outcome spaces with a finite number of items.

    This type of outcome-space acts as a standard python `Collection` which
    means that its length can be found using `len()` and it can be iterated over
    to return outcomes.
    """

    def __len__(self) -> int:
        return self.cardinality

    def __iter__(self):
        return self.enumerate().__iter__()

    @property
    @abstractmethod
    def cardinality(self) -> int:
        """The space cardinality = the number of outcomes"""

    def is_discrete(self) -> bool:
        """Checks whether there are no continua components of the space"""
        return True

    @abstractmethod
    def enumerate(self) -> Collection[Outcome]:
        """Enumerates the outcome space returning all its outcomes (or up to max_cardinality for infinite ones)"""

    def to_discrete(self, *args, **kwargs) -> DiscreteOutcomeSpace:
        return self

    def to_single_issue(
        self, numeric: bool = False, stringify: bool = True
    ) -> "CartesianOutcomeSpace":
        from negmas.outcomes import (
            CategoricalIssue,
            ContiguousIssue,
            DiscreteCartesianOutcomeSpace,
        )

        if not self.is_discrete():
            raise ValueError(
                f"Cannot convert an infinite outcome space to a single issue"
            )
        outcomes = list(self.enumerate())
        values = (
            range(len(outcomes))
            if numeric
            else [str(_) for _ in outcomes]
            if stringify
            else outcomes
        )
        return DiscreteCartesianOutcomeSpace(
            issues=[
                ContiguousIssue(len(outcomes)) if numeric else CategoricalIssue(values)
            ],
        )

    def sample(
        self,
        n_outcomes: int,
        with_replacement: bool = False,
        fail_if_not_enough=True,
    ) -> Iterable[Outcome]:
        """
        Samples up to n_outcomes with or without replacement.

        This methor provides a base implementation that is not memory efficient.
        It will simply create a list of all outcomes using `enumerate()` and then
        samples from it. Specific outcome space types should override this method
        to improve its efficiency if possible.

        """
        outcomes = self.enumerate()
        outcomes = list(outcomes)
        if with_replacement:
            return random.choices(outcomes, k=n_outcomes)
        if fail_if_not_enough and n_outcomes > self.cardinality:
            raise ValueError("Cannot sample enough")
        random.shuffle(outcomes)
        return outcomes[:n_outcomes]

    @abstractmethod
    def limit_cardinality(
        self, max_cardinality: int | float = float("inf"), **kwargs
    ) -> "DiscreteOutcomeSpace":
        """
        Limits the cardinality of the outcome space to the given maximum (or the number of levels for each issue to `levels`)

        Args:
            max_cardinality: The maximum number of outcomes in the resulting space
            kwargs: Any extra agruments to limit individual issues for example (must have a default doing nothing)
        """


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
