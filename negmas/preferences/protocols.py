from __future__ import annotations

import random
from abc import abstractmethod
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    Type,
    TypeVar,
    runtime_checkable,
)

from negmas.common import Value
from negmas.helpers.prob import Distribution, Real, ScipyDistribution
from negmas.outcomes import Outcome, OutcomeSpace
from negmas.protocols import HasMinMax, XmlSerializable
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

if TYPE_CHECKING:
    from negmas.outcomes.base_issue import Issue

__all__ = [
    "BasePref",
    "Ordinal",
    "CardinalProb",
    "CardinalCrisp",
    "UFun",
    "UFunProb",
    "UFunCrisp",
    "OrdinalRanking",
    "CardinalRanking",
    "HasReservedOutcome",
    "HasReservedValue",
    "HasReservedDistribution",
    "Randomizable",
    "Scalable",
    "Shiftable",
    "PartiallyShiftable",
    "PartiallyScalable",
    "Normalizable",
    "PartiallyNormalizable",
    "HasRange",
    "InverseUFun",
    "MultiInverseUFun",
    "IndIssues",
    "XmlSerializableUFun",
    "SingleIssueFun",
    "MultiIssueFun",
]

X = TypeVar("X", bound="XmlSerializable")


@runtime_checkable
class XmlSerializableUFun(XmlSerializable, Protocol):
    """Can be serialized to XML format (compatible with GENIUS)"""

    @classmethod
    def from_genius(
        cls: Type[X], issues: list[Issue], file_name: PathLike, **kwargs
    ) -> X:
        ...

    @abstractmethod
    def xml(self, issues: list[Issue]) -> str:
        ...


@runtime_checkable
class BasePref(Protocol):
    """Base Protcol for all preferences in NegMAS. All Preferences objects implement this interface"""

    @property
    @abstractmethod
    def type(self) -> str:
        """Returns the preferences type."""

    @property
    @abstractmethod
    def base_type(self) -> str:
        """Returns the utility_function base type ignoring discounting and similar wrappings."""

    @abstractmethod
    def is_volatile(self):
        """
        Does the utiltiy of an outcome depend on factors outside the negotiation?


        Remarks:
            - A volatile preferences is one that can change even for the same mechanism state due to outside influence
        """

    @abstractmethod
    def is_session_dependent(self):
        """
        Does the utiltiy of an outcome depend on the `NegotiatorMechanismInterface`?
        """

    @abstractmethod
    def is_state_dependent(self):
        """
        Does the utiltiy of an outcome depend on the negotiation state?
        """

    def is_stationary(self) -> bool:
        """Is the ufun stationary (i.e. utility value of an outcome is a constant)?"""
        return not self.is_state_dependent() and not self.is_volatile()


@runtime_checkable
class HasReservedDistribution(Protocol):
    """In case of disagreement, a value sampled from `reserved_distribution` will be received by the entity"""

    reserved_distribution: Distribution


@runtime_checkable
class HasReservedValue(Protocol):
    """In case of disagreement, `reserved_value` will be received by the entity"""

    reserved_value: float

    @property
    def reserved_distribution(self) -> Distribution:
        return ScipyDistribution(type="uniform", loc=self.reserved_value, scale=0.0)


@runtime_checkable
class HasReservedOutcome(Protocol):
    """In case of disagreement, the value of `reserved_outcome` will be received by the entity"""

    reserved_outcome: Outcome


@runtime_checkable
class StationaryConvertible(Protocol):
    """Can be converted to stationary Prefereences (i.e. one indepndent of the negotiation session, state or external factors). The conversion is only accurate at the instant it is done"""

    def to_stationary(self):
        ...


@runtime_checkable
class Ordinal(BasePref, Protocol):
    """
    Can be ordered (at least partially)
    """

    @abstractmethod
    def is_not_worse(self, first: Outcome, second: Outcome) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            state: The negotiation state at which the comparison is done

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """

    def is_better(self, first: Outcome, second: Outcome) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return self.is_not_worse(first, second) and not self.is_not_worse(second, first)

    def is_equivalent(self, first: Outcome, second: Outcome) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly equivelent than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return self.is_not_worse(first, second) and self.is_not_worse(second, first)

    def is_not_better(self, first: Outcome, second: Outcome) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is worse or equivalent than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return self.is_not_worse(second, first)

    def is_worse(self, first: Outcome, second: Outcome) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly worse than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return not self.is_not_worse(first, second)


@runtime_checkable
class CardinalProb(Ordinal, Protocol):
    """
    Differences between outcomes are meaningfull but probabilistic.

    Remarks:
        Inheriting from this class adds `is_not_worse` implementation that is
        extremely conservative. It declares that `first` is not worse than `second`
        only if any sample form `first` is ALWAYS not worse than any sample from
        `second`.
    """

    @abstractmethod
    def difference_prob(self, first: Outcome, second: Outcome) -> Distribution:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """

    def is_not_worse(self, first: Outcome, second: Outcome) -> bool:
        return self.difference_prob(first, second) >= 0.0


@runtime_checkable
class CardinalCrisp(CardinalProb, Protocol):
    """
    Differences between outcomes are meaningfull and crisp (i.e. real numbers)
    """

    @abstractmethod
    def difference(self, first: Outcome, second: Outcome) -> float:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """

    def is_not_worse(self, first: Outcome, second: Outcome) -> bool:
        return self.difference(first, second) > 0

    def difference_prob(self, first: Outcome, second: Outcome) -> Distribution:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """
        return ScipyDistribution(
            loc=self.difference(first, second), scale=0.0, type="uniform"
        )


@runtime_checkable
class UFun(CardinalProb, Protocol):
    """Can be called to map an `Outcome` to a `Distribution` or a `float`"""

    def __call__(self, offer: Outcome | None) -> Value:
        ...

    def eval(self, offer: Outcome) -> Value:
        ...

    def difference(self, first: Outcome, second: Outcome) -> Value:
        u1 = self(first)
        u2 = self(second)
        if isinstance(u1, float):
            u1 = Real(u1)
        if isinstance(u2, float):
            u2 = Real(u2)
        return u1 - u2  # type: ignore

    def difference_prob(self, first: Outcome, second: Outcome) -> Distribution:
        u1 = self(first)
        u2 = self(second)
        if isinstance(u1, float):
            u1 = Real(u1)
        if isinstance(u2, float):
            u2 = Real(u2)
        return u1 - u2  # type: ignore


@runtime_checkable
class UFunCrisp(UFun, Protocol):
    """Can be called to map an `Outcome` to a `float`"""

    def __call__(self, offer: Outcome | None) -> float:
        ...

    def eval(self, offer: Outcome) -> float:
        ...


@runtime_checkable
class UFunProb(UFun, Protocol):
    """Can be called to map an `Outcome` to a `Distribution`"""

    def __call__(self, offer: Outcome | None) -> Distribution:
        ...

    def eval(self, offer: Outcome) -> Distribution:
        ...


@runtime_checkable
class OrdinalRanking(Protocol):
    """Outcomes can be ranked. Supports equality"""

    @abstractmethod
    def rank(
        self, outcomes: list[Outcome | None], descending=True
    ) -> list[list[Outcome | None]]:
        """Ranks the given list of outcomes with weights. `None` stands for the null outcome.

        Returns:
            A list of lists of integers giving the outcome index in the input. The list is sorted by utlity value

        """

    @abstractmethod
    def argrank(
        self, outcomes: list[Outcome | None], descending=True
    ) -> list[list[int | None]]:
        """Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:
            A list of lists of integers giving the outcome index in the input. The list is sorted by utlity value

        """


@runtime_checkable
class CardinalRanking(Protocol):
    """Implements ranking of outcomes with meaningful differences (i.e. each rank is given a value and nearer values are more similar)"""

    @abstractmethod
    def rank_with_weights(
        self, outcomes: list[Outcome | None], descending=True
    ) -> list[tuple[tuple[Outcome | None], float]]:
        """
        Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:

            - A list of tuples each with two values:
                - an list of integers giving the index in the input array (outcomes) of an outcome (at the given utility level)
                - the weight of that outcome
            - The list is sorted by weights descendingly

        """

    @abstractmethod
    def argrank_with_weights(
        self, outcomes: list[Outcome | None], descending=True
    ) -> list[tuple[tuple[Outcome | None], float]]:
        """
        Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:

            - A list of tuples each with two values:
                - an list of integers giving the index in the input array (outcomes) of an outcome (at the given utility level)
                - the weight of that outcome
            - The list is sorted by weights descendingly

        """


@runtime_checkable
class MultiInverseUFun(Protocol):
    """Can be used to get some outcomes in a given utility range"""

    @abstractmethod
    def some(self, rng: float | tuple[float, float]) -> list[Outcome]:
        """
        Finds some outcomes in the given utility range
        """


@runtime_checkable
class InverseUFun(Protocol):
    """Can be used to get one or more outcomes at a given range"""

    @abstractmethod
    def some(
        self, rng: float | tuple[float, float], n: int | None = None
    ) -> list[Outcome]:
        """
        Finds a list of outcomes with utilities in the given range.

        Args:
            rng: The range (or single value) of utility values to search for outcomes
            n: The maximum number of outcomes to return

        Remarks:
            - If the ufun outcome space is continuous a sample of outcomes is returned
            - If the ufun outcome space is discrete **all** outcomes in the range are returned
        """

    def one_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        """
        Finds an outcmoe with the given utility value
        """
        return self.worst_in(rng) if random.random() < 0.5 else self.best_in(rng)

    def best_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        """
        Finds an outcome with highest utility within the given range
        """

    def worst_in(self, rng: float | tuple[float, float]) -> Outcome | None:
        """
        Finds an outcome with lowest utility within the given range
        """


@runtime_checkable
class HasRange(HasMinMax, UFun, Protocol):
    """Has a defined range of utility values (a minimum and a maximum) and defined best and worst outcomes"""

    def minmax(
        self,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | int | None = None,
        max_cardinality=1000,
    ) -> tuple[float, float]:
        """Finds the range of the given utility function for the given outcomes

        Args:
            self: The utility function
            issues: List of issues (optional)
            outcomes: A collection of outcomes (optional)
            max_cardinality: the maximum number of outcomes to try sampling (if sampling is used and outcomes are not
                            given)

        Returns:
            (lowest, highest) utilities in that order

        """
        (worst, best) = self.extreme_outcomes(
            outcome_space, issues, outcomes, max_cardinality
        )
        if isinstance(self, UFunCrisp):
            return self(worst), self(best)
        w, b = self(worst), self(best)
        if isinstance(w, Distribution):
            w = w.min
        if isinstance(b, Distribution):
            b = b.max
        return w, b

    @abstractmethod
    def extreme_outcomes(
        self,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | int | None = None,
        max_cardinality=1000,
    ) -> tuple[Outcome, Outcome]:
        """Finds the best and worst outcomes

        Args:
            ufun: The utility function
            issues: list of issues (optional)
            outcomes: A collection of outcomes (optional)
            max_cardinality: the maximum number of outcomes to try sampling (if sampling is used and outcomes are not
                            given)
        Returns:
            (worst, best) outcomes

        """

    @property
    def max_value(self):
        _, mx = self.minmax()
        return mx

    @property
    def min_value(self):
        mn, _ = self.minmax()
        return mn


@runtime_checkable
class IndIssues(BasePref, Protocol):
    """The utility value depends on each `Issue` value through a value function that does not depend on any other issue. (i.e. can be modeled as a `LinearAdditiveUtilityFunction`"""

    values: list[Callable[[Any], float]]
    weights: list[float]
    issues: list[Issue]


@runtime_checkable
class Fun(Protocol):
    """A value function mapping values from one or more issues to a real number"""

    def __call__(self, x) -> float:
        ...

    @property
    def dim(self) -> int:
        ...

    def minmax(self, input) -> tuple[float, float]:
        ...

    @abstractmethod
    def shift_by(self, offset: float) -> Fun:
        ...

    @abstractmethod
    def scale_by(self, scale: float) -> Fun:
        ...


@runtime_checkable
class SingleIssueFun(Fun, Protocol):
    """A value function mapping values from a **single** issue to a real number"""

    @property
    def dim(self) -> int:
        return 1

    def minmax(self, input: Issue) -> tuple[float, float]:
        ...

    def shift_by(self, offset: float) -> Fun:
        ...

    def scale_by(self, scale: float) -> Fun:
        ...

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        ...

    @classmethod
    def from_dict(cls, d: dict) -> "SingleIssueFun":
        if isinstance(d, cls):
            return d
        _ = d.pop(PYTHON_CLASS_IDENTIFIER, None)
        return cls(**deserialize(d))  # type: ignore Concrete classes will have constructor params

    def to_dict(self) -> dict[str, Any]:
        return serialize(vars(self))  # type: ignore Not sure but it should be OK


@runtime_checkable
class MultiIssueFun(Fun, Protocol):
    """A value function mapping values from **multiple** issues to a real number"""

    def __call__(self, x: tuple) -> float:
        ...

    @property
    def dim(self) -> int:
        ...

    def minmax(self, input: tuple[Issue]) -> tuple[float, float]:
        ...

    def shift_by(self, offset: float, change_bias: bool = False) -> MultiIssueFun:
        ...

    def scale_by(self, scale: float) -> MultiIssueFun:
        ...

    def xml(self, indx: int, issues: list[Issue], bias=0.0) -> str:
        ...


@runtime_checkable
class Shiftable(CardinalProb, Protocol):
    """Can be shifted by a constant amount (i.e. utility values are all shifted by this amount)"""

    @abstractmethod
    def shift_by(self, offset: float, shift_reserved=True) -> Shiftable:
        ...

    @abstractmethod
    def shift_min(self, to: float, rng: tuple[float, float] | None = None) -> Shiftable:
        ...

    @abstractmethod
    def shift_max(self, to: float, rng: tuple[float, float] | None = None) -> Shiftable:
        ...


@runtime_checkable
class Scalable(UFun, Protocol):
    """Can be scaled by a constant amount (i.e. utility values are all multiplied by this amount)"""

    @abstractmethod
    def scale_by(self, scale: float, scale_reserved=True) -> Scalable:
        ...

    @abstractmethod
    def scale_min(self, to: float, rng: tuple[float, float] | None = None) -> Scalable:
        ...

    @abstractmethod
    def scale_max(self, to: float, rng: tuple[float, float] | None = None) -> Scalable:
        ...


@runtime_checkable
class PartiallyShiftable(Scalable, Protocol):
    """Can be shifted by a constant amount for a specific part of the outcome space"""

    @abstractmethod
    def shift_min_for(
        self,
        to,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> PartiallyScalable:
        ...

    @abstractmethod
    def shift_max_for(
        self,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> PartiallyScalable:
        ...


@runtime_checkable
class PartiallyScalable(Scalable, BasePref, Protocol):
    """Can be scaled by a constant amount for a specific part of the outcome space"""

    @abstractmethod
    def scale_min_for(
        self,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> PartiallyScalable:
        ...

    @abstractmethod
    def scale_max_for(
        self,
        to: float,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
        rng: tuple[float, float] | None = None,
    ) -> PartiallyScalable:
        ...

    def scale_min(
        self, to: float, rng: tuple[float, float] | None = None
    ) -> PartiallyScalable:
        return self.scale_min_for(to, outcome_space=self.outcome_space, rng=rng)  # type: ignore

    def scale_max(
        self, to: float, rng: tuple[float, float] | None = None
    ) -> PartiallyScalable:
        return self.scale_max_for(to, outcome_space=self.outcome_space, rng=rng)  # type: ignore


N = TypeVar("N", bound="Normalizable")


@runtime_checkable
class Normalizable(Shiftable, Scalable, Protocol):
    """Can be normalized to a given range of values (default is 0-1)"""

    @abstractmethod
    def normalize(
        self: N,
        to: tuple[float, float] = (0.0, 1.0),
        minmax: tuple[float, float] | None = None,
    ) -> N:
        ...


@runtime_checkable
class PartiallyNormalizable(PartiallyScalable, PartiallyShiftable, Protocol):
    """Can be normalized to a given range of values for a given part of the outcome space (default is 0-1)"""

    def normalize(
        self,
        to: tuple[float, float] = (0.0, 1.0),
        minmax: tuple[float, float] | None = None,
        **kwargs,
    ) -> HasRange:
        return self.normalize_for(
            to, outcome_space=self.outcome_space, minmax=minmax, **kwargs  # type: ignore
        )

    def normalize_for(
        self,
        to: tuple[float, float] = (0.0, 1.0),
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | int | None = None,
        minmax: tuple[float, float] | None = None,
        **kwargs,
    ) -> HasRange:
        ...


@runtime_checkable
class Randomizable(Protocol):
    """Random Preferences of this type can be created using a `random` method"""

    @classmethod
    @abstractmethod
    def random(
        cls, outcome_space, reserved_value, normalized=True, **kwargs
    ) -> Randomizable:
        """Generates a random ufun of the given type"""
