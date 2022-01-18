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
    from negmas.common import MechanismState
    from negmas.outcomes.base_issue import Issue

__all__ = [
    "BasePref",
    "Ordinal",
    "CardinalProb",
    "CardinalCrisp",
    "UFun",
    "UFunCrisp",
    "UFunProb",
    "NonStationaryOrdinal",
    "NonStationaryCardinalProb",
    "NonStationaryCardinal",
    "NonStationaryUFun",
    "NonStationaryCrisp",
    "NonStationaryProb",
    "StationaryOrdinal",
    "StationaryCardinalProb",
    "StationaryCardinal",
    "StationaryUFun",
    "StationaryCrisp",
    "StationaryProb",
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
    outcome_space: OutcomeSpace | None

    @property
    def type(self) -> str:
        """Returns the preferences type."""

    @property
    def base_type(self) -> str:
        """Returns the utility_function base type ignoring discounting and similar wrappings."""

    def is_non_stationary(self):
        """
        Does the utiltiy of an outcome depend on the negotiation state?
        """

    def is_stationary(self) -> bool:
        """Is the ufun stationary (i.e. utility value of an outcome is a constant)?"""


@runtime_checkable
class HasReservedDistribution(Protocol):
    reserved_distribution: Distribution


@runtime_checkable
class HasReservedValue(Protocol):
    reserved_value: float

    @property
    def reserved_distribution(self) -> Distribution:
        return ScipyDistribution(type="uniform", loc=self.reserved_value, scale=0.0)


@runtime_checkable
class HasReservedOutcome(Protocol):
    reserved_outcome: Outcome


@runtime_checkable
class StarionaryConvertible(Protocol):
    def to_stationary(self):
        ...


@runtime_checkable
class Ordinal(Protocol):
    """
    Can be ordered (at least partially)
    """

    @abstractmethod
    def is_not_worse(self, first: Outcome, second: Outcome, **kwargs) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            state: The negotiation state at which the comparison is done

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """

    def is_better(self, first: Outcome, second: Outcome, **kwargs) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return self.is_not_worse(first, second, **kwargs) and not self.is_not_worse(
            second, first, **kwargs
        )

    def is_equivalent(self, first: Outcome, second: Outcome, **kwargs) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly equivelent than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return self.is_not_worse(first, second, **kwargs) and self.is_not_worse(
            second, first, **kwargs
        )

    def is_not_better(self, first: Outcome, second: Outcome, **kwargs) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is worse or equivalent than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return self.is_not_worse(second, first, **kwargs)

    def is_worse(self, first: Outcome, second: Outcome, **kwargs) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly worse than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return not self.is_not_worse(first, second, **kwargs)


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

    def is_not_worse(self, first: Outcome, second: Outcome, **kwargs) -> bool:
        return self.difference_prob(first, second, **kwargs) >= 0.0

    @abstractmethod
    def difference_prob(
        self, first: Outcome, second: Outcome, **kwargs
    ) -> Distribution:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """


@runtime_checkable
class CardinalCrisp(CardinalProb, Protocol):
    """
    Differences between outcomes are meaningfull
    """

    def is_not_worse(self, first: Outcome, second: Outcome, **kwargs) -> bool:
        return self.difference(first, second, **kwargs) > 0

    def difference_prob(
        self, first: Outcome, second: Outcome, **kwargs
    ) -> Distribution:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """
        return ScipyDistribution(
            loc=self.difference(first, second, **kwargs), scale=0.0, type="uniform"
        )

    @abstractmethod
    def difference(self, first: Outcome, second: Outcome, **kwargs) -> float:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """


@runtime_checkable
class NonStationaryOrdinal(Protocol):
    """
    Can be ordered (at least partially)
    """

    def is_stationary(self):
        return False

    def is_non_stationary(self):
        return True

    @abstractmethod
    def is_not_worse(
        self, first: Outcome, second: Outcome, *, state: "MechanismState"
    ) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            state: The negotiation state at which the comparison is done

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """

    def is_better(
        self, first: Outcome, second: Outcome, *, state: "MechanismState"
    ) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return self.is_not_worse(first, second, state=state) and not self.is_not_worse(
            second, first, state=state
        )

    def is_equivalent(
        self, first: Outcome, second: Outcome, *, state: "MechanismState"
    ) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly equivelent than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return self.is_not_worse(first, second, state=state) and self.is_not_worse(
            second, first, state=state
        )

    def is_not_better(
        self, first: Outcome, second: Outcome, *, state: "MechanismState"
    ) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is worse or equivalent than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return self.is_not_worse(second, first, state=state)

    def is_worse(
        self, first: Outcome, second: Outcome, *, state: "MechanismState"
    ) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly worse than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return not self.is_not_worse(first, second, state=state)


@runtime_checkable
class NonStationaryCardinalProb(NonStationaryOrdinal, Protocol):
    """
    Differences between outcomes are meaningfull but probabilistic
    """

    def is_stationary(self):
        return False

    def is_non_stationary(self):
        return True

    @abstractmethod
    def difference_prob(
        self, first: Outcome, second: Outcome, *, state: "MechanismState"
    ) -> Distribution:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """


@runtime_checkable
class NonStationaryCardinal(NonStationaryOrdinal, Protocol):
    """
    Differences between outcomes are meaningfull
    """

    def is_stationary(self):
        return False

    def is_non_stationary(self):
        return True

    def is_not_worse(
        self, first: Outcome, second: Outcome, *, state: "MechanismState"
    ) -> bool:
        return self.difference(first, second, state=state) > 0

    def difference_prob(
        self, first: Outcome, second: Outcome, *, state: "MechanismState"
    ) -> Distribution:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """
        return ScipyDistribution(
            type="uniform", loc=self.difference(first, second, state=state), scale=0.0
        )

    @abstractmethod
    def difference(
        self, first: Outcome, second: Outcome, *, state: "MechanismState"
    ) -> float:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """


@runtime_checkable
class StationaryOrdinal(Protocol):
    """
    Can be ordered (at least partially)
    """

    def is_stationary(self):
        return True

    def is_non_stationary(self):
        return False

    @abstractmethod
    def is_not_worse(self, first: Outcome, second: Outcome) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

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
class StationaryCardinalProb(StationaryOrdinal, Protocol):
    """
    Differences between outcomes are meaningfull but probabilistic
    """

    def is_stationary(self):
        return True

    def is_non_stationary(self):
        return False

    @abstractmethod
    def difference_prob(self, first: Outcome, second: Outcome) -> Distribution:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """


@runtime_checkable
class StationaryCardinal(StationaryOrdinal, Protocol):
    """
    Differences between outcomes are meaningfull
    """

    def is_stationary(self):
        return True

    def is_non_stationary(self):
        return False

    def is_not_worse(self, first: Outcome, second: Outcome) -> bool:
        return self.difference(first, second) > 0

    def difference_prob(self, first: Outcome, second: Outcome) -> Distribution:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """
        return ScipyDistribution(
            type="uniform", loc=self.difference(first, second), scale=0.0
        )

    @abstractmethod
    def difference(self, first: Outcome, second: Outcome) -> float:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """


@runtime_checkable
class UFun(CardinalProb, Protocol):
    def __call__(self, offer: Outcome | None, **kwargs) -> Value:
        ...

    def eval(self, offer: Outcome, **kwargs) -> Value:
        ...

    def difference(self, first: Outcome, second: Outcome, **kwargs) -> Value:
        u1 = self(first)
        u2 = self(second)
        if isinstance(u1, float):
            u1 = Real(u1)
        if isinstance(u2, float):
            u2 = Real(u2)
        return u1 - u2  # type: ignore

    def difference_prob(
        self, first: Outcome, second: Outcome, **kwargs
    ) -> Distribution:
        u1 = self(first)
        u2 = self(second)
        if isinstance(u1, float):
            u1 = Real(u1)
        if isinstance(u2, float):
            u2 = Real(u2)
        return u1 - u2  # type: ignore


@runtime_checkable
class UFunCrisp(UFun, Protocol):
    def __call__(self, offer: Outcome | None, **kwargs) -> float:
        ...

    def eval(self, offer: Outcome, **kwargs) -> float:
        ...


@runtime_checkable
class UFunProb(UFun, Protocol):
    def __call__(self, offer: Outcome | None, **kwargs) -> Distribution:
        ...

    def eval(self, offer: Outcome, **kwargs) -> Distribution:
        ...


@runtime_checkable
class StationaryUFun(Protocol):
    def is_stationary(self):
        return True

    def is_non_stationary(self):
        return False

    def __call__(self, offer: Outcome | None) -> Distribution | float:
        ...

    def eval(self, offer: Outcome) -> Distribution | float:
        ...


@runtime_checkable
class StationaryCrisp(StationaryUFun, Protocol):
    def is_non_stationary(self):
        return False

    def is_stationary(self):
        return True

    def __call__(self, offer: Outcome | None) -> float:
        ...

    def eval(self, offer: Outcome) -> float:
        ...


@runtime_checkable
class StationaryProb(StationaryUFun, Protocol):
    def is_non_stationary(self):
        return False

    def is_stationary(self):
        return True

    def __call__(self, offer: Outcome | None) -> Distribution:
        ...

    def eval(self, offer: Outcome) -> Distribution:
        ...


@runtime_checkable
class NonStationaryUFun(Protocol):
    def is_stationary(self):
        return False

    def is_non_stationary(self):
        return True

    def __call__(
        self, offer: Outcome | None, *, state: MechanismState
    ) -> Distribution | float:
        ...

    def eval(self, offer: Outcome, *, state: MechanismState) -> Distribution | float:
        ...


@runtime_checkable
class NonStationaryCrisp(NonStationaryUFun, Protocol):
    def is_stationary(self):
        return False

    def is_non_stationary(self):
        return True

    def __call__(self, offer: Outcome | None, *, state: MechanismState) -> float:
        ...

    def eval(self, offer: Outcome, *, state: MechanismState) -> float:
        ...


@runtime_checkable
class NonStationaryProb(NonStationaryUFun, Protocol):
    def is_stationary(self):
        return False

    def is_non_stationary(self):
        return True

    def __call__(self, offer: Outcome | None, *, state: MechanismState) -> Distribution:
        ...

    def eval(self, offer: Outcome, *, state: MechanismState) -> Distribution:
        ...


@runtime_checkable
class OrdinalRanking(Protocol):
    """
    Implements ranking of outcomes
    """

    def rank(
        self, outcomes: list[Outcome | None], descending=True
    ) -> list[list[Outcome | None]]:
        """Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:
            A list of lists of integers giving the outcome index in the input. The list is sorted by utlity value

        """

    def argrank(
        self, outcomes: list[Outcome | None], descending=True
    ) -> list[list[int | None]]:
        """Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:
            A list of lists of integers giving the outcome index in the input. The list is sorted by utlity value

        """


@runtime_checkable
class CardinalRanking(Protocol):
    """
    Implements ranking of outcomes
    """

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
    def some(
        self,
        rng: float | tuple[float, float],
    ) -> list[Outcome]:
        """
        Finds some outcomes in the given utility range
        """


@runtime_checkable
class InverseUFun(Protocol):
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
        return self(worst).min(), self(best).max()

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
    values: list[Callable[[Any], float]]
    weights: list[float]
    issues: list[Issue]


@runtime_checkable
class Fun(Protocol):
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
        return cls(**deserialize(d))

    def to_dict(self) -> dict[str, Any]:
        return serialize(vars(self))


@runtime_checkable
class MultiIssueFun(Fun, Protocol):
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
        return self.scale_min_for(to, outcome_space=self.outcome_space, rng=rng)

    def scale_max(
        self, to: float, rng: tuple[float, float] | None = None
    ) -> PartiallyScalable:
        return self.scale_max_for(to, outcome_space=self.outcome_space, rng=rng)


N = TypeVar("N", bound="Normalizable")


@runtime_checkable
class Normalizable(Shiftable, Scalable, Protocol):
    @abstractmethod
    def normalize(
        self: N,
        to: tuple[float, float] = (0.0, 1.0),
        minmax: tuple[float, float] | None = None,
    ) -> N:
        ...


@runtime_checkable
class PartiallyNormalizable(PartiallyScalable, PartiallyShiftable, Protocol):
    def normalize(
        self,
        to: tuple[float, float] = (0.0, 1.0),
        minmax: tuple[float, float] | None = None,
        **kwargs,
    ) -> HasRange:
        return self.normalize_for(
            to, outcome_space=self.outcome_space, minmax=minmax, **kwargs
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
    @classmethod
    def random(
        cls, outcome_space, reserved_value, normalized=True, **kwargs
    ) -> "Randomizable":
        """Generates a random ufun of the given type"""
