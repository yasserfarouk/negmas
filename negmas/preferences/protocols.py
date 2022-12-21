from __future__ import annotations

from abc import abstractmethod
from os import PathLike
from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeVar, runtime_checkable

from negmas.common import Distribution, Value
from negmas.outcomes import Outcome, OutcomeSpace
from negmas.protocols import HasMinMax, XmlSerializable

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
    "HasRange",
    "InverseUFun",
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
        cls: type[X], issues: list[Issue], file_name: PathLike, **kwargs
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
    def is_volatile(self) -> bool:
        """
        Does the utiltiy of an outcome depend on factors outside the negotiation?


        Remarks:
            - A volatile preferences is one that can change even for the same mechanism state due to outside influence
        """

    @abstractmethod
    def is_session_dependent(self) -> bool:
        """
        Does the utiltiy of an outcome depend on the `NegotiatorMechanismInterface`?
        """

    @abstractmethod
    def is_state_dependent(self) -> bool:
        """
        Does the utiltiy of an outcome depend on the negotiation state?
        """

    @abstractmethod
    def is_stationary(self) -> bool:
        """Is the ufun stationary (i.e. utility value of an outcome is a constant)?"""


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
        ...


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
    def is_not_worse(self, first: Outcome | None, second: Outcome | None) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            state: The negotiation state at which the comparison is done

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """

    @abstractmethod
    def is_better(self, first: Outcome | None, second: Outcome | None) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """

    @abstractmethod
    def is_equivalent(self, first: Outcome | None, second: Outcome | None) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly equivelent than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """

    @abstractmethod
    def is_not_better(self, first: Outcome, second: Outcome | None) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is worse or equivalent than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """

    @abstractmethod
    def is_worse(self, first: Outcome | None, second: Outcome | None) -> bool:
        """
        Compares two offers using the `ufun` returning whether the first is strictly worse than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """


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


@runtime_checkable
class UFun(CardinalProb, Protocol):
    """Can be called to map an `Outcome` to a `Distribution` or a `float`"""

    def eval(self, offer: Outcome) -> Value:
        """
        Evaluates the ufun without normalization (See `eval_normalized` )
        """
        ...

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """
        Evaluates the ufun normalizing the result between zero and one

        Args:
            offer (Outcome | None): offer
            above_reserve (bool): If True, zero corresponds to the reserved value not the minimum
            expected_limits (bool): If True, the expectation of the utility limits will be used for normalization instead of the maximum range and minimum lowest limit

        Remarks:
            - If the maximum and the minium are equal, finite and above reserve, will return 1.0.
            - If the maximum and the minium are equal, initinte or below reserve, will return 0.0.
            - For probabilistic ufuns, a distribution will still be returned.
            - The minimum and maximum will be evaluated freshly every time. If they are already caached in the ufun, the cache will be used.

        """
        ...

    @abstractmethod
    def minmax(self) -> tuple[float, float]:
        """
        Finds the minimum and maximum for the ufun
        """

    def __call__(self, offer: Outcome | None) -> Value:
        ...


T = TypeVar("T", bound="UFunCrisp")


@runtime_checkable
class UFunCrisp(UFun, Protocol):
    """Can be called to map an `Outcome` to a `float`"""

    def eval(self, offer: Outcome) -> float:
        ...

    def to_stationary(self: T) -> T:
        ...

    def __call__(self, offer: Outcome | None) -> float:
        ...


@runtime_checkable
class UFunProb(UFun, Protocol):
    """Can be called to map an `Outcome` to a `Distribution`"""

    @abstractmethod
    def eval(self, offer: Outcome) -> Distribution:
        ...

    def __call__(self, offer: Outcome | None) -> Distribution:
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
class InverseUFun(Protocol):
    """Can be used to get one or more outcomes at a given range"""

    ufun: UFun
    initialized: bool

    def __init__(self, ufun: UFun) -> None:
        ...

    def init(self):
        """
        Used to intialize the inverse ufun. Any computationally expensive initialization should be  done here not in the constructor.
        """

    @abstractmethod
    def some(
        self, rng: float | tuple[float, float], normalized: bool, n: int | None = None
    ) -> list[Outcome]:
        """
        Finds a list of outcomes with utilities in the given range.

        Args:
            rng: The range (or single value) of utility values to search for outcomes
            normalized: if `True`, the input `rng` will be understood as ranging from 0-1 (1=max, 0=min) independent of the ufun actual range
            n: The maximum number of outcomes to return

        Remarks:
            - If the ufun outcome space is continuous a sample of outcomes is returned
            - If the ufun outcome space is discrete **all** outcomes in the range are returned
        """

    @abstractmethod
    def one_in(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        """
        Finds an outcmoe with the given utility value.

        Args:
            rng: The range (or single value) of utility values to search for outcomes
            normalized: if `True`, the input `rng` will be understood as ranging from 0-1 (1=max, 0=min) independent of the ufun actual range
        """

    @abstractmethod
    def best_in(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        """
        Finds an outcome with highest utility within the given range

        Args:
            rng: The range (or single value) of utility values to search for outcomes
            normalized: if `True`, the input `rng` will be understood as ranging from 0-1 (1=max, 0=min) independent of the ufun actual range
        """

    @abstractmethod
    def worst_in(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        """
        Finds an outcome with lowest utility within the given range

        Args:
            rng: The range (or single value) of utility values to search for outcomes
            normalized: if `True`, the input `rng` will be understood as ranging from 0-1 (1=max, 0=min) independent of the ufun actual range
        """

    @abstractmethod
    def within_fractions(self, rng: tuple[float, float]) -> list[Outcome]:
        """
        Finds outocmes within the given fractions of utility values. `rng` is always assumed to be normalized between 0-1
        """

    @abstractmethod
    def within_indices(self, rng: tuple[int, int]) -> list[Outcome]:
        """
        Finds outocmes within the given indices with the best at index 0 and the worst at largest index.

        Remarks:
            - Works only for discrete outcome spaces
        """

    @abstractmethod
    def min(self) -> float:
        """
        Finds the minimum utility value that can be returned.

        Remarks:
            - May be different from the minimum of the whole ufun if there is approximation
        """

    @abstractmethod
    def max(self) -> float:
        """
        Finds the maximum utility value that can be returned.

        Remarks:
            - May be different from the maximum of the whole ufun if there is approximation
        """

    @abstractmethod
    def worst(self) -> Outcome:
        """
        Finds the worst  outcome
        """

    @abstractmethod
    def best(self) -> Outcome:
        """
        Finds the best  outcome
        """

    @abstractmethod
    def minmax(self) -> tuple[float, float]:
        """
        Finds the minimum and maximum utility values that can be returned.

        Remarks:
            These may be different from the results of `ufun.minmax()` as they can be approximate.
        """

    @abstractmethod
    def extreme_outcomes(self) -> tuple[Outcome, Outcome]:
        """
        Finds the worst and best outcomes that can be returned.

        Remarks:
            These may be different from the results of `ufun.extreme_outcomes()` as they can be approximate.
        """

    @abstractmethod
    def __call__(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        """
        Calling an inverse ufun directly is equivalent to calling `one_in()`
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
        above_reserve=False,
    ) -> tuple[float, float]:
        """Finds the range of the given utility function for the given outcomes

        Args:
            self: The utility function
            issues: List of issues (optional)
            outcomes: A collection of outcomes (optional)
            max_cardinality: the maximum number of outcomes to try sampling (if sampling is used and outcomes are not given)
            above_reserve: If given, the minimum and maximum will be set to reserved value if they were less than it.

        Returns:
            (lowest, highest) utilities in that order

        """
        ...

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
        ...

    def max(self) -> Value:
        """
        Returns maximum utility
        """
        ...

    def min(self) -> Value:
        """
        Returns minimum utility
        """
        ...

    def best(self) -> Outcome:
        """
        Returns best outcome
        """
        ...

    def worst(self) -> Outcome:
        """
        Returns worst outcome
        """
        ...


@runtime_checkable
class IndIssues(BasePref, Protocol):
    """The utility value depends on each `Issue` value through a value function that does not depend on any other issue. (i.e. can be modeled as a `LinearAdditiveUtilityFunction`"""

    values: list[Callable[[Any], float]]
    weights: list[float]
    issues: list[Issue]


@runtime_checkable
class Fun(Protocol):
    """A value function mapping values from one or more issues to a real number"""

    @property
    def dim(self) -> int:
        ...

    def minmax(self, input: Issue) -> tuple[float, float]:
        ...

    @abstractmethod
    def shift_by(self, offset: float) -> Fun:
        ...

    @abstractmethod
    def scale_by(self, scale: float) -> Fun:
        ...

    def __call__(self, x) -> float:
        ...


@runtime_checkable
class SingleIssueFun(Fun, Protocol):
    """A value function mapping values from a **single** issue to a real number"""

    # @property
    # def dim(self) -> int:
    #     ...
    #
    # def minmax(self, input: Issue) -> tuple[float, float]:
    #     ...
    #
    # def shift_by(self, offset: float) -> Fun:
    #     ...
    #
    # def scale_by(self, scale: float) -> Fun:
    #     ...

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        ...

    def min(self, input: Issue) -> float:
        ...

    def max(self, input: Issue) -> float:
        ...


@runtime_checkable
class MultiIssueFun(Fun, Protocol):
    """A value function mapping values from **multiple** issues to a real number"""

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

    def __call__(self, x: tuple) -> float:
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


N = TypeVar("N", bound="Normalizable")


@runtime_checkable
class Normalizable(Shiftable, Scalable, Protocol):
    """Can be normalized to a given range of values (default is 0-1)"""

    @abstractmethod
    def normalize(
        self: N,
        to: tuple[float, float] = (0.0, 1.0),
    ) -> N:
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
