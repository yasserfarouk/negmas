"""Protocol definitions."""

from __future__ import annotations

from abc import abstractmethod
from os import PathLike
from pathlib import Path
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
class XmlSerializableUFun(Protocol):
    """Can be serialized to XML format (compatible with GENIUS)"""

    @classmethod
    @abstractmethod
    def from_xml_str(cls: type[X], xml_str: str, **kwargs) -> X:
        """Imports a utility function from a GENIUS XML string.

        Args:

            xml_str (str): The string containing GENIUS style XML utility function definition

        Returns:

            A utility function object (depending on the input file)
        """

    @abstractmethod
    def to_xml_str(self, **kwargs) -> str:
        """Exports a utility function to a well formatted string"""

    def to_genius(self, file_name: PathLike, **kwargs) -> None:
        """
        Exports a utility function to a GENIUS XML file.

        Args:

            file_name (str): File name to export to

        Returns:

            None

        Remarks:
            See ``to_xml_str`` for all the parameters

        """
        file_name = Path(file_name).absolute()
        if file_name.suffix == "":
            file_name = file_name.parent / f"{file_name.stem}.xml"
        with open(file_name, "w") as f:
            f.write(self.to_xml_str(**kwargs))

    @classmethod
    def from_genius(
        cls: type[X], issues: list[Issue], file_name: PathLike, **kwargs
    ) -> X:
        """From genius.

        Args:
            issues: Issues.
            file_name: File name.
            **kwargs: Additional keyword arguments.

        Returns:
            X: The result.
        """
        ...

    @abstractmethod
    def xml(self, issues: list[Issue]) -> str:
        """Xml.

        Args:
            issues: Issues.

        Returns:
            str: The result.
        """
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
        """Reserved distribution.

        Returns:
            Distribution: The result.
        """
        ...


@runtime_checkable
class HasReservedOutcome(Protocol):
    """In case of disagreement, the value of `reserved_outcome` will be received by the entity"""

    reserved_outcome: Outcome


@runtime_checkable
class StationaryConvertible(Protocol):
    """Can be converted to stationary Prefereences (i.e. one indepndent of the negotiation session, state or external factors). The conversion is only accurate at the instant it is done"""

    def to_stationary(self):
        """To stationary."""
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
        """Make instance callable.

        Args:
            offer: Offer being considered.

        Returns:
            Value: The result.
        """
        ...


T = TypeVar("T", bound="UFunCrisp")


@runtime_checkable
class UFunCrisp(UFun, Protocol):
    """Can be called to map an `Outcome` to a `float`"""

    def eval(self, offer: Outcome) -> float:
        """Eval.

        Args:
            offer: Offer being considered.

        Returns:
            float: The result.
        """
        ...

    def to_stationary(self: T) -> T:
        """To stationary.

        Returns:
            T: The result.
        """
        ...

    def __call__(self, offer: Outcome | None) -> float:
        """Make instance callable.

        Args:
            offer: Offer being considered.

        Returns:
            float: The result.
        """
        ...


@runtime_checkable
class UFunProb(UFun, Protocol):
    """Can be called to map an `Outcome` to a `Distribution`"""

    @abstractmethod
    def eval(self, offer: Outcome) -> Distribution:
        """Eval.

        Args:
            offer: Offer being considered.

        Returns:
            Distribution: The result.
        """
        ...

    def __call__(self, offer: Outcome | None) -> Distribution:
        """Make instance callable.

        Args:
            offer: Offer being considered.

        Returns:
            Distribution: The result.
        """
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
        """Initialize the instance.

        Args:
            ufun: Ufun.
        """
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
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        fallback_to_higher: bool = True,
        fallback_to_best: bool = True,
    ) -> Outcome | None:
        """
        Finds an outcmoe with the given utility value.

        Args:
            rng: The range (or single value) of utility values to search for outcomes
            normalized: if `True`, the input `rng` will be understood as ranging from 0-1 (1=max, 0=min) independent of the ufun actual range
            fall_back_to_higher: if `True`, any outcome above the minimum in the range will be returned if nothing can be found in the range
            fall_back_to_best: if `True`, the best outcome will always be offered if no outcome in the given range is found.
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

    def next_worse(self) -> Outcome | None:
        """Returns the rational outcome with utility just below the last one returned from this function"""
        raise NotImplementedError(
            f"next_below is not implemented for {self.__class__.__name__}"
        )

    def next_better(self) -> Outcome | None:
        """Returns the rational outcome with utility just below the last one returned from this function"""
        raise NotImplementedError(
            f"next_above is not implemented for {self.__class__.__name__}"
        )


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
        """Dim.

        Returns:
            int: The result.
        """
        ...

    def minmax(self, input) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        ...

    @abstractmethod
    def shift_by(self, offset: float) -> Fun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            Fun: The result.
        """
        ...

    @abstractmethod
    def scale_by(self, scale: float) -> Fun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            Fun: The result.
        """
        ...

    def __call__(self, x) -> float:
        """Make instance callable.

        Args:
            x: X.

        Returns:
            float: The result.
        """
        ...


@runtime_checkable
class SingleIssueFun(Fun, Protocol):
    """A value function mapping values from a **single** issue to a real number"""

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        """Xml.

        Args:
            indx: Indx.
            issue: Issue.
            bias: Bias.

        Returns:
            str: The result.
        """
        ...

    def min(self, input: Issue) -> float:
        """Min.

        Args:
            input: Input.

        Returns:
            float: The result.
        """
        ...

    def max(self, input: Issue) -> float:
        """Max.

        Args:
            input: Input.

        Returns:
            float: The result.
        """
        ...


@runtime_checkable
class MultiIssueFun(Fun, Protocol):
    """A value function mapping values from **multiple** issues to a real number"""

    @property
    def dim(self) -> int:
        """Dim.

        Returns:
            int: The result.
        """
        ...

    def minmax(self, input: tuple[Issue, ...]) -> tuple[float, float]:
        """Minmax.

        Args:
            input: Input.

        Returns:
            tuple[float, float]: The result.
        """
        ...

    def shift_by(self, offset: float) -> MultiIssueFun:
        """Shift by.

        Args:
            offset: Offset.

        Returns:
            MultiIssueFun: The result.
        """
        ...

    def scale_by(self, scale: float) -> MultiIssueFun:
        """Scale by.

        Args:
            scale: Scale.

        Returns:
            MultiIssueFun: The result.
        """
        ...

    def xml(self, indx: int, issues: list[Issue] | tuple[Issue, ...], bias=0.0) -> str:
        """Xml.

        Args:
            indx: Indx.
            issues: Issues.
            bias: Bias.

        Returns:
            str: The result.
        """
        ...

    def __call__(self, x: tuple) -> float:
        """Make instance callable.

        Args:
            x: X.

        Returns:
            float: The result.
        """
        ...


@runtime_checkable
class Shiftable(CardinalProb, Protocol):
    """Can be shifted by a constant amount (i.e. utility values are all shifted by this amount)"""

    @abstractmethod
    def shift_by(self, offset: float, shift_reserved=True) -> Shiftable:
        """Shift by.

        Args:
            offset: Offset.
            shift_reserved: Shift reserved.

        Returns:
            Shiftable: The result.
        """
        ...

    @abstractmethod
    def shift_min(self, to: float, rng: tuple[float, float] | None = None) -> Shiftable:
        """Shift min.

        Args:
            to: To.
            rng: Rng.

        Returns:
            Shiftable: The result.
        """
        ...

    @abstractmethod
    def shift_max(self, to: float, rng: tuple[float, float] | None = None) -> Shiftable:
        """Shift max.

        Args:
            to: To.
            rng: Rng.

        Returns:
            Shiftable: The result.
        """
        ...


@runtime_checkable
class Scalable(UFun, Protocol):
    """Can be scaled by a constant amount (i.e. utility values are all multiplied by this amount)"""

    @abstractmethod
    def scale_by(self, scale: float, scale_reserved=True) -> Scalable:
        """Scale by.

        Args:
            scale: Scale.
            scale_reserved: Scale reserved.

        Returns:
            Scalable: The result.
        """
        ...

    @abstractmethod
    def scale_min(self, to: float, rng: tuple[float, float] | None = None) -> Scalable:
        """Scale min.

        Args:
            to: To.
            rng: Rng.

        Returns:
            Scalable: The result.
        """
        ...

    @abstractmethod
    def scale_max(self, to: float, rng: tuple[float, float] | None = None) -> Scalable:
        """Scale max.

        Args:
            to: To.
            rng: Rng.

        Returns:
            Scalable: The result.
        """
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
        """Shift min for.

        Args:
            to: To.
            outcome_space: Outcome space.
            issues: Issues.
            outcomes: Outcomes.
            rng: Rng.

        Returns:
            PartiallyScalable: The result.
        """
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
        """Shift max for.

        Args:
            to: To.
            outcome_space: Outcome space.
            issues: Issues.
            outcomes: Outcomes.
            rng: Rng.

        Returns:
            PartiallyScalable: The result.
        """
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
        """Scale min for.

        Args:
            to: To.
            outcome_space: Outcome space.
            issues: Issues.
            outcomes: Outcomes.
            rng: Rng.

        Returns:
            PartiallyScalable: The result.
        """
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
        """Scale max for.

        Args:
            to: To.
            outcome_space: Outcome space.
            issues: Issues.
            outcomes: Outcomes.
            rng: Rng.

        Returns:
            PartiallyScalable: The result.
        """
        ...


N = TypeVar("N", bound="Normalizable")


@runtime_checkable
class Normalizable(Shiftable, Scalable, Protocol):
    """Can be normalized to a given range of values (default is 0-1)"""

    @abstractmethod
    def normalize(self: N, to: tuple[float, float] = (0.0, 1.0)) -> N:
        """Normalize.

        Args:
            to: To.

        Returns:
            N: The result.
        """
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
