from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from typing import List

from negmas.common import NegotiatorMechanismInterface
from negmas.helpers import Distribution, snake_case
from negmas.outcomes import Issue, Outcome
from negmas.types import NamedObject

__all__ = [
    "Preferences",
    "OrdinalPreferences",
    "ProbCardinalPreferences",
    "CardinalPreferences",
]


class Preferences(ABC, NamedObject):
    """
    Base class for all preferences
    """

    def __init__(
        self,
        reserved_outcome: Outcome = None,
        issues: list["Issue"] = None,
        ami: NegotiatorMechanismInterface = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.reserved_outocme = reserved_outcome
        self._ami = ami
        self._issues = issues if ami is None or ami.issues is None else ami.issues

    @property
    def ami(self):
        return self._ami

    @ami.setter
    def ami(self, value: NegotiatorMechanismInterface):
        if not self.issues and value:
            self._issues = value.issues
        self._ami = value

    @property
    def issues(self):
        return self._issues

    @issues.setter
    def issues(self, value: List[Issue]):
        self._issues = value

    @property
    @classmethod
    def is_static(cls) -> bool:
        """Is the ufun stationary (i.e. utility value of an outcome is a constant)?"""
        return not cls.is_dynamic

    @classmethod
    def is_dynamic(cls) -> bool:
        """Can the ufun change during the negotiation?"""
        return True

    @property
    def type(self) -> str:
        """Returns the utility_function type.

        Each class inheriting from this ``UtilityFunction`` class will have its own type. The default type is the empty
        string.

        Examples:
            >>> from negmas.preferences import *
            >>> print(LinearUtilityAggregationFunction((lambda x:x, lambda x:x)).type)
            linear_aggregation
            >>> print(MappingUtilityFunction(lambda x: x).type)
            mapping
            >>> print(NonLinearUtilityAggregationFunction([lambda x:x], f=lambda x: x).type)
            non_linear_aggregation

        Returns:
            str: utility_function type
        """
        return snake_case(
            self.__class__.__name__.replace("Function", "").replace("Utility", "")
        )

    @property
    def base_type(self) -> str:
        """Returns the utility_function base type ignoring discounting and similar wrappings."""
        from .discounted import DiscountedUtilityFunction

        u = self
        while isinstance(u, DiscountedUtilityFunction):
            u = u.ufun
        return self.type

    def is_dynamic(self):
        """
        By default all prefrences are assumed dynamic.

        For a type of preferences to be static, it needs to return the same results for
        any calls involving the same outcomes. If that is the case, the type should
        return `False` from its `is_dynamic` method. This can be used by neogtiators
        and other entities to optimize their behavior (i.e. caching, pre-sorting, etc)
        """

        return True


class OrdinalPreferences(Preferences):
    """
    The base class of all types of preferences in NegMAS.

    The most general preferences specification requires the specification of
    exactly one method: `is_better` with three possible outcomes:

        - `None` indicating that the two arguments are equally good.
        - `True` indicating the first argument is better than the second.
        - `False` indicating the first argument is worse than the second.
    """

    @abstractmethod
    def is_better_or_equal(self, first: Outcome, second: Outcome) -> bool | None:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Returns:
            None if the two arguments are equivalent, else whether first is better than second

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """

    def is_better(self, first: Outcome, second: Outcome) -> bool | None:
        """
        Compares two offers using the `ufun` returning whether the first is strictly better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Returns:
            None if the two arguments are equivalent, else whether first is better than second

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return self.is_better_or_equal(first, second) and not self.is_better_or_equal(
            second, first
        )

    def is_equivalent(self, first: Outcome, second: Outcome) -> bool | None:
        """
        Compares two offers using the `ufun` returning whether the first is strictly equivelent than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Returns:
            None if the two arguments are equivalent, else whether first is better than second

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return not self.is_better_or_equal(
            first, second
        ) and not self.is_better_or_equal(second, first)

    def is_worse_or_equal(self, first: Outcome, second: Outcome) -> bool | None:
        """
        Compares two offers using the `ufun` returning whether the first is worse or equivalent than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Returns:
            None if the two arguments are equivalent, else whether first is better than second

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return not self.is_better_or_equal(first, second)

    def is_worse(self, first: Outcome, second: Outcome) -> bool | None:
        """
        Compares two offers using the `ufun` returning whether the first is strictly worse than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Returns:
            None if the two arguments are equivalent, else whether first is better than second

        Remarks:

            - Should raise `ValueError` if the comparison cannot be done

        """
        return not self.is_better_or_equal(first, second) and self.is_better_or_equal(
            second, first
        )


class ProbCardinalPreferences(OrdinalPreferences):
    """
    Differences between outcomes are meaningfull
    """

    def __init__(
        self,
        reserved_value: float = float("-inf"),
        reduction_method="max",
        *args,
        epsilon=1e-8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._reduction_method = dict(
            mean=Distribution.mean, min=Distribution.min, max=Distribution.max
        )[reduction_method]
        self._epsilon = epsilon
        self.reserved_value = reserved_value

    def is_better_or_equal(self, first: Outcome, second: Outcome) -> bool | None:
        diff = self.utility_difference_prob(first, second)
        d = self._reduction_method(diff) if not isinstance(diff, numbers.Real) else diff
        return d >= self._epsilon

    @abstractmethod
    def utility_difference_prob(self, first: Outcome, second: Outcome) -> Distribution:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """


class CardinalPreferences(ProbCardinalPreferences):
    """
    Differences between outcomes are meaningfull
    """

    def __init__(
        self, reserved_value: float = float("-inf"), epsilon=1e-10, *args, **kwargs
    ):
        super().__init__(*args, **kwargs, epsilon=epsilon)
        self.reserved_value = reserved_value
        self._epsilon = epsilon

    def is_better_or_equal(self, first: Outcome, second: Outcome) -> bool | None:
        d = self.utility_difference(first, second)
        return d >= self._epsilon

    @abstractmethod
    def utility_difference(self, first: Outcome, second: Outcome) -> float:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """

    def utility_difference_prob(self, first: Outcome, second: Outcome) -> Distribution:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """
        return Distribution(
            type="uniform", loc=self.utility_difference(first, second), scale=0.0
        )
