from __future__ import annotations

from abc import ABC, abstractmethod

from negmas.helpers import snake_case
from negmas.outcomes import Outcome
from negmas.outcomes.base_issue import Issue
from negmas.outcomes.outcome_space import make_os
from negmas.outcomes.protocols import OutcomeSpace
from negmas.types import NamedObject

from .protocols import BasePref, HasReservedOutcome

__all__ = ["Preferences"]


class Preferences(NamedObject, HasReservedOutcome, BasePref, ABC):
    """
    Base class for all preferences
    """

    def __init__(
        self,
        *args,
        outcome_space: OutcomeSpace = None,
        issues: tuple[Issue] = None,
        reserved_outcome: Outcome = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reserved_outocme = reserved_outcome
        self.outcome_space: OutcomeSpace | None = (
            outcome_space if issues is None else make_os(issues, name=self.name)
        )

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

    @abstractmethod
    def is_non_stationary(self):
        """
        Does the utiltiy of an outcome depend on the negotiation state?
        """

    @abstractmethod
    def is_stationary(self) -> bool:
        """Is the ufun stationary (i.e. utility value of an outcome is a constant)?"""
