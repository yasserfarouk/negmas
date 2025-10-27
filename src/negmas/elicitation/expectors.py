"""Preference elicitation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from ..common import MechanismState, NegotiatorMechanismInterface, Value
from ..negotiators.helpers import PolyAspiration


class Expector(ABC):
    """
    Finds an `expectation` of a utility value.

    This is not necessarily the mathematical expected value but it can be any
    reduction method that receives a utility value and return a real number.
    """

    def __init__(self, nmi: NegotiatorMechanismInterface | None = None):
        """Initialize the instance.

        Args:
            nmi: Nmi.
        """
        self.nmi = nmi

    @abstractmethod
    def is_dependent_on_negotiation_info(self) -> bool:
        """Returns `True` if the expected value depends in any way on the negotiation state/settings"""
        ...

    @abstractmethod
    def __call__(self, u: Value, state: MechanismState = None) -> float:
        """Make instance callable.

        Args:
            u: U.
            state: Current state.

        Returns:
            float: The result.
        """
        ...


class StaticExpector(Expector):
    """StaticExpector implementation."""

    def is_dependent_on_negotiation_info(self) -> bool:
        """Check if dependent on negotiation info.

        Returns:
            bool: The result.
        """
        return False

    @abstractmethod
    def __call__(self, u: Value, state: MechanismState = None) -> float:
        """Make instance callable.

        Args:
            u: U.
            state: Current state.

        Returns:
            float: The result.
        """
        ...


class MeanExpector(StaticExpector):
    """MeanExpector implementation."""

    def __call__(self, u: Value, state: MechanismState = None) -> float:
        """Make instance callable.

        Args:
            u: U.
            state: Current state.

        Returns:
            float: The result.
        """
        return u if isinstance(u, float) else float(u)


class MaxExpector(StaticExpector):
    """MaxExpector implementation."""

    def __call__(self, u: Value, state: MechanismState = None) -> float:
        """Make instance callable.

        Args:
            u: U.
            state: Current state.

        Returns:
            float: The result.
        """
        return u if isinstance(u, float) else u.loc + u.scale


class MinExpector(StaticExpector):
    """MinExpector implementation."""

    def __call__(self, u: Value, state: MechanismState = None) -> float:
        """Make instance callable.

        Args:
            u: U.
            state: Current state.

        Returns:
            float: The result.
        """
        return u if isinstance(u, float) else u.loc


class BalancedExpector(Expector):
    """BalancedExpector implementation."""

    def is_dependent_on_negotiation_info(self) -> bool:
        """Check if dependent on negotiation info.

        Returns:
            bool: The result.
        """
        return True

    def __call__(self, u: Value, state: MechanismState = None) -> float:
        """Make instance callable.

        Args:
            u: U.
            state: Current state.

        Returns:
            float: The result.
        """
        if state is None:
            state = self.nmi.state
        if isinstance(u, float):
            return u
        else:
            return state.relative_time * u.loc + (1.0 - state.relative_time) * (
                u.loc + u.scale
            )


class AspiringExpector(Expector):
    """AspiringExpector implementation."""

    def __init__(
        self,
        nmi: NegotiatorMechanismInterface | None = None,
        max_aspiration=1.0,
        aspiration_type: (
            Literal["linear"] | Literal["conceder"] | Literal["boulware"] | float
        ) = "linear",
    ):
        """Initialize the instance.

        Args:
            nmi: Nmi.
            max_aspiration: Max aspiration.
            aspiration_type: Aspiration type.
        """
        Expector.__init__(self, nmi=nmi)
        self.__asp = PolyAspiration(max_aspiration, aspiration_type)

    def utility_at(self, x):
        """Utility at.

        Args:
            x: X.
        """
        return self.__asp.utility_at(x)

    def is_dependent_on_negotiation_info(self) -> bool:
        """Check if dependent on negotiation info.

        Returns:
            bool: The result.
        """
        return True

    def __call__(self, u: Value, state: MechanismState = None) -> float:
        """Make instance callable.

        Args:
            u: U.
            state: Current state.

        Returns:
            float: The result.
        """
        if state is None:
            state = self.nmi.state
        if isinstance(u, float):
            return u
        else:
            alpha = self.__asp.utility_at(state.relative_time)
            return alpha * u.loc + (1.0 - alpha) * (u.loc + u.scale)
