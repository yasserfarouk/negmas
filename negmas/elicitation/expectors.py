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
        self.nmi = nmi

    @abstractmethod
    def is_dependent_on_negotiation_info(self) -> bool:
        """Returns `True` if the expected value depends in any way on the negotiation state/settings"""
        ...

    @abstractmethod
    def __call__(self, u: Value, state: MechanismState = None) -> float:
        ...


class StaticExpector(Expector):
    def is_dependent_on_negotiation_info(self) -> bool:
        return False

    @abstractmethod
    def __call__(self, u: Value, state: MechanismState = None) -> float:
        ...


class MeanExpector(StaticExpector):
    def __call__(self, u: Value, state: MechanismState = None) -> float:
        return u if isinstance(u, float) else float(u)


class MaxExpector(StaticExpector):
    def __call__(self, u: Value, state: MechanismState = None) -> float:
        return u if isinstance(u, float) else u.loc + u.scale


class MinExpector(StaticExpector):
    def __call__(self, u: Value, state: MechanismState = None) -> float:
        return u if isinstance(u, float) else u.loc


class BalancedExpector(Expector):
    def is_dependent_on_negotiation_info(self) -> bool:
        return True

    def __call__(self, u: Value, state: MechanismState = None) -> float:
        if state is None:
            state = self.nmi.state
        if isinstance(u, float):
            return u
        else:
            return state.relative_time * u.loc + (1.0 - state.relative_time) * (
                u.loc + u.scale
            )


class AspiringExpector(Expector):
    def __init__(
        self,
        nmi: NegotiatorMechanismInterface | None = None,
        max_aspiration=1.0,
        aspiration_type: (
            Literal["linear"] | Literal["conceder"] | Literal["boulware"] | float
        ) = "linear",
    ):
        Expector.__init__(self, nmi=nmi)
        self.__asp = PolyAspiration(max_aspiration, aspiration_type)

    def utility_at(self, x):
        return self.__asp.utility_at(x)

    def is_dependent_on_negotiation_info(self) -> bool:
        return True

    def __call__(self, u: Value, state: MechanismState = None) -> float:
        if state is None:
            state = self.nmi.state
        if isinstance(u, float):
            return u
        else:
            alpha = self.__asp.utility_at(state.relative_time)
            return alpha * u.loc + (1.0 - alpha) * (u.loc + u.scale)
