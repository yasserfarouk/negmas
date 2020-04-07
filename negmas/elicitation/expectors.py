from abc import ABC, abstractmethod

from typing import Optional, Union

from ..common import AgentMechanismInterface, MechanismState
from ..negotiators import AspirationMixin
from ..utilities import UtilityValue


class Expector(ABC):
    """
    Finds an `expectation` of a utility value.

    This is not necessarily the mathematical expected value but it can be any
    reduction method that receives a utility value and return a real number.
    """

    def __init__(self, ami: Optional[AgentMechanismInterface] = None):
        self.ami = ami

    @abstractmethod
    def is_dependent_on_negotiation_info(self) -> bool:
        """Returns `True` if the expected value depends in any way on the negotiation state/settings"""
        ...

    @abstractmethod
    def __call__(self, u: UtilityValue, state: MechanismState = None) -> float:
        ...


class StaticExpector(Expector):
    def is_dependent_on_negotiation_info(self) -> bool:
        return False

    @abstractmethod
    def __call__(self, u: UtilityValue, state: MechanismState = None) -> float:
        ...


class MeanExpector(StaticExpector):
    def __call__(self, u: UtilityValue, state: MechanismState = None) -> float:
        return u if isinstance(u, float) else float(u)


class MaxExpector(StaticExpector):
    def __call__(self, u: UtilityValue, state: MechanismState = None) -> float:
        return u if isinstance(u, float) else u.loc + u.scale


class MinExpector(StaticExpector):
    def __call__(self, u: UtilityValue, state: MechanismState = None) -> float:
        return u if isinstance(u, float) else u.loc


class BalancedExpector(Expector):
    def is_dependent_on_negotiation_info(self) -> bool:
        return True

    def __call__(self, u: UtilityValue, state: MechanismState = None) -> float:
        if state is None:
            state = self.ami.state
        if isinstance(u, float):
            return u
        else:
            return state.relative_time * u.loc + (1.0 - state.relative_time) * (
                u.loc + u.scale
            )


class AspiringExpector(Expector, AspirationMixin):
    def __init__(
        self,
        ami: Optional[AgentMechanismInterface] = None,
        max_aspiration=1.0,
        aspiration_type: Union[str, int, float] = "linear",
    ) -> bool:
        Expector.__init__(self, ami=ami)
        self.aspiration_init(
            max_aspiration=max_aspiration,
            aspiration_type=aspiration_type,
            above_reserved_value=False,
        )

    def is_dependent_on_negotiation_info(self) -> bool:
        return True

    def __call__(self, u: UtilityValue, state: MechanismState = None) -> float:
        if state is None:
            state = self.ami.state
        if isinstance(u, float):
            return u
        else:
            alpha = self.aspiration(state.relative_time)
            return alpha * u.loc + (1.0 - alpha) * (u.loc + u.scale)
