from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from attr import define

from negmas.common import Value

from .base import GBComponent

if TYPE_CHECKING:
    from negmas.gb import GBNegotiator, GBState

    from .inverter import UtilityInverter

__all__ = [
    "ConcessionRecommender",
    "KindConcessionRecommender",
]


class ConcessionRecommender(GBComponent):
    """
    Decides the level of concession to use
    """

    @abstractmethod
    def __call__(self, partner_concession: Value, state: GBState) -> float:
        """
        Returns a recommended concession in the range [0, 1]
        """


@define
class KindConcessionRecommender(ConcessionRecommender):
    """
    A simple recommender that does one small concession first then a tit-for-tat response

    Args:
        kindness: A fraction of the utility range to concede everytime no matter what.
        punish: If True, the partner will be punished by pushing our lower utility limit up if the concession (or its expectation) was negative
        initial_concession: The amount of concession to do in the first step
        must_concede: If `True` the agent is guaranteed to concede in the first step
        inverter: Used only if `must_concede` is `True` to determine the lowest level of concession possible
    """

    kindness: float = 0.0
    punish: float = True
    initial_concession: float = 0.0
    must_concede: bool = True
    inverter: UtilityInverter | None = None

    def set_inverter(self, inverter: UtilityInverter | None) -> None:
        self.inverter = inverter

    def set_negotiator(self, negotiator: GBNegotiator) -> None:
        super().set_negotiator(negotiator)
        self._did_my_kindness = False

    def __call__(self, partner_concession: Value, state: GBState) -> float:
        """
        Returns an estimate of the concession to be made given the partner_concession
        """
        # expected value of the partner concession
        pc = float(partner_concession)

        # concession goes negative if we are punishing
        concession = 0.0 if pc < 0 and not self.punish else pc

        # I do not conced in the first call
        if state.step == 0:
            return 0.0
        # I add some kindness at every step
        if state.step > 2:
            return concession + self.kindness

        # in my second call, I will do my initial concession
        concession += self.initial_concession
        # If I do not or cannot force conession, I will just return
        if not self.must_concede or not self.negotiator or not self.negotiator.ufun:
            return concession
        # I will try to make the smallest concession possible
        inv = self.negotiator.ufun.invert()
        increment = (
            0.5 * self.initial_concession if self.initial_concession > 1e-2 else 0.01
        )
        i = 0
        while concession < 1.0:
            concession += increment * (1 + i / 20)
            outcomes = inv.some((1.0 - concession, 1.0), normalized=True)
            if len(outcomes) > 1:
                break
            i += 1
        return concession
