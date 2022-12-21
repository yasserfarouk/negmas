from __future__ import annotations

from abc import abstractmethod
from typing import Callable

from negmas import PreferencesChange, warnings
from negmas.gb.components.inverter import UtilityInverter
from negmas.gb.components.selectors import OfferSelector
from negmas.gb.negotiators.base import GBNegotiator
from negmas.preferences import BaseUtilityFunction, InverseUFun

from ..common import ResponseType

__all__ = [
    "UtilBasedNegotiator",
]


class UtilBasedNegotiator(GBNegotiator):
    """
    A negotiator that bases its decisions on the utility value of outcomes only.

    Args:
        inverter (UtilityInverter): A component used to keep track of the ufun inverse
        stochastic (bool): If `False` the worst outcome in the current utility range will be used
        rank_only (bool): If `True` then the ranks of outcomes not their actual utilities will be used for decision making
        max_cardinality (int): The number of outocmes at which we switch to use the slower `SamplingInverseUtilityFunction` instead of the `PresortingInverseUtilityFunction` . Will only be used if `ufun_inverter` is `None`
        eps (float): A tolearnace around the utility range used when sampling outocmes
    """

    def __init__(
        self,
        *args,
        stochastic: bool = False,
        rank_only: bool = False,
        ufun_inverter: Callable[[BaseUtilityFunction], InverseUFun] | None = None,
        offer_selector: OfferSelector | None = None,
        max_cardinality: int = 10_000,
        eps: float = 0.0001,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._inverter = UtilityInverter(
            rank_only=rank_only,
            ufun_inverter=ufun_inverter,
            max_cardinality=max_cardinality,
            eps=eps,
            offer_selector="min" if not stochastic else offer_selector,
        )
        self._inverter.set_negotiator(self)
        self._selector = offer_selector
        if self._selector:
            self._selector.set_negotiator(self)

    @abstractmethod
    def utility_range_to_propose(self, state) -> tuple[float, float]:
        ...

    @abstractmethod
    def utility_range_to_accept(self, state) -> tuple[float, float]:
        ...

    def respond(self, state, offer, source):
        if self._selector:
            self._selector.before_responding(state, offer, source)
        if self.ufun is None:
            warnings.warn(
                f"Utility based negotiators need a ufun but I am asked to respond without one ({self.name} [id:{self.id}]. Will just reject hoping that a ufun will be set later",
                warnings.NegmasUnexpectedValueWarning,
            )
            return ResponseType.REJECT_OFFER
        urange = self._inverter.scale_utilities(self.utility_range_to_accept(state))
        u = self.ufun(offer)
        if u is None:
            warnings.warn(
                f"Cannot find utility for {offer}",
                warnings.NegmasUnexpectedValueWarning,
            )
            return ResponseType.REJECT_OFFER
        if urange[0] <= u <= urange[1]:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state):
        self._inverter.before_proposing(state)
        if self.ufun is None:
            warnings.warn(
                f"TimeBased negotiators need a ufun but I am asked to offer without one ({self.name} [id:{self.id}]. Will just offer `None` waiting for next round if any"
            )
            return None
        return self._inverter(self.utility_range_to_propose(state), state)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        self._inverter.on_preferences_changed(changes)
        if self._selector:
            self._selector.on_preferences_changed(changes)
        return super().on_preferences_changed(changes)
