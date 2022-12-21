from __future__ import annotations

from typing import TYPE_CHECKING

from attr import define, field

from negmas.preferences import RankOnlyUtilityFunction
from negmas.preferences.base_ufun import BaseUtilityFunction
from negmas.preferences.mixins import StationaryMixin

from ..base import GBComponent

if TYPE_CHECKING:
    from negmas import PreferencesChange, Value
    from negmas.outcomes import Outcome

__all__ = [
    "UFunModel",
    "FrequencyUFunModel",
    "FrequencyLinearUFunModel",
    "ZeroSumModel",
]

SENTINAL = object()


class UFunModel(GBComponent, BaseUtilityFunction):
    """
    A `SAOComponent` that can model the opponent's utility function.

    Classes implementing this ufun-model, must implement the abstract `eval()`
    method to return the utility value of an outcome. They can use any callbacks
    available to `SAOComponent` to update the model.
    """


@define
class ZeroSumModel(StationaryMixin, UFunModel):
    """
    Assumes a zero-sum negotiation (i.e. $u_o$ = $-u_s$ )

    Remarks:

        - Because some negotiators do not work well with negative ufun values, we return (max - u(w)) instead of (- u(w))
    """

    above_reserve: bool = True
    rank_only: bool = False
    _effective_ufun: BaseUtilityFunction = field(init=False, default=SENTINAL)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        if not self.negotiator or not self.negotiator.ufun:
            raise ValueError("Negotiator or ufun are not known")
        self._effective_ufun = (
            self.negotiator.ufun
            if not self.rank_only
            else RankOnlyUtilityFunction(self.negotiator.ufun)
        )

    def eval(self, offer: Outcome) -> Value:
        uo = self._effective_ufun.eval_normalized(offer, self.above_reserve) * -1 + 1.0
        mn, mx = self._effective_ufun.minmax(above_reserve=False)
        return uo * (mx - mn) + mn

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        if offer is None:
            return 0.0
        return (
            self._effective_ufun.eval_normalized(offer, above_reserve, expected_limits)
            * -1
            + 1.0
        )


class FrequencyUFunModel(UFunModel):
    """
    A `PartnerUfunModel` that uses a simple frequency-based model of the opponent offers.
    """

    def eval(self, offer: Outcome) -> Value:
        raise NotImplementedError()


class FrequencyLinearUFunModel(UFunModel):
    """
    A `PartnerUfunModel` that uses a simple frequency-based model of the opponent offers assuming the ufun is `LinearAdditiveUtilityFunction` .
    """

    def eval(self, offer: Outcome) -> Value:
        raise NotImplementedError()
