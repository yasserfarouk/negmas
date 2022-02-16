from __future__ import annotations

from typing import TYPE_CHECKING

from negmas.preferences.base_ufun import BaseUtilityFunction

from ..base import SAOComponent

if TYPE_CHECKING:
    from negmas import ResponseType, Value
    from negmas.outcomes import Outcome

__all__ = [
    "UFunModel",
    "FrequencyUFunModel",
]


class UFunModel(SAOComponent, BaseUtilityFunction):
    """
    A `SAOComponent` that can model the opponent's utility function.

    Classes implementing this ufun-model, must implement the abstract `eval()`
    method to return the utility value of an outcome. They can use any callbacks
    available to `SAOComponent` to update the model.
    """


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
