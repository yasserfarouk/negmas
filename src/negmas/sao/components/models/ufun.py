"""Utility function implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

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

    def _update_private_info(self, partner_id: str | None = None) -> None:
        """Update the negotiator's private_info with this model.

        For bilateral negotiations, sets private_info["opponent_ufun"].
        For multilateral negotiations, sets private_info["opponent_ufuns"][partner_id].

        Args:
            partner_id: The partner's ID for multilateral negotiations.
                       If None, assumes bilateral and uses "opponent_ufun".
        """
        if not self.negotiator:
            return

        # Ensure private_info exists
        if not hasattr(self.negotiator, "private_info"):
            return

        private_info = self.negotiator.private_info
        if private_info is None:
            return

        # Check if this is a multilateral negotiation
        nmi = self.negotiator.nmi
        is_multilateral = nmi is not None and nmi.n_negotiators > 2

        if is_multilateral and partner_id is not None:
            # Multilateral: store in opponent_ufuns dict
            if "opponent_ufuns" not in private_info:
                private_info["opponent_ufuns"] = {}
            private_info["opponent_ufuns"][partner_id] = self
        else:
            # Bilateral: store directly
            private_info["opponent_ufun"] = self


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
        """On preferences changed.

        Args:
            changes: Changes.
        """
        if not self.negotiator or not self.negotiator.ufun:
            raise ValueError("Negotiator or ufun are not known")
        self._effective_ufun = (
            self.negotiator.ufun
            if not self.rank_only
            else RankOnlyUtilityFunction(self.negotiator.ufun)
        )
        # Update private_info so negotiators can access this model
        self._update_private_info()

    def eval(self, offer: Outcome) -> Value:
        """Eval.

        Args:
            offer: Offer being considered.

        Returns:
            Value: The result.
        """
        uo = self._effective_ufun.eval_normalized(offer, self.above_reserve) * -1 + 1.0
        mn, mx = self._effective_ufun.minmax(above_reserve=False)
        return uo * (mx - mn) + mn

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """Eval normalized.

        Args:
            offer: Offer being considered.
            above_reserve: Above reserve.
            expected_limits: Expected limits.

        Returns:
            Value: The result.
        """
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
        """Eval.

        Args:
            offer: Offer being considered.

        Returns:
            Value: The result.
        """
        raise NotImplementedError()


class FrequencyLinearUFunModel(UFunModel):
    """
    A `PartnerUfunModel` that uses a simple frequency-based model of the opponent offers assuming the ufun is `LinearAdditiveUtilityFunction` .
    """

    def eval(self, offer: Outcome) -> Value:
        """Eval.

        Args:
            offer: Offer being considered.

        Returns:
            Value: The result.
        """
        raise NotImplementedError()
