"""Negotiators base classes."""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable

from negmas import PreferencesChange, warnings
from negmas.gb.common import ExtendedResponseType, ResponseType
from negmas.gb.components.inverter import UtilityInverter
from negmas.gb.components.selectors import OfferSelector
from negmas.gb.negotiators.base import GBNegotiator
from negmas.preferences import BaseUtilityFunction, InverseUFun

__all__ = ["UtilBasedNegotiator"]


class UtilBasedNegotiator(GBNegotiator):
    """
    A negotiator that bases its decisions on the utility value of outcomes only.

    It uses an `InverseUFun` (via the `UtilityInverter` component) to find
    outcomes with utilities in a desired range, and an optional `OfferSelector`
    to pick among multiple candidate outcomes.

    Args:
        stochastic (bool): If ``False`` (default), the inverter's ``worst_in``
            is used so the negotiator proposes the outcome with the *lowest*
            utility still within its aspiration band (i.e. just above the
            aspiration level). If ``True``, ``one_in`` is used so a random
            in-range outcome is proposed.
        rank_only (bool): If ``True``, only the relative ranks of outcomes (not
            their actual utilities) are used for inversion. This maps all
            equal-utility outcomes to the same rank, which can be useful for
            non-stationary or noisy ufuns.
        ufun_inverter (Callable[[BaseUtilityFunction], InverseUFun] | None):
            A factory that constructs an `InverseUFun` from the negotiator's
            utility function. If ``None``, a `DefaultInverseUtilityFunction`
            (i.e. `AdaptiveInverseUtilityFunction`) is used.
        offer_selector (OfferSelector | None): A callable that selects one
            outcome from a sequence of candidates given the current state. If
            ``None`` and ``stochastic`` is ``True``, a random candidate is
            chosen. If ``None`` and ``stochastic`` is ``False``, ``worst_in``
            is used directly (no selection needed).
        max_cardinality (int): The number of outcomes at which the default
            inverter may switch away from exact presorting to a scalable
            approximation. Used only if ``ufun_inverter`` is ``None``.
        eps (float): A tolerance around the utility range used when sampling
            outcomes (passed to the inverter).

    Remarks:
        - ``propose`` recovers from a ``None`` inverter result (which would
          otherwise break the SAO mechanism) by falling back to the best
          outcome. This matters for **strict** inverters (e.g.
          `BruteForceInverseUtilityFunction`) whose aspiration range contains
          no outcome, and as a safety net when a clamping inverter's fallbacks
          are exhausted.
    """

    def __init__(
        self,
        *args,
        stochastic: bool = False,
        rank_only: bool = False,
        ufun_inverter: Callable[[BaseUtilityFunction], InverseUFun] | None = None,
        offer_selector: OfferSelector | None = None,
        max_cardinality: int = 10_000_000,
        eps: float = 0.0001,
        **kwargs,
    ):
        """Initialize the instance.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
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
        """Utility range to propose.

        Args:
            state: Current state.

        Returns:
            tuple[float, float]: The result.
        """
        ...

    @abstractmethod
    def utility_range_to_accept(self, state) -> tuple[float, float]:
        """Utility range to accept.

        Args:
            state: Current state.

        Returns:
            tuple[float, float]: The result.
        """
        raise NotImplementedError(
            "utility_range_to_accept not implemented by  UtilBasedNegotiator"
        )

    def respond(
        self, state, source: str | None = None
    ) -> ResponseType | ExtendedResponseType:
        """Respond.

        Args:
            state: Current state.
            source: Source identifier.

        Returns:
            ResponseType: The result.
        """
        offer = state.current_offer  # type: ignore
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

    def propose(self, state, dest: str | None = None):
        """Propose.

        Recovers from a ``None`` proposal (which would otherwise break the SAO
        mechanism — see ``negmas.sao.mechanism``) by falling back to the best
        outcome. This is important when the inverter is **strict** (e.g.
        `BruteForceInverseUtilityFunction`) and the requested aspiration range
        contains no outcome: the negotiator would rather offer its best
        outcome than break the negotiation.

        Args:
            state: Current state.
            dest: Dest.
        """
        self._inverter.before_proposing(state)
        if self.ufun is None:
            warnings.warn(
                f"TimeBased negotiators need a ufun but I am asked to offer without one ({self.name} [id:{self.id}]. Will just offer `None` waiting for next round if any"
            )
            return None
        outcome = self._inverter(self.utility_range_to_propose(state), state)
        if outcome is None:
            # The inverter found no outcome in the requested aspiration range
            # (this happens for strict inverters like BruteForce, or when a
            # clamping inverter's fallbacks are exhausted). Fall back to the
            # best outcome rather than returning None, which would break the
            # SAO mechanism (see negmas.sao.mechanism: a None proposal after a
            # rejection sets state.broken = True).
            best = self._inverter.recommender.best
            if best is None and self.ufun is not None:
                best = self.ufun.extreme_outcomes()[1]
            return best
        return outcome

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """On preferences changed.

        Args:
            changes: Changes.
        """
        self._inverter.on_preferences_changed(changes)
        if self._selector:
            self._selector.on_preferences_changed(changes)
        return super().on_preferences_changed(changes)
