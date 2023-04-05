from typing import Any

import numpy as np
from attrs import define, field
from numpy.typing import NDArray

from negmas import SAOResponse
from negmas.common import NegotiatorMechanismInterface
from negmas.gb.common import GBState, ResponseType
from negmas.gb.components.acceptance import AcceptancePolicy, AcceptAnyRational
from negmas.gb.negotiators.base import GBNegotiator
from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao.common import SAOState
from negmas.sao.negotiators.base import SAONegotiator

__all__ = ["TAUNegotiatorAdapter"]


@define(frozen=False)
class UtilityAdapter:
    """
    Responsible of changing an offer to match TAU's rules by selecting the nearest outcome in self-utility.

    Args:
        ufun: The utility function (In the future, may be we can just use preferences here)
        nmi: `NegotiatorMechanismInterface` for the negotiation
        lower_switching_threshold: The relative time at which we check smaller utilities first.
        extend_negotiation: Whether to repeat last rational offers when no more rational offers are available
        rational: Only offer rational outcomes
        udiff_limit: Change the outcome only if it is possible to find an outcome with a utility within this from the input
        try_above: Try outcomes with higher utilities
        try_below: Try outcomes with lower utilities

    Remarks:
        - We always choose the outcome nearest in utility to the input whether it is higher or lower.
        - `lower_switching_threshold` Only affects the speed (and the edge case in which the input is exactly in the middle of
           the nearest outcomes above and below in terms of utility.)
    """

    ufun: BaseUtilityFunction | None = None
    nmi: NegotiatorMechanismInterface | None = None
    lower_switching_threshold = 0.75
    extend_negotiation: bool = True
    rational: bool = True
    udiff_limit: float = float("inf")
    try_above: bool = True
    try_below: bool = True
    offered: set[Outcome] = field(init=False, factory=set)
    utils: list[float] = field(init=False, default=None)
    sorted_outcomes: list[Outcome] = field(init=False, default=None)
    outcome_index: dict[Outcome, int] = field(init=False, default=dict)
    udiff: float = field(init=False, default=0.0)
    n_offers: int = field(init=False, default=0)
    n_rational: int = field(init=False, default=float("inf"))
    last_offer: Outcome | None = field(init=False, default=None)
    nearest_above: NDArray[np.integer[Any]] = field(init=False, default=None)
    nearest_below: NDArray[np.integer[Any]] = field(init=False, default=None)

    # @profile
    def __call__(self, outcome: Outcome | None) -> Outcome | None:
        if outcome is None:
            return outcome
        self.n_offers += 1
        if outcome not in self.offered:
            self.offered.add(outcome)
            self.last_offer = outcome
            return outcome
        if self.ufun is None:
            raise ValueError(
                "Unknown ufun. Cannot adapt an SAO offering policy to TAU without knowing the utility function."
            )
        if not self.nmi:
            raise ValueError(
                "Unknown NMI. Cannot adapt an SAO offering policy to TAU without knowing the utility function."
            )
        # repeat last offer if there are no more rational offers that we can use
        if self.extend_negotiation and len(self.offered) >= self.n_rational:
            return self.last_offer
        reserve = self.ufun.reserved_value
        if self.n_rational == float("inf"):
            outcomes = self.nmi.outcomes
            if outcomes is None:
                raise ValueError(
                    "Unknown outcome space. Cannot adapt an SAO offering policy to TAU without knowing the utility function."
                )
            if self.rational:
                uoutcomes = sorted(
                    (float(u), _) for _ in outcomes if (u := self.ufun(_)) >= reserve
                )
            else:
                uoutcomes = sorted((float(self.ufun(_)), _) for _ in outcomes)
            self.n_rational = len(uoutcomes)
            self.outcome_index = dict(
                zip([_[1] for _ in uoutcomes], range(len(uoutcomes)))
            )
            self.sorted_outcomes = [_[1] for _ in uoutcomes]
            self.utils = [_[0] for _ in uoutcomes]
            n = len(self.utils)
            self.nearest_above = np.arange(-1, n - 1, dtype=int)
            self.nearest_below = np.arange(1, n + 1, dtype=int)
            # assert all(_ >= self.ufun.reserved_value for _ in self.utils)
        indx = self.outcome_index[outcome]
        nearest, diff = indx, float("inf")
        u = float(self.ufun(outcome))
        # try higher utilities then lower utilities and choose the nearest
        above = (
            range(self.nearest_above[indx], -1, -1) if self.try_above else range(0, 0)
        )
        below = (
            range(self.nearest_below[indx], len(self.sorted_outcomes))
            if self.try_below
            else range(0, 0)
        )
        if self.nmi.state.relative_time <= self.lower_switching_threshold:
            lsts = above, below
            nearest_lists = self.nearest_above, self.nearest_below
        else:
            lsts = below, above
            nearest_lists = self.nearest_below, self.nearest_above
        for near, lst in zip(nearest_lists, lsts):
            for i in lst:
                current = self.sorted_outcomes[i]
                if current in self.offered:
                    near[indx] = i
                    continue
                # if util < self.ufun.reserved_value:
                #     continue
                d = abs(self.utils[i] - u)
                if d < diff:
                    nearest, diff = i, d
                    break
        # IF nearest is too far, just repeat
        if diff > self.udiff_limit:
            self.offered.add(outcome)
            self.last_offer = outcome
            return outcome
        # If we are changing the offer, record the utility difference
        if nearest != indx:
            self.udiff += diff
        outcome = self.sorted_outcomes[nearest]
        self.offered.add(outcome)
        self.last_offer = outcome
        return outcome

    @property
    def average_u_diff(self) -> float:
        if not self.n_offers:
            return 0.0
        return self.udiff / self.n_offers

    @property
    def total_u_diff(self) -> float:
        return self.udiff

    def on_preferences_changed(self, ufun: BaseUtilityFunction | None):
        self.ufun = ufun
        self.n_rational = float("inf")  # type: ignore
        self.outcome_index = dict()


# @define(frozen=False)
# class TAUOfferingAdapter(OfferingPolicy):
#     base: OfferingPolicy
#     adapter: UtilityAdapter = field(init=False, factory=UtilityAdapter)
#
#     def __call__(self, state: GBState) -> Outcome | None:
#         self.adapter.ufun = self.base.negotiator.ufun
#         self.adapter.nmi = self.base.negotiator.nmi
#         return self.adapter(super().__call__(state))
#
#     def on_preferences_changed(self, changes):
#         self.base.on_preferences_changed(changes)
#         self.adapter.on_preferences_changed(self.base.negotiator.ufun)
#
#     def before_proposing(self, state: GBState):
#         return self.base.before_proposing(state)
#
#     def after_proposing(self, state: GBState, offer: Outcome | None):
#         return self.base.after_proposing(state, offer)
#
#     def before_responding(self, state: GBState, offer: Outcome | None, source: str):
#         return self.base.before_responding(state, offer, source)
#
#     def after_responding(
#         self,
#         state: GBState,
#         offer: Outcome | None,
#         response: ResponseType,
#         source: str,
#     ):
#         return self.base.after_responding(state, offer, response, source,)
#
#     def on_partner_joined(self, partner: str):
#         return self.base.on_partner_joined(partner)
#
#     def on_partner_left(self, partner: str):
#         return self.base.on_partner_left(partner)
#
#     def on_partner_ended(self, partner: str):
#         return self.base.on_partner_ended(partner)
#
#     def on_partner_proposal(
#         self, state: GBState, partner_id: str, offer: Outcome
#     ) -> None:
#         return self.base.on_partner_proposal(state, partner_id, offer)
#
#     def on_partner_refused_to_propose(self, state: GBState, partner_id: str) -> None:
#         return self.base.on_partner_refused_to_propose(state, partner_id )
#
#     def on_partner_response(
#         self,
#         state: GBState,
#         partner_id: str,
#         outcome: Outcome | None,
#         response: ResponseType,
#     ) -> None:
#         return self.base.on_partner_response(state, partner_id, outcome, response)
#


class TAUNegotiatorAdapter(GBNegotiator):
    """Adapts any `GBNegotiator` to act as a TAU negotiator."""

    def __init__(
        self,
        *args,
        base: SAONegotiator,
        acceptance_policy: AcceptancePolicy | None = AcceptAnyRational(),
        **kwargs,
    ):
        self.base = base
        if acceptance_policy is not None:
            acceptance_policy._negotiator = base
        self.acceptance_policy = acceptance_policy
        self.adapter = UtilityAdapter()
        super().__init__(*args, **kwargs)

    def _sao_stat_from_gb_state(self, state: GBState) -> SAOState:
        if not state.last_thread:
            return SAOState(
                running=state.running,
                waiting=state.waiting,
                started=state.started,
                step=state.step,
                time=state.time,
                relative_time=state.relative_time,
                broken=state.broken,
                timedout=state.timedout,
                agreement=state.agreement,
                results=state.results,
                n_negotiators=state.n_negotiators,
                has_error=state.has_error,
                error_details=state.error_details,
            )
        thread = state.threads[state.last_thread]
        return SAOState(
            running=state.running,
            waiting=state.waiting,
            started=state.started,
            step=state.step,
            time=state.time,
            relative_time=state.relative_time,
            broken=state.broken,
            timedout=state.timedout,
            agreement=state.agreement,
            results=state.results,
            n_negotiators=state.n_negotiators,
            has_error=state.has_error,
            error_details=state.error_details,
            current_offer=thread.current_offer,
            current_proposer=state.last_thread,
            current_proposer_agent=None,
            n_acceptances=len(
                [_ for _ in thread.new_responses if _ == ResponseType.ACCEPT_OFFER]
            ),
            new_offers=[(k, _.new_offer) for k, _ in state.threads.items()],
            new_offerer_agents=[None for _ in state.threads.keys()],
            last_negotiator=state.last_thread,
        )

    def propose(self, state: GBState) -> Outcome | None:
        return self.adapter(self.base.propose(self._sao_stat_from_gb_state(state)))

    def respond(self, state: GBState, offer: Outcome, source: str) -> ResponseType:
        adapted_state = self._sao_stat_from_gb_state(state)
        response = self.base.respond(adapted_state, offer, source)
        if response == ResponseType.ACCEPT_OFFER:
            return response
        if self.acceptance_policy is not None:
            response = self.acceptance_policy(state, offer, source)
        return response

    def on_partner_proposal(self, state: GBState, *args, **kwargs) -> None:
        return self.base.on_partner_proposal(
            self._sao_stat_from_gb_state(state), *args, **kwargs
        )

    def on_partner_response(self, state: GBState, *args, **kwargs) -> None:
        return self.base.on_partner_response(
            self._sao_stat_from_gb_state(state), *args, **kwargs
        )

    def on_partner_ended(self, partner: str):
        return self.base.on_partner_ended(partner)

    # compatibility with SAOMechanism
    def __call__(self, state: GBState, *args, **kwargs) -> SAOResponse:
        return self.base(self._sao_stat_from_gb_state(state), *args, **kwargs)

    def propose_(self, state: GBState, *args, **kwargs) -> Outcome | None:
        return self.base.propose_(self._sao_stat_from_gb_state(state), *args, **kwargs)

    def respond_(self, state: GBState, *args, **kwargs) -> ResponseType:
        return self.base.respond_(self._sao_stat_from_gb_state(state), *args, **kwargs)

    def on_preferences_changed(self, *args, **kwargs):
        self.base.on_preferences_changed(*args, **kwargs)
        self.adapter.on_preferences_changed(self.base.ufun)

    def set_preferences(self, value, force=False):
        super().set_preferences(value, force=force)
        r = self.base.set_preferences(value, force)
        self.adapter.ufun = self.base.ufun
        return r

    def before_death(self, cntxt: dict[str, Any]) -> bool:
        return self.base.before_death(cntxt)

    def isin(self, negotiation_id: str | None) -> bool:
        return self.base.isin(negotiation_id)

    def add_capabilities(self, capabilities: dict) -> None:
        super().add_capabilities(capabilities)
        self.base.add_capabilities(capabilities)

    def join(
        self, nmi, state: GBState, *, preferences=None, ufun=None, role="negotiator"
    ) -> bool:
        joined = self.base.join(
            nmi=nmi,
            state=self._sao_stat_from_gb_state(state),
            preferences=preferences,
            ufun=ufun,
            role=role,
        )
        self.adapter.nmi = self.base.nmi
        self.adapter.ufun = self.base.ufun
        return joined

    @property
    def parent(self):
        """Returns the parent controller."""
        return self.base.parent

    @property
    def capabilities(self) -> dict[str, Any]:
        """Agent capabilities."""
        return self.base.capabilities

    @property
    def owner(self):
        """Returns the owner agent of the negotiator."""
        return self.base.owner

    @owner.setter
    def owner(self, owner):
        """Sets the owner."""
        self.base.owner = owner

    def is_acceptable_as_agreement(self, outcome: Outcome) -> bool:
        return self.base.is_acceptable_as_agreement(outcome)

    def remove_capability(self, name: str) -> None:
        self.base.remove_capability(name)

    def on_negotiation_start(self, state: GBState) -> None:
        return self.base.on_negotiation_start(self._sao_stat_from_gb_state(state))

    def on_round_start(self, state: GBState) -> None:
        return self.base.on_round_start(self._sao_stat_from_gb_state(state))

    def on_mechanism_error(self, state: GBState) -> None:
        return self.base.on_mechanism_error(self._sao_stat_from_gb_state(state))

    def on_round_end(self, state: GBState) -> None:
        return self.base.on_round_end(self._sao_stat_from_gb_state(state))

    def on_leave(self, state: GBState) -> None:
        return self.base.on_leave(self._sao_stat_from_gb_state(state))

    def on_negotiation_end(self, state: GBState) -> None:
        return self.base.on_negotiation_start(self._sao_stat_from_gb_state(state))

    def on_notification(self, notification, notifier: str):
        super().on_notification(notification, notifier)
        self.base.on_notification(notification, notifier)

    def cancel(self, reason=None) -> None:
        return self.base.cancel(reason)

    @property
    def preferences(self):
        """The utility function attached to that object."""
        return self.base.preferences

    @property
    def crisp_ufun(self):
        """Returns the preferences if it is a CrispUtilityFunction else
        None."""
        return self.base.crisp_ufun

    @crisp_ufun.setter
    def crisp_ufun(self, v):
        self.base.crisp_ufun = v

    @property
    def prob_ufun(self):
        """Returns the preferences if it is a ProbUtilityFunction else None."""
        return self.base.prob_ufun

    @prob_ufun.setter
    def prob_ufun(self, v):
        self.base.prob_ufun = v

    @property
    def ufun(self) -> BaseUtilityFunction | None:
        return self.base.ufun

    @ufun.setter
    def ufun(self, v: BaseUtilityFunction):
        self.base.ufun = v

    @property
    def has_preferences(self) -> bool:
        """Does the entity has an associated ufun?"""
        return self.base.has_preferences

    @property
    def has_cardinal_preferences(self) -> bool:
        return self.base.has_cardinal_preferences

    @property
    def reserved_outcome(self) -> Outcome | None:
        return self.base.reserved_outcome

    @property
    def reserved_value(self) -> float:
        return self.base.reserved_value
