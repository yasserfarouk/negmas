from typing import Any

from attr import define, field
from matplotlib.axes import itertools

from negmas import SAOResponse
from negmas.common import NegotiatorMechanismInterface
from negmas.gb.common import GBState, ResponseType
from negmas.gb.negotiators.base import GBNegotiator
from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao.common import SAOState
from negmas.sao.negotiators.base import SAONegotiator

__all__ = ["TAUNegotiatorAdapter"]


@define(frozen=False)
class UtilityAdapter:
    ufun: BaseUtilityFunction | None = None
    nmi: NegotiatorMechanismInterface | None = None
    offered: set[Outcome] = field(init=False, factory=set)
    utils: list[float] | None = field(init=False, default=None)
    sorted_outcomes: list[Outcome] | None = field(init=False, default=None)
    outcome_index: dict[Outcome, int] = field(init=False, default=dict)
    udiff_limit: float = float("inf")
    udiff: float = 0.0
    n_offers: int = 0

    def __call__(self, outcome: Outcome | None) -> Outcome | None:
        if outcome is None:
            return outcome
        self.n_offers += 1
        if outcome not in self.offered:
            self.offered.add(outcome)
            return outcome
        if self.ufun is None:
            raise ValueError(
                "Unknown ufun. Cannot adapt an SAO offering policy to TAU without knowing the utility function."
            )
        if not self.nmi:
            raise ValueError(
                "Unknown NMI. Cannot adapt an SAO offering policy to TAU without knowing the utility function."
            )
        if not self.utils or not self.sorted_outcomes:
            outcomes = self.nmi.outcomes
            if outcomes is None:
                raise ValueError(
                    "Unknown outcome space. Cannot adapt an SAO offering policy to TAU without knowing the utility function."
                )
            uoutcomes = sorted((float(self.ufun(_)), _) for _ in outcomes)
            self.outcome_index = dict(
                zip([_[1] for _ in uoutcomes], range(len(uoutcomes)))
            )
            self.sorted_outcomes = [_[1] for _ in uoutcomes]
            self.utils = [_[0] for _ in uoutcomes]
        indx = self.outcome_index[outcome]
        nearest, diff = indx, float("inf")
        u = float(self.ufun(outcome))
        for i in itertools.chain(
            range(indx - 1, -1, -1), range(indx + 1, len(self.sorted_outcomes))
        ):
            current = self.sorted_outcomes[i]
            if current in self.offered:
                continue
            if self.ufun(current) < self.ufun.reserved_value:
                continue
            d = abs(self.utils[i] - u)
            if d < diff:
                nearest, diff = i, d
        if diff > self.udiff_limit:
            self.offered.add(outcome)
            return outcome
        if nearest != indx:
            self.udiff += diff
        outcome = self.sorted_outcomes[nearest]
        self.offered.add(outcome)
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
        self.utils = self.sorted_outcomes = None
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

    def __init__(self, *args, base: SAONegotiator, **kwargs):
        self.base = base
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
        return self.base.respond(self._sao_stat_from_gb_state(state), offer, source)

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
        self, nmi, state, *, preferences=None, ufun=None, role="negotiator"
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

    def on_negotiation_start(self, state) -> None:
        return self.base.on_negotiation_start(self._sao_stat_from_gb_state(state))

    def on_round_start(self, state) -> None:
        return self.base.on_round_start(self._sao_stat_from_gb_state(state))

    def on_mechanism_error(self, state) -> None:
        return self.base.on_mechanism_error(self._sao_stat_from_gb_state(state))

    def on_round_end(self, state) -> None:
        return self.base.on_round_end(self._sao_stat_from_gb_state(state))

    def on_leave(self, state) -> None:
        return self.base.on_leave(self._sao_stat_from_gb_state(state))

    def on_negotiation_end(self, state) -> None:
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
