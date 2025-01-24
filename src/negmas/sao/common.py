"""
Common data-structures for supporting the Stacked Alternating Offers Protocol
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from attrs import define, field

from negmas.common import NegotiatorMechanismInterface, MechanismAction
from negmas.gb.common import GBState, ResponseType

if TYPE_CHECKING:
    from negmas.outcomes import Outcome
    from negmas.sao.negotiators.base import SAONegotiator

__all__ = ["ResponseType", "SAOResponse", "SAOState", "SAONMI", "all_negotiator_types"]


@define
class SAOResponse(MechanismAction):
    """A response to an offer given by a negotiator in the alternating offers protocol"""

    response: ResponseType = ResponseType.NO_RESPONSE
    outcome: Outcome | None = None
    data: dict[str, Any] | None = None


@define
class SAOState(GBState):
    """The `MechanismState` of SAO"""

    current_offer: Outcome | None = None
    current_proposer: str | None = None
    current_proposer_agent: str | None = None
    n_acceptances: int = 0
    new_offers: list[tuple[str, Outcome | None]] = field(factory=list)
    new_offerer_agents: list[str | None] = field(factory=list)
    last_negotiator: str | None = None
    current_data: dict[str, Any] | None = None
    new_data: list[tuple[str, dict[str, Any] | None]] = field(factory=list)


@define(frozen=True)
class SAONMI(NegotiatorMechanismInterface):
    """The `NegotiatorMechanismInterface` of SAO"""

    end_on_no_response: bool = True
    """End the negotiation if any agent responded with None"""

    one_offer_per_step: bool = False
    """If true, a step should be atomic with only one action from one negotiator"""

    @property
    def state(self) -> SAOState:
        return self._mechanism.state  # type: ignore

    @property
    def history(self) -> list[SAOState]:
        return self._mechanism.history

    @property
    def extended_trace(self) -> list[tuple[int, str, Outcome]]:
        """Returns the negotiation history as a list of step, negotiator, offer tuples"""
        return self._mechanism.extended_trace  # type: ignore

    @property
    def trace(self) -> list[tuple[str, Outcome]]:
        """Returns the negotiation history as a list of negotiator, offer tuples"""
        return self._mechanism.trace  # type: ignore

    @property
    def offers(self) -> list[Outcome]:
        """Returns offers exchanged in order"""
        return self._mechanism.offers  # type: ignore

    def negotiator_offers(self, negotiator_id: str) -> list[Outcome]:
        return self._mechanism.negotiator_offers(negotiator_id)  # type: ignore


@lru_cache(1)
def all_negotiator_types() -> list[SAONegotiator]:
    """
    Returns all the negotiator types defined in negmas.sao.negotiators
    """
    import negmas
    from negmas.helpers import get_class
    from negmas.sao import SAONegotiator

    results = []
    for _ in dir(negmas.sao.negotiators):
        try:
            type = get_class(f"negmas.sao.negotiators.{_}")
            type()
        except Exception:
            continue
        if issubclass(type, SAONegotiator):
            results.append(type)
    return results
