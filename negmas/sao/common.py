"""
Common data-structures for supporting the Stacked Alternating Offers Protocol
"""
from __future__ import annotations

from collections import namedtuple
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING

from attr import define, field

from negmas.common import NegotiatorMechanismInterface
from negmas.gb.common import GBState, ResponseType

if TYPE_CHECKING:
    from negmas.outcomes import Outcome
    from negmas.sao.negotiators.base import SAONegotiator

__all__ = [
    "ResponseType",
    "SAOResponse",
    "SAOState",
    "SAONMI",
    "all_negotiator_types",
]


@define
class SAOResponse:
    """A response to an offer given by an agent in the alternating offers protocol"""

    response: ResponseType = ResponseType.NO_RESPONSE
    outcome: Outcome | None = None


@define
class SAOState(GBState):
    current_offer: Outcome | None = None
    current_proposer: str | None = None
    current_proposer_agent: str | None = None
    n_acceptances: int = 0
    new_offers: list[tuple[str, Outcome | None]] = field(default=list)
    new_offerer_agents: list[str] = field(default=list)
    last_negotiator: str | None = None


@define
class SAONMI(NegotiatorMechanismInterface):
    end_on_no_response: bool = True


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
        except:
            continue
        if issubclass(type, SAONegotiator):
            results.append(type)
    return results
