"""
Common data-structures for supporting the Generalized Bargaining Protocol
"""
from __future__ import annotations

from enum import IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Union

from attrs import asdict, define, field

from negmas.common import MechanismState, NegotiatorMechanismInterface
from negmas.outcomes import Outcome

if TYPE_CHECKING:
    from negmas.gb.negotiators.base import GBNegotiator

__all__ = [
    "ResponseType",
    "GBResponse",
    "GBState",
    "ThreadState",
    "NegotiatorMechanismInterface",
    "all_negotiator_types",
]

GBResponse = Union[Outcome, None, Literal["continue"]]


class ResponseType(IntEnum):
    """Possible responses to offers during negotiation."""

    ACCEPT_OFFER = 0
    REJECT_OFFER = 1
    END_NEGOTIATION = 2
    NO_RESPONSE = 3
    WAIT = 4


@define
class ThreadState:
    new_offer: Outcome | None = None
    new_responses: dict[str, ResponseType] = field(factory=dict)
    accepted_offers: list[Outcome] = field(factory=list)


@define
class GBState(MechanismState):
    threads: dict[str, ThreadState] = field(factory=dict)
    last_thread: str = ""

    @property
    def base_state(self) -> MechanismState:
        d = asdict(self)
        del d["threads"]
        del d["last_thread"]
        return MechanismState(**d)

    @classmethod
    def thread_history(cls, history: list[GBState], source: str) -> list[ThreadState]:
        return [_.threads[source] for _ in history]


@lru_cache(1)
def all_negotiator_types() -> list[GBNegotiator]:
    """
    Returns all the negotiator types defined in negmas.gb.negotiators
    """
    import negmas
    from negmas.gb.negotiators.base import GBNegotiator
    from negmas.helpers import get_class

    results = []
    for _ in dir(negmas.gb.negotiators):
        try:
            type = get_class(f"negmas.gb.negotiators.{_}")
            type()
        except:
            continue
        if issubclass(type, GBNegotiator):
            results.append(type)
    return results


def current_thread_id(state: GBState, source: str | None) -> str:
    """
    Returns the ID of the source thread if given or the last thread if not given. Will return an empty string if no such thread exists
    """
    return state.last_thread if not source else source


def current_thread_accepeted_offers(
    state: GBState, source: str | None
) -> list[Outcome] | None:
    """
    Returns the accepted offers of the thread associated with the source if given or last thread activated otherwise
    """
    thread = None
    if source:
        thread = state.threads[source]
    elif state.last_thread:
        thread = state.threads[state.last_thread]
    return thread.accepted_offers if thread else []


def get_offer(state: GBState, source: str | None) -> Outcome | None:
    from negmas.sao import SAOState

    if isinstance(state, SAOState):
        return state.current_offer
    if isinstance(state, GBState):
        tid = source if source else state.last_thread
        if not tid:
            return None
        return state.threads[tid].new_offer
    if hasattr(state, "current_offer"):
        return state.current_offer
    return None


GBAction = dict[str, Outcome | None]
"""
An action for a GB mechanism is a mapping from one or more thread IDs to outcomes to offer.
"""
