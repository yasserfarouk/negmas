"""
Common data-structures for supporting the Generalized Bargaining Protocol
"""
from __future__ import annotations

from enum import IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Union

from attr import asdict, define, field

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
    current_offer: Outcome | None = None
    new_offer: Outcome | None = None
    new_responses: dict[str, ResponseType] = field(factory=dict)


@define
class GBState(MechanismState):
    threads: dict[str, ThreadState] = field(factory=dict)

    @property
    def base_state(self) -> MechanismState:
        d = asdict(self)
        del d["threads"]
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
