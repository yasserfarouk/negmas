"""
Common data-structures for supporting the Stacked Alternating Offers Protocol
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from negmas.common import AgentMechanismInterface, MechanismState
from negmas.outcomes import Outcome, ResponseType

__all__ = [
    "SAOResponse",
    "SAOState",
    "SAOAMI",
]


@dataclass
class SAOResponse:
    """A response to an offer given by an agent in the alternating offers protocol"""

    response: ResponseType = ResponseType.NO_RESPONSE
    outcome: Optional["Outcome"] = None


@dataclass
class SAOState(MechanismState):
    current_offer: Optional["Outcome"] = None
    current_proposer: Optional[str] = None
    current_proposer_agent: Optional[str] = None
    n_acceptances: int = 0
    new_offers: List[Tuple[str, "Outcome"]] = field(default_factory=list)
    new_offerer_agents: List[str] = field(default_factory=list)


@dataclass
class SAOAMI(AgentMechanismInterface):
    end_on_no_response: bool = True
    publish_proposer: bool = True
    publish_n_acceptances: bool = False
