from __future__ import annotations

import uuid
from enum import Enum

from attr import define, field

from .contract import Contract

__all__ = ["BreachProcessing", "Breach"]


class BreachProcessing(Enum):
    """The way breaches are to be handled"""

    NONE = 0
    """The breach should always be reported in the breach list and no re-negotiation is allowed."""
    VICTIM_THEN_PERPETRATOR = 1
    """The victim is asked to set the re-negotiation agenda then the perpetrator."""
    META_NEGOTIATION = 2
    """A meta negotiation is instantiated between victim and perpetrator to set re-negotiation issues."""


@define
class Breach:
    contract: Contract
    """The agreement being breached"""
    perpetrator: str
    """ID of the agent committing the breach"""
    type: str
    """The type of the breach. Can be one of: `refusal`, `product`, `money`, `penalty`."""
    victims: list[str] = field(factory=list)
    """Specific victims of the breach. If not given all partners in the agreement (except perpetrator) are considered
    victims"""
    level: float = 1.0
    """Breach level defaulting to full breach (a number between 0 and 1)"""
    step: int = -1
    """The simulation step at which the breach occurred"""
    id: str = field(factory=lambda: str(uuid.uuid4()), init=True)
    """Object name"""

    def as_dict(self):
        return {
            "contract": str(self.contract),
            "contract_id": self.contract.id,
            "type": self.type,
            "level": self.level,
            "id": self.id,
            "perpetrator": self.perpetrator,
            "perpetrator_type": self.perpetrator.__class__.__name__,
            "victims": [_ for _ in self.victims],
            "step": self.step,
            "resolved": None,
        }

    def __hash__(self):
        """The hash depends only on the name"""
        return self.id.__hash__()


#
#     def __str__(self):
#         return f"Breach ({self.level} {self.type}) by {self.perpetrator} on {self.contract.id} at {self.step}"
