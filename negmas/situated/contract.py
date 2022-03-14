from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from attr import define, field

if TYPE_CHECKING:
    from negmas.common import MechanismState
    from negmas.outcomes import Issue, Outcome, OutcomeSpace

__all__ = ["Contract"]


@define
class Contract:
    """A agreement definition which encapsulates an agreement with partners and extra information"""

    partners: tuple[str] = field(factory=tuple, converter=tuple)
    """The partners"""
    agreement: Outcome | OutcomeSpace | None = field(default=None, hash=False)
    """The actual agreement of the negotiation in the form of an `Outcome` in the `Issue` space defined by `issues`"""
    annotation: dict[str, Any] = field(factory=dict, hash=False)
    """Misc. information to be kept with the agreement."""
    issues: tuple[Issue] = field(factory=tuple, converter=tuple)
    """Issues of the negotiations from which this agreement was concluded. It may be empty"""
    signed_at: int = -1
    """The time-step at which the contract was signed"""
    executed_at: int = -1
    """The time-step at which the contract was executed/breached"""
    concluded_at: int = -1
    """The time-step at which the contract was concluded (but it is still not binding until signed)"""
    nullified_at: int = -1
    """The time-step at which the contract was nullified after being signed. That can happen if a partner declares
    bankruptcy"""
    to_be_signed_at: int = -1
    """The time-step at which the contract should be signed"""
    signatures: dict[str, str | None] = field(factory=dict, hash=False)
    """A mapping from each agent to its signature"""
    mechanism_state: MechanismState | None = field(default=None, hash=False)
    """The mechanism state at the contract conclusion"""
    mechanism_id: str | None = None
    """The Id of the mechanism that led to this contract"""
    id: str = field(factory=lambda: str(uuid.uuid4()), init=True)
    """Object name"""

    def __hash__(self):
        """The hash depends only on the name"""
        if hasattr(self, "id"):
            return self.id.__hash__()
        s = super()
        if s:
            return s.__hash__()
        return None


#
#     def __str__(self):
#         return (
#             f'{", ".join(self.partners)} agreed on {str(self.agreement)} [id {self.id}]'
#         )
#
