from __future__ import annotations

from ...common import MechanismState
from ...outcomes import Outcome
from ..common import ResponseType
from .base import SAONegotiator

__all__ = [
    "ControlledSAONegotiator",
]


class ControlledSAONegotiator(SAONegotiator):
    """
    A negotiator that acts as an end point to a parent Controller.

    This negotiator simply calls its controler for everything.
    """

    def propose(self, state: MechanismState) -> Outcome | None:
        """Calls parent controller"""
        if self._Negotiator__parent:  # type: ignore
            return self._Negotiator__parent.propose(self.id, state)  # type: ignore

    def respond(
        self, state: MechanismState, offer: Outcome, source: str
    ) -> ResponseType:
        """Calls parent controller"""
        if self._Negotiator__parent:  # type: ignore
            return self._Negotiator__parent.respond(self.id, state, offer, source)  # type: ignore
        return ResponseType.REJECT_OFFER

    # def _on_negotiation_start(self, state: MechanismState) -> None:
    #     """Calls parent controller"""
    #     if self._Negotiator__parent:  # type: ignore
    #         return self._Negotiator__parent._on_negotiation_start(self.id, state)  # type: ignore

    def on_negotiation_start(self, state: MechanismState) -> None:
        """Calls parent controller"""
        if self._Negotiator__parent:  # type: ignore
            return self._Negotiator__parent.on_negotiation_start(self.id, state)  # type: ignore

    def on_negotiation_end(self, state: MechanismState) -> None:
        """Calls parent controller"""
        if self._Negotiator__parent:  # type: ignore
            return self._Negotiator__parent.on_negotiation_end(self.id, state)  # type: ignore

    def join(
        self,
        nmi,
        state,
        *,
        preferences=None,
        role="negotiator",
    ) -> bool:
        """
        Joins a negotiation.

        Remarks:

            This method first gets permission from the parent controller by
            calling `before_join` on it and confirming the result is `True`,
            it then joins the negotiation and calls `after_join` of the
            controller to inform it that joining is completed if joining was
            successful.
        """
        permission = (
            self._Negotiator__parent is None  # type: ignore
            or self._Negotiator__parent.before_join(  # type: ignore
                self.id, nmi, state, preferences=preferences, role=role
            )
        )
        if not permission:
            return False
        if super().join(nmi, state, preferences=preferences, role=role):
            if self._Negotiator__parent:  # type: ignore
                self._Negotiator__parent.after_join(  # type: ignore
                    self.id, nmi, state, preferences=preferences, role=role
                )
            return True
        return False
