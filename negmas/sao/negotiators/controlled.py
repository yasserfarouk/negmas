from __future__ import annotations
from ...outcomes import Outcome
from ..common import ResponseType, SAOState
from .base import SAONegotiator
from negmas.negotiators.controlled import ControlledNegotiator

__all__ = ["ControlledSAONegotiator"]


class ControlledSAONegotiator(SAONegotiator, ControlledNegotiator):
    """
    A negotiator that acts as an end point to a parent Controller.

    This negotiator simply calls its controler for everything.
    """

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        """Calls parent controller"""
        if self._Negotiator__parent:  # type: ignore
            return self._Negotiator__parent.propose(self.id, state)  # type: ignore

    def respond(self, state, source: str | None = None) -> ResponseType:
        """Calls parent controller"""
        if self._Negotiator__parent:  # type: ignore
            try:
                return self._Negotiator__parent.respond(self.id, state, source)  # type: ignore
            except TypeError:
                return self._Negotiator__parent.respond(self.id, state)  # type: ignore
        return ResponseType.REJECT_OFFER

    # def _on_negotiation_start(self, state: MechanismState) -> None:
    #     """Calls parent controller"""
    #     if self._Negotiator__parent:  # type: ignore
    #         return self._Negotiator__parent._on_negotiation_start(self.id, state)  # type: ignore

    def on_negotiation_start(self, state) -> None:
        """Calls parent controller"""
        if self._Negotiator__parent:  # type: ignore
            return self._Negotiator__parent.on_negotiation_start(self.id, state)  # type: ignore

    def on_negotiation_end(self, state) -> None:
        """Calls parent controller"""
        if self._Negotiator__parent:  # type: ignore
            return self._Negotiator__parent.on_negotiation_end(self.id, state)  # type: ignore

    def join(
        self, nmi, state, *, preferences=None, ufun=None, role: str = "negotiator"
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
        if ufun is not None:
            preferences = ufun
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
