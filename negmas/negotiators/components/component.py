from __future__ import annotations

from typing import TYPE_CHECKING

from attr import define


def __setattr__(self, name, value):
    pass


def __delattr__(self, name):
    pass


if TYPE_CHECKING:
    from ...common import (
        MechanismState,
        NegotiatorMechanismInterface,
        PreferencesChange,
    )
    from ...negotiators import Negotiator
    from ...preferences import BaseUtilityFunction, Preferences

__all__ = ["Component"]


@define
class Component:
    """
    A component that can be added to a `ModularNegotiator`
    """

    _negotiator: Negotiator

    @property
    def negotiator(self):
        return self._negotiator

    def set_negotiator(self, negotiator: Negotiator) -> None:
        """
        Sets the negotiator of which this component is a part.
        """
        self._negotiator = negotiator

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """
        Called to inform the component that the ufun has changed and the kinds of change that happened.
        """

    def can_join(
        self,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: Preferences | None = None,
        ufun: BaseUtilityFunction | None = None,
        role: str = "negotiator",
    ) -> bool:
        """
        A call back called before joining a negotiation to confirm that we can join it.
        """
        return True

    def after_join(self, nmi: NegotiatorMechanismInterface) -> None:
        """
        A call back called after joining a negotiation to confirm wwe joined.
        """

    def on_negotiation_start(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation start
        """

    def on_round_start(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation round start
        """

    def on_round_end(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation round end
        """

    def on_leave(self, state: MechanismState) -> None:
        """
        A call back called after leaving a negotiation.
        """

    def on_negotiation_end(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation end
        """

    def on_mechanism_error(self, state: MechanismState) -> None:
        """
        A call back called whenever an error happens in the mechanism. The error and its explanation are accessible in
        `state`
        """
