from __future__ import annotations

import warnings
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Optional

from negmas.common import MechanismState, NegotiatorMechanismInterface
from negmas.events import Notifiable, Notification
from negmas.preferences import Preferences
from negmas.types import Rational

if TYPE_CHECKING:
    from negmas.outcomes import Outcome
    from negmas.preferences import Preferences, UFun
    from negmas.situated import Agent

    from .controller import Controller

__all__ = [
    "Negotiator",
]


class Negotiator(Rational, Notifiable, ABC):
    r"""Abstract negotiation agent. Base class for all negotiators

    Args:

           name: Negotiator name. If not given it is assigned by the system (unique 16 characters).
           preferences: The preferences of the agent (pass either this or ufun)
           ufun: The ufun of the agent (overrides preferences if given)
           parent: The `Controller` that controls this neogtiator (if any)
           owner: The `Agent` that own this negotiator (if any)
           id: The unique ID of the negotiator

       Returns:
           bool: True if participating in the given negotiation (or any negotiation if it was None)

       Remarks:
           - `ufun` overrides `preferences`. You need to pass only one of them

    """

    def __init__(
        self,
        name: str = None,
        preferences: Preferences | None = None,
        ufun: UFun | None = None,
        parent: "Controller" = None,
        owner: "Agent" = None,
        id: str = None,
    ) -> None:
        super().__init__(name=name, ufun=ufun, preferences=preferences, id=id)
        self.__parent = parent
        self._capabilities = {"enter": True, "leave": True, "ultimatum": True}
        self._mechanism_id = None
        self._nmi = None
        self._initial_state = None
        self._role = None
        self.__owner = owner

    @property
    def nmi(self):
        return self._nmi

    @property
    def owner(self):
        """Returns the owner agent of the negotiator"""
        return self.__owner

    @owner.setter
    def owner(self, owner):
        """Sets the owner"""
        self.__owner = owner

    @Rational.preferences.setter
    def preferences(self, value: "Preferences"):
        """Sets tha utility function."""
        if self._nmi is not None and self._nmi.state.started:
            warnings.warn(
                "Changing the utility function by direct assignment after the negotiation is "
                "started is deprecated."
            )
        Rational.preferences.fset(self, value)  # type: ignore
        # if self._nmi is not None:
        # else:
        #     self._preferences = value
        #     self._preferences_modified = True

    @property
    def parent(self) -> "Controller" | None:
        """Returns the parent controller"""
        return self.__parent

    def before_death(self, cntxt: Dict[str, Any]) -> bool:
        """Called whenever the parent is about to kill this negotiator. It should return False if the negotiator
        does not want to be killed but the controller can still force-kill it"""
        return True

    def _dissociate(self):
        self._mechanism_id = None
        self._nmi = None
        self._preferences = self._init_preferences
        self._role = None

    def is_acceptable_as_agreement(self, outcome: "Outcome") -> bool:
        """
        Whether the given outcome is acceptable as a final agreement of a negotiation.

        The default behavior is to reject only if a reserved value is defined for the agent and is known to be higher
        than the utility of the outcome.

        """
        if not self.preferences:
            return False
        if self.reserved_outcome is not None:
            if not hasattr(self.preferences, "is_not_worse"):
                return False
            return self.preferences.is_not_worse(outcome, self.reserved_outcome)  # type: ignore
        if not self.ufun:
            return False
        return self.ufun(outcome) >= self.reserved_value

    def isin(self, negotiation_id: Optional[str]) -> bool:
        """Is that agent participating in the given negotiation?
        Tests if the agent is participating in the given negotiation.

        Args:

            negotiation_id (Optional[str]): The negotiation ID tested. If
             None, it means ANY negotiation

        Returns:
            bool: True if participating in the given negotiation (or any
                negotiation if it was None)

        """
        return self._mechanism_id == negotiation_id

    @property
    def capabilities(self) -> Dict[str, Any]:
        """Agent capabilities"""
        return self._capabilities

    def add_capabilities(self, capabilities: dict) -> None:
        """Adds named capabilities to the agent.

        Args:
            capabilities: The capabilities to be added as a dict

        Returns:
            None

        Remarks:
            It is the responsibility of the caller to be really capable of added capabilities.

        """
        if hasattr(self, "_capabilities"):
            self._capabilities.update(capabilities)
        else:
            self._capabilities = capabilities

            # CALL BACKS

    def join(
        self,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: Optional["Preferences"] = None,
        role: str = "agent",
    ) -> bool:
        """
        Called by the mechanism when the agent is about to enter a negotiation. It can prevent the agent from entering

        Args:
            nmi  (AgentMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            ufun (UtilityFunction): The ufun function to use before any discounting.
            role (str): role of the agent.

        Returns:
            bool indicating whether or not the agent accepts to enter.
            If False is returned it will not enter the negotiation

        """
        if self._mechanism_id is not None:
            return False
        self._role = role
        self._mechanism_id = nmi.id
        self._nmi = nmi
        self._initial_state = state
        if preferences is not None and (
            self.preferences is None or id(preferences) != id(self.preferences)
        ):
            self.preferences = preferences
        if self._preferences and self._preferences_modified:
            if self._preferences_modified:
                self.on_preferences_changed()
        return True

    def on_negotiation_start(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation start

        Args:

            state: `MechanismState` giving current state of the negotiation.

        Remarks:

            - You MUST call the super() version of this function either before or after your code when you are
              overriding it.
            - `on_negotiation_start` and `on_negotiation_end` will always be called once for every agent.

        """
        if self._preferences_modified:
            self.on_preferences_changed()

    def on_round_start(self, state: MechanismState) -> None:
        """A call back called at each negotiation round start

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action.

        """

    def on_mechanism_error(self, state: MechanismState) -> None:
        """
        A call back called whenever an error happens in the mechanism. The error and its explanation are accessible in
        `state`

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action

        """

    def on_round_end(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation round end

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action

        """

    def on_leave(self, state: MechanismState) -> None:
        """A call back called after leaving a negotiation.

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Remarks:
            - **MUST** call the baseclass `on_leave` using `super` () if you are going to override this.
            - The default behavior is to do nothing.
            - Override this to hook some action

        """
        self._dissociate()

    def on_negotiation_end(self, state: MechanismState) -> None:
        """
        A call back called at each negotiation end

        Args:
            state: `MechanismState` or one of its descendants giving the state at which the negotiation ended.

        Remarks:
            - The default behavior is to do nothing.
            - Override this to hook some action
            - `on_negotiation_start` and `on_negotiation_end` will always be called once for every agent.

        """

    def on_notification(self, notification: Notification, notifier: str):
        """
        Called whenever the agent receives a notification

        Args:
            notification: The notification!!
            notifier: The notifier!!

        Returns:
            None

        Remarks:

            - You MUST call the super() version of this function either before or after your code when you are
              overriding it.

        """
        if notifier != self._mechanism_id:
            raise ValueError(f"Notification is coming from unknown {notifier}")
        if notification.type == "negotiation_start":
            self.on_negotiation_start(state=notification.data)
        elif notification.type == "round_start":
            self.on_round_start(state=notification.data)
        elif notification.type == "round_end":
            self.on_round_end(state=notification.data)
        elif notification.type == "negotiation_end":
            self.on_negotiation_end(state=notification.data)
        elif notification.type == "ufun_modified":
            self.on_preferences_changed()

    def cancel(self, reason=None) -> None:
        """
        A method that may be called by a mechanism to make the negotiator cancel whatever it is currently
        processing.

        Negotiators can just ignore this message (default behavior) but if there is a way to actually cancel
        work, it should be implemented here to improve the responsiveness of the negotiator.
        """

    def __str__(self):
        return f"{self.name}"

    class Java:
        implements = ["jnegmas.negotiators.Negotiator"]
